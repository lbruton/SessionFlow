"""
RAG engine for session transcripts.

Embeds conversation turns via mlx-embeddings. Stores vectors in Milvus — either
a remote Standalone instance (via SESSIONFLOW_MILVUS_URI) or embedded Milvus Lite
at ~/.sessionflow/milvus.db (fallback).

Full-text search via SQLite FTS5 sidecar for hybrid search (vector + keyword).
Results merged with Reciprocal Rank Fusion (RRF).

Each turn is tagged with a project_root field, enabling per-project or cross-project search.

Supports multiple embedding models via SESSIONFLOW_MODEL env var (default: embeddinggemma).
"""

import hashlib
import heapq
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Block all HuggingFace network access at runtime.
# Models must be pre-downloaded via setup.sh / download-model.sh.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from pymilvus import MilvusClient, CollectionSchema, FieldSchema
from pymilvus.exceptions import MilvusException
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Dict, NamedTuple, Optional, Sequence, Union
import asyncio
import logging
import sys
import threading
import time

from collection_specs import (
    ISSUE_ID_FIELD_MAX,
    CollectionSpec,
    memories_spec,
    sessions_spec,
)
from fts_hybrid import FTSIndex, fts_backfill_required, rrf_merge
from embedding_control import (
    EmbeddingIdentity,
    get_embedding_budget,
    _env_float,
    _env_int,
)
from provider_adapters import (
    LEGAL_PROVIDERS,
    LEGAL_SORT_BY,
    LEGAL_SOURCE_KINDS,
    REPLACE_SAME_TURN,
    default_provider_metadata,
    is_valid_issue_token,
    path_has_prefix,
)
import boilerplate
import passages
import secret_redaction
from transcript_parser import CUT_MARK, MAX_TURN_CHARS

RECENCY_WEIGHT_DEFAULT = 0.15
RECENCY_DECAY_DAYS_DEFAULT = 7
MISSING_TIMESTAMP_RECENCY = 0.5
_RANKING_SCRATCH_KEYS = ("_rrf_score", "_score", "_semantic_score", "_recency_score")
# FTS weight in RRF for identifier-shaped queries. Rare identifiers
# are absent from the vector top-k, so at equal weight unrelated vector hits
# tie with exact keyword hits. 2.0 ranks exact identifier matches first.
IDENTIFIER_FTS_WEIGHT = 2.0

logger = logging.getLogger("sessionflow.milvus")

# Issue-ID extraction: technical-standard prefixes that match the
# issue-token regex but are never issue references. Dropped during extraction.
_ISSUE_ID_PREFIX_DENYLIST = frozenset(
    {"UTF", "SHA", "HTTP", "HTTPS", "ISO", "RFC", "IPV", "MD", "BASE"}
)
# Milvus VARCHAR field length the extracted ids are stored in.
_ISSUE_ID_FIELD_MAX = ISSUE_ID_FIELD_MAX
# Default number of timeline entries returned by get_issue_timeline.
DEFAULT_TIMELINE_LIMIT = 50
# Generous FTS fetch window for the timeline fallback so the chronological slice
# isn't biased by BM25 rank (the older matching turn may be outside the top-N).
_TIMELINE_FTS_FETCH_CAP = 500
# Observability threshold: warn when one issue matches an unexpectedly large
# structured set. Rows stream through `_OldestN`, so this flags scan cost, not
# memory. No server-side cap is possible: Milvus query_iterator order is undefined
# and cannot sort by a scalar, so a first-N cut would drop arbitrary rows.
_TIMELINE_ROWS_WARN = 50000

# Shared Milvus output_fields for vector search and recency listing. Includes
# ``issue_ids`` so issue tags propagate through ``_row_to_result``.
_SEARCH_OUTPUT_FIELDS = [
    "document", "doc_id", "session_id", "transcript_file",
    "turn_index", "timestamp", "git_branch", "chunk_type",
    "project_root", "logical_session_id", "provider",
    "source_kind", "source_class", "source_id", "source_path",
    "issue_ids",
]


class FtsBackfillTransientError(Exception):
    """Raised by backfill_fts on a transient Milvus / schema-drift failure.

    Signals the FTS heal worker to retry on a later cadence tick rather than treat the
    failure as terminal. The originating exception is preserved as __cause__.
    """


# Serializes FTS heal runs: a non-blocking acquire in backfill_fts ensures a second
# heal attempt (e.g. an overlapping cadence tick) returns early instead of double-work.
_fts_backfill_lock = threading.Lock()


def _extract_issue_ids(text: str) -> str:
    """Extract issue references (e.g. ``SESF-25``) from a turn's text.

    Matches the issue-token regex ``\\b[A-Z][A-Z0-9]+-\\d+\\b`` case-insensitively
    (canonicalizing matches to upper case), drops technical-standard prefixes in
    ``_ISSUE_ID_PREFIX_DENYLIST`` (UTF-8, SHA-256, HTTP-2, ...), and
    deduplicates the survivors in first-seen order.

    Args:
        text: Raw turn text to scan.

    Returns:
        A delimiter-wrapped, comma-joined string of issue ids with a leading
        and trailing comma (e.g. ``",SESF-25,SESF-26,"``), or ``""`` when no
        issue token is found. The result is capped to ``_ISSUE_ID_FIELD_MAX``
        characters so a Milvus insert cannot overflow the storage field; if the
        next id would exceed the cap, extraction stops and logs one warning.
    """
    if not text or not isinstance(text, str):
        return ""
    seen: List[str] = []
    seen_set: set[str] = set()
    # Match case-insensitively and canonicalize only the matched tokens, rather
    # than allocating an uppercased copy of the whole (possibly large) turn text.
    for match in re.finditer(r"\b[A-Z][A-Z0-9]+-\d+\b", text, re.IGNORECASE):
        token = match.group(0).upper()
        prefix = token.split("-", 1)[0]
        if prefix in _ISSUE_ID_PREFIX_DENYLIST:
            continue
        if token in seen_set:
            continue
        seen_set.add(token)
        seen.append(token)

    if not seen:
        return ""

    result = ","
    for token in seen:
        candidate = result + token + ","
        if len(candidate) > _ISSUE_ID_FIELD_MAX:
            logger.warning(
                "issue-id list truncated at %d chars (field cap %d)",
                len(result),
                _ISSUE_ID_FIELD_MAX,
            )
            break
        result = candidate
    # Guard the pathological case where even the first token exceeds the cap:
    # ``result`` would still be the bare delimiter ",", which is neither "" nor a
    # valid comma-wrapped list. Normalize to "".
    return result if len(result) > 1 else ""


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate `text` so its UTF-8 encoding fits in `max_bytes`.

    Milvus VARCHAR caps are measured in bytes, not Python characters; a naive
    `text[:max_bytes]` slice happily produces a 65k-character string that
    serializes to >65k bytes once any multibyte codepoint is present.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _is_remote_uri(uri: str) -> bool:
    """True when uri points to a remote Milvus Standalone (http:// or https://)."""
    return uri.startswith("http://") or uri.startswith("https://")


# --- Model registry ---

_MODEL_REGISTRY = {
    "modernbert": {
        "model_id": "nomic-ai/modernbert-embed-base",
        "embed_dim": 768,
        "max_tokens": 8192,
        "search_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "cache_subdir": "models--nomic-ai--modernbert-embed-base",
    },
    "embeddinggemma": {
        "model_id": "mlx-community/embeddinggemma-300m-bf16",
        "embed_dim": 768,
        "max_tokens": 2048,
        "search_prefix": "task: search result | query: ",
        "document_prefix": "title: none | text: ",
        "cache_subdir": "models--mlx-community--embeddinggemma-300m-bf16",
    },
}

_MODEL_NAME = os.getenv("SESSIONFLOW_MODEL", "embeddinggemma").lower()
if _MODEL_NAME not in _MODEL_REGISTRY:
    raise ValueError(
        f"Unknown model '{_MODEL_NAME}'. "
        f"Valid options: {', '.join(_MODEL_REGISTRY.keys())}"
    )

_MODEL_CFG = _MODEL_REGISTRY[_MODEL_NAME]
_EMBED_DIM = _MODEL_CFG["embed_dim"]
_MODEL_ID = _MODEL_CFG["model_id"]
_MODEL_CACHE = Path.home() / ".cache/huggingface/hub" / _MODEL_CFG["cache_subdir"]
_SEARCH_PREFIX = _MODEL_CFG["search_prefix"]
_DOCUMENT_PREFIX = _MODEL_CFG["document_prefix"]

SESSIONS_SPEC = sessions_spec(_EMBED_DIM)
MEMORIES_SPEC = memories_spec(_EMBED_DIM)
COLLECTION_NAME = SESSIONS_SPEC.name

# --- Model identity check ---

_IDENTITY_FILE = Path.home() / ".sessionflow" / "model_identity.json"


def _check_model_identity(db_path: Optional[str] = None):
    """Verify that the active model matches what was used to build the index.

    On first run, stamps model_identity.json. On subsequent runs, if the stored
    model differs and the index has data, raises an error to prevent mixing
    incompatible vectors.
    """
    _IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)

    if _IDENTITY_FILE.exists():
        stored = json.loads(_IDENTITY_FILE.read_text())
        stored_model = stored.get("model_name", "")
        if stored_model and stored_model != _MODEL_NAME:
            # Check if the index actually has data before raising
            has_data = False
            if db_path:
                try:
                    client = MilvusClient(db_path)
                    if client.has_collection(COLLECTION_NAME):
                        count = client.query(
                            collection_name=COLLECTION_NAME,
                            filter="",
                            limit=1,
                            output_fields=["id"],
                        )
                        has_data = len(count) > 0
                    client.close()
                except Exception:
                    pass
            if has_data:
                raise RuntimeError(
                    f"Model mismatch: index was built with '{stored_model}' but "
                    f"SESSIONFLOW_MODEL is '{_MODEL_NAME}'. "
                    f"Run cleanup.py reset or clear the index before switching models."
                )
            # Index is empty — safe to overwrite the stamp
    # The stamp covers the transcript collection only; the memory model
    # generation lives in Postgres memory_index_meta.
    _IDENTITY_FILE.write_text(
        json.dumps({"model_name": _MODEL_NAME, "collection": COLLECTION_NAME})
    )


def get_model_name() -> str:
    """Return the active model's short name (e.g. 'modernbert', 'embeddinggemma')."""
    return _MODEL_NAME


def get_embedding_identity() -> Dict[str, object]:
    """Return the active local embedding identity for health/status output."""
    try:
        identity = EmbeddingIdentity.current_local()
    except ValueError as exc:
        logger.warning("Invalid embedding identity: %s", exc)
        return {
            "embedding_provider": "unknown",
            "model_name": "unknown",
            "dimension": None,
            "collection_name": COLLECTION_NAME,
            "created_at": "",
            "error": str(exc),
        }
    return {
        "embedding_provider": identity.embedding_provider,
        "model_name": identity.model_name,
        "dimension": identity.dimension,
        "collection_name": identity.collection_name,
        "created_at": identity.created_at,
    }

_mlx_model = None
_mlx_tokenizer = None
_mlx_load = None
_mlx_generate = None
_mlx_core = None
# The HTTP server preloads the model on a background thread, so a
# tool call during startup can reach get_model() at the same time.
_model_load_lock = threading.Lock()
# MLX's Metal command encoder is not thread-safe. Server mode routes every
# embed to the mlx-embed thread; this lock also covers inline callers (before
# init_server_mode, or outside the server), so two evals can never overlap.
_mlx_eval_lock = threading.Lock()


def _load_mlx_runtime():
    """Import MLX lazily so non-embedding tests/status paths cannot crash at import time."""
    global _mlx_load, _mlx_generate, _mlx_core
    if _mlx_load is None or _mlx_generate is None or _mlx_core is None:
        from mlx_embeddings.utils import load as mlx_load, generate as mlx_generate
        import mlx.core as mx
        _mlx_load = mlx_load
        _mlx_generate = mlx_generate
        _mlx_core = mx
    return _mlx_load, _mlx_generate, _mlx_core


def get_model():
    """Get or load the MLX embedding model (one-time load)."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    with _model_load_lock:
        if _mlx_model is not None:
            return _mlx_model, _mlx_tokenizer

        if not _MODEL_CACHE.exists():
            raise RuntimeError(
                f"Embedding model not cached at {_MODEL_CACHE}. "
                f"Run ./setup.sh or ./download-model.sh to download it."
            )

        print(f"Loading {_MODEL_ID} via mlx-embeddings...", file=sys.stderr)
        mlx_load, _, _ = _load_mlx_runtime()
        model, tokenizer = mlx_load(_MODEL_ID)
        _mlx_tokenizer = tokenizer
        _mlx_model = model
        print(f"{_MODEL_ID} ready ({_EMBED_DIM} dims, {_MODEL_CFG['max_tokens']} token context)", file=sys.stderr)
        return _mlx_model, _mlx_tokenizer


def _needs_input_remap() -> bool:
    """Check if the model's __call__ uses 'inputs' instead of 'input_ids'.

    Works around mlx-embeddings gemma3_text models where __call__ expects
    'inputs' but the tokenizer returns 'input_ids'.
    """
    return "gemma" in _MODEL_NAME


def _key_padding_mask(attention_mask):
    """Turn a (B, L) 0/1 tokenizer mask into a (B, 1, L, L) boolean SDPA mask.

    True marks a real key token. mx.fast.scaled_dot_product_attention treats a
    float mask as additive, so the raw 0/1 mask that mlx-embeddings passes
    through only nudges padded keys down by 1 logit.
    """
    import mlx.core as mx

    keys = attention_mask.astype(mx.bool_)[:, None, None, :]
    length = attention_mask.shape[-1]
    return mx.broadcast_to(keys, (keys.shape[0], 1, length, length))


def _attention_masks(attention_mask, window: Optional[int]):
    """Return ``(global_mask, local_mask)`` boolean SDPA masks for a gemma3 forward.

    Global layers see every real key. Sliding layers also need ``|q - k| <= window``,
    the inclusive bidirectional radius Hugging Face applies for EmbeddingGemma. A text
    that fits inside the window gets one shared mask, so its output is unchanged.
    """
    import mlx.core as mx

    full = _key_padding_mask(attention_mask)
    length = attention_mask.shape[-1]
    if window is None or length - 1 <= window:
        return full, full
    positions = mx.arange(length)
    band = mx.abs(positions[:, None] - positions[None, :]) <= window
    return full, mx.logical_and(full, band[None, None, :, :])


def _sliding_window(model) -> Optional[int]:
    return getattr(model.model.config, "sliding_window", None)


def _embed_gemma(model, input_ids, attention_mask):
    """Masked bidirectional forward for gemma3_text, mirroring Gemma3Model and Model.__call__.

    mlx-embeddings passes one mask to every layer. Here padded keys are excluded
    everywhere, so a vector does not depend on its batch, and sliding layers keep
    only keys inside the window (see ``_attention_masks``).
    """
    import mlx.core as mx
    from mlx_embeddings.models.base import mean_pooling, normalize_embeddings

    inner = model.model
    global_mask, local_mask = _attention_masks(attention_mask, _sliding_window(model))
    pattern = inner.config.sliding_window_pattern

    h = inner.embed_tokens(input_ids)
    h *= mx.array(inner.config.hidden_size ** 0.5, inner.embed_tokens.weight.dtype).astype(h.dtype)
    for i, layer in enumerate(inner.layers):
        is_global = i % pattern == pattern - 1
        h = layer(h, global_mask if is_global else local_mask, None)
    out = inner.norm(h)
    for dense in model.dense:
        out = dense(out)
    return normalize_embeddings(mean_pooling(out, attention_mask))


def embed_texts(texts: List[str], is_query: bool = False,
                titles: Optional[Sequence[Optional[str]]] = None) -> List[List[float]]:
    """Embed texts using the configured model. Adds model-specific prefix.

    In server mode the MLX work always runs on the single ``mlx-embed``
    executor thread, whatever thread calls this. MLX's Metal command encoder is
    not thread-safe, and two threads evaluating at once SIGSEGV the server.
    A caller already on that thread, or any caller before
    ``init_server_mode``, runs inline.

    Args:
        texts: Texts to embed.
        is_query: True for search queries, which take the query prefix.
        titles: Per-document titles for ``_document_prefix``; None keeps the default.
    """
    args = (texts, is_query) if titles is None else (texts, is_query, titles)
    executor = _embed_executor
    if executor is not None and not threading.current_thread().name.startswith("mlx-embed"):
        return executor.submit(_embed_texts_local, *args).result()
    return _embed_texts_local(*args)


def _document_prefix(title: Optional[str]) -> str:
    """Return the document prefix with ``title`` in the model's title slot, if it has one."""
    if title and "title: none" in _DOCUMENT_PREFIX:
        return _DOCUMENT_PREFIX.replace("title: none", f"title: {title}", 1)
    return _DOCUMENT_PREFIX


def _embed_texts_local(texts: List[str], is_query: bool = False,
                       titles: Optional[Sequence[Optional[str]]] = None) -> List[List[float]]:
    model, tokenizer = get_model()
    _, mlx_generate, mx = _load_mlx_runtime()
    if is_query:
        prefixed = [_SEARCH_PREFIX + t for t in texts]
    else:
        prefixed = [_document_prefix(titles[i] if titles else None) + t
                    for i, t in enumerate(texts)]

    with _mlx_eval_lock:
        if _needs_input_remap():
            # gemma3_text models expect (inputs, attention_mask) not (input_ids, ...)
            encoded = tokenizer.batch_encode_plus(
                prefixed, return_tensors="mlx", padding=True,
                truncation=True, max_length=_MODEL_CFG["max_tokens"],
            )
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = mx.ones(encoded["input_ids"].shape)
            text_embeds = _embed_gemma(model, encoded["input_ids"], attention_mask)
        else:
            text_embeds = mlx_generate(model, tokenizer, texts=prefixed,
                                       max_length=_MODEL_CFG["max_tokens"]).text_embeds

        embeddings = text_embeds.tolist()
        mx.clear_cache()
    return embeddings


# --- Milvus client management ---

_persistent_clients: Dict[str, MilvusClient] = {}
_fts = FTSIndex("turns_fts", [
    "session_id", "git_branch", "turn_index", "timestamp", "chunk_type",
    "project_root", "logical_session_id", "provider", "source_kind",
    "source_class", "source_id", "source_path", "issue_ids",
])
# A project filter: one root, or several (a root with its subfolders, or a name match).
ProjectRoots = Union[str, Sequence[str]]
# db_path -> (monotonic time, roots); see known_project_roots.
_known_roots_cache: Dict[Optional[str], tuple] = {}
KNOWN_ROOTS_TTL_S = 300.0
_write_lock: Optional[asyncio.Lock] = None
_embed_semaphore: Optional[asyncio.Semaphore] = None
# Dedicated single-worker executor so every MLX/Metal call runs on the same OS
# thread. The asyncio semaphore already serializes calls in time, but the
# default executor can rotate workers between calls, and MLX command-buffer
# state is not safe to migrate across threads.
_embed_executor: Optional[ThreadPoolExecutor] = None
_server_mode = False
# Milvus path given to init_server_mode; the default for memory-context lookups.
_server_db_path: Optional[str] = None


def init_server_mode(db_path: Optional[str] = None):
    """Initialize async concurrency primitives for HTTP server mode."""
    global _write_lock, _embed_semaphore, _embed_executor, _server_mode, _server_db_path
    _check_model_identity(db_path=db_path)
    _server_db_path = db_path
    _write_lock = asyncio.Lock()
    _embed_semaphore = asyncio.Semaphore(1)
    _embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-embed")
    _server_mode = True
    _fts.set_server_mode(True)
    print(f"Server mode initialized (model: {_MODEL_NAME})", file=sys.stderr)


def close_server_mode():
    """Close all persistent clients (Milvus + FTS) and reset server mode."""
    global _write_lock, _embed_semaphore, _embed_executor, _server_mode
    for path, client in list(_persistent_clients.items()):
        try:
            client.close()
            logger.info("Closed Milvus client: %s", path)
        except Exception as e:
            logger.warning("Error closing Milvus client %s: %s", path, _scrub_exception(e))
    _persistent_clients.clear()
    _fts.close_all()
    # Nil the semaphore and lock FIRST so any coroutine that wakes up while we
    # are shutting down sees None and takes the CLI fallback path instead of
    # trying to enqueue work onto a torn-down executor.
    _embed_semaphore = None
    _write_lock = None
    if _embed_executor is not None:
        _embed_executor.shutdown(wait=True)
        _embed_executor = None
    _server_mode = False


def _get_persistent_client(db_path: str) -> MilvusClient:
    """Get or create a persistent client for the given DB path.
    On failure, evicts the stale client and retries once."""
    if db_path in _persistent_clients:
        try:
            _persistent_clients[db_path].has_collection(COLLECTION_NAME)
            return _persistent_clients[db_path]
        except Exception as e:
            logger.warning("Stale Milvus client for %s: %s — reconnecting", db_path, _scrub_exception(e))
            try:
                _persistent_clients[db_path].close()
            except Exception:
                pass
            del _persistent_clients[db_path]

    if not _is_remote_uri(db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        if _is_remote_uri(db_path):
            # Remote Milvus Standalone — default gRPC settings are fine.
            _persistent_clients[db_path] = MilvusClient(db_path)
        else:
            # Milvus Lite — increase gRPC keepalive to 120s to prevent
            # GOAWAY/ENHANCE_YOUR_CALM (Lite rejects default 10s as too_many_pings).
            _persistent_clients[db_path] = MilvusClient(
                db_path,
                grpc_options={
                    "grpc.keepalive_time_ms": 120_000,
                    "grpc.keepalive_timeout_ms": 20_000,
                },
            )
        logger.info("Opened client: %s", db_path)
    except Exception as e:
        logger.error("Failed to connect to Milvus at %s: %s", db_path, _scrub_exception(e))
        raise
    return _persistent_clients[db_path]


def _resolve_db_path(db_path: Optional[str]) -> str:
    if not db_path:
        raise ValueError("db_path is required. Global index is at ~/.sessionflow/milvus.db")
    return db_path


def _expected_schema_fields() -> List[FieldSchema]:
    """Return the sessions collection field list from ``SESSIONS_SPEC``."""
    return SESSIONS_SPEC.field_schemas()


def _spec_kwargs(spec: Optional[CollectionSpec]) -> Dict[str, CollectionSpec]:
    """Forward ``spec`` to a helper only when the caller named one.

    The sessions call form ``helper(client, db_path)`` stays unchanged, so
    callers and test doubles written for it keep working.
    """
    return {} if spec is None else {"spec": spec}


def _describe_value(field, key: str):
    if isinstance(field, dict):
        return field.get(key)
    return getattr(field, key, None)


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def detect_schema_drift(client: MilvusClient,
                        spec: Optional[CollectionSpec] = None) -> List[str]:
    """Return how the live collection differs from ``spec``; empty means no drift.

    Reports ``missing:``, ``extra:``, then ``type:`` and ``dim:`` for a
    same-named field whose type or vector dimension changed. Lite and
    Standalone describe output differs, so an absent type or dim is not drift.
    """
    spec = spec or SESSIONS_SPEC
    if not client.has_collection(spec.name):
        return []
    try:
        info = client.describe_collection(spec.name)
    except Exception as exc:
        print(f"Schema drift check skipped: describe_collection failed: {exc}", file=sys.stderr)
        return []
    expected = {f.name: f for f in spec.fields}
    actual: Dict[str, object] = {}
    for field in info.get("fields", []) or []:
        name = _describe_value(field, "name")
        if name:
            actual[name] = field
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed_type: List[str] = []
    changed_dim: List[str] = []
    for name in sorted(set(expected) & set(actual)):
        want = expected[name]
        live = actual[name]
        live_type = _as_int(_describe_value(live, "type"))
        if live_type is not None and live_type != int(want.dtype):
            changed_type.append(name)
            continue
        if want.dim is not None:
            params = _describe_value(live, "params") or {}
            live_dim = _as_int(params.get("dim")) if isinstance(params, dict) else None
            if live_dim is not None and live_dim != want.dim:
                changed_dim.append(name)
    return (
        [f"missing:{n}" for n in missing]
        + [f"extra:{n}" for n in extra]
        + [f"type:{n}" for n in changed_type]
        + [f"dim:{n}" for n in changed_dim]
    )


def migrate_schema(client: MilvusClient, db_path: str = "",
                   spec: Optional[CollectionSpec] = None) -> None:
    """Drop the collection and recreate it with the current schema.

    DESTRUCTIVE: every row in the collection is lost. Provided as the explicit
    recovery path for `python cleanup.py migrate-schema` and for the
    auto-migrate env opt-in (SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1).
    """
    name = (spec or SESSIONS_SPEC).name
    if client.has_collection(name):
        print(
            f"Dropping collection {name!r} for schema migration "
            "(all indexed rows will be lost)",
            file=sys.stderr,
        )
        client.drop_collection(name)
    _create_collection(client, db_path, **_spec_kwargs(spec))
    # drop_collection already clears the schema cache. Every schema mutation
    # still goes through _invalidate_schema_cache.
    _invalidate_schema_cache(client, **_spec_kwargs(spec))


def _invalidate_schema_cache(client: MilvusClient,
                             spec: Optional[CollectionSpec] = None) -> None:
    """Invalidate pymilvus's process-global data-path schema cache for one collection.

    pymilvus keeps a process-wide schema cache that ``add_collection_field`` does
    not clear, so a live client raises ``code=65535 "field not exist"`` after a
    field add. Call this after every schema change. It uses pymilvus internals and
    degrades to a logged no-op if they move.
    """
    name = (spec or SESSIONS_SPEC).name
    try:
        from pymilvus.client.cache import GlobalCache
    except Exception as exc:  # pragma: no cover - pymilvus internals moved
        logger.debug("Schema cache invalidation skipped (no GlobalCache): %s", exc)
        return
    try:
        endpoint = client._get_connection().server_address
        # SessionFlow uses the default database; SchemaCache normalizes "" -> "default".
        GlobalCache.schema.invalidate(endpoint, "", name)
    except Exception as exc:  # pragma: no cover - defense in depth
        logger.warning("Schema cache invalidation failed for %s: %s", name, exc)


def _ensure_collection(client: MilvusClient, db_path: str = "",
                       spec: Optional[CollectionSpec] = None) -> None:
    """Create the collection if missing; refuse to start on schema drift.

    ``spec`` defaults to the sessions collection. A collection that lacks a
    field from its spec rejects every insert, so drift must stop startup. With
    SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 it drops and recreates the collection;
    otherwise it raises RuntimeError naming the recovery options.
    """
    name = (spec or SESSIONS_SPEC).name
    if not client.has_collection(name):
        _create_collection(client, db_path, **_spec_kwargs(spec))
        return

    drift = detect_schema_drift(client, **_spec_kwargs(spec))
    if not drift:
        return

    # Re-describe once (cache-free) before gating either branch.
    # detect_schema_drift issues a fresh describe_collection, so a stale/cached
    # first read that clears on the second describe must NOT raise or migrate.
    # This re-verify gates BOTH the auto-migrate and the raise branches.
    drift = detect_schema_drift(client, **_spec_kwargs(spec))
    if not drift:
        return

    auto = os.getenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", "").lower() in {"1", "true", "yes", "on"}
    if auto:
        print(
            f"SESSIONFLOW_AUTO_MIGRATE_SCHEMA detected schema drift {drift!r} in "
            f"{name!r}; dropping and recreating (all rows lost).",
            file=sys.stderr,
        )
        migrate_schema(client, db_path, **_spec_kwargs(spec))
        return

    if name != COLLECTION_NAME:
        raise RuntimeError(
            f"Milvus collection {name!r} schema is out of date (drift={drift}). "
            f"First restart the server: a transient describe can clear on a "
            f"fresh read. If drift persists, set SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 "
            f"to drop and recreate it; its rows are re-indexed from Postgres."
        )
    raise RuntimeError(
        f"Milvus collection {COLLECTION_NAME!r} schema is out of date "
        f"(drift={drift}). First try the non-destructive option: restart the "
        f"server — a transient describe can clear on a fresh read. If drift "
        f"persists, recover with one of these DESTRUCTIVE options (both lose "
        f"all turns): run `python cleanup.py migrate-schema` to drop and "
        f"recreate it (destructive — all turns lost), or set "
        f"SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 to migrate on startup "
        f"(destructive — all turns lost)."
    )


def _create_collection(client: MilvusClient, db_path: str = "",
                       spec: Optional[CollectionSpec] = None) -> None:
    spec = spec or SESSIONS_SPEC
    print(f"Creating collection: {spec.name} (dim={spec.vector_dim()})", file=sys.stderr)
    schema = CollectionSchema(fields=spec.field_schemas())

    index_params = client.prepare_index_params()
    # Standalone gets HNSW; Milvus Lite silently ignores non-FLAT indexes.
    index_params.add_index(**spec.index_spec(remote=_is_remote_uri(db_path)))

    create_kwargs = {
        "collection_name": spec.name,
        "schema": schema,
        "index_params": index_params,
    }
    if spec.consistency_level is not None:
        create_kwargs["consistency_level"] = spec.consistency_level
    client.create_collection(**create_kwargs)

    print(f"Collection created: {spec.name}", file=sys.stderr)

    # Standalone requires explicit load_collection before query/dedup paths work.
    # create_collection with index_params does not auto-load.
    if _is_remote_uri(db_path):
        client.load_collection(collection_name=spec.name)
        print(f"Collection loaded: {spec.name}", file=sys.stderr)


@contextmanager
def milvus_client_for_migration(db_path: Optional[str] = None):
    """Open a Milvus client WITHOUT _ensure_collection.

    `_ensure_collection` refuses to open a drifted collection, and
    `cleanup.py migrate-schema` must open it to repair the drift. Use this only
    in migration code.
    """
    path = _resolve_db_path(db_path)
    if not _is_remote_uri(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(path)
    try:
        yield client
    finally:
        client.close()


@contextmanager
def milvus_client(db_path: Optional[str] = None):
    """Get a Milvus client. In server mode, reuses persistent client."""
    path = _resolve_db_path(db_path)

    if _server_mode:
        client = _get_persistent_client(path)
        _ensure_collection(client, path)
        yield client
    else:
        if not _is_remote_uri(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(path)
        _ensure_collection(client, path)
        try:
            yield client
        finally:
            client.close()


# --- Secret redaction guard ---

# Truthy values for the SESSIONFLOW_REDACT on/off flag (boolean idiom, rag_engine
# precedent at the SESSIONFLOW_AUTO_MIGRATE_SCHEMA read).
_REDACT_TRUE = {"1", "true", "yes", "on"}
_REDACT_MODES = {"enforce", "report"}

# Durable, process-lifetime per-rule detection counts surfaced via get_stats under
# the "redaction" key (AC-10). Rule names only — never a secret value (AC-18).
# Guarded by _redaction_lock: the check-then-set update is a read-modify-write that
# would race under concurrent ingestion despite the GIL.
_redaction_counters: Dict[str, int] = {}
_redaction_lock = threading.Lock()

# mtime-keyed cache for the operator allowlist so a hot backfill path does not
# re-read + re-compile the file on every add_turns batch. {path: (mtime, patterns)}.
_allowlist_cache: Dict[str, tuple] = {}


def _redaction_settings() -> tuple[bool, str, Optional[str]]:
    """Read the redaction config from the environment (AC-11/12/13).

    Returns:
        ``(enabled, mode, allowlist_path)``. ``SESSIONFLOW_REDACT`` unset defaults to
        enabled in ``report`` mode; an explicit off value disables redaction.
    """
    raw = os.getenv("SESSIONFLOW_REDACT")
    enabled = True if raw is None else raw.strip().lower() in _REDACT_TRUE
    mode = os.getenv("SESSIONFLOW_REDACT_MODE", "report").strip().lower()
    if mode not in _REDACT_MODES:
        mode = "report"
    return enabled, mode, os.getenv("SESSIONFLOW_REDACT_ALLOWLIST")


def load_allowlist(path: Optional[str]) -> List[re.Pattern]:
    """Load operator allowlist regex patterns from ``path`` (one per line).

    Impure on purpose: keeps file I/O out of the pure ``secret_redaction`` module
    (D-4, AC-16). Blank lines and ``#`` comments are ignored; invalid patterns are
    skipped with a warning. Returns an empty list when ``path`` is falsy/unreadable.
    """
    if not path:
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        logger.warning("Could not read redaction allowlist %s: %s", path, exc)
        return []
    cached = _allowlist_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    patterns: List[re.Pattern] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    patterns.append(re.compile(stripped))
                except re.error as exc:
                    logger.warning("Skipping invalid redaction allowlist pattern: %s", exc)
    except OSError as exc:
        logger.warning("Could not read redaction allowlist %s: %s", path, exc)
        return []
    _allowlist_cache[path] = (mtime, patterns)
    return patterns


def _apply_redaction(turns: List[Dict]) -> None:
    """Redact secrets in ``turns`` in place before embed/store.

    Runs once over the already-deduped turns so all three durable sinks and the
    async wrapper are covered with no per-Provider change (AC-1/2/3), and before
    ``_extract_issue_ids`` so issue IDs survive. In ``report`` mode it counts and
    logs detections without mutating the text (AC-10); when disabled it is a no-op
    (AC-12). Rule names only ever reach the log/counters (AC-18).
    """
    enabled, mode, allowlist_path = _redaction_settings()
    if not enabled:
        return
    allowlist = load_allowlist(allowlist_path)
    rule_counts: Dict[str, int] = {}
    for turn in turns:
        redacted, hits = secret_redaction.redact(
            turn.get("text", ""), mode=mode, allowlist=allowlist
        )
        for hit in hits:
            rule_counts[hit.rule_name] = rule_counts.get(hit.rule_name, 0) + 1
        if mode == "enforce":
            turn["text"] = redacted
    if rule_counts:
        with _redaction_lock:
            for rule_name, count in rule_counts.items():
                _redaction_counters[rule_name] = _redaction_counters.get(rule_name, 0) + count
        histogram = ", ".join(f"{name}={count}" for name, count in sorted(rule_counts.items()))
        logger.info("Redaction (%s mode) detected: %s", mode, histogram)


def _redaction_status() -> Dict:
    """Return the operator-facing redaction status surface (AC-10)."""
    enabled, mode, _ = _redaction_settings()
    with _redaction_lock:
        counts = dict(_redaction_counters)
    return {"enabled": enabled, "mode": mode, "counts": counts}


# Process-lifetime counts of stripped boilerplate blocks per rule, and of turns
# dropped because nothing searchable was left. Read by get_stats under "boilerplate".
_boilerplate_counters: Dict[str, int] = {}
_boilerplate_dropped = 0
_boilerplate_lock = threading.Lock()

# Same floor as transcript_parser._build_turn: shorter text holds nothing to search.
MIN_TURN_CHARS = 20


def _boilerplate_enabled() -> bool:
    raw = os.getenv("SESSIONFLOW_STRIP_BOILERPLATE")
    return True if raw is None else raw.strip().lower() in _REDACT_TRUE


def _apply_boilerplate(turns: List[Dict]) -> List[Dict]:
    """Strip harness boilerplate from ``turns`` in place and return the turns worth keeping.

    Runs before redaction and issue-ID extraction, so neither sees harness text. A turn
    that stripping leaves under ``MIN_TURN_CHARS`` is dropped; other turns keep their
    length. ``SESSIONFLOW_STRIP_BOILERPLATE=off`` disables it.
    """
    global _boilerplate_dropped
    if not _boilerplate_enabled():
        return turns
    kept: List[Dict] = []
    rule_counts: Dict[str, int] = {}
    for turn in turns:
        clean, removed = boilerplate.strip(turn.get("text", ""))
        for rule, count in removed.items():
            rule_counts[rule] = rule_counts.get(rule, 0) + count
        if removed:
            turn["text"] = clean
            if len(clean.strip()) < MIN_TURN_CHARS:
                continue
        kept.append(turn)
    with _boilerplate_lock:
        for rule, count in rule_counts.items():
            _boilerplate_counters[rule] = _boilerplate_counters.get(rule, 0) + count
        _boilerplate_dropped += len(turns) - len(kept)
    return kept


def would_store(turn: Dict) -> bool:
    """Return False for a turn that ``_apply_boilerplate`` would drop, without changing it."""
    if not _boilerplate_enabled():
        return True
    clean, removed = boilerplate.strip(turn.get("text", ""))
    return not removed or len(clean.strip()) >= MIN_TURN_CHARS


def _boilerplate_status() -> Dict:
    with _boilerplate_lock:
        return {
            "enabled": _boilerplate_enabled(),
            "counts": dict(_boilerplate_counters),
            "dropped_turns": _boilerplate_dropped,
        }


def _scrub_exception(exc: BaseException) -> str:
    """Return the exception text with any secret value redacted (AC-17)."""
    redacted, _ = secret_redaction.redact(str(exc), mode="enforce")
    return redacted


def _scrub_exception_args(exc: BaseException) -> None:
    """Redact every string arg on ``exc`` in place, preserving non-string args (AC-17).

    Scrubbing each string arg (rather than collapsing to a single message) keeps
    status codes and other structured metadata in ``exc.args[1:]`` intact while
    ensuring no secret survives in any stringified form of the re-raised exception.
    """
    if exc.args:
        exc.args = tuple(
            secret_redaction.redact(arg, mode="enforce")[0] if isinstance(arg, str) else arg
            for arg in exc.args
        )


# --- Core operations ---

def add_turns(turns: List[Dict], db_path: Optional[str] = None) -> int:
    """Insert conversation turn chunks into Milvus. Dedup by doc_id.

    Each turn dict should have:
        text, doc_id, session_id, transcript_file, turn_index,
        timestamp, git_branch, chunk_type
    """
    if not turns:
        return 0

    # Dedup: check which doc_ids already exist
    with milvus_client(db_path) as client:
        existing_ids = set()
        for turn in turns:
            doc_id = turn["doc_id"]
            try:
                results = client.query(
                    collection_name=COLLECTION_NAME,
                    filter=f'doc_id == "{doc_id}"',
                    limit=1,
                    output_fields=["doc_id"],
                )
                if results:
                    existing_ids.add(doc_id)
            except Exception as e:
                logger.warning("Dedup check failed for doc_id %s: %s", doc_id, _scrub_exception(e))

    already_indexed = [t for t in turns if t["doc_id"] in existing_ids]
    new_turns = [t for t in turns if t["doc_id"] not in existing_ids]
    new_turns = _apply_boilerplate(new_turns)
    if not new_turns:
        _replace_stale_turns(already_indexed, db_path)
        return 0

    # Ingestion-time secret redaction guard. One hook over the deduped
    # turns rewrites turn["text"] in place, covering all three durable sinks
    # (embedding, Milvus document, FTS content) and add_turns_async with no
    # per-Provider change, and runs before _extract_issue_ids so issue IDs survive.
    _apply_redaction(new_turns)

    units = _with_passages(new_turns)
    items = [t for unit in units for t in unit]

    # Embed texts in local, resource-controlled batches. Query embedding stays
    # untouched in search(); this path is ingestion/backfill only.
    budget = get_embedding_budget()
    all_embeddings = []
    for batch in budget.split_batches(items):
        texts = [t["text"] for t in batch]
        titles = [t.get("title") for t in batch]
        decision = budget.before_batch(
            batch_size=len(batch),
            estimated_chars=sum(len(t) for t in texts),
        )
        if not decision.allowed and decision.retry_after_seconds > 0:
            time.sleep(decision.retry_after_seconds)
            decision = budget.before_batch(
                batch_size=len(batch),
                estimated_chars=sum(len(t) for t in texts),
            )
        if not decision.allowed:
            logger.info("Embedding batch deferred: %s", decision.reason)
            break

        started = time.monotonic()
        try:
            if any(titles):
                embeddings = embed_texts(texts, is_query=False, titles=titles)
            else:
                embeddings = embed_texts(texts, is_query=False)
        except Exception as e:
            budget.after_batch(time.monotonic() - started, 0, error=e)
            # Scrub the exception's string args before the bare
            # re-raise so every upstream site that later stringifies it is already
            # clean, while preserving any structured status-code args.
            _scrub_exception_args(e)
            raise
        budget.after_batch(time.monotonic() - started, len(batch))
        all_embeddings.extend(embeddings)

    # A deferred batch drops every unit it reached, so no parent lands without its passages.
    new_turns, done = [], 0
    for unit in units:
        if done + len(unit) > len(all_embeddings):
            break
        new_turns.extend(unit)
        done += len(unit)
    embeddings = all_embeddings[:done]
    if not new_turns:
        _replace_stale_turns(already_indexed, db_path)
        return 0

    provider_defaults = default_provider_metadata()
    data = [_milvus_row(turn, emb, provider_defaults)
            for turn, emb in zip(new_turns, embeddings)]

    with milvus_client(db_path) as client:
        client.insert(collection_name=COLLECTION_NAME, data=data)

    # Dual-write into FTS5 sidecar
    try:
        if db_path:
            fts_conn = _fts.connection(db_path)
            _fts.insert(fts_conn, [_fts_record(t, provider_defaults) for t in new_turns])
            _fts.close_ephemeral(fts_conn)
    except Exception as e:
        # The FTS payload holds Turn content, so scrub the exception
        # text before logging so no secret fragment can echo through the warning.
        logger.warning("FTS insert failed (non-fatal): %s", _scrub_exception(e))

    _replace_stale_turns(already_indexed + new_turns, db_path)
    return len(data)


def _milvus_row(turn: Dict, vector: List[float], defaults: Dict) -> Dict:
    """Return the Milvus row of ``turn``; ``turn["_issue_ids"]`` is already set."""
    return {
        # sha256-derived, because Python's hash() differs per process.
        "id": _pk_from_doc_id(turn["doc_id"]),
        "vector": vector,
        "document": _truncate_utf8(turn["text"], 65535),
        "doc_id": turn["doc_id"],
        "session_id": turn.get("session_id", ""),
        "logical_session_id": turn.get("logical_session_id", turn.get("session_id", "")),
        "provider": turn.get("provider", defaults["provider"]),
        "source_kind": turn.get("source_kind", defaults["source_kind"]),
        "source_class": turn.get("source_class", defaults["source_class"]),
        "source_id": turn.get("source_id", ""),
        "source_path": turn.get("source_path", turn.get("transcript_file", "")),
        "transcript_file": turn.get("transcript_file", ""),
        "turn_index": turn.get("turn_index", 0),
        "timestamp": turn.get("timestamp", ""),
        "git_branch": turn.get("git_branch", ""),
        "chunk_type": turn.get("chunk_type", "turn"),
        "project_root": turn.get("project_root", ""),
        "issue_ids": turn["_issue_ids"],
    }


def _fts_record(turn: Dict, defaults: Dict) -> Dict:
    """Return the FTS sidecar record of ``turn``; ``turn["_issue_ids"]`` is already set."""
    return {
        "doc_id": turn["doc_id"],
        "content": turn["text"],
        "session_id": turn.get("session_id", ""),
        "logical_session_id": turn.get("logical_session_id", turn.get("session_id", "")),
        "provider": turn.get("provider", defaults["provider"]),
        "source_kind": turn.get("source_kind", defaults["source_kind"]),
        "source_class": turn.get("source_class", defaults["source_class"]),
        "source_id": turn.get("source_id", ""),
        "source_path": turn.get("source_path", turn.get("transcript_file", "")),
        "git_branch": turn.get("git_branch", ""),
        "turn_index": turn.get("turn_index", 0),
        "timestamp": turn.get("timestamp", ""),
        "chunk_type": turn.get("chunk_type", "turn"),
        "project_root": turn.get("project_root", ""),
        "issue_ids": turn["_issue_ids"],
    }


def replace_passages(parent_doc_id: str, built: List[Dict], vectors: List[List[float]],
                     db_path: Optional[str] = None) -> bool:
    """Replace the stored passages of ``parent_doc_id`` with ``built`` and their vectors.

    Deletes the parent's passages in Milvus and FTS, then inserts the new ones, so
    a repeated call leaves one copy. Passages carry ``_issue_ids``.

    Args:
        parent_doc_id: Doc_id of the parent turn.
        built: Passage dicts from ``passages.build_passages``, all of one turn.
        vectors: One vector per passage.
        db_path: Milvus DB path; also derives the FTS sidecar path.

    Returns:
        False when the FTS write failed; ``write_passage_fts`` retries it without
        Milvus. The server's FTS healing runs only when its sentinel is set.
    """
    if not built:
        return True
    defaults = default_provider_metadata()
    with milvus_client(db_path) as client:
        _delete_milvus_passages(client, parent_doc_id, built[0])
        client.insert(collection_name=COLLECTION_NAME,
                      data=[_milvus_row(p, v, defaults) for p, v in zip(built, vectors)])
    return write_passage_fts(parent_doc_id, built, db_path=db_path)


def write_passage_fts(parent_doc_id: str, built: List[Dict],
                      db_path: Optional[str] = None) -> bool:
    """Replace the FTS rows of ``parent_doc_id``'s passages with ``built``.

    Args:
        parent_doc_id: Doc_id of the parent turn.
        built: Passage dicts from ``passages.build_passages``, carrying ``_issue_ids``.
        db_path: Milvus DB path; derives the FTS sidecar path.

    Returns:
        False when the FTS write failed.
    """
    if not db_path:
        return True
    defaults = default_provider_metadata()
    conn = None
    try:
        conn = _fts.connection(db_path)
        _fts.delete_where(conn, "doc_id >= ? AND doc_id < ?",
                          (f"{parent_doc_id}::p", f"{parent_doc_id}::q"))
        _fts.insert(conn, [_fts_record(p, defaults) for p in built])
        return True
    except Exception as e:
        logger.warning("FTS passage write failed: %s", _scrub_exception(e))
        return False
    finally:
        if conn is not None:
            _fts.close_ephemeral(conn)


_offset_tokenizer = None
_offset_tokenizer_lock = threading.Lock()
_passage_warned = False


def _token_offsets(text: str) -> List[tuple]:
    """Return the ``(start, end)`` character span of each document token of ``text``.

    Uses the transformers tokenizer of the cached model, which matches the MLX
    one and needs no MLX thread. The Rust tokenizer is not safe across threads.
    """
    global _offset_tokenizer
    with _offset_tokenizer_lock:
        if _offset_tokenizer is None:
            from transformers import AutoTokenizer

            snapshots = sorted(_MODEL_CACHE.glob("snapshots/*/config.json"))
            if not snapshots:
                raise RuntimeError(f"Embedding model not cached at {_MODEL_CACHE}")
            _offset_tokenizer = AutoTokenizer.from_pretrained(str(snapshots[0].parent))
            # Offsets cover the whole turn; only embedding input is length-capped.
            _offset_tokenizer.model_max_length = sys.maxsize
        encoded = _offset_tokenizer(text, add_special_tokens=False,
                                    return_offsets_mapping=True)
    return [tuple(span) for span in encoded["offset_mapping"]]


def _with_passages(turns: List[Dict]) -> List[List[Dict]]:
    """Group each turn with its passages, parent first, and stamp ``_issue_ids`` on all.

    Passages copy the parent's issue IDs, so an issue filter finds every row of
    a turn. A tokenizer failure stores the parent alone.
    """
    global _passage_warned
    units = []
    for turn in turns:
        turn["_issue_ids"] = _extract_issue_ids(turn.get("text", ""))
        unit = [turn]
        text = turn.get("text", "")
        # A token covers at least one character, so a shorter text has no passages.
        if turn.get("chunk_type", "turn") == "turn" and len(text) > passages.PASSAGE_TOKENS:
            try:
                built = passages.build_passages(turn, _token_offsets(text))
            except Exception as e:
                if not _passage_warned:
                    logger.warning("Passages skipped, tokenizer failed: %s", _scrub_exception(e))
                    _passage_warned = True
                built = []
            for p in built:
                p["_issue_ids"] = turn["_issue_ids"]
            unit.extend(built)
        units.append(unit)
    return units


def _replace_stale_turns(landed: List[Dict], db_path: Optional[str]) -> int:
    """Delete other rows of the same session and turn_index as each landed turn.

    Only turns flagged with ``REPLACE_SAME_TURN`` take part, and only rows with
    ``chunk_type == "turn"`` are deleted, so summaries and other providers' rows
    stay. Call it after the landed turns are stored, never before.
    """
    keep: Dict[str, Dict[int, str]] = {}
    for t in landed:
        if t.get(REPLACE_SAME_TURN) and t.get("chunk_type", "turn") == "turn":
            keep.setdefault(t.get("session_id", ""), {})[int(t.get("turn_index", 0))] = t["doc_id"]
    if not keep:
        return 0

    stale: List[str] = []
    with milvus_client(db_path) as client:
        for session_id, by_index in keep.items():
            expr = (
                f'session_id == "{_escape_filter_scalar(session_id)}" and chunk_type == "turn" '
                f"and turn_index in {json.dumps(sorted(by_index))}"
            )
            try:
                rows = client.query(collection_name=COLLECTION_NAME, filter=expr,
                                    output_fields=["doc_id", "turn_index"])
            except Exception as e:
                logger.warning("Stale-turn lookup failed: %s", _scrub_exception(e))
                continue
            stale.extend(r["doc_id"] for r in rows
                         if r["doc_id"] != by_index.get(int(r["turn_index"])))

    for doc_id in stale:
        result = delete_by_doc_id(doc_id, db_path=db_path)
        if not result.fts_ok:
            logger.warning("Stale turn %s left in FTS; cleanup.py dedupe-turns removes it", doc_id)
    return len(stale)


def _escape_filter_scalar(value: str) -> str:
    """Escape ``value`` for a double-quoted Milvus filter literal; reject NUL bytes.

    Milvus accepts backslash escapes only (no ``""`` doubling) and no raw newlines.
    Double backslashes first, then escape quotes, newlines and tabs, so no added
    escape is doubled and a trailing backslash cannot swallow the closing quote.
    """
    if "\x00" in value:
        raise ValueError("Filter scalar value must not contain NUL bytes")
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _issue_id_containment_token(issue_id: str) -> str:
    """Normalize an issue id into a safe ``%,TOKEN,%`` containment token.

    The token is matched as ``%,TOKEN,%`` against the comma-wrapped ``issue_ids``
    field. Surrounding whitespace is stripped so ``" sesf-25 "`` can't yield a
    never-matching ``%, SESF-25 ,%``; ``%``/``_`` are stripped so a malformed id
    can't broaden the match into a wildcard scan; a valid token
    (``[A-Z][A-Z0-9]+-\\d+``) contains none of these, so this is a no-op for
    legitimate input. NUL bytes are rejected outright (mirroring
    ``_escape_filter_scalar``) since the FTS path consumes this token without
    going through that guard. Shared by the Milvus filter and the FTS filter so
    both halves of hybrid search stay in lockstep.
    """
    if "\x00" in issue_id:
        raise ValueError("issue_id must not contain NUL bytes")
    return issue_id.strip().upper().replace("%", "").replace("_", "")


def _row_to_result(entity: Dict, defaults: Dict, distance: Optional[float] = None) -> Dict:
    """Map a Milvus entity dict to the standard internal result format.

    Shared by ``search`` (vector hit entities) and ``_recent_listing`` (query
    result rows) so any new schema field added to the collection propagates to
    both code paths automatically.

    ``distance`` is the COSINE distance from the query vector. Query-less
    listings have no query vector, so they keep the default ``None`` (no score)
    and ``tools.format_results`` labels them ``Listing``.
    """
    return {
        "content": entity.get("document", ""),
        "doc_id": entity.get("doc_id", ""),
        "session_id": entity.get("session_id", ""),
        "logical_session_id": entity.get("logical_session_id", entity.get("session_id", "")),
        "provider": entity.get("provider", defaults["provider"]),
        "source_kind": entity.get("source_kind", defaults["source_kind"]),
        "source_class": entity.get("source_class", defaults["source_class"]),
        "source_id": entity.get("source_id", ""),
        "source_path": entity.get("source_path", entity.get("transcript_file", "")),
        "transcript_file": entity.get("transcript_file", ""),
        "turn_index": entity.get("turn_index", 0),
        "timestamp": entity.get("timestamp", ""),
        "git_branch": entity.get("git_branch", ""),
        "chunk_type": entity.get("chunk_type", ""),
        "project_root": entity.get("project_root", ""),
        "issue_ids": entity.get("issue_ids", ""),
        "distance": distance,
    }


def _project_root_filter(project_root: Optional[ProjectRoots]) -> Optional[str]:
    """Milvus clause for one root (``==``) or several (``in``); None when unscoped."""
    if not project_root:
        return None
    if isinstance(project_root, str):
        project_root = [project_root]
    if len(project_root) == 1:
        return f'project_root == "{_escape_filter_scalar(project_root[0])}"'
    values = ", ".join(f'"{_escape_filter_scalar(r)}"' for r in project_root)
    return f"project_root in [{values}]"


def _fts_project_filter(project_root: ProjectRoots):
    if isinstance(project_root, str):
        return project_root
    return ("in", list(project_root))


def _build_milvus_filter(session_id: Optional[str], git_branch: Optional[str],
                         project_root: Optional[ProjectRoots], provider: Optional[str],
                         source_kind: Optional[str], date_from: Optional[str],
                         date_to: Optional[str],
                         issue_id: Optional[str] = None) -> Optional[str]:
    """Compose the Milvus boolean-expression filter shared by vector search and
    the query-less recency listing. Returns None when no filters apply.

    provider/source_kind are validated against allowlists before reaching here,
    so they are interpolated directly; user-supplied scalars are escaped.
    """
    filters = []
    if session_id:
        filters.append(f'session_id == "{_escape_filter_scalar(session_id)}"')
    if git_branch:
        filters.append(f'git_branch == "{_escape_filter_scalar(git_branch)}"')
    if project_root:
        filters.append(_project_root_filter(project_root))
    if provider:
        filters.append(f'provider == "{provider}"')
    if source_kind:
        filters.append(f'source_kind == "{source_kind}"')
    if date_from:
        filters.append(f'timestamp >= "{_escape_filter_scalar(date_from)}"')
    if date_to:
        # Strip any existing time component before appending T23:59:59 so the
        # ISO string is always well-formed even if the caller passed a full
        # datetime string.
        date_to_date = date_to.split("T")[0]
        filters.append(f'timestamp <= "{_escape_filter_scalar(date_to_date)}T23:59:59"')
    if issue_id:
        # Outer %,...,% are the intended containment wildcards; the token itself
        # is wildcard-stripped (see _issue_id_containment_token) then escaped for
        # the Milvus double-quoted literal.
        token = _escape_filter_scalar(_issue_id_containment_token(issue_id))
        filters.append(f'issue_ids like "%,{token},%"')
    return " && ".join(filters) if filters else None


def _is_identifier_query(query: Optional[str]) -> bool:
    """True for a single token with a digit or ``_-.``: hostnames, ticket IDs, symbols."""
    q = (query or "").strip()
    if not q or any(c.isspace() for c in q):
        return False
    return any(c.isdigit() or c in "_-." for c in q)


def search(query: Optional[str], n: int = 5, session_id: Optional[str] = None,
           git_branch: Optional[str] = None, project_root: Optional[ProjectRoots] = None,
           sort_by: str = "hybrid",
           date_from: Optional[str] = None, date_to: Optional[str] = None,
           provider: Optional[str] = None, source_kind: Optional[str] = None,
           issue_id: Optional[str] = None,
           db_path: Optional[str] = None) -> List[Dict]:
    """Hybrid search: vector similarity + FTS5 keyword search, merged with RRF.

    Both engines run with an expanded candidate pool (n*3), then RRF merges
    the two ranked lists. The merged pool is re-ordered by ``sort_by``:
    ``relevance`` (pure RRF order), ``recency`` (timestamp-descending re-rank),
    or ``hybrid`` (blended semantic + recency score; the default).

    project_root: when set, restricts results to that root, or to any of a list
    of roots. When None, searches across all projects (cross-project search).
    date_from/date_to: ISO 8601 date strings (e.g. '2026-04-02') to restrict
    results to a time range. Timestamps are VARCHAR and sort lexicographically.
    issue_id: when set, restricts results to turns tagged with that issue id
    (e.g. 'SESF-25'); uppercased and matched as an exact comma-delimited token.
    """
    # Validate sort_by before any embedding/Milvus work (mirrors the
    # provider/source_kind entry-point validation below).
    if sort_by not in LEGAL_SORT_BY:
        allowed = ", ".join(sorted(LEGAL_SORT_BY))
        raise ValueError(
            f"Invalid sort_by: {sort_by!r}; expected one of: {allowed}"
        )

    # Validate provider/source_kind before they reach Milvus filter strings.
    # These flow through to a server-side expression as raw quoted values;
    # rejecting unknown inputs early prevents filter-expression injection.
    if provider is not None and provider not in LEGAL_PROVIDERS:
        allowed = ", ".join(sorted(LEGAL_PROVIDERS))
        raise ValueError(
            f"Invalid provider: {provider!r}; expected one of: {allowed}"
        )
    if source_kind is not None and source_kind not in LEGAL_SOURCE_KINDS:
        allowed = ", ".join(sorted(LEGAL_SOURCE_KINDS))
        raise ValueError(
            f"Invalid source_kind: {source_kind!r}; expected one of: {allowed}"
        )

    # Query-less recency listing: with no query there is no vector to
    # match and nothing for FTS, so fall back to a filter-only chronological
    # listing regardless of sort_by. Branches before embed_texts so an empty
    # string never produces a degenerate query vector.
    if not query or not query.strip():
        return _recent_listing(
            n, session_id=session_id, git_branch=git_branch,
            project_root=project_root, provider=provider,
            source_kind=source_kind, date_from=date_from, date_to=date_to,
            issue_id=issue_id, db_path=db_path,
        )

    # Expanded candidate pool for both engines
    fetch_n = n * 3

    # --- Vector search ---
    query_embedding = embed_texts([query], is_query=True)[0]

    filter_expr = _build_milvus_filter(
        session_id, git_branch, project_root, provider, source_kind,
        date_from, date_to, issue_id,
    )

    search_params = {"metric_type": "COSINE"}
    if _is_remote_uri(db_path or ""):
        search_params["params"] = {"ef": 128}  # HNSW search parameter

    with milvus_client(db_path) as client:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=fetch_n,
            filter=filter_expr,
            search_params=search_params,
            output_fields=_SEARCH_OUTPUT_FIELDS,
        )

    provider_defaults = default_provider_metadata()
    vector_results = []
    if results and results[0]:
        for hit in results[0]:
            # Milvus COSINE reports similarity (higher = closer) in "distance";
            # convert so the result's distance is a true distance.
            vector_results.append(
                _row_to_result(
                    hit["entity"], provider_defaults,
                    distance=1.0 - hit["distance"],
                )
            )

    # --- FTS5 keyword search ---
    fts_filters = {}
    if session_id:
        fts_filters["session_id"] = session_id
    if git_branch:
        fts_filters["git_branch"] = git_branch
    if project_root:
        fts_filters["project_root"] = _fts_project_filter(project_root)
    if provider:
        fts_filters["provider"] = provider
    if source_kind:
        fts_filters["source_kind"] = source_kind
    if date_from:
        fts_filters["timestamp_gte"] = (">=", date_from)
    if date_to:
        # Strip any existing time component before appending end-of-day so the
        # bound stays a valid ISO-8601 timestamp (date_to may arrive as a date
        # or a full datetime).
        fts_filters["timestamp_lte"] = ("<=", date_to.split("T")[0] + "T23:59:59")
    if issue_id:
        # Use the same comma-delimited containment match as the Milvus half, so
        # FTS results cannot include turns without the issue tag.
        token = _issue_id_containment_token(issue_id)
        fts_filters["issue_ids"] = ("like", f"%,{token},%")
    fts_results = _fts.search(query, n=fetch_n, filters=fts_filters or None, db_path=db_path)

    # --- Merge with RRF ---
    # Route *every* candidate set through rrf_merge (it handles an empty list
    # natively and is rank-order-preserving) so each row carries `_rrf_score` —
    # the only relevance signal that spans both vector and FTS-only rows. The
    # `_rrf_score` is kept alive until after scoring (see _rank_results).
    fts_weight = IDENTIFIER_FTS_WEIGHT if _is_identifier_query(query) else 1.0
    # Collapse the whole fused pool before the fetch_n cut, so sibling passages
    # of one turn cannot push other turns out of it.
    pool = len(vector_results) + len(fts_results)
    merged = _collapse_passages(
        rrf_merge(vector_results, fts_results, n=pool, fts_weight=fts_weight))[:fetch_n]

    if not merged:
        return []

    # Populate provider/source defaults; do NOT strip `_rrf_score` here — the
    # strategy dispatcher needs it and strips ranking scratch keys afterwards.
    merge_defaults = default_provider_metadata()
    for r in merged:
        r.setdefault("provider", merge_defaults["provider"])
        r.setdefault("source_kind", merge_defaults["source_kind"])
        r.setdefault("source_class", merge_defaults["source_class"])
        r.setdefault("logical_session_id", r.get("session_id", ""))

    merged = _rank_results(merged, sort_by, n)

    # If the FTS table was recently dropped + recreated and backfill hasn't
    # caught up, surface a one-line warning on each row's metadata so callers
    # can render it without us blocking the search.
    try:
        from fts_hybrid import fts_backfill_required
        if fts_backfill_required():
            notice = "keyword index rebuilding, results may be vector-only"
            for r in merged[:n]:
                r["_fts_warning"] = notice
    except Exception:  # pragma: no cover - sentinel check is best-effort
        pass

    return merged[:n]


_DISPLAY_CUT = "\n\n[cut: get_turns shows the whole turn]"


def _collapse_passages(merged: List[Dict]) -> List[Dict]:
    """Keep the first row of each parent and its passages, in ``merged`` order.

    A kept passage gets ``passage_index``. A kept turn goes through ``_cap_for_display``.
    """
    seen = set()
    out = []
    for row in merged:
        doc_id = row.get("doc_id", "")
        key = passages.passage_parent(doc_id) or doc_id
        if key in seen:
            continue
        seen.add(key)
        if row.get("chunk_type") == passages.PASSAGE_CHUNK_TYPE:
            row["passage_index"] = passages.passage_index(doc_id)
        out.append(_cap_for_display(row))
    return out


def _cap_for_display(row: Dict) -> Dict:
    """Cut a turn longer than ``MAX_TURN_CHARS`` plus the stored cut marker; ``get_turns`` keeps it whole."""
    if (row.get("chunk_type") == "turn"
            and len(row.get("content", "")) > MAX_TURN_CHARS + len(CUT_MARK)):
        row["content"] = row["content"][:MAX_TURN_CHARS] + _DISPLAY_CUT
    return row


_NOT_PASSAGE = f'chunk_type != "{passages.PASSAGE_CHUNK_TYPE}"'


def _without_passages(filter_expr: Optional[str]) -> str:
    """Return ``filter_expr`` restricted to rows that are not passages."""
    return f"({filter_expr}) && {_NOT_PASSAGE}" if filter_expr else _NOT_PASSAGE


def _recent_listing(n: int, session_id: Optional[str] = None,
                    git_branch: Optional[str] = None,
                    project_root: Optional[ProjectRoots] = None,
                    provider: Optional[str] = None,
                    source_kind: Optional[str] = None,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None,
                    issue_id: Optional[str] = None,
                    db_path: Optional[str] = None) -> List[Dict]:
    """Query-less chronological listing: filter-only streamed scan ranked by
    timestamp descending.

    Milvus ``query()`` has no ORDER BY, and a single capped ``query()`` call
    materializes every filter-matching row server-side — broad filters (e.g.
    ``provider="codex"`` across all projects) overflow Milvus's per-query
    result-size buffer and raise ``code=65535: query results exceed the limit
    size``. Rows therefore stream through ``_query_batches`` into a
    ``_NewestN`` bounded heap: memory stays O(n) and the kept set is the true
    global newest-``n`` regardless of collection size.
    """
    if n <= 0:
        return []

    filter_expr = _without_passages(_build_milvus_filter(
        session_id, git_branch, project_root, provider, source_kind,
        date_from, date_to, issue_id,
    ))

    collector = _NewestN(n)
    for batch in _query_batches(
        _SEARCH_OUTPUT_FIELDS, filter_expr=filter_expr, db_path=db_path,
    ):
        for row in batch:
            collector.add(row)

    rows = collector.result()
    if not rows:
        return []

    defaults = default_provider_metadata()
    # distance=None: no query vector exists for recency listings, so there is
    # no score to report.
    mapped = [_cap_for_display(_row_to_result(row, defaults, distance=None)) for row in rows]

    return _rank_results(mapped, "recency", n)


def _timeline_entry(row: Dict) -> Dict:
    """Normalize a structured or FTS source row to the timeline entry shape.

    Both sources are reduced to one shape so the merge/dedup/sort logic is
    source-agnostic. ``text`` is read from any of ``text``/``document``/
    ``content`` (the structured field uses ``document``; the FTS sidecar uses
    ``content``). Each entry carries provider, session_id, timestamp, role and
    chunk_type, text, doc_id, and issue_ids (Req 4.6).
    """
    role = row.get("role", row.get("chunk_type", ""))
    return {
        "doc_id": row.get("doc_id", ""),
        "timestamp": row.get("timestamp", ""),
        "provider": row.get("provider", ""),
        "session_id": row.get("session_id", ""),
        "role": role,
        "chunk_type": row.get("chunk_type", role),
        "text": row.get("text", row.get("document", row.get("content", ""))),
        "issue_ids": row.get("issue_ids", ""),
        "project_root": row.get("project_root", ""),
        "logical_session_id": row.get("logical_session_id", ""),
    }


class _ReverseKey:
    """Wraps a sort key so ``heapq`` (a min-heap) surfaces the *largest* key.

    ``heapq`` always pops the smallest element; inverting the comparison turns it
    into a max-heap, so ``heap[0]`` is the newest-kept timeline entry — the one
    evicted first when a strictly-older candidate arrives.
    """

    __slots__ = ("key",)

    def __init__(self, key):
        """Store the wrapped ``(timestamp, doc_id)`` sort key."""
        self.key = key

    def __lt__(self, other: "_ReverseKey") -> bool:
        """Invert ordering so a larger key compares as "smaller" to heapq."""
        return self.key > other.key


class _OldestN:
    """Bounded collector that retains the globally-oldest ``limit`` entries.

    Replaces draining every structured match into a list: rows stream
    in via :meth:`add` and at most ``limit`` are ever held, so memory is O(limit)
    no matter how many turns reference an issue. Correctness is preserved because
    the newest-kept entry is evicted only when a strictly-older candidate arrives,
    so the kept set always equals the true oldest ``limit`` seen so far — even
    though Milvus ``query_iterator`` yields rows in an undefined order.

    Entries are keyed by ``(parsed-UTC timestamp, doc_id)`` — timestamps go
    through ``_parse_timestamp_utc`` (the same ordering ``_rank_results`` uses)
    because the index mixes ``Z``/``+00:00``/naive ISO forms, which do not sort
    reliably as strings.
    Unparseable/empty timestamps fall back to ``datetime.min`` (oldest, kept
    first). Entries are deduplicated by ``doc_id``
    (first writer wins, so the structured source — streamed before the FTS
    fallback — takes precedence, matching the prior ``setdefault`` merge). Once a
    ``doc_id`` is rejected or evicted the kept maximum only decreases, so a later
    re-arrival of the same key is rejected again: no duplicates, no resurrection.
    """

    def __init__(self, limit: int):
        """Create a collector bounded to ``limit`` entries."""
        self._limit = limit
        self._heap: List = []   # (_ReverseKey(key), doc_id, entry); heap[0] = newest kept
        self._ids: set = set()  # doc_ids currently retained

    def add(self, entry: Dict) -> None:
        """Offer one timeline entry to the bounded set.

        No-op when its ``doc_id`` is already retained (dedup) or when the set is
        full and the entry is not strictly older than the newest currently kept.
        """
        # Coerce to str so a null (FTS rows may carry SQLite NULLs) or any
        # non-str field can't raise TypeError in the heap-key comparison: `or ""`
        # maps falsy/None to "", str() handles a truthy non-str (e.g. int epoch).
        doc_id = str(entry.get("doc_id") or "")
        if doc_id in self._ids:
            return
        parsed = _parse_timestamp_utc(str(entry.get("timestamp") or ""))
        key = (parsed or datetime.min.replace(tzinfo=timezone.utc), doc_id)
        item = (_ReverseKey(key), doc_id, entry)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, item)
            self._ids.add(doc_id)
        elif self._heap and key < self._heap[0][0].key:  # older than the newest kept
            self._ids.discard(self._heap[0][1])          # drop the evicted doc_id
            heapq.heapreplace(self._heap, item)          # evict newest, insert candidate
            self._ids.add(doc_id)

    def __len__(self) -> int:
        """Number of entries currently retained (never exceeds ``limit``)."""
        return len(self._heap)

    def result(self) -> List[Dict]:
        """Return the retained entries sorted oldest-first by (timestamp, doc_id)."""
        # Sort by the stored heap key (parsed-UTC timestamp, doc_id) so the
        # final order matches the retention order; the key is unique within
        # the set, so the comparison never reaches the entry dict.
        return [entry for _, _, entry in sorted(self._heap, key=lambda item: item[0].key)]


class _NewestN:
    """Bounded collector that retains the globally-newest ``limit`` rows.

    The newest-first mirror of :class:`_OldestN`, used by the query-less
    recency listing: rows stream in via :meth:`add` and at most
    ``limit`` are ever held, so memory is O(limit) no matter how many rows
    match the filter. A plain ``heapq`` min-heap suffices — ``heap[0]`` is the
    oldest-kept row, evicted only when a strictly-newer candidate arrives, so
    the kept set always equals the true newest ``limit`` seen so far even
    though Milvus ``query_iterator`` yields rows in an undefined order.

    Rows are keyed by ``(parsed-UTC timestamp, doc_id)`` — timestamps go
    through ``_parse_timestamp_utc`` (the same ordering ``_rank_results``
    uses) because the index mixes ``Z``/``+00:00``/naive ISO forms, which do
    not sort reliably as strings. Unparseable/empty timestamps fall back to
    ``datetime.min`` (oldest, evicted first). Rows are deduplicated by
    ``doc_id``; once a ``doc_id`` is rejected or evicted the kept minimum
    only increases, so a later re-arrival of the same key is rejected again.
    """

    def __init__(self, limit: int):
        """Create a collector bounded to ``limit`` rows."""
        self._limit = limit
        self._heap: List = []   # (key, doc_id, row); heap[0] = oldest kept
        self._ids: set = set()  # doc_ids currently retained

    def add(self, row: Dict) -> None:
        """Offer one Milvus row to the bounded set.

        No-op when its ``doc_id`` is already retained (dedup) or when the set
        is full and the row is not strictly newer than the oldest currently
        kept.
        """
        # Coerce to str so a None or non-str field can't raise TypeError in
        # the heap-key comparison (mirrors _OldestN).
        doc_id = str(row.get("doc_id") or "")
        if doc_id in self._ids:
            return
        parsed = _parse_timestamp_utc(str(row.get("timestamp") or ""))
        key = (parsed or datetime.min.replace(tzinfo=timezone.utc), doc_id)
        item = (key, doc_id, row)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, item)
            self._ids.add(doc_id)
        elif self._heap and key > self._heap[0][0]:  # newer than the oldest kept
            self._ids.discard(self._heap[0][1])      # drop the evicted doc_id
            heapq.heapreplace(self._heap, item)      # evict oldest, insert candidate
            self._ids.add(doc_id)

    def __len__(self) -> int:
        """Number of rows currently retained (never exceeds ``limit``)."""
        return len(self._heap)

    def result(self) -> List[Dict]:
        """Return the retained rows sorted newest-first by (timestamp, doc_id)."""
        # No explicit key: heap items are (key, doc_id, row) and key is unique
        # within the set, so tuple comparison never reaches the row dict.
        return [row for _, _, row in sorted(self._heap, reverse=True)]


def get_issue_timeline(issue_id: str, *, limit: int = DEFAULT_TIMELINE_LIMIT,
                       providers: Optional[List[str]] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       db_path: Optional[str] = None,
                       newest: bool = False) -> List[Dict]:
    """Cross-harness, deduplicated, chronological feed of turns for an issue.

    Unions a structured Milvus source (``issue_ids like "%,ID,%"`` streamed batch
    by batch via ``_query_batches``) with an FTS keyword fallback (a literal MATCH
    on the issue token) so any un-tagged turn remains visible (Req 6.1). Both
    sources are normalized to the same shape and routed through an ``_OldestN``
    collector that bounds memory to O(``limit``) while retaining the true oldest
    matches: it deduplicates by ``doc_id`` (Req 4.2/6.2, structured
    wins) and yields entries sorted oldest-first by ``(timestamp asc, doc_id asc)``
    (Req 4.1), capped at ``limit`` (Req 4.5). Each row is filtered to an optional
    ``providers`` subset (Req 4.4) and ``date_from``/``date_to`` bounds (Req 4.3)
    before it reaches the collector. An FTS failure is non-fatal — the feed
    degrades to the structured source with a logged warning.

    Args:
        issue_id: The tracker issue token (e.g. ``"SESF-25"``); uppercased and
            escaped at this boundary before it reaches Milvus.
        limit: Maximum number of entries to return (default
            ``DEFAULT_TIMELINE_LIMIT``).
        providers: Optional allow-list of provider names to keep.
        date_from: Optional inclusive ISO-8601 lower bound on ``timestamp``.
        date_to: Optional inclusive ISO-8601 upper bound on ``timestamp``.
        db_path: Optional Milvus DB path override (also drives the FTS sidecar).
        newest: When true, keep the newest ``limit`` turns (via ``_NewestN``) and
            return them newest first; the default keeps the oldest ``limit``.

    Returns:
        A list of timeline entry dicts, oldest first (newest first when
        ``newest`` is set); ``[]`` when nothing
        references the issue (Req 4.7).
    """
    if not is_valid_issue_token(issue_id):
        raise ValueError("issue_id must be a valid issue token like 'SESF-25'")
    canonical = issue_id.strip().upper()
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    filter_expr = _build_milvus_filter(
        None, None, None, None, None, date_from, date_to, issue_id=canonical,
    )

    output_fields = ["document", "doc_id", "session_id", "timestamp",
                     "chunk_type", "provider", "issue_ids", "project_root",
                     "logical_session_id"]

    # Per-entry filters, hoisted so they run during the stream over each source
    # row rather than over a fully-materialized list — only survivors reach the
    # bounded collector. Equivalent to the prior post-merge filtering.
    allowed = set(providers) if providers else None
    upper = date_to.split("T")[0] + "T23:59:59" if date_to else None
    token = f",{canonical},"

    def _passes(entry: Dict) -> bool:
        """Whether one entry clears the passage, provider, date-bound and issue-token guards."""
        if entry.get("chunk_type") == passages.PASSAGE_CHUNK_TYPE:
            return False
        if allowed is not None and entry.get("provider") not in allowed:  # Req 4.4
            return False
        ts = str(entry.get("timestamp") or "")  # coerce null/non-str so date compares can't TypeError
        if date_from and ts < date_from:  # Req 4.3 lower bound
            return False
        if upper and ts > upper:          # Req 4.3 upper bound
            return False
        # Boundary-aware guard: keep only turns that actually reference the token
        # (FTS may surface substring/word-stem noise; the structured source is
        # already exact via the comma-wrapped LIKE, so this is a no-op for it).
        return (token in (entry.get("issue_ids") or "")
                or _references_issue(entry.get("text", ""), canonical))

    # Structured source: stream Milvus matches batch by batch into a memory-bounded
    # oldest-N collector rather than draining them into a list. The
    # collector retains only the true oldest ``limit``, so memory is O(limit) no
    # matter how many turns reference the issue — and a server-side cap stays
    # impossible (Milvus query_iterator order is undefined; it can't sort by a
    # scalar), so streaming is the only correctness-preserving bound.
    collector = _NewestN(limit) if newest else _OldestN(limit)
    structured_count = 0
    for batch in _query_batches(output_fields, filter_expr=filter_expr, db_path=db_path):
        structured_count += len(batch)
        for row in batch:
            entry = _timeline_entry(row)
            if _passes(entry):
                collector.add(entry)
    if structured_count > _TIMELINE_ROWS_WARN:
        logger.warning(
            "issue timeline for %s matched %d structured rows (>%d) — oldest-%d "
            "retained via bounded pagination; narrow with date_from/date_to to "
            "shrink the scan",
            canonical, structured_count, _TIMELINE_ROWS_WARN, limit,
        )

    # FTS keyword fallback (Req 6.1) for turns not yet tagged with issue_ids
    # (e.g. pre-DEVS-39-reindex rows). Quote the token as an FTS5 phrase so the
    # hyphen isn't parsed as query syntax (a bare ``SESF-25`` MATCH errors/misses);
    # the boundary guard in ``_passes`` enforces the exact issue token. Push the
    # date bounds into the FTS query so the fetch window isn't spent on out-of-range
    # rows (provider is a list, so it stays a post-filter in ``_passes``). Fetch a
    # generous window (not just ``limit``) so the chronological slice isn't biased
    # by BM25 rank, then route the hits through the SAME bounded collector so the
    # combined feed stays bounded. Non-fatal — degrade to the structured source on
    # any FTS error.
    fts_filters: Dict = {}
    if date_from:
        fts_filters["timestamp_gte"] = (">=", date_from)
    if date_to:
        fts_filters["timestamp_lte"] = ("<=", upper)
    try:
        fts_hits = _fts.search(
            f'"{canonical}"', n=max(limit, _TIMELINE_FTS_FETCH_CAP),
            filters=fts_filters or None, db_path=db_path,
        )
    except Exception as e:
        logger.warning("FTS fallback failed for issue timeline %s (non-fatal): %s", canonical, _scrub_exception(e))
        fts_hits = []

    for row in fts_hits:
        entry = _timeline_entry(row)
        if _passes(entry):
            collector.add(entry)

    # Dedup by doc_id (structured wins, streamed first), the oldest-first sort and
    # the ``limit`` slice all happen inside the collector (Req 4.1/4.2/4.5/6.2).
    return collector.result()


def _references_issue(text: str, canonical: str) -> bool:
    """Whether text contains the issue token as a whole word (case-insensitive).

    Boundary-anchored so ``SESF-42`` does not match ``SESF-420`` (Req 6.3).
    """
    if not text:
        return False
    pattern = r"\b" + re.escape(canonical) + r"\b"
    return re.search(pattern, text.upper()) is not None


async def get_issue_timeline_async(issue_id: str, *,
                                   limit: int = DEFAULT_TIMELINE_LIMIT,
                                   providers: Optional[List[str]] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None,
                                   db_path: Optional[str] = None) -> List[Dict]:
    """Async wrapper over ``get_issue_timeline`` (mirrors ``search_async``).

    Runs the synchronous core in the default executor. The timeline does no
    embedding, so it does not require the MLX embed semaphore.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_issue_timeline(
            issue_id, limit=limit, providers=providers,
            date_from=date_from, date_to=date_to, db_path=db_path,
        ),
    )


def _parse_timestamp_utc(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime.

    Handles both naive and aware inputs (naive is assumed UTC). Returns None
    on empty/unparseable input so callers can apply a deterministic fallback.
    """
    if not ts:
        return None
    # Real-world transcripts (Claude Code JSONL et al.) emit a trailing "Z" for
    # UTC. datetime.fromisoformat only accepts it on Python 3.11+, so normalize
    # explicitly to stay correct regardless of interpreter version.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recency_score(ts: str, now: datetime, decay_days: float) -> float:
    """Age-aware recency in (0, 1]: exp(-days_old / decay_days).

    A result `decay_days` old scores ≈0.367. Missing/unparseable timestamps
    return a neutral fallback (EARS-8). Future-dated timestamps clamp to
    days_old=0 so clock skew can't earn an inflated boost.
    """
    parsed = _parse_timestamp_utc(ts)
    if parsed is None:
        return MISSING_TIMESTAMP_RECENCY
    days_old = max(0.0, (now - parsed).total_seconds() / 86400.0)
    return math.exp(-days_old / decay_days)


def _semantic_scores(results: List[Dict]) -> List[float]:
    """Min-max normalize each row's `_rrf_score` into [0, 1].

    A degenerate pool (single row, or all-equal scores) has a zero range; rather
    than dividing by zero, every row gets a neutral 1.0.
    """
    raw = [r.get("_rrf_score", 0.0) for r in results]
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    span = hi - lo
    if span == 0:
        return [1.0] * len(raw)
    return [(s - lo) / span for s in raw]


def ranking_settings() -> Dict[str, Union[float, int]]:
    """Return the hybrid ranking settings this process applies, read from the environment.

    ``SESSIONFLOW_RECENCY_WEIGHT`` outside ``[0, 1]`` and an unparsable value fall
    back to the default; ``SESSIONFLOW_RECENCY_DECAY_DAYS`` is at least 1.
    """
    return {
        "recency_weight": _env_float(
            "SESSIONFLOW_RECENCY_WEIGHT", RECENCY_WEIGHT_DEFAULT, 0.0, 1.0
        ),
        "recency_decay_days": _env_int(
            "SESSIONFLOW_RECENCY_DECAY_DAYS", RECENCY_DECAY_DAYS_DEFAULT, minimum=1
        ),
    }


def _rank_results(results: List[Dict], sort_by: str, n: int,
                  now: Optional[datetime] = None, *,
                  timestamp_key: str = "timestamp",
                  tie_break: Optional[str] = None) -> List[Dict]:
    """Order the merged candidate pool by the chosen strategy, then truncate.

    - ``relevance``: keep the post-RRF order (no recency re-rank).
    - ``recency``: re-rank by ``timestamp_key`` descending; missing-timestamp rows last.
    - ``hybrid``: final = (1-w)*semantic_score + w*recency_score.

    ``tie_break`` names a key that orders exact ties descending; without it,
    ties keep their incoming order. Strips ranking scratch keys (`_rrf_score`,
    `_score`, `_semantic_score`, `_recency_score`) before returning but keeps
    non-score engine metadata such as `_fts_warning`.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    def _with_tie(primary):
        if tie_break is None:
            return primary
        return lambda r: (primary(r), str(r.get(tie_break, "")))

    if sort_by == "relevance":
        ranked = list(results)
    elif sort_by == "recency":
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        ranked = sorted(
            results,
            key=_with_tie(lambda r: _parse_timestamp_utc(r.get(timestamp_key, "")) or _epoch),
            reverse=True,
        )
    else:  # hybrid
        settings = ranking_settings()
        weight = settings["recency_weight"]
        decay_days = settings["recency_decay_days"]
        semantic = _semantic_scores(results)
        for r, sem in zip(results, semantic, strict=True):
            rec = _recency_score(r.get(timestamp_key, ""), now, decay_days)
            r["_semantic_score"] = sem
            r["_recency_score"] = rec
            r["_score"] = (1 - weight) * sem + weight * rec
        ranked = sorted(
            results, key=_with_tie(lambda r: r.get("_score", 0.0)), reverse=True
        )

    for r in ranked:
        for key in _RANKING_SCRATCH_KEYS:
            r.pop(key, None)

    return ranked[:n]


_CHICAGO = None
# Timeline rows read per issue when grouping sessions for a memory's context.
SESSION_SCAN_LIMIT = 500
# Characters of each source-session turn returned with a memory's context.
SESSION_TURN_CHARS = 2000


def chicago_day_bounds(day) -> tuple:
    """Return the half-open UTC interval ``[start, end)`` of one America/Chicago day."""
    global _CHICAGO
    if _CHICAGO is None:
        from zoneinfo import ZoneInfo

        _CHICAGO = ZoneInfo("America/Chicago")
    start = datetime(day.year, day.month, day.day, tzinfo=_CHICAGO)
    nxt = day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=_CHICAGO)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _session_key(row: Dict) -> tuple:
    # Codex rows of one thread carry per-payload session ids; the thread id
    # lives in logical_session_id, so Codex groups on it.
    provider = row.get("provider") or ""
    if provider == "codex":
        return provider, row.get("logical_session_id") or row.get("session_id") or ""
    return provider, row.get("session_id") or ""


def _root_matches(root: str, roots: List[str]) -> bool:
    return any(root == r or root.startswith(r.rstrip("/") + "/") for r in roots)


def _rows_on_day(day, project_roots: List[str], db_path: Optional[str]) -> List[Dict]:
    start, end = chicago_day_bounds(day)
    # String bounds a day wide on each side; the exact Chicago bounds apply after parsing.
    lo = (start - timedelta(days=1)).strftime("%Y-%m-%d")
    hi = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    roots = " or ".join(f'project_root like "{_escape_filter_scalar(r.rstrip("/"))}%"'
                        for r in project_roots)
    filter_expr = f'timestamp >= "{lo}" and timestamp < "{hi}" and ({roots}) and {_NOT_PASSAGE}'
    fields = ["doc_id", "session_id", "logical_session_id", "provider", "timestamp",
              "project_root", "issue_ids"]
    rows: List[Dict] = []
    for batch in _query_batches(fields, filter_expr=filter_expr, db_path=db_path):
        rows.extend(batch)
    return rows


def sessions_by_issue(issue_ids: List[str], *, limit: int,
                      project_roots: Optional[List[str]] = None, day=None,
                      exclude_session: Optional[str] = None,
                      db_path: Optional[str] = None) -> List[Dict]:
    """Group transcript turns into sessions, newest first, for a memory's context.

    Rows come from each issue's timeline; with no issue IDs and a ``day``, from
    that day's turns in ``project_roots``. ``project_roots`` keeps a root and its
    subfolders (worktrees). ``day`` keeps one America/Chicago day.

    Returns:
        Dicts of ``session_id``, ``provider``, ``turns``, ``first_timestamp``,
        ``last_timestamp`` and ``issue_ids``, at most ``limit``.
    """
    db_path = db_path or _server_db_path
    rows: List[Dict] = []
    if issue_ids:
        for issue in issue_ids:
            rows.extend(get_issue_timeline(issue, limit=SESSION_SCAN_LIMIT,
                                           db_path=db_path, newest=True))
    elif day is not None and project_roots:
        rows = _rows_on_day(day, project_roots, db_path)
    bounds = chicago_day_bounds(day) if day is not None else None
    sessions: Dict[tuple, Dict] = {}
    seen_docs = set()
    for row in rows:
        doc = row.get("doc_id")
        if doc in seen_docs:
            continue
        seen_docs.add(doc)
        if project_roots and not _root_matches(row.get("project_root") or "", project_roots):
            continue
        ts = _parse_timestamp_utc(row.get("timestamp", ""))
        if bounds is not None and (ts is None or not bounds[0] <= ts < bounds[1]):
            continue
        key = _session_key(row)
        if not key[1] or key[1] == exclude_session:
            continue
        entry = sessions.setdefault(key, {
            "session_id": key[1], "provider": key[0], "turns": 0, "issue_ids": set(),
            "first_timestamp": row.get("timestamp", ""), "last_timestamp": row.get("timestamp", ""),
            "_first": ts, "_last": ts,
        })
        entry["turns"] += 1
        entry["issue_ids"].update(i for i in (row.get("issue_ids") or "").split(",") if i)
        if ts is not None and (entry["_first"] is None or ts < entry["_first"]):
            entry["_first"], entry["first_timestamp"] = ts, row.get("timestamp", "")
        if ts is not None and (entry["_last"] is None or ts > entry["_last"]):
            entry["_last"], entry["last_timestamp"] = ts, row.get("timestamp", "")
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    ordered = sorted(sessions.values(), key=lambda s: s["_last"] or _epoch, reverse=True)
    out = []
    for s in ordered[:limit]:
        s.pop("_first")
        s.pop("_last")
        s["issue_ids"] = sorted(s["issue_ids"])
        out.append(s)
    return out


def session_turns(session_id: str, provider: Optional[str], *, limit: int,
                  db_path: Optional[str] = None) -> List[Dict]:
    """Return the last ``limit`` turns of one session in order, each trimmed for display.

    Codex sessions are matched on ``logical_session_id``, others on ``session_id``.
    """
    db_path = db_path or _server_db_path
    column = "logical_session_id" if provider == "codex" else "session_id"
    parts = [f'{column} == "{_escape_filter_scalar(session_id)}"', _NOT_PASSAGE]
    if provider:
        parts.append(f'provider == "{_escape_filter_scalar(provider)}"')
    fields = ["turn_index", "timestamp", "chunk_type", "document"]
    rows: List[Dict] = []
    for batch in _query_batches(fields, filter_expr=" and ".join(parts), db_path=db_path):
        rows.extend(batch)
    rows.sort(key=lambda r: (r.get("turn_index") or 0))
    return [{"turn_index": r.get("turn_index"), "timestamp": r.get("timestamp", ""),
             "chunk_type": r.get("chunk_type", ""),
             "content": (r.get("document") or "")[:SESSION_TURN_CHARS]}
            for r in rows[-limit:]]


MEMORY_RANGE_LIMIT = 16384
"""Milvus caps one search at this many hits; a range search returns at most this many."""


def memory_threshold_search(client, vector: List[float], *, project: str, user_id: str,
                            generation: int, threshold: float,
                            collection: str = "memories",
                            exclude_memory_id: Optional[str] = None) -> List[Dict]:
    """Return every active memory in one project whose vector is at least ``threshold`` similar.

    A COSINE range search, not a top-N search, so no match above the threshold
    is dropped. Only vectors of the current ``generation`` are compared.

    Returns:
        ``{"memory_id", "score"}`` dicts, most similar first.
    """
    parts = [
        f'project == "{_escape_filter_scalar(project)}"',
        f'user_id == "{_escape_filter_scalar(user_id)}"',
        'status == "active"',
        f"generation == {int(generation)}",
    ]
    if exclude_memory_id:
        parts.append(f'memory_id != "{_escape_filter_scalar(exclude_memory_id)}"')
    results = client.search(
        collection_name=collection, data=[vector], limit=MEMORY_RANGE_LIMIT,
        filter=" and ".join(parts), output_fields=["memory_id"],
        search_params={"metric_type": "COSINE",
                       "params": {"radius": float(threshold), "range_filter": 1.0}},
    )
    hits = [{"memory_id": h["entity"]["memory_id"], "score": float(h["distance"])}
            for h in (results[0] if results else []) if h["distance"] >= threshold]
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def get_turns(session_id: str, turn_index: int, context: int = 2,
              db_path: Optional[str] = None) -> List[Dict]:
    """Retrieve turns around a specific turn_index within a session.

    turn_index is a byte offset into the transcript file. context is the
    number of neighboring turns (before and after) to include. We fetch all
    turns for the session, sort by turn_index, find the target, and return
    the surrounding window.

    Returns turns sorted by turn_index ascending, with the same field
    mapping as search() (document → content).
    """
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=_without_passages(f'session_id == "{_escape_filter_scalar(session_id)}"'),
            output_fields=["document", "doc_id", "session_id", "transcript_file",
                           "turn_index", "timestamp", "git_branch", "chunk_type",
                           "logical_session_id", "provider", "source_kind",
                           "source_class", "source_id", "source_path"],
            limit=16384,
        )

    if not results:
        return []

    # Sort all turns by turn_index (byte offset)
    results.sort(key=lambda r: r.get("turn_index", 0))

    # Find the target turn (closest match to requested turn_index)
    target_idx = 0
    min_dist = float("inf")
    for i, row in enumerate(results):
        dist = abs(row.get("turn_index", 0) - turn_index)
        if dist < min_dist:
            min_dist = dist
            target_idx = i

    # Extract window: context turns before and after
    start = max(0, target_idx - context)
    end = min(len(results), target_idx + context + 1)

    turn_defaults = default_provider_metadata()
    formatted = []
    for row in results[start:end]:
        formatted.append({
            "content": row["document"],
            "doc_id": row.get("doc_id", ""),
            "session_id": row.get("session_id", ""),
            "logical_session_id": row.get("logical_session_id", row.get("session_id", "")),
            "provider": row.get("provider", turn_defaults["provider"]),
            "source_kind": row.get("source_kind", turn_defaults["source_kind"]),
            "source_class": row.get("source_class", turn_defaults["source_class"]),
            "source_id": row.get("source_id", ""),
            "source_path": row.get("source_path", row.get("transcript_file", "")),
            "transcript_file": row.get("transcript_file", ""),
            "turn_index": row.get("turn_index", 0),
            "timestamp": row.get("timestamp", ""),
            "git_branch": row.get("git_branch", ""),
            "chunk_type": row.get("chunk_type", ""),
        })

    return formatted


def get_stats(project_root: Optional[ProjectRoots] = None,
              db_path: Optional[str] = None) -> Dict:
    """Get index statistics. Optionally filter to one project root or a list of roots."""
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            # Keep the redaction surface on every return path so
            # get_stats()["redaction"] never KeyErrors before the collection exists.
            return {
                "total_turns": 0,
                "sessions": 0,
                "by_type": {},
                "providers": {},
                "redaction": _redaction_status(),
                "boilerplate": _boilerplate_status(),
            }

    # Query for breakdowns (capped by Milvus offset limit)
    all_results = _query_all(
        ["session_id", "chunk_type", "git_branch", "project_root", "provider"],
        filter_expr=_project_root_filter(project_root),
        db_path=db_path,
    )

    # FTS holds passages too, so the lag check compares every row.
    stored = len(all_results)
    total = sum(1 for r in all_results if r.get("chunk_type") != passages.PASSAGE_CHUNK_TYPE)
    sessions = set(r["session_id"] for r in all_results if r.get("session_id"))
    branches = set(r["git_branch"] for r in all_results if r.get("git_branch"))

    by_type = {}
    providers = {}
    defaults = default_provider_metadata()
    for r in all_results:
        t = r.get("chunk_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        provider_name = r.get("provider", defaults["provider"])
        providers[provider_name] = providers.get(provider_name, 0) + 1

    stats = {
        "total_turns": total,
        "sessions": len(sessions),
        "branches": sorted(branches),
        "by_type": by_type,
        "providers": providers,
    }

    # Additive, project-scoped FTS lag keys. Pass the Milvus total
    # we just computed so fts_lag_status does not re-enter get_stats.
    lag = fts_lag_status(
        db_path=db_path, project_root=project_root, milvus_turn_count=stored
    )
    stats["fts_row_count"] = lag["fts_row_count"]
    stats["fts_lag"] = lag["fts_lag"]
    stats["fts_backfill_required"] = lag["fts_backfill_required"]

    # Operator-facing redaction status + durable per-rule counts (AC-10).
    stats["redaction"] = _redaction_status()
    stats["boilerplate"] = _boilerplate_status()
    return stats


def _count_milvus_turns(db_path=None, project_root=None) -> int:
    """Count Milvus turn rows through the DRIFT-TOLERANT migration client.

    ``fts_lag_status`` cannot route this count through ``get_stats`` (which opens
    the drift-GUARDED ``milvus_client`` and RAISES under persistent schema drift):
    the lag readout must survive the very incident it exists to report. This helper
    issues a server-side ``count(*)`` query via the migration client, applying the
    same ``project_root`` filter ``get_stats`` builds so the lag stays
    apples-to-apples — without streaming every ``doc_id`` row back to the client.

    Args:
        db_path: Milvus DB path.
        project_root: when provided, scope the count to that project.

    Returns:
        int: the number of turn rows (project-scoped when ``project_root`` is set).
    """
    filter_expr = _project_root_filter(project_root)
    with milvus_client_for_migration(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return 0
        res = client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr or "",
            output_fields=["count(*)"],
        )
        return int(res[0]["count(*)"]) if res else 0


def fts_lag_status(
    db_path: Optional[str] = None,
    project_root: Optional[ProjectRoots] = None,
    milvus_turn_count: Optional[int] = None,
) -> Dict[str, object]:
    """Return static FTS-vs-Milvus lag data for observability.

    Computes the difference between the Milvus turn count and the FTS row count,
    both scoped to ``project_root`` when provided. The lag is static — no worker
    state (``consecutive_failures``/``last_error``) is included here.

    Args:
        db_path: Milvus DB path; also derives the FTS sidecar path.
        project_root: when provided, scope both counts to that project.
        milvus_turn_count: pre-computed Milvus total; when None it is counted via
            the drift-tolerant ``_count_milvus_turns`` (get_stats passes its own
            already-computed total to avoid a double scan and recursion).

    Returns:
        dict: keys ``milvus_turn_count``, ``fts_row_count``, ``fts_lag``
        (milvus_turn_count - fts_row_count), and ``fts_backfill_required``.
    """
    if milvus_turn_count is None:
        # Standalone path (e.g. /health): count through the drift-tolerant
        # migration client, NOT get_stats, so observability survives drift (D-6).
        milvus_turn_count = _count_milvus_turns(
            db_path=db_path, project_root=project_root
        )

    # FTS count on the calling thread: open an
    # ephemeral connection, count, and close it.
    conn = _fts.connection(db_path)
    try:
        fts_row_count = _fts.count_rows(conn, project_root=project_root)
    finally:
        _fts.close_ephemeral(conn)

    fts_lag = milvus_turn_count - fts_row_count
    return {
        "milvus_turn_count": milvus_turn_count,
        "fts_row_count": fts_row_count,
        "fts_lag": fts_lag,
        # Pure sentinel state (D-7/AC-6): normal indexing lag must NOT report a
        # required rebuild — only the explicit backfill sentinel does. Referenced
        # via the module-level binding so it stays one patchable seam.
        "fts_backfill_required": fts_backfill_required(),
    }


def _fts_roots_complete(conn, db_path: Optional[str]) -> bool:
    """Return True when the FTS sidecar holds every indexed turn.

    A set backfill sentinel or an FTS row count below the Milvus turn count
    means FTS is rebuilding or missed inserts. A failed Milvus count keeps the
    FTS answer.
    """
    if fts_backfill_required():
        return False
    try:
        return _fts.count_rows(conn) >= _count_milvus_turns(db_path=db_path)
    except Exception as e:
        logger.warning("Milvus turn count failed, keeping FTS roots: %s", _scrub_exception(e))
        return True


def known_project_roots(db_path: Optional[str] = None) -> List[str]:
    """Return every distinct ``project_root`` in the index.

    Reads the FTS sidecar, which is local, and reads Milvus instead when FTS
    fails or is incomplete. The result is cached per ``db_path`` for
    ``KNOWN_ROOTS_TTL_S``.
    """
    cached = _known_roots_cache.get(db_path)
    if cached and time.monotonic() - cached[0] < KNOWN_ROOTS_TTL_S:
        return cached[1]
    roots = None
    try:
        conn = _fts.connection(db_path)
        try:
            if _fts_roots_complete(conn, db_path):
                roots = _fts.distinct_values(conn, "project_root")
        finally:
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS project_root scan failed, reading Milvus: %s", _scrub_exception(e))
    if roots is None:
        found = set()
        for batch in _query_batches(["project_root"], db_path=db_path):
            found.update(row.get("project_root") or "" for row in batch)
        roots = list(found)
    roots = sorted(r for r in roots if r)
    _known_roots_cache[db_path] = (time.monotonic(), roots)
    return roots


def _query_batches(output_fields: list, batch_size: int = 1000,
                   filter_expr: Optional[str] = None,
                   db_path: Optional[str] = None) -> Iterator[list]:
    """Yield Milvus query_iterator results one batch at a time.

    Uses pymilvus's server-side iterator instead of offset pagination so the
    full collection is drained regardless of size. Offset pagination stops at
    16,384 rows.
    """
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return
        iterator = client.query_iterator(
            collection_name=COLLECTION_NAME,
            batch_size=batch_size,
            filter=filter_expr or "",
            output_fields=output_fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                yield batch
        finally:
            iterator.close()


def _query_all(output_fields: list, batch_size: int = 1000,
               filter_expr: Optional[str] = None,
               db_path: Optional[str] = None) -> list:
    """Query all rows via Milvus query_iterator. Optional filter expression.

    Keeps the public list-returning behavior for callers that need aggregate
    results while allowing streaming callers to use _query_batches directly.
    """
    all_results = []
    for batch in _query_batches(output_fields, batch_size, filter_expr, db_path):
        all_results.extend(batch)
    return all_results


# --- Cleanup operations ---

def delete_by_session(session_id: str, db_path: Optional[str] = None) -> int:
    """Delete all turns for a given session ID."""
    escaped_sid = _escape_filter_scalar(session_id)
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{escaped_sid}"',
            output_fields=["id"],
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'session_id == "{escaped_sid}"',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete(conn, "session_id", session_id)
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete by session failed (non-fatal): %s", _scrub_exception(e))

    return len(results)


def _pk_from_doc_id(doc_id: str) -> int:
    """Derive the stable INT64 Milvus primary key from a doc_id (SHA-256, 60-bit)."""
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:15], 16)


class UpsertResult(NamedTuple):
    """Outcome of an in-place document overwrite across both stores.

    Attributes:
        milvus_ok: True when the Milvus upsert (atomic by primary key) succeeded.
        fts_ok: True when the FTS delete-then-insert of the redacted content
            succeeded. Reported distinctly from ``milvus_ok``: a Milvus-clean but
            FTS-failed row still satisfies a keyword search for the old secret, so
            the caller must treat ``fts_ok is False`` as unfinished/retryable and
            NOT mark the row done. FTS background healing only hydrates *missing*
            doc_ids, so a stale row left in place is never auto-redacted.
    """

    milvus_ok: bool
    fts_ok: bool


class DeleteResult(NamedTuple):
    """Outcome of a doc-id-scoped dual delete across both stores.

    Attributes:
        deleted: Number of Milvus rows deleted (0 or 1).
        fts_ok: True when the FTS delete of ``doc_id`` succeeded. Surfaced
            distinctly so the drop path can mirror the redact contract: a turn is
            only marked sanitized when its FTS row is gone too. A swallowed FTS
            failure would otherwise leave the secret-bearing keyword row alive
            while the caller marks the doc done.
    """

    deleted: int
    fts_ok: bool


def get_row_by_doc_id(doc_id: str,
                      db_path: Optional[str] = None) -> Optional[dict]:
    """Fetch a single Milvus row by ``doc_id``, or ``None`` when absent.

    Used by the sanitizer to read an affected row's full payload
    (document + metadata) just-in-time at apply time, so a large scan never has
    to cache every row's text in memory. Looks the row up by its integer primary
    key (``_pk_from_doc_id``) for an O(1) point lookup rather than a VARCHAR scan;
    the numeric filter needs no escaping. The returned row still carries every
    field, including the original ``id`` and ``doc_id``.

    Args:
        doc_id: Stable document id of the turn to read.
        db_path: Optional Milvus DB path.

    Returns:
        The row as a plain dict (every field, ``output_fields=["*"]``), or
        ``None`` when the collection is missing or no row matches.
    """
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return None
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id == {_pk_from_doc_id(doc_id)}",
            output_fields=["*"],
        )
        if not rows:
            return None
        return dict(rows[0])


def upsert_document(doc_id: str, *, new_document: str,
                    new_vector: Optional[list] = None,
                    db_path: Optional[str] = None) -> UpsertResult:
    """Overwrite a single turn's document (and optionally vector) in place, then rewrite FTS.

    Reads the existing Milvus row for ``doc_id`` (preserving every metadata
    field), swaps in the new ``document`` (UTF-8 truncated to the VARCHAR cap),
    and upserts it by primary key (atomic — the row is either fully old or fully
    new). Then rewrites the FTS sidecar as a delete-then-insert of the new
    content (``FTSIndex.insert`` skips existing doc_ids, so the prior row must be
    deleted first).

    When ``new_vector`` is None, the existing embedding is left untouched and only
    the ``document`` field + FTS row are converged. This is the resume/no-spans
    path: a row already redacted in Milvus must not have its fixed-dim HNSW vector
    clobbered by a zero-length list — so callers that only need to retry the FTS
    rewrite pass ``new_vector=None``.

    Unlike ``delete_by_session``, an FTS failure here is surfaced rather than
    swallowed: a failed rewrite leaves the old secret-bearing row searchable, and
    the FTS heal path only hydrates *missing* doc_ids, so it would never be
    re-redacted by background healing.

    Args:
        doc_id: Stable document id of the turn to overwrite.
        new_document: Replacement document text (the redacted content).
        new_vector: Replacement embedding vector for ``new_document``. When None,
            the stored vector is preserved (document + FTS converge only).
        db_path: Optional Milvus DB path; also derives the FTS sidecar path.

    Returns:
        UpsertResult: ``milvus_ok`` and ``fts_ok`` reported independently. The
        row is fully sanitized only when both are True.
    """
    milvus_ok = False
    row: dict = {}
    try:
        with milvus_client(db_path) as client:
            if not client.has_collection(COLLECTION_NAME):
                return UpsertResult(False, False)
            # Point lookup by integer PK (O(1)) rather than a VARCHAR doc_id
            # scan; kept inline so the fetched row stays inside this client/
            # upsert context for the in-place rewrite.
            rows = client.query(
                collection_name=COLLECTION_NAME,
                filter=f"id == {_pk_from_doc_id(doc_id)}",
                output_fields=["*"],
            )
            if not rows:
                return UpsertResult(False, False)
            row = dict(rows[0])
            row["document"] = _truncate_utf8(new_document, 65535)
            # Leave the existing fixed-dim vector intact when no replacement is
            # supplied — never write a 0-length vector into an HNSW/COSINE row.
            if new_vector is not None:
                row["vector"] = new_vector
            client.upsert(collection_name=COLLECTION_NAME, data=[row])
        milvus_ok = True
    except Exception as e:
        logger.warning("Milvus upsert failed for doc_id: %s", _scrub_exception(e))
        return UpsertResult(False, False)

    return UpsertResult(milvus_ok, _rewrite_fts_row(doc_id, row, db_path))


def _rewrite_fts_row(doc_id: str, row: dict, db_path: Optional[str]) -> bool:
    """Replace one FTS row with a record built from the Milvus ``row``; True on success.

    Delete-then-insert (insert dedups by doc_id, so the stale row must go
    first). Failure is surfaced, never swallowed. The record is built from the
    already-fetched row so every metadata column the normal ingest FTS record
    carries is preserved. Without a ``db_path`` there is no sidecar, so it
    reports success.
    """
    if not db_path:
        return True
    conn = None
    try:
        conn = _fts.connection(db_path)
        _fts.delete(conn, "doc_id", doc_id)
        _fts.insert(conn, [{
            "doc_id": doc_id,
            # Use the UTF-8-truncated text stored in Milvus (row["document"]),
            # not the caller's text — a redacted payload can expand past the
            # 65535 cap, and the two stores must index identical content.
            "content": row["document"],
            "session_id": row.get("session_id", ""),
            "logical_session_id": row.get(
                "logical_session_id", row.get("session_id", "")),
            "provider": row.get("provider", ""),
            "source_kind": row.get("source_kind", ""),
            "source_class": row.get("source_class", ""),
            "source_id": row.get("source_id", ""),
            "source_path": row.get("source_path", ""),
            "git_branch": row.get("git_branch", ""),
            "turn_index": row.get("turn_index", 0),
            "timestamp": row.get("timestamp", ""),
            "chunk_type": row.get("chunk_type", "turn"),
            "project_root": row.get("project_root", ""),
            "issue_ids": row.get("issue_ids", ""),
        }])
        return True
    except Exception as e:
        logger.warning("FTS rewrite failed for doc_id: %s", _scrub_exception(e))
        return False
    finally:
        # Always release the ephemeral connection, even when the rewrite
        # raised partway through — otherwise a failed run leaks a SQLite
        # handle per affected row.
        if conn is not None:
            _fts.close_ephemeral(conn)


# Metadata a caller may rewrite in place without re-embedding.
RETAGGABLE_FIELDS = frozenset({"project_root", "issue_ids", "timestamp", "git_branch"})


def update_row_metadata(doc_id: str, *, db_path: Optional[str] = None,
                        **fields: str) -> UpsertResult:
    """Rewrite metadata fields on one turn in place, keeping its document and vector.

    Reads the Milvus row by primary key, sets ``fields``, upserts it (atomic by
    primary key) and then rewrites the FTS row the same way ``upsert_document``
    does. Nothing is embedded. An FTS failure is reported as ``fts_ok=False``;
    :func:`find_doc_ids_by_project_root` also reads FTS, so a re-run picks the
    turn up again.

    Args:
        doc_id: Stable document id of the turn to rewrite.
        db_path: Optional Milvus DB path; also derives the FTS sidecar path.
        **fields: New string values, keyed by names in ``RETAGGABLE_FIELDS``.

    Returns:
        UpsertResult: ``(False, False)`` when the row is missing or the Milvus
        upsert failed; otherwise ``milvus_ok`` is True and ``fts_ok`` reports
        the sidecar rewrite.

    Raises:
        ValueError: ``fields`` is empty, names a field outside
            ``RETAGGABLE_FIELDS``, or holds a non-string value.
    """
    if not fields:
        raise ValueError("update_row_metadata needs at least one field")
    unknown = set(fields) - RETAGGABLE_FIELDS
    if unknown:
        raise ValueError(f"Fields not retaggable: {', '.join(sorted(unknown))}")
    if not all(isinstance(value, str) for value in fields.values()):
        raise ValueError("Retagged field values must be strings")

    try:
        with milvus_client(db_path) as client:
            if not client.has_collection(COLLECTION_NAME):
                return UpsertResult(False, False)
            rows = client.query(
                collection_name=COLLECTION_NAME,
                filter=f"id == {_pk_from_doc_id(doc_id)}",
                output_fields=["*"],
            )
            if not rows:
                return UpsertResult(False, False)
            row = dict(rows[0])
            row.update(fields)
            client.upsert(collection_name=COLLECTION_NAME, data=[row])
    except Exception as e:
        logger.warning("Milvus metadata upsert failed for doc_id: %s", _scrub_exception(e))
        return UpsertResult(False, False)

    return UpsertResult(True, _rewrite_fts_row(doc_id, row, db_path))


def find_doc_ids_by_project_root(prefix: str, *, provider: Optional[str] = None,
                                 db_path: Optional[str] = None) -> Dict[str, str]:
    """Find turns whose ``project_root`` is ``prefix`` or lies under it.

    Matches on a path-component boundary (``/a/CoWork`` and ``/a/CoWork/x``,
    never ``/a/CoWorkX``). Milvus is streamed with a LIKE prefilter and each hit
    is re-checked exactly, since LIKE treats ``%`` and ``_`` as wildcards. The
    FTS sidecar is read too, so a turn whose Milvus row was already rewritten
    but whose FTS rewrite failed is still found. An FTS read failure is logged
    and the Milvus results are returned.

    Args:
        prefix: Absolute project root to look under.
        provider: Optional provider to restrict to.
        db_path: Optional Milvus DB path; also derives the FTS sidecar path.

    Returns:
        ``{doc_id: project_root}`` with the root each turn currently has under
        ``prefix`` (the FTS value for a turn only FTS still has there).
    """
    prefix = prefix.rstrip("/")
    escaped = _escape_filter_scalar(prefix)
    filter_expr = f'(project_root == "{escaped}" or project_root like "{escaped}/%")'
    if provider:
        filter_expr += f' and provider == "{_escape_filter_scalar(provider)}"'

    found: Dict[str, str] = {}
    for batch in _query_batches(["doc_id", "project_root", "provider"],
                                filter_expr=filter_expr, db_path=db_path):
        for row in batch:
            root = row.get("project_root") or ""
            if path_has_prefix(root, prefix) and (
                    not provider or row.get("provider") == provider):
                found[row["doc_id"]] = root

    if db_path:
        conn = None
        try:
            conn = _fts.connection(db_path)
            where = "(project_root = ? OR substr(project_root, 1, ?) = ?)"
            params: tuple = (prefix, len(prefix) + 1, prefix + "/")
            if provider:
                where += " AND provider = ?"
                params += (provider,)
            for doc_id, root in _fts.select_where(
                    conn, ["doc_id", "project_root"], where, params):
                found.setdefault(doc_id, root)
        except Exception as e:
            logger.warning("FTS project_root lookup failed: %s", _scrub_exception(e))
        finally:
            if conn is not None:
                _fts.close_ephemeral(conn)

    return dict(sorted(found.items()))


def stored_metadata_by_provider(provider: str, fields: List[str], *,
                                db_path: Optional[str] = None) -> Dict[str, List[dict]]:
    """Read ``fields`` for every turn of one provider from Milvus and the FTS sidecar.

    Each store contributes its own values, so a caller can tell when a prior
    rewrite reached Milvus but failed in FTS. An FTS read failure raises, since
    Milvus values alone cannot show a stale FTS row.

    Args:
        provider: Provider whose turns to read.
        fields: Column names held by both stores, such as ``timestamp``.
        db_path: Optional Milvus DB path; also derives the FTS sidecar path.

    Returns:
        ``{doc_id: [values, ...]}`` with one ``{field: value}`` dict per store
        that holds the turn.

    Raises:
        ValueError: ``fields`` names a column outside ``RETAGGABLE_FIELDS``.
        RuntimeError: The FTS sidecar could not be read.
    """
    unknown = set(fields) - RETAGGABLE_FIELDS
    if unknown:
        raise ValueError(f"Fields not readable here: {', '.join(sorted(unknown))}")

    stored: Dict[str, List[dict]] = {}
    filter_expr = f'provider == "{_escape_filter_scalar(provider)}"'
    for batch in _query_batches(["doc_id", "provider", *fields],
                                filter_expr=filter_expr, db_path=db_path):
        for row in batch:
            if row.get("provider") == provider:
                stored.setdefault(row["doc_id"], []).append(
                    {field: row.get(field) or "" for field in fields})

    if db_path:
        conn = None
        try:
            conn = _fts.connection(db_path)
            for doc_id, *values in _fts.select_where(
                    conn, ["doc_id", *fields], "provider = ?", (provider,)):
                stored.setdefault(doc_id, []).append(
                    {field: value or "" for field, value in zip(fields, values)})
        except Exception as e:
            raise RuntimeError(f"FTS metadata read failed: {_scrub_exception(e)}") from e
        finally:
            if conn is not None:
                _fts.close_ephemeral(conn)

    return stored


def get_rows_by_ids(ids: List[int], db_path: Optional[str] = None) -> List[dict]:
    """Fetch full Milvus rows (every field, vector included) for a batch of primary keys.

    Used by the re-embed to read a batch of rows just in time. Rows that
    no longer exist are simply absent from the result, so callers compare lengths
    to count missing rows. The integer-only ``id in [...]`` filter needs no escaping.

    Args:
        ids: Integer primary keys to fetch.
        db_path: Optional Milvus DB path.

    Returns:
        The matching rows as plain dicts, in whatever order Milvus returns them.
    """
    if not ids:
        return []
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return []
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id in [{', '.join(str(int(i)) for i in ids)}]",
            output_fields=["*"],
        )
        return [dict(r) for r in rows]


def upsert_rows(rows: List[dict], db_path: Optional[str] = None) -> int:
    """Upsert complete Milvus rows by primary key in one call, leaving FTS alone.

    The re-embed only replaces ``vector``; ``document`` and every FTS
    column are unchanged, so unlike ``upsert_document`` there is no FTS rewrite.
    Each row is replaced atomically by PK, so concurrent searches see it either
    fully old or fully new.

    Args:
        rows: Full rows as returned by ``get_rows_by_ids`` (every schema field).
        db_path: Optional Milvus DB path.

    Returns:
        The number of rows sent.
    """
    if not rows:
        return 0
    with milvus_client(db_path) as client:
        client.upsert(collection_name=COLLECTION_NAME, data=rows)
    return len(rows)


def delete_by_doc_id(doc_id: str, db_path: Optional[str] = None) -> DeleteResult:
    """Delete a single turn from both Milvus (by PK) and FTS (by doc_id).

    Mirrors ``delete_by_session``'s symmetric dual-write: the Milvus delete keys
    on the stable primary key derived from ``doc_id`` (the same
    ``int(sha256(doc_id)[:15], 16)`` derivation used at insert time).

    Unlike ``delete_by_session``, the FTS delete outcome is surfaced rather than
    swallowed: a turn that is gone from Milvus but still keyword-searchable in FTS
    has not been sanitized, and FTS healing only hydrates *missing* doc_ids, so a
    surviving stale row is never auto-redacted. Callers must treat
    ``fts_ok is False`` as unfinished and not mark the doc done.

    Args:
        doc_id: Stable document id of the turn to delete.
        db_path: Optional Milvus DB path; also derives the FTS sidecar path.

    Returns:
        DeleteResult: ``deleted`` (Milvus rows removed, 0 or 1) and ``fts_ok``
        reported independently. The turn is fully removed only when the FTS
        delete also succeeded.
    """
    pk = _pk_from_doc_id(doc_id)
    deleted = 0
    is_parent = passages.passage_parent(doc_id) is None
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return DeleteResult(0, True)
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id == {pk}",
            output_fields=["id", "session_id", "turn_index"],
        )
        if results:
            if is_parent:
                _delete_milvus_passages(client, doc_id, results[0])
            client.delete(collection_name=COLLECTION_NAME, filter=f"id == {pk}")
            deleted = len(results)

    # Also delete from FTS. The outcome is surfaced (not swallowed) so the caller
    # can keep an FTS-failed doc_id on the worklist instead of marking it done.
    fts_ok = False
    if db_path:
        conn = None
        try:
            conn = _fts.connection(db_path)
            _fts.delete(conn, "doc_id", doc_id)
            if is_parent:
                # Every passage doc_id sorts between "<parent>::p" and "<parent>::q".
                _fts.delete_where(conn, "doc_id >= ? AND doc_id < ?",
                                  (f"{doc_id}::p", f"{doc_id}::q"))
            fts_ok = True
        except Exception as e:
            logger.warning(
                "FTS delete by doc_id failed: %s", _scrub_exception(e))
            fts_ok = False
        finally:
            # Release the ephemeral connection on every path so a failed FTS
            # delete doesn't leak a SQLite handle.
            if conn is not None:
                _fts.close_ephemeral(conn)
    else:
        fts_ok = True

    return DeleteResult(deleted, fts_ok)


def _delete_milvus_passages(client, parent_doc_id: str, parent: Dict) -> int:
    """Delete the Milvus passage rows of ``parent_doc_id``; ``parent`` holds its session and turn."""
    expr = (
        f'session_id == "{_escape_filter_scalar(parent.get("session_id", ""))}" '
        f'and turn_index == {int(parent.get("turn_index", 0))} '
        f'and chunk_type == "{passages.PASSAGE_CHUNK_TYPE}"'
    )
    rows = client.query(collection_name=COLLECTION_NAME, filter=expr,
                        output_fields=["id", "doc_id"])
    pks = [r["id"] for r in rows
           if passages.passage_parent(r.get("doc_id", "")) == parent_doc_id]
    if pks:
        client.delete(collection_name=COLLECTION_NAME, filter=f"id in {json.dumps(pks)}")
    return len(pks)


def delete_by_branch(git_branch: str, db_path: Optional[str] = None) -> int:
    """Delete all turns for a given git branch."""
    escaped_branch = _escape_filter_scalar(git_branch)
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'git_branch == "{escaped_branch}"',
            output_fields=["id"],
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'git_branch == "{escaped_branch}"',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete(conn, "git_branch", git_branch)
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete by branch failed (non-fatal): %s", _scrub_exception(e))

    return len(results)


def delete_older_than(max_age_days: int, db_path: Optional[str] = None) -> int:
    """Delete all turns with timestamps older than max_age_days ago.

    Returns the number of deleted turns.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    # Milvus Lite varchar comparison works lexicographically,
    # and ISO 8601 timestamps sort correctly this way.
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return 0

        # Query to count before deleting
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'timestamp < "{cutoff_str}" && timestamp != ""',
            output_fields=["id"],
            limit=16384,
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'timestamp < "{cutoff_str}" && timestamp != ""',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete_where(conn, "timestamp < ? AND timestamp != ''", (cutoff_str,))
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete older_than failed (non-fatal): %s", _scrub_exception(e))

    return len(results)


def backfill_fts(db_path: Optional[str] = None) -> int:
    """Populate FTS from Milvus for any records missing from the FTS index.

    Two-pass to stay under Milvus's 64MB per-segment query result limit:
      1. Fetch doc_id only (~100 bytes/row) to identify what's in Milvus.
      2. Diff against FTS via batched IN-clause to find missing doc_ids.
      3. Fetch full documents only for missing doc_ids, in chunks of 100.
    A single pass that pulls the wide `document` field on all rows can cross
    the 64MB message ceiling and corrupt Woodpecker WAL state.
    """
    if not db_path:
        return 0

    # Serialize heal runs: if another backfill is already in flight (e.g. an
    # overlapping cadence tick) skip rather than double-hydrate.
    if not _fts_backfill_lock.acquire(blocking=False):
        logger.debug("FTS backfill already running; skipping overlapping heal")
        return 0

    try:
        fts_conn = _fts.connection(db_path)
        try:
            # Pass 2: hydrate missing rows in small batches and stream into FTS
            # one batch at a time so peak memory stays at O(BATCH_FETCH) regardless
            # of how many rows are missing.
            # Reuse the shared search field list so backfill and vector search
            # return the same fields.
            output_fields = _SEARCH_OUTPUT_FIELDS
            backfill_defaults = default_provider_metadata()
            BATCH_FETCH = 100
            inserted = 0
            # Truth row count streamed from Milvus during the diff pass; the
            # sentinel clear is gated on the FTS count reaching this.
            milvus_count = 0

            # Convert transient Milvus / schema-drift faults (a RuntimeError at
            # context-manager open from the drift guard, or a MilvusException from
            # a query) into the typed transient error so the heal worker retries
            # on a later tick instead of treating them as terminal.
            try:
                with milvus_client(db_path) as client:
                    def hydrate_and_insert(doc_ids: list) -> None:
                        nonlocal inserted
                        for i in range(0, len(doc_ids), BATCH_FETCH):
                            fetch_chunk = doc_ids[i:i + BATCH_FETCH]
                            ids_quoted = ", ".join(json.dumps(d) for d in fetch_chunk)
                            batch = client.query(
                                collection_name=COLLECTION_NAME,
                                filter=f"doc_id in [{ids_quoted}]",
                                limit=len(fetch_chunk),
                                output_fields=output_fields,
                            )
                            records = [
                                {
                                    "doc_id": r["doc_id"],
                                    "content": r.get("document", ""),
                                    "session_id": r.get("session_id", ""),
                                    "logical_session_id": r.get("logical_session_id", r.get("session_id", "")),
                                    "provider": r.get("provider", backfill_defaults["provider"]),
                                    "source_kind": r.get("source_kind", backfill_defaults["source_kind"]),
                                    "source_class": r.get("source_class", backfill_defaults["source_class"]),
                                    "source_id": r.get("source_id", ""),
                                    "source_path": r.get("source_path", r.get("transcript_file", "")),
                                    "git_branch": r.get("git_branch", ""),
                                    "turn_index": r.get("turn_index", 0),
                                    "timestamp": r.get("timestamp", ""),
                                    "chunk_type": r.get("chunk_type", "turn"),
                                    "project_root": r.get("project_root", ""),
                                    "issue_ids": r.get("issue_ids", ""),
                                }
                                for r in batch
                            ]
                            if records:
                                _fts.insert(fts_conn, records)
                                inserted += len(records)

                    # Diff against FTS in bounded chunks, then hydrate each chunk
                    # before moving on so missing doc IDs never grow with
                    # collection size.
                    for batch in _query_batches(["doc_id"], batch_size=500, db_path=db_path):
                        chunk = [r.get("doc_id", "") for r in batch if r.get("doc_id", "")]
                        if not chunk:
                            continue
                        milvus_count += len(chunk)
                        placeholders = ",".join("?" for _ in chunk)
                        rows = fts_conn.execute(
                            f"SELECT doc_id FROM {_fts.table_name} WHERE doc_id IN ({placeholders})",
                            chunk,
                        ).fetchall()
                        existing = {row[0] for row in rows}
                        missing_doc_ids = [d for d in chunk if d not in existing]
                        if missing_doc_ids:
                            hydrate_and_insert(missing_doc_ids)
            except RuntimeError as exc:
                # Only the schema-drift guard (_ensure_collection) is transient and
                # worth retrying; other RuntimeErrors — model mismatch, model not
                # cached, or genuine logic bugs — must surface, not retry forever.
                if "schema is out of date" not in str(exc):
                    raise
                raise FtsBackfillTransientError(
                    "FTS backfill aborted on transient Milvus / schema-drift fault"
                ) from exc
            except MilvusException as exc:
                raise FtsBackfillTransientError(
                    "FTS backfill aborted on transient Milvus query fault"
                ) from exc

            logger.info("FTS backfill: inserted %d records", inserted)
            # Only clear the sentinel once FTS has caught up to the Milvus truth
            # count; a still-degraded keyword index keeps the warning live so a
            # later heal tick retries. If the count can't be read
            # we err toward the legacy clear-on-completion behavior rather than
            # leaving a healthy index flagged forever.
            should_clear = True
            try:
                fts_count = _fts.count_rows(fts_conn)
                should_clear = fts_count >= milvus_count
                if not should_clear:
                    logger.info(
                        "FTS backfill incomplete (%d/%d rows); leaving sentinel set",
                        fts_count, milvus_count,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read FTS row count: %s", exc)
            if should_clear:
                try:
                    from fts_hybrid import clear_fts_backfill_sentinel
                    clear_fts_backfill_sentinel()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to clear FTS sentinel: %s", exc)
            return inserted
        finally:
            _fts.close_ephemeral(fts_conn)
    finally:
        _fts_backfill_lock.release()


def clear_collection(db_path: Optional[str] = None):
    """Drop and recreate the collection (full reset). Also clears FTS."""
    with milvus_client(db_path) as client:
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            print(f"Collection dropped: {COLLECTION_NAME}", file=sys.stderr)

    # Clear FTS database
    if db_path:
        _fts.clear(db_path)


def list_sessions(project_root: Optional[str] = None,
                  db_path: Optional[str] = None) -> List[Dict]:
    """List all sessions with turn counts and date ranges. Optionally filter by project."""
    all_results = _query_all(
        ["session_id", "timestamp", "git_branch", "chunk_type", "project_root"],
        filter_expr=_without_passages(
            f'project_root == "{_escape_filter_scalar(project_root)}"' if project_root else None),
        db_path=db_path,
    )

    sessions: Dict[str, Dict] = {}
    for r in all_results:
        sid = r.get("session_id", "")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "turns": 0,
                "branches": set(),
                "min_ts": "",
                "max_ts": "",
            }
        s = sessions[sid]
        s["turns"] += 1
        branch = r.get("git_branch", "")
        if branch:
            s["branches"].add(branch)
        ts = r.get("timestamp", "")
        if ts:
            if not s["min_ts"] or ts < s["min_ts"]:
                s["min_ts"] = ts
            if not s["max_ts"] or ts > s["max_ts"]:
                s["max_ts"] = ts

    result = []
    for s in sessions.values():
        s["branches"] = sorted(s["branches"])
        result.append(s)

    # Sort by most recent first
    result.sort(key=lambda s: s["max_ts"], reverse=True)
    return result


# --- Async wrappers ---

async def search_async(query: str, n: int = 5, session_id: Optional[str] = None,
                       git_branch: Optional[str] = None,
                       project_root: Optional[ProjectRoots] = None,
                       sort_by: str = "hybrid",
                       date_from: Optional[str] = None, date_to: Optional[str] = None,
                       provider: Optional[str] = None, source_kind: Optional[str] = None,
                       issue_id: Optional[str] = None,
                       db_path: Optional[str] = None) -> List[Dict]:
    """Async search with embed semaphore; takes the same filters as ``search``."""
    loop = asyncio.get_event_loop()
    # Snapshot globals at function entry (before any await) to close the TOCTOU
    # window: close_server_mode can clear _embed_executor between the semaphore
    # guard and the run_in_executor call.
    executor = _embed_executor
    semaphore = _embed_semaphore

    def run():
        return search(
            query, n, session_id=session_id, git_branch=git_branch,
            project_root=project_root, sort_by=sort_by,
            date_from=date_from, date_to=date_to,
            provider=provider, source_kind=source_kind,
            issue_id=issue_id, db_path=db_path,
        )

    if semaphore is None or executor is None:
        # Before init_server_mode (server still starting, or no server): keep the
        # event loop free. _mlx_eval_lock still serializes any MLX work.
        return await asyncio.to_thread(run)
    async with semaphore:
        return await loop.run_in_executor(
            executor,
            run,
        )


async def add_turns_async(turns: List[Dict], db_path: Optional[str] = None) -> int:
    """Async add_turns with embed semaphore + write lock."""
    loop = asyncio.get_event_loop()
    # Snapshot globals at function entry (before any await) to close the TOCTOU
    # window: close_server_mode can clear _embed_executor between the semaphore
    # guard and the run_in_executor call.
    executor = _embed_executor
    semaphore = _embed_semaphore
    write_lock = _write_lock

    if semaphore is None or write_lock is None or executor is None:
        raise RuntimeError(
            "MLX embed executor not initialized — call init_server_mode() before "
            "using add_turns_async(). Running embeddings on the default executor can "
            "hop OS threads and trigger Metal SIGSEGV."
        )
    async with semaphore:
        async with write_lock:
            return await loop.run_in_executor(executor, lambda: add_turns(turns, db_path))
