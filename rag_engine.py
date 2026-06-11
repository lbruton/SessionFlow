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

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from pymilvus.exceptions import MilvusException
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Dict, Optional
import asyncio
import logging
import sys
import threading
import time

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
    default_provider_metadata,
    is_valid_issue_token,
)
import secret_redaction

RECENCY_WEIGHT_DEFAULT = 0.3
RECENCY_DECAY_DAYS_DEFAULT = 7
MISSING_TIMESTAMP_RECENCY = 0.5
_RANKING_SCRATCH_KEYS = ("_rrf_score", "_score", "_semantic_score", "_recency_score")

logger = logging.getLogger("sessionflow.milvus")

# Issue-ID extraction (SESF-25): technical-standard prefixes that match the
# issue-token regex but are never issue references. Dropped during extraction.
_ISSUE_ID_PREFIX_DENYLIST = frozenset(
    {"UTF", "SHA", "HTTP", "HTTPS", "ISO", "RFC", "IPV", "MD", "BASE"}
)
# Milvus VARCHAR field length the extracted ids are stored in.
_ISSUE_ID_FIELD_MAX = 4096
# Default number of timeline entries returned by get_issue_timeline (SESF-25/26).
DEFAULT_TIMELINE_LIMIT = 50
# Generous FTS fetch window for the timeline fallback so the chronological slice
# isn't biased by BM25 rank (the older matching turn may be outside the top-N).
_TIMELINE_FTS_FETCH_CAP = 500
# Observability threshold: warn when one issue matches an unexpectedly large
# structured set. Memory is bounded (SESF-34: rows stream through `_OldestN`
# rather than draining into a list), so this is now a scan-cost heads-up — narrow
# with date_from/date_to to shrink the iterator window. A server-side cap is still
# impossible: Milvus query_iterator order is undefined and it can't sort by a
# scalar, so any first-N truncation would drop arbitrary (not newest) rows.
_TIMELINE_ROWS_WARN = 50000

# Shared Milvus output_fields for vector search and recency listing. Includes
# ``issue_ids`` so SESF-25 issue tags propagate through ``_row_to_result``.
_SEARCH_OUTPUT_FIELDS = [
    "document", "doc_id", "session_id", "transcript_file",
    "turn_index", "timestamp", "git_branch", "chunk_type",
    "project_root", "logical_session_id", "provider",
    "source_kind", "source_class", "source_id", "source_path",
    "issue_ids",
]


class FtsBackfillTransientError(Exception):
    """Raised by backfill_fts on a transient Milvus / schema-drift failure (SESF-38).

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

COLLECTION_NAME = "sessions"

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
    # Stamp current model
    _IDENTITY_FILE.write_text(json.dumps({"model_name": _MODEL_NAME}))


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

    if not _MODEL_CACHE.exists():
        raise RuntimeError(
            f"Embedding model not cached at {_MODEL_CACHE}. "
            f"Run ./setup.sh or ./download-model.sh to download it."
        )

    print(f"Loading {_MODEL_ID} via mlx-embeddings...", file=sys.stderr)
    mlx_load, _, _ = _load_mlx_runtime()
    _mlx_model, _mlx_tokenizer = mlx_load(_MODEL_ID)
    print(f"{_MODEL_ID} ready ({_EMBED_DIM} dims, {_MODEL_CFG['max_tokens']} token context)", file=sys.stderr)
    return _mlx_model, _mlx_tokenizer


def _needs_input_remap() -> bool:
    """Check if the model's __call__ uses 'inputs' instead of 'input_ids'.

    Works around mlx-embeddings gemma3_text models where __call__ expects
    'inputs' but the tokenizer returns 'input_ids'.
    """
    return "gemma" in _MODEL_NAME


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed texts using the configured model. Adds model-specific prefix."""
    model, tokenizer = get_model()
    _, mlx_generate, mx = _load_mlx_runtime()
    prefix = _SEARCH_PREFIX if is_query else _DOCUMENT_PREFIX
    prefixed = [prefix + t for t in texts]

    if _needs_input_remap():
        # gemma3_text models expect (inputs, attention_mask) not (input_ids, ...)
        encoded = tokenizer.batch_encode_plus(
            prefixed, return_tensors="mlx", padding=True,
            truncation=True, max_length=_MODEL_CFG["max_tokens"],
        )
        output = model(encoded["input_ids"], attention_mask=encoded.get("attention_mask"))
    else:
        output = mlx_generate(model, tokenizer, texts=prefixed,
                              max_length=_MODEL_CFG["max_tokens"])

    embeddings = output.text_embeds.tolist()
    mx.clear_cache()
    return embeddings


# --- Milvus client management ---

_persistent_clients: Dict[str, MilvusClient] = {}
_fts = FTSIndex("turns_fts", [
    "session_id", "git_branch", "turn_index", "timestamp", "chunk_type",
    "project_root", "logical_session_id", "provider", "source_kind",
    "source_class", "source_id", "source_path", "issue_ids",
])
_write_lock: Optional[asyncio.Lock] = None
_embed_semaphore: Optional[asyncio.Semaphore] = None
# Dedicated single-worker executor so every MLX/Metal call runs on the same OS
# thread. The asyncio semaphore already serializes calls in time, but the
# default executor can rotate workers between calls — and MLX command-buffer
# state is not safe to migrate across threads. See SESF-8.
_embed_executor: Optional[ThreadPoolExecutor] = None
_server_mode = False


def init_server_mode(db_path: Optional[str] = None):
    """Initialize async concurrency primitives for HTTP server mode."""
    global _write_lock, _embed_semaphore, _embed_executor, _server_mode
    _check_model_identity(db_path=db_path)
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
            logger.warning("Error closing Milvus client %s: %s", path, e)
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
            logger.warning("Stale Milvus client for %s: %s — reconnecting", db_path, e)
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
        logger.error("Failed to connect to Milvus at %s: %s", db_path, e)
        raise
    return _persistent_clients[db_path]


def _resolve_db_path(db_path: Optional[str]) -> str:
    if not db_path:
        raise ValueError("db_path is required. Global index is at ~/.sessionflow/milvus.db")
    return db_path


def _expected_schema_fields() -> List[FieldSchema]:
    """Source-of-truth Milvus field list for the sessions collection.

    Used by both _ensure_collection() (create path) and _detect_schema_drift()
    (startup validation) so the two can't drift out of sync.
    """
    return [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=_EMBED_DIM),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="logical_session_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="provider", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_kind", dtype=DataType.VARCHAR, max_length=96),
        FieldSchema(name="source_class", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="transcript_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="turn_index", dtype=DataType.INT64),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="git_branch", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="project_root", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="issue_ids", dtype=DataType.VARCHAR, max_length=_ISSUE_ID_FIELD_MAX),
    ]


def detect_schema_drift(client: MilvusClient) -> List[str]:
    """Return a list of missing or extra field names if the live collection
    schema differs from `_expected_schema_fields()`. Empty list = no drift.

    Only field NAMES are diffed today — pymilvus's describe_collection output
    shape varies across Milvus Lite vs Standalone, and we have not hit a
    case where a same-named field changed dtype/length silently.
    """
    if not client.has_collection(COLLECTION_NAME):
        return []
    try:
        info = client.describe_collection(COLLECTION_NAME)
    except Exception as exc:
        print(f"Schema drift check skipped: describe_collection failed: {exc}", file=sys.stderr)
        return []
    expected = {f.name for f in _expected_schema_fields()}
    actual: set[str] = set()
    for field in info.get("fields", []) or []:
        if isinstance(field, dict):
            name = field.get("name")
        else:
            name = getattr(field, "name", None)
        if name:
            actual.add(name)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return [f"missing:{n}" for n in missing] + [f"extra:{n}" for n in extra]


def migrate_schema(client: MilvusClient, db_path: str = "") -> None:
    """Drop the sessions collection and recreate it with the current schema.

    DESTRUCTIVE: all indexed turns are lost. Provided as the explicit recovery
    path for `python cleanup.py migrate-schema` and for the auto-migrate
    env opt-in (SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1).
    """
    if client.has_collection(COLLECTION_NAME):
        print(
            f"Dropping collection {COLLECTION_NAME!r} for schema migration "
            "(all indexed turns will be lost)",
            file=sys.stderr,
        )
        client.drop_collection(COLLECTION_NAME)
    _create_collection(client, db_path)
    # SESF-40: drop_collection already clears the data-path schema cache, so
    # this is a no-op today. It funnels the one place schema mutates through the
    # canonical invalidation hook so a future in-place add_collection_field path
    # can't leave a warm persistent client raising code=65535 "field not exist".
    _invalidate_schema_cache(client)


def _invalidate_schema_cache(client: MilvusClient) -> None:
    """Invalidate pymilvus's process-global data-path schema cache for the collection.

    SESF-40: pymilvus 2.6 caches collection schemas in a class-level LRU
    (``GlobalCache.schema``, keyed by endpoint/db/collection and consumed by
    insert/upsert/search/hybrid_search via ``_get_schema``). It is invalidated on
    ``drop_collection`` but NOT on ``add_collection_field``, so an in-place field
    add can leave a long-lived persistent client raising ``code=65535 "field not
    exist"`` until LRU eviction or process restart — a fresh ``MilvusClient`` does
    not clear it (the cache is a singleton). Any schema-mutating op must funnel
    through here.

    No public schema-cache refresh API exists, so this reaches into pymilvus
    internals; every step is guarded and degrades to a logged no-op on a pymilvus
    upgrade rather than breaking a real migration.
    """
    try:
        from pymilvus.client.cache import GlobalCache
    except Exception as exc:  # pragma: no cover - pymilvus internals moved
        logger.debug("Schema cache invalidation skipped (no GlobalCache): %s", exc)
        return
    try:
        endpoint = client._get_connection().server_address
        # SessionFlow uses the default database; SchemaCache normalizes "" -> "default".
        GlobalCache.schema.invalidate(endpoint, "", COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover - defense in depth
        logger.warning("Schema cache invalidation failed for %s: %s", COLLECTION_NAME, exc)


def _ensure_collection(client: MilvusClient, db_path: str = "") -> None:
    """Create the sessions collection if missing; refuse to start on schema drift.

    SESF-11: previously this was create-if-missing only, so adding a field to
    `_expected_schema_fields()` silently broke every insert with
    DataNotMatchException against the pre-existing Milvus collection. Now:
      - missing collection → create
      - present + no drift → no-op
      - present + drift → if SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 drop+recreate,
        else raise RuntimeError telling the operator to run
        `python cleanup.py migrate-schema`.
    """
    if not client.has_collection(COLLECTION_NAME):
        _create_collection(client, db_path)
        return

    drift = detect_schema_drift(client)
    if not drift:
        return

    # SESF-38 AC-3: re-describe once (cache-free) before gating either branch.
    # detect_schema_drift issues a fresh describe_collection, so a stale/cached
    # first read that clears on the second describe must NOT raise or migrate.
    # This re-verify gates BOTH the auto-migrate and the raise branches.
    drift = detect_schema_drift(client)
    if not drift:
        return

    auto = os.getenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", "").lower() in {"1", "true", "yes", "on"}
    if auto:
        print(
            f"SESSIONFLOW_AUTO_MIGRATE_SCHEMA detected schema drift {drift!r}; "
            "dropping and recreating (all turns lost).",
            file=sys.stderr,
        )
        migrate_schema(client, db_path)
        return

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


def _create_collection(client: MilvusClient, db_path: str = "") -> None:
    print(f"Creating collection: {COLLECTION_NAME} (dim={_EMBED_DIM})", file=sys.stderr)
    schema = CollectionSchema(fields=_expected_schema_fields())

    index_params = client.prepare_index_params()
    if _is_remote_uri(db_path):
        # Standalone supports HNSW — O(log n) search vs O(n) FLAT.
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )
    else:
        # Milvus Lite silently ignores non-FLAT indexes.
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(f"Collection created: {COLLECTION_NAME}", file=sys.stderr)

    # Standalone requires explicit load_collection before query/dedup paths work.
    # create_collection with index_params does not auto-load.
    if _is_remote_uri(db_path):
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"Collection loaded: {COLLECTION_NAME}", file=sys.stderr)


@contextmanager
def milvus_client_for_migration(db_path: Optional[str] = None):
    """Open a Milvus client WITHOUT _ensure_collection.

    SESF-11: needed because _ensure_collection refuses to start on schema
    drift — but the whole point of `cleanup.py migrate-schema` is to repair
    that drift. This bypass MUST NOT be used outside migration code paths.
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


# --- Secret redaction guard (SESF-41) ---

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
    """Redact secrets in ``turns`` in place before embed/store (SESF-41 hook).

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
                logger.warning("Dedup check failed for doc_id %s: %s", doc_id, e)

    new_turns = [t for t in turns if t["doc_id"] not in existing_ids]
    if not new_turns:
        return 0

    # SESF-41: ingestion-time secret redaction guard. One hook over the deduped
    # turns rewrites turn["text"] in place, covering all three durable sinks
    # (embedding, Milvus document, FTS content) and add_turns_async with no
    # per-Provider change, and runs before _extract_issue_ids so issue IDs survive.
    _apply_redaction(new_turns)

    # Embed texts in local, resource-controlled batches. Query embedding stays
    # untouched in search(); this path is ingestion/backfill only.
    budget = get_embedding_budget()
    all_embeddings = []
    for batch in budget.split_batches(new_turns):
        texts = [t["text"] for t in batch]
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
            embeddings = embed_texts(texts, is_query=False)
        except Exception as e:
            budget.after_batch(time.monotonic() - started, 0, error=e)
            # SESF-41 AC-17: scrub the exception's string args before the bare
            # re-raise so every upstream site that later stringifies it is already
            # clean, while preserving any structured status-code args.
            _scrub_exception_args(e)
            raise
        budget.after_batch(time.monotonic() - started, len(batch))
        all_embeddings.extend(embeddings)

    new_turns = new_turns[:len(all_embeddings)]
    embeddings = all_embeddings
    if not new_turns:
        return 0

    provider_defaults = default_provider_metadata()
    data = []
    for turn, emb in zip(new_turns, embeddings):
        # Stable hash: SHA-256 truncated to int64. Python's hash() is
        # randomized per process, so the same doc_id would get different
        # primary keys across server restarts.
        int_id = int(hashlib.sha256(turn["doc_id"].encode()).hexdigest()[:15], 16)
        data.append({
            "id": int_id,
            "vector": emb,
            "document": _truncate_utf8(turn["text"], 65535),
            "doc_id": turn["doc_id"],
            "session_id": turn.get("session_id", ""),
            "logical_session_id": turn.get("logical_session_id", turn.get("session_id", "")),
            "provider": turn.get("provider", provider_defaults["provider"]),
            "source_kind": turn.get("source_kind", provider_defaults["source_kind"]),
            "source_class": turn.get("source_class", provider_defaults["source_class"]),
            "source_id": turn.get("source_id", ""),
            "source_path": turn.get("source_path", turn.get("transcript_file", "")),
            "transcript_file": turn.get("transcript_file", ""),
            "turn_index": turn.get("turn_index", 0),
            "timestamp": turn.get("timestamp", ""),
            "git_branch": turn.get("git_branch", ""),
            "chunk_type": turn.get("chunk_type", "turn"),
            "project_root": turn.get("project_root", ""),
            "issue_ids": _extract_issue_ids(turn.get("text", "")),
        })

    with milvus_client(db_path) as client:
        client.insert(collection_name=COLLECTION_NAME, data=data)

    # Dual-write into FTS5 sidecar
    try:
        if db_path:
            fts_conn = _fts.connection(db_path)
            fts_records = [{
                "doc_id": t["doc_id"],
                "content": t["text"],
                "session_id": t.get("session_id", ""),
                "logical_session_id": t.get("logical_session_id", t.get("session_id", "")),
                "provider": t.get("provider", provider_defaults["provider"]),
                "source_kind": t.get("source_kind", provider_defaults["source_kind"]),
                "source_class": t.get("source_class", provider_defaults["source_class"]),
                "source_id": t.get("source_id", ""),
                "source_path": t.get("source_path", t.get("transcript_file", "")),
                "git_branch": t.get("git_branch", ""),
                "turn_index": t.get("turn_index", 0),
                "timestamp": t.get("timestamp", ""),
                "chunk_type": t.get("chunk_type", "turn"),
                "project_root": t.get("project_root", ""),
                "issue_ids": _extract_issue_ids(t.get("text", "")),
            } for t in new_turns]
            _fts.insert(fts_conn, fts_records)
            _fts.close_ephemeral(fts_conn)
    except Exception as e:
        # SESF-41 AC-17: the FTS payload holds Turn content, so scrub the exception
        # text before logging so no secret fragment can echo through the warning.
        logger.warning("FTS insert failed (non-fatal): %s", _scrub_exception(e))

    return len(data)


def _escape_filter_scalar(value: str) -> str:
    """Escape a string value for use in a Milvus boolean-expression filter literal.

    Milvus filter literals are C-style double-quoted strings (e.g. field ==
    "value"). Per the Milvus expression grammar (Plan.g4), an embedded
    double-quote is written as backslash-quote and a literal backslash as a
    doubled backslash; Milvus does NOT honor ""-doubling.

    Rules:
      - NUL bytes are never valid in identifiers or scalar values; reject them
        outright so a malformed input cannot truncate the filter expression.
      - Each backslash is doubled, then each double-quote becomes backslash-quote.
        Order matters: backslashes are escaped first, otherwise the backslash
        introduced when escaping a quote would itself be doubled. This also stops
        a trailing backslash from escaping the literal's closing quote and
        consuming the rest of the filter expression (SESF-33).
      - A literal newline or carriage return is not a valid character inside a
        Milvus string literal (the grammar's DoubleSChar excludes them), so each
        is rewritten to its escape form (backslash-n / backslash-r); tabs are
        likewise escaped for consistency. These run after the backslash doubling
        so the escape backslash they introduce is not itself re-doubled.
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
    both halves of hybrid search stay in lockstep (SESF-32).
    """
    if "\x00" in issue_id:
        raise ValueError("issue_id must not contain NUL bytes")
    return issue_id.strip().upper().replace("%", "").replace("_", "")


def _row_to_result(entity: Dict, defaults: Dict, distance: float = 1.0) -> Dict:
    """Map a Milvus entity dict to the standard internal result format.

    Shared by ``search`` (vector hit entities) and ``_recent_listing`` (query
    result rows) so any new schema field added to the collection propagates to
    both code paths automatically.

    ``distance`` is the COSINE distance from the query vector. Use the default
    ``1.0`` sentinel for query-less listings where no similarity score exists
    (1.0 distance → 0.0 similarity, which is the honest value).
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


def _build_milvus_filter(session_id: Optional[str], git_branch: Optional[str],
                         project_root: Optional[str], provider: Optional[str],
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
        filters.append(f'project_root == "{_escape_filter_scalar(project_root)}"')
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


def search(query: Optional[str], n: int = 5, session_id: Optional[str] = None,
           git_branch: Optional[str] = None, project_root: Optional[str] = None,
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

    project_root: when set, restricts results to that project. When None,
    searches across all projects (cross-project search).
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

    # Query-less recency listing (SESF-16): with no query there is no vector to
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
            vector_results.append(
                _row_to_result(hit["entity"], provider_defaults, distance=hit["distance"])
            )

    # --- FTS5 keyword search ---
    fts_filters = {}
    if session_id:
        fts_filters["session_id"] = session_id
    if git_branch:
        fts_filters["git_branch"] = git_branch
    if project_root:
        fts_filters["project_root"] = project_root
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
        # SESF-32 — apply the same comma-delimited containment match the Milvus
        # half uses (issue_ids is an FTS metadata column as of SESF-25), so the
        # hybrid FTS results can't leak turns that aren't tagged with the issue.
        token = _issue_id_containment_token(issue_id)
        fts_filters["issue_ids"] = ("like", f"%,{token},%")
    fts_results = _fts.search(query, n=fetch_n, filters=fts_filters or None, db_path=db_path)

    # --- Merge with RRF ---
    # Route *every* candidate set through rrf_merge (it handles an empty list
    # natively and is rank-order-preserving) so each row carries `_rrf_score` —
    # the only relevance signal that spans both vector and FTS-only rows. The
    # `_rrf_score` is kept alive until after scoring (see _rank_results).
    merged = rrf_merge(vector_results, fts_results, n=fetch_n)

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


def _recent_listing(n: int, session_id: Optional[str] = None,
                    git_branch: Optional[str] = None,
                    project_root: Optional[str] = None,
                    provider: Optional[str] = None,
                    source_kind: Optional[str] = None,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None,
                    issue_id: Optional[str] = None,
                    db_path: Optional[str] = None) -> List[Dict]:
    """Query-less chronological listing: filter-only streamed scan ranked by
    timestamp descending (SESF-16).

    Milvus ``query()`` has no ORDER BY, and a single capped ``query()`` call
    materializes every filter-matching row server-side — broad filters (e.g.
    ``provider="codex"`` across all projects) overflow Milvus's per-query
    result-size buffer and raise ``code=65535: query results exceed the limit
    size`` (SESF-36). Rows therefore stream through ``_query_batches`` into a
    ``_NewestN`` bounded heap: memory stays O(n) and the kept set is the true
    global newest-``n`` regardless of collection size.
    """
    if n <= 0:
        return []

    filter_expr = _build_milvus_filter(
        session_id, git_branch, project_root, provider, source_kind,
        date_from, date_to, issue_id,
    )

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
    # distance=1.0 sentinel: no query vector exists for recency listings, so
    # 1.0 (maximum COSINE distance → 0 similarity) is the honest placeholder
    # and prevents the 1 - None TypeError in format_results.
    mapped = [_row_to_result(row, defaults, distance=1.0) for row in rows]

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

    Replaces draining every structured match into a list (SESF-34): rows stream
    in via :meth:`add` and at most ``limit`` are ever held, so memory is O(limit)
    no matter how many turns reference an issue. Correctness is preserved because
    the newest-kept entry is evicted only when a strictly-older candidate arrives,
    so the kept set always equals the true oldest ``limit`` seen so far — even
    though Milvus ``query_iterator`` yields rows in an undefined order.

    Entries are keyed by ``(parsed-UTC timestamp, doc_id)`` — timestamps go
    through ``_parse_timestamp_utc`` (the same ordering ``_rank_results`` uses)
    because the index mixes ``Z``/``+00:00``/naive ISO forms, which do not sort
    reliably as strings (SESF-43; mirrors the ``_NewestN`` fix from SESF-36).
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
    recency listing (SESF-36): rows stream in via :meth:`add` and at most
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
                       db_path: Optional[str] = None) -> List[Dict]:
    """Cross-harness, deduplicated, chronological feed of turns for an issue.

    Unions a structured Milvus source (``issue_ids like "%,ID,%"`` streamed batch
    by batch via ``_query_batches``) with an FTS keyword fallback (a literal MATCH
    on the issue token) so any un-tagged turn remains visible (Req 6.1). Both
    sources are normalized to the same shape and routed through an ``_OldestN``
    collector that bounds memory to O(``limit``) while retaining the true oldest
    matches (SESF-34): it deduplicates by ``doc_id`` (Req 4.2/6.2, structured
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

    Returns:
        A list of timeline entry dicts, oldest first; ``[]`` when nothing
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
                     "chunk_type", "provider", "issue_ids"]

    # Per-entry filters, hoisted so they run during the stream over each source
    # row rather than over a fully-materialized list — only survivors reach the
    # bounded collector. Equivalent to the prior post-merge filtering.
    allowed = set(providers) if providers else None
    upper = date_to.split("T")[0] + "T23:59:59" if date_to else None
    token = f",{canonical},"

    def _passes(entry: Dict) -> bool:
        """Whether one entry clears the provider, date-bound and issue-token guards."""
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
    # oldest-N collector rather than draining them into a list (SESF-34). The
    # collector retains only the true oldest ``limit``, so memory is O(limit) no
    # matter how many turns reference the issue — and a server-side cap stays
    # impossible (Milvus query_iterator order is undefined; it can't sort by a
    # scalar), so streaming is the only correctness-preserving bound.
    collector = _OldestN(limit)
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
        logger.warning("FTS fallback failed for issue timeline %s (non-fatal): %s", canonical, e)
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


def _rank_results(results: List[Dict], sort_by: str, n: int,
                  now: Optional[datetime] = None) -> List[Dict]:
    """Order the merged candidate pool by the chosen strategy, then truncate.

    - ``relevance``: keep the post-RRF order (no recency re-rank).
    - ``recency``: re-rank by timestamp descending; missing-timestamp rows last.
    - ``hybrid``: final = (1-w)*semantic_score + w*recency_score.

    Strips ranking scratch keys (`_rrf_score`, `_score`, `_semantic_score`,
    `_recency_score`) before returning — but preserves non-score engine metadata
    such as `_fts_warning`.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if sort_by == "relevance":
        ranked = list(results)
    elif sort_by == "recency":
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        ranked = sorted(
            results,
            key=lambda r: _parse_timestamp_utc(r.get("timestamp", "")) or _epoch,
            reverse=True,
        )
    else:  # hybrid
        weight = _env_float(
            "SESSIONFLOW_RECENCY_WEIGHT", RECENCY_WEIGHT_DEFAULT, 0.0, 1.0
        )
        decay_days = _env_int(
            "SESSIONFLOW_RECENCY_DECAY_DAYS", RECENCY_DECAY_DAYS_DEFAULT, minimum=1
        )
        semantic = _semantic_scores(results)
        for r, sem in zip(results, semantic, strict=True):
            rec = _recency_score(r.get("timestamp", ""), now, decay_days)
            r["_semantic_score"] = sem
            r["_recency_score"] = rec
            r["_score"] = (1 - weight) * sem + weight * rec
        ranked = sorted(results, key=lambda r: r.get("_score", 0.0), reverse=True)

    for r in ranked:
        for key in _RANKING_SCRATCH_KEYS:
            r.pop(key, None)

    return ranked[:n]


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
            filter=f'session_id == "{_escape_filter_scalar(session_id)}"',
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


def get_stats(project_root: Optional[str] = None, db_path: Optional[str] = None) -> Dict:
    """Get index statistics. Optionally filter to a specific project."""
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            # SESF-41 AC-10: keep the redaction surface on every return path so
            # get_stats()["redaction"] never KeyErrors before the collection exists.
            return {
                "total_turns": 0,
                "sessions": 0,
                "by_type": {},
                "providers": {},
                "redaction": _redaction_status(),
            }

    # Query for breakdowns (capped by Milvus offset limit)
    all_results = _query_all(
        ["session_id", "chunk_type", "git_branch", "project_root", "provider"],
        filter_expr=f'project_root == "{_escape_filter_scalar(project_root)}"' if project_root else None,
        db_path=db_path,
    )

    total = len(all_results)
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

    # Additive, project-scoped FTS lag keys (SESF-38 AC-6). Pass the Milvus total
    # we just computed so fts_lag_status does not re-enter get_stats.
    lag = fts_lag_status(
        db_path=db_path, project_root=project_root, milvus_turn_count=total
    )
    stats["fts_row_count"] = lag["fts_row_count"]
    stats["fts_lag"] = lag["fts_lag"]
    stats["fts_backfill_required"] = lag["fts_backfill_required"]

    # SESF-41: operator-facing redaction status + durable per-rule counts (AC-10).
    stats["redaction"] = _redaction_status()
    return stats


def _count_milvus_turns(db_path=None, project_root=None) -> int:
    """Count Milvus turn rows through the DRIFT-TOLERANT migration client (SESF-38 D-6).

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
    filter_expr = (
        f'project_root == "{_escape_filter_scalar(project_root)}"'
        if project_root
        else None
    )
    with milvus_client_for_migration(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return 0
        res = client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr or "",
            output_fields=["count(*)"],
            limit=1,
        )
        return int(res[0]["count(*)"]) if res else 0


def fts_lag_status(
    db_path: Optional[str] = None,
    project_root: Optional[str] = None,
    milvus_turn_count: Optional[int] = None,
) -> Dict[str, object]:
    """Return static FTS-vs-Milvus lag data for observability (SESF-38 AC-6).

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

    # FTS count on the calling thread (SESF-13 thread affinity): open an
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


def _query_batches(output_fields: list, batch_size: int = 1000,
                   filter_expr: Optional[str] = None,
                   db_path: Optional[str] = None) -> Iterator[list]:
    """Yield Milvus query_iterator results one batch at a time.

    Uses pymilvus's server-side iterator instead of offset pagination so the
    full collection is drained regardless of size. The previous implementation
    hard-capped at 16,384 rows, silently truncating any collection larger than
    that — see SESF-4.
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
        logger.warning("FTS delete by session failed (non-fatal): %s", e)

    return len(results)


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
        logger.warning("FTS delete by branch failed (non-fatal): %s", e)

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
        logger.warning("FTS delete older_than failed (non-fatal): %s", e)

    return len(results)


def backfill_fts(db_path: Optional[str] = None) -> int:
    """Populate FTS from Milvus for any records missing from the FTS index.

    Two-pass to stay under Milvus's 64MB per-segment query result limit:
      1. Fetch doc_id only (~100 bytes/row) to identify what's in Milvus.
      2. Diff against FTS via batched IN-clause to find missing doc_ids.
      3. Fetch full documents only for missing doc_ids, in chunks of 100.
    The original single-pass that pulled the wide `document` field on all
    rows crossed the 64MB ceiling once a segment accumulated enough text
    and corrupted Woodpecker WAL state. See SESF-2.
    """
    if not db_path:
        return 0

    # Serialize heal runs: if another backfill is already in flight (e.g. an
    # overlapping cadence tick) skip rather than double-hydrate (SESF-38 D-3).
    if not _fts_backfill_lock.acquire(blocking=False):
        logger.debug("FTS backfill already running; skipping overlapping heal")
        return 0

    try:
        fts_conn = _fts.connection(db_path)
        try:
            # Pass 2: hydrate missing rows in small batches and stream into FTS
            # one batch at a time so peak memory stays at O(BATCH_FETCH) regardless
            # of how many rows are missing — see SESF-5.
            # Reuse the shared search field list so backfill and vector search
            # stay in sync — a drifted copy is how transcript_file went missing.
            output_fields = _SEARCH_OUTPUT_FIELDS
            backfill_defaults = default_provider_metadata()
            BATCH_FETCH = 100
            inserted = 0
            # Truth row count streamed from Milvus during the diff pass; the
            # sentinel clear is gated on the FTS count reaching this (SESF-38 D-5).
            milvus_count = 0

            # Convert transient Milvus / schema-drift faults (a RuntimeError at
            # context-manager open from the drift guard, or a MilvusException from
            # a query) into the typed transient error so the heal worker retries
            # on a later tick instead of treating them as terminal (SESF-38 D-4).
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
            # later heal tick retries (SESF-38 D-5). If the count can't be read
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
        filter_expr=f'project_root == "{_escape_filter_scalar(project_root)}"' if project_root else None,
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
                       git_branch: Optional[str] = None, project_root: Optional[str] = None,
                       sort_by: str = "hybrid",
                       provider: Optional[str] = None, source_kind: Optional[str] = None,
                       db_path: Optional[str] = None) -> List[Dict]:
    """Async search with embed semaphore."""
    loop = asyncio.get_event_loop()
    # Snapshot globals at function entry (before any await) to close the TOCTOU
    # window: close_server_mode can clear _embed_executor between the semaphore
    # guard and the run_in_executor call.
    executor = _embed_executor
    semaphore = _embed_semaphore

    if semaphore is None or executor is None:
        raise RuntimeError(
            "MLX embed executor not initialized — call init_server_mode() before "
            "using search_async(). Running embeddings on the default executor can "
            "hop OS threads and trigger Metal SIGSEGV (see SESF-8)."
        )
    async with semaphore:
        return await loop.run_in_executor(
            executor,
            lambda: search(
                query, n, session_id=session_id, git_branch=git_branch,
                project_root=project_root, sort_by=sort_by,
                provider=provider, source_kind=source_kind, db_path=db_path,
            ),
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
            "hop OS threads and trigger Metal SIGSEGV (see SESF-8)."
        )
    async with semaphore:
        async with write_lock:
            return await loop.run_in_executor(executor, lambda: add_turns(turns, db_path))
