"""Optional MLX reranker that reorders the top hybrid candidates of a query search.

``SESSIONFLOW_RERANKER`` picks ``qwen3`` (the default), ``jina`` or ``off``. The
server loads the model on the ``mlx-embed`` thread; CLI processes never load it,
so their searches skip reranking.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import embedding_control
from embedding_control import _env_int

log = logging.getLogger(__name__)

KINDS = ("off", "jina", "qwen3")
RERANKER_DEFAULT = "qwen3"
RERANK_TOP_DEFAULT = 10
RERANK_MAX_TOKENS_DEFAULT = 512

QWEN3_REPO = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
# Jina's weights and code are CC BY-NC 4.0: they load from the pinned snapshot
# and are never copied into this repository.
JINA_REPO = "jinaai/jina-reranker-v3.5-mlx"
JINA_REVISION = "3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21"
# RRF constant that fuses the hybrid rank with the Qwen3 rank.
QWEN3_RRF_K = 10
QWEN3_BATCH = 8
ERROR_LOG_INTERVAL_S = 60.0

_QWEN3_INSTRUCT = ("Given a question about past coding-agent sessions, judge whether this "
                   "conversation turn contains the answer")
_QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based "
                 "on the Query and the Instruct provided. Note that the answer can only be "
                 "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

Scorer = Callable[[str, Sequence[str]], List[float]]

_state: Dict[str, object] = {"kind": "off", "status": "off", "scorer": None}
_state_lock = threading.Lock()
_warned_values: set = set()
_last_error_log = 0.0


def settings() -> Dict[str, object]:
    """Return the reranker settings read from the environment.

    An unknown ``SESSIONFLOW_RERANKER`` value logs one warning and means ``off``.
    """
    raw = (os.getenv("SESSIONFLOW_RERANKER") or RERANKER_DEFAULT).strip().lower()
    if raw not in KINDS:
        if raw not in _warned_values:
            _warned_values.add(raw)
            log.warning("SESSIONFLOW_RERANKER=%r is not one of %s; reranking is off",
                        raw, ", ".join(KINDS))
        raw = "off"
    return {
        "reranker": raw,
        "rerank_top": _env_int("SESSIONFLOW_RERANK_TOP", RERANK_TOP_DEFAULT, minimum=1),
        "rerank_max_tokens": _env_int("SESSIONFLOW_RERANK_MAX_TOKENS",
                                      RERANK_MAX_TOKENS_DEFAULT, minimum=32),
    }


def status() -> str:
    """Return ``off``, ``loading``, ``ready`` or ``failed: <reason>``."""
    return str(_state["status"])


def is_ready() -> bool:
    """Return True when a scorer is loaded and search may rerank."""
    return _state["status"] == "ready" and _state["scorer"] is not None


def reset() -> None:
    """Drop any loaded scorer and report ``off``."""
    with _state_lock:
        _state.update(kind="off", status="off", scorer=None)


def install(kind: str, scorer: Scorer) -> None:
    """Make ``scorer`` the active reranker of type ``kind`` and report ``ready``."""
    with _state_lock:
        _state.update(kind=kind, status="ready", scorer=scorer)


def begin(kind: str) -> None:
    """Report ``loading`` for ``kind`` until ``load`` finishes."""
    with _state_lock:
        _state.update(kind=kind, status="loading", scorer=None)


def fail(kind: str, exc: BaseException) -> None:
    """Report ``failed: <reason>`` for ``kind``; search then skips reranking."""
    reason = f"{type(exc).__name__}: {exc}"
    log.warning("Reranker %s failed to load: %s", kind, reason)
    with _state_lock:
        _state.update(kind=kind, status=f"failed: {reason}", scorer=None)


_REPOS = {"jina": (JINA_REPO, JINA_REVISION), "qwen3": (QWEN3_REPO, QWEN3_REVISION)}


def _snapshot(kind: str) -> str:
    """Return the cached snapshot folder; the server never downloads (``download`` does)."""
    from huggingface_hub import snapshot_download

    repo, revision = _REPOS[kind]
    try:
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"{repo} is not downloaded; run ./download-model.sh {kind}") from exc


def download(kind: str) -> str:
    """Download the pinned snapshot for ``kind`` and return its folder (``download-model.sh``)."""
    from huggingface_hub import snapshot_download

    repo, revision = _REPOS[kind]
    return snapshot_download(repo, revision=revision)


def load(kind: str) -> None:
    """Build the ``kind`` scorer and install it, or report why it failed.

    Loading evaluates MLX arrays, so the server runs it on the ``mlx-embed`` thread.
    """
    if kind not in _BUILDERS:
        reset()
        return
    begin(kind)
    try:
        scorer = _BUILDERS[kind](settings()["rerank_max_tokens"])
    except Exception as exc:
        fail(kind, exc)
        return
    install(kind, scorer)
    log.info("Reranker %s ready", kind)


def rerank(query: str, rows: List[Dict], *, lock, fuse: bool = False) -> List[Dict]:
    """Reorder the first ``SESSIONFLOW_RERANK_TOP`` rows; rows past them keep their order.

    ``jina`` sorts those rows by score. ``qwen3``, and any kind with ``fuse``, sorts
    them by RRF of their incoming rank and the reranker rank. Each reranked row gets
    ``rerank_rank``. A scoring error logs at most once a minute and returns ``rows``
    unchanged.

    Args:
        query: The search query.
        rows: Ranked search rows with ``content``.
        lock: Held while the scorer runs MLX.
        fuse: Keep the incoming order as a signal, for an order the scorer cannot see.
    """
    scorer = _state["scorer"]
    if not is_ready() or not rows or scorer is None:
        return rows
    top = rows[:settings()["rerank_top"]]
    throttle = embedding_control.get_embed_throttle()
    # Scoring is a heavy MLX batch, so it keeps the same cooldown as embeds.
    throttle.wait_blocking()
    try:
        with lock:
            scores = list(scorer(query, [r.get("content") or "" for r in top]))
        if len(scores) != len(top):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(top)} rows")
    except Exception as exc:
        _log_error(exc)
        return rows
    finally:
        throttle.done()

    by_score = sorted(range(len(top)), key=lambda i: (-scores[i], i))
    rank = {i: k + 1 for k, i in enumerate(by_score)}
    if fuse or _state["kind"] == "qwen3":
        order = sorted(range(len(top)), key=lambda i: (
            -(1.0 / (QWEN3_RRF_K + i + 1) + 1.0 / (QWEN3_RRF_K + rank[i])), i))
    else:
        order = by_score
    for i in order:
        top[i]["rerank_rank"] = rank[i]
    return [top[i] for i in order] + rows[len(top):]


def _log_error(exc: BaseException) -> None:
    global _last_error_log
    now = time.monotonic()
    if not _last_error_log or now - _last_error_log >= ERROR_LOG_INTERVAL_S:
        _last_error_log = now
        log.warning("Reranking failed, returning the hybrid order: %s: %s",
                    type(exc).__name__, exc)


class _Qwen3Scorer:
    """Qwen3-Reranker yes/no scorer that applies the output head at the last position only."""

    def __init__(self, path: str, max_tokens: int):
        from mlx_lm import load as mlx_lm_load

        self.model, self.tokenizer = mlx_lm_load(path)
        self.max_tokens = max_tokens
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self._prefix = self.tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
        self._suffix = self.tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)

    def sequences(self, query: str, docs: Sequence[str]) -> List[List[int]]:
        """Return the token IDs of each prompt, with each document cut to ``max_tokens``."""
        head = self.tokenizer.encode(
            f"<Instruct>: {_QWEN3_INSTRUCT}\n<Query>: {query}\n<Document>: ",
            add_special_tokens=False)
        return [self._prefix + head
                + self.tokenizer.encode(d, add_special_tokens=False)[:self.max_tokens]
                + self._suffix for d in docs]

    def _head(self, hidden):
        if getattr(self.model.args, "tie_word_embeddings", False):
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)

    def __call__(self, query: str, docs: Sequence[str]) -> List[float]:
        import mlx.core as mx

        seqs = self.sequences(query, docs)
        # Batch by length so padding stays small; scores go back in input order.
        order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]))
        pad = self.tokenizer.pad_token_id or 0
        scores = [0.0] * len(seqs)
        for start in range(0, len(order), QWEN3_BATCH):
            chunk = order[start:start + QWEN3_BATCH]
            width = max(len(seqs[j]) for j in chunk)
            # Right padding: causal attention keeps each last real token clean.
            ids = mx.array([seqs[j] + [pad] * (width - len(seqs[j])) for j in chunk])
            hidden = self.model.model(ids)
            last = hidden[mx.arange(len(chunk)), mx.array([len(seqs[j]) - 1 for j in chunk])]
            logits = self._head(last)[:, [self.no_id, self.yes_id]].astype(mx.float32)
            probs = mx.softmax(logits, axis=-1)[:, 1]
            mx.eval(probs)
            for j, p in zip(chunk, probs.tolist(), strict=True):
                scores[j] = p
        mx.clear_cache()
        return scores


def _build_qwen3(max_tokens: int) -> _Qwen3Scorer:
    return _Qwen3Scorer(_snapshot("qwen3"), max_tokens)


def _load_module(name: str, path: Path, aliases: Optional[Dict[str, object]] = None):
    """Import ``path`` as module ``name``; ``aliases`` are visible in sys.modules while it runs."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in (aliases or {})}
    sys.modules[name] = module
    sys.modules.update(aliases or {})
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
    return module


def _build_jina(max_tokens: int) -> Scorer:
    folder = Path(_snapshot("jina"))
    modeling = _load_module("sessionflow_jina_modeling", folder / "modeling.py")
    # The snapshot's rerank.py imports its sibling as the top-level name "modeling".
    rerank_module = _load_module("sessionflow_jina_rerank", folder / "rerank.py",
                                 aliases={"modeling": modeling})
    model = rerank_module.MLXReranker(model_path=str(folder), max_doc_length=max_tokens + 1)

    def score(query: str, docs: Sequence[str]) -> List[float]:
        import mlx.core as mx

        scores = [0.0] * len(docs)
        for hit in model.rerank(query, list(docs)):
            scores[hit["index"]] = hit["relevance_score"]
        mx.clear_cache()
        return scores

    return score


_BUILDERS: Dict[str, Callable[[int], Scorer]] = {"jina": _build_jina, "qwen3": _build_qwen3}
