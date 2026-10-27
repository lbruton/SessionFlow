"""In-place re-embed of the existing index with the current embedding code.

A vector embedded without the padding mask depends on the other texts in its
batch, because padding leaks into attention. Such a stored row keeps its drifted
vector until it is re-embedded. This module recomputes every stored vector from the row's stored
``document`` and writes it back **in place**: the same primary key, every
metadata field kept, FTS untouched. Search keeps working throughout, since each
row is atomically either old or new.

Flow:

* **Snapshot** every primary key once into ``~/.sessionflow/reembed_worklist.json``,
  so the run never iterates a collection it is upserting into.
* **Run** in chunks. Each ``run`` stops at ``max_minutes`` / ``max_turns`` and
  exits, which keeps every process under the ~50 min MLX Metal crash limit.
  ``cleanup.py reembed drive`` spawns one process per chunk with idle gaps.
* **Checkpoint** after every batch to ``~/.sessionflow/reembed_state.json``
  (atomic replace, 0600). The next run resumes at ``cursor``. Re-embedding a row
  twice is harmless, so a crash mid-batch only repeats that batch.
* **Back up** each old vector (float16) to ``~/.sessionflow/audit/reembed-<run>.f16``
  before overwriting it, for rollback and the before/after cosine readout.

Embedding goes through the shared ``EmbeddingBudget``, so batch size and the
200 ms cooldown floor apply exactly as in ``rag_engine.add_turns``.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import rag_engine
from embedding_control import get_embedding_budget

_STATE_DIRNAME = ".sessionflow"
_AUDIT_DIRNAME = "audit"
_STATE_NAME = "reembed_state.json"
_WORKLIST_NAME = "reembed_worklist.json"
_FILE_MODE = 0o600

# Throughput of the DEVS-39 full reindex (18,508 turns in ~56 min). Used for the
# dry-run estimate until a chunk has measured the real rate.
DEFAULT_TURNS_PER_SECOND = 5.5

# Patchable seams so tests can drive time without sleeping.
_monotonic = time.monotonic
_sleep = time.sleep


@dataclass
class ReembedReport:
    """Outcome of a ``plan`` or ``run`` call.

    Attributes:
        status: ``dry-run`` | ``refused`` | ``paused`` | ``done`` | ``error``.
        total: Rows in the snapshot (or in the live index for a fresh dry run).
        done: Rows re-embedded so far across all chunks of this run.
        missing: Snapshot rows that no longer existed when their batch came up.
        errors: Failed batches so far (each one is retried by the next run).
        cursor: Index into the worklist where the next run resumes.
        estimated_seconds: Estimated compute time left (dry run only).
        backup_path: Old-vector backup file for this run, if any.
        run_id: Identifier of the run the state belongs to.
        message: Human-readable detail for refused/paused/error outcomes.
    """

    status: str
    total: int = 0
    done: int = 0
    missing: int = 0
    errors: int = 0
    cursor: int = 0
    estimated_seconds: float = 0.0
    backup_path: str = ""
    run_id: str = ""
    message: str = ""


@dataclass
class VerifyReport:
    """Cosine check of stored vectors against fresh single-text embeddings.

    Attributes:
        checked: Rows compared.
        passed: Rows whose cosine met ``threshold``.
        min_cos: Lowest cosine seen (1.0 when nothing was checked).
        mean_cos: Mean cosine (1.0 when nothing was checked).
        threshold: Pass threshold used.
    """

    checked: int
    passed: int
    min_cos: float
    mean_cos: float
    threshold: float


# --- State files -------------------------------------------------------------


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def _state_dir() -> Path:
    return _ensure_private_dir(Path.home() / _STATE_DIRNAME)


def _collection_file(name: str, collection: str) -> str:
    # The sessions files keep their original names so existing checkpoints resume.
    if collection == rag_engine.COLLECTION_NAME:
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}-{collection}{dot}{ext}"


def _state_path(collection: str = "sessions") -> Path:
    return _state_dir() / _collection_file(_STATE_NAME, collection)


def _worklist_path(collection: str = "sessions") -> Path:
    return _state_dir() / _collection_file(_WORKLIST_NAME, collection)


def _backup_path(run_id: str) -> Path:
    return _ensure_private_dir(_state_dir() / _AUDIT_DIRNAME) / f"reembed-{run_id}.f16"


def _write_private_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a 0600 temp file and ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(tmp, _FILE_MODE)
    os.replace(tmp, path)


def _load_state() -> Optional[dict]:
    path = Path.home() / _STATE_DIRNAME / _collection_file(_STATE_NAME, rag_engine.COLLECTION_NAME)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_state(state: dict) -> None:
    _write_private_atomic(_state_path(), json.dumps(state))


def _load_worklist() -> List[int]:
    return json.loads(_worklist_path().read_text())


def _db_path() -> Optional[str]:
    """Return the configured Milvus DB path, as the cleanup CLI resolves it."""
    try:
        import cleanup

        return cleanup.get_db_path()
    except Exception:  # pragma: no cover - cleanup import/lookup never fails in tests
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_commit() -> str:
    """Best-effort git SHA of this checkout, recorded so a run is traceable to the fix."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git missing is not an error
        return ""


def _snapshot_ids(db_path: Optional[str]) -> List[int]:
    ids: List[int] = []
    for batch in rag_engine._query_batches(["id"], db_path=db_path):
        ids.extend(int(r["id"]) for r in batch)
    return ids


def status() -> dict:
    """Return the current checkpoint state, or an empty dict when no run exists."""
    return _load_state() or {}


# --- Embedding ---------------------------------------------------------------


class _BudgetDenied(Exception):
    """The shared embedding budget refused a batch (paused or capped)."""


def _embed_documents(budget, texts: List[str]) -> List[list]:
    """Embed ``texts`` as documents through the shared budget (cooldown + accounting)."""
    chars = sum(len(t) for t in texts)
    decision = budget.before_batch(batch_size=len(texts), estimated_chars=chars)
    if not decision.allowed and decision.retry_after_seconds > 0:
        _sleep(decision.retry_after_seconds)
        decision = budget.before_batch(batch_size=len(texts), estimated_chars=chars)
    if not decision.allowed:
        raise _BudgetDenied(decision.reason)
    started = _monotonic()
    try:
        vectors = rag_engine.embed_texts(texts, is_query=False)
    except Exception as e:
        budget.after_batch(_monotonic() - started, 0, error=e)
        raise
    budget.after_batch(_monotonic() - started, len(texts))
    return vectors


def _append_backup(path: Path, rows: List[dict]) -> None:
    """Append ``(pk, old vector as float16)`` records for ``rows`` to the backup file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
    with os.fdopen(fd, "ab") as fh:
        for row in rows:
            vec = row["vector"]
            fh.write(struct.pack(f"<q{len(vec)}e", int(row["id"]), *vec))
    os.chmod(path, _FILE_MODE)


def _cos(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- Public API --------------------------------------------------------------


def _rate_from_state(state: Optional[dict]) -> float:
    if state:
        turns = sum(c.get("turns", 0) for c in state.get("chunks", []))
        seconds = sum(c.get("seconds", 0.0) for c in state.get("chunks", []))
        if turns and seconds > 0:
            return turns / seconds
    return DEFAULT_TURNS_PER_SECOND


MEMORY_COLLECTION = "memories"


def _open_memory_store():
    import memory_store

    store = memory_store.MemoryStore(memory_store.fetch_dsn())
    store.migrate()
    return store


def _reembed_memories() -> Optional[int]:
    """Start a new memory embedding generation for the active model.

    Memory vectors are rebuilt by the server's sync loop from the Postgres
    text, never in this process: the bump makes every Milvus acknowledgement
    pending, and the reset clears them. Returns the new generation, or None
    when the memory store is unreachable (transcripts still re-embed).
    """
    try:
        store = _open_memory_store()
        generation = store.bump_generation(
            model_name=rag_engine.get_model_name(),
            vector_dim=rag_engine.MEMORIES_SPEC.vector_dim(),
        )
        store.reset_acks(consumer="milvus")
    except Exception as exc:  # noqa: BLE001 - a memory outage must not block transcripts
        print(f"Memory re-embed skipped: {rag_engine._scrub_exception(exc)}")
        return None
    print(f"Memory embedding generation {generation} started; the server re-indexes memories.")
    return generation


def plan(db_path: Optional[str] = None) -> ReembedReport:
    """Report what a run would do, without writing anything.

    With an unfinished run in the checkpoint, reports its progress and the time
    left. Otherwise counts the live index's primary keys. The estimate uses the
    rate measured by earlier chunks when there is one, else the DEVS-39 rate.

    Args:
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        A ``dry-run`` report with ``total``, ``done``, ``cursor`` and
        ``estimated_seconds``.
    """
    db_path = db_path or _db_path()
    state = _load_state()
    if state and state.get("run_id") and state.get("status") != "done":
        total = int(state.get("snapshot_count", 0))
        cursor = int(state.get("cursor", 0))
        return ReembedReport(
            status="dry-run", total=total, done=int(state.get("done", 0)),
            missing=int(state.get("missing", 0)), errors=int(state.get("errors", 0)),
            cursor=cursor,
            estimated_seconds=(total - cursor) / _rate_from_state(state),
            run_id=state["run_id"],
            message="resuming an unfinished run",
        )
    total = len(_snapshot_ids(db_path))
    return ReembedReport(
        status="dry-run", total=total,
        estimated_seconds=total / _rate_from_state(state),
        message="a run would snapshot and re-embed every row",
    )


def _new_state(ids: List[int]) -> dict:
    run_id = uuid.uuid4().hex[:12]
    return {
        "run_id": run_id,
        "status": "running",
        "model_name": rag_engine.get_model_name(),
        "code_commit": _code_commit(),
        "created_at": _now_iso(),
        "snapshot_count": len(ids),
        "cursor": 0,
        "done": 0,
        "missing": 0,
        "errors": 0,
        "backup_path": str(_backup_path(run_id)),
        "chunks": [],
    }


def _report(state: dict, status: str, message: str = "") -> ReembedReport:
    return ReembedReport(
        status=status,
        total=int(state.get("snapshot_count", 0)),
        done=int(state.get("done", 0)),
        missing=int(state.get("missing", 0)),
        errors=int(state.get("errors", 0)),
        cursor=int(state.get("cursor", 0)),
        backup_path=state.get("backup_path", ""),
        run_id=state.get("run_id", ""),
        message=message,
    )


def run(*, confirmed: bool, max_minutes: Optional[float] = None,
        max_turns: Optional[int] = None, restart: bool = False,
        db_path: Optional[str] = None) -> ReembedReport:
    """Re-embed stored rows in place, one chunk per call, resuming from the checkpoint.

    The first call snapshots every primary key; later calls continue at
    ``cursor``. A call stops when the worklist is finished (``done``), when
    ``max_minutes`` or ``max_turns`` is reached (``paused``), or when the
    embedding budget refuses a batch (``paused``). An embedding or Milvus failure
    checkpoints without advancing past the failed batch and returns ``error``.

    Args:
        confirmed: Must be True; otherwise nothing is read or written (``refused``).
        max_minutes: Stop after this much wall time in this call.
        max_turns: Stop after this many worklist entries in this call.
        restart: Ignore any existing checkpoint and take a fresh snapshot.
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        A report with the run's cumulative counters and this call's outcome.
    """
    if not confirmed:
        return ReembedReport(status="refused", message="confirmation required (--yes)")
    db_path = db_path or _db_path()

    state = None if restart else _load_state()
    if state and state.get("run_id"):
        if state.get("model_name") != rag_engine.get_model_name():
            return _report(state, "refused",
                           f"checkpoint was built with model {state.get('model_name')!r}; "
                           f"active model is {rag_engine.get_model_name()!r}. "
                           "Use --restart to start over.")
        if state.get("status") == "done":
            return _report(state, "done", "run already complete; use --restart to redo")
        worklist = _load_worklist()
    else:
        worklist = _snapshot_ids(db_path)
        _write_private_atomic(_worklist_path(), json.dumps(worklist))
        state = _new_state(worklist)
        state["memory_generation"] = _reembed_memories()
        _write_private_atomic(_state_path(MEMORY_COLLECTION), json.dumps({
            "run_id": state["run_id"], "generation": state["memory_generation"],
            "model_name": state["model_name"], "created_at": state["created_at"],
        }))

    state["status"] = "running"
    _save_state(state)

    budget = get_embedding_budget()
    backup = Path(state["backup_path"])
    started_at, started = _now_iso(), _monotonic()
    chunk_turns = 0
    outcome, message = "done", ""

    while state["cursor"] < len(worklist):
        if max_turns is not None and chunk_turns >= max_turns:
            outcome, message = "paused", f"max_turns={max_turns} reached"
            break
        if max_minutes is not None and _monotonic() - started >= max_minutes * 60:
            outcome, message = "paused", f"max_minutes={max_minutes} reached"
            break

        size = budget.batch_size
        if max_turns is not None:
            size = min(size, max_turns - chunk_turns)
        ids = worklist[state["cursor"]:state["cursor"] + size]

        try:
            rows = rag_engine.get_rows_by_ids(ids, db_path=db_path)
            if rows:
                vectors = _embed_documents(budget, [r["document"] for r in rows])
                _append_backup(backup, rows)
                for row, vec in zip(rows, vectors):
                    row["vector"] = vec
                rag_engine.upsert_rows(rows, db_path=db_path)
        except _BudgetDenied as e:
            outcome, message = "paused", f"embedding budget refused: {e}"
            break
        except Exception as e:
            state["errors"] += 1
            outcome, message = "error", rag_engine._scrub_exception(e)
            break

        state["cursor"] += len(ids)
        state["done"] += len(rows)
        state["missing"] += len(ids) - len(rows)
        chunk_turns += len(ids)
        _save_state(state)

    seconds = _monotonic() - started
    state["chunks"].append({
        "started_at": started_at,
        "ended_at": _now_iso(),
        "turns": chunk_turns,
        "seconds": round(seconds, 3),
        "outcome": outcome,
    })
    state["status"] = "done" if outcome == "done" else "paused"
    if outcome == "done":
        state["finished_at"] = _now_iso()
    _save_state(state)
    return _report(state, outcome, message)


def verify(sample: int = 200, threshold: float = 0.9999,
           db_path: Optional[str] = None, seed: Optional[int] = None) -> VerifyReport:
    """Compare stored vectors with fresh single-text embeddings of their documents.

    A batch of one has no padding, so after a correct re-embed every stored vector
    matches its single-text embedding. Read-only: nothing is written to Milvus,
    FTS or the checkpoint.

    Args:
        sample: How many random rows to check.
        threshold: Minimum cosine for a row to pass.
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.
        seed: Optional RNG seed for a reproducible sample.

    Returns:
        Counts and cosine stats. Stops early (fewer ``checked``) if the embedding
        budget refuses a batch.
    """
    db_path = db_path or _db_path()
    try:
        ids = _load_worklist()
    except (OSError, ValueError):
        ids = _snapshot_ids(db_path)
    picked = random.Random(seed).sample(ids, min(sample, len(ids)))

    budget = get_embedding_budget()
    cosines: List[float] = []
    try:
        for start in range(0, len(picked), 100):
            rows = rag_engine.get_rows_by_ids(picked[start:start + 100], db_path=db_path)
            for row in rows:
                fresh = _embed_documents(budget, [row["document"]])[0]
                cosines.append(_cos(row["vector"], fresh))
    except _BudgetDenied:
        pass

    passed = sum(1 for c in cosines if c >= threshold)
    return VerifyReport(
        checked=len(cosines),
        passed=passed,
        min_cos=min(cosines) if cosines else 1.0,
        mean_cos=sum(cosines) / len(cosines) if cosines else 1.0,
        threshold=threshold,
    )
