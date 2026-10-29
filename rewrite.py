"""In-place rewrite of stored rows that ingest would now store differently.

Two kinds of row qualify. A row whose ``document`` still holds harness boilerplate
is stripped with ``boilerplate.strip``. A row longer than the model's sliding window
was embedded without the window, so its stored vector is stale. Each qualifying row
is stripped, redacted per ``SESSIONFLOW_REDACT_MODE``, re-embedded and upserted by
its primary key with a recomputed ``issue_ids``, and its FTS row is rewritten. A row
that stripping leaves under ``rag_engine.MIN_TURN_CHARS`` is deleted, as ingest
would drop it.

The run reuses the ``reembed`` machinery: a worklist snapshot, chunked runs under
the MLX crash limit, a 0600 checkpoint after every batch, and a new memory
embedding generation on a fresh run. Before a batch is written, its old documents
and ``issue_ids`` go to ``~/.sessionflow/audit/rewrite-<run>.jsonl`` and its old
vectors to ``rewrite-<run>.f16``. Stripping cannot be undone from a transcript
that no longer exists, so the backup is the only copy of the old text.

An FTS rewrite or delete that fails stays in the checkpoint and is retried first
by the next run, because FTS healing only adds missing rows.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boilerplate
import rag_engine
import reembed
from embedding_control import get_embedding_budget

_STATE_NAME = "rewrite_state.json"
_WORKLIST_NAME = "rewrite_worklist.json"
_SCAN_BATCH = 256

# Patchable seam so tests can drive time without sleeping.
_monotonic = reembed._monotonic

_tokenizer = None


@dataclass
class RewriteReport:
    """Outcome of a ``plan`` or ``run`` call.

    Attributes:
        status: ``dry-run`` | ``refused`` | ``paused`` | ``done`` | ``error``.
        total: Rows in the worklist.
        done: Rows re-embedded and upserted so far.
        deleted: Rows deleted because stripping left too little text.
        missing: Worklist rows that no longer existed when their batch came up.
        errors: Failed batches so far (each one is retried by the next run).
        fts_pending: FTS rewrites or deletes waiting for a retry.
        cursor: Index into the worklist where the next run resumes.
        scanned: Rows read when the worklist was built.
        stripped: Rows whose document stripping changes.
        long: Rows over the sliding window that stripping does not change.
        dropped: Rows that stripping leaves under the minimum turn length.
        estimated_seconds: Estimated compute time left (dry run only).
        backup_path: Backup file of the old documents for this run.
        run_id: Identifier of the run the state belongs to.
        message: Human-readable detail.
    """

    status: str
    total: int = 0
    done: int = 0
    deleted: int = 0
    missing: int = 0
    errors: int = 0
    fts_pending: int = 0
    cursor: int = 0
    scanned: int = 0
    stripped: int = 0
    long: int = 0
    dropped: int = 0
    estimated_seconds: float = 0.0
    backup_path: str = ""
    run_id: str = ""
    message: str = ""


@dataclass
class VerifyReport:
    """Check of rewritten rows: no boilerplate left, vector matches a fresh embedding.

    Attributes:
        checked: Rows compared.
        passed: Rows with nothing to strip and a cosine at or above ``threshold``.
        unstripped: Rows that ``boilerplate.strip`` would still change.
        min_cos: Lowest cosine seen (1.0 when nothing was checked).
        mean_cos: Mean cosine (1.0 when nothing was checked).
        threshold: Pass threshold used.
        complete: False when the embedding budget stopped the check before the
            whole sample was read.
    """

    checked: int
    passed: int
    unstripped: int
    min_cos: float
    mean_cos: float
    threshold: float
    complete: bool = True


# --- Selection ---------------------------------------------------------------


def clean_text(document: str) -> str:
    """Return ``document`` with harness boilerplate removed."""
    return boilerplate.strip(document)[0]


def _is_long(token_length: int, window: Optional[int]) -> bool:
    # A sliding layer masks keys only when the sequence spans more than the window.
    return window is not None and token_length - 1 > window


def needs_rewrite(document: str, *, token_length: int, window: Optional[int]) -> bool:
    """Return True when ingest would store ``document`` with other text or another vector.

    Args:
        document: The stored document.
        token_length: Token count of the stripped document with the document prefix.
        window: The model's sliding window, or None when it has none.

    Returns:
        True when stripping changes the text or the text spans more than the window.
    """
    return clean_text(document) != document or _is_long(token_length, window)


def _model_dir() -> Optional[Path]:
    snapshots = sorted(rag_engine._MODEL_CACHE.glob("snapshots/*/config.json"))
    return snapshots[0].parent if snapshots else None


def _window() -> Optional[int]:
    """Return the cached model's ``sliding_window``, or None when it has none."""
    model_dir = _model_dir()
    if model_dir is None:
        raise RuntimeError(f"Embedding model not cached at {rag_engine._MODEL_CACHE}")
    return json.loads((model_dir / "config.json").read_text()).get("sliding_window")


def _token_lengths(texts: List[str]) -> List[int]:
    """Count tokens as ``embed_texts`` encodes documents, without loading MLX."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(str(_model_dir()))
    encoded = _tokenizer(
        [rag_engine._DOCUMENT_PREFIX + t for t in texts],
        truncation=True, max_length=rag_engine._MODEL_CFG["max_tokens"],
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _scan(db_path: Optional[str]) -> Tuple[List[int], Dict[str, int]]:
    """Read every stored document once; return the worklist and its counts."""
    window = _window()
    worklist: List[int] = []
    counts = {"scanned": 0, "stripped": 0, "long": 0, "dropped": 0}
    pending: List[Tuple[int, str, str]] = []

    def flush() -> None:
        lengths = _token_lengths([clean for _, _, clean in pending])
        for (pk, document, clean), length in zip(pending, lengths):
            if clean != document:
                counts["stripped"] += 1
                if len(clean.strip()) < rag_engine.MIN_TURN_CHARS:
                    counts["dropped"] += 1
                worklist.append(pk)
            elif _is_long(length, window):
                counts["long"] += 1
                worklist.append(pk)
        pending.clear()

    for batch in rag_engine._query_batches(["id", "document"], db_path=db_path):
        for row in batch:
            counts["scanned"] += 1
            document = row.get("document") or ""
            pending.append((int(row["id"]), document, clean_text(document)))
            if len(pending) >= _SCAN_BATCH:
                flush()
    if pending:
        flush()
    return worklist, counts


# --- State files -------------------------------------------------------------


def _state_path() -> Path:
    return reembed._state_dir() / _STATE_NAME


def _worklist_path() -> Path:
    return reembed._state_dir() / _WORKLIST_NAME


def _audit_path(run_id: str, ext: str) -> Path:
    audit = reembed._ensure_private_dir(reembed._state_dir() / reembed._AUDIT_DIRNAME)
    return audit / f"rewrite-{run_id}.{ext}"


def _load_state() -> Optional[dict]:
    path = Path.home() / reembed._STATE_DIRNAME / _STATE_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_state(state: dict) -> None:
    reembed._write_private_atomic(_state_path(), json.dumps(state))


def _load_worklist() -> List[int]:
    return json.loads(_worklist_path().read_text())


def status() -> dict:
    """Return the current checkpoint state, or an empty dict when no run exists."""
    return _load_state() or {}


def _append_documents(path: Path, rows: List[dict]) -> None:
    """Append the old document and ``issue_ids`` of ``rows`` as JSON lines (0600).

    The file is synced before returning, so the backup reaches disk before the upsert.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, reembed._FILE_MODE)
    with os.fdopen(fd, "a") as fh:
        for row in rows:
            fh.write(json.dumps({
                "id": int(row["id"]), "doc_id": row["doc_id"],
                "document": row["document"], "issue_ids": row.get("issue_ids", ""),
            }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(path, reembed._FILE_MODE)


# --- Public API --------------------------------------------------------------


def _report(state: dict, status: str, message: str = "") -> RewriteReport:
    return RewriteReport(
        status=status,
        total=int(state.get("snapshot_count", 0)),
        done=int(state.get("done", 0)),
        deleted=int(state.get("deleted", 0)),
        missing=int(state.get("missing", 0)),
        errors=int(state.get("errors", 0)),
        fts_pending=len(state.get("fts_retry", [])),
        cursor=int(state.get("cursor", 0)),
        scanned=int(state.get("scanned", 0)),
        stripped=int(state.get("stripped", 0)),
        long=int(state.get("long", 0)),
        dropped=int(state.get("dropped", 0)),
        backup_path=state.get("backup_path", ""),
        run_id=state.get("run_id", ""),
        message=message,
    )


def plan(db_path: Optional[str] = None) -> RewriteReport:
    """Report what a run would do, without writing anything.

    With an unfinished run in the checkpoint, reports its progress. Otherwise scans
    every stored document and counts the rows a run would rewrite or delete.

    Args:
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        A ``dry-run`` report with the selection counts and a time estimate.
    """
    db_path = db_path or reembed._db_path()
    state = _load_state()
    if state and state.get("run_id") and state.get("status") != "done":
        report = _report(state, "dry-run", "resuming an unfinished run")
        report.estimated_seconds = ((report.total - report.cursor)
                                    / reembed._rate_from_state(state))
        return report
    worklist, counts = _scan(db_path)
    return RewriteReport(
        status="dry-run", total=len(worklist), **counts,
        estimated_seconds=len(worklist) / reembed._rate_from_state(state),
        message="a run would snapshot this worklist and rewrite it",
    )


def _new_state(worklist: List[int], counts: Dict[str, int]) -> dict:
    run_id = uuid.uuid4().hex[:12]
    return {
        "run_id": run_id,
        "status": "running",
        "model_name": rag_engine.get_model_name(),
        "code_commit": reembed._code_commit(),
        "created_at": reembed._now_iso(),
        "snapshot_count": len(worklist),
        **counts,
        "cursor": 0,
        "done": 0,
        "deleted": 0,
        "missing": 0,
        "errors": 0,
        "fts_retry": [],
        "backup_path": str(_audit_path(run_id, "jsonl")),
        "vector_backup_path": str(_audit_path(run_id, "f16")),
        "chunks": [],
    }


def _retry_fts(state: dict, db_path: Optional[str]) -> None:
    """Retry the FTS writes that failed in an earlier batch; keep the ones that fail again."""
    still: List[dict] = []
    for entry in state.get("fts_retry", []):
        try:
            rows = rag_engine.get_rows_by_ids([entry["id"]], db_path=db_path)
            if entry["op"] == "rewrite" and rows:
                ok = rag_engine._rewrite_fts_row(entry["doc_id"], rows[0], db_path)
            else:
                ok = rag_engine.delete_by_doc_id(entry["doc_id"], db_path=db_path).fts_ok
        except Exception:  # noqa: BLE001 - keep the entry for the next run
            ok = False
        if not ok:
            still.append(entry)
    state["fts_retry"] = still
    _save_state(state)


def _rewrite_batch(state: dict, rows: List[dict], budget, db_path: Optional[str]) -> None:
    """Strip, redact, re-embed and write one batch; drop rows left too short."""
    keep: List[Tuple[dict, str]] = []
    drop: List[dict] = []
    for row in rows:
        clean = clean_text(row["document"])
        if clean != row["document"] and len(clean.strip()) < rag_engine.MIN_TURN_CHARS:
            drop.append(row)
            continue
        turn = {"text": clean}
        rag_engine._apply_redaction([turn])
        keep.append((row, turn["text"]))

    vectors = reembed._embed_documents(budget, [text for _, text in keep]) if keep else []
    _append_documents(Path(state["backup_path"]), rows)
    reembed._append_backup(Path(state["vector_backup_path"]), rows)

    updated = []
    for (row, text), vector in zip(keep, vectors):
        updated.append(dict(row, document=rag_engine._truncate_utf8(text, 65535),
                            vector=vector, issue_ids=rag_engine._extract_issue_ids(text)))
    rag_engine.upsert_rows(updated, db_path=db_path)
    for row in updated:
        if not rag_engine._rewrite_fts_row(row["doc_id"], row, db_path):
            state["fts_retry"].append({"op": "rewrite", "id": int(row["id"]),
                                       "doc_id": row["doc_id"]})
    # A resumed batch skips rows gone from Milvus, so each delete is checkpointed as
    # pending before it runs and cleared only once FTS confirms it.
    pending = [{"op": "delete", "id": int(row["id"]), "doc_id": row["doc_id"]} for row in drop]
    state["fts_retry"].extend(pending)
    _save_state(state)
    for entry in pending:
        result = rag_engine.delete_by_doc_id(entry["doc_id"], db_path=db_path)
        state["deleted"] += result.deleted
        if result.fts_ok:
            state["fts_retry"].remove(entry)
    state["done"] += len(updated)


def run(*, confirmed: bool, max_minutes: Optional[float] = None,
        max_turns: Optional[int] = None, restart: bool = False,
        db_path: Optional[str] = None) -> RewriteReport:
    """Rewrite the worklist in place, one chunk per call, resuming from the checkpoint.

    The first call scans the index and snapshots the worklist; later calls first
    retry pending FTS writes, then continue at ``cursor``. A call stops when the
    worklist is finished, at ``max_minutes`` or ``max_turns``, when the embedding
    budget refuses a batch, or on a failure, which leaves the cursor before the
    failed batch. The run is ``done`` only when no FTS write is pending.

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
        return RewriteReport(status="refused", message="confirmation required (--yes)")
    db_path = db_path or reembed._db_path()

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
        worklist, counts = _scan(db_path)
        reembed._write_private_atomic(_worklist_path(), json.dumps(worklist))
        state = _new_state(worklist, counts)
        state["memory_generation"] = reembed._reembed_memories()

    state["status"] = "running"
    _save_state(state)
    _retry_fts(state, db_path)

    budget = get_embedding_budget()
    started_at, started = reembed._now_iso(), _monotonic()
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
                _rewrite_batch(state, rows, budget, db_path)
        except reembed._BudgetDenied as e:
            outcome, message = "paused", f"embedding budget refused: {e}"
            break
        except Exception as e:
            state["errors"] += 1
            outcome, message = "error", rag_engine._scrub_exception(e)
            break

        state["cursor"] += len(ids)
        state["missing"] += len(ids) - len(rows)
        chunk_turns += len(ids)
        _save_state(state)

    if outcome == "done" and state["fts_retry"]:
        outcome = "paused"
        message = f"{len(state['fts_retry'])} FTS writes pending; run again to retry"

    state["chunks"].append({
        "started_at": started_at,
        "ended_at": reembed._now_iso(),
        "turns": chunk_turns,
        "seconds": round(_monotonic() - started, 3),
        "outcome": outcome,
    })
    state["status"] = "done" if outcome == "done" else "paused"
    if outcome == "done":
        state["finished_at"] = reembed._now_iso()
    _save_state(state)
    return _report(state, outcome, message)


def verify(sample: int = 200, threshold: float = 0.9999,
           db_path: Optional[str] = None, seed: Optional[int] = None) -> VerifyReport:
    """Check a sample of worklist rows against what ingest would store now.

    A row passes when ``boilerplate.strip`` leaves it unchanged and its stored
    vector matches a fresh single-text embedding of its document. Read-only.

    Args:
        sample: How many worklist rows to check.
        threshold: Minimum cosine for a row to pass.
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.
        seed: Optional RNG seed for a reproducible sample.

    Returns:
        Counts and cosine stats. Rows deleted by the run are skipped.
    """
    db_path = db_path or reembed._db_path()
    ids = _load_worklist()
    picked = random.Random(seed).sample(ids, min(sample, len(ids)))

    budget = get_embedding_budget()
    cosines: List[float] = []
    passed = unstripped = 0
    complete = True
    try:
        for start in range(0, len(picked), 100):
            for row in rag_engine.get_rows_by_ids(picked[start:start + 100], db_path=db_path):
                dirty = clean_text(row["document"]) != row["document"]
                fresh = reembed._embed_documents(budget, [row["document"]])[0]
                cos = reembed._cos(row["vector"], fresh)
                cosines.append(cos)
                unstripped += dirty
                passed += (not dirty) and cos >= threshold
    except reembed._BudgetDenied:
        complete = False

    return VerifyReport(
        checked=len(cosines),
        passed=passed,
        unstripped=unstripped,
        min_cos=min(cosines) if cosines else 1.0,
        mean_cos=sum(cosines) / len(cosines) if cosines else 1.0,
        threshold=threshold,
        complete=complete,
    )
