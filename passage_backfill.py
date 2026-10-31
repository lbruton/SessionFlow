"""Backfill passages for turns indexed before turns kept passages.

A run has two phases. The uncut phase re-parses, from disk and with no cursor,
each session that holds a turn cut at the old ``MAX_TURN_CHARS`` limit, and passes
those turns to ``rag_engine.add_turns``. The new turn replaces the cut row, and
ingest adds its passages. The passages phase then scans again and builds passages
from the stored text of every long turn that has none.

The run reuses the ``reembed`` machinery: chunked runs under the MLX crash limit and
a 0600 checkpoint after every session or turn. A session gone from disk or one
that fails to read keeps its cut row, which gets passages from its stored text.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import passages
import rag_engine
import reembed
from embedding_control import get_embedding_budget
from transcript_parser import CUT_MARK, MAX_TURN_CHARS

REPARSE_PROVIDERS = ("claude_code_cli", "claude_cowork_legacy", "codex")
"""Providers whose turns ``_parse`` can rebuild from disk by ``turn_index``."""

_STATE_NAME = "passages_state.json"
_ID_CHUNK = 200

_adapters: Dict[str, object] = {}

SessionKey = Tuple[str, str]


@dataclass
class PassageReport:
    """Outcome of a ``plan`` or ``run`` call.

    Attributes:
        status: ``dry-run`` | ``refused`` | ``paused`` | ``done`` | ``error``.
        phase: ``uncut`` or ``passages``.
        cut_rows: Turns cut at the old limit whose provider can re-parse them.
        sessions: Sessions holding those turns.
        missing: Long turns without passages (by length, before tokenizing).
        uncut: Cut turns replaced by their whole text so far.
        gone: Sessions whose transcript is no longer on disk.
        unreadable: Sessions whose transcript failed to parse.
        added: Passages stored so far.
        short: Turns that turned out too short for passages.
        fts_failed: Turns whose FTS passage write failed (FTS healing adds them).
        errors: Failed steps so far (each one is retried by the next run).
        cursor: Position in the current phase's worklist.
        total: Length of the current phase's worklist.
        run_id: Identifier of the run the state belongs to.
        message: Human-readable detail.
    """

    status: str
    phase: str = "uncut"
    cut_rows: int = 0
    sessions: int = 0
    missing: int = 0
    uncut: int = 0
    gone: int = 0
    unreadable: int = 0
    added: int = 0
    short: int = 0
    fts_failed: int = 0
    errors: int = 0
    cursor: int = 0
    total: int = 0
    run_id: str = ""
    message: str = ""


@dataclass
class VerifyResult:
    """What a run would still do.

    Attributes:
        missing: Turns long enough for passages that have none.
        cut_rows: Turns still cut at the old limit whose provider can re-parse them.
    """

    missing: int
    cut_rows: int


def is_old_cut(document: str) -> bool:
    """Return True for a turn cut at the old ``MAX_TURN_CHARS`` limit."""
    return document.endswith(CUT_MARK) and len(document) <= MAX_TURN_CHARS + len(CUT_MARK)


# --- Selection ---------------------------------------------------------------


def _scan(db_path: Optional[str]) -> Tuple[Dict[SessionKey, List[int]], List[int], int]:
    """Return the cut turns by session, the long turns without passages, and the cut count."""
    fields = ["id", "doc_id", "document", "chunk_type", "provider", "session_id",
              "logical_session_id", "turn_index"]
    expr = f'chunk_type in ["turn", "{passages.PASSAGE_CHUNK_TYPE}"]'
    parents: Dict[str, dict] = {}
    with_passages: Set[str] = set()
    for batch in rag_engine._query_batches(fields, filter_expr=expr, db_path=db_path):
        for row in batch:
            if row.get("chunk_type") == passages.PASSAGE_CHUNK_TYPE:
                with_passages.add(passages.passage_parent(row.get("doc_id", "")) or "")
            elif row.get("chunk_type") == "turn":
                parents[row["doc_id"]] = row

    cut: Dict[SessionKey, List[int]] = {}
    missing: List[int] = []
    cut_rows = 0
    for doc_id, row in parents.items():
        document = row.get("document") or ""
        if is_old_cut(document) and row.get("provider") in REPARSE_PROVIDERS:
            key = (row["provider"], row.get("logical_session_id") or row.get("session_id", ""))
            cut.setdefault(key, []).append(int(row.get("turn_index", 0)))
            cut_rows += 1
        if doc_id not in with_passages and len(document) > passages.PASSAGE_TOKENS:
            missing.append(int(row["id"]))
    return cut, missing, cut_rows


def _sources(keys: List[SessionKey]) -> Dict[SessionKey, list]:
    """Return the transcript sources on disk of each wanted session."""
    from provider_ingestion import default_provider_adapters

    _adapters.update(default_provider_adapters())
    wanted = set(keys)
    found: Dict[SessionKey, list] = {}
    for provider in sorted({p for p, _ in wanted}):
        for source in _adapters[provider].discover_sources():
            key = (provider, source.logical_session_id)
            if key in wanted:
                found.setdefault(key, []).append(source)
    return found


def _parse(provider: str, source):
    """Parse ``source`` from its start, ignoring any stored cursor."""
    return _adapters[provider].parse_source(source, cursor=None)


def _stored_doc_ids(doc_ids: List[str], db_path: Optional[str]) -> Set[str]:
    """Return the doc_ids among ``doc_ids`` that Milvus holds."""
    found: Set[str] = set()
    with rag_engine.milvus_client(db_path) as client:
        for start in range(0, len(doc_ids), _ID_CHUNK):
            chunk = doc_ids[start:start + _ID_CHUNK]
            rows = client.query(collection_name=rag_engine.COLLECTION_NAME,
                                filter=f"doc_id in {json.dumps(chunk)}",
                                output_fields=["doc_id"])
            found.update(r["doc_id"] for r in rows)
    return found


# --- State -------------------------------------------------------------------


def _state_path():
    return reembed._state_dir() / _STATE_NAME


def _load_state() -> Optional[dict]:
    path = _state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_state(state: dict) -> None:
    reembed._write_private_atomic(_state_path(), json.dumps(state))


def status() -> dict:
    """Return the current checkpoint state, or an empty dict when no run exists."""
    return _load_state() or {}


_COUNTERS = ("uncut", "gone", "unreadable", "added", "short", "fts_failed", "errors")


def _report(state: dict, status: str, message: str = "") -> PassageReport:
    worklist = state["sessions"] if state["phase"] == "uncut" else state["parents"]
    return PassageReport(
        status=status, phase=state["phase"], cut_rows=state["cut_rows"],
        sessions=len(state["sessions"]), missing=state["missing"],
        cursor=state["cursor"], total=len(worklist), run_id=state["run_id"],
        message=message, **{k: state[k] for k in _COUNTERS},
    )


def _new_state(db_path: Optional[str]) -> dict:
    cut, missing, cut_rows = _scan(db_path)
    return {
        "run_id": uuid.uuid4().hex[:12],
        "status": "running",
        "model_name": rag_engine.get_model_name(),
        "code_commit": reembed._code_commit(),
        "created_at": reembed._now_iso(),
        "phase": "uncut",
        "sessions": [[p, s, sorted(set(t))] for (p, s), t in sorted(cut.items())],
        "parents": [],
        "cut_rows": cut_rows,
        "missing": len(missing),
        "cursor": 0,
        **{k: 0 for k in _COUNTERS},
        "chunks": [],
    }


# --- Public API --------------------------------------------------------------


def plan(db_path: Optional[str] = None) -> PassageReport:
    """Report what a run would do, without writing anything.

    Args:
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        A ``dry-run`` report; an unfinished run reports its own progress.
    """
    db_path = db_path or reembed._db_path()
    state = _load_state()
    if state and state.get("status") != "done":
        return _report(state, "dry-run", "resuming an unfinished run")
    cut, missing, cut_rows = _scan(db_path)
    return PassageReport(status="dry-run", cut_rows=cut_rows, sessions=len(cut),
                         missing=len(missing), total=len(cut),
                         message="a run would re-parse these sessions, then add passages")


def _uncut_session(state: dict, entry: list, sources: Dict[SessionKey, list],
                   db_path: Optional[str]) -> bool:
    """Re-parse one session's cut turns; return False when the budget deferred them."""
    provider, session, turn_indexes = entry
    found = sources.get((provider, session))
    if not found:
        state["gone"] += 1
        return True
    wanted = set(turn_indexes)
    turns = []
    for source in found:
        result = _parse(provider, source)
        if result.errors:
            state["unreadable"] += 1
            return True
        turns.extend(t for t in result.turns
                     if t.get("chunk_type", "turn") == "turn" and t.get("turn_index") in wanted)
    if not turns:
        return True
    rag_engine.add_turns(turns, db_path=db_path)
    doc_ids = [t["doc_id"] for t in turns]
    if len(_stored_doc_ids(doc_ids, db_path)) < len(set(doc_ids)):
        return False
    state["uncut"] += len(turns)
    return True


def _add_passages(state: dict, row: dict, budget, db_path: Optional[str]) -> None:
    """Build, embed and store the passages of one stored turn."""
    document = row.get("document") or ""
    turn = {**row, "text": document}
    built = passages.build_passages(turn, rag_engine._token_offsets(document))
    if not built:
        state["short"] += 1
        return
    for p in built:
        p["_issue_ids"] = row.get("issue_ids", "")
    vectors = reembed._embed_documents(budget, [p["text"] for p in built],
                                       titles=[p["title"] for p in built])
    if not rag_engine.replace_passages(row["doc_id"], built, vectors, db_path=db_path):
        state["fts_failed"] += 1
    state["added"] += len(built)


def run(*, confirmed: bool, max_minutes: Optional[float] = None,
        max_turns: Optional[int] = None, restart: bool = False,
        db_path: Optional[str] = None) -> PassageReport:
    """Run the backfill one chunk per call, resuming from the checkpoint.

    A call stops when both phases are finished, at ``max_minutes`` or ``max_turns``
    (sessions in the uncut phase, turns in the passages phase), when the embedding
    budget defers work, or on a failure, which leaves the cursor before that step.

    Args:
        confirmed: Must be True; otherwise nothing is read or written (``refused``).
        max_minutes: Stop after this much wall time in this call.
        max_turns: Stop after this many worklist entries in this call.
        restart: Ignore any existing checkpoint and scan again.
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        A report with the run's cumulative counters and this call's outcome.
    """
    if not confirmed:
        return PassageReport(status="refused", message="confirmation required (--yes)")
    db_path = db_path or reembed._db_path()

    state = None if restart else _load_state()
    if state and state.get("run_id"):
        if state.get("model_name") != rag_engine.get_model_name():
            return _report(state, "refused",
                           f"checkpoint was built with model {state.get('model_name')!r}; "
                           "use --restart to start over")
        if state.get("status") == "done":
            return _report(state, "done", "run already complete; use --restart to redo")
    else:
        state = _new_state(db_path)
    state["status"] = "running"
    _save_state(state)

    budget = get_embedding_budget()
    started_at, started = reembed._now_iso(), reembed._monotonic()
    steps = 0
    outcome, message = "done", ""

    def _stop() -> Optional[str]:
        if max_turns is not None and steps >= max_turns:
            return f"max_turns={max_turns} reached"
        if max_minutes is not None and reembed._monotonic() - started >= max_minutes * 60:
            return f"max_minutes={max_minutes} reached"
        return None

    try:
        if state["phase"] == "uncut":
            remaining = state["sessions"][state["cursor"]:]
            sources = _sources([(p, s) for p, s, _ in remaining]) if remaining else {}
            while state["cursor"] < len(state["sessions"]):
                if (reason := _stop()) is not None:
                    outcome, message = "paused", reason
                    break
                if not _uncut_session(state, state["sessions"][state["cursor"]], sources, db_path):
                    outcome, message = "paused", "embedding budget deferred re-parsed turns"
                    break
                state["cursor"] += 1
                steps += 1
                _save_state(state)
            if outcome == "done":
                _, missing, _ = _scan(db_path)
                state.update(phase="passages", parents=missing, missing=len(missing), cursor=0)
                _save_state(state)

        while outcome == "done" and state["cursor"] < len(state["parents"]):
            if (reason := _stop()) is not None:
                outcome, message = "paused", reason
                break
            ids = state["parents"][state["cursor"]:state["cursor"] + budget.batch_size]
            if max_turns is not None:
                ids = ids[:max_turns - steps]
            rows = {int(r["id"]): r for r in rag_engine.get_rows_by_ids(ids, db_path=db_path)}
            for pk in ids:
                if pk in rows:
                    _add_passages(state, rows[pk], budget, db_path)
                state["cursor"] += 1
                steps += 1
                _save_state(state)
    except reembed._BudgetDenied as e:
        outcome, message = "paused", f"embedding budget refused: {e}"
    except Exception as e:
        state["errors"] += 1
        outcome, message = "error", rag_engine._scrub_exception(e)

    state["chunks"].append({
        "started_at": started_at,
        "ended_at": reembed._now_iso(),
        "steps": steps,
        "outcome": outcome,
    })
    state["status"] = "done" if outcome == "done" else "paused"
    if outcome == "done":
        state["finished_at"] = reembed._now_iso()
    _save_state(state)
    return _report(state, outcome, message)


def verify(db_path: Optional[str] = None) -> VerifyResult:
    """Count the turns a run would still change, tokenizing each long turn. Read-only.

    Args:
        db_path: Milvus DB path; defaults to the cleanup CLI's configured path.

    Returns:
        Long turns without passages and turns still cut at the old limit.
    """
    db_path = db_path or reembed._db_path()
    _, missing, cut_rows = _scan(db_path)
    long_enough = 0
    for start in range(0, len(missing), _ID_CHUNK):
        for row in rag_engine.get_rows_by_ids(missing[start:start + _ID_CHUNK], db_path=db_path):
            document = row.get("document") or ""
            long_enough += bool(passages.passage_spans(rag_engine._token_offsets(document)))
    return VerifyResult(missing=long_enough, cut_rows=cut_rows)
