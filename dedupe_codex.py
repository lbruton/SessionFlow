"""Remove Codex rows left by keying turns on rollout line numbers.

A session whose rollout is on disk loses an old row only when a new task row holds
its reply, and only once the backfill has indexed every parsed turn, so search never
loses text. A session whose rollout is gone keeps one row per distinct turn.
Nothing here embeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import os
import uuid

import boilerplate
import rag_engine

PROVIDER = "codex"
_REPLY_MARKER = "\n\nAssistant: "


@dataclass
class CodexDedupePlan:
    """Codex rows to delete, by session.

    Attributes:
        stale: Rows of sessions on disk that the current parse does not yield and
            whose reply a new task row holds (see ``_covered``).
        duplicates: Repeated rows of sessions whose rollout is gone.
        uncovered_rows: Rows the parse does not yield and no new row covers; kept.
        waiting_sessions: Sessions on disk with parsed turns not yet in Milvus.
        waiting_turns: Parsed turns not yet in Milvus.
        unreadable_sessions: Sessions whose rollout failed to read; their rows are kept.
        sessions_on_disk: Indexed sessions whose rollout was found.
        sessions_missing: Indexed sessions whose rollout is gone.
    """

    stale: Dict[str, List[str]] = field(default_factory=dict)
    duplicates: Dict[str, List[str]] = field(default_factory=dict)
    waiting_sessions: int = 0
    waiting_turns: int = 0
    uncovered_rows: int = 0
    unreadable_sessions: int = 0
    sessions_on_disk: int = 0
    sessions_missing: int = 0


@dataclass
class CodexDedupeResult:
    """What ``apply`` deleted.

    Attributes:
        deleted_rows: Rows passed to ``rag_engine.delete_by_doc_id``.
        fts_failed: Deletes that reached Milvus but not FTS; a re-run retries them.
        audit_path: The JSONL file that lists every deleted doc_id.
    """

    deleted_rows: int = 0
    fts_failed: int = 0
    audit_path: str = ""


def _milvus_rows(db_path: Optional[str]) -> List[dict]:
    expr = f'provider == "{PROVIDER}" and chunk_type == "turn"'
    fields = ["doc_id", "session_id", "turn_index", "document"]
    return [row for batch in rag_engine._query_batches(fields, filter_expr=expr, db_path=db_path)
            for row in batch]


def _fts_rows(db_path: Optional[str]) -> List[dict]:
    if not db_path:
        return []
    conn = rag_engine._fts.connection(db_path)
    try:
        found = rag_engine._fts.select_where(
            conn, ["doc_id", "session_id", "turn_index", "content"],
            "provider = ? AND chunk_type = 'turn'", (PROVIDER,))
    finally:
        rag_engine._fts.close_ephemeral(conn)
    return [{"doc_id": d, "session_id": s, "turn_index": int(t), "document": c}
            for d, s, t, c in found]


def _indexed_rows(db_path: Optional[str]) -> List[dict]:
    """Return every Codex turn row once; ``in_milvus`` is False for rows only FTS holds."""
    rows = {row["doc_id"]: {**row, "in_milvus": True} for row in _milvus_rows(db_path)}
    for row in _fts_rows(db_path):
        rows.setdefault(row["doc_id"], {**row, "in_milvus": False})
    return list(rows.values())


def _parse_sessions() -> Dict[str, Optional[List[dict]]]:
    """Parse every Codex rollout on disk.

    A session whose tasks are all skipped maps to []; one whose rollout failed to
    read maps to None.
    """
    from provider_codex import CodexAdapter

    adapter = CodexAdapter()
    parsed: Dict[str, Optional[List[dict]]] = {}
    for source in adapter.discover_sources():
        result = adapter.parse_source(source, cursor=None)
        session_id = source.logical_session_id
        if result.errors or (session_id in parsed and parsed[session_id] is None):
            parsed[session_id] = None
            continue
        parsed.setdefault(session_id, []).extend(result.turns)
    return parsed


def _would_store(turn: dict) -> bool:
    # Mirrors rag_engine._apply_boilerplate, which drops a turn stripping leaves too short.
    if not rag_engine._boilerplate_enabled():
        return True
    clean, removed = boilerplate.strip(turn.get("text", ""))
    return not removed or len(clean.strip()) >= rag_engine.MIN_TURN_CHARS


def _duplicate_rows(rows: List[dict]) -> List[str]:
    """Return repeats of an earlier row and assistant-only copies of a User + Assistant row.

    The rollout is gone, so a deleted vector cannot be rebuilt: a Milvus row goes only
    when a Milvus row with the same turn stays, and Milvus copies are kept first.
    """
    # Each User + Assistant document maps to whether a Milvus row holds it.
    with_user: Dict[str, bool] = {}
    for row in rows:
        document = row.get("document") or ""
        if _REPLY_MARKER in document:
            with_user[document] = with_user.get(document, False) or row["in_milvus"]
    seen: Set[str] = set()
    duplicates: List[str] = []
    order = sorted(rows, key=lambda r: (not r["in_milvus"], int(r["turn_index"]), r["doc_id"]))
    for row in order:
        document = row.get("document") or ""
        if document in seen or (
            not document.startswith("User: ")
            and any(doc.endswith(_REPLY_MARKER + document)
                    and (backed or not row["in_milvus"])
                    for doc, backed in with_user.items())
        ):
            duplicates.append(row["doc_id"])
        else:
            seen.add(document)
    return duplicates


def plan(db_path: Optional[str] = None) -> CodexDedupePlan:
    """Read the indexed Codex rows and the rollouts on disk, and choose rows to delete. Writes nothing."""
    by_session: Dict[str, List[dict]] = {}
    for row in _indexed_rows(db_path):
        by_session.setdefault(row["session_id"], []).append(row)
    parsed = _parse_sessions()

    result = CodexDedupePlan()
    for session_id in sorted(by_session):
        rows = by_session[session_id]
        if session_id not in parsed:
            result.sessions_missing += 1
            duplicates = _duplicate_rows(rows)
            if duplicates:
                result.duplicates[session_id] = duplicates
            continue
        result.sessions_on_disk += 1
        turns = parsed[session_id]
        if turns is None:
            result.unreadable_sessions += 1
            continue
        in_milvus = {row["doc_id"] for row in rows if row["in_milvus"]}
        missing = {t["doc_id"] for t in turns if _would_store(t)} - in_milvus
        if missing:
            result.waiting_sessions += 1
            result.waiting_turns += len(missing)
            continue
        current = {t["doc_id"] for t in turns}
        texts = _covering_texts(turns)
        stale: List[str] = []
        for row in rows:
            if row["doc_id"] in current:
                continue
            if _covered(row, texts):
                stale.append(row["doc_id"])
            else:
                result.uncovered_rows += 1
        if stale:
            result.stale[session_id] = sorted(stale)
    return result


def _covering_texts(turns: List[dict]) -> List[str]:
    # Stored documents went through boilerplate stripping, so compare both forms.
    texts: List[str] = []
    for turn in turns:
        text = turn.get("text", "")
        texts.extend((text, boilerplate.strip(text)[0]))
    return texts


def _covered(row: dict, texts: List[str]) -> bool:
    """Whether a new task row holds this row's whole reply.

    Import turns, text past the task cut and text from an older rollout layout are
    not covered, and their rows stay until a parse can rebuild them.
    """
    document = row.get("document") or ""
    if document.startswith("User: ") and _REPLY_MARKER in document:
        document = document.split(_REPLY_MARKER, 1)[1]
    reply = document.removesuffix("[truncated]").strip()
    return bool(reply) and any(reply in text for text in texts)


def _audit_dir() -> Path:
    return Path.home() / ".sessionflow" / "audit"


def apply(plan: CodexDedupePlan, db_path: Optional[str] = None) -> CodexDedupeResult:
    """Delete the planned rows, recording each doc_id in a 0600 audit file before its delete."""
    import reembed

    audit_dir = reembed._ensure_private_dir(_audit_dir())
    audit_path = audit_dir / f"dedupe-codex-{uuid.uuid4().hex[:12]}.jsonl"
    result = CodexDedupeResult(audit_path=str(audit_path))
    fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as audit:
        for reason, groups in (("stale", plan.stale), ("duplicate", plan.duplicates)):
            for session_id, doc_ids in sorted(groups.items()):
                for doc_id in doc_ids:
                    audit.write(json.dumps(
                        {"doc_id": doc_id, "session_id": session_id, "reason": reason}) + "\n")
                    audit.flush()
                    outcome = rag_engine.delete_by_doc_id(doc_id, db_path=db_path)
                    result.deleted_rows += 1
                    if not outcome.fts_ok:
                        result.fts_failed += 1
    return result
