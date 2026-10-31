"""Remove Codex rows left by keying turns on rollout line numbers.

A session whose rollout is on disk keeps the rows its parse yields. Its other rows
are deleted only once the backfill has indexed every parsed turn, so search never
loses a turn. A session whose rollout is gone keeps one row per distinct turn.
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
        stale: Rows of sessions on disk that the current parse does not yield.
        duplicates: Repeated rows of sessions whose rollout is gone.
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
    """Return repeats of an earlier row and assistant-only copies of a User + Assistant row."""
    documents = [row.get("document") or "" for row in rows]
    with_user = [doc for doc in documents if _REPLY_MARKER in doc]
    seen: Set[str] = set()
    duplicates: List[str] = []
    for row in sorted(rows, key=lambda r: (int(r["turn_index"]), r["doc_id"])):
        document = row.get("document") or ""
        if document in seen or (
            not document.startswith("User: ")
            and any(doc.endswith(_REPLY_MARKER + document) for doc in with_user)
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
        stale = sorted(row["doc_id"] for row in rows if row["doc_id"] not in current)
        if stale:
            result.stale[session_id] = stale
    return result


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
