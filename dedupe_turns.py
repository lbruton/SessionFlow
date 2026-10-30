"""Repair Claude Code turns indexed mid-write: duplicates and turns missing their reply.

A turn indexed before its reply was written is stored user-only. When another
parse indexed the completed turn, both rows remain under one session and
``turn_index``. ``plan`` finds these rows; ``apply`` re-parses each affected
session's transcript and passes the affected turns to ``rag_engine.add_turns``,
whose same-turn replacement deletes the stale rows. A session whose transcript
is gone keeps the longest row of each duplicate group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import rag_engine

PROVIDER = "claude_code_cli"
_REPLY_MARKER = "\n\nAssistant: "

Group = Tuple[str, int]


@dataclass
class DedupePlan:
    """Indexed turn groups that need repair.

    Attributes:
        groups: Affected ``(session_id, turn_index)`` groups and their rows.
        duplicate_groups: Groups holding more than one row.
        user_only_rows: Single-row groups whose row has no reply.
    """

    groups: Dict[Group, List[dict]] = field(default_factory=dict)
    duplicate_groups: int = 0
    user_only_rows: int = 0

    @property
    def affected_sessions(self) -> Set[str]:
        """Return the sessions that hold an affected group."""
        return {session_id for session_id, _ in self.groups}


@dataclass
class DedupeResult:
    """What ``apply`` changed.

    Attributes:
        rebuilt_turns: Turns re-parsed from a transcript and confirmed stored.
        deferred_turns: Rebuilt turns ``add_turns`` did not store; their old rows stay.
        deleted_rows: Rows deleted by ``apply`` itself, outside ``add_turns``.
        fts_failed: Deletes that reached Milvus but not FTS.
        sessions_on_disk: Affected sessions whose transcript was found.
        sessions_missing: Affected sessions whose transcript is gone.
    """

    rebuilt_turns: int = 0
    deferred_turns: int = 0
    deleted_rows: int = 0
    fts_failed: int = 0
    sessions_on_disk: int = 0
    sessions_missing: int = 0


def _has_reply(row: dict) -> bool:
    return _REPLY_MARKER in (row.get("document") or "")


def _longest(rows: List[dict]) -> dict:
    return max(rows, key=lambda r: (_has_reply(r), len(r.get("document") or "")))


def _milvus_rows(db_path: Optional[str]) -> List[dict]:
    expr = f'provider == "{PROVIDER}" and chunk_type == "turn"'
    fields = ["doc_id", "session_id", "turn_index", "document", "project_root"]
    return [row for batch in rag_engine._query_batches(fields, filter_expr=expr, db_path=db_path)
            for row in batch]


def _fts_rows(db_path: Optional[str]) -> List[dict]:
    if not db_path:
        return []
    conn = rag_engine._fts.connection(db_path)
    try:
        found = rag_engine._fts.select_where(
            conn, ["doc_id", "session_id", "turn_index", "content", "project_root"],
            "provider = ? AND chunk_type = 'turn'", (PROVIDER,))
    finally:
        rag_engine._fts.close_ephemeral(conn)
    return [{"doc_id": d, "session_id": s, "turn_index": int(t), "document": c,
             "project_root": p} for d, s, t, c, p in found]


def _stored_doc_ids(doc_ids: List[str], db_path: Optional[str]) -> Set[str]:
    found: Set[str] = set()
    with rag_engine.milvus_client(db_path) as client:
        for start in range(0, len(doc_ids), 500):
            chunk = doc_ids[start:start + 500]
            quoted = ", ".join(f'"{rag_engine._escape_filter_scalar(d)}"' for d in chunk)
            rows = client.query(collection_name=rag_engine.COLLECTION_NAME,
                                filter=f"doc_id in [{quoted}]", output_fields=["doc_id"])
            found.update(r["doc_id"] for r in rows)
    return found


def _indexed_rows(db_path: Optional[str]) -> List[dict]:
    """Return every Claude Code turn row once, from Milvus and from FTS.

    FTS is read too, because a failed FTS delete leaves a row that Milvus no
    longer holds but keyword search still returns.
    """
    rows = {row["doc_id"]: row for row in _milvus_rows(db_path)}
    for row in _fts_rows(db_path):
        rows.setdefault(row["doc_id"], row)
    return list(rows.values())


def _parse_sessions(session_ids: Iterable[str]) -> Dict[str, List[dict]]:
    from provider_claude import ClaudeCodeCliAdapter

    wanted = set(session_ids)
    adapter = ClaudeCodeCliAdapter()
    parsed: Dict[str, List[dict]] = {}
    for source in adapter.discover_sources():
        if source.logical_session_id in wanted:
            parsed.setdefault(source.logical_session_id, []).extend(
                adapter.parse_source(source, cursor=None).turns)
    return parsed


def plan(db_path: Optional[str] = None) -> DedupePlan:
    """Scan the Claude Code turns in Milvus for duplicate and user-only groups. Writes nothing."""
    by_group: Dict[Group, List[dict]] = {}
    for row in _indexed_rows(db_path):
        by_group.setdefault((row["session_id"], int(row["turn_index"])), []).append(row)

    result = DedupePlan()
    for key, rows in by_group.items():
        if len(rows) > 1:
            result.duplicate_groups += 1
            result.groups[key] = rows
        elif not _has_reply(rows[0]):
            result.user_only_rows += 1
            result.groups[key] = rows
    return result


def _delete(doc_id: str, db_path: Optional[str], result: DedupeResult) -> None:
    outcome = rag_engine.delete_by_doc_id(doc_id, db_path=db_path)
    result.deleted_rows += 1
    if not outcome.fts_ok:
        result.fts_failed += 1


def _keep_longest(rows: List[dict], db_path: Optional[str], result: DedupeResult) -> None:
    keep = _longest(rows)
    for row in rows:
        if row is not keep:
            _delete(row["doc_id"], db_path, result)


def apply(plan: DedupePlan, db_path: Optional[str] = None) -> DedupeResult:
    """Repair every planned group: rebuild from the transcript, else keep the longest row.

    A rebuilt turn keeps the ``project_root`` of its longest indexed row, so the
    repair does not move turns between projects.
    """
    result = DedupeResult()
    parsed = _parse_sessions(plan.affected_sessions)
    by_session: Dict[str, Dict[int, List[dict]]] = {}
    for (session_id, turn_index), rows in plan.groups.items():
        by_session.setdefault(session_id, {})[turn_index] = rows

    for session_id in sorted(by_session):
        groups = by_session[session_id]
        turns = {int(t["turn_index"]): t for t in parsed.get(session_id, [])
                 if t.get("chunk_type") == "turn"}
        if session_id in parsed:
            result.sessions_on_disk += 1
        else:
            result.sessions_missing += 1

        rebuilt, superseded = [], {}
        for turn_index, rows in sorted(groups.items()):
            turn = turns.get(turn_index)
            if turn is not None:
                rebuilt.append({**turn, "project_root": _longest(rows).get("project_root", "")})
                superseded[turn["doc_id"]] = [r for r in rows if r["doc_id"] != turn["doc_id"]]
            elif len(rows) > 1:
                _keep_longest(rows, db_path, result)
        if not rebuilt:
            continue
        rag_engine.add_turns(rebuilt, db_path=db_path)
        # add_turns may defer turns when the embedding budget refuses a batch,
        # so a predecessor is deleted only once its replacement is stored.
        stored = _stored_doc_ids(list(superseded), db_path)
        for doc_id, old_rows in superseded.items():
            if doc_id not in stored:
                result.deferred_turns += 1
                continue
            result.rebuilt_turns += 1
            # Deleting by doc_id also clears rows that only FTS still holds.
            for row in old_rows:
                _delete(row["doc_id"], db_path, result)
    return result
