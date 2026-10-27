"""Memory validation, write orchestration, search and related-conversation lookups.

Memory records are curated notes written by agents (session summaries,
decisions, handoffs, facts). They are stored verbatim with fixed metadata and
no LLM step. The agent vocabulary here is separate from the transcript
provider allowlist in ``provider_adapters``: one names who wrote a memory, the
other names where a transcript came from.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Pattern

import memory_store
import secret_redaction
from collection_specs import memories_spec
from embedding_control import _env_float, _env_int
from memory_store import (
    MemoryFilters,
    MemoryNotFound,
    MemoryStore,
    MemoryStoreError,
    NewMemory,
    content_hash,
    run_db,
)

LEGAL_MEMORY_TYPES = frozenset({
    "session-summary",
    "decision",
    "handoff",
    "fact",
    "workflow",
    "discovery",
    "contact",
    "blocker",
    "preference",
    "open-item",
    "milestone",
    "guardrail",
})

LEGAL_MEMORY_AGENTS = frozenset({
    "claude-code",
    "codex",
    "gemini",
    "opencode",
    "antigravity",
})

# Accepted only on the import path, for mem0 records with no usable writer.
# Later updates that omit agent_id keep it.
IMPORT_ONLY_AGENT = "unknown"

MAX_MEMORY_BYTES = 65536
"""Largest accepted memory text, counted in UTF-8 bytes."""

DEFAULT_USER_ID = memory_store.DEFAULT_USER_ID


def _index_key_limit(name: str) -> int:
    # The embedding dimension does not affect scalar field limits.
    return next(f.max_length for f in memories_spec(1).fields if f.name == name)


# Longest project and user_id the memories index collection accepts, in UTF-8 bytes.
MAX_PROJECT_BYTES = _index_key_limit("project")
MAX_USER_ID_BYTES = _index_key_limit("user_id")

# Types the audit lists when they stay active longer than the audit age.
STALE_TYPES = frozenset({"open-item", "blocker"})

# Calibrated on the local backup with the transcript model: duplicate wordings of
# one fact score 0.88 and up, while distinct facts on one topic score below it.
SIMILAR_THRESHOLD = 0.88
# Open items and blockers older than this are listed by the audit.
AUDIT_AGE_DAYS = 21

_PROVENANCE_TEXT = ("source_session_id", "source_provider", "source_project_root")
_PROVENANCE_TIME = ("source_started_at", "source_ended_at")
# Fields a caller may change with update_memory; text changes recompute the
# content hash and issue IDs.
_UPDATE_FIELDS = frozenset({"text", "agent_id", "type", "topic_key", "extra",
                            *_PROVENANCE_TEXT, *_PROVENANCE_TIME})


class MemoryInputError(MemoryStoreError):
    """A request that fails validation; nothing was written."""


def _issue_ids(text: str) -> List[str]:
    import rag_engine

    return [i for i in rag_engine._extract_issue_ids(text).split(",") if i]


def _matches(row: Dict[str, Any], f: MemoryFilters) -> bool:
    """Return whether a canonical row passes every listing filter."""
    extra = row.get("extra") or {}
    checks = (
        row["user_id"] == f.user_id,
        f.project == "*" or row["project"] == f.project,
        f.status is None or row["status"] == f.status,
        f.type is None or row["type"] == f.type,
        f.agent_id is None or row["agent_id"] == f.agent_id,
        f.issue_id is None or f.issue_id.strip().upper() in row["issue_ids"],
        f.source is None or extra.get("source") == f.source,
        f.category is None or extra.get("category") == f.category,
        f.source_session_id is None or row["source_session_id"] == f.source_session_id,
        f.created_from is None
        or row["created_at"] >= memory_store.chicago_day_start(f.created_from),
        f.created_to is None
        or row["created_at"] < memory_store.chicago_day_start(f.created_to + timedelta(days=1)),
    )
    return all(checks)


def _memory_day(row: Dict[str, Any]):
    """Return the memory's day: ``extra.date`` when set, else its America/Chicago creation day."""
    raw = (row.get("extra") or {}).get("date")
    if raw:
        try:
            return datetime.fromisoformat(str(raw)[:10]).date()
        except ValueError:
            pass
    return row["created_at"].astimezone(memory_store.CHICAGO).date()


def _known_project_slugs() -> List[tuple[str, str]]:
    """Return ``(root, project)`` for each registered root with a readable ``.claude/project.json``.

    The roots are the ones the server has registered from transcripts, so no
    folder layout is assumed. The project is ``memory.project``, else ``tag``.
    """
    import json
    from pathlib import Path

    import file_watcher

    pairs = []
    for root in file_watcher.known_project_roots():
        try:
            data = json.loads((Path(root) / ".claude" / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        slug = (data.get("memory") or {}).get("project") or str(data.get("tag") or "").lower()
        if slug:
            pairs.append((root, slug))
    return pairs


def resolve_project_roots(project: str) -> List[str]:
    """Return the known project roots whose ``.claude/project.json`` names ``project``."""
    return [root for root, slug in _known_project_slugs() if slug == project]


def project_inventory() -> set[str]:
    """Return every project named by a known root's ``.claude/project.json``."""
    return {slug for _, slug in _known_project_slugs()}


def _parse_time(name: str, value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise MemoryInputError("invalid_field", f"{name} must be an ISO-8601 timestamp",
                                   field=name) from exc
    if parsed.tzinfo is None:
        raise MemoryInputError("invalid_field", f"{name} needs a UTC offset", field=name)
    return parsed


class MemoryService:
    """Validates memory requests, writes them to the store, then updates the indexes.

    The store commit is the success boundary. An index failure after it
    returns success with ``index: "pending"``, and the sync loop repairs it.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        index: Any,
        embed_query: Callable[[str], Awaitable[List[float]]],
        allowlist: Optional[List[Pattern[str]]] = None,
        similar_threshold: Optional[float] = None,
        audit_age_days: Optional[int] = None,
        project_inventory: Optional[Iterable[str]] = None,
    ) -> None:
        """Wire the store, the index channels and the query embedder.

        ``allowlist`` holds the redaction allowlist patterns; a match is never
        treated as a secret.
        """
        self.store = store
        self.index = index
        self.embed_query = embed_query
        self.allowlist = list(allowlist or [])
        self.similar_threshold = similar_threshold if similar_threshold is not None else _env_float(
            "SESSIONFLOW_MEMORY_SIMILAR_THRESHOLD", SIMILAR_THRESHOLD, 0.0, 1.0)
        self.audit_age_days = audit_age_days if audit_age_days is not None else _env_int(
            "SESSIONFLOW_MEMORY_AUDIT_AGE_DAYS", AUDIT_AGE_DAYS, minimum=1)
        self.project_inventory = set(project_inventory or ())

    # --- Validation ---------------------------------------------------------

    def _check_text(self, text: Any) -> str:
        if not isinstance(text, str) or text == "":
            raise MemoryInputError("missing_field", "text is required", field="text")
        size = len(text.encode("utf-8"))
        if size > MAX_MEMORY_BYTES:
            raise MemoryInputError(
                "too_large",
                f"text is {size} bytes; the limit is {MAX_MEMORY_BYTES} bytes (UTF-8)",
                limit=MAX_MEMORY_BYTES, received=size,
            )
        _, spans = secret_redaction.scan_spans(text, mode="report", allowlist=self.allowlist)
        if spans:
            rules = sorted({s.rule_name for s in spans})
            raise MemoryInputError(
                "secret_detected",
                f"text matches the secret pattern {', '.join(rules)}; nothing was stored",
                rules=rules,
            )
        return text

    @staticmethod
    def _check_agent(agent_id: Any) -> str:
        if agent_id in (None, ""):
            raise MemoryInputError("missing_field", "agent_id is required", field="agent_id")
        if agent_id not in LEGAL_MEMORY_AGENTS:
            raise MemoryInputError(
                "invalid_agent",
                f"agent_id {agent_id!r} is not allowed; use one of "
                f"{', '.join(sorted(LEGAL_MEMORY_AGENTS))}",
                allowed=sorted(LEGAL_MEMORY_AGENTS),
            )
        return agent_id

    @staticmethod
    def _check_type(type_: Any) -> str:
        if type_ in (None, ""):
            raise MemoryInputError("missing_field", "type is required", field="type")
        if type_ not in LEGAL_MEMORY_TYPES:
            raise MemoryInputError(
                "invalid_type",
                f"type {type_!r} is not allowed; use one of {', '.join(sorted(LEGAL_MEMORY_TYPES))}",
                allowed=sorted(LEGAL_MEMORY_TYPES),
            )
        return type_

    @staticmethod
    def _check_index_key(name: str, value: str, limit: int) -> str:
        size = len(value.encode("utf-8"))
        if size > limit:
            raise MemoryInputError(
                "invalid_field",
                f"{name} is {size} bytes; the limit is {limit} bytes (UTF-8)",
                field=name, limit=limit, received=size,
            )
        return value

    @classmethod
    def _write_project(cls, project: Any) -> str:
        if not isinstance(project, str) or not project.strip():
            raise MemoryInputError("missing_field", "project is required", field="project")
        if project.strip() == "*":
            raise MemoryInputError("invalid_project", "A write needs one project, not \"*\"")
        return cls._check_index_key("project", project.strip(), MAX_PROJECT_BYTES)

    @classmethod
    def _write_user_id(cls, user_id: Any) -> str:
        if not user_id:
            return DEFAULT_USER_ID
        if not isinstance(user_id, str):
            raise MemoryInputError("invalid_field", "user_id must be a string", field="user_id")
        return cls._check_index_key("user_id", user_id, MAX_USER_ID_BYTES)

    @staticmethod
    def _scope_project(project: Any) -> str:
        if not isinstance(project, str) or not project.strip():
            raise MemoryInputError(
                "project_required", "project is required; pass \"*\" to span every project"
            )
        return project.strip()

    @staticmethod
    def _check_extra(extra: Any) -> Dict[str, Any]:
        if extra is None:
            return {}
        if not isinstance(extra, dict):
            raise MemoryInputError("invalid_field", "extra must be an object", field="extra")
        return extra

    # --- Writes -------------------------------------------------------------

    async def add(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and add one memory, then index it and list similar records."""
        text = self._check_text(args.get("text"))
        agent_id = self._check_agent(args.get("agent_id"))
        project = self._write_project(args.get("project"))
        type_ = self._check_type(args.get("type"))
        rec = NewMemory(
            text=text,
            agent_id=agent_id,
            project=project,
            type=type_,
            content_hash=content_hash(text),
            user_id=self._write_user_id(args.get("user_id")),
            topic_key=args.get("topic_key") or None,
            extra=self._check_extra(args.get("extra")),
            issue_ids=_issue_ids(text),
            supersedes=list(args.get("supersedes") or []),
            **{k: args.get(k) or None for k in _PROVENANCE_TEXT},
            **{k: _parse_time(k, args.get(k)) for k in _PROVENANCE_TIME},
        )
        result = await run_db(self.store.add, rec)
        out = self._write_payload(result)
        out["index"] = await self._apply(result.memory_id)
        out["similar"], out["similar_state"] = await self._similar(text, project, rec.user_id,
                                                                   result.memory_id)
        return out

    async def update(self, memory_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Change the given fields of one memory; ``args["project"]`` scopes the call."""
        project = args.get("project")
        changes: Dict[str, Any] = {}
        unknown = set(args) - _UPDATE_FIELDS - {"project", "format"}
        if unknown:
            raise MemoryInputError("invalid_field", f"Cannot update {', '.join(sorted(unknown))}",
                                   fields=sorted(unknown))
        if "text" in args:
            changes["text"] = self._check_text(args["text"])
            changes["content_hash"] = content_hash(changes["text"])
            changes["issue_ids"] = _issue_ids(changes["text"])
        if "agent_id" in args:
            changes["agent_id"] = self._check_agent(args["agent_id"])
        if "type" in args:
            changes["type"] = self._check_type(args["type"])
        if "topic_key" in args:
            changes["topic_key"] = args["topic_key"] or None
        if "extra" in args:
            changes["extra"] = self._check_extra(args["extra"])
        for key in _PROVENANCE_TEXT:
            if key in args:
                changes[key] = args[key] or None
        for key in _PROVENANCE_TIME:
            if key in args:
                changes[key] = _parse_time(key, args[key])
        if not changes:
            raise MemoryInputError("missing_field", "Pass at least one field to change")
        result = await run_db(self.store.update, memory_id, changes, expect_project=project)
        out = self._write_payload(result)
        out["index"] = await self._apply(result.memory_id)
        return out

    async def delete(self, memory_id: str, project: Optional[str] = None) -> Dict[str, Any]:
        """Delete one memory as a tombstone, then remove it from the indexes."""
        result = await run_db(self.store.delete, memory_id, expect_project=project)
        out = self._write_payload(result)
        out["index"] = await self._apply(result.memory_id)
        return out

    @staticmethod
    def _write_payload(result: memory_store.WriteResult) -> Dict[str, Any]:
        return {
            "memory_id": result.memory_id,
            "outcome": result.outcome,
            "version": result.version,
            "created_at": result.created_at,
            "updated_at": result.updated_at,
        }

    async def _apply(self, memory_id: str) -> str:
        try:
            return await self.index.apply(memory_id)
        except Exception:  # noqa: BLE001 - the sync loop retries every pending record
            return "pending"

    async def _similar(self, text: str, project: str, user_id: str,
                       memory_id: str) -> tuple[List[Dict[str, Any]], str]:
        try:
            found, state = await self.index.similar(
                text, project=project, user_id=user_id, threshold=self.similar_threshold
            )
        except Exception:  # noqa: BLE001 - similarity is advisory; report it as pending
            return [], "pending"
        return [s for s in found if s.get("memory_id") != memory_id], state

    # --- Reads --------------------------------------------------------------

    async def get(self, memory_id: str, project: Optional[str] = None) -> Dict[str, Any]:
        """Return one live memory; an unknown or deleted ID raises ``not_found``."""
        row = await run_db(self.store.get, memory_id, expect_project=project)
        if row is None:
            raise MemoryNotFound(memory_id)
        return row

    async def list(self, filters: Dict[str, Any], limit: int = 20,
                   cursor: Optional[str] = None) -> Dict[str, Any]:
        """Return one page newest first, the next cursor, and the nightly dump status.

        ``dump`` carries the dump state so the session-start hook can report a
        failed or missing run; a failed status read reports ``unknown``.
        """
        parsed = self._filters(filters)
        rows, next_cursor = await run_db(self.store.list, parsed, limit, cursor)
        try:
            dump = await run_db(self.store.dump_status)
        except Exception:  # noqa: BLE001 - the listing stays useful without it
            dump = {"state": "unknown"}
        return {"results": rows, "next_cursor": next_cursor, "dump": dump}

    # --- Audit --------------------------------------------------------------

    async def audit(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        """Report likely duplicates, aged open items and blockers, counts and unknown projects.

        Read-only: nothing is written. Clusters join active records whose
        vectors are at least the similarity threshold apart, per project.
        """
        rows = await run_db(self.store.audit_rows, user_id)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.audit_age_days)
        projects: Dict[str, Dict[str, Any]] = {}
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_project: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_project.setdefault(row["project"], []).append(row)
        for project, items in sorted(by_project.items()):
            counts: Dict[str, Dict[str, int]] = {}
            for row in items:
                bucket = counts.setdefault(row["type"], {})
                bucket[row["status"]] = bucket.get(row["status"], 0) + 1
            active = [r for r in items if r["status"] == "active"]
            ids = {r["memory_id"] for r in active}
            for mid in ids:
                parent[mid] = mid
            for mid in sorted(ids):
                for other in await self.index.neighbours(mid, threshold=self.similar_threshold):
                    if other in ids:
                        parent[find(other)] = find(mid)
            groups: Dict[str, List[str]] = {}
            for mid in ids:
                groups.setdefault(find(mid), []).append(mid)
            clusters = sorted(sorted(g) for g in groups.values() if len(g) > 1)
            stale = sorted(r["memory_id"] for r in active
                           if r["type"] in STALE_TYPES and r["created_at"] < cutoff)
            projects[project] = {"clusters": clusters, "stale": stale, "counts": counts}
        unknown = sorted(p for p in projects if self.project_inventory
                         and p not in self.project_inventory)
        return {"projects": projects, "unknown_projects": unknown,
                "inventory_checked": bool(self.project_inventory),
                "similar_threshold": self.similar_threshold,
                "audit_age_days": self.audit_age_days}

    # --- Related conversations ---------------------------------------------

    async def context(self, memory_id: str, project: Optional[str] = None,
                      limit: int = 10) -> Dict[str, Any]:
        """Return a memory, the session that produced it, and related sessions.

        With a source session: its last ``limit`` turns, and other sessions that
        share the memory's issue IDs, newest first. Without one: the union of
        sessions sharing its issue IDs and sessions in the project on its day,
        with a note that no source session is linked.
        """
        import asyncio

        import rag_engine

        row = await self.get(memory_id, project)
        issue_ids = list(row.get("issue_ids") or [])
        limit = max(1, int(limit))
        source_id = row.get("source_session_id")
        if source_id:
            provider = row.get("source_provider")
            turns = await asyncio.to_thread(
                lambda: rag_engine.session_turns(source_id, provider, limit=limit))
            related = []
            if issue_ids:
                related = await asyncio.to_thread(lambda: rag_engine.sessions_by_issue(
                    issue_ids, limit=limit, project_roots=None, exclude_session=source_id))
            return {"memory": row,
                    "source_session": {"session_id": source_id, "provider": provider,
                                       "project_root": row.get("source_project_root"),
                                       "turns": turns},
                    "related_sessions": related, "note": None}

        found: List[Dict[str, Any]] = []
        if issue_ids:
            found += await asyncio.to_thread(
                lambda: rag_engine.sessions_by_issue(issue_ids, limit=limit))
        day = _memory_day(row)
        roots = await asyncio.to_thread(resolve_project_roots, row["project"])
        if roots:
            found += await asyncio.to_thread(lambda: rag_engine.sessions_by_issue(
                [], limit=limit, project_roots=roots, day=day))
        merged: Dict[tuple, Dict[str, Any]] = {}
        for s in found:
            merged.setdefault((s.get("provider"), s.get("session_id")), s)
        related = sorted(merged.values(), key=lambda s: str(s.get("last_timestamp") or ""),
                         reverse=True)[:limit]
        return {"memory": row, "source_session": None, "related_sessions": related,
                "note": "No source session is linked to this memory; showing sessions that "
                        "share its issue IDs and sessions in its project on "
                        f"{day.isoformat()}."}

    # --- Search -------------------------------------------------------------

    async def search(self, query: str, filters: Dict[str, Any], n: int = 10,
                     sort_by: str = "hybrid") -> Dict[str, Any]:
        """Hybrid memory search validated against Postgres.

        Both channels overfetch ``3*n``. Every hit is checked against the
        canonical row (version, and generation for vectors) and the filters,
        so a stale index entry never hides a current one. A short result is
        refilled once with a larger fetch; ``incomplete`` is True when it is
        still short while the indexes lag.
        """
        parsed = self._filters(filters)
        n = max(1, int(n))
        fetch = 3 * n
        results, channels = await self._search_once(query, parsed, n, fetch, sort_by)
        if len(results) < n:
            results, channels = await self._search_once(query, parsed, n, fetch * 3, sort_by)
        incomplete = len(results) < n and bool(self.index.lagging())
        return {"results": results, "channels": channels, "incomplete": incomplete}

    @staticmethod
    def _channel_filters(f: MemoryFilters) -> Dict[str, Any]:
        return {"project": f.project, "user_id": f.user_id, "status": f.status, "type": f.type,
                "agent_id": f.agent_id, "issue_id": f.issue_id, "source": f.source,
                "category": f.category}

    async def _search_once(self, query: str, f: MemoryFilters, n: int, fetch: int,
                           sort_by: str) -> tuple[List[Dict[str, Any]], List[str]]:
        import rag_engine
        from fts_hybrid import rrf_merge

        channel_filters = self._channel_filters(f)
        channels: List[str] = []
        vector_hits: List[Dict[str, Any]] = []
        fts_hits: List[Dict[str, Any]] = []
        try:
            vector = await self.embed_query(query)
            vector_hits = await self.index.vector_search(vector, filters=channel_filters,
                                                         limit=fetch)
            channels.append("vector")
        except Exception:  # noqa: BLE001 - Milvus down: keyword search alone still answers
            vector_hits = []
        try:
            fts_hits = await self.index.fts_search(query, filters=channel_filters, limit=fetch)
            channels.append("fts")
        except Exception:  # noqa: BLE001 - a broken FTS file is rebuilt; vectors still answer
            fts_hits = []

        ids = {h["memory_id"] for h in vector_hits} | {h["memory_id"] for h in fts_hits}
        rows = await run_db(self.store.get_many, sorted(ids)) if ids else {}
        generation = (await run_db(self.store.index_meta))["generation"] if vector_hits else None

        def keep(hit: Dict[str, Any], check_generation: bool) -> bool:
            row = rows.get(hit["memory_id"])
            if row is None or hit.get("version") != row["version"]:
                return False
            if check_generation and hit.get("generation") != generation:
                return False
            return _matches(row, f)

        valid_vector = [{"doc_id": h["memory_id"]} for h in vector_hits if keep(h, True)]
        valid_fts = [{"doc_id": h["memory_id"]} for h in fts_hits if keep(h, False)]
        merged = rrf_merge(valid_vector, valid_fts, n=len(valid_vector) + len(valid_fts))
        scored = [{"memory_id": m["doc_id"], "_rrf_score": m["_rrf_score"],
                   "created_at": rows[m["doc_id"]]["created_at"].isoformat()} for m in merged]
        ranked = rag_engine._rank_results(scored, sort_by, n, timestamp_key="created_at",
                                          tie_break="memory_id")
        return [rows[r["memory_id"]] for r in ranked], channels

    def _filters(self, filters: Dict[str, Any]) -> MemoryFilters:
        project = self._scope_project(filters.get("project"))
        status = filters.get("status", "active")
        if status == "any":
            status = None
        days = {}
        for key in ("created_from", "created_to"):
            value = filters.get(key)
            if value:
                try:
                    days[key] = datetime.fromisoformat(str(value)).date()
                except ValueError as exc:
                    raise MemoryInputError("invalid_field", f"{key} must be YYYY-MM-DD",
                                           field=key) from exc
        return MemoryFilters(
            project=project,
            user_id=filters.get("user_id") or DEFAULT_USER_ID,
            agent_id=filters.get("agent_id") or None,
            type=filters.get("type") or None,
            status=status,
            issue_id=filters.get("issue_id") or None,
            source=filters.get("source") or None,
            category=filters.get("category") or None,
            source_session_id=filters.get("source_session_id") or None,
            **days,
        )
