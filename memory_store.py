"""Canonical memory store on Postgres.

Every memory write commits here first. The ``memories`` Milvus collection and
the local FTS files are derived indexes that follow ``memory_index_acks``. A
write resets the acknowledgements of the record it changes in the same
transaction, so a crash between commit and indexing leaves it pending.

All methods are blocking. Async callers go through ``run_db``, which uses a
dedicated executor and never the ``mlx-embed`` thread.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import psycopg
from psycopg import errors as pg_errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

CHICAGO = ZoneInfo("America/Chicago")
DEFAULT_USER_ID = "lbruton"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "sql" / "memory"

# A unique_violation means a concurrent writer won; re-reading resolves it.
MAX_WRITE_RETRIES = 3

DSN_ENV = "SESSIONFLOW_MEMORY_DSN"

DUMP_JOB = "pg-dump"
DUMP_HOUR, DUMP_MINUTE = 2, 30
# A scheduled dump counts as due once it is this old.
DUMP_GRACE = timedelta(minutes=45)

_COLUMNS = (
    "memory_id", "user_id", "agent_id", "project", "type", "status", "topic_key",
    "content_hash", "text", "extra", "issue_ids", "supersedes",
    "source_session_id", "source_provider", "source_project_root",
    "source_started_at", "source_ended_at", "created_at", "updated_at",
    "deleted_at", "version", "change_seq",
)
_COLUMN_SQL = ", ".join(_COLUMNS)

# Fields an update may change. Identity, scope and bookkeeping stay fixed.
_UPDATABLE = frozenset({
    "text", "content_hash", "agent_id", "type", "status", "topic_key", "extra",
    "issue_ids", "source_session_id", "source_provider", "source_project_root",
    "source_started_at", "source_ended_at", "updated_at",
})


# --- Errors -----------------------------------------------------------------


class MemoryStoreError(Exception):
    """A store failure with a stable ``code`` for structured tool errors."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        """Store the error code, the message and any structured details."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class MemoryNotFound(MemoryStoreError):
    """No live record has this ID."""

    def __init__(self, memory_id: str) -> None:
        """Name the ID that matched no live record."""
        super().__init__("not_found", f"No memory with id {memory_id}", memory_id=memory_id)
        self.memory_id = memory_id


class MemoryConflict(MemoryStoreError):
    """A topic key and a content hash point at different records."""

    def __init__(self, memory_ids: Sequence[str], message: str = "") -> None:
        """Name every conflicting record ID."""
        ids = sorted(set(memory_ids))
        super().__init__(
            "conflict",
            message or f"Topic key and content hash match different records: {', '.join(ids)}",
            memory_ids=ids,
        )
        self.memory_ids = ids


class InvalidSupersede(MemoryStoreError):
    """A supersede target is missing, deleted or in another project."""

    def __init__(self, memory_ids: Sequence[str]) -> None:
        """Name every rejected supersede target."""
        ids = sorted(set(memory_ids))
        super().__init__(
            "invalid_supersede",
            f"Cannot supersede {', '.join(ids)}: each target must be a live record "
            "in the same user and project",
            memory_ids=ids,
        )
        self.memory_ids = ids


# --- Values -----------------------------------------------------------------


@dataclass
class NewMemory:
    """A record to add. ``created_at`` and ``updated_at`` default to now."""

    text: str
    agent_id: str
    project: str
    type: str
    content_hash: str
    user_id: str = DEFAULT_USER_ID
    status: str = "active"
    topic_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    issue_ids: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    source_session_id: Optional[str] = None
    source_provider: Optional[str] = None
    source_project_root: Optional[str] = None
    source_started_at: Optional[datetime] = None
    source_ended_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(frozen=True)
class WriteResult:
    """Outcome of one write: created, replaced, deduplicated, updated, unchanged or deleted."""

    memory_id: str
    outcome: str
    version: int
    created_at: datetime
    updated_at: datetime
    change_seq: Optional[int] = None


@dataclass(frozen=True)
class MemoryFilters:
    """Listing filters. ``project="*"`` spans every project; ``status=None`` any status.

    ``created_from`` and ``created_to`` are inclusive America/Chicago days.
    """

    project: str
    user_id: str = DEFAULT_USER_ID
    agent_id: Optional[str] = None
    type: Optional[str] = None
    status: Optional[str] = "active"
    issue_id: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    created_from: Optional[date] = None
    created_to: Optional[date] = None
    source_session_id: Optional[str] = None

    def fingerprint(self) -> str:
        """Return a short stable hash that binds a cursor to these filters."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chicago_day_start(day: date) -> datetime:
    """Return the UTC instant at which ``day`` starts in America/Chicago."""
    return datetime.combine(day, time(0), CHICAGO).astimezone(timezone.utc)


def content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- DSN --------------------------------------------------------------------


def fetch_dsn() -> str:
    """Return the memory store DSN from ``SESSIONFLOW_MEMORY_DSN``, without ever logging it.

    How the value reaches the environment (a launcher, a secrets manager, by
    hand) is the user's choice; SessionFlow reads nothing else.
    """
    value = (os.environ.get(DSN_ENV) or "").strip()
    if not value:
        raise MemoryStoreError(
            "dsn_unavailable", f"memory store not configured: set {DSN_ENV}"
        )
    return value


# --- Executor ---------------------------------------------------------------

_db_executor: Optional[ThreadPoolExecutor] = None


DB_WORKERS = 4
_in_flight = 0
_in_flight_lock = threading.Lock()


def db_executor() -> ThreadPoolExecutor:
    """Return the executor that runs every blocking store call in the server."""
    global _db_executor
    if _db_executor is None:
        _db_executor = ThreadPoolExecutor(max_workers=DB_WORKERS, thread_name_prefix="memory-db")
    return _db_executor


def db_calls_in_flight() -> int:
    """Return how many store calls are running on memory-db threads now.

    A cancelled ``run_db`` await does not stop its thread, so this counts calls
    that outlived a timeout as well.
    """
    return _in_flight


def _counted(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    global _in_flight
    with _in_flight_lock:
        _in_flight += 1
    try:
        return fn(*args, **kwargs)
    finally:
        with _in_flight_lock:
            _in_flight -= 1


async def run_db(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking store call on the memory-db executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(db_executor(), _counted, fn, args, kwargs)


# --- Nightly dump status ----------------------------------------------


def latest_due_dump_slot(now: datetime) -> datetime:
    """Return the latest scheduled 02:30 America/Chicago run at least 45 minutes old, in UTC."""
    local_day = now.astimezone(CHICAGO).date()
    for back in range(3):
        day = local_day - timedelta(days=back)
        slot = datetime(day.year, day.month, day.day, DUMP_HOUR, DUMP_MINUTE,
                        tzinfo=CHICAGO).astimezone(timezone.utc)
        if slot <= now - DUMP_GRACE:
            return slot
    raise AssertionError("unreachable: a slot two days back is always due")


def evaluate_dump_status(rows: Iterable[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
    """Classify the nightly dump as ``ok``, ``failed`` or ``missing``.

    ``ok`` needs a successful result that started at or after the latest due
    slot. A failed result, or an attempt with no result, since that slot is
    ``failed``. Nothing since the slot is ``missing``.
    """
    slot = latest_due_dump_slot(now)
    runs = [r for r in rows if r.get("job", DUMP_JOB) == DUMP_JOB]
    ok_runs = [r["started_at"] for r in runs if r.get("phase") == "result" and r.get("ok")]
    since = [r for r in runs if r["started_at"] >= slot]
    if any(r.get("phase") == "result" and r.get("ok") for r in since):
        state = "ok"
    elif since:
        state = "failed"
    else:
        state = "missing"
    return {
        "state": state,
        "scheduled_for": slot,
        "last_ok": max(ok_runs) if ok_runs else None,
        "last_run": max((r["started_at"] for r in runs), default=None),
    }


# --- Store ------------------------------------------------------------------


def _uuid_or_none(value: str) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _row(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    out = dict(raw)
    out["memory_id"] = str(out["memory_id"])
    if "supersedes" in out:
        out["supersedes"] = sorted(str(s) for s in out["supersedes"] or [])
    if "issue_ids" in out:
        out["issue_ids"] = list(out["issue_ids"] or [])
    return out


def _encode_cursor(created_at: datetime, memory_id: str, fingerprint: str) -> str:
    payload = json.dumps({"c": created_at.isoformat(), "i": memory_id, "f": fingerprint})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str, fingerprint: str) -> Tuple[datetime, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        created_at = datetime.fromisoformat(data["c"])
        memory_id = str(uuid.UUID(data["i"]))
        bound = data["f"]
    except (ValueError, KeyError, TypeError) as exc:
        raise MemoryStoreError("invalid_cursor", "The cursor is malformed") from exc
    if bound != fingerprint:
        raise MemoryStoreError(
            "invalid_cursor", "The cursor belongs to a listing with other filters"
        )
    return created_at, memory_id


class MemoryStore:
    """Postgres-backed canonical memory store with a small connection pool."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4,
                 timeout: float = 10.0) -> None:
        """Open a pool on ``dsn``; ``timeout`` bounds the wait for a free connection."""
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"row_factory": dict_row},
            open=True,
            name="sessionflow-memory",
        )
        # Test seam: called as hook(stage, conn) with stage "before_commit".
        self._fault_hook: Optional[Callable[[str, psycopg.Connection], None]] = None

    def close(self) -> None:
        """Close every pooled connection."""
        self._pool.close()

    # --- Migrations ---------------------------------------------------------

    def migrate(self) -> List[str]:
        """Apply every unapplied ``sql/memory/*.sql`` file in name order.

        Each file and its ``schema_migrations`` row commit together. An
        advisory lock keeps two servers from migrating at once. Returns the
        versions applied by this call.
        """
        applied: List[str] = []
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('sessionflow-memory-migrate'))")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    " version text PRIMARY KEY,"
                    " applied_at timestamptz NOT NULL DEFAULT now())"
                )
                done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
                for path in files:
                    version = path.stem
                    if version in done:
                        continue
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                    applied.append(version)
        return applied

    # --- Reads --------------------------------------------------------------

    def get(self, memory_id: str, expect_project: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the live record, or None when it is missing or deleted.

        Raises ``project_mismatch`` when ``expect_project`` names another project.
        """
        mid = _uuid_or_none(memory_id)
        if mid is None:
            return None
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMN_SQL} FROM memories WHERE memory_id = %s AND deleted_at IS NULL",
                (mid,),
            ).fetchone()
        row = _row(row)
        if row is not None:
            self._check_project(row, expect_project)
        return row

    def get_many(self, memory_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Return live records keyed by ID in one query; deleted and unknown IDs are absent."""
        ids = [m for m in (_uuid_or_none(x) for x in memory_ids) if m is not None]
        if not ids:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMN_SQL} FROM memories "
                "WHERE memory_id = ANY(%s) AND deleted_at IS NULL",
                (ids,),
            ).fetchall()
        return {r["memory_id"]: r for r in map(_row, rows)}

    def versions(self, memory_id: str) -> List[Dict[str, Any]]:
        """Return every stored version of a record, oldest first, tombstone included."""
        mid = _uuid_or_none(memory_id)
        if mid is None:
            return []
        cols = ", ".join(c for c in _COLUMNS) + ", written_at"
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {cols} FROM memory_versions WHERE memory_id = %s ORDER BY version",
                (mid,),
            ).fetchall()
        return [_row(r) for r in rows]

    def list(self, filters: MemoryFilters, limit: int = 20,
             cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Return one page newest first, ties by ``memory_id`` descending, and the next cursor.

        The cursor is keyset on ``(created_at, memory_id)`` and bound to the
        filters; reusing it with other filters raises ``invalid_cursor``.
        """
        limit = max(1, int(limit))
        fingerprint = filters.fingerprint()
        where, params = self._filter_sql(filters)
        if cursor:
            after_created, after_id = _decode_cursor(cursor, fingerprint)
            where.append("(created_at, memory_id) < (%s, %s)")
            params.extend([after_created, uuid.UUID(after_id)])
        sql = (
            f"SELECT {_COLUMN_SQL} FROM memories WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC, memory_id DESC LIMIT %s"
        )
        params.append(limit + 1)
        with self._pool.connection() as conn:
            rows = [_row(r) for r in conn.execute(sql, params).fetchall()]
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1]
            next_cursor = _encode_cursor(last["created_at"], last["memory_id"], fingerprint)
        return rows, next_cursor

    def count(self, filters: MemoryFilters) -> int:
        """Return how many records match ``filters``."""
        where, params = self._filter_sql(filters)
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT count(*) AS n FROM memories WHERE {' AND '.join(where)}", params
            ).fetchone()
        return int(row["n"])

    def find_by_mem0_id(self, mem0_id: str) -> Optional[Dict[str, Any]]:
        """Return the record whose ``extra.mem0_ids`` holds ``mem0_id``, tombstones included."""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMN_SQL} FROM memories "
                "WHERE (extra -> 'mem0_ids') @> %s ORDER BY deleted_at NULLS FIRST LIMIT 1",
                (Jsonb([mem0_id]),),
            ).fetchone()
        return _row(row)

    def _filter_sql(self, f: MemoryFilters) -> Tuple[List[str], List[Any]]:
        where = ["deleted_at IS NULL", "user_id = %s"]
        params: List[Any] = [f.user_id]
        if f.project != "*":
            where.append("project = %s")
            params.append(f.project)
        if f.status is not None:
            where.append("status = %s")
            params.append(f.status)
        if f.type:
            where.append("type = %s")
            params.append(f.type)
        if f.agent_id:
            where.append("agent_id = %s")
            params.append(f.agent_id)
        if f.issue_id:
            where.append("issue_ids @> %s")
            params.append([f.issue_id.strip().upper()])
        if f.source:
            where.append("extra ->> 'source' = %s")
            params.append(f.source)
        if f.category:
            where.append("extra ->> 'category' = %s")
            params.append(f.category)
        if f.source_session_id:
            where.append("source_session_id = %s")
            params.append(f.source_session_id)
        if f.created_from:
            where.append("created_at >= %s")
            params.append(chicago_day_start(f.created_from))
        if f.created_to:
            where.append("created_at < %s")
            params.append(chicago_day_start(f.created_to + timedelta(days=1)))
        return where, params

    @staticmethod
    def _check_project(row: Dict[str, Any], expect_project: Optional[str]) -> None:
        if expect_project and expect_project != "*" and row["project"] != expect_project:
            raise MemoryStoreError(
                "project_mismatch",
                f"Memory {row['memory_id']} belongs to project {row['project']!r}, "
                f"not {expect_project!r}",
                memory_id=row["memory_id"], project=row["project"],
            )

    # --- Writes -------------------------------------------------------------

    def add(self, rec: NewMemory) -> WriteResult:
        """Add a record, or resolve it against an existing one.

        A topic-key match replaces that record in place; a content-hash match
        returns the existing record; both on different records raise
        ``MemoryConflict``. Supersede targets are marked in the same transaction.
        """
        topic = rec.topic_key or None
        return self._with_retries(lambda conn: self._add_tx(conn, rec, topic))

    def update(self, memory_id: str, changes: Dict[str, Any],
               expect_project: Optional[str] = None) -> WriteResult:
        """Change fields of a live record; omitted fields keep their values."""
        unknown = set(changes) - _UPDATABLE
        if unknown:
            raise MemoryStoreError(
                "invalid_field", f"Cannot update {', '.join(sorted(unknown))}",
                fields=sorted(unknown),
            )
        changes = dict(changes)
        if "text" in changes and "content_hash" not in changes:
            changes["content_hash"] = content_hash(changes["text"])
        if "topic_key" in changes:
            changes["topic_key"] = changes["topic_key"] or None
        return self._with_retries(
            lambda conn: self._update_tx(conn, memory_id, changes, expect_project)
        )

    def delete(self, memory_id: str, expect_project: Optional[str] = None) -> WriteResult:
        """Delete a record as a tombstone with a final version row."""
        return self._with_retries(lambda conn: self._delete_tx(conn, memory_id, expect_project))

    def _with_retries(self, body: Callable[[psycopg.Connection], WriteResult]) -> WriteResult:
        last: Optional[pg_errors.UniqueViolation] = None
        for _ in range(MAX_WRITE_RETRIES + 1):
            try:
                with self._pool.connection() as conn:
                    with conn.transaction():
                        result = body(conn)
                        if self._fault_hook is not None:
                            self._fault_hook("before_commit", conn)
                    return result
            except pg_errors.UniqueViolation as exc:
                last = exc
        raise MemoryConflict([], "The write lost every retry to concurrent writers") from last

    def _lock(self, conn: psycopg.Connection, where: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
        rows = conn.execute(
            f"SELECT {_COLUMN_SQL} FROM memories WHERE {where} ORDER BY memory_id FOR UPDATE",
            params,
        ).fetchall()
        return [_row(r) for r in rows]

    def _add_tx(self, conn: psycopg.Connection, rec: NewMemory, topic: Optional[str]) -> WriteResult:
        targets = sorted({str(t) for t in rec.supersedes})
        target_uuids = [u for u in (_uuid_or_none(t) for t in targets) if u is not None]
        if len(target_uuids) != len(targets):
            raise InvalidSupersede(targets)
        locked = self._lock(
            conn,
            "memory_id = ANY(%s) OR (user_id = %s AND project = %s AND deleted_at IS NULL "
            "AND (content_hash = %s OR (%s::text IS NOT NULL AND topic_key = %s)))",
            (target_uuids, rec.user_id, rec.project, rec.content_hash, topic, topic),
        )
        by_id = {r["memory_id"]: r for r in locked}
        bad = [
            t for t in targets
            if t not in by_id or by_id[t]["deleted_at"] is not None
            or by_id[t]["user_id"] != rec.user_id or by_id[t]["project"] != rec.project
        ]
        if bad:
            raise InvalidSupersede(bad)
        scoped = [
            r for r in locked
            if r["deleted_at"] is None and r["user_id"] == rec.user_id
            and r["project"] == rec.project
        ]
        topic_row = next((r for r in scoped if topic is not None and r["topic_key"] == topic), None)
        hash_row = next((r for r in scoped if r["content_hash"] == rec.content_hash), None)

        if topic_row and hash_row and topic_row["memory_id"] != hash_row["memory_id"]:
            raise MemoryConflict([topic_row["memory_id"], hash_row["memory_id"]])

        if topic_row is not None and topic_row["content_hash"] != rec.content_hash:
            changes = {
                "text": rec.text, "content_hash": rec.content_hash, "agent_id": rec.agent_id,
                "type": rec.type, "status": rec.status, "extra": rec.extra,
                "issue_ids": rec.issue_ids, "source_session_id": rec.source_session_id,
                "source_provider": rec.source_provider,
                "source_project_root": rec.source_project_root,
                "source_started_at": rec.source_started_at,
                "source_ended_at": rec.source_ended_at,
            }
            if targets:
                changes["supersedes"] = targets
            if rec.updated_at is not None:
                changes["updated_at"] = rec.updated_at
            result = self._apply_change(conn, topic_row, changes, "replaced")
            self._supersede(conn, [by_id[t] for t in targets if t != topic_row["memory_id"]])
            return result

        existing = topic_row or hash_row
        if existing is not None:
            return WriteResult(existing["memory_id"], "deduplicated", existing["version"],
                               existing["created_at"], existing["updated_at"],
                               existing["change_seq"])

        now = datetime.now(timezone.utc)
        created = rec.created_at or now
        updated = rec.updated_at or created
        memory_id = uuid.uuid4()
        row = conn.execute(
            "INSERT INTO memories (memory_id, user_id, agent_id, project, type, status, "
            "topic_key, content_hash, text, extra, issue_ids, supersedes, source_session_id, "
            "source_provider, source_project_root, source_started_at, source_ended_at, "
            "created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s::uuid[], %s, %s, %s, %s, %s, %s, %s) RETURNING version, change_seq",
            (memory_id, rec.user_id, rec.agent_id, rec.project, rec.type, rec.status, topic,
             rec.content_hash, rec.text, Jsonb(rec.extra or {}), list(rec.issue_ids),
             targets, rec.source_session_id, rec.source_provider, rec.source_project_root,
             rec.source_started_at, rec.source_ended_at, created, updated),
        ).fetchone()
        self._snapshot(conn, memory_id)
        self._supersede(conn, [by_id[t] for t in targets])
        return WriteResult(str(memory_id), "created", row["version"], created, updated,
                           row["change_seq"])

    def _update_tx(self, conn: psycopg.Connection, memory_id: str, changes: Dict[str, Any],
                   expect_project: Optional[str]) -> WriteResult:
        mid = _uuid_or_none(memory_id)
        if mid is None:
            raise MemoryNotFound(memory_id)
        current = conn.execute(
            "SELECT user_id, project FROM memories WHERE memory_id = %s AND deleted_at IS NULL",
            (mid,),
        ).fetchone()
        if current is None:
            raise MemoryNotFound(memory_id)
        new_hash = changes.get("content_hash")
        new_topic = changes.get("topic_key")
        locked = self._lock(
            conn,
            "memory_id = %s OR (user_id = %s AND project = %s AND deleted_at IS NULL AND "
            "(content_hash = %s OR (%s::text IS NOT NULL AND topic_key = %s)))",
            (mid, current["user_id"], current["project"], new_hash, new_topic, new_topic),
        )
        target = next((r for r in locked if r["memory_id"] == str(mid)), None)
        if target is None or target["deleted_at"] is not None:
            raise MemoryNotFound(memory_id)
        self._check_project(target, expect_project)
        others = [
            r["memory_id"] for r in locked
            if r["memory_id"] != target["memory_id"] and r["deleted_at"] is None
            and r["user_id"] == target["user_id"] and r["project"] == target["project"]
        ]
        if others:
            raise MemoryConflict([target["memory_id"], *others])
        return self._apply_change(conn, target, changes, "updated")

    def _delete_tx(self, conn: psycopg.Connection, memory_id: str,
                   expect_project: Optional[str]) -> WriteResult:
        mid = _uuid_or_none(memory_id)
        if mid is None:
            raise MemoryNotFound(memory_id)
        locked = self._lock(conn, "memory_id = %s", (mid,))
        if not locked or locked[0]["deleted_at"] is not None:
            raise MemoryNotFound(memory_id)
        target = locked[0]
        self._check_project(target, expect_project)
        now = self._next_time(target)
        row = conn.execute(
            "UPDATE memories SET deleted_at = %s, updated_at = %s, version = version + 1, "
            "change_seq = nextval('memory_change_seq') WHERE memory_id = %s "
            "RETURNING version, change_seq, created_at, updated_at",
            (now, now, mid),
        ).fetchone()
        self._snapshot(conn, mid)
        self._reset_acks(conn, [mid])
        return WriteResult(str(mid), "deleted", row["version"], row["created_at"],
                           row["updated_at"], row["change_seq"])

    def _apply_change(self, conn: psycopg.Connection, target: Dict[str, Any],
                      changes: Dict[str, Any], outcome: str) -> WriteResult:
        effective = {
            k: v for k, v in changes.items()
            if k != "updated_at" and _normalize(k, v) != _normalize(k, target.get(k))
        }
        if not effective:
            return WriteResult(target["memory_id"], "unchanged", target["version"],
                               target["created_at"], target["updated_at"], target["change_seq"])
        updated = changes.get("updated_at") or self._next_time(target)
        assignments, params = [], []
        for key, value in effective.items():
            if key == "extra":
                value = Jsonb(value or {})
            elif key == "supersedes":
                assignments.append("supersedes = %s::uuid[]")
                params.append(sorted(str(v) for v in value))
                continue
            elif key == "issue_ids":
                value = list(value or [])
            assignments.append(f"{key} = %s")
            params.append(value)
        assignments += ["updated_at = %s", "version = version + 1",
                        "change_seq = nextval('memory_change_seq')"]
        params += [updated, uuid.UUID(target["memory_id"])]
        row = conn.execute(
            f"UPDATE memories SET {', '.join(assignments)} WHERE memory_id = %s "
            "RETURNING version, change_seq, created_at, updated_at",
            params,
        ).fetchone()
        mid = uuid.UUID(target["memory_id"])
        self._snapshot(conn, mid)
        self._reset_acks(conn, [mid])
        return WriteResult(target["memory_id"], outcome, row["version"], row["created_at"],
                           row["updated_at"], row["change_seq"])

    def _supersede(self, conn: psycopg.Connection, targets: List[Dict[str, Any]]) -> None:
        for target in targets:
            if target["status"] == "superseded":
                continue
            self._apply_change(conn, target, {"status": "superseded"}, "superseded")

    @staticmethod
    def _next_time(target: Dict[str, Any]) -> datetime:
        # updated_at must move forward even when two writes share a clock tick.
        now = datetime.now(timezone.utc)
        floor = target["updated_at"] + timedelta(microseconds=1)
        return now if now >= floor else floor

    @staticmethod
    def _snapshot(conn: psycopg.Connection, memory_id: uuid.UUID) -> None:
        conn.execute(
            f"INSERT INTO memory_versions ({_COLUMN_SQL}) "
            f"SELECT {_COLUMN_SQL} FROM memories WHERE memory_id = %s",
            (memory_id,),
        )

    @staticmethod
    def _reset_acks(conn: psycopg.Connection, memory_ids: List[uuid.UUID]) -> None:
        conn.execute("DELETE FROM memory_index_acks WHERE memory_id = ANY(%s)", (memory_ids,))

    # --- Acknowledgements ---------------------------------------------

    def ack(self, consumer: str, memory_id: str, *, version: int, generation: int,
            deleted: bool) -> bool:
        """Record that ``consumer`` holds ``version`` at ``generation``.

        Returns False, changing nothing, when the stored acknowledgement is
        already at that version and generation or later, so a late ack can
        never move it backwards.
        """
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO memory_index_acks AS a (consumer, memory_id, version, generation, deleted) "
                "SELECT %s, m.memory_id, %s, %s, %s FROM memories m WHERE m.memory_id = %s "
                "ON CONFLICT (consumer, memory_id) DO UPDATE SET version = EXCLUDED.version, "
                "generation = EXCLUDED.generation, deleted = EXCLUDED.deleted, acked_at = now() "
                "WHERE (a.version, a.generation) < (EXCLUDED.version, EXCLUDED.generation) "
                "RETURNING a.version",
                (consumer, version, generation, deleted, uuid.UUID(str(memory_id))),
            ).fetchone()
        return row is not None

    def is_acked(self, consumer: str, memory_id: str, *, version: int, generation: int) -> bool:
        """Return True when ``consumer`` already holds ``version`` at ``generation`` or later."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_index_acks WHERE consumer = %s AND memory_id = %s "
                "AND (version, generation) >= (%s, %s)",
                (consumer, uuid.UUID(str(memory_id)), version, generation),
            ).fetchone()
        return row is not None

    def reset_acks(self, memory_ids: Optional[Sequence[str]] = None,
                   consumer: Optional[str] = None) -> int:
        """Delete acknowledgements so the records become pending again; returns the count.

        With no ``memory_ids`` every record is reset; with no ``consumer`` every consumer.
        """
        where, params = [], []
        if memory_ids is not None:
            where.append("memory_id = ANY(%s)")
            params.append([uuid.UUID(str(m)) for m in memory_ids])
        if consumer is not None:
            where.append("consumer = %s")
            params.append(consumer)
        sql = "DELETE FROM memory_index_acks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._pool.connection() as conn:
            return conn.execute(sql, params).rowcount

    def pending(self, consumer: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return records whose ``consumer`` acknowledgement is missing or behind, by ``change_seq``.

        The ``milvus`` consumer is also behind when its generation is older than
        the current one; FTS consumers hold no vectors and ignore generations.
        """
        sql = (
            "SELECT m.memory_id, m.version, meta.generation, (m.deleted_at IS NOT NULL) AS deleted "
            "FROM memories m CROSS JOIN memory_index_meta meta "
            "LEFT JOIN memory_index_acks a ON a.consumer = %s AND a.memory_id = m.memory_id "
            "WHERE a.memory_id IS NULL OR a.version < m.version "
            "OR (%s AND a.generation < meta.generation) "
            "ORDER BY m.change_seq"
        )
        params: List[Any] = [consumer, consumer == "milvus"]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row(r) for r in rows]

    def canonical(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Return the current row, tombstone included, with the current ``generation``."""
        mid = _uuid_or_none(memory_id)
        if mid is None:
            return None
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMN_SQL}, meta.generation FROM memories "
                "CROSS JOIN memory_index_meta meta WHERE memory_id = %s",
                (mid,),
            ).fetchone()
        return _row(row)

    @contextmanager
    def locked_canonical(self, memory_id: str) -> Iterator[Optional[Dict[str, Any]]]:
        """Yield the current row under a transaction-scoped advisory lock on its ID.

        Index workers on every machine take this lock, so two workers never
        interleave a read and a Milvus write for the same record. The row also
        carries the memory model (``index_model_name``, ``index_vector_dim``)
        read in the same transaction.
        """
        mid = uuid.UUID(str(memory_id))
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                             (str(mid),))
                row = conn.execute(
                    f"SELECT {_COLUMN_SQL}, meta.generation, "
                    "meta.model_name AS index_model_name, meta.vector_dim AS index_vector_dim "
                    "FROM memories CROSS JOIN memory_index_meta meta WHERE memory_id = %s",
                    (mid,),
                ).fetchone()
                yield _row(row)

    def index_state(self) -> List[Dict[str, Any]]:
        """Return every record's ID, version, hash and tombstone flag in ``change_seq`` order."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT memory_id, version, content_hash, (deleted_at IS NOT NULL) AS deleted, "
                "meta.generation FROM memories CROSS JOIN memory_index_meta meta "
                "ORDER BY change_seq"
            ).fetchall()
        return [_row(r) for r in rows]

    def audit_rows(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """Return ID, project, type, status and created_at of every live record, for the audit."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT memory_id, project, type, status, created_at FROM memories "
                "WHERE user_id = %s AND deleted_at IS NULL ORDER BY project, created_at",
                (user_id,),
            ).fetchall()
        return [_row(r) for r in rows]

    def vectors_pending(self, project: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Return whether any active record in the project lacks a current-generation vector."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM memories m CROSS JOIN memory_index_meta meta "
                "LEFT JOIN memory_index_acks a ON a.consumer = 'milvus' AND a.memory_id = m.memory_id "
                "WHERE m.user_id = %s AND m.project = %s AND m.deleted_at IS NULL "
                "AND m.status = 'active' AND (a.memory_id IS NULL OR a.version < m.version "
                "OR a.generation < meta.generation)) AS pending",
                (user_id, project),
            ).fetchone()
        return bool(row["pending"])

    def ack_lag(self) -> Dict[str, int]:
        """Return how many records each known consumer still has to apply."""
        with self._pool.connection() as conn:
            consumers = [r["consumer"] for r in conn.execute(
                "SELECT DISTINCT consumer FROM memory_index_acks")]
        return {c: len(self.pending(c)) for c in sorted(set(consumers) | {"milvus"})}

    # --- Embedding generation -----------------------------------------

    def index_meta(self) -> Dict[str, Any]:
        """Return the memory model name, vector dimension and generation."""
        with self._pool.connection() as conn:
            return dict(conn.execute(
                "SELECT model_name, vector_dim, generation FROM memory_index_meta"
            ).fetchone())

    def claim_index_model(self, model_name: str, vector_dim: int) -> Dict[str, Any]:
        """Record the memory model when none is set; return the stored meta either way."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE memory_index_meta SET model_name = %s, vector_dim = %s "
                "WHERE model_name IS NULL",
                (model_name, vector_dim),
            )
        return self.index_meta()

    def bump_generation(self, model_name: Optional[str] = None,
                        vector_dim: Optional[int] = None) -> int:
        """Start a new embedding generation, optionally for a new model; returns it."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE memory_index_meta SET generation = generation + 1, "
                "model_name = COALESCE(%s, model_name), vector_dim = COALESCE(%s, vector_dim) "
                "RETURNING generation",
                (model_name, vector_dim),
            ).fetchone()
        return int(row["generation"])

    # --- Maintenance --------------------------------------------------------

    def dump_status(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Evaluate the nightly dump from ``maintenance_runs``."""
        now = now or datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT job, phase, started_at, ok FROM maintenance_runs WHERE job = %s "
                "AND started_at >= %s ORDER BY started_at DESC",
                (DUMP_JOB, now - timedelta(days=45)),
            ).fetchall()
        return evaluate_dump_status(rows, now)


def _normalize(key: str, value: Any) -> Any:
    if key in ("issue_ids", "supersedes"):
        return sorted(str(v) for v in value or [])
    if key == "extra":
        return json.dumps(value or {}, sort_keys=True, default=str)
    return value
