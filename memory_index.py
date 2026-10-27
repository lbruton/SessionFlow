"""Memory indexes: the ``memories`` Milvus collection and the local FTS file.

Postgres is the canonical store. Everything here is derived from it and can be
rebuilt, so an index write that fails after a commit leaves the record pending
for the sync loop instead of failing the write. Each machine is two consumers
in ``memory_index_acks``: the shared ``milvus`` collection and its own
``fts:<host>`` file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import embedding_control
import rag_engine
from memory_store import MemoryStore, run_db

logger = logging.getLogger("sessionflow.memory_index")

MILVUS_CONSUMER = "milvus"
# Records applied per consumer per tick; the rest wait for the next tick.
TICK_BATCH = 256
# A record that changes between embedding and the locked write is re-embedded
# this many times before it is left pending for the next tick.
MAX_EMBED_RACES = 3
# Upper bound for one Milvus query of every memory row (reconcile and rebuild).
MILVUS_SCAN_LIMIT = 16384

_FTS_COLUMNS = ("memory_id", "version", "content_hash", "user_id", "agent_id", "project",
                "type", "status", "issue_ids", "source", "category", "created_at")


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


def _wrapped_ids(ids: List[str]) -> str:
    return f",{','.join(ids)}," if ids else ""


def _extra(row: Dict[str, Any]) -> Dict[str, Any]:
    extra = row.get("extra") or {}
    return extra if isinstance(extra, dict) else json.loads(extra)


async def embed_for_index(texts: List[str]) -> List[List[float]]:
    """Embed memory texts as documents on the ``mlx-embed`` thread.

    Waits on the shared throttle instead of the transcript budget, which drops
    batches. ``rag_engine.embed_texts`` routes the MLX work to ``mlx-embed``.
    """
    throttle = embedding_control.get_embed_throttle()
    await throttle.wait()
    try:
        loop = asyncio.get_running_loop()
        executor = rag_engine._embed_executor
        if executor is not None:
            return await loop.run_in_executor(executor, rag_engine.embed_texts, texts)
        return await asyncio.to_thread(rag_engine.embed_texts, texts)
    finally:
        throttle.done()


class MemoryFTS:
    """Keyword index of memory text in a local SQLite FTS5 file.

    A watermark table keeps the highest version applied per record, tombstones
    included, and every change commits with its watermark in one transaction,
    so a delayed older write can never bring back a deleted or newer record.
    """

    def __init__(self, path: Path) -> None:
        """Open or create the FTS file; ``recreated`` is True when it had to be built."""
        self.path = Path(path)
        self.recreated = self._open()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _create(self, conn: sqlite3.Connection) -> None:
        cols = ", ".join(f"{c} UNINDEXED" for c in _FTS_COLUMNS)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, {cols}, "
            "tokenize='unicode61')"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories_fts_watermark ("
            " memory_id TEXT PRIMARY KEY, version INTEGER NOT NULL, deleted INTEGER NOT NULL)"
        )

    def _open(self) -> bool:
        existed = self.path.exists()
        if existed:
            try:
                with closing(self._connect()) as conn:
                    if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise sqlite3.DatabaseError("quick_check failed")
                    self._create(conn)
                    conn.execute("SELECT count(*) FROM memories_fts").fetchone()
                return False
            except sqlite3.DatabaseError as exc:
                logger.warning("Memory FTS file %s is damaged (%s); rebuilding", self.path, exc)
                self._remove_files()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._create(conn)
        return True

    def _remove_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(f"{self.path}{suffix}")
            except FileNotFoundError:
                pass

    def reset(self) -> None:
        """Delete every row and watermark."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM memories_fts")
            conn.execute("DELETE FROM memories_fts_watermark")
            conn.execute("COMMIT")

    def apply(self, row: Dict[str, Any]) -> bool:
        """Apply one canonical row; return False when the watermark already holds this version or later."""
        memory_id = str(row["memory_id"])
        version = int(row["version"])
        deleted = row.get("deleted_at") is not None
        extra = _extra(row)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                mark = conn.execute(
                    "SELECT version FROM memories_fts_watermark WHERE memory_id = ?", (memory_id,)
                ).fetchone()
                if mark is not None and mark[0] >= version:
                    conn.execute("ROLLBACK")
                    return False
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
                if not deleted:
                    values = {
                        "memory_id": memory_id,
                        "version": version,
                        "content_hash": row.get("content_hash") or "",
                        "user_id": row["user_id"],
                        "agent_id": row["agent_id"],
                        "project": row["project"],
                        "type": row["type"],
                        "status": row["status"],
                        "issue_ids": _wrapped_ids(list(row.get("issue_ids") or [])),
                        "source": str(extra.get("source") or ""),
                        "category": str(extra.get("category") or ""),
                        "created_at": _iso(row.get("created_at")),
                    }
                    placeholders = ", ".join("?" for _ in range(len(_FTS_COLUMNS) + 1))
                    conn.execute(
                        f"INSERT INTO memories_fts (text, {', '.join(_FTS_COLUMNS)}) "
                        f"VALUES ({placeholders})",
                        [row["text"], *(values[c] for c in _FTS_COLUMNS)],
                    )
                conn.execute(
                    "INSERT INTO memories_fts_watermark (memory_id, version, deleted) "
                    "VALUES (?, ?, ?) ON CONFLICT (memory_id) DO UPDATE SET "
                    "version = excluded.version, deleted = excluded.deleted",
                    (memory_id, version, int(deleted)),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return True

    def watermark(self, memory_id: str) -> Optional[int]:
        """Return the highest version applied for ``memory_id``, or None."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT version FROM memories_fts_watermark WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return None if row is None else int(row[0])

    def forget(self, memory_ids: List[str]) -> None:
        """Remove rows and watermarks so the records are applied again from scratch."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for mid in memory_ids:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))
                conn.execute("DELETE FROM memories_fts_watermark WHERE memory_id = ?", (mid,))
            conn.execute("COMMIT")

    def rows(self) -> Dict[str, Dict[str, Any]]:
        """Return the indexed version and content hash of every row, keyed by ID."""
        with closing(self._connect()) as conn:
            return {
                r[0]: {"version": int(r[1]), "content_hash": r[2]}
                for r in conn.execute("SELECT memory_id, version, content_hash FROM memories_fts")
            }

    def search(self, query: str, *, project: str, limit: int,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return BM25-ranked hits as ``memory_id`` and ``version``.

        Each whitespace-separated term is quoted, so punctuation such as an
        issue ID's hyphen never breaks the FTS5 query syntax.
        """
        terms = [t for t in query.split() if t]
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        where, params = ["memories_fts MATCH ?"], [match]
        if project != "*":
            where.append("project = ?")
            params.append(project)
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if key == "issue_id":
                where.append("issue_ids LIKE ?")
                params.append(f"%,{str(value).upper()},%")
            elif key in _FTS_COLUMNS:
                where.append(f"{key} = ?")
                params.append(value)
        params.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT memory_id, version FROM memories_fts WHERE {' AND '.join(where)} "
                "ORDER BY bm25(memories_fts) LIMIT ?",
                params,
            ).fetchall()
        return [{"memory_id": r[0], "version": int(r[1])} for r in rows]


class MemoryIndex:
    """Keeps this machine's memory consumers in step with Postgres.

    ``apply`` runs right after a write; ``tick`` applies every record whose
    acknowledgement differs; ``reconcile`` compares the indexes with Postgres
    and resets what differs; ``rebuild`` replays every record.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        milvus: Any,
        fts_path: Path,
        host: str,
        embed: Callable[[List[str]], Awaitable[List[List[float]]]] = embed_for_index,
        model_name: Optional[str] = None,
        milvus_uri: str = "",
    ) -> None:
        """Bind the store, the Milvus client and this machine's FTS file."""
        self.store = store
        self.milvus = milvus
        self.fts_path = Path(fts_path)
        self.host = host
        self.embed = embed
        self.model_name = model_name or rag_engine.get_model_name()
        self.milvus_uri = milvus_uri
        self.spec = rag_engine.MEMORIES_SPEC
        self.fts: Optional[MemoryFTS] = None
        self.milvus_blocked: Optional[str] = None
        self.last_tick: Optional[datetime] = None

    @property
    def fts_consumer(self) -> str:
        """Return this machine's FTS consumer name."""
        return f"fts:{self.host}"

    async def start(self) -> None:
        """Ensure the collection, check the memory model, and rebuild a missing or damaged FTS file."""
        await run_db(rag_engine._ensure_collection, self.milvus, self.milvus_uri, spec=self.spec)
        meta = await run_db(self.store.claim_index_model, self.model_name, self.spec.vector_dim())
        if meta["model_name"] != self.model_name or meta["vector_dim"] != self.spec.vector_dim():
            self.milvus_blocked = (
                f"local model {self.model_name} ({self.spec.vector_dim()} dims) does not match "
                f"the memory model {meta['model_name']} ({meta['vector_dim']} dims); "
                "memory vectors are not written from this machine"
            )
            logger.warning("Memory Milvus writes refused: %s", self.milvus_blocked)
        self.fts = await run_db(MemoryFTS, self.fts_path)
        if self.fts.recreated:
            await self._rebuild_fts()

    def health(self) -> Dict[str, Any]:
        """Return the consumer names, the Milvus block reason and the last tick time."""
        return {
            "host": self.host,
            "consumers": [MILVUS_CONSUMER, self.fts_consumer],
            "milvus_blocked": self.milvus_blocked,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
        }

    # --- One record ---------------------------------------------------------

    async def apply(self, memory_id: str) -> str:
        """Bring both consumers up to date for one record; ``ok`` or ``pending``."""
        milvus_ok = await self._apply_milvus(memory_id)
        fts_ok = await self._apply_fts(memory_id)
        return "ok" if milvus_ok and fts_ok else "pending"

    async def _apply_milvus(self, memory_id: str) -> bool:
        if self.milvus_blocked:
            return False
        try:
            for _ in range(MAX_EMBED_RACES):
                row = await run_db(self.store.canonical, memory_id)
                if row is None:
                    return False
                vector = None
                if row["deleted_at"] is None:
                    vector = (await self.embed([row["text"]]))[0]
                written = await run_db(self._milvus_write, memory_id, row["content_hash"], vector)
                if written is not None:
                    await run_db(self.store.ack, MILVUS_CONSUMER, memory_id,
                                 version=written["version"], generation=written["generation"],
                                 deleted=written["deleted_at"] is not None)
                    return True
            return False
        except Exception as exc:  # noqa: BLE001 - the record stays pending for the next tick
            logger.warning("Memory Milvus apply failed for %s: %s", memory_id,
                           rag_engine._scrub_exception(exc))
            return False

    def _milvus_write(self, memory_id: str, embedded_hash: str,
                      vector: Optional[List[float]]) -> Optional[Dict[str, Any]]:
        """Write the canonical row to Milvus under the record lock.

        Returns the row written, or None when the text changed after it was
        embedded, so the caller embeds again.
        """
        pk = rag_engine._pk_from_doc_id(memory_id)
        with self.store.locked_canonical(memory_id) as row:
            if row is None:
                return None
            if row["deleted_at"] is not None:
                self.milvus.delete(collection_name=self.spec.name, ids=[pk])
                return row
            if vector is None or row["content_hash"] != embedded_hash:
                return None
            extra = _extra(row)
            self.milvus.upsert(collection_name=self.spec.name, data=[{
                "id": pk,
                "memory_id": row["memory_id"],
                "vector": vector,
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "project": row["project"],
                "type": row["type"],
                "status": row["status"],
                "issue_ids": rag_engine._truncate_utf8(
                    _wrapped_ids(row["issue_ids"]), rag_engine._ISSUE_ID_FIELD_MAX),
                "source": rag_engine._truncate_utf8(str(extra.get("source") or ""), 128),
                "category": rag_engine._truncate_utf8(str(extra.get("category") or ""), 128),
                "created_at": _iso(row["created_at"]),
                "version": row["version"],
                "generation": row["generation"],
            }])
            return row

    async def _apply_fts(self, memory_id: str) -> bool:
        try:
            row = await run_db(self.store.canonical, memory_id)
            if row is None:
                return False
            applied = await run_db(self.fts.apply, row)
            if not applied:
                mark = await run_db(self.fts.watermark, memory_id)
                if mark is None or mark < row["version"]:
                    return False
            await run_db(self.store.ack, self.fts_consumer, memory_id,
                         version=row["version"], generation=row["generation"],
                         deleted=row["deleted_at"] is not None)
            return True
        except Exception as exc:  # noqa: BLE001 - the record stays pending for the next tick
            logger.warning("Memory FTS apply failed for %s: %s", memory_id,
                           rag_engine._scrub_exception(exc))
            return False

    # --- Loop, reconcile and rebuild -----------------------------------------

    async def tick(self) -> Dict[str, int]:
        """Apply every record whose acknowledgement differs, up to ``TICK_BATCH`` per consumer."""
        done = {MILVUS_CONSUMER: 0, self.fts_consumer: 0}
        if not self.milvus_blocked:
            for row in await run_db(self.store.pending, MILVUS_CONSUMER, TICK_BATCH):
                done[MILVUS_CONSUMER] += int(await self._apply_milvus(row["memory_id"]))
        for row in await run_db(self.store.pending, self.fts_consumer, TICK_BATCH):
            done[self.fts_consumer] += int(await self._apply_fts(row["memory_id"]))
        self.last_tick = datetime.now(timezone.utc)
        return done

    def _milvus_rows(self) -> Dict[str, Dict[str, Any]]:
        rows = self.milvus.query(
            collection_name=self.spec.name, filter='memory_id != ""',
            output_fields=["id", "memory_id", "version", "generation"], limit=MILVUS_SCAN_LIMIT,
        )
        if len(rows) >= MILVUS_SCAN_LIMIT:
            logger.warning("Memory Milvus scan hit its %d-row limit", MILVUS_SCAN_LIMIT)
        return {r["memory_id"]: r for r in rows}

    async def reconcile(self) -> int:
        """Reset the acknowledgement of every record whose index copy differs; return how many."""
        state = await run_db(self.store.index_state)
        canonical = {r["memory_id"]: r for r in state}
        reset = 0

        if not self.milvus_blocked:
            milvus_rows = await run_db(self._milvus_rows)
            stale = []
            for mid, row in canonical.items():
                held = milvus_rows.get(mid)
                if row["deleted"]:
                    if held is not None:
                        stale.append(mid)
                elif held is None or held["version"] != row["version"] \
                        or held["generation"] != row["generation"]:
                    stale.append(mid)
            strays = [r["id"] for mid, r in milvus_rows.items() if mid not in canonical]
            if strays:
                await run_db(self.milvus.delete, collection_name=self.spec.name, ids=strays)
            if stale:
                reset += await run_db(self.store.reset_acks, stale, MILVUS_CONSUMER)

        fts_rows = await run_db(self.fts.rows)
        stale = []
        for mid, row in canonical.items():
            held = fts_rows.get(mid)
            if row["deleted"]:
                if held is not None:
                    stale.append(mid)
            elif held is None or held["version"] != row["version"] \
                    or held["content_hash"] != row["content_hash"]:
                stale.append(mid)
        strays = [mid for mid in fts_rows if mid not in canonical]
        if stale or strays:
            await run_db(self.fts.forget, stale + strays)
        if stale:
            reset += await run_db(self.store.reset_acks, stale, self.fts_consumer)
        return reset

    async def _rebuild_fts(self) -> None:
        await run_db(self.store.reset_acks, None, self.fts_consumer)
        await run_db(self.fts.reset)
        for row in await run_db(self.store.index_state):
            await self._apply_fts(row["memory_id"])

    async def rebuild(self) -> Dict[str, Any]:
        """Replay every record in ``change_seq`` order into both consumers, then check acknowledgements."""
        state = await run_db(self.store.index_state)
        live = {r["memory_id"] for r in state if not r["deleted"]}
        if not self.milvus_blocked:
            await run_db(self.store.reset_acks, None, MILVUS_CONSUMER)
            held = await run_db(self._milvus_rows)
            strays = [r["id"] for mid, r in held.items() if mid not in live]
            if strays:
                await run_db(self.milvus.delete, collection_name=self.spec.name, ids=strays)
        await run_db(self.store.reset_acks, None, self.fts_consumer)
        await run_db(self.fts.reset)
        for row in state:
            await self._apply_milvus(row["memory_id"])
            await self._apply_fts(row["memory_id"])
        pending = {self.fts_consumer: len(await run_db(self.store.pending, self.fts_consumer))}
        if not self.milvus_blocked:
            pending[MILVUS_CONSUMER] = len(await run_db(self.store.pending, MILVUS_CONSUMER))
        return {"records": len(state), "pending": pending}
