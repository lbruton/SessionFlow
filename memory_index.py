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


class MemoryModelMismatch(RuntimeError):
    """This machine's embedding model is not the memory model, so its vectors cannot be compared."""


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
    batch = getattr(throttle, "batch", None)
    if batch is not None:
        # _embed_after_cooldown stamps on mlx-embed; a second stamp here would
        # push back a transcript batch already waiting out the first one.
        async with batch(stamp=False):
            return await _embed_on_mlx_thread(texts)
    await throttle.wait()
    return await _embed_on_mlx_thread(texts)


def _embed_after_cooldown(texts: List[str]) -> List[List[float]]:
    # Runs on mlx-embed. A transcript batch can finish between the async wait
    # and this call, and the next one checks the stamp on this thread.
    throttle = embedding_control.get_embed_throttle()
    wait_blocking = getattr(throttle, "wait_blocking", None)
    if wait_blocking is not None:
        wait_blocking()
    try:
        return rag_engine.embed_texts(texts)
    finally:
        throttle.done()


async def _embed_on_mlx_thread(texts: List[str]) -> List[List[float]]:
    loop = asyncio.get_running_loop()
    executor = rag_engine._embed_executor
    if executor is not None:
        return await loop.run_in_executor(executor, _embed_after_cooldown, texts)
    return await asyncio.to_thread(_embed_after_cooldown, texts)


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
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            # A damaged file fails here, before any caller can close the connection.
            conn.close()
            raise
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
        spec: Optional[Any] = None,
    ) -> None:
        """Bind the store, the Milvus client and this machine's FTS file."""
        self.store = store
        self.milvus = milvus
        self.fts_path = Path(fts_path)
        self.host = host
        self.embed = embed
        self.model_name = model_name or rag_engine.get_model_name()
        self.milvus_uri = milvus_uri
        self.spec = spec or rag_engine.MEMORIES_SPEC
        self.fts: Optional[MemoryFTS] = None
        self.milvus_blocked: Optional[str] = None
        self.last_tick: Optional[datetime] = None
        self._lagging = True

    @property
    def fts_consumer(self) -> str:
        """Return this machine's FTS consumer name."""
        return f"fts:{self.host}"

    async def start(self) -> None:
        """Ensure the collection, check the memory model, and rebuild a missing or damaged FTS file."""
        await run_db(rag_engine._ensure_collection, self.milvus, self.milvus_uri, spec=self.spec)
        meta = await run_db(self.store.claim_index_model, self.model_name, self.spec.vector_dim())
        self._set_model_block(meta["model_name"], meta["vector_dim"])
        self.fts = await run_db(MemoryFTS, self.fts_path)
        if self.fts.recreated:
            await self._rebuild_fts()

    def _model_mismatch(self, model_name: Any, vector_dim: Any) -> Optional[str]:
        if model_name == self.model_name and vector_dim == self.spec.vector_dim():
            return None
        return (
            f"local model {self.model_name} ({self.spec.vector_dim()} dims) does not match "
            f"the memory model {model_name} ({vector_dim} dims); "
            "memory vectors are not written from this machine"
        )

    def _set_model_block(self, model_name: Any, vector_dim: Any) -> None:
        reason = self._model_mismatch(model_name, vector_dim)
        if reason and reason != self.milvus_blocked:
            logger.warning("Memory Milvus writes refused: %s", reason)
        self.milvus_blocked = reason

    async def _refresh_model_block(self) -> None:
        """Compare the stored memory model with this machine's and set or clear the block."""
        meta = await run_db(self.store.index_meta)
        if meta["model_name"] is None:
            meta = await run_db(self.store.claim_index_model, self.model_name,
                                self.spec.vector_dim())
        self._set_model_block(meta["model_name"], meta["vector_dim"])

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
        if milvus_ok and fts_ok:
            return "ok"
        self._lagging = True
        return "pending"

    async def _apply_milvus(self, memory_id: str) -> bool:
        if self.milvus_blocked:
            return False
        try:
            for _ in range(MAX_EMBED_RACES):
                if self.milvus_blocked:
                    return False
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
        embedded, so the caller embeds again. A vector is refused, and the
        block set, when the stored memory model is not this machine's model:
        the check reads the model under the same lock as the write.
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
            self._set_model_block(row["index_model_name"], row["index_vector_dim"])
            if self.milvus_blocked:
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

    # --- Search channels ----------------------------------------------------

    def _milvus_filter(self, filters: Dict[str, Any]) -> str:
        parts = []
        for key in ("project", "user_id", "status", "type", "agent_id", "source", "category"):
            value = filters.get(key)
            if value is None or (key == "project" and value == "*"):
                continue
            parts.append(f'{key} == "{rag_engine._escape_filter_scalar(str(value))}"')
        if filters.get("issue_id"):
            token = rag_engine._issue_id_containment_token(str(filters["issue_id"]))
            parts.append(f'issue_ids like "%,{rag_engine._escape_filter_scalar(token)},%"')
        return " and ".join(parts)

    def _vector_search_sync(self, vector: List[float], filters: Dict[str, Any],
                            limit: int) -> List[Dict[str, Any]]:
        search_params: Dict[str, Any] = {"metric_type": self.spec.metric_type}
        if rag_engine._is_remote_uri(self.milvus_uri):
            search_params["params"] = {"ef": 128}
        results = self.milvus.search(
            collection_name=self.spec.name, data=[vector], limit=int(limit),
            filter=self._milvus_filter(filters), search_params=search_params,
            output_fields=["memory_id", "version", "generation"],
        )
        hits = []
        for hit in (results[0] if results else []):
            entity = hit["entity"]
            hits.append({"memory_id": entity["memory_id"], "version": entity["version"],
                         "generation": entity["generation"], "score": hit["distance"]})
        return hits

    async def vector_search(self, vector: List[float], *, filters: Dict[str, Any],
                            limit: int) -> List[Dict[str, Any]]:
        """Return Milvus hits (``memory_id``, ``version``, ``generation``, ``score``), best first.

        Raises ``MemoryModelMismatch`` when the stored memory model differs from
        this machine's: same-size vectors from another model would still match.
        """
        await self._refresh_model_block()
        if self.milvus_blocked:
            raise MemoryModelMismatch(self.milvus_blocked)
        return await run_db(self._vector_search_sync, vector, filters, limit)

    async def fts_search(self, query: str, *, filters: Dict[str, Any],
                         limit: int) -> List[Dict[str, Any]]:
        """Return keyword hits (``memory_id``, ``version``) from this machine's FTS file."""
        rest = {k: v for k, v in filters.items() if k != "project"}
        return await run_db(self.fts.search, query, project=filters.get("project", "*"),
                            limit=limit, filters=rest)

    async def similar(self, text: str, *, project: str, user_id: str,
                      threshold: float) -> tuple[List[Dict[str, Any]], str]:
        """List every active same-project memory at least ``threshold`` similar to ``text``.

        Each hit is checked against Postgres. The state is ``ok`` when every
        active record in the project has a current-generation vector, and
        ``pending`` otherwise, so an empty list never hides missing vectors.
        """
        state = "pending" if self.milvus_blocked or await run_db(
            self.store.vectors_pending, project, user_id) else "ok"
        if self.milvus_blocked:
            return [], state
        vector = (await self.embed([text]))[0]
        meta = await run_db(self.store.index_meta)
        hits = await run_db(lambda: rag_engine.memory_threshold_search(
            self.milvus, vector, project=project, user_id=user_id,
            generation=meta["generation"], threshold=threshold, collection=self.spec.name))
        rows = await run_db(self.store.get_many, [h["memory_id"] for h in hits])
        found = []
        for hit in hits:
            row = rows.get(hit["memory_id"])
            if row is None or row["status"] != "active" or row["project"] != project:
                continue
            found.append({"memory_id": hit["memory_id"], "score": hit["score"],
                          "text": row["text"][:200]})
        return found, state

    async def neighbours(self, memory_id: str, *, threshold: float) -> List[str]:
        """Return IDs of active same-project memories at least ``threshold`` similar to this one."""
        pk = rag_engine._pk_from_doc_id(memory_id)
        rows = await run_db(lambda: self.milvus.query(
            collection_name=self.spec.name, filter=f"id == {pk}",
            output_fields=["vector", "project", "user_id", "generation"], limit=1))
        if not rows:
            return []
        row = rows[0]
        hits = await run_db(lambda: rag_engine.memory_threshold_search(
            self.milvus, row["vector"], project=row["project"], user_id=row["user_id"],
            generation=row["generation"], threshold=threshold, collection=self.spec.name,
            exclude_memory_id=memory_id))
        return [h["memory_id"] for h in hits]

    def lagging(self) -> bool:
        """Return whether the last tick left records pending on this machine."""
        return self._lagging

    # --- Loop, reconcile and rebuild -----------------------------------------

    async def tick(self) -> Dict[str, int]:
        """Apply every record whose acknowledgement differs, up to ``TICK_BATCH`` per consumer."""
        await self._refresh_model_block()
        done = {MILVUS_CONSUMER: 0, self.fts_consumer: 0}
        seen = 0
        if not self.milvus_blocked:
            batch = await run_db(self.store.pending, MILVUS_CONSUMER, TICK_BATCH)
            seen += len(batch)
            for row in batch:
                done[MILVUS_CONSUMER] += int(await self._apply_milvus(row["memory_id"]))
        batch = await run_db(self.store.pending, self.fts_consumer, TICK_BATCH)
        seen += len(batch)
        for row in batch:
            done[self.fts_consumer] += int(await self._apply_fts(row["memory_id"]))
        # Lagging until a tick finds nothing left to apply.
        self._lagging = seen > 0
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
        milvus_error: Optional[Exception] = None
        await self._refresh_model_block()

        if not self.milvus_blocked:
            # The FTS half runs even when Milvus fails; the failure is raised after it.
            try:
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
            except Exception as exc:  # noqa: BLE001 - re-raised after the FTS half
                milvus_error = exc

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
        if milvus_error is not None:
            raise milvus_error
        return reset

    async def _rebuild_fts(self) -> None:
        await run_db(self.store.reset_acks, None, self.fts_consumer)
        await run_db(self.fts.reset)
        for row in await run_db(self.store.index_state):
            await self._apply_fts(row["memory_id"])

    async def rebuild(self) -> Dict[str, Any]:
        """Replay every record in ``change_seq`` order into both consumers, then check acknowledgements."""
        await self._refresh_model_block()
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


# --- Restore check ------------------------------------------------------------

RESTORE_COLLECTION = "memories_restore"
RESTORE_HOST = "restore"


def _compare_restore(manifest: List[Dict[str, Any]], milvus_rows: Dict[str, Dict[str, Any]],
                     fts_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    expected = {m["memory_id"]: m for m in manifest if not m.get("deleted")}
    missing, extra, changed = [], [], []
    for name, rows in (("milvus", milvus_rows), ("fts", fts_rows)):
        for mid, want in expected.items():
            held = rows.get(mid)
            if held is None:
                missing.append(f"{name}:{mid}")
            elif held["version"] != want["version"] or (
                "content_hash" in held and held["content_hash"] != want["content_hash"]
            ):
                changed.append(f"{name}:{mid}")
        extra.extend(f"{name}:{mid}" for mid in rows if mid not in expected)
    return {"ok": not (missing or extra or changed), "expected": len(expected),
            "missing": sorted(missing), "extra": sorted(extra), "changed": sorted(changed)}


async def run_restore_check(*, target_dsn: str, manifest: List[Dict[str, Any]], milvus: Any,
                            fts_dir: Path, embed: Callable[[List[str]], Awaitable[List[List[float]]]],
                            model_name: str, timeout_s: float) -> Dict[str, Any]:
    """Index a restored database into temporary targets and compare it with the dump manifest.

    The rows are indexed into the ``memories_restore`` collection and a
    temporary FTS file, never the production ones; both are removed
    afterwards. The result is ``ok`` only when every live manifest record is
    present in both with the same version and hash, with nothing extra,
    before ``timeout_s`` passes.
    """
    import dataclasses

    spec = dataclasses.replace(rag_engine.MEMORIES_SPEC, name=RESTORE_COLLECTION)
    fts_path = Path(fts_dir) / f"{RESTORE_HOST}-memories_fts.db"
    store = await run_db(MemoryStore, target_dsn)
    index = MemoryIndex(store=store, milvus=milvus, fts_path=fts_path, host=RESTORE_HOST,
                        embed=embed, model_name=model_name, spec=spec)
    try:
        await run_db(store.reset_acks)
        try:
            await asyncio.wait_for(_restore_index(index), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"indexing did not finish within {timeout_s:g}s"}
        if index.milvus_blocked:
            return {"ok": False, "error": index.milvus_blocked}
        milvus_rows = {
            mid: {"version": r["version"]} for mid, r in (await run_db(index._milvus_rows)).items()
        }
        fts_rows = await run_db(index.fts.rows)
        return _compare_restore(manifest, milvus_rows, fts_rows)
    finally:
        await run_db(_drop_quietly, milvus, RESTORE_COLLECTION)
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(f"{fts_path}{suffix}")
            except FileNotFoundError:
                pass
        await run_db(store.close)


async def _restore_index(index: MemoryIndex) -> None:
    await index.start()
    await index.rebuild()


def _drop_quietly(milvus: Any, name: str) -> None:
    try:
        if milvus.has_collection(name):
            milvus.drop_collection(name)
    except Exception as exc:  # noqa: BLE001 - cleanup must not hide the check result
        logger.warning("Could not drop %s: %s", name, rag_engine._scrub_exception(exc))
