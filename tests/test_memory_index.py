"""Memory indexes derived from Postgres: Milvus `memories` and the local memories_fts.db.

Milvus is an in-memory fake, embedding returns fixed vectors, SQLite is a real
file under tmp_path and Postgres is the memory_pg test schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading

import pytest

# A collection-time ImportError would stop the whole test session.
try:
    import memory_index
    from memory_index import MemoryFTS, MemoryIndex
    from memory_store import MemoryStore, NewMemory
except ImportError:
    memory_index = MemoryFTS = MemoryIndex = MemoryStore = NewMemory = None


class FakeMilvus:
    """Just enough of MilvusClient for the memories collection."""

    def __init__(self):
        self.collections: dict[str, dict[int, dict]] = {}
        self.fail_next_upsert = False

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, collection_name, **kwargs):
        self.collections.setdefault(collection_name, {})

    def drop_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def load_collection(self, collection_name):
        return None

    def prepare_index_params(self):
        class _P:
            def add_index(self, **kwargs):
                return None
        return _P()

    def describe_collection(self, name):
        import collection_specs
        spec = collection_specs.memories_spec(768)
        return {"fields": [{"name": n} for n in spec.field_names()]}

    def upsert(self, collection_name, data):
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("milvus down")
        rows = self.collections.setdefault(collection_name, {})
        for row in data:
            rows[row["id"]] = dict(row)
        return {"upsert_count": len(data)}

    def delete(self, collection_name, ids=None, filter=None):
        rows = self.collections.setdefault(collection_name, {})
        for pk in ids or []:
            rows.pop(pk, None)
        return {"delete_count": len(ids or [])}

    def query(self, collection_name, filter="", output_fields=None, limit=None, **kwargs):
        return [dict(r) for r in self.collections.get(collection_name, {}).values()]

    def rows(self):
        return {r["memory_id"]: r for r in self.collections.get("memories", {}).values()}


def _vec(text):
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest] * 24  # 768 floats


async def _embed(texts):
    return [_vec(t) for t in texts]


def _rec(text, **over):
    fields = dict(text=text, agent_id="claude-code", project="sessionflow", type="fact",
                  content_hash=hashlib.sha256(text.encode()).hexdigest())
    fields.update(over)
    return NewMemory(**fields)


@pytest.fixture
def store(memory_pg):
    s = MemoryStore(memory_pg)
    s.migrate()
    yield s
    s.close()


@pytest.fixture
def milvus():
    return FakeMilvus()


def _index(store, milvus, tmp_path, host="minitron", embed=_embed, model_name="embeddinggemma"):
    idx = MemoryIndex(store=store, milvus=milvus, fts_path=tmp_path / f"{host}-memories_fts.db",
                      host=host, embed=embed, model_name=model_name)
    asyncio.run(idx.start())
    return idx


def _fts_ids(path):
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memories_fts")}


# --- Separate stores (AC-1) --------------------------------------------------


def test_memories_use_their_own_collection_and_fts_file(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("own collection"))
    asyncio.run(idx.tick())

    assert set(milvus.collections) == {"memories"}
    assert written.memory_id in milvus.rows()
    assert idx.fts_path.name.endswith("memories_fts.db")
    assert not (tmp_path / "fts.db").exists()
    assert written.memory_id in _fts_ids(idx.fts_path)


# --- Repair after a post-commit failure (AC-27) ------------------------------


def test_index_failure_after_commit_is_repaired_next_tick(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("repair me"))
    milvus.fail_next_upsert = True

    state = asyncio.run(idx.apply(written.memory_id))

    assert state == "pending"
    assert [r["memory_id"] for r in store.pending("milvus")] == [written.memory_id]
    asyncio.run(idx.tick())
    assert store.pending("milvus") == [] and store.pending("fts:minitron") == []
    assert milvus.rows()[written.memory_id]["version"] == 1


# --- Ordering, tombstones and watermarks (AC-28, D-24) -----------------------


def test_fts_watermark_rejects_an_older_version_after_a_tombstone(tmp_path):
    fts = MemoryFTS(tmp_path / "memories_fts.db")
    base = {"memory_id": "m1", "text": "secret plans", "user_id": "lbruton", "agent_id": "codex",
            "project": "sessionflow", "type": "fact", "status": "active", "issue_ids": [],
            "extra": {}, "created_at": "2026-10-01T00:00:00+00:00", "deleted_at": None}
    assert fts.apply(dict(base, version=2, deleted_at="2026-10-02T00:00:00+00:00")) is True
    assert fts.apply(dict(base, version=1)) is False
    assert fts.search("secret", project="sessionflow", limit=5) == []


def test_slow_worker_writes_the_current_version(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("v1 text"))
    store.update(written.memory_id, {"text": "v2 text",
                                     "content_hash": hashlib.sha256(b"v2 text").hexdigest()})

    asyncio.run(idx.apply(written.memory_id))

    assert milvus.rows()[written.memory_id]["version"] == 2


def test_late_old_ack_leaves_the_newer_version_pending(store):
    written = store.add(_rec("first"))
    store.update(written.memory_id, {"text": "second",
                                     "content_hash": hashlib.sha256(b"second").hexdigest()})
    store.ack("milvus", written.memory_id, version=1, generation=1, deleted=False)
    assert [r["version"] for r in store.pending("milvus")] == [2]


def test_index_write_then_failed_ack_replays_safely(store, milvus, tmp_path, monkeypatch):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("replay"))
    real_ack = store.ack
    calls = {"n": 0}

    def flaky_ack(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection lost after the index write")
        return real_ack(*args, **kwargs)

    monkeypatch.setattr(store, "ack", flaky_ack)
    asyncio.run(idx.tick())
    asyncio.run(idx.tick())

    assert len(milvus.rows()) == 1 and milvus.rows()[written.memory_id]["version"] == 1
    assert store.pending("milvus") == []


def test_out_of_order_commit_is_delivered_on_the_next_tick(memory_pg, store, milvus, tmp_path):
    """A takes the lower change_seq, B commits first, another host ticks, then A commits."""
    idx_blue = _index(store, milvus, tmp_path, host="blue")
    writer_a = MemoryStore(memory_pg)
    in_txn, release = threading.Event(), threading.Event()

    def hold(stage, conn):
        if stage == "before_commit":
            in_txn.set()
            release.wait(30)

    writer_a._fault_hook = hold
    result_a = {}
    t = threading.Thread(target=lambda: result_a.update(r=writer_a.add(_rec("record a"))))
    t.start()
    assert in_txn.wait(30)

    b = store.add(_rec("record b"))
    asyncio.run(idx_blue.tick())
    assert set(milvus.rows()) == {b.memory_id}

    release.set()
    t.join(30)
    writer_a.close()
    a = result_a["r"]
    assert store.get(a.memory_id)["change_seq"] < store.get(b.memory_id)["change_seq"]

    asyncio.run(idx_blue.tick())
    assert set(milvus.rows()) == {a.memory_id, b.memory_id}
    assert a.memory_id in _fts_ids(idx_blue.fts_path)


def test_each_fts_consumer_converges_on_its_own(store, milvus, tmp_path):
    minitron = _index(store, milvus, tmp_path, host="minitron")
    blue = _index(store, milvus, tmp_path, host="blue")
    written = store.add(_rec("shared memory"))

    asyncio.run(minitron.tick())
    assert store.pending("milvus") == [] and store.pending("fts:minitron") == []
    assert [r["memory_id"] for r in store.pending("fts:blue")] == [written.memory_id]

    upserts_before = dict(milvus.rows())
    asyncio.run(blue.tick())
    assert store.pending("fts:blue") == []
    assert written.memory_id in _fts_ids(blue.fts_path)
    assert milvus.rows() == upserts_before


# --- Rebuild and startup repair (AC-29, AC-34) -------------------------------


def _snapshot(milvus, path):
    rows = {k: (v["version"], v["generation"], v["project"]) for k, v in milvus.rows().items()}
    with sqlite3.connect(path) as conn:
        fts = sorted(conn.execute("SELECT memory_id, version, text FROM memories_fts").fetchall())
    return rows, fts


def test_rebuild_twice_gives_the_same_committed_state(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    keep = store.add(_rec("keep me"))
    gone = store.add(_rec("delete me"))
    store.update(keep.memory_id, {"text": "keep me v2",
                                  "content_hash": hashlib.sha256(b"keep me v2").hexdigest()})
    store.delete(gone.memory_id)

    asyncio.run(idx.rebuild())
    first = _snapshot(milvus, idx.fts_path)
    asyncio.run(idx.rebuild())
    second = _snapshot(milvus, idx.fts_path)

    assert first == second
    assert set(first[0]) == {keep.memory_id}
    assert [row[0] for row in first[1]] == [keep.memory_id]
    assert first[1][0][1:] == (2, "keep me v2")


@pytest.mark.parametrize("damage", ["missing", "garbage"])
def test_missing_or_damaged_fts_is_rebuilt_at_startup(store, milvus, tmp_path, damage):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("survives damage"))
    asyncio.run(idx.tick())
    milvus_acks = [r for r in store.pending("milvus")]

    if damage == "missing":
        idx.fts_path.unlink()
    else:
        idx.fts_path.write_bytes(b"not a sqlite file at all")

    again = _index(store, milvus, tmp_path)

    assert written.memory_id in _fts_ids(again.fts_path)
    assert store.pending("fts:minitron") == []
    assert store.pending("milvus") == milvus_acks


# --- Generations (AC-35) -----------------------------------------------------


def test_generation_bump_makes_only_milvus_pending(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("generation"))
    asyncio.run(idx.tick())

    store.bump_generation()

    assert [r["memory_id"] for r in store.pending("milvus")] == [written.memory_id]
    assert store.pending("fts:minitron") == []
    asyncio.run(idx.tick())
    assert milvus.rows()[written.memory_id]["generation"] == 2


def test_mismatched_local_model_refuses_memory_vectors(store, milvus, tmp_path):
    _index(store, milvus, tmp_path, host="minitron", model_name="embeddinggemma")
    other = _index(store, milvus, tmp_path, host="blue", model_name="modernbert")
    written = store.add(_rec("wrong model"))

    asyncio.run(other.tick())

    assert written.memory_id not in milvus.rows()
    assert [r["memory_id"] for r in store.pending("milvus")] == [written.memory_id]
    assert store.pending("fts:blue") == []
    assert "modernbert" in other.health()["milvus_blocked"]


# --- Full reconcile (D-23) ---------------------------------------------------


def test_reconcile_resets_a_row_that_differs(store, milvus, tmp_path):
    idx = _index(store, milvus, tmp_path)
    written = store.add(_rec("tampered"))
    asyncio.run(idx.tick())
    pk = next(k for k, v in milvus.collections["memories"].items() if v["memory_id"] == written.memory_id)
    milvus.collections["memories"][pk]["version"] = 99
    with sqlite3.connect(idx.fts_path) as conn:
        conn.execute("UPDATE memories_fts SET version = 7 WHERE memory_id = ?", (written.memory_id,))

    reset = asyncio.run(idx.reconcile())

    assert reset == 2
    assert {r["memory_id"] for r in store.pending("milvus")} == {written.memory_id}
    asyncio.run(idx.tick())
    assert milvus.rows()[written.memory_id]["version"] == 1


# --- Similarity warnings (AC-36) ---------------------------------------------


def test_similar_lists_active_same_project_matches(store, milvus, tmp_path, monkeypatch):
    import rag_engine

    idx = _index(store, milvus, tmp_path)
    near = store.add(_rec("x" * 300))
    old = store.add(_rec("superseded neighbour"))
    store.add(_rec("newer", supersedes=[old.memory_id]))
    asyncio.run(idx.tick())

    def fake_threshold_search(client, vector, *, project, user_id, generation, threshold, **kw):
        assert project == "sessionflow" and threshold == 0.9
        return [{"memory_id": near.memory_id, "score": 0.95},
                {"memory_id": old.memory_id, "score": 0.93}]

    monkeypatch.setattr(rag_engine, "memory_threshold_search", fake_threshold_search)
    similar, state = asyncio.run(idx.similar("some new text", project="sessionflow",
                                             user_id="lbruton", threshold=0.9))

    assert state == "ok"
    assert similar == [{"memory_id": near.memory_id, "score": 0.95, "text": "x" * 200}]


def test_similar_is_pending_while_vectors_lag(store, milvus, tmp_path, monkeypatch):
    import rag_engine

    idx = _index(store, milvus, tmp_path)
    store.add(_rec("not yet indexed"))
    monkeypatch.setattr(rag_engine, "memory_threshold_search", lambda *a, **k: [])

    _, state = asyncio.run(idx.similar("anything", project="sessionflow",
                                       user_id="lbruton", threshold=0.9))
    assert state == "pending"
