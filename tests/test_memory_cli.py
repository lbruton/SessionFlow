"""memory_cli: rebuild and reindex reset acknowledgements, audit reports, import
delegates, and restore-check proves a dump rebuilds the same committed state.

The CLI never embeds; the server's sync loop and restore-check route do.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

# A collection-time ImportError would stop the whole test session.
try:
    import memory_cli
    import memory_index
    from memory_store import MemoryStore, NewMemory
except ImportError:
    memory_cli = memory_index = MemoryStore = NewMemory = None

from tests.test_memory_index import FakeMilvus, _embed  # noqa: E402 - shared fakes


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
def no_embedding(monkeypatch):
    import rag_engine

    def forbidden(*a, **k):
        raise AssertionError("memory_cli must never embed")

    monkeypatch.setattr(rag_engine, "embed_texts", forbidden)
    if memory_index is not None:
        monkeypatch.setattr(memory_index, "embed_for_index", forbidden)


@pytest.fixture
def cli_store(monkeypatch, store):
    monkeypatch.setattr(memory_cli, "_open_store", lambda: store)
    return store


# --- rebuild and reindex (AC-29, D-8) ----------------------------------------


def test_rebuild_resets_every_acknowledgement(cli_store, no_embedding):
    store = cli_store
    ids = [store.add(_rec(f"memory {i}")).memory_id for i in range(3)]
    for mid in ids:
        store.ack("milvus", mid, version=1, generation=1, deleted=False)
        store.ack("fts:minitron", mid, version=1, generation=1, deleted=False)

    assert memory_cli.main(["rebuild", "--yes"]) == 0

    assert {r["memory_id"] for r in store.pending("milvus")} == set(ids)
    assert {r["memory_id"] for r in store.pending("fts:minitron")} == set(ids)


def test_rebuild_needs_confirmation(cli_store, no_embedding, capsys):
    store = cli_store
    mid = store.add(_rec("kept acked")).memory_id
    store.ack("milvus", mid, version=1, generation=1, deleted=False)
    assert memory_cli.main(["rebuild"]) != 0
    assert store.pending("milvus") == []
    assert "--yes" in capsys.readouterr().err


def test_reindex_one_memory(cli_store, no_embedding):
    store = cli_store
    a = store.add(_rec("a")).memory_id
    b = store.add(_rec("b")).memory_id
    for mid in (a, b):
        store.ack("milvus", mid, version=1, generation=1, deleted=False)

    assert memory_cli.main(["reindex", "--memory-id", a]) == 0
    assert [r["memory_id"] for r in store.pending("milvus")] == [a]


# --- audit and import --------------------------------------------------------


def test_audit_prints_the_report_and_changes_nothing(monkeypatch, capsys):
    calls = []

    class _Svc:
        async def audit(self):
            calls.append("audit")
            return {"projects": {"sessionflow": {"clusters": [], "stale": [], "counts": {}}},
                    "unknown_projects": []}

    monkeypatch.setattr(memory_cli, "_open_service", lambda: _Svc())
    assert memory_cli.main(["audit", "--format", "json"]) == 0
    assert calls == ["audit"]
    assert '"unknown_projects"' in capsys.readouterr().out


def test_import_delegates_to_the_importer(monkeypatch, tmp_path):
    seen = {}

    async def fake_import(data, *, service, report_path):
        seen.update(data=data, report_path=report_path, service=service)
        return {"totals": {"read": 1}, "records": [], "counts": {}}

    svc = object()
    monkeypatch.setattr(memory_cli, "_open_service", lambda: svc)
    monkeypatch.setattr(memory_import_module(), "import_records", fake_import)
    src = tmp_path / "import.json"
    src.write_text('{"source": "mem0", "records": []}')
    report = tmp_path / "report.json"

    assert memory_cli.main(["import", str(src), "--report", str(report)]) == 0
    assert seen["data"] == {"source": "mem0", "records": []}
    assert seen["report_path"] == report and seen["service"] is svc


def memory_import_module():
    import memory_import
    return memory_import


# --- restore-check (AC-33) ---------------------------------------------------

restore_dsn = os.environ.get("SESSIONFLOW_TEST_PG_RESTORE_DSN")
needs_restore = pytest.mark.skipif(
    not restore_dsn or shutil.which("pg_restore") is None or shutil.which("pg_dump") is None,
    reason="SESSIONFLOW_TEST_PG_RESTORE_DSN unset or pg_dump/pg_restore 17 missing",
)


def _dump(memory_pg, path):
    params = conninfo_to_dict(memory_pg)
    schema = params.pop("options").split("search_path=")[1]
    base = psycopg.conninfo.make_conninfo(**params)
    subprocess.run(["pg_dump", "-Fc", "--no-owner", "--schema", schema, "-f", str(path), base],
                   check=True)
    return schema


def _target_tables():
    with psycopg.connect(restore_dsn) as conn:
        return conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
        ).fetchone()[0]


@pytest.fixture
def server_side(monkeypatch, tmp_path, store):
    """Route the CLI's restore-check call to the server-side handler, in process."""
    prod_milvus = FakeMilvus()
    prod_milvus.create_collection("memories")
    prod_milvus.collections["memories"][1] = {"id": 1, "memory_id": "prod", "version": 1,
                                              "generation": 1, "project": "sessionflow"}

    def post(payload):
        return asyncio.run(memory_index.run_restore_check(
            target_dsn=payload["target_dsn"], manifest=payload["manifest"],
            milvus=prod_milvus, fts_dir=tmp_path, embed=_embed,
            model_name="embeddinggemma", timeout_s=payload.get("timeout_s", 30)))

    monkeypatch.setattr(memory_cli, "_post_restore_check", post)
    return prod_milvus


@needs_restore
def test_restore_check_rebuilds_and_matches(memory_pg, store, server_side, tmp_path, no_embedding):
    keep = store.add(_rec("keep"))
    gone = store.add(_rec("gone"))
    store.update(keep.memory_id, {"text": "keep v2", "content_hash": hashlib.sha256(b"keep v2").hexdigest()})
    store.delete(gone.memory_id)
    before_rows = store.versions(keep.memory_id)
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)

    rc = memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn])

    assert rc == 0
    assert _target_tables() == 0
    assert set(server_side.collections) == {"memories"}
    assert set(r["memory_id"] for r in server_side.rows().values()) == {"prod"}
    assert not list(tmp_path.glob("*restore*fts*"))
    assert store.versions(keep.memory_id) == before_rows
    with psycopg.connect(memory_pg) as conn:
        assert conn.execute("SELECT count(*) FROM memory_index_acks").fetchone()[0] == 0


@needs_restore
def test_restore_check_refuses_a_non_empty_target(memory_pg, store, server_side, tmp_path):
    store.add(_rec("anything"))
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)
    with psycopg.connect(restore_dsn, autocommit=True) as conn:
        conn.execute("CREATE TABLE leftover (i int)")
    try:
        assert memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn]) != 0
    finally:
        with psycopg.connect(restore_dsn, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS leftover")


@needs_restore
def test_restore_check_fails_on_a_changed_record(memory_pg, store, server_side, tmp_path, monkeypatch):
    store.add(_rec("original"))
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)

    real_post = memory_cli._post_restore_check

    def tampered(payload):
        payload = dict(payload)
        payload["manifest"] = [dict(m, content_hash="0" * 64) for m in payload["manifest"]]
        return real_post(payload)

    monkeypatch.setattr(memory_cli, "_post_restore_check", tampered)
    assert memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn]) != 0
    assert _target_tables() == 0


@needs_restore
def test_restore_check_fails_on_timeout(memory_pg, store, server_side, tmp_path, monkeypatch):
    store.add(_rec("slow"))
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)

    async def never(texts):
        await asyncio.sleep(3600)

    def slow(payload):
        return asyncio.run(memory_index.run_restore_check(
            target_dsn=payload["target_dsn"], manifest=payload["manifest"],
            milvus=server_side, fts_dir=tmp_path, embed=never,
            model_name="embeddinggemma", timeout_s=payload["timeout_s"]))

    monkeypatch.setattr(memory_cli, "_post_restore_check", slow)
    assert memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn,
                            "--timeout", "0.5"]) != 0
    assert _target_tables() == 0
