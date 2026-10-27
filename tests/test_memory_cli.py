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
import urllib.error

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


def _project_root(tmp_path, name, config):
    import json

    root = tmp_path / name
    (root / ".claude").mkdir(parents=True)
    (root / ".claude" / "project.json").write_text(json.dumps(config), encoding="utf-8")
    return str(root)


def test_project_inventory_reads_each_known_root(monkeypatch, tmp_path):
    import file_watcher
    import memory_service

    roots = [
        _project_root(tmp_path, "a", {"memory": {"project": "sessionflow"}, "tag": "ignored"}),
        _project_root(tmp_path, "b", {"tag": "DevOps"}),
        str(tmp_path / "no-config"),
    ]
    monkeypatch.setattr(file_watcher, "known_project_roots", lambda: roots)

    assert memory_service.project_inventory() == {"sessionflow", "devops"}


def test_cli_audit_reports_projects_missing_from_the_inventory(
        cli_store, no_embedding, monkeypatch, tmp_path, capsys):
    import file_watcher

    root = _project_root(tmp_path, "sf", {"memory": {"project": "sessionflow"}})
    monkeypatch.setattr(file_watcher, "known_project_roots", lambda: [root])
    monkeypatch.setattr(memory_cli, "_open_neighbour_index", lambda: None)
    cli_store.add(_rec("known project"))
    cli_store.add(_rec("typo project", project="sesionflow"))

    assert memory_cli.main(["audit"]) == 0

    assert "Unknown projects: sesionflow" in capsys.readouterr().out


def test_cli_audit_says_when_no_inventory_is_available(
        cli_store, no_embedding, monkeypatch, capsys):
    import file_watcher

    monkeypatch.setattr(file_watcher, "known_project_roots", lambda: [])
    monkeypatch.setattr(memory_cli, "_open_neighbour_index", lambda: None)
    cli_store.add(_rec("any project", project="sesionflow"))

    assert memory_cli.main(["audit", "--format", "json"]) == 0

    captured = capsys.readouterr()
    assert '"inventory_checked": false' in captured.out
    assert "Unknown-project check skipped" in captured.err


class _StoredVectorMilvus:
    """A Milvus stand-in holding one stored vector and answering the threshold search."""

    def __init__(self, memory_id, neighbour_ids):
        import rag_engine

        self.row = {"id": rag_engine._pk_from_doc_id(memory_id), "vector": [0.5] * 4,
                    "project": "sessionflow", "user_id": "lbruton", "generation": 1}
        self.neighbour_ids = neighbour_ids
        self.searched = []

    def query(self, collection_name, filter="", output_fields=None, limit=None, **kwargs):
        return [dict(self.row)]

    def search(self, collection_name, data, **kwargs):
        self.searched.append(data)
        return [[{"entity": {"memory_id": m}, "distance": 0.99} for m in self.neighbour_ids]]


def test_cli_index_returns_stored_vector_neighbours_without_embedding(no_embedding, tmp_path):
    memory_id = "00000000-0000-0000-0000-000000000001"
    milvus = _StoredVectorMilvus(memory_id, ["other"])

    async def forbidden(texts):
        raise AssertionError("memory_cli must never embed")

    inner = memory_index.MemoryIndex(store=None, milvus=milvus, fts_path=tmp_path / "fts.db", host="cli",
                                     embed=forbidden, model_name="embeddinggemma")
    index = memory_cli._StoredVectorIndex(inner)

    assert asyncio.run(index.neighbours(memory_id, threshold=0.9)) == ["other"]
    assert milvus.searched == [[[0.5] * 4]]
    assert index.unavailable is None
    assert asyncio.run(index.apply(memory_id)) == "pending"


def test_cli_index_reports_unavailable_when_milvus_fails(no_embedding):
    class _Broken:
        async def neighbours(self, memory_id, **kwargs):
            raise RuntimeError("database is locked")

    index = memory_cli._StoredVectorIndex(_Broken())

    assert asyncio.run(index.neighbours("m1", threshold=0.9)) == []
    assert "database is locked" in index.unavailable


def test_audit_says_clusters_are_unavailable_when_milvus_cannot_open(monkeypatch, capsys):
    class _Svc:
        index = object()

        async def audit(self):
            return {"projects": {}, "unknown_projects": []}

    monkeypatch.setattr(memory_cli, "_open_service", lambda: _Svc())
    monkeypatch.setattr(memory_cli, "_open_neighbour_index", lambda: None)

    assert memory_cli.main(["audit"]) == 0
    assert "clusters are unavailable" in capsys.readouterr().err


def test_audit_uses_the_stored_vector_index(monkeypatch):
    svc = type("Svc", (), {"index": object(), "audit": lambda self: _done()})()

    async def _done():
        return {"projects": {}, "unknown_projects": []}

    stored = memory_cli._StoredVectorIndex(object())
    monkeypatch.setattr(memory_cli, "_open_service", lambda: svc)
    monkeypatch.setattr(memory_cli, "_open_neighbour_index", lambda: stored)

    assert memory_cli.main(["audit"]) == 0
    assert svc.index is stored


def test_post_restore_check_survives_transport_failures(monkeypatch):
    def refused(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(memory_cli.urllib.request, "urlopen", refused)
    assert memory_cli._post_restore_check({}) == {"ok": False, "error": "connection refused"}

    def slow(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(memory_cli.urllib.request, "urlopen", slow)
    assert memory_cli._post_restore_check({}) == {"ok": False, "error": "timed out"}


def test_post_restore_check_handles_a_non_json_http_error(monkeypatch):
    import io

    def bad_gateway(*a, **k):
        raise urllib.error.HTTPError("http://x", 502, "Bad Gateway", {}, io.BytesIO(b"<html>"))

    monkeypatch.setattr(memory_cli.urllib.request, "urlopen", bad_gateway)
    assert memory_cli._post_restore_check({}) == {"ok": False,
                                                  "error": "server returned HTTP 502"}


def test_restore_check_needs_pg_restore(monkeypatch, capsys):
    monkeypatch.setattr(memory_cli.shutil, "which", lambda name: None)
    assert memory_cli.main(["restore-check", "x.dump", "--target-dsn", "dbname=none"]) == 2
    assert "pg_restore is not on PATH" in capsys.readouterr().err


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


needs_restore_dsn = pytest.mark.skipif(not restore_dsn, reason="SESSIONFLOW_TEST_PG_RESTORE_DSN unset")


@needs_restore_dsn
def test_restore_check_reports_a_pg_restore_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(memory_cli.shutil, "which", lambda name: "/usr/bin/pg_restore")
    stderr = "pg_restore: connecting to database\npg_restore: error: could not execute query\n"
    monkeypatch.setattr(memory_cli.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, stdout="", stderr=stderr))

    assert memory_cli.main(["restore-check", str(tmp_path / "x.dump"),
                            "--target-dsn", restore_dsn]) == 1

    err = capsys.readouterr().err
    assert "pg_restore failed: pg_restore: error: could not execute query" in err
    assert "[" not in err
    assert _target_tables() == 0


@needs_restore
def test_restore_check_fails_when_the_server_is_unreachable(memory_pg, store, tmp_path,
                                                            monkeypatch):
    store.add(_rec("anything"))
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)

    def refused(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(memory_cli.urllib.request, "urlopen", refused)

    assert memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn]) == 1
    assert _target_tables() == 0


@needs_restore
def test_restore_check_refuses_a_non_empty_target(memory_pg, store, server_side, tmp_path):
    store.add(_rec("anything"))
    dump = tmp_path / "memory.dump"
    _dump(memory_pg, dump)
    with psycopg.connect(restore_dsn, autocommit=True) as conn:
        conn.execute("CREATE TABLE leftover (i int)")
    try:
        assert memory_cli.main(["restore-check", str(dump), "--target-dsn", restore_dsn]) == 2
        assert _target_tables() == 1
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
