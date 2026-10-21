"""SESF-73 tests: remap moved project folders at ingest and re-tag indexed rows.

Three layers:

* ``provider_adapters`` path helpers: ``parse_project_path_map``,
  ``path_has_prefix`` and ``remap_project_root`` (driven by
  ``SESSIONFLOW_PROJECT_PATH_MAP``).
* ``rag_engine`` primitives: ``update_row_metadata`` rewrites metadata on one
  row without touching its vector, and ``find_doc_ids_by_project_root`` finds
  rows under an old root in Milvus *and* FTS (so an FTS-failed row stays
  retryable).
* ``cleanup.py retag-project``: dry run by default, refuses ``--apply`` without
  ``--yes``, and maps each row's old root to the new one.

Milvus is faked; FTS is a real SQLite sidecar under ``tmp_path``.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path

import pytest

import rag_engine
from fts_hybrid import FTSIndex

OLD = "/Users/someone/CoWork"
NEW = "/Users/someone/Claude/Projects/CoWork"


# --- provider_adapters path helpers -----------------------------------------


def test_parse_project_path_map_orders_longest_prefix_first():
    from provider_adapters import parse_project_path_map

    mapping = parse_project_path_map(f"/a=>/x; /a/b/=>/y ;{OLD}=>{NEW}")
    assert mapping == [("/Users/someone/CoWork", NEW), ("/a/b", "/y"), ("/a", "/x")]


@pytest.mark.parametrize("raw", ["", None, "   ", ";;"])
def test_parse_project_path_map_empty(raw):
    from provider_adapters import parse_project_path_map

    assert parse_project_path_map(raw) == []


def test_parse_project_path_map_skips_malformed_entries():
    from provider_adapters import parse_project_path_map

    mapping = parse_project_path_map(f"no-arrow;relative=>/x;/a=>;=>/b;{OLD}=>{NEW}")
    assert mapping == [(OLD, NEW)]


@pytest.mark.parametrize("path,prefix,expected", [
    (OLD, OLD, True),
    (OLD + "/sub/dir", OLD, True),
    (OLD + "X", OLD, False),
    ("/Users/someone", OLD, False),
    (OLD + "/", OLD, True),
])
def test_path_has_prefix_respects_component_boundary(path, prefix, expected):
    from provider_adapters import path_has_prefix

    assert path_has_prefix(path, prefix) is expected


def test_remap_project_root_exact_and_subpath(monkeypatch):
    from provider_adapters import remap_project_root

    monkeypatch.setenv("SESSIONFLOW_PROJECT_PATH_MAP", f"{OLD}=>{NEW}")
    assert remap_project_root(OLD) == NEW
    assert remap_project_root(OLD + "/notes") == NEW + "/notes"


def test_remap_project_root_leaves_partial_component_alone(monkeypatch):
    from provider_adapters import remap_project_root

    monkeypatch.setenv("SESSIONFLOW_PROJECT_PATH_MAP", f"{OLD}=>{NEW}")
    assert remap_project_root(OLD + "X") == OLD + "X"


def test_remap_project_root_longest_prefix_wins(monkeypatch):
    from provider_adapters import remap_project_root

    monkeypatch.setenv(
        "SESSIONFLOW_PROJECT_PATH_MAP", f"{OLD}=>{NEW};{OLD}/special=>/elsewhere",
    )
    assert remap_project_root(OLD + "/special/x") == "/elsewhere/x"
    assert remap_project_root(OLD + "/other") == NEW + "/other"


def test_remap_project_root_never_touches_unknown_or_empty(monkeypatch):
    from provider_adapters import remap_project_root

    monkeypatch.setenv("SESSIONFLOW_PROJECT_PATH_MAP", "unknown=>/x")
    assert remap_project_root("unknown") == "unknown"
    assert remap_project_root("") == ""


def test_remap_project_root_noop_without_env(monkeypatch):
    from provider_adapters import remap_project_root

    monkeypatch.delenv("SESSIONFLOW_PROJECT_PATH_MAP", raising=False)
    assert remap_project_root(OLD) == OLD


# --- Cowork legacy adapter ----------------------------------------------------


def _make_session(root: Path, name: str, metadata: dict) -> None:
    space = root / "space-uuid" / "account-uuid"
    transcript = (
        space / f"local_{name}" / ".claude" / "projects" / f"-sessions-{name}"
        / f"{name}-session.jsonl"
    )
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({
        "type": "user", "cwd": f"/sessions/{name}",
        "timestamp": "2026-03-02T10:00:00Z",
        "message": {"content": "synthetic cowork request"},
    }) + "\n")
    (space / f"local_{name}.json").write_text(json.dumps(metadata))


def test_cowork_adapter_remaps_selected_folder(tmp_path, monkeypatch):
    from provider_claude import ClaudeCoworkLegacyAdapter

    root = tmp_path / "local-agent-mode-sessions"
    _make_session(root, "alpha", {"userSelectedFolders": [OLD]})
    _make_session(root, "beta", {"cwd": OLD + "/sub"})
    _make_session(root, "gamma", {"userSelectedFolders": []})
    monkeypatch.setenv("SESSIONFLOW_PROJECT_PATH_MAP", f"{OLD}=>{NEW}")

    roots = {
        s.logical_session_id: s.project_root
        for s in ClaudeCoworkLegacyAdapter(root=root).discover_sources()
    }
    assert roots == {
        "alpha-session": NEW,
        "beta-session": NEW + "/sub",
        "gamma-session": "unknown",
    }


def test_cowork_adapter_unchanged_without_map(tmp_path, monkeypatch):
    from provider_claude import ClaudeCoworkLegacyAdapter

    root = tmp_path / "local-agent-mode-sessions"
    _make_session(root, "alpha", {"userSelectedFolders": [OLD]})
    monkeypatch.delenv("SESSIONFLOW_PROJECT_PATH_MAP", raising=False)

    (source,) = ClaudeCoworkLegacyAdapter(root=root).discover_sources()
    assert source.project_root == OLD


# --- rag_engine primitives ----------------------------------------------------


VECTOR = [0.25] * 4


def _row(doc_id: str, project_root: str, provider: str = "claude_cowork_legacy") -> dict:
    return {
        "id": rag_engine._pk_from_doc_id(doc_id),
        "doc_id": doc_id,
        "vector": list(VECTOR),
        "document": f"synthetic text for {doc_id}",
        "session_id": "s1",
        "provider": provider,
        "project_root": project_root,
        "issue_ids": ",SESF-73,",
        "turn_index": 7,
        "timestamp": "2026-06-10T10:00:00Z",
        "chunk_type": "turn",
    }


@pytest.fixture
def fake_milvus(monkeypatch):
    """A tiny in-memory Milvus keyed by primary key, recording calls."""
    state = {"rows": {}, "queries": [], "upserts": []}

    class Client:
        def has_collection(self, name):
            return True

        def query(self, **kwargs):
            state["queries"].append(kwargs)
            flt = kwargs.get("filter", "")
            if flt.startswith("id == "):
                pk = int(flt.split("==")[1])
                row = state["rows"].get(pk)
                return [dict(row)] if row else []
            return [dict(r) for r in state["rows"].values()]

        def query_iterator(self, **kwargs):
            # One batch of every row: the code under test must re-check the
            # prefix itself, as it does against real LIKE wildcard matches.
            state["queries"].append(kwargs)
            batches = [[dict(r) for r in state["rows"].values()]]

            class _Iterator:
                def next(self):
                    return batches.pop(0) if batches else []

                def close(self):
                    pass

            return _Iterator()

        def upsert(self, **kwargs):
            state["upserts"].append(kwargs)
            for row in kwargs["data"]:
                state["rows"][row["id"]] = dict(row)
            return {"upsert_count": len(kwargs["data"])}

    @contextlib.contextmanager
    def fake_client(db_path=None):
        yield Client()

    monkeypatch.setattr(rag_engine, "milvus_client", fake_client)

    def add(row):
        state["rows"][row["id"]] = dict(row)

    state["add"] = add
    return state


@pytest.fixture
def fts_db(tmp_path, monkeypatch):
    """Point ``rag_engine._fts`` at a fresh FTS sidecar and return the Milvus db_path."""
    fts = FTSIndex("turns_fts", list(rag_engine._fts.metadata_columns))
    monkeypatch.setattr(rag_engine, "_fts", fts)
    return str(tmp_path / "milvus.db")


def _fts_insert(db_path: str, row: dict) -> None:
    conn = rag_engine._fts.connection(db_path)
    rag_engine._fts.insert(conn, [{**row, "content": row["document"]}])
    rag_engine._fts.close_ephemeral(conn)


def _fts_project_root(db_path: str, doc_id: str):
    conn = rag_engine._fts.connection(db_path)
    try:
        hit = conn.execute(
            "SELECT project_root, content, issue_ids FROM turns_fts WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    finally:
        rag_engine._fts.close_ephemeral(conn)
    return hit


def test_update_row_metadata_keeps_vector_and_rewrites_both_stores(fake_milvus, fts_db):
    row = _row("s1::a", OLD)
    fake_milvus["add"](row)
    _fts_insert(fts_db, row)

    result = rag_engine.update_row_metadata("s1::a", project_root=NEW, db_path=fts_db)

    assert result == rag_engine.UpsertResult(True, True)
    (upsert,) = fake_milvus["upserts"]
    (written,) = upsert["data"]
    assert written["project_root"] == NEW
    assert written["vector"] == VECTOR
    assert written["document"] == row["document"]
    assert written["issue_ids"] == ",SESF-73,"
    project_root, content, issue_ids = _fts_project_root(fts_db, "s1::a")
    assert (project_root, content, issue_ids) == (NEW, row["document"], ",SESF-73,")


def test_update_row_metadata_rejects_unknown_fields(fake_milvus, fts_db):
    fake_milvus["add"](_row("s1::a", OLD))
    with pytest.raises(ValueError):
        rag_engine.update_row_metadata("s1::a", vector=[0.0], db_path=fts_db)
    with pytest.raises(ValueError):
        rag_engine.update_row_metadata("s1::a", db_path=fts_db)
    assert fake_milvus["upserts"] == []


def test_update_row_metadata_missing_row(fake_milvus, fts_db):
    result = rag_engine.update_row_metadata("s1::missing", project_root=NEW, db_path=fts_db)
    assert result == rag_engine.UpsertResult(False, False)
    assert fake_milvus["upserts"] == []


def test_update_row_metadata_reports_fts_failure(fake_milvus, fts_db, monkeypatch):
    row = _row("s1::a", OLD)
    fake_milvus["add"](row)
    _fts_insert(fts_db, row)

    def boom(conn, records):
        raise RuntimeError("synthetic FTS failure")

    monkeypatch.setattr(rag_engine._fts, "insert", boom)
    result = rag_engine.update_row_metadata("s1::a", project_root=NEW, db_path=fts_db)
    assert result == rag_engine.UpsertResult(True, False)


def test_find_doc_ids_by_project_root_matches_prefix_on_boundary(fake_milvus, fts_db):
    for row in (
        _row("s1::exact", OLD),
        _row("s1::sub", OLD + "/notes"),
        _row("s1::partial", OLD + "X"),
        _row("s1::other", "/elsewhere"),
        _row("s1::cli", OLD, provider="claude_code_cli"),
    ):
        fake_milvus["add"](row)

    found = rag_engine.find_doc_ids_by_project_root(
        OLD, provider="claude_cowork_legacy", db_path=fts_db,
    )

    assert found == {"s1::exact": OLD, "s1::sub": OLD + "/notes"}
    flt = fake_milvus["queries"][0]["filter"]
    assert f'project_root == "{OLD}"' in flt
    assert f'project_root like "{OLD}/%"' in flt
    assert 'provider == "claude_cowork_legacy"' in flt


def test_find_doc_ids_by_project_root_includes_fts_stale_rows(fake_milvus, fts_db):
    # Milvus already moved (a prior run's FTS rewrite failed); FTS still has OLD.
    moved = _row("s1::stale", NEW)
    fake_milvus["add"](moved)
    _fts_insert(fts_db, {**moved, "project_root": OLD})

    found = rag_engine.find_doc_ids_by_project_root(OLD, db_path=fts_db)
    assert found == {"s1::stale": OLD}


def test_find_doc_ids_by_project_root_escapes_filter(fake_milvus, fts_db):
    rag_engine.find_doc_ids_by_project_root('/odd"path', db_path=fts_db)
    flt = fake_milvus["queries"][0]["filter"]
    assert 'project_root == "/odd\\"path"' in flt


# --- cleanup.py retag-project ---------------------------------------------------


@pytest.fixture
def cleanup_mod():
    return importlib.import_module("cleanup")


@pytest.fixture
def stub_engine(monkeypatch, cleanup_mod):
    calls = {"find": [], "update": []}
    found = {"s1::a": OLD, "s1::b": OLD + "/notes"}
    results = {}

    def fake_find(prefix, *, provider=None, db_path=None):
        calls["find"].append((prefix, provider))
        return dict(found)

    def fake_update(doc_id, *, db_path=None, **fields):
        calls["update"].append((doc_id, fields))
        return results.get(doc_id, rag_engine.UpsertResult(True, True))

    # Other suites swap ``rag_engine`` for a stub namespace before ``cleanup`` is
    # first imported, so pin the real module before patching it.
    monkeypatch.setattr(cleanup_mod, "rag_engine", rag_engine)
    monkeypatch.setattr(rag_engine, "find_doc_ids_by_project_root", fake_find)
    monkeypatch.setattr(rag_engine, "update_row_metadata", fake_update)
    monkeypatch.setattr(cleanup_mod, "get_db_path", lambda: "/tmp/synthetic/milvus.db")
    calls["results"] = results
    return calls


def _run(cleanup_mod, argv):
    args = cleanup_mod.build_parser().parse_args(argv)
    return cleanup_mod.cmd_retag_project(args)


def test_retag_project_dry_run_writes_nothing(cleanup_mod, stub_engine, capsys):
    rc = _run(cleanup_mod, [
        "retag-project", "--from", OLD, "--to", NEW, "--provider", "claude_cowork_legacy",
    ])
    assert rc == 0
    assert stub_engine["find"] == [(OLD, "claude_cowork_legacy")]
    assert stub_engine["update"] == []
    out = capsys.readouterr().out
    assert "2" in out and "dry run" in out.lower()


def test_retag_project_apply_without_yes_refuses_before_any_call(
    cleanup_mod, stub_engine, capsys,
):
    rc = _run(cleanup_mod, ["retag-project", "--from", OLD, "--to", NEW, "--apply"])
    assert rc != 0
    assert stub_engine["find"] == []
    assert stub_engine["update"] == []
    assert "--yes" in capsys.readouterr().err


def test_retag_project_apply_maps_each_row(cleanup_mod, stub_engine):
    rc = _run(cleanup_mod, ["retag-project", "--from", OLD, "--to", NEW, "--apply", "--yes"])
    assert rc == 0
    assert sorted(stub_engine["update"]) == [
        ("s1::a", {"project_root": NEW}),
        ("s1::b", {"project_root": NEW + "/notes"}),
    ]


def test_retag_project_apply_reports_failures(cleanup_mod, stub_engine, capsys):
    stub_engine["results"]["s1::b"] = rag_engine.UpsertResult(True, False)
    rc = _run(cleanup_mod, ["retag-project", "--from", OLD, "--to", NEW, "--apply", "--yes"])
    assert rc == 1
    assert "1" in capsys.readouterr().out


def test_retag_project_rejects_relative_or_identical_paths(cleanup_mod, stub_engine):
    assert _run(cleanup_mod, ["retag-project", "--from", "CoWork", "--to", NEW]) == 2
    assert _run(cleanup_mod, ["retag-project", "--from", OLD, "--to", OLD]) == 2
    assert stub_engine["find"] == []
