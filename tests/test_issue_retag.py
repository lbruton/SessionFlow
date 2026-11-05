"""Tests for ``cleanup.py retag-issues`` (``issue_retag``).

The retag re-runs issue-ID extraction on each stored ``document`` and rewrites
``issue_ids`` in Milvus and FTS without embedding. Every test runs against an
in-memory Milvus and a scratch FTS file, so no test reaches the live index.
"""

from __future__ import annotations

import contextlib
import importlib

import pytest

from fts_hybrid import FTSIndex

rag_engine = importlib.import_module("rag_engine")
issue_patterns = importlib.import_module("issue_patterns")
issue_retag = importlib.import_module("issue_retag")

VECTOR = [0.25] * 4


@pytest.fixture(autouse=True)
def fake_milvus(monkeypatch):
    """An in-memory Milvus keyed by primary key; autouse so nothing reaches a live index."""
    state = {"rows": {}, "upserts": []}

    class Client:
        def has_collection(self, name):
            return True

        def query(self, **kwargs):
            flt = kwargs.get("filter", "")
            if flt.startswith("id == "):
                row = state["rows"].get(int(flt.split("==")[1]))
                return [dict(row)] if row else []
            return [dict(r) for r in state["rows"].values()]

        def query_iterator(self, **kwargs):
            fields = kwargs.get("output_fields") or []
            rows = [{k: v for k, v in r.items() if k in fields} for r in state["rows"].values()]
            batches = [rows[i:i + 2] for i in range(0, len(rows), 2)]

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
    return state


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    fts = FTSIndex("turns_fts", list(rag_engine._fts.metadata_columns))
    monkeypatch.setattr(rag_engine, "_fts", fts)
    return str(tmp_path / "milvus.db")


@pytest.fixture
def servicenow(tmp_path, monkeypatch):
    path = tmp_path / "issue_ids.conf"
    path.write_text("preset:servicenow\n", encoding="utf-8")
    monkeypatch.setenv(issue_patterns.ENV_VAR, str(path))
    issue_patterns.reset()


def _row(doc_id, document, issue_ids, chunk_type="turn"):
    return {
        "id": rag_engine._pk_from_doc_id(doc_id),
        "vector": list(VECTOR),
        "document": document,
        "doc_id": doc_id,
        "session_id": "s1",
        "logical_session_id": "s1",
        "provider": "claude_code_cli",
        "source_kind": "claude_code_jsonl",
        "source_class": "native",
        "source_id": "",
        "source_path": "",
        "transcript_file": "",
        "turn_index": 1,
        "timestamp": "2026-11-01T10:00:00Z",
        "git_branch": "main",
        "chunk_type": chunk_type,
        "project_root": "/repo",
        "issue_ids": issue_ids,
    }


def _add(fake_milvus, db_path, row, fts_issue_ids=None):
    fake_milvus["rows"][row["id"]] = dict(row)
    fts_row = {k: v for k, v in row.items() if k not in ("id", "vector", "document", "transcript_file")}
    if fts_issue_ids is not None:
        fts_row["issue_ids"] = fts_issue_ids
    conn = rag_engine._fts.connection(db_path)
    rag_engine._fts.insert(conn, [{**fts_row, "content": row["document"]}])
    rag_engine._fts.close_ephemeral(conn)


def _fts_issue_ids(db_path, doc_id):
    conn = rag_engine._fts.connection(db_path)
    try:
        hit = conn.execute("SELECT issue_ids FROM turns_fts WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        rag_engine._fts.close_ephemeral(conn)
    return hit[0] if hit else None


@pytest.fixture
def seeded(fake_milvus, db_path, servicenow):
    parent = "s1::t1"
    _add(fake_milvus, db_path, _row(parent, "User: see INC0000001 and SESF-1", ",SESF-1,"))
    _add(fake_milvus, db_path, _row(parent + "::p0", "tail text with no ids", ",SESF-1,",
                                    chunk_type="turn_passage"))
    _add(fake_milvus, db_path, _row("s1::t2", "User: SESF-2 only", ",SESF-2,"))
    _add(fake_milvus, db_path, _row("s1::t3", "User: legacy SESF-3", ""))
    _add(fake_milvus, db_path, _row("s1::t4", "User: OPS-4 here", ",OPS-4,"), fts_issue_ids="")
    _add(fake_milvus, db_path, _row("gone::t9::p0", "orphan INC0000009", "",
                                    chunk_type="turn_passage"))
    return db_path


def test_plan_finds_changed_rows_only(seeded):
    plan = issue_retag.plan(db_path=seeded)
    assert plan.changes == {
        "s1::t1": ",INC0000001,SESF-1,",
        "s1::t1::p0": ",INC0000001,SESF-1,",   # passages take the parent's tags
        "s1::t3": ",SESF-3,",                  # turn indexed before tagging
        "s1::t4": ",OPS-4,",                   # Milvus right, FTS stale
    }
    assert plan.scanned == 6
    assert plan.fts_stale == 1
    assert plan.added["INC0000001"] == 2
    assert plan.added["SESF-3"] == 1


def test_plan_lists_a_row_missing_from_fts(fake_milvus, db_path):
    # A failed FTS insert after the delete leaves Milvus right and FTS without the row.
    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)
    plan = issue_retag.plan(db_path=db_path)
    assert plan.changes == {"s1::t1": ",SESF-1,"}
    assert plan.fts_stale == 1


def test_missing_fts_rows_are_ignored_when_fts_cannot_be_read(fake_milvus, db_path, monkeypatch):
    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)

    def boom(db_path):
        raise RuntimeError("sidecar locked")

    monkeypatch.setattr(issue_retag, "_fts_issue_ids", boom)
    plan = issue_retag.plan(db_path=db_path)
    assert plan.changes == {}
    assert plan.fts_error


def test_plan_leaves_an_fts_sidecar_with_an_old_schema_alone(fake_milvus, db_path, tmp_path, monkeypatch):
    # Opening the sidecar for writes drops a table whose schema differs; planning must not.
    import sqlite3

    import fts_hybrid

    sentinel = tmp_path / "fts_backfill_needed"
    monkeypatch.setattr(fts_hybrid, "FTS_BACKFILL_SENTINEL", sentinel)
    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)
    fts_path = FTSIndex.db_path(db_path)
    conn = sqlite3.connect(fts_path)
    conn.execute("CREATE TABLE _fts_schema (table_name TEXT PRIMARY KEY, create_sql TEXT NOT NULL)")
    conn.execute("INSERT INTO _fts_schema VALUES ('turns_fts', 'old schema')")
    conn.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(doc_id, content)")
    conn.execute("INSERT INTO turns_fts VALUES ('s1::t1', 'old row')")
    conn.commit()
    conn.close()

    plan = issue_retag.plan(db_path=db_path)

    assert plan.fts_error
    assert not sentinel.exists()
    conn = sqlite3.connect(fts_path)
    try:
        assert conn.execute("SELECT count(*) FROM turns_fts").fetchone()[0] == 1
    finally:
        conn.close()


def test_plan_does_not_create_a_missing_fts_sidecar(fake_milvus, db_path):
    import os

    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)
    plan = issue_retag.plan(db_path=db_path)
    assert plan.fts_error is None
    assert not os.path.exists(FTSIndex.db_path(db_path))


def test_orphan_passage_with_stale_fts_is_listed(fake_milvus, db_path):
    # A passage without a parent keeps its Milvus tags; FTS must still match them.
    _add(fake_milvus, db_path, _row("gone::t9::p0", "orphan text", ",SESF-9,",
                                    chunk_type="turn_passage"), fts_issue_ids="")
    plan = issue_retag.plan(db_path=db_path)
    assert plan.changes == {"gone::t9::p0": ",SESF-9,"}
    assert plan.fts_stale == 1


def test_apply_restores_a_missing_fts_row(fake_milvus, db_path):
    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)
    issue_retag.apply(issue_retag.plan(db_path=db_path), db_path=db_path)
    assert _fts_issue_ids(db_path, "s1::t1") == ",SESF-1,"
    assert issue_retag.plan(db_path=db_path).changes == {}


def test_plan_writes_nothing(seeded, fake_milvus):
    issue_retag.plan(db_path=seeded)
    assert fake_milvus["upserts"] == []
    assert _fts_issue_ids(seeded, "s1::t4") == ""


def test_plan_reports_removed_tags(fake_milvus, db_path, tmp_path, monkeypatch):
    path = tmp_path / "issue_ids.conf"
    path.write_text("deny:OPS\n", encoding="utf-8")
    monkeypatch.setenv(issue_patterns.ENV_VAR, str(path))
    issue_patterns.reset()
    _add(fake_milvus, db_path, _row("s1::t1", "User: OPS-4 and SESF-1", ",OPS-4,SESF-1,"))
    plan = issue_retag.plan(db_path=db_path)
    assert plan.changes == {"s1::t1": ",SESF-1,"}
    assert plan.removed["OPS-4"] == 1


def test_apply_keeps_vectors_and_rewrites_both_stores(seeded, fake_milvus):
    plan = issue_retag.plan(db_path=seeded)
    result = issue_retag.apply(plan, db_path=seeded)
    assert result.written == 4
    assert result.milvus_failed == result.fts_failed == 0
    rows = {r["doc_id"]: r for r in fake_milvus["rows"].values()}
    assert rows["s1::t1"]["issue_ids"] == ",INC0000001,SESF-1,"
    assert rows["s1::t1::p0"]["issue_ids"] == ",INC0000001,SESF-1,"
    assert all(r["vector"] == VECTOR for r in rows.values())
    assert rows["s1::t1"]["document"] == "User: see INC0000001 and SESF-1"
    assert _fts_issue_ids(seeded, "s1::t4") == ",OPS-4,"
    assert _fts_issue_ids(seeded, "s1::t3") == ",SESF-3,"
    assert issue_retag.plan(db_path=seeded).changes == {}


def test_apply_counts_failures(seeded, monkeypatch):
    plan = issue_retag.plan(db_path=seeded)
    outcomes = {"s1::t1": rag_engine.UpsertResult(False, False),
                "s1::t3": rag_engine.UpsertResult(True, False)}

    def fake_update(doc_id, *, db_path=None, **fields):
        return outcomes.get(doc_id, rag_engine.UpsertResult(True, True))

    monkeypatch.setattr(rag_engine, "update_row_metadata", fake_update)
    result = issue_retag.apply(plan, db_path=seeded)
    assert result.written == 2
    assert result.milvus_failed == 1
    assert result.fts_failed == 1


def test_apply_skips_a_row_whose_document_changed_after_the_scan(seeded, fake_milvus):
    # rewrite and sanitize change ``document`` in place under the same doc_id.
    plan = issue_retag.plan(db_path=seeded)
    pk = rag_engine._pk_from_doc_id("s1::t1")
    fake_milvus["rows"][pk]["document"] = "User: redacted text with SESF-1"
    fake_milvus["rows"][pk]["issue_ids"] = ",SESF-1,"
    result = issue_retag.apply(plan, db_path=seeded)
    assert result.changed == 1
    assert result.written == 3
    assert fake_milvus["rows"][pk]["issue_ids"] == ",SESF-1,"


def test_update_row_metadata_refuses_a_changed_document(fake_milvus, db_path):
    row = _row("s1::t1", "User: SESF-1", ",SESF-1,")
    fake_milvus["rows"][row["id"]] = dict(row)
    with pytest.raises(rag_engine.RowChanged):
        rag_engine.update_row_metadata(
            "s1::t1", db_path=db_path, issue_ids=",X-1,",
            expect_document_sha256=rag_engine.document_sha256("other text"))
    assert fake_milvus["upserts"] == []
    result = rag_engine.update_row_metadata(
        "s1::t1", db_path=db_path, issue_ids=",SESF-2,",
        expect_document_sha256=rag_engine.document_sha256("User: SESF-1"))
    assert result.milvus_ok


# --- CLI -------------------------------------------------------------------------


@pytest.fixture
def cleanup_mod(monkeypatch, seeded):
    module = importlib.import_module("cleanup")
    # Other suites swap ``rag_engine`` for a stub before ``cleanup`` is imported.
    monkeypatch.setattr(module, "rag_engine", rag_engine)
    monkeypatch.setattr(module, "get_db_path", lambda: seeded)
    return module


def _run(cleanup_mod, argv):
    args = cleanup_mod.build_parser().parse_args(argv)
    return cleanup_mod.cmd_retag_issues(args)


def test_cli_dry_run_prints_counts_and_writes_nothing(cleanup_mod, fake_milvus, capsys):
    assert _run(cleanup_mod, ["retag-issues"]) == 0
    out = capsys.readouterr().out
    assert "4 of 6 rows" in out
    assert "INC0000001" in out
    assert "preset:servicenow" in out
    assert "Dry run" in out
    assert fake_milvus["upserts"] == []


def test_cli_apply_without_yes_refuses_before_scanning(cleanup_mod, monkeypatch, capsys):
    def boom(**kwargs):
        raise AssertionError("scanned before the confirmation gate")

    monkeypatch.setattr(issue_retag, "plan", boom)
    assert _run(cleanup_mod, ["retag-issues", "--apply"]) == 1
    assert "--yes" in capsys.readouterr().err


def test_cli_apply_writes_and_returns_zero(cleanup_mod, fake_milvus, capsys):
    assert _run(cleanup_mod, ["retag-issues", "--apply", "--yes"]) == 0
    assert "Rewrote 4 rows" in capsys.readouterr().out
    assert len(fake_milvus["upserts"]) == 4


def _unreadable_fts(monkeypatch):
    def boom(db_path):
        raise RuntimeError("sidecar locked")

    monkeypatch.setattr(issue_retag, "_fts_issue_ids", boom)


def test_cli_dry_run_returns_one_when_fts_cannot_be_read(cleanup_mod, monkeypatch, capsys):
    _unreadable_fts(monkeypatch)
    assert _run(cleanup_mod, ["retag-issues"]) == 1
    assert "FTS" in capsys.readouterr().out


def test_cli_apply_refuses_when_fts_cannot_be_read(cleanup_mod, fake_milvus, monkeypatch):
    _unreadable_fts(monkeypatch)
    assert _run(cleanup_mod, ["retag-issues", "--apply", "--yes"]) == 1
    assert fake_milvus["upserts"] == []


def test_cli_apply_returns_one_on_failure(cleanup_mod, monkeypatch):
    monkeypatch.setattr(rag_engine, "update_row_metadata",
                        lambda doc_id, *, db_path=None, **f: rag_engine.UpsertResult(False, False))
    assert _run(cleanup_mod, ["retag-issues", "--apply", "--yes"]) == 1


def _use_patterns(monkeypatch, path):
    monkeypatch.setenv(issue_patterns.ENV_VAR, str(path))
    issue_patterns.reset()


def test_cli_apply_refuses_when_the_config_cannot_be_read(
        cleanup_mod, fake_milvus, monkeypatch, tmp_path, capsys):
    # Only the dashed preset applies then, so applying would strip every other tag.
    _use_patterns(monkeypatch, tmp_path / "missing.conf")
    assert _run(cleanup_mod, ["retag-issues", "--apply", "--yes"]) == 1
    assert "config" in capsys.readouterr().out
    assert fake_milvus["upserts"] == []


def test_cli_apply_refuses_when_config_lines_were_skipped(
        cleanup_mod, fake_milvus, monkeypatch, tmp_path):
    conf = tmp_path / "issue_ids.conf"
    conf.write_text("preset:servicenow\npreset:servicenwo\n")
    _use_patterns(monkeypatch, conf)
    assert _run(cleanup_mod, ["retag-issues", "--apply", "--yes"]) == 1
    assert fake_milvus["upserts"] == []


def test_cli_dry_run_still_reports_with_a_config_error(
        cleanup_mod, monkeypatch, tmp_path, capsys):
    _use_patterns(monkeypatch, tmp_path / "missing.conf")
    assert _run(cleanup_mod, ["retag-issues"]) == 1
    assert "rows change" in capsys.readouterr().out
