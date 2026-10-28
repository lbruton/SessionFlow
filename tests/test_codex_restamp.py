"""SESF-80: re-stamp indexed Codex turns with their task start time and session branch.

* ``rag_engine.stored_metadata_by_provider`` reads chosen fields for one
  provider from Milvus and the FTS sidecar.
* ``update_row_metadata`` accepts ``timestamp`` and ``git_branch``.
* ``cleanup.py restamp-codex``: dry run by default, refuses ``--apply`` without
  ``--yes``, and rewrites only turns whose stored values differ.
"""

from __future__ import annotations

import importlib
import json

import rag_engine
from tests.test_project_path_remap import _fts_insert, _row, fake_milvus, fts_db  # noqa: F401

import pytest


# --- rag_engine primitives ----------------------------------------------------


def test_update_row_metadata_accepts_timestamp_and_branch(fake_milvus, fts_db):  # noqa: F811
    row = _row("c1::a", "/repo", provider="codex")
    fake_milvus["add"](row)
    _fts_insert(fts_db, row)

    result = rag_engine.update_row_metadata(
        "c1::a", db_path=fts_db, timestamp="2026-10-26T02:48:50+00:00", git_branch="dev",
    )

    assert result == rag_engine.UpsertResult(True, True)
    (upsert,) = fake_milvus["upserts"]
    (written,) = upsert["data"]
    assert (written["timestamp"], written["git_branch"]) == ("2026-10-26T02:48:50+00:00", "dev")
    assert written["vector"] == row["vector"]


def test_stored_metadata_by_provider_reads_both_stores(fake_milvus, fts_db):  # noqa: F811
    codex = {**_row("c1::a", "/repo", provider="codex"), "git_branch": ""}
    other = _row("s1::b", "/repo", provider="claude_code_cli")
    fake_milvus["add"](codex)
    fake_milvus["add"](other)
    # FTS still holds an older timestamp for the Codex turn.
    _fts_insert(fts_db, {**codex, "timestamp": "2026-10-27T10:46:05Z"})

    stored = rag_engine.stored_metadata_by_provider(
        "codex", ["timestamp", "git_branch"], db_path=fts_db,
    )

    assert set(stored) == {"c1::a"}
    assert {"timestamp": codex["timestamp"], "git_branch": ""} in stored["c1::a"]
    assert {"timestamp": "2026-10-27T10:46:05Z", "git_branch": ""} in stored["c1::a"]
    assert 'provider == "codex"' in fake_milvus["queries"][0]["filter"]


# --- cleanup.py restamp-codex -------------------------------------------------


REWRITE_TS = "2026-10-26T14:51:34.961Z"


def _codex_home(tmp_path):
    home = tmp_path / "home"
    root = home / ".codex" / "sessions" / "2026" / "10" / "26"
    root.mkdir(parents=True)
    lines = [
        {"type": "session_meta", "payload": {"id": "sess", "cwd": "/repo", "git": {"branch": "dev"}}},
        {"type": "event_msg", "payload": {"type": "task_started", "started_at": 1792982930}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": "q1"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": "a1"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": "a2"}},
    ]
    (root / "rollout-sess.jsonl").write_text(
        "".join(json.dumps({"timestamp": REWRITE_TS, **line}) + "\n" for line in lines)
    )
    return home


@pytest.fixture
def cleanup_mod():
    return importlib.import_module("cleanup")


@pytest.fixture
def restamp(monkeypatch, cleanup_mod, tmp_path):
    from provider_codex import CodexAdapter

    adapter = CodexAdapter(home=_codex_home(tmp_path))
    doc_ids = [t["doc_id"] for t in adapter.parse_source(adapter.discover_sources()[0], None).turns]
    state = {"doc_ids": doc_ids, "stored": {}, "update": [], "results": {}}

    def fake_stored(provider, fields, *, db_path=None):
        state["provider"] = provider
        return {doc_id: list(values) for doc_id, values in state["stored"].items()}

    def fake_update(doc_id, *, db_path=None, **fields):
        state["update"].append((doc_id, fields))
        return state["results"].get(doc_id, rag_engine.UpsertResult(True, True))

    monkeypatch.setattr(cleanup_mod, "rag_engine", rag_engine)
    monkeypatch.setattr(rag_engine, "stored_metadata_by_provider", fake_stored)
    monkeypatch.setattr(rag_engine, "update_row_metadata", fake_update)
    monkeypatch.setattr(cleanup_mod, "get_db_path", lambda: "/tmp/synthetic/milvus.db")
    monkeypatch.setattr(cleanup_mod, "_codex_adapter", lambda: adapter)
    return state


def _run(cleanup_mod, argv):
    args = cleanup_mod.build_parser().parse_args(argv)
    return cleanup_mod.cmd_restamp_codex(args)


STALE = {"timestamp": REWRITE_TS, "git_branch": ""}
GOOD = {"timestamp": "2026-10-26T02:48:50+00:00", "git_branch": "dev"}


def test_restamp_codex_dry_run_writes_nothing(cleanup_mod, restamp, capsys):
    a, b = restamp["doc_ids"]
    restamp["stored"] = {a: [STALE], b: [GOOD]}

    assert _run(cleanup_mod, ["restamp-codex"]) == 0
    assert restamp["provider"] == "codex"
    assert restamp["update"] == []
    out = capsys.readouterr().out.lower()
    assert "dry run" in out and "1" in out


def test_restamp_codex_apply_without_yes_refuses_before_any_read(cleanup_mod, restamp, capsys):
    assert _run(cleanup_mod, ["restamp-codex", "--apply"]) == 1
    assert "provider" not in restamp
    assert restamp["update"] == []
    assert "--yes" in capsys.readouterr().err


def test_restamp_codex_apply_rewrites_only_changed_turns(cleanup_mod, restamp):
    a, b = restamp["doc_ids"]
    # b's Milvus row is current, but its FTS row is stale from a failed rewrite.
    restamp["stored"] = {a: [STALE, STALE], b: [GOOD, STALE], "codex:gone:x": [STALE]}

    assert _run(cleanup_mod, ["restamp-codex", "--apply", "--yes"]) == 0
    assert sorted(restamp["update"]) == sorted([(a, GOOD), (b, GOOD)])


def test_restamp_codex_skips_current_turns(cleanup_mod, restamp):
    a, b = restamp["doc_ids"]
    restamp["stored"] = {a: [GOOD, GOOD], b: [GOOD]}

    assert _run(cleanup_mod, ["restamp-codex", "--apply", "--yes"]) == 0
    assert restamp["update"] == []


def test_restamp_codex_reports_failures(cleanup_mod, restamp):
    a, b = restamp["doc_ids"]
    restamp["stored"] = {a: [STALE], b: [STALE]}
    restamp["results"][b] = rag_engine.UpsertResult(True, False)

    assert _run(cleanup_mod, ["restamp-codex", "--apply", "--yes"]) == 1
