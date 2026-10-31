"""SESF-80: prune indexed Codex turns that a full parse of their rollout no longer produces.

* Only sessions whose rollout is still on disk are touched.
* Dry run by default; ``--apply`` refuses without ``--yes`` before any read.
* A delete whose FTS half fails makes the command exit 1.
"""

from __future__ import annotations

import importlib

import rag_engine
from tests.test_codex_restamp import _codex_home

import pytest


@pytest.fixture
def cleanup_mod():
    return importlib.import_module("cleanup")


@pytest.fixture
def prune(monkeypatch, cleanup_mod, tmp_path):
    from provider_codex import CodexAdapter

    adapter = CodexAdapter(home=_codex_home(tmp_path))
    doc_ids = [t["doc_id"] for t in adapter.parse_source(adapter.discover_sources()[0], None).turns]
    session = doc_ids[0].split(":")[1]
    state = {"doc_ids": doc_ids, "session": session, "stored": [], "deleted": [], "results": {}}

    def fake_stored(provider, fields, *, db_path=None):
        state["provider"] = provider
        return {doc_id: [{"timestamp": ""}] for doc_id in state["stored"]}

    def fake_delete(doc_id, db_path=None):
        state["deleted"].append(doc_id)
        return state["results"].get(doc_id, rag_engine.DeleteResult(1, True))

    monkeypatch.setattr(cleanup_mod, "rag_engine", rag_engine)
    monkeypatch.setattr(rag_engine, "stored_metadata_by_provider", fake_stored)
    monkeypatch.setattr(rag_engine, "delete_by_doc_id", fake_delete)
    monkeypatch.setattr(cleanup_mod, "get_db_path", lambda: "/tmp/synthetic/milvus.db")
    monkeypatch.setattr(cleanup_mod, "_codex_adapter", lambda: adapter)
    return state


def _run(cleanup_mod, argv):
    args = cleanup_mod.build_parser().parse_args(argv)
    return cleanup_mod.cmd_prune_codex(args)


def test_prune_codex_dry_run_writes_nothing(cleanup_mod, prune, capsys):
    orphan = f"codex:{prune['session']}:stale"
    prune["stored"] = [*prune["doc_ids"], orphan]

    assert _run(cleanup_mod, ["prune-codex"]) == 0
    assert prune["provider"] == "codex"
    assert prune["deleted"] == []
    out = capsys.readouterr().out
    assert "Orphan turns to delete:       1\n" in out
    assert "dry run" in out.lower()


def test_prune_codex_apply_without_yes_refuses_before_any_read(cleanup_mod, prune, capsys):
    assert _run(cleanup_mod, ["prune-codex", "--apply"]) == 1
    assert "provider" not in prune
    assert prune["deleted"] == []
    assert "--yes" in capsys.readouterr().err


def test_prune_codex_apply_deletes_only_orphans_of_sessions_on_disk(cleanup_mod, prune):
    orphan = f"codex:{prune['session']}:stale"
    gone = "codex:gone-session:x"
    prune["stored"] = [*prune["doc_ids"], orphan, gone]

    assert _run(cleanup_mod, ["prune-codex", "--apply", "--yes"]) == 0
    assert prune["deleted"] == [orphan]


def test_prune_codex_reports_fts_failures(cleanup_mod, prune, capsys):
    orphan = f"codex:{prune['session']}:stale"
    prune["stored"] = [*prune["doc_ids"], orphan]
    prune["results"][orphan] = rag_engine.DeleteResult(1, False)

    assert _run(cleanup_mod, ["prune-codex", "--apply", "--yes"]) == 1
    assert "FTS" in capsys.readouterr().out


def test_prune_codex_skips_a_session_whose_rollout_fails_to_read(cleanup_mod, prune, monkeypatch):
    import provider_codex

    def _fail(path):
        raise OSError("gone")
        yield  # pragma: no cover

    prune["stored"] = [*prune["doc_ids"]]
    monkeypatch.setattr(provider_codex, "iter_rollout_lines", _fail)

    assert _run(cleanup_mod, ["prune-codex", "--apply", "--yes"]) == 0
    assert prune["deleted"] == []
