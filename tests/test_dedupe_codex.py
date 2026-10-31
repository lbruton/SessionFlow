"""SESF-98: remove Codex rows left by the line-number turn keys.

* A session whose rollout is on disk keeps only the rows its parse yields, once
  every parsed turn is indexed; until then nothing of it is deleted.
* A session whose rollout is gone keeps one row per distinct turn: exact
  repeats and assistant-only copies of a User + Assistant row go.
* Dry run by default; ``--apply`` refuses without ``--yes`` before any read.
"""

from __future__ import annotations

import json
import os
import stat
import types

import pytest

import dedupe_codex

FULL = "User: q\n\nAssistant: the reply"


def _row(doc_id, session, turn_index, document, in_milvus=True):
    return {"doc_id": doc_id, "session_id": session, "turn_index": turn_index,
            "document": document, "in_milvus": in_milvus}


def _turn(doc_id, session, turn_index, text):
    return {"doc_id": doc_id, "session_id": session, "turn_index": turn_index,
            "text": text, "chunk_type": "turn"}


@pytest.fixture
def world(monkeypatch, tmp_path):
    state = {"rows": [], "parsed": {}, "deleted": [], "fts_fail": set(), "reads": 0}

    def _rows(db_path):
        state["reads"] += 1
        return list(state["rows"])

    def _delete(doc_id, db_path=None):
        state["deleted"].append(doc_id)
        return types.SimpleNamespace(deleted=1, fts_ok=doc_id not in state["fts_fail"])

    monkeypatch.setattr(dedupe_codex, "_indexed_rows", _rows)
    monkeypatch.setattr(dedupe_codex, "_parse_sessions", lambda: dict(state["parsed"]))
    monkeypatch.setattr(dedupe_codex.rag_engine, "delete_by_doc_id", _delete)
    monkeypatch.setattr(dedupe_codex, "_audit_dir", lambda: tmp_path / "audit")
    state["audit_dir"] = tmp_path / "audit"
    return state


@pytest.fixture
def cleanup(monkeypatch):
    import cleanup as module

    monkeypatch.setattr(module, "get_db_path", lambda: "db")
    return module


def _run(cleanup, argv):
    return cleanup.cmd_dedupe_codex(cleanup.build_parser().parse_args(argv))


def test_session_on_disk_keeps_only_parsed_rows(world):
    world["rows"] = [_row("codex:s:new", "s", 0, FULL), _row("codex:s:old1", "s", 230, FULL),
                     _row("codex:s:old2", "s", 233, "the reply")]
    world["parsed"] = {"s": [_turn("codex:s:new", "s", 0, FULL)]}

    plan = dedupe_codex.plan(db_path="db")
    result = dedupe_codex.apply(plan, db_path="db")

    assert sorted(world["deleted"]) == ["codex:s:old1", "codex:s:old2"]
    assert result.deleted_rows == 2


def test_session_on_disk_waits_until_every_parsed_turn_is_indexed(world):
    world["rows"] = [_row("codex:s:new", "s", 0, FULL), _row("codex:s:old", "s", 230, FULL)]
    world["parsed"] = {"s": [_turn("codex:s:new", "s", 0, FULL),
                             _turn("codex:s:new2", "s", 1, "User: q2\n\nAssistant: a2 long enough")]}

    plan = dedupe_codex.plan(db_path="db")
    dedupe_codex.apply(plan, db_path="db")

    assert world["deleted"] == []
    assert plan.waiting_sessions == 1
    assert plan.waiting_turns == 1


def test_fts_only_copy_does_not_count_as_indexed(world):
    world["rows"] = [_row("codex:s:new", "s", 0, FULL, in_milvus=False),
                     _row("codex:s:old", "s", 230, FULL)]
    world["parsed"] = {"s": [_turn("codex:s:new", "s", 0, FULL)]}

    plan = dedupe_codex.plan(db_path="db")

    assert plan.waiting_sessions == 1
    assert plan.stale == {}


def test_parsed_turn_that_stripping_drops_is_not_waited_for(world):
    dropped = "<system-reminder>\nonly harness text\n</system-reminder>"
    world["rows"] = [_row("codex:s:new", "s", 0, FULL), _row("codex:s:old", "s", 230, FULL)]
    world["parsed"] = {"s": [_turn("codex:s:new", "s", 0, FULL),
                             _turn("codex:s:empty", "s", 1, dropped)]}

    plan = dedupe_codex.plan(db_path="db")

    assert plan.waiting_sessions == 0
    assert plan.stale == {"s": ["codex:s:old"]}


def test_gone_session_keeps_one_row_per_distinct_turn(world):
    world["rows"] = [
        _row("codex:g:a", "g", 230, FULL),
        _row("codex:g:b", "g", 233, FULL),
        _row("codex:g:c", "g", 231, "the reply"),
        _row("codex:g:d", "g", 5, "User: other\n\nAssistant: something else"),
    ]

    plan = dedupe_codex.plan(db_path="db")
    dedupe_codex.apply(plan, db_path="db")

    assert sorted(world["deleted"]) == ["codex:g:b", "codex:g:c"]


def test_apply_writes_a_private_audit_file(world):
    world["rows"] = [_row("codex:g:a", "g", 1, FULL), _row("codex:g:b", "g", 2, FULL)]

    result = dedupe_codex.apply(dedupe_codex.plan(db_path="db"), db_path="db")

    lines = [json.loads(line) for line in open(result.audit_path)]
    assert lines == [{"doc_id": "codex:g:b", "session_id": "g", "reason": "duplicate"}]
    assert stat.S_IMODE(os.stat(result.audit_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(world["audit_dir"]).st_mode) == 0o700


def test_cli_dry_run_writes_nothing(world, cleanup, capsys):
    world["rows"] = [_row("codex:g:a", "g", 1, FULL), _row("codex:g:b", "g", 2, FULL)]

    assert _run(cleanup, ["dedupe-codex"]) == 0
    assert world["deleted"] == []
    assert not world["audit_dir"].exists()
    assert "Dry run" in capsys.readouterr().out


def test_cli_apply_refuses_without_yes_before_any_read(world, cleanup, capsys):
    world["rows"] = [_row("codex:g:a", "g", 1, FULL), _row("codex:g:b", "g", 2, FULL)]

    assert _run(cleanup, ["dedupe-codex", "--apply"]) == 1
    assert world["reads"] == 0
    assert "--yes" in capsys.readouterr().err


def test_cli_apply_deletes_and_reports_fts_failures(world, cleanup, capsys):
    world["rows"] = [_row("codex:g:a", "g", 1, FULL), _row("codex:g:b", "g", 2, FULL)]
    world["fts_fail"] = {"codex:g:b"}

    assert _run(cleanup, ["dedupe-codex", "--apply", "--yes"]) == 1
    assert world["deleted"] == ["codex:g:b"]
    assert "FTS" in capsys.readouterr().out


def test_cli_exits_nonzero_while_sessions_wait_for_backfill(world, cleanup, capsys):
    world["rows"] = [_row("codex:s:old", "s", 230, FULL)]
    world["parsed"] = {"s": [_turn("codex:s:new", "s", 0, FULL)]}

    assert _run(cleanup, ["dedupe-codex", "--apply", "--yes"]) == 1
    assert world["deleted"] == []
    assert "backfill" in capsys.readouterr().out


def test_unreadable_session_is_left_alone(world):
    world["rows"] = [_row("codex:s:old", "s", 230, FULL), _row("codex:s:old2", "s", 233, FULL)]
    world["parsed"] = {"s": None}

    plan = dedupe_codex.plan(db_path="db")

    assert plan.stale == {} and plan.duplicates == {}
    assert plan.unreadable_sessions == 1


def test_gone_session_keeps_the_milvus_copy_of_a_repeat(world):
    world["rows"] = [_row("codex:g:fts", "g", 1, FULL, in_milvus=False),
                     _row("codex:g:vec", "g", 2, FULL)]

    plan = dedupe_codex.plan(db_path="db")

    assert plan.duplicates == {"g": ["codex:g:fts"]}


def test_assistant_only_vector_row_needs_a_vector_backed_full_row(world):
    world["rows"] = [_row("codex:g:full", "g", 1, FULL, in_milvus=False),
                     _row("codex:g:tail", "g", 2, "the reply")]

    plan = dedupe_codex.plan(db_path="db")

    assert plan.duplicates == {}
