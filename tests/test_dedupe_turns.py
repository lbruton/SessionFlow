"""SESF-74: cleanup of duplicate and user-only Claude turns already indexed."""

from __future__ import annotations

import types

import pytest

import dedupe_turns

PARTIAL = "User: second question still running tools"
FULL = PARTIAL + "\n\nAssistant: second answer arrived later"


def _row(doc_id, session, turn_index, document, project_root="/p"):
    return {"doc_id": doc_id, "session_id": session, "turn_index": turn_index,
            "document": document, "project_root": project_root}


def _turn(doc_id, session, turn_index, text):
    return {"doc_id": doc_id, "session_id": session, "turn_index": turn_index,
            "text": text, "chunk_type": "turn", "project_root": "/detected"}


@pytest.fixture
def world(monkeypatch):
    """Stub the index scan, the transcript parse and both writers."""
    state = {"rows": [], "parsed": {}, "added": [], "deleted": [], "deferred": set()}

    monkeypatch.setattr(dedupe_turns, "_indexed_rows", lambda db_path: list(state["rows"]))
    monkeypatch.setattr(dedupe_turns, "_parse_sessions",
                        lambda session_ids: {s: state["parsed"][s]
                                             for s in session_ids if s in state["parsed"]})
    monkeypatch.setattr(
        dedupe_turns, "_stored_doc_ids",
        lambda doc_ids, db_path: {d for d in doc_ids
                                  if d not in state["deferred"]
                                  and (d in {t["doc_id"] for t in state["added"]}
                                       or d in {r["doc_id"] for r in state["rows"]})})

    def _add(turns, db_path=None):
        state["added"].extend(turns)
        return len(turns)

    def _delete(doc_id, db_path=None):
        state["deleted"].append(doc_id)
        return types.SimpleNamespace(deleted=1, fts_ok=True)

    monkeypatch.setattr(dedupe_turns.rag_engine, "add_turns", _add)
    monkeypatch.setattr(dedupe_turns.rag_engine, "delete_by_doc_id", _delete)
    return state


@pytest.fixture
def cleanup(monkeypatch):
    import cleanup as module

    monkeypatch.setattr(module, "get_db_path", lambda: "db")
    return module


def _run(cleanup, argv):
    return cleanup.cmd_dedupe_turns(cleanup.build_parser().parse_args(argv))


def test_row_whose_user_part_was_stripped_counts_as_a_reply(world):
    world["rows"] = [_row("s::stripped", "s", 7, "Assistant: a reply whose user part was stripped")]

    plan = dedupe_turns.plan(db_path="db")

    assert plan.user_only_rows == 0
    assert plan.groups == {}


def test_plan_finds_duplicates_and_user_only_rows(world):
    world["rows"] = [
        _row("s::partial", "s", 7, PARTIAL), _row("s::full", "s", 7, FULL),
        _row("u::partial", "u", 3, PARTIAL),
        _row("ok::full", "ok", 1, FULL),
    ]

    plan = dedupe_turns.plan(db_path="db")

    assert plan.duplicate_groups == 1
    assert plan.user_only_rows == 1
    assert plan.affected_sessions == {"s", "u"}


def test_session_on_disk_is_rebuilt_from_its_transcript(world):
    world["rows"] = [_row("u::partial", "u", 3, PARTIAL, project_root="/indexed")]
    world["parsed"] = {"u": [_turn("u::full", "u", 3, FULL), _turn("u::other", "u", 9, FULL)]}

    plan = dedupe_turns.plan(db_path="db")
    result = dedupe_turns.apply(plan, db_path="db")

    assert [t["doc_id"] for t in world["added"]] == ["u::full"]
    assert world["added"][0]["project_root"] == "/indexed"
    # The partial row is deleted by doc_id too, which also clears an FTS-only copy.
    assert world["deleted"] == ["u::partial"]
    assert result.rebuilt_turns == 1


def test_rebuilt_turn_already_indexed_is_not_deleted(world):
    world["rows"] = [_row("s::partial", "s", 7, PARTIAL), _row("s::full", "s", 7, FULL)]
    world["parsed"] = {"s": [_turn("s::full", "s", 7, FULL)]}

    dedupe_turns.apply(dedupe_turns.plan(db_path="db"), db_path="db")

    assert world["deleted"] == ["s::partial"]


def test_deferred_rebuild_keeps_its_old_rows(world):
    world["rows"] = [_row("u::partial", "u", 3, PARTIAL)]
    world["parsed"] = {"u": [_turn("u::full", "u", 3, FULL)]}
    world["deferred"] = {"u::full"}

    result = dedupe_turns.apply(dedupe_turns.plan(db_path="db"), db_path="db")

    assert world["deleted"] == []
    assert result.rebuilt_turns == 0
    assert result.deferred_turns == 1


def test_cli_exits_nonzero_when_rebuilds_are_deferred(world, cleanup):
    world["rows"] = [_row("u::partial", "u", 3, PARTIAL)]
    world["parsed"] = {"u": [_turn("u::full", "u", 3, FULL)]}
    world["deferred"] = {"u::full"}

    assert _run(cleanup, ["dedupe-turns", "--apply", "--yes"]) == 1
    assert world["deleted"] == []


def test_indexed_rows_include_fts_only_rows(monkeypatch):
    milvus = [_row("s::full", "s", 7, FULL)]
    fts = [_row("s::full", "s", 7, FULL), _row("s::orphan", "s", 7, PARTIAL)]
    monkeypatch.setattr(dedupe_turns, "_milvus_rows", lambda db_path: list(milvus))
    monkeypatch.setattr(dedupe_turns, "_fts_rows", lambda db_path: list(fts))

    rows = dedupe_turns._indexed_rows("db")

    assert sorted(r["doc_id"] for r in rows) == ["s::full", "s::orphan"]


def test_session_without_transcript_keeps_the_longest_row(world):
    world["rows"] = [_row("g::partial", "g", 7, PARTIAL), _row("g::full", "g", 7, FULL)]

    plan = dedupe_turns.plan(db_path="db")
    result = dedupe_turns.apply(plan, db_path="db")

    assert world["added"] == []
    assert world["deleted"] == ["g::partial"]
    assert result.deleted_rows == 1


def test_user_only_row_without_transcript_is_kept(world):
    world["rows"] = [_row("g::partial", "g", 7, PARTIAL)]

    result = dedupe_turns.apply(dedupe_turns.plan(db_path="db"), db_path="db")

    assert world["deleted"] == []
    assert result.deleted_rows == 0


def test_turn_missing_from_the_parse_falls_back_to_keep_longest(world):
    world["rows"] = [_row("s::partial", "s", 7, PARTIAL), _row("s::full", "s", 7, FULL)]
    world["parsed"] = {"s": [_turn("s::other", "s", 9, FULL)]}

    dedupe_turns.apply(dedupe_turns.plan(db_path="db"), db_path="db")

    assert world["added"] == []
    assert world["deleted"] == ["s::partial"]


def test_cli_dry_run_writes_nothing(world, cleanup, capsys):
    world["rows"] = [_row("g::partial", "g", 7, PARTIAL), _row("g::full", "g", 7, FULL)]

    code = _run(cleanup, ["dedupe-turns"])

    assert code == 0
    assert world["added"] == world["deleted"] == []
    assert "Dry run" in capsys.readouterr().out


def test_cli_apply_refuses_without_yes(world, cleanup, capsys):
    world["rows"] = [_row("g::partial", "g", 7, PARTIAL), _row("g::full", "g", 7, FULL)]

    code = _run(cleanup, ["dedupe-turns", "--apply"])

    assert code == 1
    assert world["added"] == world["deleted"] == []


def test_cli_apply_with_yes_writes(world, cleanup):
    world["rows"] = [_row("g::partial", "g", 7, PARTIAL), _row("g::full", "g", 7, FULL)]

    assert _run(cleanup, ["dedupe-turns", "--apply", "--yes"]) == 0
    assert world["deleted"] == ["g::partial"]
