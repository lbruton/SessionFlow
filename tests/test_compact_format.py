"""SESF-89: format='compact' prints one line per hit for search_all_sessions and search_session."""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest

from project_scope import ProjectScope

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")

ROOT = "/Volumes/DATA/GitHub/SessionFlow"
SESSION = "4a9320b2-656d-47fc-812f-ca5ae824d086"
REPLY_BODY = "the reply body that compact output leaves out"


def _row(content=f"User: why did the deploy fail\n\nAssistant: {REPLY_BODY}", **extra) -> dict:
    return {
        "session_id": SESSION,
        "turn_index": 29375,
        "timestamp": "2026-11-01T23:20:52.445Z",
        "provider": "claude_code_cli",
        "project_root": ROOT,
        "git_branch": "main",
        "chunk_type": "turn",
        "content": content,
        **extra,
    }


def _lines(rows):
    return tools.format_results_compact(rows, ProjectScope(roots=[ROOT])).splitlines()


def test_compact_has_no_reply_and_keeps_scope_line():
    lines = _lines([_row()])
    assert lines[0].startswith("Scope: ")
    assert REPLY_BODY not in "\n".join(lines)
    assert "why did the deploy fail" in lines[-1]


def test_each_hit_is_one_line_with_full_ids_and_metadata():
    lines = _lines([_row(), _row(turn_index=7)])
    hits = [line for line in lines if SESSION in line]
    assert len(hits) == 2
    assert f"session:{SESSION} turn:29375" in hits[0]
    assert f"session:{SESSION} turn:7" in hits[1]
    assert "2026-11-01T23:20:52" in hits[0]
    assert "claude_code_cli" in hits[0]
    assert "project:SessionFlow" in hits[0]
    assert ROOT not in hits[0]


def test_preview_is_cut_at_150_characters_on_one_line():
    line = _lines([_row(content=f"User: first\nsecond   {'x' * 300}\n\nAssistant: reply")])[-1]
    preview = line.split(" | ", 1)[1]
    assert preview.startswith("first second x")
    assert len(preview) == tools.COMPACT_PREVIEW_CHARS == 150


def test_empty_user_text_falls_back_to_marked_assistant_preview():
    line = _lines([_row(content="User: \n\nAssistant: only a reply")])[-1]
    assert line.endswith("| assistant: only a reply")


def test_reply_only_turn_is_marked_assistant():
    line = _lines([_row(content="Assistant: only a reply")])[-1]
    assert line.endswith("| assistant: only a reply")


def test_mid_turn_passage_is_marked_with_its_part():
    row = _row(content="text from the middle of a long reply", chunk_type="turn_passage",
               passage_index=2)
    assert _lines([row])[-1].endswith("| part 3: text from the middle of a long reply")


def test_first_passage_shows_the_user_text():
    row = _row(content="User: the question\n\nAssistant: start of the reply",
               chunk_type="turn_passage", passage_index=0)
    assert _lines([row])[-1].endswith("| the question")


def test_row_without_a_prefix_prints_its_text():
    assert _lines([_row(content="plain planner note")])[-1].endswith("| plain planner note")


def test_no_results():
    assert _lines([])[-1] == "No results found."


def test_markdown_is_unchanged():
    scope = ProjectScope(roots=[ROOT])
    rendered = tools.format_results([_row()], scope)
    assert REPLY_BODY in rendered
    assert "**Result 1**" in rendered


def _register():
    captured: dict = {}

    class _Server:
        def list_tools(self):
            def deco(fn):
                captured["list"] = fn
                return fn
            return deco

        def call_tool(self):
            def deco(fn):
                captured["call"] = fn
                return fn
            return deco

    tools.register_tools(_Server())
    return {t.name: t for t in asyncio.run(captured["list"]())}, captured["call"]


@pytest.fixture
def engine(monkeypatch):
    async def _search_async(query, n=5, **kwargs):
        return [_row()]

    monkeypatch.setattr(rag_engine, "known_project_roots", lambda db_path=None: [ROOT])
    monkeypatch.setattr(rag_engine, "search_async", _search_async)
    monkeypatch.setattr(tools, "get_current_project_root", lambda: None)


TOOLS = ["search_all_sessions", "search_session"]


@pytest.mark.parametrize("name", TOOLS)
def test_tools_print_compact_output(engine, name):
    _, call = _register()
    text = asyncio.run(call(name, {"project_root": ROOT, "format": "compact"}))[0].text
    assert text.startswith("Scope: ")
    assert REPLY_BODY not in text
    assert f"session:{SESSION} turn:29375" in text


@pytest.mark.parametrize("name", TOOLS)
def test_tools_default_to_markdown(engine, name):
    _, call = _register()
    text = asyncio.run(call(name, {"project_root": ROOT}))[0].text
    assert REPLY_BODY in text


@pytest.mark.parametrize("name", TOOLS)
def test_tools_print_json(engine, name):
    _, call = _register()
    text = asyncio.run(call(name, {"project_root": ROOT, "format": "json"}))[0].text
    assert json.loads(text)["results"][0]["session_id"] == SESSION


@pytest.mark.parametrize("name", TOOLS)
def test_schemas_offer_compact_and_name_get_turns(name):
    listed, _ = _register()
    tool = listed[name]
    assert tool.inputSchema["properties"]["format"]["enum"] == ["compact", "json", "markdown"]
    assert "get_turns" in tool.inputSchema["properties"]["format"]["description"]


def test_unprefixed_assistant_row_gets_marker():
    line = _lines([_row(content="only a reply", chunk_type="assistant")])[-1]
    assert line.endswith("| assistant: only a reply")


def test_unprefixed_user_row_stays_unmarked():
    line = _lines([_row(content="only a question", chunk_type="user")])[-1]
    assert line.endswith("| only a question")
