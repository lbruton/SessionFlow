"""Search usage log: off by default, redacts queries, records search and follow-up calls."""

import asyncio
import importlib
import json
import stat

import pytest

import search_log

tools = importlib.import_module("tools")

GITHUB_TOKEN = "ghp_" + "A1b2C3d4" * 4 + "Zz9Y"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SESSIONFLOW_SEARCH_LOG", raising=False)
    return tmp_path


def _lines(home):
    path = home / ".sessionflow" / "search-log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_off_by_default_writes_nothing(home):
    search_log.record_search("search_memories", client="c", project="p", sort_by="hybrid",
                             query="anything", ids=["m1"])
    search_log.record_follow_up("get_memory", client="c", item_id="m1")
    assert not (home / ".sessionflow" / "search-log.jsonl").exists()
    assert not (home / ".sessionflow").exists()


@pytest.mark.parametrize("value", ["0", "off", "false", ""])
def test_falsy_values_keep_it_off(home, monkeypatch, value):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", value)
    assert search_log.enabled() is False


def test_on_redacts_the_query_and_keeps_ids_in_order(home, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", "1")
    search_log.record_search("search_all_sessions", client="c", project="/repo", sort_by="relevance",
                             query=f"why did {GITHUB_TOKEN} leak", ids=["s:3", "s:1", "s:2"])
    raw = (home / ".sessionflow" / "search-log.jsonl").read_text()
    assert GITHUB_TOKEN not in raw
    (line,) = _lines(home)
    assert line["kind"] == "search"
    assert line["tool"] == "search_all_sessions"
    assert line["ids"] == ["s:3", "s:1", "s:2"]
    assert line["project"] == "/repo" and line["sort_by"] == "relevance"
    assert "[REDACTED:" in line["query"]
    assert line["ts"].endswith("+00:00")


def test_log_file_is_private(home, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", "1")
    search_log.record_follow_up("get_turns", client="c", item_id="s:1")
    path = home / ".sessionflow" / "search-log.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _lines(home)[0] == {**_lines(home)[0], "kind": "follow_up", "tool": "get_turns", "id": "s:1"}


def test_write_failure_does_not_raise(home, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", "1")
    (home / ".sessionflow").write_text("not a directory")
    search_log.record_follow_up("get_memory", client=None, item_id="m1")


# --- Hooks in the MCP tools ----------------------------------------------------


class StubService:
    async def search(self, query, filters, n, sort_by="hybrid"):
        return {"results": [{"memory_id": "m-2"}, {"memory_id": "m-1"}], "channels": ["fts"],
                "incomplete": False}

    async def get(self, memory_id, project=None):
        return {"memory_id": memory_id, "text": "t"}


def _call(name, arguments):
    captured = {}

    class _Server:
        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            def deco(fn):
                captured["call"] = fn
                return fn
            return deco

    tools.register_tools(_Server())
    return asyncio.run(captured["call"](name, arguments))


@pytest.fixture
def service():
    tools.set_memory_service(StubService())
    yield
    tools.set_memory_service(None)


def test_memory_search_and_get_are_logged(home, monkeypatch, service):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", "1")
    _call("search_memories", {"query": "queue drain", "project": "sessionflow"})
    _call("get_memory", {"memory_id": "m-1", "project": "sessionflow"})
    search, follow = _lines(home)
    assert search["tool"] == "search_memories" and search["ids"] == ["m-2", "m-1"]
    assert search["project"] == "sessionflow" and search["sort_by"] == "hybrid"
    assert follow == {**follow, "kind": "follow_up", "tool": "get_memory", "id": "m-1"}


def test_get_turns_is_logged_as_a_turn_id(home, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_SEARCH_LOG", "1")
    monkeypatch.setattr(tools.rag_engine, "get_turns", lambda *a, **k: [])
    _call("get_turns", {"session_id": "abc", "turn_index": 42})
    (line,) = _lines(home)
    assert line["tool"] == "get_turns" and line["id"] == "abc:42"


def test_tools_write_nothing_when_off(home, service):
    _call("search_memories", {"query": "queue drain", "project": "sessionflow"})
    assert _lines(home) == []
