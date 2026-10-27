"""MCP surface for memories: seven tools, Markdown or JSON output, structured errors."""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest
from mcp import types

tools = importlib.import_module("tools")

MEMORY_TOOLS = {"add_memory", "search_memories", "list_memories", "get_memory",
                "update_memory", "delete_memory", "get_memory_context"}


class StubError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class StubService:
    """Records calls and returns the structured responses the renderers format."""

    def __init__(self):
        self.calls = []

    async def add(self, args):
        self.calls.append(("add", args))
        return {"memory_id": "m-1", "outcome": "created", "version": 1, "index": "ok",
                "similar": [], "similar_state": "ok"}

    async def get(self, memory_id, project=None):
        self.calls.append(("get", memory_id, project))
        if memory_id == "missing":
            raise StubError("not_found", f"No memory {memory_id}")
        return {"memory_id": memory_id, "text": "hello", "project": "sessionflow",
                "type": "fact", "status": "active", "version": 1}

    async def update(self, memory_id, args):
        self.calls.append(("update", memory_id, args))
        if memory_id == "missing":
            raise StubError("not_found", f"No memory {memory_id}")
        return {"memory_id": memory_id, "outcome": "updated", "version": 2}

    async def delete(self, memory_id, project=None):
        self.calls.append(("delete", memory_id, project))
        if memory_id == "missing":
            raise StubError("not_found", f"No memory {memory_id}")
        return {"memory_id": memory_id, "outcome": "deleted", "version": 2}

    async def list(self, filters, limit, cursor=None):
        self.calls.append(("list", filters, limit, cursor))
        return {"results": [{"memory_id": "m-1", "text": "hello"}], "next_cursor": None}

    async def search(self, query, filters, n):
        self.calls.append(("search", query, filters, n))
        return {"results": [{"memory_id": "m-1", "text": "hello"}], "channels": ["vector", "fts"],
                "incomplete": False}

    async def context(self, memory_id, project=None, limit=10):
        self.calls.append(("context", memory_id, project, limit))
        return {"memory": {"memory_id": memory_id}, "source_session": None,
                "related_sessions": [], "note": "No source session is linked."}


def _register():
    captured = {}

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
    return captured


@pytest.fixture
def service(monkeypatch):
    svc = StubService()
    tools.set_memory_service(svc)
    yield svc
    tools.set_memory_service(None)


def _call(name, arguments):
    return asyncio.run(_register()["call"](name, arguments))


def _text(result):
    content = result.content if isinstance(result, types.CallToolResult) else result
    return content[0].text


def test_seven_memory_tools_are_listed():
    listed = asyncio.run(_register()["list"]())
    names = {t.name for t in listed}
    assert MEMORY_TOOLS <= names
    for tool in listed:
        if tool.name in MEMORY_TOOLS:
            assert "format" in tool.inputSchema["properties"]


@pytest.mark.parametrize("name,args", [
    ("add_memory", {"text": "t", "agent_id": "codex", "project": "sessionflow", "type": "fact"}),
    ("get_memory", {"memory_id": "m-1", "project": "sessionflow"}),
    ("list_memories", {"project": "sessionflow"}),
    ("search_memories", {"query": "q", "project": "sessionflow"}),
    ("update_memory", {"memory_id": "m-1", "text": "new", "project": "sessionflow"}),
    ("delete_memory", {"memory_id": "m-1", "project": "sessionflow"}),
    ("get_memory_context", {"memory_id": "m-1", "project": "sessionflow"}),
])
def test_each_tool_renders_markdown_by_default_and_json_on_request(service, name, args):
    markdown = _text(_call(name, dict(args)))
    assert not markdown.lstrip().startswith("{")
    assert "m-1" in markdown

    payload = json.loads(_text(_call(name, dict(args, format="json"))))
    assert isinstance(payload, dict)
    assert "error" not in payload


@pytest.mark.parametrize("name", ["get_memory", "update_memory", "delete_memory"])
def test_unknown_id_is_a_structured_error(service, name):
    args = {"memory_id": "missing", "project": "sessionflow", "format": "json"}
    if name == "update_memory":
        args["text"] = "x"
    result = _call(name, args)
    assert isinstance(result, types.CallToolResult) and result.isError is True
    body = json.loads(_text(result))
    assert body["error"]["code"] == "not_found"
    assert "missing" in body["error"]["message"]


def test_markdown_error_also_sets_is_error(service):
    result = _call("get_memory", {"memory_id": "missing", "project": "sessionflow"})
    assert isinstance(result, types.CallToolResult) and result.isError is True
    assert "not_found" in _text(result)


def test_tools_report_unavailable_before_memory_starts():
    tools.set_memory_service(None)
    result = _call("list_memories", {"project": "sessionflow", "format": "json"})
    assert result.isError is True
    assert json.loads(_text(result))["error"]["code"] == "memory_unavailable"


def test_search_all_sessions_accepts_json_format(monkeypatch):
    async def fake_search_async(query, n=5, **kwargs):
        return [{"doc_id": "d1", "content": "turn text", "timestamp": "2026-10-01T00:00:00Z",
                 "session_id": "s", "provider": "codex", "project_root": "/p"}]

    monkeypatch.setattr(tools.rag_engine, "search_async", fake_search_async)
    payload = json.loads(_text(_call("search_all_sessions",
                                     {"query": "q", "project_root": "*", "format": "json"})))
    assert payload["results"][0]["doc_id"] == "d1"


def test_bind_address_and_routes(stub_rag_engine):
    import sys

    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    assert http_server.HOST == "127.0.0.1" and http_server.PORT == 7102
    paths = {getattr(r, "path", None) for r in http_server.app.routes}
    assert paths == {"/health", "/backfill", "/index", "/watch", "/timeline", "/mcp",
                     "/memory/restore-check"}
