"""SESF-86: the read tools take the current project from the client's MCP roots.

Order: an explicit project_root, then the client's first file:// root, then the
X-Project-Root header, then refusal. Roots are asked once per MCP session and
forgotten on notifications/roots/list_changed.
"""

from __future__ import annotations

import asyncio
import importlib

import anyio
import pytest
from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session

client_roots = importlib.import_module("client_roots")
rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")

ROOT = "/Volumes/DATA/GitHub/SessionFlow"
OTHER = "/Volumes/DATA/GitHub/StakTrakr"
KNOWN = [ROOT, OTHER]


@pytest.fixture(autouse=True)
def _clear_cache():
    client_roots.forget_all()
    yield
    client_roots.forget_all()


# ---------------------------------------------------------------------------
# URI to path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uri, expected", [
    ("file:///Volumes/DATA/GitHub/SessionFlow", ROOT),
    ("file:///Volumes/DATA/GitHub/SessionFlow/", ROOT),
    ("file:///Users/me/My%20Project", "/Users/me/My Project"),
    ("file://localhost/srv/app", "/srv/app"),
    ("file:///", "/"),
    ("https://example.com/repo", None),
    ("file://otherhost/srv/app", None),
    ("", None),
])
def test_root_path(uri, expected):
    assert client_roots.root_path(uri) == expected


def test_first_file_root_wins():
    roots = [
        types.Root.model_construct(uri="https://example.com/x"),
        types.Root(uri="file:///Volumes/DATA/GitHub/SessionFlow"),
        types.Root(uri="file:///Volumes/DATA/GitHub/StakTrakr"),
    ]
    assert client_roots.first_root_path(roots) == ROOT


def test_no_file_root_gives_none():
    assert client_roots.first_root_path([types.Root.model_construct(uri="https://example.com/x")]) is None
    assert client_roots.first_root_path([]) is None


# ---------------------------------------------------------------------------
# Session handling with a fake session
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, roots=None, supports=True, hang=False, fail=False):
        self.roots = roots or []
        self.supports = supports
        self.hang = hang
        self.fail = fail
        self.requests = 0
        self.related = []

    def check_client_capability(self, capability):
        return self.supports

    async def send_request(self, request, result_type, metadata=None, **_kw):
        self.requests += 1
        self.related.append(getattr(metadata, "related_request_id", None))
        if self.hang:
            await anyio.sleep(60)
        if self.fail:
            raise RuntimeError("client went away")
        return types.ListRootsResult(roots=[types.Root(uri=u) for u in self.roots])


def _ask(session, request_id=7):
    return asyncio.run(client_roots.project_root(session, request_id))


def test_asks_once_per_session_on_the_calling_request():
    session = _FakeSession(roots=["file:///Volumes/DATA/GitHub/SessionFlow"])
    assert _ask(session, 7) == ROOT
    assert _ask(session, 8) == ROOT
    assert session.requests == 1
    assert session.related == [7]


def test_client_without_roots_support_is_not_asked():
    session = _FakeSession(roots=["file:///x"], supports=False)
    assert _ask(session) is None
    assert session.requests == 0


def test_timeout_gives_none_and_is_not_retried(monkeypatch):
    monkeypatch.setattr(client_roots, "ROOTS_TIMEOUT_S", 0.05)
    session = _FakeSession(hang=True)
    assert _ask(session) is None
    assert _ask(session) is None
    assert session.requests == 1


def test_error_gives_none():
    session = _FakeSession(fail=True)
    assert _ask(session) is None


def test_list_changed_forgets_cached_roots():
    session = _FakeSession(roots=["file:///Volumes/DATA/GitHub/SessionFlow"])
    assert _ask(session) == ROOT
    session.roots = ["file:///Volumes/DATA/GitHub/StakTrakr"]
    asyncio.run(client_roots.handle_list_changed(types.RootsListChangedNotification(
        method="notifications/roots/list_changed")))
    assert _ask(session) == OTHER
    assert session.requests == 2


def test_no_request_context_gives_none():
    class _NoContext:
        @property
        def request_context(self):
            raise LookupError("no request")

    assert asyncio.run(client_roots.current_project_root(_NoContext())) is None


# ---------------------------------------------------------------------------
# Order of sources in the read tools, over a real in-memory MCP session
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(monkeypatch):
    seen: dict = {}

    async def _search_async(query, n=5, **kwargs):
        seen["search"] = kwargs.get("project_root")
        return []

    def _get_stats(project_root=None, db_path=None):
        seen["stats"] = project_root
        return {"total_turns": 0, "sessions": 0}

    monkeypatch.setattr(rag_engine, "known_project_roots", lambda db_path=None: KNOWN)
    monkeypatch.setattr(rag_engine, "search_async", _search_async)
    monkeypatch.setattr(rag_engine, "get_stats", _get_stats)
    return seen


def _server():
    server = Server("sessionflow-test")
    tools.register_tools(server)
    client_roots.register(server)
    return server


def _roundtrip(name, arguments, roots, header=None, monkeypatch=None):
    """Call a tool through a real client session that answers roots/list."""
    asked = {"n": 0}
    monkeypatch.setattr(tools, "get_current_project_root", lambda: header)

    async def _list_roots(context):
        asked["n"] += 1
        return types.ListRootsResult(roots=[types.Root(uri=u) for u in roots])

    async def _run():
        callback = _list_roots if roots is not None else None
        async with create_connected_server_and_client_session(
                _server(), list_roots_callback=callback) as client:
            result = await client.call_tool(name, arguments)
            return result.content[0].text

    return anyio.run(_run), asked["n"]


@pytest.mark.parametrize("name", ["search_all_sessions", "search_session", "get_session_stats"])
def test_roots_scope_a_call_with_no_project(name, engine, monkeypatch):
    text, asked = _roundtrip(name, {}, ["file:///Volumes/DATA/GitHub/SessionFlow"],
                             monkeypatch=monkeypatch)
    assert asked == 1
    assert text.startswith(f"Scope: {ROOT}")
    assert "client roots" in text.splitlines()[0]
    assert (engine.get("search") or engine.get("stats")) == [ROOT]


def test_explicit_argument_beats_roots(engine, monkeypatch):
    text, asked = _roundtrip("search_all_sessions", {"project_root": OTHER},
                             ["file:///Volumes/DATA/GitHub/SessionFlow"], monkeypatch=monkeypatch)
    assert asked == 0
    assert engine["search"] == [OTHER]


def test_star_beats_roots(engine, monkeypatch):
    text, asked = _roundtrip("search_all_sessions", {"project_root": "*"},
                             ["file:///Volumes/DATA/GitHub/SessionFlow"], monkeypatch=monkeypatch)
    assert asked == 0
    assert engine["search"] is None


def test_roots_beat_the_header(engine, monkeypatch):
    _roundtrip("search_all_sessions", {}, ["file:///Volumes/DATA/GitHub/SessionFlow"],
               header=OTHER, monkeypatch=monkeypatch)
    assert engine["search"] == [ROOT]


def test_header_used_when_client_has_no_roots(engine, monkeypatch):
    text, _ = _roundtrip("search_all_sessions", {}, None, header=OTHER, monkeypatch=monkeypatch)
    assert engine["search"] == [OTHER]
    assert "client roots" not in text.splitlines()[0]


def test_empty_roots_and_no_header_refuse(engine, monkeypatch):
    text, asked = _roundtrip("search_all_sessions", {}, [], monkeypatch=monkeypatch)
    assert asked == 1
    assert text == tools.UNNAMED_SCOPE_ERROR
    assert "search" not in engine


def test_search_session_with_session_id_treats_roots_like_the_header(engine, monkeypatch):
    _, asked = _roundtrip("search_session", {"session_id": "abc", "query": "x"},
                          ["file:///Volumes/DATA/GitHub/SessionFlow"], monkeypatch=monkeypatch)
    assert asked == 1
    assert engine["search"] == [ROOT]


# ---------------------------------------------------------------------------
# The HTTP server runs stateful sessions so it can ask the client
# ---------------------------------------------------------------------------

def test_http_server_uses_stateful_streamed_sessions():
    http_server = importlib.import_module("http_server")
    manager = http_server.session_manager
    assert manager.stateless is False
    assert manager.json_response is False
    assert types.RootsListChangedNotification in http_server.mcp_server.notification_handlers


# ---------------------------------------------------------------------------
# The header fallback reads the current HTTP request, not the session's context
# ---------------------------------------------------------------------------

class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class _StubServer:
    """A server whose request context carries this call's HTTP request."""

    def __init__(self, header):
        class _Request:
            headers = _Headers({"x-project-root": header} if header else {})

        class _Ctx:
            session = _FakeSession(supports=False)
            request_id = 1
            request = _Request()

        self._ctx = _Ctx()
        self.captured = {}

    @property
    def request_context(self):
        return self._ctx

    def list_tools(self):
        def deco(fn):
            self.captured["list"] = fn
            return fn
        return deco

    def call_tool(self):
        def deco(fn):
            self.captured["call"] = fn
            return fn
        return deco


def test_header_comes_from_the_tool_calls_own_request(engine, monkeypatch):
    # The ContextVar holds the initialize request's header in a stateful session.
    monkeypatch.setattr(tools, "get_current_project_root", lambda: ROOT)
    server = _StubServer(OTHER)
    tools.register_tools(server)
    asyncio.run(server.captured["call"]("search_all_sessions", {}))
    assert engine["search"] == [OTHER]


def test_no_header_on_the_tool_call_means_no_header(engine, monkeypatch):
    monkeypatch.setattr(tools, "get_current_project_root", lambda: ROOT)
    server = _StubServer(None)
    tools.register_tools(server)
    out = asyncio.run(server.captured["call"]("search_all_sessions", {}))
    assert out[0].text == tools.UNNAMED_SCOPE_ERROR
