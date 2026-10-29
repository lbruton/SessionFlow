"""SESF-85: transcript read tools take project_root and refuse an unnamed scope.

A client that sends no X-Project-Root header (Claude Desktop, Codex) must name
the project in the call, or pass '*' on purpose. Otherwise the tool refuses
instead of searching every project.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from project_scope import resolve_project_scope

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")

ROOT = "/Volumes/DATA/GitHub/SessionFlow"
OTHER = "/Volumes/DATA/GitHub/StakTrakr"
KNOWN = [ROOT, OTHER]
SCOPED_TOOLS = ["search_all_sessions", "search_session", "get_session_stats"]


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
    listed = asyncio.run(captured["list"]())
    return {t.name: t for t in listed}, captured["call"]


@pytest.fixture
def engine(monkeypatch):
    """Stub the engine; record the project filter each call receives."""
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


def _call(name, arguments, header=None, monkeypatch=None):
    _, call = _register()
    monkeypatch.setattr(tools, "get_current_project_root", lambda: header)
    out = asyncio.run(call(name, arguments))
    return out[0].text


# ---------------------------------------------------------------------------
# resolve_project_scope marks an unnamed scope
# ---------------------------------------------------------------------------

def test_no_arg_and_no_header_is_unnamed():
    scope = resolve_project_scope(None, None, KNOWN)
    assert scope.unnamed is True
    assert scope.roots is None


def test_star_is_not_unnamed():
    assert resolve_project_scope("*", None, KNOWN).unnamed is False


def test_header_scope_is_not_unnamed():
    assert resolve_project_scope(None, ROOT, KNOWN).unnamed is False


def test_named_miss_is_not_unnamed():
    scope = resolve_project_scope("Nope", None, KNOWN)
    assert scope.roots is None
    assert scope.unnamed is False


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_schema_offers_project_root(name):
    listed, _ = _register()
    props = listed[name].inputSchema["properties"]
    assert "project_root" in props
    assert "working directory" in props["project_root"]["description"]


@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_description_asks_for_the_working_directory(name):
    listed, _ = _register()
    assert "working directory" in listed[name].description


def test_scoped_tools_share_one_project_root_description():
    listed, _ = _register()
    descs = {listed[n].inputSchema["properties"]["project_root"]["description"]
             for n in SCOPED_TOOLS}
    assert len(descs) == 1


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_unnamed_scope_is_refused(name, engine, monkeypatch):
    text = _call(name, {"query": "x"}, header=None, monkeypatch=monkeypatch)
    assert text.startswith("No project named.")
    assert "'*'" in text
    assert engine == {}


@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_explicit_project_root_is_honoured(name, engine, monkeypatch):
    text = _call(name, {"query": "x", "project_root": OTHER}, header=ROOT,
                 monkeypatch=monkeypatch)
    assert text.startswith(f"Scope: {OTHER}")
    assert list(engine.values()) == [[OTHER]]


@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_header_applies_when_arg_is_missing(name, engine, monkeypatch):
    text = _call(name, {"query": "x"}, header=ROOT, monkeypatch=monkeypatch)
    assert text.startswith(f"Scope: {ROOT}")
    assert list(engine.values()) == [[ROOT]]


@pytest.mark.parametrize("name", SCOPED_TOOLS)
def test_star_searches_every_project_without_a_header(name, engine, monkeypatch):
    text = _call(name, {"query": "x", "project_root": "*"}, header=None,
                 monkeypatch=monkeypatch)
    assert text.startswith("Scope: all projects")
    assert list(engine.values()) == [None]


def test_search_session_with_session_id_needs_no_project(engine, monkeypatch):
    text = _call("search_session", {"query": "x", "session_id": "abc"}, header=None,
                 monkeypatch=monkeypatch)
    assert not text.startswith("No project named.")
    assert engine["search"] is None
