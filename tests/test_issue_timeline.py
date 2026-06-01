"""SESF-25/SESF-26 red-phase tests for the cross-harness issue timeline.

Per design.md Component 5::

    get_issue_timeline(issue_id, *, limit=50, providers=None,
                       date_from=None, date_to=None, db_path=None) -> List[Dict]

Algorithm: query a structured source (Milvus ``_query_all`` filtered
``issue_ids like "%,ID,%"``) AND an FTS keyword fallback (``_fts.search`` on the
literal token), merge and dedup by ``doc_id``, sort by ``(timestamp asc,
doc_id asc)``, then apply optional ``providers`` subset, ``date_from``/``date_to``
bounds, and a ``limit`` (default 50). An empty match returns ``[]``. A
``get_issue_timeline_async`` wrapper mirrors ``search_async``. The MCP tool
``get_issue_timeline`` and the HTTP ``GET /timeline`` route return an equivalent
contract.

Covers Requirements 4.1 (order), 4.2/6.2 (dedup by doc_id), 4.3 (date bounds),
4.4 (provider subset), 4.5 (limit), 4.7 (empty feed), 5.1/5.2/5.3 (MCP + HTTP
equivalence), 6.1/6.3 (structured + FTS merge, boundary-aware).
"""

from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
from contextlib import contextmanager

import pytest

rag_engine = importlib.import_module("rag_engine")


def _entry(doc_id, ts, provider="claude_code_cli", text=None, role="user"):
    """A timeline-shaped row (the union shape both sources hydrate to)."""
    return {
        "doc_id": doc_id,
        "timestamp": ts,
        "provider": provider,
        "session_id": f"sess-{doc_id}",
        "role": role,
        "chunk_type": role,
        "text": text if text is not None else f"text {doc_id}",
        "issue_ids": ",SESF-25,",
    }


def _patch_sources(monkeypatch, structured, fts):
    """Patch the structured (_query_all) and FTS (_fts.search) timeline sources.

    Both are patched to return pre-shaped entry dicts so the test exercises the
    engine's merge/dedup/sort/limit logic rather than Milvus/FTS internals.
    """
    monkeypatch.setattr(rag_engine, "_query_all", lambda *a, **kw: [dict(r) for r in structured])
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [dict(r) for r in fts])

    @contextmanager
    def _fake_client(db_path=None):
        class _C:
            def query(self, *a, **kw):
                return [dict(r) for r in structured]

            def search(self, *a, **kw):
                raise AssertionError("vector search must not run for timeline")
        yield _C()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)


DB = os.path.join(tempfile.gettempdir(), "sessionflow-timeline-test.db")


# ---------------------------------------------------------------------------
# Engine: merge + dedup + order
# ---------------------------------------------------------------------------

def test_timeline_merges_structured_and_fts_sources(monkeypatch):
    # Req 6.1 — feed unions the structured field hits with the FTS fallback hits.
    structured = [_entry("a", "2026-05-01T10:00:00")]
    fts = [_entry("b", "2026-05-02T10:00:00")]
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert {r["doc_id"] for r in out} == {"a", "b"}


def test_timeline_dedups_by_doc_id(monkeypatch):
    # Req 4.2 / 6.2 — a turn matched by both sources appears once.
    structured = [_entry("dup", "2026-05-01T10:00:00")]
    fts = [_entry("dup", "2026-05-01T10:00:00")]
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert [r["doc_id"] for r in out] == ["dup"]


def test_timeline_sorted_by_timestamp_then_doc_id(monkeypatch):
    # Req 4.1 — oldest first; doc_id is the deterministic secondary sort key.
    structured = [
        _entry("z", "2026-05-03T10:00:00"),
        _entry("m", "2026-05-01T10:00:00"),
    ]
    fts = [
        _entry("a", "2026-05-01T10:00:00"),  # same ts as "m" → doc_id breaks tie
        _entry("q", "2026-05-02T10:00:00"),
    ]
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert [r["doc_id"] for r in out] == ["a", "m", "q", "z"]


def test_timeline_empty_feed_when_no_match(monkeypatch):
    # Req 4.7 — nothing references the issue → [].
    _patch_sources(monkeypatch, [], [])
    out = rag_engine.get_issue_timeline("SESF-99999", db_path=DB)
    assert out == []


# ---------------------------------------------------------------------------
# Engine: filters + limit
# ---------------------------------------------------------------------------

def test_timeline_provider_subset(monkeypatch):
    # Req 4.4 — restrict the feed to the requested provider subset.
    structured = [
        _entry("a", "2026-05-01T10:00:00", provider="claude_code_cli"),
        _entry("b", "2026-05-02T10:00:00", provider="codex"),
        _entry("c", "2026-05-03T10:00:00", provider="opencode"),
    ]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", providers=["codex", "opencode"], db_path=DB)
    assert {r["doc_id"] for r in out} == {"b", "c"}
    assert all(r["provider"] in ("codex", "opencode") for r in out)


def test_timeline_date_bounds(monkeypatch):
    # Req 4.3 — only turns within [date_from, date_to] are returned.
    structured = [
        _entry("early", "2026-04-15T10:00:00"),
        _entry("mid", "2026-05-10T10:00:00"),
        _entry("late", "2026-06-20T10:00:00"),
    ]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline(
        "SESF-25", date_from="2026-05-01", date_to="2026-05-31", db_path=DB
    )
    assert [r["doc_id"] for r in out] == ["mid"]


def test_timeline_default_limit_is_50(monkeypatch):
    # Req 4.5 — omitted limit applies the documented default of 50.
    structured = [
        _entry(f"{i:04d}", f"2026-05-01T10:{i:02d}:00") for i in range(60)
    ]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert len(out) == 50


def test_timeline_explicit_limit(monkeypatch):
    # Req 4.5 — an explicit limit caps the feed length.
    structured = [
        _entry(f"{i:04d}", f"2026-05-01T10:{i:02d}:00") for i in range(20)
    ]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", limit=5, db_path=DB)
    assert len(out) == 5
    # Oldest-first: the first five timestamps are the earliest minutes.
    assert [r["doc_id"] for r in out] == ["0000", "0001", "0002", "0003", "0004"]


def test_timeline_entry_carries_required_fields(monkeypatch):
    # Req 4.6 — each entry exposes provider, session_id, timestamp, role, text, doc_id.
    structured = [_entry("a", "2026-05-01T10:00:00")]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert out
    entry = out[0]
    for key in ("provider", "session_id", "timestamp", "doc_id", "text"):
        assert key in entry
    assert "role" in entry or "chunk_type" in entry


# ---------------------------------------------------------------------------
# Engine: async wrapper
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_timeline_async_wrapper_matches_sync(monkeypatch):
    # Design — get_issue_timeline_async mirrors search_async.
    structured = [_entry("a", "2026-05-01T10:00:00")]
    _patch_sources(monkeypatch, structured, [])
    out = await rag_engine.get_issue_timeline_async("SESF-25", db_path=DB)
    assert [r["doc_id"] for r in out] == ["a"]


# ---------------------------------------------------------------------------
# Transport: MCP tool + HTTP endpoint equivalence
# ---------------------------------------------------------------------------

tools = importlib.import_module("tools")


def _tool_registry():
    """Register tools against a capturing fake server; return {name: schema}, call_tool."""
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
    return {t.name: t.inputSchema for t in listed}, captured["call"]


def test_mcp_timeline_tool_is_registered():
    # Req 5.1 — a get_issue_timeline tool is listed and takes issue_id.
    schemas, _ = _tool_registry()
    assert "get_issue_timeline" in schemas
    assert "issue_id" in schemas["get_issue_timeline"]["properties"]


def test_mcp_timeline_tool_invokes_engine(monkeypatch):
    # Req 5.1 — the tool routes issue_id to rag_engine.get_issue_timeline.
    seen: dict = {}

    def _spy(issue_id, **kwargs):
        seen["issue_id"] = issue_id
        return [_entry("a", "2026-05-01T10:00:00")]

    monkeypatch.setattr(tools.rag_engine, "get_issue_timeline", _spy)
    monkeypatch.setattr(tools, "get_current_project_root", lambda: None)

    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("get_issue_timeline", {"issue_id": "SESF-25"}))

    assert seen.get("issue_id") == "SESF-25"
    assert result and result[0].type == "text"
    assert not result[0].text.startswith("Error")


def test_http_timeline_route_registered():
    # Req 5.2 — a /timeline route exists on the Starlette app.
    http_server = importlib.import_module("http_server")
    routes = {getattr(route, "path", "") for route in http_server.app.routes}
    assert "/timeline" in routes


def test_http_timeline_matches_engine_feed(stub_rag_engine, tmp_path, monkeypatch):
    # Req 5.2 / 5.3 — GET /timeline returns the same feed the engine produces.
    import importlib
    import sys

    from starlette.testclient import TestClient

    feed = [_entry("a", "2026-05-01T10:00:00"), _entry("b", "2026-05-02T10:00:00")]
    stub_rag_engine.get_issue_timeline = lambda *a, **kw: feed

    monkeypatch.setenv("HOME", str(tmp_path))
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    client = TestClient(http_server.app)
    try:
        resp = client.get("/timeline", params={"issue_id": "SESF-25"})
        assert resp.status_code == 200
        body = resp.json()
        # Accept either a bare list or a {"timeline": [...]} envelope.
        entries = body if isinstance(body, list) else body.get("timeline", body.get("results"))
        doc_ids = [e["doc_id"] for e in entries]
        assert doc_ids == ["a", "b"]
    finally:
        client.close()


def test_http_and_mcp_equivalent_for_same_input(stub_rag_engine, tmp_path, monkeypatch):
    # Req 5.3 — MCP tool and HTTP endpoint produce equivalent doc_id feeds.
    import importlib
    import sys

    from starlette.testclient import TestClient

    feed = [_entry("a", "2026-05-01T10:00:00"), _entry("b", "2026-05-02T10:00:00")]

    # MCP side
    monkeypatch.setattr(tools.rag_engine, "get_issue_timeline", lambda *a, **kw: feed)
    monkeypatch.setattr(tools, "get_current_project_root", lambda: None)
    _, call_tool = _tool_registry()
    mcp_result = asyncio.run(call_tool("get_issue_timeline", {"issue_id": "SESF-25"}))
    mcp_text = mcp_result[0].text

    # HTTP side
    stub_rag_engine.get_issue_timeline = lambda *a, **kw: feed
    monkeypatch.setenv("HOME", str(tmp_path))
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    client = TestClient(http_server.app)
    try:
        resp = client.get("/timeline", params={"issue_id": "SESF-25"})
        body = resp.json()
        entries = body if isinstance(body, list) else body.get("timeline", body.get("results"))
        http_ids = [e["doc_id"] for e in entries]
    finally:
        client.close()

    assert http_ids == ["a", "b"]
    # The MCP text rendering should reference the same turns.
    assert "a" in mcp_text and "b" in mcp_text
