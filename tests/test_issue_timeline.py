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
import random
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
    """Patch the structured (_query_batches) and FTS (_fts.search) timeline sources.

    Both are patched to return pre-shaped entry dicts so the test exercises the
    engine's stream/bound/dedup/sort/limit logic rather than Milvus/FTS internals.
    ``get_issue_timeline`` streams the structured source via ``_query_batches``
    (a single batch here) and routes both sources through the bounded oldest-N
    collector (SESF-34). ``_query_all`` is patched too in case any other path
    reaches for it, but the timeline no longer drains it.
    """
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *a, **kw: iter([[dict(r) for r in structured]]),
    )
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
# SESF-34: memory-bounded oldest-N pagination
# ---------------------------------------------------------------------------

def test_oldest_n_collector_keeps_bound_and_oldest():
    # SESF-34 — the bounded collector never exceeds ``limit`` mid-stream and
    # yields the globally-oldest entries regardless of insertion order.
    collector = rag_engine._OldestN(3)
    for i in [5, 1, 4, 2, 0, 3]:  # deliberately out of chronological order
        collector.add(_entry(f"d{i}", f"2026-05-01T00:00:0{i}"))
        assert len(collector) <= 3
    assert [r["doc_id"] for r in collector.result()] == ["d0", "d1", "d2"]


def test_oldest_n_collector_robust_to_zero_limit_and_null_fields():
    # SESF-34 hardening (PR #29 bot review) — a zero-capacity collector is a
    # no-op (no IndexError on the eviction branch), and null timestamp/doc_id
    # fields don't crash the heap key comparison or the final sort (an unguarded
    # ``None`` vs ``str`` compare raises TypeError; FTS NULLs make this reachable).
    zero = rag_engine._OldestN(0)
    zero.add(_entry("a", "2026-05-01T10:00:00"))  # must not raise
    assert zero.result() == []

    collector = rag_engine._OldestN(3)
    collector.add({"doc_id": None, "timestamp": None})        # null fields, must not raise
    collector.add({"doc_id": 12345, "timestamp": 1700000000})  # truthy non-str fields, must not raise
    collector.add(_entry("b", "2026-05-02T10:00:00"))
    assert len(collector) <= 3
    assert "b" in [r.get("doc_id") for r in collector.result()]  # sort survives null + non-str entries


def test_oldest_n_orders_mixed_timezone_formats_chronologically():
    """Z/+offset/naive ISO forms must order by instant, not by string
    (SESF-43, mirrors the _NewestN fix from PR #38):
    "2026-01-01T12:00:00+05:00" is lexicographically the largest here but
    chronologically the oldest, so it must be kept and sorted first."""
    collector = rag_engine._OldestN(2)
    collector.add(_entry("zulu", "2026-01-01T09:00:00Z"))
    collector.add(_entry("naive", "2026-01-01T08:00:00"))  # naive = UTC
    collector.add(_entry("offset", "2026-01-01T12:00:00+05:00"))  # 07:00 UTC
    assert [r["doc_id"] for r in collector.result()] == ["offset", "naive"]


def test_timeline_retains_oldest_n_from_shuffled_large_input(monkeypatch):
    # SESF-34 (core) — a pathologically large structured set is streamed in
    # shuffled timestamp order across many batches; the bounded collector must
    # still return the true oldest ``limit``, not an arbitrary iterator prefix.
    # Milvus query_iterator order is undefined, so order-independence is the
    # guarantee. The timeline must STREAM (_query_batches), not drain _query_all.
    n = 5000
    # Strictly-increasing, zero-padded ISO timestamps so lexicographic order ==
    # chronological order, and doc_id ordering tracks timestamp ordering.
    rows = [
        _entry(f"{i:05d}", f"2026-05-01T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}")
        for i in range(n)
    ]
    shuffled = list(rows)
    random.Random(1234).shuffle(shuffled)
    batches = [shuffled[i:i + 256] for i in range(0, len(shuffled), 256)]
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *a, **kw: iter([[dict(r) for r in b] for b in batches]),
    )
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [])

    def _no_query_all(*a, **kw):
        raise AssertionError("timeline must stream via _query_batches, not drain _query_all (SESF-34)")
    monkeypatch.setattr(rag_engine, "_query_all", _no_query_all)

    out = rag_engine.get_issue_timeline("SESF-25", limit=50, db_path=DB)
    assert [r["doc_id"] for r in out] == [f"{i:05d}" for i in range(50)]


def test_timeline_bound_applies_across_structured_and_fts(monkeypatch):
    # SESF-34 — the FTS fallback flows through the SAME bound: an older FTS-only
    # turn displaces the newest structured turn when the feed is already at limit.
    structured = [_entry(f"s{i:03d}", f"2026-05-02T00:{i:02d}:00") for i in range(50)]
    fts = [_entry("fts-old", "2026-05-01T00:00:00")]  # older than every structured row
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", limit=50, db_path=DB)
    ids = [r["doc_id"] for r in out]
    assert len(out) == 50
    assert ids[0] == "fts-old"   # the oldest turn overall is retained
    assert "s049" not in ids     # the newest structured turn is displaced by the bound


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


# ---------------------------------------------------------------------------
# Sessions by issue: timeline rows grouped into sessions, newest first, for
# get_memory_context. Claude Code rows join on session_id and Codex rows on
# logical_session_id; day bounds are America/Chicago calendar days.
# ---------------------------------------------------------------------------


def _trow(session_id, ts, provider="claude_code_cli", root="/Volumes/DATA/GitHub/SessionFlow",
          logical=None, issue="DEVS-145"):
    return {"doc_id": f"{session_id}:{ts}", "timestamp": ts, "provider": provider,
            "session_id": session_id, "logical_session_id": logical or session_id,
            "project_root": root, "issue_ids": f",{issue},"}


def _fake_timeline(monkeypatch, rows_by_issue):
    def fake(issue_id, *, limit=50, **kwargs):
        return [r for r in rows_by_issue.get(issue_id.upper(), [])]

    monkeypatch.setattr(rag_engine, "get_issue_timeline", fake)


def test_sessions_by_issue_groups_newest_first(monkeypatch):
    _fake_timeline(monkeypatch, {
        "DEVS-145": [_trow("a", "2026-10-01T10:00:00Z"), _trow("a", "2026-10-03T10:00:00Z"),
                     _trow("b", "2026-10-02T10:00:00Z")],
        "DEVS-146": [_trow("c", "2026-10-05T10:00:00Z", issue="DEVS-146"),
                     _trow("a", "2026-10-04T10:00:00Z", issue="DEVS-146")],
    })
    out = rag_engine.sessions_by_issue(["DEVS-145", "DEVS-146"], limit=10)
    assert [s["session_id"] for s in out] == ["c", "a", "b"]
    a = next(s for s in out if s["session_id"] == "a")
    assert a["turns"] == 3 and a["last_timestamp"] == "2026-10-04T10:00:00Z"
    assert sorted(a["issue_ids"]) == ["DEVS-145", "DEVS-146"]
    assert len(rag_engine.sessions_by_issue(["DEVS-145", "DEVS-146"], limit=2)) == 2


def test_timeline_newest_mode_keeps_newest_turns_newest_first(monkeypatch):
    structured = [_entry(f"d{i}", f"2026-05-0{i}T10:00:00") for i in range(1, 6)]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", limit=2, newest=True, db_path=DB)
    assert [r["doc_id"] for r in out] == ["d5", "d4"]


def test_timeline_default_order_stays_oldest_first(monkeypatch):
    structured = [_entry(f"d{i}", f"2026-05-0{i}T10:00:00") for i in range(5, 0, -1)]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", limit=2, db_path=DB)
    assert [r["doc_id"] for r in out] == ["d1", "d2"]


def test_sessions_by_issue_keeps_newest_session_past_scan_limit(monkeypatch):
    structured = [_entry(f"d{i}", f"2026-05-0{i}T10:00:00") for i in range(1, 6)]
    for row in structured:
        row["issue_ids"] = ",DEVS-145,"
    _patch_sources(monkeypatch, structured, [])
    monkeypatch.setattr(rag_engine, "SESSION_SCAN_LIMIT", 2)
    out = rag_engine.sessions_by_issue(["DEVS-145"], limit=10, db_path=DB)
    assert [s["session_id"] for s in out] == ["sess-d5", "sess-d4"]


def test_sessions_by_issue_joins_codex_on_logical_id(monkeypatch):
    _fake_timeline(monkeypatch, {"DEVS-145": [
        _trow("payload-1", "2026-10-01T10:00:00Z", provider="codex", logical="thread-x"),
        _trow("payload-2", "2026-10-01T11:00:00Z", provider="codex", logical="thread-x"),
        _trow("thread-x", "2026-10-01T12:00:00Z", provider="claude_code_cli"),
    ]})
    out = rag_engine.sessions_by_issue(["DEVS-145"], limit=10)
    keys = sorted((s["provider"], s["session_id"]) for s in out)
    assert keys == [("claude_code_cli", "thread-x"), ("codex", "thread-x")]
    assert next(s for s in out if s["provider"] == "codex")["turns"] == 2


def test_sessions_by_issue_filters_project_roots_and_worktrees(monkeypatch):
    root = "/Volumes/DATA/GitHub/SessionFlow"
    _fake_timeline(monkeypatch, {"DEVS-145": [
        _trow("main", "2026-10-01T10:00:00Z", root=root),
        _trow("wt", "2026-10-01T11:00:00Z", root=f"{root}/.worktrees/spec-DEVS-145"),
        _trow("other", "2026-10-01T12:00:00Z", root="/Volumes/DATA/GitHub/Devops"),
        _trow("prefix", "2026-10-01T13:00:00Z", root=f"{root}-old"),
    ]})
    out = rag_engine.sessions_by_issue(["DEVS-145"], limit=10, project_roots=[root])
    assert {s["session_id"] for s in out} == {"main", "wt"}


def test_sessions_by_issue_day_uses_chicago_bounds(monkeypatch):
    from datetime import date

    _fake_timeline(monkeypatch, {"DEVS-145": [
        _trow("before", "2026-10-20T04:59:59Z"),
        _trow("inside", "2026-10-20T05:00:00Z"),
        _trow("last", "2026-10-21T04:59:59Z"),
        _trow("after", "2026-10-21T05:00:00Z"),
    ]})
    out = rag_engine.sessions_by_issue(["DEVS-145"], limit=10, day=date(2026, 10, 20))
    assert {s["session_id"] for s in out} == {"inside", "last"}


def test_sessions_by_issue_excludes_the_source_session(monkeypatch):
    _fake_timeline(monkeypatch, {"DEVS-145": [_trow("src", "2026-10-01T10:00:00Z"),
                                              _trow("other", "2026-10-02T10:00:00Z")]})
    out = rag_engine.sessions_by_issue(["DEVS-145"], limit=10, exclude_session="src")
    assert [s["session_id"] for s in out] == ["other"]


def test_chicago_day_bounds_are_half_open_and_follow_dst():
    from datetime import date, datetime, timezone

    start, end = rag_engine.chicago_day_bounds(date(2026, 11, 1))
    assert start == datetime(2026, 11, 1, 5, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 11, 2, 6, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Subagent and compaction transcripts (agent-* sessions) are hidden by default.
# ---------------------------------------------------------------------------


def _agent(doc_id, ts, session_id):
    row = _entry(doc_id, ts)
    row["session_id"] = session_id
    return row


def test_timeline_hides_agent_sessions_by_default(monkeypatch):
    structured = [_entry("main", "2026-05-01T10:00:00"),
                  _agent("sub", "2026-05-02T10:00:00", "agent-a1b2c3")]
    fts = [_agent("compact", "2026-05-03T10:00:00", "agent-acompact-5c789c")]
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert [r["doc_id"] for r in out] == ["main"]
    assert out.hidden_subagents == 2


def test_timeline_include_subagents_keeps_agent_sessions(monkeypatch):
    structured = [_entry("main", "2026-05-01T10:00:00"),
                  _agent("sub", "2026-05-02T10:00:00", "agent-a1b2c3")]
    fts = [_agent("compact", "2026-05-03T10:00:00", "agent-acompact-5c789c")]
    _patch_sources(monkeypatch, structured, fts)
    out = rag_engine.get_issue_timeline("SESF-25", include_subagents=True, db_path=DB)
    assert [r["doc_id"] for r in out] == ["main", "sub", "compact"]
    assert out.hidden_subagents == 0


def test_timeline_hidden_count_dedups_across_sources(monkeypatch):
    row = _agent("sub", "2026-05-02T10:00:00", "agent-a1b2c3")
    _patch_sources(monkeypatch, [row], [row])
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert list(out) == [] and out.hidden_subagents == 1


def test_timeline_hidden_count_skips_rows_other_filters_drop(monkeypatch):
    structured = [_agent("old", "2026-04-01T10:00:00", "agent-a1"),
                  _agent("codex", "2026-05-02T10:00:00", "agent-a2")]
    structured[1]["provider"] = "codex"
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", providers=["claude_code_cli"],
                                        date_from="2026-05-01", db_path=DB)
    assert out.hidden_subagents == 0


def test_timeline_limit_counts_only_visible_turns(monkeypatch):
    structured = [_agent(f"s{i}", f"2026-05-0{i}T09:00:00", f"agent-a{i}") for i in range(1, 8)]
    structured += [_entry(f"m{i}", f"2026-05-0{i}T10:00:00") for i in range(5, 8)]
    _patch_sources(monkeypatch, structured, [])
    out = rag_engine.get_issue_timeline("SESF-25", limit=3, db_path=DB)
    assert [r["doc_id"] for r in out] == ["m5", "m6", "m7"]


@pytest.mark.anyio
async def test_timeline_async_forwards_newest_and_include_subagents(monkeypatch):
    seen: dict = {}

    def fake(issue_id, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(rag_engine, "get_issue_timeline", fake)
    await rag_engine.get_issue_timeline_async("SESF-25", newest=True,
                                              include_subagents=True, db_path=DB)
    assert seen["newest"] is True and seen["include_subagents"] is True


def test_sessions_by_issue_keeps_agent_sessions(monkeypatch):
    seen: dict = {}

    def fake(issue_id, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(rag_engine, "get_issue_timeline", fake)
    rag_engine.sessions_by_issue(["DEVS-145"], limit=5)
    assert seen["include_subagents"] is True


def test_mcp_timeline_schema_has_include_subagents():
    schemas, _ = _tool_registry()
    prop = schemas["get_issue_timeline"]["properties"]["include_subagents"]
    assert prop["type"] == "boolean" and prop["default"] is False


def _call_timeline(monkeypatch, args, feed=None):
    seen: dict = {}

    def _spy(issue_id, **kwargs):
        seen.update(kwargs)
        return feed if feed is not None else [_entry("a", "2026-05-01T10:00:00")]

    monkeypatch.setattr(tools.rag_engine, "get_issue_timeline", _spy)
    monkeypatch.setattr(tools, "get_current_project_root", lambda: None)
    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("get_issue_timeline", {"issue_id": "SESF-25", **args}))
    return seen, result[0].text


def test_mcp_timeline_passes_include_subagents(monkeypatch):
    seen, _ = _call_timeline(monkeypatch, {})
    assert seen["include_subagents"] is False
    seen, _ = _call_timeline(monkeypatch, {"include_subagents": True})
    assert seen["include_subagents"] is True


def test_mcp_timeline_refuses_non_bool_include_subagents(monkeypatch):
    seen, text = _call_timeline(monkeypatch, {"include_subagents": "yes"})
    assert seen == {} and "include_subagents must be true or false" in text


def test_format_timeline_names_hidden_subagent_turns():
    feed = rag_engine.TimelineResult([_entry("a", "2026-05-01T10:00:00")], hidden_subagents=3)
    text = tools.format_timeline(feed)
    assert "3 subagent turns hidden; pass include_subagents=true to show them." in text
    assert "hidden" not in tools.format_timeline([_entry("a", "2026-05-01T10:00:00")])


def test_format_timeline_empty_feed_names_hidden_turns():
    text = tools.format_timeline(rag_engine.TimelineResult([], hidden_subagents=1))
    assert text.startswith("No turns reference that issue.")
    assert "1 subagent turn hidden" in text


def _http_client(stub_rag_engine, tmp_path, monkeypatch, fake):
    import sys

    from starlette.testclient import TestClient

    stub_rag_engine.get_issue_timeline = fake
    monkeypatch.setenv("HOME", str(tmp_path))
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    return TestClient(http_server.app)


def test_http_timeline_parses_include_subagents(stub_rag_engine, tmp_path, monkeypatch):
    seen: list = []

    def fake(issue_id, **kwargs):
        seen.append(kwargs["include_subagents"])
        return rag_engine.TimelineResult([], hidden_subagents=4)

    client = _http_client(stub_rag_engine, tmp_path, monkeypatch, fake)
    try:
        body = client.get("/timeline", params={"issue_id": "SESF-25"}).json()
        assert body["hidden_subagents"] == 4 and body["timeline"] == []
        client.get("/timeline", params={"issue_id": "SESF-25", "include_subagents": "true"})
        client.get("/timeline", params={"issue_id": "SESF-25", "include_subagents": "false"})
        bad = client.get("/timeline", params={"issue_id": "SESF-25", "include_subagents": "1"})
        assert bad.status_code == 400
    finally:
        client.close()
    assert seen == [False, True, False]


def test_timeline_hidden_count_is_a_lower_bound_when_fts_fills_its_window(monkeypatch):
    fts = [_agent("s1", "2026-05-02T10:00:00", "agent-a1"), _entry("m1", "2026-05-03T10:00:00")]
    _patch_sources(monkeypatch, [], fts)
    monkeypatch.setattr(rag_engine, "_TIMELINE_FTS_FETCH_CAP", 2)
    out = rag_engine.get_issue_timeline("SESF-25", limit=1, db_path=DB)
    assert out.hidden_subagents == 1 and out.hidden_subagents_capped is True


def test_timeline_hidden_count_is_exact_below_the_fts_window(monkeypatch):
    _patch_sources(monkeypatch, [], [_agent("s1", "2026-05-02T10:00:00", "agent-a1")])
    out = rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert out.hidden_subagents == 1 and out.hidden_subagents_capped is False


def test_format_timeline_says_at_least_when_the_count_is_capped():
    feed = rag_engine.TimelineResult([_entry("a", "2026-05-01T10:00:00")], hidden_subagents=2,
                                     hidden_subagents_capped=True)
    assert "At least 2 subagent turns hidden" in tools.format_timeline(feed)


def test_http_timeline_reports_a_capped_count(stub_rag_engine, tmp_path, monkeypatch):
    client = _http_client(stub_rag_engine, tmp_path, monkeypatch, lambda *a, **kw: rag_engine.TimelineResult(
        [], hidden_subagents=2, hidden_subagents_capped=True))
    try:
        body = client.get("/timeline", params={"issue_id": "SESF-25"}).json()
    finally:
        client.close()
    assert body["hidden_subagents"] == 2 and body["hidden_subagents_capped"] is True


def _patch_filtering_fts(monkeypatch, structured, fts):
    """FTS fake that honours the session_id LIKE / NOT LIKE filter and the n limit."""
    calls: list = []

    def search(query, n=15, filters=None, db_path=None):
        calls.append(dict(filters or {}))
        rows = [dict(r) for r in fts]
        op = (filters or {}).get("session_id")
        if op == ("NOT LIKE", "agent-%"):
            rows = [r for r in rows if not r["session_id"].startswith("agent-")]
        elif op == ("LIKE", "agent-%"):
            rows = [r for r in rows if r["session_id"].startswith("agent-")]
        return rows[:n]

    _patch_sources(monkeypatch, structured, [])
    monkeypatch.setattr(rag_engine._fts, "search", search)
    return calls


def test_timeline_agent_fts_hits_do_not_use_the_visible_window(monkeypatch):
    fts = [_agent(f"s{i}", f"2026-05-0{i}T10:00:00", f"agent-a{i}") for i in range(1, 4)]
    fts.append(_entry("main", "2026-05-05T10:00:00"))
    _patch_filtering_fts(monkeypatch, [], fts)
    monkeypatch.setattr(rag_engine, "_TIMELINE_FTS_FETCH_CAP", 2)
    out = rag_engine.get_issue_timeline("SESF-25", limit=1, db_path=DB)
    assert [r["doc_id"] for r in out] == ["main"]
    assert out.hidden_subagents == 2 and out.hidden_subagents_capped is True


def test_timeline_fts_filters_follow_include_subagents(monkeypatch):
    calls = _patch_filtering_fts(monkeypatch, [], [])
    rag_engine.get_issue_timeline("SESF-25", db_path=DB)
    assert sorted(c.get("session_id") for c in calls) == [("LIKE", "agent-%"), ("NOT LIKE", "agent-%")]
    calls.clear()
    rag_engine.get_issue_timeline("SESF-25", include_subagents=True, db_path=DB)
    assert [c.get("session_id") for c in calls] == [None]


def test_timeline_session_filter_runs_on_a_real_fts_index(monkeypatch, tmp_path):
    from fts_hybrid import FTSIndex

    fts = FTSIndex("turns_fts", ["session_id", "timestamp", "chunk_type", "provider", "issue_ids"])
    db_path = str(tmp_path / "milvus.db")
    rows = [("a1", "agent-a1"), ("a2", "agent-acompact-9"), ("m1", "sess-main")]
    fts.insert(fts.connection(db_path), [
        {"doc_id": d, "content": "work on SESF-25", "session_id": s, "timestamp": "2026-05-01T10:00:00",
         "chunk_type": "turn", "provider": "claude_code_cli", "issue_ids": ""} for d, s in rows])
    monkeypatch.setattr(rag_engine, "_fts", fts)
    monkeypatch.setattr(rag_engine, "_query_batches", lambda *a, **kw: iter([]))
    out = rag_engine.get_issue_timeline("SESF-25", db_path=db_path)
    assert [r["doc_id"] for r in out] == ["m1"] and out.hidden_subagents == 2
    out = rag_engine.get_issue_timeline("SESF-25", include_subagents=True, db_path=db_path)
    assert sorted(r["doc_id"] for r in out) == ["a1", "a2", "m1"]


def test_format_timeline_notes_a_capped_zero_count():
    feed = rag_engine.TimelineResult([_entry("a", "2026-05-01T10:00:00")], hidden_subagents=0,
                                     hidden_subagents_capped=True)
    assert ("Subagent turns may be hidden; pass include_subagents=true to show them."
            in tools.format_timeline(feed))
