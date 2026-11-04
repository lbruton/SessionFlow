"""list_sessions: one row per session, with a title and the wrapped flag.

Covers ``rag_engine.list_sessions`` (the streamed aggregate scan),
``rag_engine.session_titles`` (the title fetch), ``session_listing.list_sessions``
(subagent, wrapped and count handling) and the ``list_sessions`` MCP tool.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from contextlib import contextmanager

import pytest

rag_engine = importlib.import_module("rag_engine")
session_listing = importlib.import_module("session_listing")
tools = importlib.import_module("tools")

DB = "/tmp/sessionflow-test.db"


def _row(doc_id, session_id, ts, turn_index, document="", provider="claude_code_cli",
         project_root="/p/a", git_branch="main", chunk_type="turn"):
    return {
        "doc_id": doc_id, "session_id": session_id, "timestamp": ts,
        "turn_index": turn_index, "document": document, "provider": provider,
        "project_root": project_root, "git_branch": git_branch, "chunk_type": chunk_type,
    }


class _Iterator:
    def __init__(self, rows, batch_size):
        self._rows, self._size, self._pos = rows, batch_size, 0

    def next(self):
        batch = self._rows[self._pos:self._pos + self._size]
        self._pos += self._size
        return batch

    def close(self):
        pass


class _Client:
    """Serves query_iterator; an ``id in [...]`` filter selects rows by primary key."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[dict] = []

    def has_collection(self, name):
        return True

    def query_iterator(self, *, collection_name, batch_size, filter, output_fields):
        self.calls.append({"filter": filter, "output_fields": list(output_fields)})
        rows = self.rows
        ids = re.search(r"id in (\[[^\]]*\])", filter)
        if ids:
            wanted = set(json.loads(ids.group(1)))
            rows = [r for r in rows if rag_engine._pk_from_doc_id(r["doc_id"]) in wanted]
        for field in ("session_id", "provider"):
            eq = re.search(rf'\b{field} == "([^"]*)"', filter)
            if eq:
                rows = [r for r in rows if r[field] == eq.group(1)]
        projected = [{k: r[k] for k in output_fields if k in r} for r in rows]
        return _Iterator(projected, batch_size)

    def query(self, *args, **kwargs):
        raise AssertionError("single-shot query() must not run")

    def search(self, *args, **kwargs):
        raise AssertionError("vector search must not run")


def _patch(monkeypatch, rows) -> _Client:
    client = _Client(rows)

    @contextmanager
    def fake(db_path=None):
        yield client

    monkeypatch.setattr(rag_engine, "milvus_client", fake)
    return client


# ---------------------------------------------------------------------------
# rag_engine.list_sessions: aggregates
# ---------------------------------------------------------------------------

def test_groups_by_session_and_provider(monkeypatch):
    rows = [
        _row("a1", "s1", "2026-11-01T10:00:00", 100, git_branch="main"),
        _row("a2", "s1", "2026-11-01T12:00:00", 300, git_branch="feat/x"),
        _row("a3", "s1", "2026-11-01T11:00:00", 200),
        _row("b1", "s2", "2026-11-02T09:00:00", 5),
        _row("c1", "s1", "2026-10-30T09:00:00", 1, provider="codex"),
    ]
    _patch(monkeypatch, rows)
    out = rag_engine.list_sessions(db_path=DB)
    keys = [(s["session_id"], s["provider"]) for s in out]
    assert keys == [("s2", "claude_code_cli"), ("s1", "claude_code_cli"), ("s1", "codex")]
    s1 = out[1]
    assert s1["turns"] == 3
    assert s1["min_ts"] == "2026-11-01T10:00:00"
    assert s1["max_ts"] == "2026-11-01T12:00:00"
    assert s1["last_turn_index"] == 300
    assert s1["branches"] == ["feat/x", "main"]
    assert s1["project_root"] == "/p/a"


def test_scan_reads_no_document_and_skips_passages(monkeypatch):
    client = _patch(monkeypatch, [_row("a1", "s1", "2026-11-01T10:00:00", 1)])
    rag_engine.list_sessions(db_path=DB)
    scan = client.calls[0]
    assert "document" not in scan["output_fields"]
    assert 'chunk_type != "turn_passage"' in scan["filter"]


def test_filters_reach_the_scan(monkeypatch):
    client = _patch(monkeypatch, [])
    rag_engine.list_sessions(project_root=["/p/a", "/p/b"], provider="codex",
                             date_from="2026-11-01", date_to="2026-11-02", db_path=DB)
    expr = client.calls[0]["filter"]
    assert 'project_root in ["/p/a", "/p/b"]' in expr
    assert 'provider == "codex"' in expr
    assert 'timestamp >= "2026-11-01"' in expr
    assert 'timestamp <= "2026-11-02T23:59:59"' in expr


def test_single_root_uses_equality(monkeypatch):
    client = _patch(monkeypatch, [])
    rag_engine.list_sessions(project_root="/p/a", db_path=DB)
    assert 'project_root == "/p/a"' in client.calls[0]["filter"]


def test_unscoped_scan_has_only_the_passage_clause(monkeypatch):
    client = _patch(monkeypatch, [])
    rag_engine.list_sessions(project_root=None, db_path=DB)
    assert client.calls[0]["filter"] == 'chunk_type != "turn_passage"'


def test_title_candidates_are_bounded_per_session(monkeypatch):
    rows = [_row(f"d{i}", "s1", f"2026-11-01T10:{i:02d}:00", i) for i in range(50)]
    _patch(monkeypatch, rows)
    [s1] = rag_engine.list_sessions(db_path=DB)
    assert s1["turns"] == 50
    assert s1["first_doc_ids"] == [f"d{i}" for i in range(rag_engine.SESSION_TITLE_CANDIDATES)]


# ---------------------------------------------------------------------------
# rag_engine.session_titles
# ---------------------------------------------------------------------------

def test_title_skips_rows_with_empty_user_text(monkeypatch):
    rows = [
        _row("t0", "s1", "2026-11-01T10:00:00", 0, "User: \n\nAssistant: side-session brief"),
        _row("t1", "s1", "2026-11-01T10:01:00", 1,
             "User: <system-reminder>\nnoise\n</system-reminder>\n\nAssistant: ok"),
        _row("t2", "s1", "2026-11-01T10:02:00", 2,
             "User: Fix the   list_sessions\ntitle\n\nAssistant: Done."),
    ]
    _patch(monkeypatch, rows)
    sessions = rag_engine.list_sessions(db_path=DB)
    titles = rag_engine.session_titles(sessions, db_path=DB)
    assert titles[("s1", "claude_code_cli")] == "Fix the list_sessions title"


def test_title_prefers_words_over_a_command_marker(monkeypatch):
    rows = [
        _row("c0", "s1", "2026-11-01T10:00:00", 0, "User: [/start]\n\nAssistant: report"),
        _row("c1", "s1", "2026-11-01T10:01:00", 1, "User: Pull, then SESF-63\n\nAssistant: ok"),
        _row("k0", "s2", "2026-11-01T10:00:00", 0, "User: [/wrap]\n\nAssistant: wrapped"),
    ]
    _patch(monkeypatch, rows)
    sessions = rag_engine.list_sessions(db_path=DB)
    titles = rag_engine.session_titles(sessions, db_path=DB)
    assert titles[("s1", "claude_code_cli")] == "Pull, then SESF-63"
    assert titles[("s2", "claude_code_cli")] == "[/wrap]"


def test_title_scans_past_the_candidates_for_real_user_text(monkeypatch):
    k = rag_engine.SESSION_TITLE_CANDIDATES
    rows = [_row(f"m{i}", "s1", f"2026-11-01T10:{i:02d}:00", i, "User: [/start]\n\nAssistant: x")
            for i in range(k + 2)]
    rows.append(_row("late", "s1", "2026-11-01T11:00:00", k + 5, "User: The real ask\n\nAssistant: ok"))
    rows.append(_row("later", "s1", "2026-11-01T11:30:00", k + 9, "User: Another ask"))
    rows.append(_row("o0", "s1", "2026-11-01T09:00:00", 0, "User: other provider",
                     provider="codex"))
    client = _patch(monkeypatch, rows)
    sessions = [s for s in rag_engine.list_sessions(db_path=DB) if s["provider"] == "claude_code_cli"]
    titles = rag_engine.session_titles(sessions, db_path=DB)
    assert titles[("s1", "claude_code_cli")] == "The real ask"
    scan = client.calls[-1]["filter"]
    assert 'session_id == "s1"' in scan and 'provider == "claude_code_cli"' in scan


def test_no_fallback_scan_when_candidates_hold_a_title(monkeypatch):
    rows = [_row(f"d{i}", "s1", f"2026-11-01T10:{i:02d}:00", i, f"User: q{i}") for i in range(30)]
    client = _patch(monkeypatch, rows)
    sessions = rag_engine.list_sessions(db_path=DB)
    rag_engine.session_titles(sessions, db_path=DB)
    assert len(client.calls) == 2


def test_cleanup_list_prints_provider(monkeypatch, capsys):
    import cleanup

    rows = [_row("a1", "s1", "2026-11-01T10:00:00", 1),
            _row("c1", "s1", "2026-11-01T09:00:00", 1, provider="codex")]
    _patch(monkeypatch, rows)
    monkeypatch.setattr(cleanup, "get_db_path", lambda: DB)
    cleanup.cmd_list(type("A", (), {"project": None})())
    out = capsys.readouterr().out
    assert "Provider" in out.splitlines()[0]
    assert sum("claude_code_cli" in line for line in out.splitlines()) == 1
    assert sum("codex" in line for line in out.splitlines()) == 1


def test_title_is_cut_and_empty_when_no_user_text(monkeypatch):
    long = "word " * 100
    rows = [
        _row("x0", "s1", "2026-11-01T10:00:00", 0, f"User: {long}\n\nAssistant: ok"),
        _row("y0", "s2", "2026-11-01T10:00:00", 0, "Assistant: only a reply"),
    ]
    _patch(monkeypatch, rows)
    sessions = rag_engine.list_sessions(db_path=DB)
    titles = rag_engine.session_titles(sessions, db_path=DB)
    assert len(titles[("s1", "claude_code_cli")]) == rag_engine.SESSION_TITLE_CHARS
    assert titles[("s2", "claude_code_cli")] == ""


def test_title_fetch_reads_only_candidate_rows(monkeypatch):
    rows = [_row(f"d{i}", "s1", f"2026-11-01T10:{i:02d}:00", i, f"User: q{i}") for i in range(30)]
    client = _patch(monkeypatch, rows)
    sessions = rag_engine.list_sessions(db_path=DB)
    rag_engine.session_titles(sessions, db_path=DB)
    fetch = client.calls[1]
    ids = json.loads(re.search(r"id in (\[[^\]]*\])", fetch["filter"]).group(1))
    assert len(ids) == rag_engine.SESSION_TITLE_CANDIDATES
    assert "document" in fetch["output_fields"]


# ---------------------------------------------------------------------------
# session_listing.list_sessions: subagents, wrapped, count
# ---------------------------------------------------------------------------

class _Memory:
    def __init__(self, wrapped=(), fail=False):
        self.wrapped = set(wrapped)
        self.fail = fail
        self.asked: list[list[str]] = []

    async def wrapped_sessions(self, session_ids):
        self.asked.append(list(session_ids))
        if self.fail:
            raise RuntimeError("database down")
        return {s for s in session_ids if s in self.wrapped}


def _listing_rows():
    return [
        _row("n1", "new", "2026-11-03T10:00:00", 1, "User: newest"),
        _row("g1", "agent-a123", "2026-11-03T09:00:00", 1, "User: subagent"),
        _row("m1", "mid", "2026-11-02T10:00:00", 1, "User: middle"),
        _row("o1", "old", "2026-11-01T10:00:00", 1, "User: oldest"),
    ]


def _run(coro):
    return asyncio.run(coro)


def test_excludes_subagents_by_default(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, db_path=DB, memory=_Memory()))
    assert [s["session_id"] for s in out["sessions"]] == ["new", "mid", "old"]


def test_include_subagents(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, include_subagents=True, db_path=DB,
                                             memory=_Memory()))
    assert "agent-a123" in [s["session_id"] for s in out["sessions"]]


def test_wrapped_flag_and_title(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, db_path=DB, memory=_Memory({"mid"})))
    by_id = {s["session_id"]: s for s in out["sessions"]}
    assert by_id["mid"]["wrapped"] is True
    assert by_id["new"]["wrapped"] is False
    assert by_id["new"]["title"] == "newest"
    assert "first_doc_ids" not in by_id["new"]
    assert out["wrapped_known"] is True


def test_wrapped_false_filter_keeps_count(monkeypatch):
    rows = [_row(f"r{i}", f"s{i}", f"2026-11-01T10:{i:02d}:00", 1, f"User: q{i}")
            for i in range(10)]
    _patch(monkeypatch, rows)
    memory = _Memory({"s9", "s8", "s7"})
    out = _run(session_listing.list_sessions(None, n=3, wrapped=False, db_path=DB,
                                             memory=memory))
    assert [s["session_id"] for s in out["sessions"]] == ["s6", "s5", "s4"]
    assert all(s["wrapped"] is False for s in out["sessions"])


def test_n_limits_the_rows(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, n=1, db_path=DB, memory=_Memory()))
    assert [s["session_id"] for s in out["sessions"]] == ["new"]


def test_unknown_wrapped_without_memory(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, db_path=DB, memory=None))
    assert {s["wrapped"] for s in out["sessions"]} == {None}
    assert out["wrapped_known"] is False


def test_unknown_wrapped_when_lookup_fails(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    out = _run(session_listing.list_sessions(None, db_path=DB, memory=_Memory(fail=True)))
    assert {s["wrapped"] for s in out["sessions"]} == {None}
    assert out["wrapped_known"] is False


def test_wrapped_filter_refuses_without_memory(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    with pytest.raises(session_listing.WrappedUnavailable):
        _run(session_listing.list_sessions(None, wrapped=False, db_path=DB, memory=None))


def test_wrapped_filter_refuses_when_lookup_fails(monkeypatch):
    _patch(monkeypatch, _listing_rows())
    with pytest.raises(session_listing.WrappedUnavailable):
        _run(session_listing.list_sessions(None, wrapped=True, db_path=DB,
                                           memory=_Memory(fail=True)))


# ---------------------------------------------------------------------------
# MemoryStore.wrapped_sessions (Postgres)
# ---------------------------------------------------------------------------

def test_store_wrapped_sessions(memory_pg):
    import hashlib

    from memory_store import MemoryStore, NewMemory

    store = MemoryStore(memory_pg)
    store.migrate()
    try:
        def add(text, **over):
            fields = dict(text=text, agent_id="claude-code", project="sessionflow",
                          type="session-summary",
                          content_hash=hashlib.sha256(text.encode()).hexdigest())
            fields.update(over)
            return store.add(NewMemory(**fields))

        add("wrapped one", source_session_id="s-wrapped")
        add("other project", project="devops", source_session_id="s-elsewhere")
        add("a decision", type="decision", source_session_id="s-decision")
        gone = add("deleted wrap", source_session_id="s-deleted")
        store.delete(gone.memory_id)

        found = store.wrapped_sessions(
            ["s-wrapped", "s-elsewhere", "s-decision", "s-deleted", "s-none"])
        assert found == {"s-wrapped", "s-elsewhere"}
        assert store.wrapped_sessions([]) == set()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Formatting and the MCP tool
# ---------------------------------------------------------------------------

def _session(**over):
    s = {"session_id": "3aba31d6-70af-4154-9a14-5db6f51b306d", "provider": "claude_code_cli",
         "project_root": "/Volumes/DATA/GitHub/SessionFlow", "turns": 12,
         "min_ts": "2026-11-03T20:19:00.123", "max_ts": "2026-11-03T21:14:00.456",
         "last_turn_index": 3532199, "branches": ["main"], "wrapped": True,
         "title": "Start the session"}
    s.update(over)
    return s


def test_format_sessions_one_line_each():
    from project_scope import ProjectScope

    text = tools.format_sessions({"sessions": [_session()], "wrapped_known": True},
                                 ProjectScope(roots=None))
    lines = text.splitlines()
    assert lines[0].startswith("Scope:")
    line = lines[2]
    assert line.startswith("2026-11-03T21:14:00 claude_code_cli project:SessionFlow")
    assert "session:3aba31d6-70af-4154-9a14-5db6f51b306d" in line
    assert "last_turn:3532199" in line
    assert "turns:12" in line
    assert "first:2026-11-03T20:19:00" in line
    assert "wrapped:yes" in line
    assert line.endswith("| Start the session")


def test_format_sessions_unknown_wrapped_note():
    from project_scope import ProjectScope

    text = tools.format_sessions(
        {"sessions": [_session(wrapped=None, title="")], "wrapped_known": False},
        ProjectScope(roots=None))
    assert "wrapped:unknown" in text
    assert "(no user message)" in text
    assert "memory store" in text.splitlines()[-1]


def test_format_sessions_empty():
    from project_scope import ProjectScope

    text = tools.format_sessions({"sessions": [], "wrapped_known": True}, ProjectScope(roots=None))
    assert text.endswith("No sessions found.")


def _tool_server():
    from mcp.server import Server

    server = Server("test")
    tools.register_tools(server)
    return server


def _call(server, name, arguments):
    from mcp import types

    handler = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(method="tools/call",
                                params=types.CallToolRequestParams(name=name, arguments=arguments))
    result = asyncio.run(handler(req))
    return result.root.content[0].text


def test_tool_is_listed():
    from mcp import types

    server = _tool_server()
    handler = server.request_handlers[types.ListToolsRequest]
    result = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
    names = {t.name for t in result.root.tools}
    assert "list_sessions" in names


def test_tool_refuses_unnamed_scope(monkeypatch):
    async def no_roots(server):
        return None

    monkeypatch.setattr(tools.client_roots, "current_project_root", no_roots)
    monkeypatch.setattr(tools, "get_current_project_root", lambda: None)
    text = _call(_tool_server(), "list_sessions", {})
    assert text == tools.UNNAMED_SCOPE_ERROR


def test_tool_rejects_bad_provider_and_n():
    server = _tool_server()
    assert "Invalid provider" in _call(server, "list_sessions",
                                       {"project_root": "*", "provider": "nope"})
    assert "n must be" in _call(server, "list_sessions", {"project_root": "*", "n": 0})


def test_tool_passes_filters_and_renders(monkeypatch):
    seen = {}

    async def fake_list(roots, **kwargs):
        seen["roots"] = roots
        seen.update(kwargs)
        return {"sessions": [_session()], "wrapped_known": True}

    monkeypatch.setattr(tools.session_listing, "list_sessions", fake_list)
    text = _call(_tool_server(), "list_sessions",
                 {"project_root": "*", "n": 5, "provider": "codex", "date_from": "2026-11-01",
                  "date_to": "2026-11-03", "wrapped": False, "include_subagents": True})
    assert seen["roots"] is None
    assert seen["n"] == 5
    assert seen["provider"] == "codex"
    assert seen["date_from"] == "2026-11-01"
    assert seen["date_to"] == "2026-11-03"
    assert seen["wrapped"] is False
    assert seen["include_subagents"] is True
    assert "session:3aba31d6-70af-4154-9a14-5db6f51b306d" in text


def test_tool_json_format(monkeypatch):
    async def fake_list(roots, **kwargs):
        return {"sessions": [_session()], "wrapped_known": True}

    monkeypatch.setattr(tools.session_listing, "list_sessions", fake_list)
    text = _call(_tool_server(), "list_sessions", {"project_root": "*", "format": "json"})
    payload = json.loads(text)
    assert payload["sessions"][0]["session_id"] == "3aba31d6-70af-4154-9a14-5db6f51b306d"
    assert payload["scope"]["roots"] is None


def test_tool_reports_wrapped_unavailable(monkeypatch):
    async def fake_list(roots, **kwargs):
        raise session_listing.WrappedUnavailable("The memory store has not started.")

    monkeypatch.setattr(tools.session_listing, "list_sessions", fake_list)
    text = _call(_tool_server(), "list_sessions", {"project_root": "*", "wrapped": False})
    assert "memory store has not started" in text
