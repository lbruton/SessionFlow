"""get_turns resolves a unique session ID prefix and explains a miss."""

import asyncio
import importlib

import pytest

import rag_engine

tools = importlib.import_module("tools")

FULL = "1513a61d-ab1f-4dbe-a4e3-68de182ae25b"
TWIN = "1513a61d-ffff-4dbe-a4e3-68de182ae25b"
OTHER = "62245a69-18a2-4478-afb3-473f0b71c4ae"


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
    return asyncio.run(captured["call"](name, arguments))[0].text


def _record(doc_id, session_id, timestamp):
    record = {column: "" for column in rag_engine._fts.metadata_columns}
    record.update(doc_id=doc_id, content="text", session_id=session_id, timestamp=timestamp)
    return record


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "milvus.db")
    conn = rag_engine._fts.connection(path)
    rag_engine._fts.insert(conn, [
        _record("a1", FULL, "2026-11-01T23:40:24.779Z"),
        _record("a2", FULL, "2026-11-01T22:00:00.000Z"),
        _record("o1", OTHER, "2026-11-01T23:47:33.000Z"),
    ])
    rag_engine._fts.close_ephemeral(conn)
    monkeypatch.setattr(tools, "get_db_path", lambda: path)
    return path


@pytest.fixture
def stored(monkeypatch):
    """Serve turns for exact session IDs, as Milvus does, and record each lookup."""
    calls = []

    def _get_turns(session_id, turn_index, context=2, db_path=None):
        calls.append(session_id)
        if session_id not in (FULL, TWIN, OTHER):
            return []
        return [{"content": "User: hi", "session_id": session_id, "turn_index": 7,
                 "timestamp": "2026-11-01T23:40:24.779Z", "chunk_type": "turn"}]

    monkeypatch.setattr(tools.rag_engine, "get_turns", _get_turns)
    return calls


def _add_twin(db):
    conn = rag_engine._fts.connection(db)
    rag_engine._fts.insert(conn, [_record("t1", TWIN, "2026-11-01T10:00:00.000Z")])
    rag_engine._fts.close_ephemeral(conn)


def test_full_id_prints_the_session_and_skips_the_prefix_lookup(db, stored, monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("an exact hit must not look up prefixes")

    monkeypatch.setattr(tools.rag_engine, "sessions_with_prefix", _fail)
    text = _call("get_turns", {"session_id": FULL, "turn_index": 7})
    assert text.startswith(f"Session: {FULL}\n")
    assert stored == [FULL]


def test_unique_prefix_resolves_and_prints_the_full_id(db, stored):
    text = _call("get_turns", {"session_id": FULL[:8], "turn_index": 7})
    assert text.startswith(f"Session: {FULL}\n")
    assert "User: hi" in text
    assert stored == [FULL[:8], FULL]


def test_ambiguous_prefix_lists_candidates_and_returns_no_turns(db, stored):
    _add_twin(db)
    text = _call("get_turns", {"session_id": FULL[:8], "turn_index": 7})
    assert FULL in text and TWIN in text
    assert OTHER not in text
    assert "User: hi" not in text
    assert text.index(FULL) < text.index(TWIN)
    assert "2026-11-01T23:40:24" in text
    assert stored == [FULL[:8]]


def test_ambiguous_prefix_lists_at_most_five(db, stored):
    conn = rag_engine._fts.connection(db)
    rag_engine._fts.insert(conn, [
        _record(f"m{i}", f"1513a61d-{i}", f"2026-10-0{i + 1}T00:00:00Z") for i in range(7)
    ])
    rag_engine._fts.close_ephemeral(conn)
    text = _call("get_turns", {"session_id": FULL[:8], "turn_index": 7})
    assert text.count("\n- ") == 5


def test_unknown_id_returns_the_explanatory_error(db, stored):
    text = _call("get_turns", {"session_id": "deadbeef", "turn_index": 7})
    assert text == "Unknown session ID deadbeef. Pass the full session_id from a search result."


def test_prefix_under_eight_characters_is_not_expanded(db, stored):
    text = _call("get_turns", {"session_id": FULL[:7], "turn_index": 7})
    assert text.startswith(f"Unknown session ID {FULL[:7]}.")
    assert stored == [FULL[:7]]


@pytest.mark.parametrize("pattern", ["1513a61%", "1513a6__"])
def test_like_wildcards_in_the_id_match_literally(db, stored, pattern):
    text = _call("get_turns", {"session_id": pattern, "turn_index": 7})
    assert text.startswith(f"Unknown session ID {pattern}.")


def test_follow_up_log_names_the_resolved_session(db, stored, monkeypatch):
    seen = {}

    async def _log(name, *, client, item_id):
        seen["id"] = item_id

    monkeypatch.setattr(tools.search_log, "log_follow_up", _log)
    _call("get_turns", {"session_id": FULL[:8], "turn_index": 7})
    assert seen["id"] == f"{FULL}:7"


def test_tool_description_says_what_turn_index_is():
    captured = {}

    class _Server:
        def list_tools(self):
            def deco(fn):
                captured["list"] = fn
                return fn
            return deco

        def call_tool(self):
            return lambda fn: fn

    tools.register_tools(_Server())
    listed = asyncio.run(captured["list"]())
    (tool,) = [t for t in listed if t.name == "get_turns"]
    props = tool.inputSchema["properties"]
    assert "not a position" in props["turn_index"]["description"]
    assert "prefix" in props["session_id"]["description"]
