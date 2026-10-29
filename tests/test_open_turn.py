"""SESF-74: the open last turn is rebuilt with its reply, and re-indexing replaces it.

A transcript that ends on a complete line still has its last turn open: the
reply may not be written yet. The parser emits that turn so it is searchable,
and the next parse resumes at its user line. ``add_turns`` then replaces the
earlier row for the same session and turn_index.
"""

from __future__ import annotations

import contextlib
import json
import types

import pytest

import rag_engine
import transcript_parser
from provider_adapters import ProviderSource
from provider_claude import ClaudeCodeCliAdapter

DB = "/tmp/sessionflow-open-turn-test.db"


def _user(text):
    return json.dumps({"type": "user", "timestamp": "2026-10-29T00:00:00Z",
                       "message": {"role": "user", "content": text}}) + "\n"


def _assistant(text):
    return json.dumps({"type": "assistant", "timestamp": "2026-10-29T00:00:01Z",
                       "message": {"role": "assistant",
                                   "content": [{"type": "text", "text": text}]}}) + "\n"


def _tool_only_assistant():
    return json.dumps({"type": "assistant", "timestamp": "2026-10-29T00:00:01Z",
                       "message": {"role": "assistant",
                                   "content": [{"type": "tool_use", "name": "Bash"}]}}) + "\n"


FIRST = _user("first question about the indexer") + _assistant("first answer about the indexer")
FIRST_LEN = len(FIRST.encode())


def _write(path, data):
    path.write_bytes(data.encode())


def _append(path, data):
    with open(path, "ab") as f:
        f.write(data.encode())


# --- Parser ---------------------------------------------------------------


def test_open_last_turn_is_emitted_and_resumed_from_its_user_line(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("second question still running tools") + _tool_only_assistant())

    result = transcript_parser.parse_transcript_incremental(str(path), "s")

    assert [t["turn_index"] for t in result.turns] == [0, FIRST_LEN]
    assert "Assistant:" not in result.turns[-1]["text"]
    assert result.consumed_offset == path.stat().st_size
    assert result.resume_offset == FIRST_LEN


def test_resumed_parse_rebuilds_the_open_turn_with_its_reply(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("second question still running tools"))
    first = transcript_parser.parse_transcript_incremental(str(path), "s")
    partial = first.turns[-1]

    _append(path, _assistant("second answer arrived later"))
    second = transcript_parser.parse_transcript_incremental(
        str(path), "s", start_offset=first.resume_offset)

    assert len(second.turns) == 1
    rebuilt = second.turns[0]
    assert rebuilt["turn_index"] == partial["turn_index"]
    assert rebuilt["doc_id"] != partial["doc_id"]
    assert "second answer arrived later" in rebuilt["text"]
    assert second.resume_offset == FIRST_LEN


def test_short_open_turn_still_resumes_at_its_user_line(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("go"))

    result = transcript_parser.parse_transcript_incremental(str(path), "s")

    assert [t["turn_index"] for t in result.turns] == [0]
    assert result.resume_offset == FIRST_LEN

    _append(path, _assistant("a reply long enough to index"))
    again = transcript_parser.parse_transcript_incremental(
        str(path), "s", start_offset=result.resume_offset)
    assert "a reply long enough to index" in again.turns[0]["text"]


def test_summary_after_last_turn_closes_it(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + json.dumps({"type": "summary", "summary": "Indexer chat"}) + "\n")

    result = transcript_parser.parse_transcript_incremental(str(path), "s")

    assert result.resume_offset == result.consumed_offset == path.stat().st_size


def test_truncated_tail_resumes_and_consumes_at_the_open_user_line(tmp_path):
    path = tmp_path / "s.jsonl"
    user2 = _user("second question")
    reply = _assistant("second answer")
    _write(path, FIRST + user2 + reply[: len(reply) // 2])

    result = transcript_parser.parse_transcript_incremental(str(path), "s")

    assert result.resume_offset == result.consumed_offset == FIRST_LEN
    assert [t["turn_index"] for t in result.turns] == [0]


def test_parse_transcript_still_returns_turns_and_consumed_offset(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("trailing question with no reply yet"))

    turns, offset = transcript_parser.parse_transcript(str(path), "s")

    assert offset == path.stat().st_size
    assert len(turns) == 2


def test_parser_turns_opt_into_replacement_and_summaries_do_not(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + json.dumps({"type": "summary", "summary": "Indexer chat"}) + "\n")

    turns = transcript_parser.parse_transcript_incremental(str(path), "s").turns

    by_type = {t["chunk_type"]: t for t in turns}
    assert by_type["turn"][transcript_parser.REPLACE_SAME_TURN] is True
    assert transcript_parser.REPLACE_SAME_TURN not in by_type["summary"]


# --- Index state ------------------------------------------------------------


def test_state_keeps_resume_and_consumed_offsets():
    state = {}
    transcript_parser.set_transcript_offset(state, "/t.jsonl", 10, consumed_offset=50)

    assert transcript_parser.get_transcript_offset(state, "/t.jsonl") == 10
    assert transcript_parser.get_consumed_offset(state, "/t.jsonl") == 50


def test_consumed_offset_defaults_to_the_resume_offset():
    state = {"transcripts": {"/t.jsonl": {"last_byte_offset": 42}}}

    assert transcript_parser.get_consumed_offset(state, "/t.jsonl") == 42


# --- Claude provider cursor ---------------------------------------------------


def _source(path):
    return ProviderSource(
        provider="claude_code_cli", source_kind="claude_code_jsonl", source_class="native",
        source_id="claude_code_cli:s", logical_session_id="s", path=str(path),
        canonical_path=str(path), project_root="/p", timestamp="", status="eligible")


def test_provider_cursor_resumes_at_the_open_turn(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("second question still running tools"))
    adapter = ClaudeCodeCliAdapter(projects_root=tmp_path)

    parsed = adapter.parse_source(_source(path), cursor=None)

    assert parsed.cursor["last_byte_offset"] == FIRST_LEN
    assert parsed.cursor["consumed_offset"] == path.stat().st_size


def test_provider_skips_an_unchanged_file(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, FIRST + _user("second question still running tools"))
    adapter = ClaudeCodeCliAdapter(projects_root=tmp_path)
    cursor = adapter.parse_source(_source(path), cursor=None).cursor

    again = adapter.parse_source(_source(path), cursor=cursor)

    assert again.turns == []
    assert again.cursor == cursor


# --- add_turns replacement ----------------------------------------------------


@pytest.fixture
def index(monkeypatch):
    """In-memory Milvus and FTS stand-ins that keep rows by doc_id."""
    rows: dict[str, dict] = {}
    deleted: list[str] = []

    class _Decision:
        allowed = True
        retry_after_seconds = 0
        reason = ""

    class _Budget:
        def split_batches(self, turns):
            return [turns] if turns else []

        def before_batch(self, **kw):
            return _Decision()

        def after_batch(self, *a, **k):
            pass

    class _Client:
        def query(self, collection_name, filter, output_fields=None, limit=None):
            if filter.startswith("doc_id == "):
                doc_id = filter.split('"')[1]
                return [{"doc_id": doc_id}] if doc_id in rows else []
            return [dict(r) for r in rows.values() if _matches(r, filter)]

        def insert(self, collection_name, data):
            for d in data:
                rows[d["doc_id"]] = d

        def has_collection(self, name):
            return True

    def _matches(row, expr):
        # Only the stale-turn lookup issues other filters.
        sid = expr.split('session_id == "')[1].split('"')[0]
        indexes = json.loads(expr.split("turn_index in ")[1].split("]")[0] + "]")
        return (row["session_id"] == sid and row["chunk_type"] == "turn"
                and row["turn_index"] in indexes)

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    def _delete(doc_id, db_path=None):
        deleted.append(doc_id)
        rows.pop(doc_id, None)
        return rag_engine.DeleteResult(1, True)

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "embed_texts",
                        lambda texts, is_query=False: [[0.0] * rag_engine._EMBED_DIM for _ in texts])
    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _Budget())
    monkeypatch.setattr(rag_engine, "delete_by_doc_id", _delete)
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(), insert=lambda conn, records: None,
        close_ephemeral=lambda conn: None))
    return rows, deleted


def _turn(doc_id, turn_index, text, session="s", replace=True, chunk_type="turn"):
    t = {"doc_id": doc_id, "text": text, "session_id": session,
         "turn_index": turn_index, "chunk_type": chunk_type}
    if replace:
        t[transcript_parser.REPLACE_SAME_TURN] = True
    return t


PARTIAL = "User: second question still running tools"
FULL = PARTIAL + "\n\nAssistant: second answer arrived later"


def test_rebuilt_turn_replaces_the_partial_row(index):
    rows, deleted = index
    rag_engine.add_turns([_turn("s::partial", 7, PARTIAL)], db_path=DB)

    rag_engine.add_turns([_turn("s::full", 7, FULL)], db_path=DB)

    assert set(rows) == {"s::full"}
    assert deleted == ["s::partial"]


def test_replacement_runs_when_the_full_turn_is_already_indexed(index):
    rows, deleted = index
    rag_engine.add_turns([_turn("s::partial", 7, PARTIAL)], db_path=DB)
    rag_engine.add_turns([_turn("s::full", 7, FULL, replace=False)], db_path=DB)

    rag_engine.add_turns([_turn("s::full", 7, FULL)], db_path=DB)

    assert set(rows) == {"s::full"}
    assert deleted == ["s::partial"]


def test_other_turns_and_sessions_are_untouched(index):
    rows, deleted = index
    rag_engine.add_turns([
        _turn("s::other-turn", 3, "User: an earlier turn in the same session"),
        _turn("t::same-index", 7, "User: same turn_index in another session", session="t"),
        _turn("s::partial", 7, PARTIAL),
    ], db_path=DB)

    rag_engine.add_turns([_turn("s::full", 7, FULL)], db_path=DB)

    assert set(rows) == {"s::other-turn", "t::same-index", "s::full"}
    assert deleted == ["s::partial"]


def test_summary_chunk_at_the_same_offset_is_kept(index):
    rows, deleted = index
    rag_engine.add_turns([_turn("s::summary", 7, "Session Summary: indexer chat",
                                replace=False, chunk_type="summary")], db_path=DB)

    rag_engine.add_turns([_turn("s::full", 7, FULL)], db_path=DB)

    assert set(rows) == {"s::summary", "s::full"}
    assert deleted == []


def test_turns_without_the_flag_never_replace(index):
    rows, deleted = index
    rag_engine.add_turns([_turn("c::a", 7, "User: codex row one", replace=False)], db_path=DB)

    rag_engine.add_turns([_turn("c::b", 7, "User: codex row two", replace=False)], db_path=DB)

    assert set(rows) == {"c::a", "c::b"}
    assert deleted == []
