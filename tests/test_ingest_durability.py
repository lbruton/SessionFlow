"""SESF-58: ingestion durability.

Half-written trailing JSONL lines must not advance the resume cursor, and
``save_index_state`` must replace ``index_state.json`` atomically.
"""

import json
import os

import pytest

import transcript_parser


def _user(text, ts="2026-10-20T00:00:00Z"):
    return json.dumps({
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }, ensure_ascii=False) + "\n"


def _assistant(text, ts="2026-10-20T00:00:01Z"):
    return json.dumps({
        "type": "assistant",
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }, ensure_ascii=False) + "\n"


def _write(path, data: str):
    path.write_bytes(data.encode("utf-8"))


def _append(path, data: bytes):
    with open(path, "ab") as f:
        f.write(data)


FIRST_TURN = (
    _user("first question about the indexer")
    + _assistant("first answer about the indexer")
)


# --- Half-written trailing line ---------------------------------------------


def test_truncated_user_line_is_not_skipped(tmp_path):
    path = tmp_path / "s.jsonl"
    second = _user("second question that is still being written")
    cut = len(second) // 2
    _write(path, FIRST_TURN + second[:cut])

    turns, offset = transcript_parser.parse_transcript(str(path), "s")

    # The first turn is still open (the partial line could be more of its
    # reply), so nothing is emitted and the cursor rewinds to its user line
    # rather than skipping past the half-written line.
    assert offset == 0
    assert turns == []

    _append(path, second[cut:].encode("utf-8") + _assistant("second answer").encode("utf-8"))
    turns2, offset2 = transcript_parser.parse_transcript(str(path), "s", start_offset=offset)

    assert offset2 == path.stat().st_size
    assert len(turns2) == 2
    assert "first answer" in turns2[0]["text"]
    assert "second question that is still being written" in turns2[1]["text"]
    assert "second answer" in turns2[1]["text"]
    assert turns2[1]["turn_index"] == len(FIRST_TURN.encode("utf-8"))


def test_truncated_assistant_reply_keeps_pending_turn_for_next_parse(tmp_path):
    path = tmp_path / "s.jsonl"
    user2 = _user("second question awaiting its reply")
    reply = _assistant("second answer arriving late")
    cut = len(reply) // 2
    _write(path, FIRST_TURN + user2 + reply[:cut])

    turns, offset = transcript_parser.parse_transcript(str(path), "s")

    # The pending user turn must not be flushed without its reply; the cursor
    # rewinds to the start of its user line so the next parse rebuilds it.
    user2_start = len(FIRST_TURN.encode("utf-8"))
    assert offset == user2_start
    assert len(turns) == 1
    assert "first question" in turns[0]["text"]

    _append(path, reply[cut:].encode("utf-8"))
    turns2, offset2 = transcript_parser.parse_transcript(str(path), "s", start_offset=offset)

    assert offset2 == path.stat().st_size
    assert len(turns2) == 1
    assert "second question awaiting its reply" in turns2[0]["text"]
    assert "second answer arriving late" in turns2[0]["text"]
    assert turns2[0]["turn_index"] == user2_start


def test_truncated_inside_multibyte_utf8_does_not_raise(tmp_path):
    path = tmp_path / "s.jsonl"
    second = _user("second question with a snowman ☃ in it").encode("utf-8")
    snowman = "☃".encode("utf-8")
    cut = second.index(snowman) + 1  # split the 3-byte character
    path.write_bytes(FIRST_TURN.encode("utf-8") + second[:cut])

    turns, offset = transcript_parser.parse_transcript(str(path), "s")

    assert offset == 0
    assert turns == []

    _append(path, second[cut:])
    turns2, offset2 = transcript_parser.parse_transcript(str(path), "s", start_offset=offset)
    assert offset2 == path.stat().st_size
    assert len(turns2) == 2
    assert "☃" in turns2[1]["text"]


def test_only_partial_line_returns_start_offset(tmp_path):
    path = tmp_path / "s.jsonl"
    partial = _user("nothing complete yet")[:10]
    _write(path, FIRST_TURN + partial)
    start = len(FIRST_TURN.encode("utf-8"))

    turns, offset = transcript_parser.parse_transcript(str(path), "s", start_offset=start)

    assert turns == []
    assert offset == start


# --- Complete files are unchanged -------------------------------------------


def test_complete_file_parses_all_turns_to_eof(tmp_path):
    path = tmp_path / "s.jsonl"
    body = (
        FIRST_TURN
        + json.dumps({"type": "summary", "summary": "Indexer chat"}) + "\n"
        + "{not valid json but newline terminated}\n"
        + _user("trailing question with no reply yet")
    )
    _write(path, body)

    turns, offset = transcript_parser.parse_transcript(str(path), "s")

    assert offset == path.stat().st_size
    texts = [t["text"] for t in turns]
    assert any("first answer" in t for t in texts)
    assert "Session Summary: Indexer chat" in texts
    # A complete trailing user line is still flushed at EOF, as before.
    assert any("trailing question with no reply yet" in t for t in texts)


def test_complete_file_incremental_matches_single_pass(tmp_path):
    path = tmp_path / "s.jsonl"
    part1 = FIRST_TURN
    part2 = _user("second question here") + _assistant("second answer here")
    _write(path, part1 + part2)

    full_turns, full_offset = transcript_parser.parse_transcript(str(path), "s")
    assert full_offset == path.stat().st_size
    assert len(full_turns) == 2

    resumed, end = transcript_parser.parse_transcript(
        str(path), "s", start_offset=len(part1.encode("utf-8")))
    assert end == full_offset
    assert [t["doc_id"] for t in resumed] == [full_turns[-1]["doc_id"]]
    assert resumed[0]["turn_index"] == len(part1.encode("utf-8"))


# --- Atomic index_state.json ------------------------------------------------


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    state_dir = tmp_path / ".sessionflow"
    path = state_dir / "index_state.json"
    monkeypatch.setattr(transcript_parser, "_STATE_DIR", state_dir)
    monkeypatch.setattr(transcript_parser, "_STATE_PATH", path)
    return path


def test_save_index_state_writes_via_temp_and_replace(state_path, monkeypatch):
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    transcript_parser.save_index_state({"transcripts": {"a": {"last_byte_offset": 5}}})

    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == str(state_path)
    assert src != dst
    assert os.path.dirname(src) == str(state_path.parent)
    assert json.loads(state_path.read_text())["transcripts"]["a"]["last_byte_offset"] == 5
    assert sorted(p.name for p in state_path.parent.iterdir()) == ["index_state.json"]


def test_save_index_state_failure_leaves_original_intact(state_path):
    original = {"transcripts": {"a": {"last_byte_offset": 42}}}
    transcript_parser.save_index_state(original)
    before = state_path.read_bytes()

    # Not JSON-serializable: the dump fails partway through the write.
    bad = {"transcripts": {"a": {"last_byte_offset": 99}}, "zzz": object()}
    with pytest.raises(TypeError):
        transcript_parser.save_index_state(bad)

    assert state_path.read_bytes() == before
    assert json.loads(before) == original
    assert sorted(p.name for p in state_path.parent.iterdir()) == ["index_state.json"]
