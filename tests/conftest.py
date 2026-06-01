"""Shared fixtures for SessionFlow tests."""

import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

# Add project root to sys.path so `import http_server` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_server_dir(tmp_path):
    """
    Create a temporary `.sessionflow` directory under the provided `tmp_path`.
    
    Returns:
        pathlib.Path: Path to the created `.sessionflow` directory.
    """
    server_dir = tmp_path / ".sessionflow"
    server_dir.mkdir()
    return server_dir


@pytest.fixture
def mock_heartbeat_file(tmp_server_dir):
    """
    Create a heartbeat JSON file with the current timestamp, current process PID, and activity "idle" inside the provided server directory.
    
    Parameters:
        tmp_server_dir (pathlib.Path): Directory in which to create the `heartbeat` file.
    
    Returns:
        pathlib.Path: Path to the created `heartbeat` file.
    """
    hb_path = tmp_server_dir / "heartbeat"
    data = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "activity": "idle",
    }
    hb_path.write_text(json.dumps(data))
    return hb_path


@pytest.fixture
def stale_heartbeat_file(tmp_server_dir):
    """
    Create a heartbeat file with its timestamp set 300 seconds in the past.
    
    Parameters:
        tmp_server_dir (pathlib.Path): Directory in which to create the `heartbeat` file.
    
    Returns:
        pathlib.Path: Path to the created `heartbeat` file containing JSON with keys `timestamp` (float), `pid` (int), and `activity` (str).
    """
    hb_path = tmp_server_dir / "heartbeat"
    data = {
        "timestamp": time.time() - 300,
        "pid": os.getpid(),
        "activity": "idle",
    }
    hb_path.write_text(json.dumps(data))
    return hb_path


@pytest.fixture
def mock_pid_file(tmp_server_dir):
    """
    Create a server PID file named `server.pid` inside the given directory containing the current process PID.
    
    Parameters:
        tmp_server_dir (pathlib.Path): Directory in which to create the `server.pid` file.
    
    Returns:
        pathlib.Path: Path to the created `server.pid` file.
    """
    pid_path = tmp_server_dir / "server.pid"
    pid_path.write_text(str(os.getpid()))
    return pid_path


@pytest.fixture
def script_path():
    """
    Compute the absolute filesystem path to the `sessionflow-server.sh` script located at the project root.
    
    Returns:
        script_path (str): Absolute path to `sessionflow-server.sh`.
    """
    return str(Path(__file__).resolve().parent.parent / "sessionflow-server.sh")


@pytest.fixture
def synthetic_claude_jsonl(tmp_path):
    """Create a small Claude Code transcript with no real user transcript content."""
    transcript = tmp_path / "claude-session.jsonl"
    entries = [
        {
            "type": "user",
            "cwd": str(tmp_path),
            "gitBranch": "main",
            "timestamp": "2026-05-21T10:00:00Z",
            "message": {"content": "synthetic user request"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-21T10:00:01Z",
            "message": {"content": [{"type": "text", "text": "synthetic assistant response"}]},
        },
    ]
    transcript.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return transcript


@pytest.fixture
def synthetic_codex_home(tmp_path):
    """Create active and archived Codex rollout files for one logical session."""
    home = tmp_path / "home"
    active = home / ".codex" / "sessions" / "2026" / "05" / "21"
    archive = home / ".codex" / "archived_sessions"
    active.mkdir(parents=True)
    archive.mkdir(parents=True)
    active_rollout = active / "rollout-synthetic-session.jsonl"
    archived_rollout = archive / "rollout-synthetic-session.jsonl"
    lines = [
        {
            "timestamp": "2026-05-21T10:00:00Z",
            "type": "session_meta",
            "payload": {"id": "synthetic-session", "cwd": str(tmp_path)},
        },
        {
            "timestamp": "2026-05-21T10:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "synthetic codex question"}],
            },
        },
        {
            "timestamp": "2026-05-21T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "synthetic codex answer"}],
            },
        },
    ]
    payload = "\n".join(json.dumps(line) for line in lines) + "\n"
    active_rollout.write_text(payload)
    archived_rollout.write_text(payload)
    return home


@pytest.fixture
def synthetic_opencode_storage(tmp_path):
    """Create a minimal OpenCode storage tree with one message and one part."""
    storage = tmp_path / ".local" / "share" / "opencode" / "storage"
    for name in ("session", "message", "part"):
        (storage / name).mkdir(parents=True)
    (storage / "session" / "synthetic-session.json").write_text(json.dumps({
        "id": "synthetic-session",
        "cwd": str(tmp_path),
        "time": {"created": "2026-05-21T10:00:00Z"},
    }))
    (storage / "message" / "synthetic-message.json").write_text(json.dumps({
        "id": "synthetic-message",
        "sessionID": "synthetic-session",
        "role": "user",
        "time": {"created": "2026-05-21T10:00:01Z"},
    }))
    part = storage / "part" / "synthetic-part.json"
    part.write_text(json.dumps({
        "id": "synthetic-part",
        "sessionID": "synthetic-session",
        "messageID": "synthetic-message",
        "type": "text",
        "text": "synthetic opencode content",
    }))
    return storage


@pytest.fixture
def synthetic_antigravity_home(tmp_path):
    """Create Antigravity CLI history plus JSONL transcript logs."""
    home = tmp_path / "home"
    brain = home / ".gemini" / "antigravity-cli" / "brain" / "synthetic-conversation"
    logs = brain / ".system_generated" / "logs"
    logs.mkdir(parents=True)
    history = home / ".gemini" / "antigravity-cli" / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps({
        "conversation_id": "synthetic-conversation",
        "workspace": str(tmp_path),
    }) + "\n")
    (logs / "transcript.jsonl").write_text("\n".join([
        json.dumps({"step_index": 1, "type": "USER_INPUT", "text": "synthetic antigravity ask"}),
        json.dumps({"step_index": 2, "type": "PLANNER_RESPONSE", "text": "synthetic antigravity plan"}),
    ]) + "\n")
    return home


def _pb_varint(value: int) -> bytes:
    """Encode a non-negative int as a protobuf base-128 varint."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _pb_len_delimited(tag: int, payload: bytes) -> bytes:
    """Frame `payload` as a wire-type-2 (length-delimited) field with `tag`."""
    return bytes([tag]) + _pb_varint(len(payload)) + payload


def _pb_varint_field(field_number: int, value: int) -> bytes:
    """Encode a wire-type-0 (varint) field: tag byte + varint value."""
    tag = (field_number << 3) | 0  # wire type 0
    return _pb_varint(tag) + _pb_varint(value)


def _pb_fixed32_field(field_number: int, value: int) -> bytes:
    """Encode a wire-type-5 (32-bit fixed) field: tag byte + 4 LE bytes."""
    import struct
    tag = (field_number << 3) | 5  # wire type 5
    return _pb_varint(tag) + struct.pack("<I", value & 0xFFFFFFFF)


def _pb_fixed64_field(field_number: int, value: int) -> bytes:
    """Encode a wire-type-1 (64-bit fixed) field: tag byte + 8 LE bytes."""
    import struct
    tag = (field_number << 3) | 1  # wire type 1
    return _pb_varint(tag) + struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)


def _wrap_record(record: bytes) -> bytes:
    """Frame a record body as a top-level `Field 1` (wt=2) length-delimited entry.

    The real Antigravity `.pb` is a sequence of repeated top-level `Field 1`
    records; the conversation UUID + workspace submessage live *inside* each
    record (SESF-29). `_pb_summaries_record` / `_pb_summaries_record_with_extra_wire_types`
    build the inner record body; this wraps one so it sits at the right depth.
    """
    return _pb_len_delimited(0x0A, record)


def _pb_summaries_record(conversation_id: str, workspace_uri: str) -> bytes:
    """
    Build one summaries record body mapping a conversation to a workspace URI.

    Wire layout (all wire type 2, length-delimited):
        Field 1 (0x0A) -> conversation UUID string
        Field 2 (0x12) -> submessage:
            Field 9 (0x4A) -> nested submessage:
                Field 1 (0x0A) -> workspace `file://` URI string
    """
    nested = _pb_len_delimited(0x0A, workspace_uri.encode("utf-8"))  # Field 1 (URI)
    field9 = _pb_len_delimited(0x4A, nested)                          # Field 9 (nested submessage)
    field2 = _pb_len_delimited(0x12, field9)                          # Field 2 (submessage)
    field1 = _pb_len_delimited(0x0A, conversation_id.encode("utf-8"))  # Field 1 (conversation UUID)
    return field1 + field2


def _pb_summaries_record_with_extra_wire_types(
    conversation_id: str, workspace_uri: str
) -> bytes:
    """
    Build a top-level summaries record that also includes non-length-delimited
    wire types interleaved, mimicking what the real Antigravity daemon emits.

    Extra fields injected (all unknown to our schema, must be silently skipped):
        - wire type 0 (varint): field 10 with a fake int64 timestamp value.
        - wire type 5 (fixed32): field 11 with a fake 32-bit flags value.
        - wire type 1 (fixed64): field 12 with a fake 64-bit monotonic clock.

    The conversation UUID (Field 1) and workspace submessage (Field 2) must
    still be decoded correctly after the fix, and the record must appear at
    field-order-swapped positions (Field 2 before Field 1) to exercise both
    wire-type skipping and field-order independence simultaneously.
    """
    nested = _pb_len_delimited(0x0A, workspace_uri.encode("utf-8"))
    field9 = _pb_len_delimited(0x4A, nested)
    field2 = _pb_len_delimited(0x12, field9)                           # workspace submessage FIRST
    varint_noise = _pb_varint_field(10, 1748649600000)                 # fake ms timestamp
    fixed32_noise = _pb_fixed32_field(11, 0xDEADBEEF)                 # fake 32-bit flags
    fixed64_noise = _pb_fixed64_field(12, 0xCAFEBABEDEADC0DE)        # fake 64-bit clock
    field1 = _pb_len_delimited(0x0A, conversation_id.encode("utf-8"))  # uuid SECOND
    # Layout: Field2 | varint_noise | fixed32_noise | fixed64_noise | Field1
    return field2 + varint_noise + fixed32_noise + fixed64_noise + field1


@pytest.fixture
def synthetic_antigravity_desktop_home(tmp_path):
    """
    Materialize a synthetic Antigravity *desktop* root for SESF-17 project-detection tests.

    Layout (under `<tmp>/home/.gemini/antigravity/`):
      - `brain/<uuid>/.system_generated/logs/transcript.jsonl` for four conversations.
      - A hand-built valid `agyhub_summaries_proto.pb` mapping a subset of those
        conversation_ids to `file://` workspace URIs via the Field 1 / Field 2->9->1
        wire structure.
      - One conversation present in `brain/` but absent from the `.pb` (AC-4 -> "unknown").
      - One record carrying a percent-encoded `file://` path (`%20` space + a unicode
        char) to exercise AC-2's `urllib.parse.unquote` decode branch.
      - A separately-addressable malformed/truncated `.pb` variant (`*.truncated`) — a
        short read of the valid bytes — for AC-5.

    Self-contained: does not read the live `~/.gemini` tree.

    Returns a dict:
      - `home`             pathlib.Path to the synthetic home (pass as `home=` to the adapter).
      - `root`             pathlib.Path to `.../.gemini/antigravity`.
      - `pb_path`          pathlib.Path to the valid `agyhub_summaries_proto.pb`.
      - `truncated_pb_path` pathlib.Path to the malformed/truncated `.pb` variant.
      - `mapped`           {conversation_id: decoded_workspace_path} for `.pb`-mapped convs.
      - `unmapped`         list of conversation_ids present in brain but absent from `.pb`.
      - `encoded_id`       conversation_id of the percent-encoded record.
      - `encoded_decoded`  the expected decoded filesystem path for `encoded_id`.
    """
    home = tmp_path / "home"
    root = home / ".gemini" / "antigravity"

    # Conversation ids (brain dir names). The first three are mapped in the `.pb`;
    # the fourth is intentionally absent from the `.pb` to exercise AC-4.
    # The fifth has a record that interleaves non-length-delimited wire types
    # (wt=0 varint, wt=5 fixed32, wt=1 fixed64) to exercise wire-type skipping.
    plain_id = "11111111-1111-1111-1111-111111111111"
    plain_ws = "/Volumes/DATA/GitHub/SessionFlow"
    second_id = "22222222-2222-2222-2222-222222222222"
    second_ws = "/Users/lbruton/Projects/Demo"
    encoded_id = "33333333-3333-3333-3333-333333333333"
    # Percent-encoded path: a literal space (%20) and a unicode char (café -> caf%C3%A9).
    encoded_decoded = "/Users/lbruton/My Projects/café"
    encoded_ws = "/Users/lbruton/My%20Projects/caf%C3%A9"
    unmapped_id = "44444444-4444-4444-4444-444444444444"
    mixed_wire_id = "55555555-5555-5555-5555-555555555555"
    mixed_wire_ws = "/Users/lbruton/Projects/MixedWire"

    brain_ids = [plain_id, second_id, encoded_id, unmapped_id, mixed_wire_id]
    for conversation_id in brain_ids:
        logs = root / "brain" / conversation_id / ".system_generated" / "logs"
        logs.mkdir(parents=True)
        (logs / "transcript.jsonl").write_text("\n".join([
            json.dumps({"step_index": 1, "type": "USER_INPUT", "text": "synthetic desktop ask"}),
            json.dumps({"step_index": 2, "type": "PLANNER_RESPONSE", "text": "synthetic desktop plan"}),
        ]) + "\n")

    # Hand-built valid `.pb` matching the REAL Antigravity framing (SESF-29):
    # the top level is a sequence of repeated `Field 1` records, and the UUID +
    # workspace live one level deeper *inside* each record. Each record body
    # (Field 1 = uuid, Field 2->9->1 = uri) is wrapped in a top-level Field 1
    # length-delimited frame via `_wrap_record`. Records: plain, second,
    # percent-encoded, plus one with mixed wire types (wt=0/1/5) and reversed
    # field order. The `file://` URIs carry the scheme prefix; AC-2's normalizer
    # strips it and `unquote`s the percent-encoded record back to `encoded_decoded`.
    pb_bytes = b"".join([
        _wrap_record(_pb_summaries_record(plain_id, "file://" + plain_ws)),
        _wrap_record(_pb_summaries_record(second_id, "file://" + second_ws)),
        _wrap_record(_pb_summaries_record(encoded_id, "file://" + encoded_ws)),
        _wrap_record(
            _pb_summaries_record_with_extra_wire_types(mixed_wire_id, "file://" + mixed_wire_ws)
        ),
        # Regression guard (SESF-29): a top-level `Field 2` "recency" entry,
        # mirroring the ~5 such entries in the real file (inner Field 1 = uuid
        # string, Field 2 = varint timestamp, NO workspace path). The walker must
        # only descend top-level `Field 1` records and IGNORE this — otherwise
        # `unmapped_id` would wrongly resolve to a path instead of "unknown" (AC-4).
        _pb_len_delimited(
            0x12,
            _pb_len_delimited(0x0A, unmapped_id.encode("utf-8"))
            + _pb_varint_field(2, 1779575744),
        ),
    ])
    pb_path = root / "agyhub_summaries_proto.pb"
    pb_path.write_bytes(pb_bytes)

    # Malformed/truncated variant: a short read of the valid bytes, slicing through
    # the middle of a length-delimited field so the walker hits a boundary error.
    truncated_pb_path = root / "agyhub_summaries_proto.pb.truncated"
    truncated_pb_path.write_bytes(pb_bytes[: len(pb_bytes) // 2])

    mapped = {
        plain_id: plain_ws,
        second_id: second_ws,
        encoded_id: encoded_decoded,
        mixed_wire_id: mixed_wire_ws,
    }

    return {
        "home": home,
        "root": root,
        "pb_path": pb_path,
        "truncated_pb_path": truncated_pb_path,
        "mapped": mapped,
        "unmapped": [unmapped_id],
        "encoded_id": encoded_id,
        "encoded_decoded": encoded_decoded,
        "mixed_wire_id": mixed_wire_id,
        "mixed_wire_ws": mixed_wire_ws,
    }


@pytest.fixture
def stub_rag_engine(monkeypatch):
    """Stub the heavy rag_engine import for tests that inspect CLI/server formatting."""
    async def add_turns_async(
        turns: list[dict[str, Any]],
        *args: object,
        **kwargs: object,
    ) -> int:
        return len(turns)

    async def get_issue_timeline_async(*args: object, **kwargs: object) -> list:
        # Mirror the real wrapper, which runs the sync core in an executor, so a
        # test that reassigns the sync `get_issue_timeline` is reflected here too.
        return module.get_issue_timeline(*args, **kwargs)

    module = types.SimpleNamespace(
        get_issue_timeline=lambda *args, **kwargs: [],
        get_issue_timeline_async=get_issue_timeline_async,
        search=lambda *args, **kwargs: [],
        get_turns=lambda *args, **kwargs: [],
        get_stats=lambda *args, **kwargs: {
            "total_turns": 0,
            "sessions": 0,
            "by_type": {},
            "providers": {},
        },
        delete_older_than=lambda *args, **kwargs: 0,
        delete_by_session=lambda *args, **kwargs: 0,
        delete_by_branch=lambda *args, **kwargs: 0,
        list_sessions=lambda *args, **kwargs: [],
        clear_collection=lambda *args, **kwargs: None,
        init_server_mode=lambda *args, **kwargs: None,
        close_server_mode=lambda *args, **kwargs: None,
        add_turns_async=add_turns_async,
        backfill_fts=lambda *args, **kwargs: 0,
        get_model=lambda *args, **kwargs: object(),
        get_model_name=lambda *args, **kwargs: "embeddinggemma",
    )
    monkeypatch.setitem(sys.modules, "rag_engine", module)
    return module
