"""SESF-6 red-phase tests for Codex rollout JSONL ingestion.

Requirements: 2, 5, 6.
"""


def test_codex_adapter_dedups_active_and_archived_paths_by_logical_session(synthetic_codex_home):
    from provider_codex import CodexAdapter

    adapter = CodexAdapter(home=synthetic_codex_home)
    sources = adapter.discover_sources()

    logical_sources = [src for src in sources if src.logical_session_id == "synthetic-session"]
    assert len(logical_sources) == 1
    assert logical_sources[0].provider == "codex"
    assert logical_sources[0].source_kind == "codex_rollout_jsonl"

    parsed = adapter.parse_source(logical_sources[0], cursor=None)
    doc_ids = [turn["doc_id"] for turn in parsed.turns]
    assert parsed.turns
    assert len(doc_ids) == len(set(doc_ids))
    assert set(parsed.cursor["known_paths"]) >= {
        str(synthetic_codex_home / ".codex" / "sessions" / "2026" / "05" / "21" / "rollout-synthetic-session.jsonl"),
        str(synthetic_codex_home / ".codex" / "archived_sessions" / "rollout-synthetic-session.jsonl"),
    }


def test_codex_parse_falls_back_when_known_paths_cache_is_stale(synthetic_codex_home):
    """Sharing one CodexAdapter instance across discover/parse pairs (and
    between threads) means _known_paths can be wiped by a concurrent
    discover_sources(). parse_source() must not crash or silently drop the
    session — fall back to the source's own path."""
    from provider_codex import CodexAdapter

    adapter = CodexAdapter(home=synthetic_codex_home)
    source = next(s for s in adapter.discover_sources() if s.logical_session_id == "synthetic-session")

    # Simulate a second discover pass clearing the cache mid-flight.
    adapter._known_paths = {}

    parsed = adapter.parse_source(source, cursor=None)
    assert parsed.turns, "fallback should still parse the source file"
    assert parsed.cursor["known_paths"], "cursor still records at least the fallback path"


def test_codex_adapter_marks_unknown_project_scope_when_cwd_missing(tmp_path):
    from provider_codex import CodexAdapter

    home = tmp_path / "home"
    root = home / ".codex" / "sessions" / "2026" / "05" / "21"
    root.mkdir(parents=True)
    (root / "rollout-no-cwd.jsonl").write_text('{"session_id":"no-cwd","type":"response_item","content":"synthetic"}\n')

    source = CodexAdapter(home=home).discover_sources()[0]

    assert source.project_root in {"", "unknown", "/"}
    assert source.status == "eligible"


REWRITE_TS = "2026-10-26T14:51:34.961Z"


def _write_rewritten_rollout(tmp_path, lines):
    import json

    home = tmp_path / "home"
    root = home / ".codex" / "sessions" / "2026" / "10" / "26"
    root.mkdir(parents=True)
    path = root / "rollout-2026-10-26T09-51-34-rewritten.jsonl"
    path.write_text("".join(json.dumps({"timestamp": REWRITE_TS, **line}) + "\n" for line in lines))
    return home


def _msg(role, text):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": [{"type": "text", "text": text}]}}


def _task_started(epoch):
    return {"type": "event_msg", "payload": {"type": "task_started", "started_at": epoch}}


def _session_meta(branch="dev"):
    return {"type": "session_meta", "payload": {
        "id": "rewritten", "cwd": "/repo", "timestamp": REWRITE_TS,
        "git": {"commit_hash": "abc", "branch": branch},
    }}


def test_codex_turns_take_task_start_time_over_rewritten_line_time(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _task_started(1792982930),  # 2026-10-26T02:48:50Z
        _msg("user", "first question"),
        _msg("assistant", "first answer"),
        _msg("assistant", "first follow-up"),
        {"type": "event_msg", "payload": {"type": "task_complete", "started_at": 1792982930}},
        _task_started(1792990956),  # 2026-10-26T05:02:36Z
        _msg("user", "second question"),
        _msg("assistant", "second answer"),
    ])
    adapter = CodexAdapter(home=home)
    source = adapter.discover_sources()[0]
    turns = adapter.parse_source(source, cursor=None).turns

    assert [t["timestamp"][:19] for t in turns] == [
        "2026-10-26T02:48:50",
        "2026-10-26T02:48:50",
        "2026-10-26T05:02:36",
    ]


def test_codex_turns_carry_session_meta_git_branch(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(branch="feature/x"),
        _task_started(1792982930),
        _msg("user", "q"),
        _msg("assistant", "a"),
    ])
    adapter = CodexAdapter(home=home)
    turns = adapter.parse_source(adapter.discover_sources()[0], cursor=None).turns

    assert turns and all(t["git_branch"] == "feature/x" for t in turns)


def test_codex_resumed_parse_does_not_pair_skipped_user_with_next_answer(tmp_path):
    """A turn skipped via the cursor must still consume its user message."""
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("assistant", "a2"),
    ])
    adapter = CodexAdapter(home=home)
    source = adapter.discover_sources()[0]
    full = adapter.parse_source(source, cursor=None).turns
    assert [t["text"] for t in full] == ["User: q1\n\nAssistant: a1", "a2"]

    resumed = adapter.parse_source(source, cursor={"emitted_ids": [full[0]["doc_id"]]}).turns

    assert [(t["doc_id"], t["text"]) for t in resumed] == [(full[1]["doc_id"], "a2")]


def _turn_started(turn_id, epoch=1792982930):
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id, "started_at": epoch}}


def test_codex_skips_external_import_turns(tmp_path):
    """Imported turns copy another harness's transcript, which is indexed from its own source."""
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _turn_started("external-import-turn-1"),
        _msg("user", "imported question"),
        _msg("assistant", "[external_agent_tool_call: Read]\nfile: x\n[/external_agent_tool_call]"),
        _msg("assistant", "imported answer"),
        _turn_started("external-import-turn-2"),
        _msg("user", "second imported question"),
        _msg("assistant", "second imported answer"),
    ])
    adapter = CodexAdapter(home=home)
    turns = adapter.parse_source(adapter.discover_sources()[0], cursor=None).turns

    assert turns == []


def test_codex_keeps_native_turns_after_imported_ones(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _turn_started("external-import-turn-1"),
        _msg("user", "imported question"),
        _msg("assistant", "imported answer"),
        _turn_started("01a173b9-8e9e-7373-874e-0a10c4c3f01c"),
        _msg("user", "native question"),
        _msg("assistant", "native answer"),
    ])
    adapter = CodexAdapter(home=home)
    turns = adapter.parse_source(adapter.discover_sources()[0], cursor=None).turns

    assert [t["text"] for t in turns] == ["User: native question\n\nAssistant: native answer"]


def test_codex_imported_user_message_is_not_paired_with_native_answer(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _turn_started("external-import-turn-1"),
        _msg("user", "imported question"),
        _turn_started("01a173b9-8e9e-7373-874e-0a10c4c3f01c"),
        _msg("assistant", "native answer"),
    ])
    adapter = CodexAdapter(home=home)
    turns = adapter.parse_source(adapter.discover_sources()[0], cursor=None).turns

    assert [t["text"] for t in turns] == ["native answer"]


def test_codex_resumed_parse_still_skips_external_import_turns(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _turn_started("external-import-turn-1"),
        _msg("user", "imported question"),
        _msg("assistant", "imported answer"),
        _turn_started("native-1"),
        _msg("user", "q"),
        _msg("assistant", "a"),
    ])
    adapter = CodexAdapter(home=home)
    source = adapter.discover_sources()[0]
    full = adapter.parse_source(source, cursor=None)

    resumed = adapter.parse_source(source, cursor=full.cursor).turns

    assert [t["text"] for t in full.turns] == ["User: q\n\nAssistant: a"]
    assert resumed == []


def test_codex_turns_fall_back_to_line_time_without_task_events(tmp_path):
    from provider_codex import CodexAdapter

    home = _write_rewritten_rollout(tmp_path, [
        _session_meta(),
        _msg("user", "q"),
        _msg("assistant", "a"),
    ])
    adapter = CodexAdapter(home=home)
    turns = adapter.parse_source(adapter.discover_sources()[0], cursor=None).turns

    assert turns[0]["timestamp"][:19] == "2026-10-26T14:51:34"
