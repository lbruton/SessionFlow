"""Tests for harness boilerplate stripping (``boilerplate.strip``) and its ``add_turns`` hook.

Fixtures copy the shapes found in indexed transcripts: Claude Code and Codex turns are
``User: <text>\\n\\nAssistant: <text>``, and harness blocks open at the start of a line.
"""

import contextlib
import types

import pytest

import boilerplate


def test_task_notification_block_is_removed_and_reply_kept():
    text = (
        "User: <task-notification>\n<task-id>breyntkmr</task-id>\n"
        "<tool-use-id>toolu_016vFJGYc97Hg6VUcnNCzFt4</tool-use-id>\n"
        "<output-file>/private/tmp/x/tasks/breyntkmr.output</output-file>\n"
        "<status>completed</status>\n<summary>Background command finished</summary>\n"
        "</task-notification>\n\nAssistant: The prune deleted 27,254 rows."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: The prune deleted 27,254 rows."
    assert removed == {"task_notification": 1}


def test_side_sessions_event_is_removed():
    text = (
        "User: <side-sessions-event>None of the sessions you started is working now."
        "</side-sessions-event>\n\nAssistant: Both sessions finished; merging next."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Both sessions finished; merging next."
    assert removed == {"side_sessions_event": 1}


def test_ci_monitor_event_is_removed():
    text = (
        "User: <ci-monitor-event>lbruton/SpecFlow PR #10 has 1 new review comment:\n"
        "- coderabbitai[bot]: <!-- auto-generated -->\n</ci-monitor-event>\n\n"
        "Assistant: Fixed the review comment in abc123."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Fixed the review comment in abc123."
    assert removed == {"ci_monitor_event": 1}


def test_system_reminder_is_removed_after_a_tool_result_marker():
    text = (
        "Assistant: [external_agent_tool_result]\n<system-reminder>This memory is 5 days old."
        "</system-reminder>\nThe file has 40 lines."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: [external_agent_tool_result]\nThe file has 40 lines."
    assert removed == {"system_reminder": 1}


def test_command_echo_becomes_a_marker_with_its_args():
    text = (
        "User: <command-message>sketch</command-message>\n<command-name>/sketch</command-name>\n"
        "<command-args>apply STRK-85</command-args>\n\nAssistant: Applying the sketch."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "User: [/sketch apply STRK-85]\n\nAssistant: Applying the sketch."
    assert removed == {"command_echo": 1}


def test_command_echo_without_args_keeps_the_name():
    text = "User: <command-message>start</command-message>\n<command-name>/start</command-name>\n\nAssistant: Session start."
    clean, _ = boilerplate.strip(text)
    assert clean == "User: [/start]\n\nAssistant: Session start."


def test_indented_command_echo_with_empty_args_becomes_one_marker():
    text = (
        "User: <command-name>/plugin</command-name>\n            <command-message>plugin</command-message>\n"
        "            <command-args></command-args>"
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "User: [/plugin]"
    assert removed == {"command_echo": 1}


def test_local_command_output_and_caveat_are_removed():
    text = (
        "User: <local-command-caveat>The command below was run directly.</local-command-caveat>\n"
        "<local-command-stdout>Set model to claude-sonnet-4-6</local-command-stdout>\n\n"
        "Assistant: Model switched; continuing the refactor."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Model switched; continuing the refactor."
    assert removed == {"local_command": 2}


def test_bash_input_is_kept_and_its_output_removed():
    text = (
        "User: <bash-input>npx agentlinter --json</bash-input>\n"
        "<bash-stdout>{\n  \"score\": 79\n}</bash-stdout><bash-stderr></bash-stderr>\n\n"
        "Assistant: The linter scored 79."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "User: [! npx agentlinter --json]\n\nAssistant: The linter scored 79."
    assert removed == {"bash_output": 2}


def test_compaction_summary_user_part_is_removed():
    text = (
        "User: This session is being continued from a previous conversation that ran out of "
        "context. The summary below covers the earlier portion of the conversation.\n\n"
        "Summary:\n1. Primary Request and Intent:\n   The user invoked /sketch.\n\n"
        "Assistant: Resuming the sketch at task 4."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Resuming the sketch at task 4."
    assert removed == {"compaction_summary": 1}


def test_relay_wrappers_are_removed():
    text = (
        "User: <wake>scheduled check-in</wake>\n<project_claude_message>From CoWork: ping"
        "</project_claude_message>\nPlease check the CoWork backlog.\n\nAssistant: Checking it."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "User: Please check the CoWork backlog.\n\nAssistant: Checking it."
    assert removed == {"wake": 1, "project_claude_message": 1}


def test_unclosed_block_from_a_truncated_turn_is_removed_to_the_section_end():
    text = (
        "User: <task-notification>\n<task-id>x</task-id>\n<summary>long text cut off\n\n"
        "Assistant: Reading the output file now."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Reading the output file now."
    assert removed == {"task_notification": 1}


def test_nested_reminder_inside_an_event_is_removed_with_it():
    text = (
        "User: <ci-monitor-event>PR #87 has comments\n<system-reminder>ignore</system-reminder>\n"
        "</ci-monitor-event>\n\nAssistant: Replied to the comments."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: Replied to the comments."
    assert removed == {"ci_monitor_event": 1}


def test_tag_mentioned_in_prose_is_kept():
    text = (
        "User: Why do results show `<wake>` and <system-reminder> wrappers?\n\n"
        "Assistant: The parser keeps the `<task-notification>` text as written."
    )
    clean, removed = boilerplate.strip(text)
    assert clean == text
    assert removed == {}


def test_text_without_boilerplate_is_unchanged():
    text = "User: How do I restart the server?\n\nAssistant: Run launchctl kickstart -k."
    assert boilerplate.strip(text) == (text, {})


def test_turn_that_is_only_boilerplate_strips_to_empty():
    text = "User: <task-notification>\n<task-id>x</task-id>\n</task-notification>"
    clean, removed = boilerplate.strip(text)
    assert clean == ""
    assert removed == {"task_notification": 1}


def test_repeated_blocks_are_all_counted():
    text = (
        "Assistant: one\n<system-reminder>a</system-reminder>\ntwo\n"
        "<system-reminder>b</system-reminder>\nthree"
    )
    clean, removed = boilerplate.strip(text)
    assert clean == "Assistant: one\ntwo\nthree"
    assert removed == {"system_reminder": 2}


# === add_turns hook =======================================================

DB = "/tmp/sessionflow-boilerplate-test.db"  # mocked; never touched on disk


@pytest.fixture
def harness(monkeypatch):
    """Run ``add_turns`` with embed, Milvus and FTS mocked and capture each sink."""
    import rag_engine

    for var in ("SESSIONFLOW_STRIP_BOILERPLATE", "SESSIONFLOW_REDACT", "SESSIONFLOW_REDACT_MODE"):
        monkeypatch.delenv(var, raising=False)
    cap = {"embed_inputs": [], "milvus_data": [], "fts_records": []}

    def fake_embed_texts(texts, is_query=False):
        cap["embed_inputs"].extend(texts)
        return [[0.0] * rag_engine._EMBED_DIM for _ in texts]

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
        def query(self, **kw):
            return []

        def insert(self, collection_name, data):
            cap["milvus_data"].extend(data)

        def has_collection(self, name):
            return True

    @contextlib.contextmanager
    def _fake_milvus_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _Budget())
    monkeypatch.setattr(rag_engine, "milvus_client", _fake_milvus_client)
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(),
        insert=lambda conn, records: cap["fts_records"].extend(records),
        close_ephemeral=lambda conn: None,
    ))
    monkeypatch.setattr(rag_engine, "_query_all", lambda *a, **k: [])
    monkeypatch.setattr(
        rag_engine, "fts_lag_status",
        lambda **k: {"fts_row_count": 0, "fts_lag": 0, "fts_backfill_required": False},
    )
    rag_engine._boilerplate_counters.clear()
    monkeypatch.setattr(rag_engine, "_boilerplate_dropped", 0)
    return rag_engine, cap


CI_TURN = (
    "User: <ci-monitor-event>PR #88 for SESF-99 has a comment</ci-monitor-event>\n\n"
    "Assistant: Fixed the parser for SESF-66 and pushed."
)


def test_add_turns_strips_every_sink(harness):
    rag_engine, cap = harness
    rag_engine.add_turns([{"doc_id": "d1", "text": CI_TURN}], db_path=DB)

    expected = "Assistant: Fixed the parser for SESF-66 and pushed."
    assert cap["embed_inputs"] == [expected]
    assert cap["milvus_data"][0]["document"] == expected
    assert cap["fts_records"][0]["content"] == expected


def test_add_turns_extracts_issue_ids_from_stripped_text(harness):
    rag_engine, cap = harness
    rag_engine.add_turns([{"doc_id": "d1", "text": CI_TURN}], db_path=DB)

    assert cap["milvus_data"][0]["issue_ids"] == ",SESF-66,"
    assert cap["fts_records"][0]["issue_ids"] == ",SESF-66,"


def test_add_turns_strips_before_redaction(harness, monkeypatch):
    rag_engine, _ = harness
    seen = []
    monkeypatch.setattr(rag_engine, "_apply_redaction", lambda turns: seen.extend(t["text"] for t in turns))
    rag_engine.add_turns([{"doc_id": "d1", "text": CI_TURN}], db_path=DB)

    assert seen == ["Assistant: Fixed the parser for SESF-66 and pushed."]


def test_add_turns_drops_a_turn_left_under_the_length_floor(harness):
    rag_engine, cap = harness
    only_noise = "User: <task-notification>\n<task-id>x</task-id>\n</task-notification>"
    stored = rag_engine.add_turns(
        [{"doc_id": "noise", "text": only_noise}, {"doc_id": "real", "text": CI_TURN}], db_path=DB,
    )

    assert stored == 1
    assert [row["doc_id"] for row in cap["milvus_data"]] == ["real"]
    assert [row["doc_id"] for row in cap["fts_records"]] == ["real"]


def test_add_turns_keeps_text_when_stripping_is_off(harness, monkeypatch):
    rag_engine, cap = harness
    monkeypatch.setenv("SESSIONFLOW_STRIP_BOILERPLATE", "off")
    rag_engine.add_turns([{"doc_id": "d1", "text": CI_TURN}], db_path=DB)

    assert cap["milvus_data"][0]["document"] == CI_TURN


def test_get_stats_reports_boilerplate_counts(harness):
    rag_engine, _ = harness
    rag_engine.add_turns([{"doc_id": "d1", "text": CI_TURN}], db_path=DB)

    surface = rag_engine.get_stats()["boilerplate"]
    assert surface == {"enabled": True, "counts": {"ci_monitor_event": 1}, "dropped_turns": 0}
