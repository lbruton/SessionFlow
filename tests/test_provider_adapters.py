"""SESF-6 red-phase tests for shared provider contracts.

Requirements: 1, 5, 6, 8.
"""

from pathlib import Path

import pytest


def test_provider_source_validates_legal_metadata_and_canonical_path(tmp_path):
    from provider_adapters import (
        LEGAL_PROVIDERS,
        LEGAL_SOURCE_CLASSES,
        LEGAL_SOURCE_KINDS,
        ProviderSource,
    )

    assert {
        "claude_code_cli",
        "codex",
        "opencode",
        "antigravity_cli",
        "antigravity_desktop",
    }.issubset(LEGAL_PROVIDERS)
    assert "native" in LEGAL_SOURCE_CLASSES
    assert "codex_rollout_jsonl" in LEGAL_SOURCE_KINDS

    real_path = tmp_path / "source.jsonl"
    real_path.write_text("{}\n")
    link_path = tmp_path / "source-link.jsonl"
    link_path.symlink_to(real_path)

    source = ProviderSource(
        provider="codex",
        source_kind="codex_rollout_jsonl",
        source_class="native",
        source_id="codex:synthetic",
        logical_session_id="synthetic",
        path=str(link_path),
        project_root=str(tmp_path),
        timestamp="2026-05-21T10:00:00Z",
        status="eligible",
    )

    assert source.path == str(link_path)
    assert Path(source.canonical_path) == real_path.resolve()


def test_retired_cowork_probe_is_not_a_legal_provider_or_source_kind():
    from provider_adapters import (
        LEGAL_HEALTH_STATUSES,
        LEGAL_PROVIDERS,
        LEGAL_SOURCE_KINDS,
    )

    assert "claude_desktop_cowork" not in LEGAL_PROVIDERS
    assert "claude_desktop_sessions" not in LEGAL_SOURCE_KINDS
    assert "probe-only" not in LEGAL_HEALTH_STATUSES


def test_backfill_enqueue_rejects_retired_cowork_provider(tmp_path):
    from backfill_manager import BackfillManager

    manager = BackfillManager(tmp_path / "backfill_state.json")

    with pytest.raises(ValueError, match="Unknown provider"):
        manager.enqueue_provider_backfill(provider="claude_desktop_cowork", mode="recent")


def test_provider_source_rejects_unknown_provider(tmp_path):
    from provider_adapters import ProviderSource

    with pytest.raises(ValueError, match="provider"):
        ProviderSource(
            provider="gemini_cli",
            source_kind="legacy_gemini_history",
            source_class="native",
            source_id="legacy",
            logical_session_id="legacy",
            path=str(tmp_path / "legacy.jsonl"),
            project_root=str(tmp_path),
            timestamp="2026-05-21T10:00:00Z",
            status="eligible",
        )


def test_provider_parse_result_carries_normalized_turn_metadata(tmp_path):
    from provider_adapters import ProviderParseResult, ProviderSource

    source = ProviderSource(
        provider="antigravity_cli",
        source_kind="antigravity_cli_transcript_jsonl",
        source_class="native",
        source_id="ag:synthetic",
        logical_session_id="synthetic",
        path=str(tmp_path / "transcript.jsonl"),
        project_root=str(tmp_path),
        timestamp="2026-05-21T10:00:00Z",
        status="eligible",
    )
    result = ProviderParseResult(
        source=source,
        turns=[{
            "content": "synthetic turn",
            "doc_id": "ag:synthetic:1",
            "session_id": "synthetic",
            "logical_session_id": "synthetic",
            "provider": "antigravity_cli",
            "source_kind": "antigravity_cli_transcript_jsonl",
            "source_class": "native",
            "source_id": "ag:synthetic",
            "source_path": str(tmp_path / "transcript.jsonl"),
            "transcript_file": "transcript.jsonl",
            "turn_index": 1,
            "timestamp": "2026-05-21T10:00:00Z",
            "git_branch": "",
            "chunk_type": "turn",
            "project_root": str(tmp_path),
        }],
        cursor={"cursor_type": "step_index", "last_step_index": 1},
    )

    assert result.turns[0]["provider"] == source.provider
    assert result.cursor["cursor_type"] == "step_index"
