"""SESF-60 tests for the legacy Cowork agent-mode transcript provider.

Fixtures are synthetic: the on-disk layout mirrors
``local-agent-mode-sessions/<id>/<id>/local_<id>/.claude/projects/<slug>/<session>.jsonl``
with a sibling ``local_<id>.json`` metadata file, but no real session text.
"""

import json
from pathlib import Path

import pytest


def _write_transcript(path: Path, cwd: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "type": "user",
            "cwd": cwd,
            "timestamp": "2026-03-02T10:00:00Z",
            "message": {"content": "synthetic cowork request"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-02T10:00:01Z",
            "message": {"content": [{"type": "text", "text": "synthetic cowork response"}]},
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return path


def _make_session(root: Path, name: str, metadata) -> Path:
    """Create one ``local_<name>`` session dir and return its main transcript path."""
    space = root / "space-uuid" / "account-uuid"
    local_dir = space / f"local_{name}"
    transcript = _write_transcript(
        local_dir / ".claude" / "projects" / f"-sessions-{name}" / f"{name}-session.jsonl",
        cwd=f"/sessions/{name}",
    )
    (local_dir / "audit.jsonl").write_text('{"event": "synthetic"}\n')
    (local_dir / "CLAUDE.md").write_text("# synthetic\n")
    meta_path = space / f"local_{name}.json"
    if isinstance(metadata, str):
        meta_path.write_text(metadata)
    elif metadata is not None:
        meta_path.write_text(json.dumps(metadata))
    return transcript


@pytest.fixture
def cowork_root(tmp_path):
    root = tmp_path / "local-agent-mode-sessions"
    selected = _make_session(root, "alpha", {
        "cwd": "/sessions/alpha",
        "userSelectedFolders": ["/Users/someone/CoWork", "/Users/someone/Other"],
        "emailAddress": "someone@example.com",
        "systemPrompt": "synthetic system prompt",
    })
    _write_transcript(
        selected.parent / "alpha-session" / "subagents" / "agent-a1b2c3.jsonl",
        cwd="/sessions/alpha",
    )
    _make_session(root, "beta", {"cwd": "/sessions/beta", "userSelectedFolders": []})
    _make_session(root, "gamma", "{not json")
    _make_session(root, "delta", None)
    outputs_cwd = str(root / "space-uuid" / "account-uuid" / "local_epsilon" / "outputs")
    _make_session(root, "epsilon", {"cwd": outputs_cwd})
    _make_session(root, "zeta", {"cwd": "/Users/someone/RealFolder"})
    (root / "space-uuid" / "account-uuid" / "scheduled-tasks.json").write_text("{}\n")
    return root


def _by_session(sources):
    return {source.logical_session_id: source for source in sources}


def test_discovery_finds_transcripts_and_skips_audit_and_non_transcripts(cowork_root):
    from provider_claude import ClaudeCoworkLegacyAdapter

    sources = ClaudeCoworkLegacyAdapter(root=cowork_root).discover_sources()
    names = {Path(source.path).name for source in sources}

    assert names == {
        "alpha-session.jsonl",
        "agent-a1b2c3.jsonl",
        "beta-session.jsonl",
        "gamma-session.jsonl",
        "delta-session.jsonl",
        "epsilon-session.jsonl",
        "zeta-session.jsonl",
    }
    assert "audit.jsonl" not in names
    for source in sources:
        assert source.provider == "claude_cowork_legacy"
        assert source.source_kind == "claude_cowork_legacy_jsonl"
        assert source.source_class == "native"
        assert source.status == "eligible"
        assert source.source_id.startswith("claude_cowork_legacy:")


def test_project_root_comes_from_session_metadata(cowork_root):
    from provider_claude import ClaudeCoworkLegacyAdapter

    sources = _by_session(ClaudeCoworkLegacyAdapter(root=cowork_root).discover_sources())

    # First user-selected folder wins over the sandbox cwd.
    assert sources["alpha-session"].project_root == "/Users/someone/CoWork"
    # Subagent transcripts inherit their session's metadata.
    assert sources["agent-a1b2c3"].project_root == "/Users/someone/CoWork"
    # A real cwd is used when no folder was selected.
    assert sources["zeta-session"].project_root == "/Users/someone/RealFolder"
    # Sandbox cwds (/sessions/..., the session's own outputs dir), malformed or
    # missing metadata all fall back to "unknown".
    assert sources["beta-session"].project_root == "unknown"
    assert sources["epsilon-session"].project_root == "unknown"
    assert sources["gamma-session"].project_root == "unknown"
    assert sources["delta-session"].project_root == "unknown"


def test_parse_source_labels_turns_with_legacy_provider_and_metadata_root(cowork_root):
    from provider_claude import ClaudeCoworkLegacyAdapter

    adapter = ClaudeCoworkLegacyAdapter(root=cowork_root)
    source = _by_session(adapter.discover_sources())["alpha-session"]

    parsed = adapter.parse_source(source, cursor=None)

    assert parsed.turns
    assert not parsed.errors
    for turn in parsed.turns:
        assert turn["provider"] == "claude_cowork_legacy"
        assert turn["source_kind"] == "claude_cowork_legacy_jsonl"
        assert turn["project_root"] == "/Users/someone/CoWork"
        assert "someone@example.com" not in json.dumps(turn)
    # The fixture ends on its last turn, which stays open: the cursor resumes
    # there and records the bytes read in consumed_offset.
    assert parsed.cursor["consumed_offset"] > 0


def test_missing_root_reports_missing_without_sources(tmp_path):
    from provider_claude import ClaudeCoworkLegacyAdapter

    adapter = ClaudeCoworkLegacyAdapter(root=tmp_path / "gone")

    assert adapter.discover_sources() == []
    health = adapter.health()
    assert health.provider == "claude_cowork_legacy"
    assert health.status == "missing"
    assert health.source_count == 0


def test_root_can_be_overridden_by_env(cowork_root, monkeypatch):
    from provider_claude import ClaudeCoworkLegacyAdapter

    monkeypatch.setenv("SESSIONFLOW_COWORK_LEGACY_ROOT", str(cowork_root))

    adapter = ClaudeCoworkLegacyAdapter()

    assert adapter.root == cowork_root
    assert len(adapter.discover_sources()) == 7


def test_default_root_is_desktop_agent_mode_folder(monkeypatch):
    from provider_claude import ClaudeCoworkLegacyAdapter

    monkeypatch.delenv("SESSIONFLOW_COWORK_LEGACY_ROOT", raising=False)

    adapter = ClaudeCoworkLegacyAdapter()

    assert adapter.root == (
        Path.home() / "Library" / "Application Support" / "Claude" / "local-agent-mode-sessions"
    )


def test_legacy_provider_is_registered():
    from provider_adapters import LEGAL_PROVIDERS, LEGAL_SOURCE_KINDS
    from provider_ingestion import default_provider_adapters

    assert "claude_cowork_legacy" in LEGAL_PROVIDERS
    assert "claude_cowork_legacy_jsonl" in LEGAL_SOURCE_KINDS
    assert default_provider_adapters()["claude_cowork_legacy"].provider == "claude_cowork_legacy"


@pytest.mark.anyio
async def test_backfill_job_ingests_legacy_sources(cowork_root, tmp_path, monkeypatch, stub_rag_engine):
    import provider_ingestion
    from backfill_manager import BackfillManager
    from provider_claude import ClaudeCoworkLegacyAdapter

    indexed = []

    async def fake_add_turns_async(turns, db_path=None):
        indexed.extend(turns)
        return len(turns)

    monkeypatch.setattr(provider_ingestion.rag_engine, "add_turns_async", fake_add_turns_async)
    monkeypatch.setattr(provider_ingestion.transcript_parser, "load_index_state", lambda: {})
    monkeypatch.setattr(provider_ingestion.transcript_parser, "save_index_state", lambda state: None)

    manager = BackfillManager(tmp_path / "backfill_state.json")
    manager.enqueue_provider_backfill(provider="claude_cowork_legacy", mode="full")
    service = provider_ingestion.ProviderIngestionService(
        manager,
        db_path=str(tmp_path / "db"),
        adapters={"claude_cowork_legacy": ClaudeCoworkLegacyAdapter(root=cowork_root)},
    )

    totals = await service.process_queued_jobs()

    assert totals["processed_sources"] == 7
    assert totals["errors"] == 0
    assert indexed
    assert {turn["provider"] for turn in indexed} == {"claude_cowork_legacy"}
