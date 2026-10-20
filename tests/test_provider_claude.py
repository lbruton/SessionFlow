"""SESF-6 red-phase tests for Claude CLI sources (Desktop/CoWork probe retired in SESF-59).

Requirements: 5, 6.5, 8.
"""


def test_claude_cli_adapter_preserves_existing_turns_with_provider_metadata(synthetic_claude_jsonl):
    from provider_claude import ClaudeCodeCliAdapter

    adapter = ClaudeCodeCliAdapter(projects_root=synthetic_claude_jsonl.parent)
    sources = adapter.discover_sources()
    source = next(src for src in sources if src.path == str(synthetic_claude_jsonl))

    parsed = adapter.parse_source(source, cursor=None)

    assert parsed.turns
    assert parsed.turns[0]["provider"] == "claude_code_cli"
    assert parsed.turns[0]["source_kind"] == "claude_code_jsonl"
    assert parsed.turns[0]["source_class"] == "native"
    assert parsed.turns[0]["project_root"]


def test_claude_desktop_cowork_probe_is_retired():
    # SESF-59: Desktop Code-tab sessions (former CoWork projects included) are
    # ordinary transcripts under ~/.claude/projects, indexed as claude_code_cli.
    import provider_claude

    assert not hasattr(provider_claude, "ClaudeDesktopCoworkProbe")
