"""SESF-56 tests for fixes ported from upstream mwgreen/claude-code-session-rag.

Covers symlink-alias path normalization (upstream 4ba7084), the
``<slug>/archived-sessions/<id>.jsonl`` layout (upstream 4ba7084) and the
``mcp<2.0`` pin (upstream 353e0d7).
"""

import asyncio
import os
import re
from pathlib import Path

import pytest

import file_watcher

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def aliased_projects(tmp_path, monkeypatch):
    """A real projects dir plus a symlink alias, with the alias as ``_CLAUDE_PROJECTS``.

    Mirrors ``~/.claude/projects -> /Volumes/DATA/claude/projects``.
    """
    real = tmp_path / "real" / "projects"
    slug_dir = real / "-Users-me-proj"
    slug_dir.mkdir(parents=True)
    (slug_dir / "abc.jsonl").write_text("")
    alias = tmp_path / "alias-projects"
    alias.symlink_to(real)
    monkeypatch.setattr(file_watcher, "_CLAUDE_PROJECTS", alias)
    return real, alias


def test_slug_resolves_from_real_path_when_root_is_symlink(aliased_projects):
    real, _alias = aliased_projects
    # FSEvents reports the resolved path, not the alias the root was configured with.
    assert file_watcher._slug_from_transcript_path(
        str(real / "-Users-me-proj" / "abc.jsonl")) == "-Users-me-proj"


def test_slug_resolves_from_alias_path(aliased_projects):
    _real, alias = aliased_projects
    assert file_watcher._slug_from_transcript_path(
        str(alias / "-Users-me-proj" / "abc.jsonl")) == "-Users-me-proj"


def test_slug_requires_path_boundary(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    sibling = tmp_path / "projects-other" / "-x"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(file_watcher, "_CLAUDE_PROJECTS", root)
    assert file_watcher._slug_from_transcript_path(str(sibling / "s.jsonl")) is None


def test_handler_enqueue_normalizes_symlinked_paths(aliased_projects):
    real, alias = aliased_projects

    async def run():
        queue: asyncio.Queue = asyncio.Queue()
        handler = file_watcher.TranscriptChangeHandler(queue, asyncio.get_running_loop())
        handler._enqueue(str(alias / "-Users-me-proj" / "abc.jsonl"))
        handler._enqueue(str(real / "-Users-me-proj" / "abc.jsonl"))
        await asyncio.sleep(0)
        return {queue.get_nowait() for _ in range(queue.qsize())}

    assert asyncio.run(run()) == {os.path.realpath(real / "-Users-me-proj" / "abc.jsonl")}


def test_backfill_keys_state_by_real_path(aliased_projects, monkeypatch):
    real, _alias = aliased_projects
    state = {"transcripts": {}}
    monkeypatch.setattr(file_watcher.transcript_parser, "load_index_state", lambda: state)
    monkeypatch.setattr(file_watcher.transcript_parser, "save_index_state", lambda s: None)
    monkeypatch.setattr(file_watcher.transcript_parser, "parse_transcript_incremental",
                        lambda path, sid, start_offset=0:
                        file_watcher.transcript_parser.ParseResult([], 0, 0))
    monkeypatch.setattr(file_watcher.transcript_parser, "detect_project_root", lambda p: "")
    (real / "-Users-me-proj" / "abc.jsonl").write_text('{"x": 1}\n')

    watcher = file_watcher.GlobalTranscriptWatcher(db_path="unused")
    asyncio.run(watcher.backfill())

    assert list(state["transcripts"]) == [
        os.path.realpath(real / "-Users-me-proj" / "abc.jsonl")]


def test_filtered_backfill_includes_archived_sessions(aliased_projects, monkeypatch):
    real, _alias = aliased_projects
    archived = real / "-Users-me-proj" / "archived-sessions" / "old.jsonl"
    archived.parent.mkdir()
    archived.write_text('{"x": 1}\n')
    (real / "-Users-me-proj" / "abc.jsonl").write_text('{"x": 1}\n')
    state = {"transcripts": {}}
    parsed = []
    monkeypatch.setattr(file_watcher.transcript_parser, "load_index_state", lambda: state)
    monkeypatch.setattr(file_watcher.transcript_parser, "save_index_state", lambda s: None)
    monkeypatch.setattr(file_watcher.transcript_parser, "parse_transcript_incremental",
                        lambda path, sid, start_offset=0:
                        file_watcher.transcript_parser.ParseResult(parsed.append(sid) or [], 0, 0))
    monkeypatch.setattr(file_watcher.transcript_parser, "detect_project_root", lambda p: "")

    watcher = file_watcher.GlobalTranscriptWatcher(db_path="unused")
    asyncio.run(watcher.backfill(slug_filter="-Users-me-proj"))

    assert sorted(parsed) == ["abc", "old"]


def test_load_index_state_migrates_alias_keys_keeping_offsets(aliased_projects, tmp_path,
                                                               monkeypatch):
    import json
    import transcript_parser

    real, alias = aliased_projects
    real_key = os.path.realpath(real / "-Users-me-proj" / "abc.jsonl")
    alias_key = str(alias / "-Users-me-proj" / "abc.jsonl")
    state_path = tmp_path / "index_state.json"
    state_path.write_text(json.dumps({"transcripts": {
        alias_key: {"last_byte_offset": 42, "project_root": "/p"},
        real_key: {"last_byte_offset": 10},
    }}))
    monkeypatch.setattr(transcript_parser, "_STATE_PATH", state_path)
    monkeypatch.setattr(transcript_parser, "_migrated", True)

    state = transcript_parser.load_index_state()

    assert list(state["transcripts"]) == [real_key]
    assert transcript_parser.get_transcript_offset(state, real_key) == 42
    assert state["transcripts"][real_key]["project_root"] == "/p"


def test_archived_sessions_layout_is_a_main_transcript(aliased_projects):
    real, _alias = aliased_projects
    archived = real / "-Users-me-proj" / "archived-sessions" / "old.jsonl"
    archived.parent.mkdir()
    archived.write_text("")

    assert file_watcher._slug_from_transcript_path(str(archived)) == "-Users-me-proj"

    from provider_claude import ClaudeCodeCliAdapter
    sources = ClaudeCodeCliAdapter(projects_root=real).discover_sources()
    src = next(s for s in sources if s.path.endswith("old.jsonl"))
    assert src.logical_session_id == "old"


def _mcp_spec(text: str) -> str:
    match = re.search(r'"?mcp([<>=,.\d\s]+)"?', text)
    assert match, "mcp requirement not found"
    return match.group(1).strip()


@pytest.mark.parametrize("path", ["requirements.txt", "setup.sh"])
def test_mcp_pinned_below_2(path):
    lines = [ln for ln in (REPO_ROOT / path).read_text().splitlines()
             if re.search(r'(^|["\s])mcp[<>=]', ln)]
    assert lines, f"no mcp requirement in {path}"
    for line in lines:
        assert "<2.0" in _mcp_spec(line), f"{path}: {line!r} must cap mcp below 2.0"
