"""Claude provider adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import transcript_parser
from provider_adapters import (
    ProviderHealth,
    ProviderParseResult,
    ProviderSource,
    ProviderWatchRoot,
    build_source_id,
    canonicalize_path,
)


class ClaudeCodeCliAdapter:
    """Adapter for native Claude Code CLI ``.jsonl`` transcripts."""

    provider = "claude_code_cli"
    source_kind = "claude_code_jsonl"

    def __init__(self, projects_root: str | Path | None = None):
        """Set the projects root, defaulting to ``~/.claude/projects``."""
        self.projects_root = (
            Path(projects_root).expanduser()
            if projects_root is not None
            else Path.home() / ".claude" / "projects"
        )

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate ``.jsonl`` transcripts under the projects root."""
        if not self.projects_root.exists():
            return []
        sources = []
        for path in sorted(self.projects_root.rglob("*.jsonl")):
            logical_session_id = path.stem
            canonical_path = canonicalize_path(path)
            project_root = transcript_parser.detect_project_root(str(path)) or ""
            source_id = build_source_id(self.provider, logical_session_id, canonical_path)
            sources.append(ProviderSource(
                provider=self.provider,
                source_kind=self.source_kind,
                source_class="native",
                source_id=source_id,
                logical_session_id=logical_session_id,
                path=str(path),
                canonical_path=canonical_path,
                project_root=project_root,
                timestamp="",
                status="eligible",
            ))
        return sources

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        """Parse a transcript into normalized turns, resuming from ``cursor``'s byte offset."""
        start_offset = int((cursor or {}).get("last_byte_offset", 0))
        try:
            turns, new_offset = transcript_parser.parse_transcript(
                source.path,
                source.logical_session_id,
                start_offset=start_offset,
            )
        except OSError as exc:
            return ProviderParseResult(
                source=source,
                turns=[],
                cursor=cursor or {},
                errors=[str(exc)],
            )
        normalized = []
        for turn in turns:
            text = turn.get("text", turn.get("content", ""))
            normalized.append({
                **turn,
                "text": text,
                "content": text,
                "logical_session_id": source.logical_session_id,
                "provider": self.provider,
                "source_kind": self.source_kind,
                "source_class": "native",
                "source_id": source.source_id,
                "source_path": source.path,
                "project_root": turn.get("project_root") or source.project_root,
            })
        return ProviderParseResult(
            source=source,
            turns=normalized,
            cursor={
                "cursor_type": "byte_offset",
                "logical_session_id": source.logical_session_id,
                "known_paths": [source.path],
                "last_byte_offset": new_offset,
                "project_root": source.project_root,
            },
        )

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the projects root as the single recursive watch root."""
        return [ProviderWatchRoot(self.provider, self.source_kind, str(self.projects_root), recursive=True)]

    def health(self) -> ProviderHealth:
        """Report adapter health and the discovered source count."""
        sources = self.discover_sources()
        return ProviderHealth(
            provider=self.provider,
            status="ok" if self.projects_root.exists() else "missing",
            source_count=len(sources),
            eligible_count=len(sources),
            roots=[str(self.projects_root)],
        )


class ClaudeDesktopCoworkProbe:
    """Probe-only adapter for Claude Desktop / CoWork sessions (SESF-6)."""

    provider = "claude_desktop_cowork"
    source_kind = "claude_desktop_sessions"

    def __init__(self, root: str | Path | None = None):
        """Set the sessions root, defaulting to the Claude Desktop support dir."""
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / "Library" / "Application Support" / "Claude" / "claude-code-sessions"
        )

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate ``local_*.json`` session files (marked unsupported/probe-only)."""
        if not self.root.exists():
            return []
        sources = []
        for path in sorted(self.root.rglob("local_*.json")):
            logical_session_id = path.parent.name
            canonical_path = canonicalize_path(path)
            source_id = build_source_id(self.provider, logical_session_id, canonical_path)
            sources.append(ProviderSource(
                provider=self.provider,
                source_kind=self.source_kind,
                source_class="native",
                source_id=source_id,
                logical_session_id=logical_session_id,
                path=str(path),
                canonical_path=canonical_path,
                project_root="unknown",
                timestamp="",
                status="unsupported",
                reason="Probe-only source; not searchable until Claude Desktop/CoWork turn content is verified.",
            ))
        return sources

    def parse_source(self, source: ProviderSource, cursor: Optional[Dict]) -> ProviderParseResult:
        """Return an empty result — Claude Desktop/CoWork content is not yet ingestible."""
        return ProviderParseResult(
            source=source,
            turns=[],
            cursor={
                "cursor_type": "record_set",
                "logical_session_id": source.logical_session_id,
                "known_paths": [source.path],
                "emitted_ids": [],
            },
            errors=["Claude Desktop/CoWork is probe-only in SESF-6."],
        )

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the sessions root as the single recursive watch root."""
        return [ProviderWatchRoot(self.provider, self.source_kind, str(self.root), recursive=True)]

    def health(self) -> ProviderHealth:
        """Report probe-only health and the discovered source count."""
        sources = self.discover_sources()
        return ProviderHealth(
            provider=self.provider,
            status="probe-only" if self.root.exists() else "missing",
            source_count=len(sources),
            unsupported_count=len(sources),
            roots=[str(self.root)],
            limitations=["Probe-only; searchable ingestion is deferred until content format is proven."],
        )
