"""Claude provider adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import transcript_parser
from provider_adapters import (
    ProviderHealth,
    ProviderParseResult,
    ProviderSource,
    ProviderWatchRoot,
    PROJECT_PATH_MAP_ENV,
    build_source_id,
    canonicalize_path,
    parse_project_path_map,
    remap_project_root,
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



class ClaudeCoworkLegacyAdapter(ClaudeCodeCliAdapter):
    """Adapter for legacy Claude Cowork agent-mode transcripts.

    Each agent-mode session ran Claude Code with a sandboxed config dir, so its
    transcripts are ordinary Claude Code JSONL under
    ``<root>/<space>/<account>/local_<id>/.claude/projects/<slug>/``. Parsing is
    inherited from :class:`ClaudeCodeCliAdapter`; only discovery and the
    project tag differ. The transcript's own ``cwd`` is a VM path
    (``/sessions/<name>``), so ``project_root`` comes from the sibling
    ``local_<id>.json`` metadata instead. Only ``userSelectedFolders`` and
    ``cwd`` are read from it; the rest (system prompt, account details) is
    never touched. A folder that has moved since is rewritten through
    ``SESSIONFLOW_PROJECT_PATH_MAP``.
    """

    provider = "claude_cowork_legacy"
    source_kind = "claude_cowork_legacy_jsonl"

    def __init__(self, root: str | Path | None = None):
        """Set the sessions root.

        Args:
            root: Directory holding the agent-mode sessions. Defaults to
                ``$SESSIONFLOW_COWORK_LEGACY_ROOT``, else Claude Desktop's
                ``local-agent-mode-sessions`` folder. Point the env var at a
                backup copy if Desktop has moved or pruned the original.
        """
        if root is None:
            root = os.environ.get("SESSIONFLOW_COWORK_LEGACY_ROOT") or (
                Path.home() / "Library" / "Application Support" / "Claude" / "local-agent-mode-sessions"
            )
        super().__init__(projects_root=root)
        self.root = self.projects_root

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate session transcripts, including subagent ones, skipping ``audit.jsonl``."""
        if not self.root.exists():
            return []
        # The selected folder may have moved since the session ran.
        path_map = parse_project_path_map(os.environ.get(PROJECT_PATH_MAP_ENV))
        project_roots: Dict[Path, str] = {}
        sources = []
        for path in sorted(self.root.glob("*/*/local_*/.claude/projects/**/*.jsonl")):
            session_dir = self._session_dir(path)
            if session_dir not in project_roots:
                project_roots[session_dir] = remap_project_root(
                    self._project_root_from_metadata(session_dir), path_map,
                )
            logical_session_id = path.stem
            canonical_path = canonicalize_path(path)
            sources.append(ProviderSource(
                provider=self.provider,
                source_kind=self.source_kind,
                source_class="native",
                source_id=build_source_id(self.provider, logical_session_id, canonical_path),
                logical_session_id=logical_session_id,
                path=str(path),
                canonical_path=canonical_path,
                project_root=project_roots[session_dir],
                timestamp="",
                status="eligible",
            ))
        return sources

    def _session_dir(self, transcript: Path) -> Path:
        # <root>/<space>/<account>/local_<id>/.claude/projects/...
        return self.root.joinpath(*transcript.relative_to(self.root).parts[:3])

    def _project_root_from_metadata(self, session_dir: Path) -> str:
        meta_path = session_dir.with_name(session_dir.name + ".json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "unknown"
        if not isinstance(meta, dict):
            return "unknown"
        for folder in meta.get("userSelectedFolders") or []:
            if isinstance(folder, str) and folder.strip():
                return folder.strip()
        cwd = meta.get("cwd")
        if isinstance(cwd, str) and cwd.startswith("/") and not self._is_sandbox_path(cwd):
            return cwd
        return "unknown"

    def _is_sandbox_path(self, path: str) -> bool:
        if path == "/sessions" or path.startswith("/sessions/"):
            return True
        try:
            Path(path).relative_to(self.root)
        except ValueError:
            return False
        return True
