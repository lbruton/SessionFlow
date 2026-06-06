"""Codex rollout JSONL provider adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json

from provider_adapters import (
    ProviderHealth,
    ProviderParseResult,
    ProviderSource,
    ProviderWatchRoot,
    build_source_id,
    canonicalize_path,
    normalize_timestamp,
)


def _content_to_text(content) -> str:
    """Extract recall-safe text from Codex content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if content:
        return str(content)
    return ""


class CodexAdapter:
    """Adapter for Codex rollout ``.jsonl`` sessions (active and archived)."""

    provider = "codex"
    source_kind = "codex_rollout_jsonl"

    def __init__(self, home: str | Path | None = None):
        """Set the Codex home, defaulting to the user home directory."""
        self.home = Path(home).expanduser() if home is not None else Path.home()
        self.active_root = self.home / ".codex" / "sessions"
        self.archive_root = self.home / ".codex" / "archived_sessions"
        self._known_paths: Dict[str, List[str]] = {}

    def _rollout_paths(self) -> List[Path]:
        paths: List[Path] = []
        if self.active_root.exists():
            paths.extend(self.active_root.rglob("rollout-*.jsonl"))
        if self.archive_root.exists():
            paths.extend(self.archive_root.glob("rollout-*.jsonl"))
        return sorted(paths)

    def _session_context(self, path: Path) -> tuple[str, str, str]:
        logical_session_id = path.stem.removeprefix("rollout-")
        project_root = ""
        timestamp = ""
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                    logical_session_id = (
                        entry.get("session_id")
                        or entry.get("sessionId")
                        or entry.get("id")
                        or payload.get("session_id")
                        or payload.get("sessionId")
                        or payload.get("id")
                        or logical_session_id
                    )
                    git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
                    project_root = (
                        entry.get("cwd")
                        or entry.get("project_root")
                        or payload.get("cwd")
                        or payload.get("project_root")
                        or git.get("cwd")
                        or project_root
                    )
                    timestamp = (
                        entry.get("timestamp")
                        or entry.get("created_at")
                        or payload.get("timestamp")
                        or payload.get("created_at")
                        or timestamp
                    )
                    if project_root and timestamp:
                        break
        except OSError:
            pass
        return logical_session_id, project_root or "unknown", timestamp

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate Codex rollout sessions across the active and archive roots."""
        grouped: Dict[str, List[Path]] = {}
        context: Dict[str, tuple[str, str]] = {}
        for path in self._rollout_paths():
            logical_session_id, project_root, timestamp = self._session_context(path)
            grouped.setdefault(logical_session_id, []).append(path)
            context.setdefault(logical_session_id, (project_root, timestamp))

        sources: List[ProviderSource] = []
        self._known_paths = {}
        for logical_session_id, paths in grouped.items():
            canonical_paths = [canonicalize_path(path) for path in paths]
            first_path = paths[0]
            source_id = build_source_id(
                self.provider,
                logical_session_id,
                "::".join(sorted(canonical_paths)),
            )
            self._known_paths[source_id] = [str(path) for path in paths]
            project_root, timestamp = context.get(logical_session_id, ("unknown", ""))
            sources.append(ProviderSource(
                provider=self.provider,
                source_kind=self.source_kind,
                source_class="native",
                source_id=source_id,
                logical_session_id=logical_session_id,
                path=str(first_path),
                canonical_path=canonicalize_path(first_path),
                project_root=project_root,
                timestamp=normalize_timestamp(timestamp),
                status="eligible",
            ))
        return sources

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        """Parse a Codex rollout into normalized turns, resuming from ``cursor``."""
        # _known_paths is populated by discover_sources() and may be stale
        # or empty when callers reuse a shared adapter instance across
        # threads. Fall back to the source's own path/canonical_path rather
        # than dropping the session silently.
        known_paths = self._known_paths.get(source.source_id)
        if not known_paths:
            fallback = source.path or source.canonical_path
            known_paths = [fallback] if fallback else []
        emitted: set[str] = set((cursor or {}).get("emitted_ids", []))
        turns = []
        for path in known_paths:
            try:
                lines = Path(path).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            pending_user = ""
            for index, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                record = payload if payload else entry
                role = record.get("role")
                content = _content_to_text(record.get("content") or record.get("text") or "")
                if role == "user" and content:
                    pending_user = str(content)
                    continue
                if role == "assistant" and content:
                    text = (
                        f"User: {pending_user}\n\nAssistant: {content}"
                        if pending_user else str(content)
                    )
                    doc_hash = hashlib.sha256(
                        f"{source.logical_session_id}:{index}:{text}".encode("utf-8")
                    ).hexdigest()[:16]
                    doc_id = f"codex:{source.logical_session_id}:{doc_hash}"
                    if doc_id in emitted:
                        continue
                    emitted.add(doc_id)
                    turns.append({
                        "text": text,
                        "content": text,
                        "doc_id": doc_id,
                        "session_id": source.logical_session_id,
                        "logical_session_id": source.logical_session_id,
                        "provider": self.provider,
                        "source_kind": self.source_kind,
                        "source_class": "native",
                        "source_id": source.source_id,
                        "source_path": path,
                        "transcript_file": Path(path).name,
                        "turn_index": index,
                        "timestamp": normalize_timestamp(
                            entry.get("timestamp") or record.get("timestamp")
                        ) or source.timestamp,
                        "git_branch": entry.get("git_branch") or record.get("git_branch", ""),
                        "chunk_type": "turn",
                        "project_root": source.project_root,
                    })
                    pending_user = ""

        return ProviderParseResult(
            source=source,
            turns=turns,
            cursor={
                "cursor_type": "record_set",
                "logical_session_id": source.logical_session_id,
                "known_paths": known_paths,
                "emitted_ids": sorted(emitted),
                "project_root": source.project_root,
            },
        )

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the active (recursive) and archive (flat) watch roots."""
        return [
            ProviderWatchRoot(self.provider, self.source_kind, str(self.active_root), recursive=True),
            ProviderWatchRoot(self.provider, self.source_kind, str(self.archive_root), recursive=False),
        ]

    def health(self) -> ProviderHealth:
        """Report adapter health, flagging any missing roots."""
        sources = self.discover_sources()
        missing_roots = [
            str(path) for path in (self.active_root, self.archive_root) if not path.exists()
        ]
        return ProviderHealth(
            provider=self.provider,
            status="ok" if sources else "missing",
            source_count=len(sources),
            eligible_count=sum(1 for source in sources if source.status == "eligible"),
            roots=[str(self.active_root), str(self.archive_root)],
            limitations=missing_roots,
        )
