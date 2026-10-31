"""Codex rollout JSONL provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import hashlib
import io
import json
import logging

from provider_adapters import (
    REPLACE_SAME_TURN,
    ProviderHealth,
    ProviderParseResult,
    ProviderSource,
    ProviderWatchRoot,
    build_source_id,
    canonicalize_path,
    normalize_timestamp,
)
from transcript_parser import truncate_turn_text

logger = logging.getLogger(__name__)

_ZST_SUFFIX = ".zst"
_zstd_warned = False


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


# Codex Desktop marks turns it imported from another harness with this turn_id prefix.
_IMPORT_TURN_PREFIX = "external-import-"


def _is_import_turn(turn_id) -> bool:
    return isinstance(turn_id, str) and turn_id.startswith(_IMPORT_TURN_PREFIX)


def _zstd_available() -> bool:
    global _zstd_warned
    try:
        import zstandard  # noqa: F401
    except ImportError:
        if not _zstd_warned:
            logger.warning("zstandard is not installed; compressed Codex rollouts are skipped")
            _zstd_warned = True
        return False
    return True


def iter_rollout_lines(path: str | Path) -> Iterator[str]:
    """Yield the lines of a Codex rollout, decompressing a ``.jsonl.zst`` file as it reads.

    Raises:
        OSError: The file cannot be read or its zstd frame is corrupt.
    """
    path = Path(path)
    if not path.name.endswith(_ZST_SUFFIX):
        with path.open("r", encoding="utf-8") as handle:
            yield from handle
        return
    import zstandard

    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        try:
            yield from io.TextIOWrapper(reader, encoding="utf-8")
        except zstandard.ZstdError as exc:
            raise OSError(f"Cannot decompress {path.name}: {exc}") from exc


def _rollout_stem(path: Path) -> str:
    return path.name.removesuffix(_ZST_SUFFIX).removesuffix(".jsonl").removeprefix("rollout-")


@dataclass
class _Task:
    """One Codex task: the user message and every reply up to ``task_complete``."""

    ordinal: int
    importing: bool = False
    started: str = ""
    user: str = ""
    parts: List[str] = field(default_factory=list)
    line_timestamp: str = ""
    branch: str = ""

    def add_user(self, content: str) -> None:
        if self.parts:
            self.parts.append(f"User: {content}")
        else:
            self.user = content

    def add_reply(self, content: str) -> None:
        if self.parts and self.parts[-1].startswith("User: "):
            content = f"Assistant: {content}"
        self.parts.append(content)

    def text(self) -> str:
        replies = "\n\n".join(self.parts)
        return truncate_turn_text(
            f"User: {self.user}\n\nAssistant: {replies}" if self.user else replies)


def _task_events(entries: List[dict]) -> bool:
    return any(
        entry.get("type") == "event_msg"
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("type") == "task_started"
        for entry in entries
    )


def _parse_tasks(entries: List[dict]) -> tuple[List[_Task], str]:
    """Group rollout entries into tasks, numbered by their position in the rollout.

    A task opens at ``task_started``; a rollout without task events opens one at
    each user message that follows a reply. Import tasks keep their ordinal so the
    native tasks after them do not move.
    """
    has_tasks = _task_events(entries)
    tasks: List[_Task] = []
    current: Optional[_Task] = None
    session_branch = ""

    def open_task(**kwargs) -> _Task:
        task = _Task(ordinal=len(tasks), **kwargs)
        tasks.append(task)
        return task

    for entry in entries:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        record = payload if payload else entry
        if entry.get("type") == "session_meta":
            git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
            session_branch = git.get("branch") or session_branch
            continue
        if entry.get("type") == "event_msg":
            # Codex Desktop can rewrite a rollout and restamp every line's
            # top-level timestamp; event_msg started_at keeps the task's real start.
            started_at = payload.get("started_at")
            started = (
                normalize_timestamp(started_at)
                if isinstance(started_at, (int, float)) and not isinstance(started_at, bool)
                else ""
            )
            if payload.get("type") == "task_started":
                current = open_task(importing=_is_import_turn(payload.get("turn_id")),
                                    started=started)
            elif payload.get("type") == "task_complete":
                if current is not None and not current.started:
                    current.started = started
                current = None
            elif current is not None and not current.started:
                current.started = started
            continue
        role = record.get("role")
        content = _content_to_text(record.get("content") or record.get("text") or "")
        if role not in ("user", "assistant") or not content:
            continue
        if current is None or (not has_tasks and role == "user" and current.parts):
            current = open_task()
        if current.importing:
            continue
        if role == "user":
            current.add_user(str(content))
            continue
        current.add_reply(str(content))
        if not current.line_timestamp:
            current.line_timestamp = normalize_timestamp(
                entry.get("timestamp") or record.get("timestamp"))
        current.branch = current.branch or entry.get("git_branch") or record.get("git_branch") or ""
    return tasks, session_branch


class CodexAdapter:
    """Adapter for Codex rollout sessions (active and archived, plain or ``.zst``)."""

    provider = "codex"
    source_kind = "codex_rollout_jsonl"

    def __init__(self, home: str | Path | None = None):
        """Set the Codex home, defaulting to the user home directory."""
        self.home = Path(home).expanduser() if home is not None else Path.home()
        self.active_root = self.home / ".codex" / "sessions"
        self.archive_root = self.home / ".codex" / "archived_sessions"
        self._known_paths: Dict[str, List[str]] = {}

    def _rollout_paths(self) -> List[Path]:
        patterns = ["rollout-*.jsonl"]
        if _zstd_available():
            patterns.append("rollout-*.jsonl" + _ZST_SUFFIX)
        found: List[Path] = []
        for root, recursive in ((self.active_root, True), (self.archive_root, False)):
            if not root.exists():
                continue
            for pattern in patterns:
                found.extend(root.rglob(pattern) if recursive else root.glob(pattern))
        plain = {path for path in found if not path.name.endswith(_ZST_SUFFIX)}
        # A plain copy next to its compressed twin is the newer one.
        return sorted(
            path for path in found
            if not (path.name.endswith(_ZST_SUFFIX) and path.with_suffix("") in plain)
        )

    def _session_context(self, path: Path) -> tuple[str, str, str]:
        logical_session_id = _rollout_stem(path)
        project_root = ""
        timestamp = ""
        try:
            for line in iter_rollout_lines(path):
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
        except (OSError, UnicodeDecodeError):
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

    def _turn(self, source: ProviderSource, path: str, task: _Task, branch: str) -> dict:
        text = task.text()
        doc_hash = hashlib.sha256(
            f"{source.logical_session_id}:{task.ordinal}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "text": text,
            "content": text,
            "doc_id": f"codex:{source.logical_session_id}:{doc_hash}",
            "session_id": source.logical_session_id,
            "logical_session_id": source.logical_session_id,
            "provider": self.provider,
            "source_kind": self.source_kind,
            "source_class": "native",
            "source_id": source.source_id,
            "source_path": path,
            "transcript_file": Path(path).name,
            "turn_index": task.ordinal,
            "timestamp": task.started or task.line_timestamp or source.timestamp,
            "git_branch": task.branch or branch,
            "chunk_type": "turn",
            "project_root": source.project_root,
            REPLACE_SAME_TURN: True,
        }

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        """Parse a Codex rollout into one turn per task, skipping turns ``cursor`` emitted.

        A task's ``turn_index`` is its ordinal in the rollout, so a rewrite that shifts
        lines keeps the doc_id. An open task is emitted again each time it grows and
        replaces its earlier row.
        """
        # _known_paths is populated by discover_sources() and may be stale
        # or empty when callers reuse a shared adapter instance across
        # threads. Fall back to the source's own path/canonical_path rather
        # than dropping the session silently.
        known_paths = self._known_paths.get(source.source_id)
        if not known_paths:
            fallback = source.path or source.canonical_path
            known_paths = [fallback] if fallback else []
        emitted: set[str] = set((cursor or {}).get("emitted_ids", []))
        by_ordinal: Dict[int, dict] = {}
        errors: List[str] = []
        for path in known_paths:
            # Callers that delete rows must tell a failed or partial read from a
            # complete one, so read errors and malformed lines both land in errors.
            malformed = 0
            try:
                entries = []
                for line in iter_rollout_lines(path):
                    if not line.strip():
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed += 1
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"{Path(path).name}: {exc}")
                continue
            if malformed:
                errors.append(f"{Path(path).name}: {malformed} malformed "
                              f"line{'s' if malformed > 1 else ''}")
            tasks, branch = _parse_tasks(entries)
            for task in tasks:
                if task.importing or not task.parts:
                    continue
                turn = self._turn(source, path, task, branch)
                # An active and an archived copy can differ; the longer holds more of the task.
                held = by_ordinal.get(task.ordinal)
                if held is None or len(turn["text"]) > len(held["text"]):
                    by_ordinal[task.ordinal] = turn

        turns = [by_ordinal[ordinal] for ordinal in sorted(by_ordinal)]
        # A partial parse can shorten a task or shift ordinals, and REPLACE_SAME_TURN
        # would then delete complete rows: emit nothing and keep the previous cursor.
        if errors:
            turns, current = [], sorted(emitted)
        else:
            current = sorted(turn["doc_id"] for turn in turns)
        return ProviderParseResult(
            source=source,
            turns=[turn for turn in turns if turn["doc_id"] not in emitted],
            cursor={
                "cursor_type": "record_set",
                "logical_session_id": source.logical_session_id,
                "known_paths": known_paths,
                "emitted_ids": current,
                "project_root": source.project_root,
            },
            errors=errors,
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
