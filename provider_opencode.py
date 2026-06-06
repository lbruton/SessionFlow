"""OpenCode storage provider adapter.

OpenCode persists session/message/part records in a SQLite database
(``~/.local/share/opencode/opencode.db``). Earlier shipping builds also wrote
per-record JSON files under ``storage/{session,message,part}/`` which some
historical installs still carry; the adapter reads SQLite first and falls back
to the JSON layout when the DB is absent (or for sessions that only exist in
the legacy tree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import logging
import sqlite3
import time

from provider_adapters import (
    ProviderHealth,
    ProviderParseResult,
    ProviderSource,
    ProviderWatchRoot,
    build_source_id,
    canonicalize_path,
    normalize_timestamp,
)

logger = logging.getLogger("sessionflow.opencode")


class OpenCodeAdapter:
    """Adapter for OpenCode storage (legacy file tree plus the SQLite store)."""

    provider = "opencode"
    source_kind = "opencode_storage"

    def __init__(
        self,
        storage_root: str | Path | None = None,
        db_path: str | Path | None = None,
        settled_seconds: int = 5,
    ):
        """Set the storage root, SQLite path, and settle window for new writes."""
        # OpenCode keeps both the DB and the legacy filesystem tree under the
        # same root, so callers normally only need to pass storage_root.
        if storage_root is not None:
            self.storage_root = Path(storage_root).expanduser()
        else:
            self.storage_root = Path.home() / ".local" / "share" / "opencode" / "storage"
        # `opencode.db` lives one directory up from the legacy storage tree.
        if db_path is not None:
            self.db_path = Path(db_path).expanduser()
        else:
            self.db_path = self.storage_root.parent / "opencode.db"
        self.settled_seconds = settled_seconds
        # Per-pass cache populated by discover_sources(); parse_source() reuses
        # message/part rows without re-querying the DB.
        self._messages_by_session: Dict[str, List[Dict]] = {}
        self._parts_by_message: Dict[str, List[Dict]] = {}
        self._session_path_for: Dict[str, str] = {}
        # Legacy JSON cache.
        self._legacy_messages_by_session: Dict[str, List[Tuple[Path, Dict]]] = {}
        self._legacy_parts_by_session: Dict[str, List[Tuple[Path, Dict]]] = {}
        self._legacy_all_message_ids: set[str] = set()
        self._legacy_all_parts: List[Tuple[Path, Dict]] = []
        self._legacy_indexes_built = False

    # ---- DB helpers --------------------------------------------------------

    def _open_db(self) -> Optional[sqlite3.Connection]:
        if not self.db_path.exists():
            return None
        # `uri=True` lets us open in read-only mode and avoid mutating WAL.
        uri = f"{self.db_path.as_uri()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        except sqlite3.Error:
            return None
        conn.row_factory = sqlite3.Row
        return conn

    def _refresh_db_indexes(self) -> List[ProviderSource]:
        self._messages_by_session = {}
        self._parts_by_message = {}
        self._session_path_for = {}
        conn = self._open_db()
        if conn is None:
            return []
        sources: List[ProviderSource] = []
        try:
            # Newest-first so the per-run file limit (max_files_per_run)
            # never clips today's session off the back of the list.
            sessions = conn.execute(
                "SELECT id, directory, title, time_created FROM session "
                "ORDER BY time_created DESC"
            ).fetchall()
            for row in sessions:
                session_id = str(row["id"])
                self._session_path_for[session_id] = str(self.db_path)
                source_id = build_source_id(
                    self.provider, session_id, canonicalize_path(self.db_path)
                )
                directory = row["directory"] or "unknown"
                sources.append(
                    ProviderSource(
                        provider=self.provider,
                        source_kind=self.source_kind,
                        source_class="native",
                        source_id=source_id,
                        logical_session_id=session_id,
                        path=str(self.db_path),
                        canonical_path=canonicalize_path(self.db_path),
                        project_root=directory,
                        timestamp=normalize_timestamp(row["time_created"]),
                        status="eligible",
                    )
                )

            for row in conn.execute(
                "SELECT id, session_id, time_created, data FROM message ORDER BY session_id, time_created, id"
            ):
                data = self._safe_json(row["data"])
                if not data:
                    continue
                data.setdefault("id", row["id"])
                data.setdefault("sessionID", row["session_id"])
                data.setdefault("__time_created_ms", row["time_created"])
                self._messages_by_session.setdefault(str(row["session_id"]), []).append(data)

            for row in conn.execute(
                "SELECT id, message_id, session_id, time_created, data FROM part ORDER BY message_id, time_created, id"
            ):
                data = self._safe_json(row["data"])
                if not data:
                    continue
                data.setdefault("id", row["id"])
                data.setdefault("messageID", row["message_id"])
                data.setdefault("sessionID", row["session_id"])
                data.setdefault("__time_created_ms", row["time_created"])
                self._parts_by_message.setdefault(str(row["message_id"]), []).append(data)
            msg_count = sum(len(v) for v in self._messages_by_session.values())
            part_count = sum(len(v) for v in self._parts_by_message.values())
            if msg_count + part_count > 50_000:
                logger.warning(
                    "_refresh_db_indexes loaded %d messages and %d parts into memory; "
                    "large OpenCode installs risk OOM — lazy per-session queries are recommended",
                    msg_count,
                    part_count,
                )
        finally:
            conn.close()
        return sources

    @staticmethod
    def _safe_json(text: object) -> Dict:
        if not isinstance(text, str) or not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    # ---- Legacy JSON tree (older opencode installs) ------------------------

    def _refresh_legacy_indexes(self) -> None:
        self._legacy_messages_by_session = {}
        self._legacy_parts_by_session = {}
        self._legacy_all_message_ids = set()
        self._legacy_all_parts = []

        message_root = self.storage_root / "message"
        if message_root.exists():
            for path in sorted(message_root.rglob("*.json")):
                data = self._load_json(path)
                session_id = data.get("sessionID") or data.get("session_id")
                if session_id:
                    self._legacy_messages_by_session.setdefault(str(session_id), []).append((path, data))
                msg_id = data.get("id")
                if msg_id:
                    self._legacy_all_message_ids.add(str(msg_id))

        part_root = self.storage_root / "part"
        if part_root.exists():
            for path in sorted(part_root.rglob("*.json")):
                data = self._load_json(path)
                self._legacy_all_parts.append((path, data))
                session_id = data.get("sessionID") or data.get("session_id")
                if session_id:
                    self._legacy_parts_by_session.setdefault(str(session_id), []).append((path, data))
        self._legacy_indexes_built = True

    def _legacy_session_files(self) -> List[Path]:
        root = self.storage_root / "session"
        return sorted(root.rglob("*.json")) if root.exists() else []

    def _load_json(self, path: Path) -> Dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ---- Public API --------------------------------------------------------

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate OpenCode sessions from the SQLite store and legacy tree."""
        sources = self._refresh_db_indexes()
        seen = {src.logical_session_id for src in sources}

        # Legacy filesystem fallback: only surface sessions the DB doesn't
        # already know about so we don't double-emit the same conversation.
        self._refresh_legacy_indexes()
        for path in self._legacy_session_files():
            data = self._load_json(path)
            logical_session_id = str(data.get("id") or path.stem)
            if logical_session_id in seen:
                continue
            canonical_path = canonicalize_path(path)
            source_id = build_source_id(self.provider, logical_session_id, canonical_path)
            created = (
                data.get("time", {}).get("created", "")
                if isinstance(data.get("time"), dict)
                else ""
            )
            sources.append(
                ProviderSource(
                    provider=self.provider,
                    source_kind=self.source_kind,
                    source_class="native",
                    source_id=source_id,
                    logical_session_id=logical_session_id,
                    path=str(path),
                    canonical_path=canonical_path,
                    project_root=data.get("cwd") or data.get("directory") or "unknown",
                    timestamp=normalize_timestamp(created),
                    status="eligible",
                )
            )
            seen.add(logical_session_id)
        return sources

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        """Parse an OpenCode session into normalized turns, resuming from ``cursor``."""
        if source.path == str(self.db_path):
            return self._parse_db_source(source, cursor)
        return self._parse_legacy_source(source, cursor)

    def _parse_db_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        # Refresh if discover wasn't called yet this pass (e.g. resumed cursor).
        if source.logical_session_id not in self._messages_by_session and not self._session_path_for:
            self._refresh_db_indexes()
        messages = self._messages_by_session.get(source.logical_session_id, [])
        emitted = set((cursor or {}).get("emitted_ids", []))
        turns: List[Dict] = []
        for index, message in enumerate(messages):
            message_id = str(message.get("id", ""))
            role = message.get("role", "")
            parts = self._parts_by_message.get(message_id, [])
            content_parts = [
                part.get("text", "")
                for part in parts
                if part.get("type") in {"text", "message"} and part.get("text")
            ]
            if not role or not content_parts:
                continue
            text = "\n".join(content_parts)
            doc_hash = hashlib.sha256(
                f"{source.logical_session_id}:{message_id}:{text}".encode("utf-8")
            ).hexdigest()[:16]
            doc_id = f"opencode:{source.logical_session_id}:{doc_hash}"
            if doc_id in emitted:
                continue
            emitted.add(doc_id)
            raw_ts: object = ""
            if isinstance(message.get("time"), dict):
                tc = message["time"].get("created")
                if tc is not None and tc != "":
                    raw_ts = tc
            if raw_ts == "":
                tc2 = message.get("__time_created_ms")
                if tc2 is not None and tc2 != "":
                    raw_ts = tc2
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
                "source_path": source.path,
                "transcript_file": Path(source.path).name,
                "turn_index": index,
                "timestamp": normalize_timestamp(raw_ts) or source.timestamp,
                "git_branch": "",
                "chunk_type": role or "turn",
                "project_root": source.project_root,
            })

        return ProviderParseResult(
            source=source,
            turns=turns,
            cursor={
                "cursor_type": "record_set",
                "logical_session_id": source.logical_session_id,
                "known_paths": [source.path],
                "emitted_ids": sorted(emitted),
                "project_root": source.project_root,
            },
        )

    def _parse_legacy_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        if not self._legacy_indexes_built:
            self._refresh_legacy_indexes()
        messages = list(self._legacy_messages_by_session.get(source.logical_session_id, []))
        parts = list(self._legacy_parts_by_session.get(source.logical_session_id, []))
        paths = [Path(source.path)] + [p for p, _ in messages] + [p for p, _ in parts]
        if not self._is_settled(paths):
            source.status = "pending"
            source.reason = "OpenCode records are still inside the settled window."
            return ProviderParseResult(source=source, turns=[], cursor=cursor or {})

        parts_by_message: Dict[str, List[Dict]] = {}
        for _, part in parts:
            message_id = part.get("messageID") or part.get("message_id")
            if message_id:
                parts_by_message.setdefault(str(message_id), []).append(part)

        emitted = set((cursor or {}).get("emitted_ids", []))
        turns = []
        for index, (_, message) in enumerate(messages):
            message_id = str(message.get("id", ""))
            role = message.get("role", "")
            content_parts = [
                part.get("text", "")
                for part in parts_by_message.get(message_id, [])
                if part.get("type") in {"text", "message"} and part.get("text")
            ]
            if not role or not content_parts:
                continue
            text = "\n".join(content_parts)
            doc_hash = hashlib.sha256(
                f"{source.logical_session_id}:{message_id}:{text}".encode("utf-8")
            ).hexdigest()[:16]
            doc_id = f"opencode:{source.logical_session_id}:{doc_hash}"
            if doc_id in emitted:
                continue
            emitted.add(doc_id)
            timestamp = ""
            if isinstance(message.get("time"), dict):
                timestamp = message["time"].get("created", "")
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
                "source_path": source.path,
                "transcript_file": Path(source.path).name,
                "turn_index": index,
                "timestamp": normalize_timestamp(timestamp) or source.timestamp,
                "git_branch": "",
                "chunk_type": role or "turn",
                "project_root": source.project_root,
            })

        return ProviderParseResult(
            source=source,
            turns=turns,
            cursor={
                "cursor_type": "record_set",
                "logical_session_id": source.logical_session_id,
                "known_paths": [str(p) for p in paths],
                "emitted_ids": sorted(emitted),
                "project_root": source.project_root,
            },
        )

    def _is_settled(self, paths: List[Path]) -> bool:
        if self.settled_seconds <= 0:
            return True
        now = time.time()
        return all(now - path.stat().st_mtime >= self.settled_seconds for path in paths if path.exists())

    def _legacy_orphan_part_count(self) -> int:
        if not self._legacy_indexes_built:
            self._refresh_legacy_indexes()
        count = 0
        for _, data in self._legacy_all_parts:
            message_id = data.get("messageID") or data.get("message_id")
            if message_id and str(message_id) not in self._legacy_all_message_ids:
                count += 1
        return count

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the storage tree and (if present) the SQLite parent as watch roots."""
        # Both the legacy tree and the SQLite WAL live under the same parent.
        roots = [ProviderWatchRoot(self.provider, self.source_kind, str(self.storage_root), recursive=True)]
        if self.db_path.exists():
            roots.append(
                ProviderWatchRoot(
                    self.provider,
                    self.source_kind,
                    str(self.db_path.parent),
                    recursive=False,
                )
            )
        return roots

    def health(self) -> ProviderHealth:
        """Report adapter health, including any legacy orphan-part count."""
        sources = self.discover_sources()
        orphan_parts = self._legacy_orphan_part_count()
        if not self.db_path.exists() and not self.storage_root.exists():
            status = "missing"
        elif orphan_parts:
            status = "warning"
        else:
            status = "ok"
        limitations: List[str] = []
        if orphan_parts:
            limitations.append("Incomplete OpenCode part/message records are left pending.")
        return ProviderHealth(
            provider=self.provider,
            status=status,
            source_count=len(sources),
            eligible_count=len(sources),
            error_count=orphan_parts,
            roots=[str(self.storage_root), str(self.db_path)],
            limitations=limitations,
        )
