"""Shared provider contracts for SessionFlow ingestion.

Provider-specific modules parse their native artifacts and return normalized
turn dictionaries that the existing storage/search layers can index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import hashlib
import re


LEGAL_PROVIDERS = frozenset({
    "claude_code_cli",
    "claude_desktop_cowork",
    "codex",
    "opencode",
    "antigravity_cli",
    "antigravity_desktop",
})

LEGAL_SOURCE_KINDS = frozenset({
    "claude_code_jsonl",
    "claude_desktop_sessions",
    "codex_rollout_jsonl",
    "opencode_storage",
    "antigravity_cli_transcript_jsonl",
    "antigravity_desktop_transcript_jsonl",
    "legacy_gemini_history",
    "terminal_log",
})

LEGAL_SOURCE_CLASSES = frozenset({
    "native",
    "fallback",
})

# Canonical issue-token shape (e.g. SESF-25), matching the ingestion extractor.
ISSUE_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


def is_valid_issue_token(raw: object) -> bool:
    """Whether ``raw`` is a well-formed issue token like ``SESF-25``.

    Case-insensitive and whitespace-tolerant (callers should ``.strip().upper()``
    before storing/matching). Rejects empty/non-str input and anything carrying
    LIKE/FTS metacharacters or whitespace, so a filter token can't broaden a
    Milvus ``LIKE`` or break an FTS ``MATCH``.
    """
    return (
        isinstance(raw, str)
        and ISSUE_ID_PATTERN.match(raw.strip().upper()) is not None
    )

LEGAL_SORT_BY = frozenset({
    "relevance",
    "recency",
    "hybrid",
})

LEGAL_SOURCE_STATUSES = frozenset({
    "eligible",
    "pending",
    "unsupported",
    "error",
})

LEGAL_CURSOR_TYPES = frozenset({
    "byte_offset",
    "record_set",
    "step_index",
})

LEGAL_HEALTH_STATUSES = frozenset({
    "ok",
    "missing",
    "warning",
    "partial",
    "probe-only",
    "error",
})


def _validate_choice(field_name: str, value: str, legal_values: frozenset[str]) -> None:
    if value not in legal_values:
        allowed = ", ".join(sorted(legal_values))
        raise ValueError(f"Invalid {field_name}: {value!r}; expected one of: {allowed}")


def canonicalize_path(path: str | Path) -> str:
    """Return a symlink-resolved absolute path for deduplication."""
    return str(Path(path).expanduser().resolve())


def build_source_id(provider: str, logical_session_id: str, canonical_path: str) -> str:
    """Create a stable provider-scoped source id without exposing huge paths."""
    _validate_choice("provider", provider, LEGAL_PROVIDERS)
    digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{logical_session_id}:{digest}"


def normalize_timestamp(value: Any) -> str:
    """Coerce a provider-emitted timestamp to a Milvus-compatible VARCHAR(64) string.

    Accepts ISO 8601 strings (returned unchanged after str()), numeric Unix epochs
    in seconds or milliseconds (auto-detected by magnitude), datetime instances,
    or None/empty. Used by every provider before turn emission so the Milvus
    schema's VARCHAR(64) `timestamp` field never receives an int.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        # Heuristic: anything > 1e12 is milliseconds (≈ year 33658 in seconds).
        seconds = float(value) / 1000.0 if abs(value) > 1e12 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    return str(value)


def default_provider_metadata() -> Dict[str, str]:
    """Provider defaults for legacy Claude rows that lack explicit metadata."""
    return {
        "provider": "claude_code_cli",
        "source_kind": "claude_code_jsonl",
        "source_class": "native",
    }


@dataclass
class ProviderSource:
    """A discovered source artifact (transcript/session file) for one provider."""

    provider: str
    source_kind: str
    source_class: str
    source_id: str
    logical_session_id: str
    path: str
    project_root: str
    timestamp: str
    status: str
    canonical_path: Optional[str] = None
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate enum-constrained fields and derive the canonical path."""
        _validate_choice("provider", self.provider, LEGAL_PROVIDERS)
        _validate_choice("source_kind", self.source_kind, LEGAL_SOURCE_KINDS)
        _validate_choice("source_class", self.source_class, LEGAL_SOURCE_CLASSES)
        _validate_choice("status", self.status, LEGAL_SOURCE_STATUSES)
        if not self.logical_session_id:
            raise ValueError("logical_session_id is required")
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.canonical_path:
            self.canonical_path = canonicalize_path(self.path)


@dataclass
class ProviderCursor:
    """Per-session ingestion progress so re-runs resume instead of reprocessing."""

    cursor_type: str
    logical_session_id: str
    known_paths: List[str] = field(default_factory=list)
    last_byte_offset: int = 0
    last_step_index: int = 0
    emitted_ids: List[str] = field(default_factory=list)
    last_mtime_ns: int = 0
    project_root: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Validate the cursor type against the legal set."""
        _validate_choice("cursor_type", self.cursor_type, LEGAL_CURSOR_TYPES)

    def as_state(self) -> Dict[str, Any]:
        """Return the cursor as a plain JSON-serializable state dict."""
        return {
            "cursor_type": self.cursor_type,
            "logical_session_id": self.logical_session_id,
            "known_paths": list(self.known_paths),
            "last_byte_offset": self.last_byte_offset,
            "last_step_index": self.last_step_index,
            "emitted_ids": list(self.emitted_ids),
            "last_mtime_ns": self.last_mtime_ns,
            "project_root": self.project_root,
            "updated_at": self.updated_at,
        }


@dataclass
class ProviderParseResult:
    """Outcome of parsing one source: normalized turns plus the updated cursor."""

    source: ProviderSource
    turns: List[Dict[str, Any]]
    cursor: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


@dataclass
class ProviderHealth:
    """Per-provider ingestion health and source-count rollup."""

    provider: str
    status: str
    source_count: int = 0
    eligible_count: int = 0
    pending_count: int = 0
    unsupported_count: int = 0
    error_count: int = 0
    limitations: List[str] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the provider and status against their legal sets."""
        _validate_choice("provider", self.provider, LEGAL_PROVIDERS)
        _validate_choice("status", self.status, LEGAL_HEALTH_STATUSES)


@dataclass
class ProviderWatchRoot:
    """A filesystem root to watch for a provider's incoming sources."""

    provider: str
    source_kind: str
    path: str
    recursive: bool = True

    def __post_init__(self) -> None:
        """Validate the provider and source kind against their legal sets."""
        _validate_choice("provider", self.provider, LEGAL_PROVIDERS)
        _validate_choice("source_kind", self.source_kind, LEGAL_SOURCE_KINDS)


class ProviderAdapter(Protocol):
    """Protocol implemented by provider-specific source adapters."""

    provider: str

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate the provider's available sources."""
        ...

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict[str, Any]],
    ) -> ProviderParseResult:
        """Parse one source into normalized turns, resuming from ``cursor``."""
        ...

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the filesystem roots to watch for new sources."""
        ...

    def health(self) -> ProviderHealth:
        """Report the provider's ingestion health."""
        ...
