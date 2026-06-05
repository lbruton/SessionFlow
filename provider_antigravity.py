"""Antigravity JSONL transcript provider adapter."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlparse
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


_VARINT_MAX_BYTES = 10  # max bytes for a 64-bit base-128 varint (D-1).


def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a base-128 varint from ``data`` at ``pos``.

    Returns ``(value, next_pos)``. Caps the varint at 10 bytes (the max for a
    64-bit value) and raises ``ValueError`` past that so a malformed/truncated
    stream can't spin in an infinite loop (D-1). Raises ``IndexError`` on a
    short read past the end of the buffer.
    """
    value = 0
    shift = 0
    for read in range(_VARINT_MAX_BYTES):
        byte = data[pos + read]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos + read + 1
        shift += 7
    raise ValueError("varint exceeds 10 bytes (truncated or malformed stream)")


def _iter_length_delimited(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(field_number, payload)`` for every wire-type-2 field in ``data``.

    Non-length-delimited fields are silently skipped by advancing ``pos`` by the
    correct byte count for each wire type:
    - wt=0 (varint): consume a varint (reuse ``_read_varint``) and discard.
    - wt=1 (fixed64): advance ``pos`` by 8 bytes.
    - wt=2 (length-delimited): read length prefix, yield ``(field_number, payload)``.
    - wt=5 (fixed32): advance ``pos`` by 4 bytes.
    - wt=3/4 (deprecated group start/end): raises ``ValueError`` — these are
      obsolete and never appear in real Antigravity data; the defensive loader in
      ``_load_summaries`` catches this and returns ``{}`` (AC-5).

    ``IndexError`` from a short read propagates up to ``_load_summaries`` for
    graceful degradation to ``"unknown"`` (AC-5).
    """
    pos = 0
    length = len(data)
    while pos < length:
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            # Varint field — consume and discard.
            _, pos = _read_varint(data, pos)
        elif wire_type == 1:
            # 64-bit fixed field — skip 8 bytes.
            pos += 8
        elif wire_type == 2:
            # Length-delimited — yield payload to caller.
            size, pos = _read_varint(data, pos)
            end = pos + size
            if end > length:
                raise IndexError("length-delimited payload runs past end of buffer")
            yield field_number, data[pos:end]
            pos = end
        elif wire_type == 5:
            # 32-bit fixed field — skip 4 bytes.
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type} for field {field_number}")


def _walk_length_delimited(data: bytes) -> Dict[str, str]:
    """Decode the Antigravity ``agyhub_summaries_proto.pb`` wire format.

    Schema-free varint / length-delimited walker (stdlib only, D-1). The file is
    a sequence of **repeated top-level ``Field 1`` records** (SESF-29): the
    conversation UUID and workspace submessage live one level *deeper*, inside
    each record, not flat at the top level. Each top-level ``Field 1`` payload is
    one conversation record decoded by ``_decode_record``; top-level ``Field 2``
    entries are unrelated recency metadata (no workspace) and are skipped.

    Returns a ``{conversation_id: workspace_uri}`` map. Records missing either the
    UUID or the workspace URI are dropped. Tolerates ``UnicodeDecodeError`` while
    decoding individual strings.
    """
    mapping: Dict[str, str] = {}
    for field_number, payload in _iter_length_delimited(data):
        if field_number != 1:
            continue
        record = _decode_record(payload)
        if record is not None:
            conversation_id, uri = record
            mapping[conversation_id] = uri
    return mapping


def _decode_record(record: bytes) -> Optional[Tuple[str, str]]:
    """Decode one conversation record into a ``(conversation_id, workspace_uri)`` pair.

    Inner ``Field 1`` carries the conversation UUID; inner ``Field 2`` carries the
    workspace submessage, decoded recursively via the nested ``Field 2->9->1`` path
    (``_extract_workspace_uri``).

    Field-order independent: protobuf does not guarantee that ``Field 1`` precedes
    ``Field 2`` within a record, so both halves are collected before pairing.
    Returns ``None`` when either half is absent (the real file carries a handful
    of workspace-less records) or fails to decode as UTF-8.

    Per-record fault isolation: a single malformed/truncated record (a varint or
    length-delimited frame that runs past its own bounds) raises ``IndexError``/
    ``ValueError`` from the inner walker. That is caught here and the one bad
    record is dropped, rather than propagating to ``_load_summaries`` and discarding
    every other valid mapping in the file (AC-5 spirit, extended per-record).
    """
    pending_id: Optional[str] = None
    pending_uri: Optional[str] = None
    try:
        for field_number, payload in _iter_length_delimited(record):
            if field_number == 1:
                try:
                    pending_id = payload.decode("utf-8")
                except UnicodeDecodeError:
                    pending_id = None
            elif field_number == 2:
                pending_uri = _extract_workspace_uri(payload)
    except (IndexError, ValueError):
        return None
    if pending_id is not None and pending_uri is not None:
        return pending_id, pending_uri
    return None


def _extract_workspace_uri(field2: bytes) -> Optional[str]:
    """Decode the nested Field 2->9->1 submessage path to a workspace URI string.

    Returns the decoded URI, or ``None`` if the expected nesting is absent or a
    string fails to decode as UTF-8.
    """
    for field_number, payload in _iter_length_delimited(field2):
        if field_number != 9:
            continue
        for inner_number, inner_payload in _iter_length_delimited(payload):
            if inner_number != 1:
                continue
            try:
                return inner_payload.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def _normalize_file_uri(value: str) -> Optional[str]:
    """Normalize a ``file://`` URI (or bare absolute path) to a filesystem path.

    Built on ``urllib.parse`` (``urlparse`` + ``unquote``): a ``file://`` URI has
    its scheme stripped and percent-encoding decoded (AC-2). A bare **absolute**
    path (starts with ``/``) is accepted and percent-decoded. Anything else
    (relative paths, non-file schemes, malformed strings) returns ``None`` so it
    can't poison project-scoped search (D-6, AC-7).
    """
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        return path if path.startswith("/") else None
    if not parsed.scheme:
        path = unquote(value)
        return path if path.startswith("/") else None
    return None


def _load_summaries(root: Path) -> Dict[str, str]:
    """Load the desktop summaries metadata into a ``{conversation_id: path}`` map.

    Reads ``<root>/agyhub_summaries_proto.pb``, walks it (``_walk_length_delimited``),
    and normalizes each workspace URI (``_normalize_file_uri``), dropping entries
    whose URI normalizes to ``None``. All file I/O and binary decoding is wrapped
    defensively: any failure (``OSError``/``PermissionError``/``IndexError``/
    ``ValueError``/``UnicodeDecodeError``) returns ``{}`` (AC-5).
    """
    pb_path = root / "agyhub_summaries_proto.pb"
    try:
        data = pb_path.read_bytes()
        raw = _walk_length_delimited(data)
    except (OSError, IndexError, ValueError, UnicodeDecodeError):
        return {}
    mapping: Dict[str, str] = {}
    for conversation_id, uri in raw.items():
        normalized = _normalize_file_uri(uri)
        if normalized is not None:
            mapping[conversation_id] = normalized
    return mapping


class AntigravityAdapter:
    """Adapter for Antigravity transcripts (CLI and Desktop variants)."""

    def __init__(self, home: str | Path | None = None, source_kind: str = "cli"):
        """Set the home and select the CLI or Desktop provider variant."""
        self.home = Path(home).expanduser() if home is not None else Path.home()
        self.variant = source_kind
        if source_kind == "desktop":
            self.provider = "antigravity_desktop"
            self.source_kind = "antigravity_desktop_transcript_jsonl"
            self.root = self.home / ".gemini" / "antigravity"
        elif source_kind == "cli":
            self.provider = "antigravity_cli"
            self.source_kind = "antigravity_cli_transcript_jsonl"
            self.root = self.home / ".gemini" / "antigravity-cli"
        else:
            raise ValueError(f"unknown source_kind: {source_kind!r}")

    def _load_history(self) -> Dict[str, str]:
        history_path = self.root / "history.jsonl"
        mapping: Dict[str, str] = {}
        if not history_path.exists():
            return mapping
        try:
            lines = history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return mapping
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversation_id = (
                entry.get("conversation_id")
                or entry.get("conversationId")
                or entry.get("id")
            )
            workspace = entry.get("workspace") or entry.get("cwd") or entry.get("project_root")
            if conversation_id and workspace:
                mapping[str(conversation_id)] = str(workspace)
        return mapping

    def discover_sources(self) -> List[ProviderSource]:
        """Enumerate Antigravity transcripts under the brain directory."""
        transcript_glob = "brain/*/.system_generated/logs/transcript.jsonl"
        history = self._load_history()
        # Desktop history.jsonl is empty; consult the summaries metadata as a
        # second resolution layer (SESF-17). Gated to the desktop variant so the
        # CLI path is provably untouched (AC-6). Parsed once per discovery pass,
        # mirroring _load_history(). The loader parses the summaries .pb, so a
        # desktop source's project_root resolves history -> summaries -> "unknown".
        summaries: Dict[str, str] = {}
        if self.variant == "desktop":
            summaries = _load_summaries(self.root)
        sources = []
        for path in sorted(self.root.glob(transcript_glob)):
            conversation_id = path.parents[2].name
            canonical_path = canonicalize_path(path)
            source_id = build_source_id(self.provider, conversation_id, canonical_path)
            sources.append(ProviderSource(
                provider=self.provider,
                source_kind=self.source_kind,
                source_class="native",
                source_id=source_id,
                logical_session_id=conversation_id,
                path=str(path),
                canonical_path=canonical_path,
                project_root=history.get(
                    conversation_id, summaries.get(conversation_id, "unknown")
                ),
                timestamp="",
                status="eligible",
            ))
        return sources

    def parse_source(
        self,
        source: ProviderSource,
        cursor: Optional[Dict],
    ) -> ProviderParseResult:
        """Parse an Antigravity transcript into normalized turns, resuming from ``cursor``."""
        last_step = int((cursor or {}).get("last_step_index", -1))
        emitted = set((cursor or {}).get("emitted_ids", []))
        turns = []
        try:
            lines = Path(source.path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return ProviderParseResult(source=source, turns=[], cursor=cursor or {}, errors=[str(exc)])

        for line_number, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # TODO(SESF-6+): step_index can collide across logical sessions if
            # cursor provenance is shared (e.g. resumed conversation with a
            # rewritten transcript). Revisit cursor scoping once we have
            # ground-truth examples of multi-session step_index reuse.
            raw_step = entry.get("step_index")
            raw_step_alt = entry.get("stepIndex")
            try:
                step_index = int(raw_step if raw_step is not None else (
                    raw_step_alt if raw_step_alt is not None else line_number
                ))
            except (TypeError, ValueError):
                warnings.warn(
                    f"provider_antigravity: malformed step_index {raw_step!r}/{raw_step_alt!r}"
                    f" at line {line_number} of {source.path!r}; skipping record",
                    stacklevel=1,
                )
                continue
            if step_index <= last_step:
                continue
            text = entry.get("text") or entry.get("content") or entry.get("message") or ""
            if not text:
                continue
            # Antigravity transcripts use `created_at`; older / desktop variants
            # have also been seen with `timestamp`. Accept either.
            raw_ts = next(
                (
                    v for k in ("timestamp", "created_at", "createdAt", "time")
                    if (v := entry.get(k)) is not None and v != ""
                ),
                None,
            )
            doc_hash = hashlib.sha256(
                f"{source.logical_session_id}:{step_index}:{text}".encode("utf-8")
            ).hexdigest()[:16]
            doc_id = f"{self.provider}:{source.logical_session_id}:{doc_hash}"
            if doc_id in emitted:
                continue
            emitted.add(doc_id)
            turns.append({
                "text": str(text),
                "content": str(text),
                "doc_id": doc_id,
                "session_id": source.logical_session_id,
                "logical_session_id": source.logical_session_id,
                "provider": self.provider,
                "source_kind": self.source_kind,
                "source_class": "native",
                "source_id": source.source_id,
                "source_path": source.path,
                "transcript_file": Path(source.path).name,
                "turn_index": step_index,
                "timestamp": normalize_timestamp(raw_ts) or source.timestamp,
                "git_branch": "",
                "chunk_type": entry.get("type", "turn"),
                "project_root": source.project_root,
            })
            last_step = max(last_step, step_index)

        return ProviderParseResult(
            source=source,
            turns=turns,
            cursor={
                "cursor_type": "step_index",
                "logical_session_id": source.logical_session_id,
                "known_paths": [source.path],
                "last_step_index": last_step,
                "emitted_ids": sorted(emitted),
                "project_root": source.project_root,
            },
        )

    def _has_opaque_binary_artifacts(self) -> bool:
        return any(self.root.glob("brain/**/*.pb")) or any(self.root.glob("brain/**/*.db"))

    def watch_roots(self) -> List[ProviderWatchRoot]:
        """Return the brain directory as the single recursive watch root."""
        return [ProviderWatchRoot(self.provider, self.source_kind, str(self.root / "brain"), recursive=True)]

    def health(self) -> ProviderHealth:
        """Report adapter health for the active CLI or Desktop variant."""
        sources = self.discover_sources()
        if self.variant == "desktop":
            # SESF-17 (AC-8 / D-7): root summaries metadata is now parsed and
            # consulted for project_root resolution. The genuinely-opaque
            # per-conversation brain/**/*.pb / *.db artifacts remain unparsed,
            # so keep naming them — but do not claim full protobuf schema parsing.
            limitations = [
                "Root summaries metadata is parsed for project resolution; "
                "per-conversation brain/**/*.pb (protobuf) and brain/**/*.db artifacts remain opaque in SESF-6."
            ]
        else:
            limitations = ["Protobuf/database artifacts are not parsed in SESF-6."]
        if self._has_opaque_binary_artifacts():
            limitations.append("Opaque protobuf/database artifacts detected; JSONL remains authoritative.")
        return ProviderHealth(
            provider=self.provider,
            status="ok" if sources else ("missing" if not self.root.exists() else "partial"),
            source_count=len(sources),
            eligible_count=len(sources),
            roots=[str(self.root)],
            limitations=limitations,
        )
