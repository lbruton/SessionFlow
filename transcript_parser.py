"""
Parse Claude Code JSONL transcripts into conversation turns for indexing.

Transcript format (one JSON object per line):
- type: "user"     → user message (content is str or list of tool_result)
- type: "assistant" → assistant message (content is list of text/tool_use/thinking blocks)
- type: "summary"  → compaction summary (short title string)
- type: "system", "file-history-snapshot", "progress", "queue-operation" → skip

Turn assembly:
1. A user entry with plain string content starts a new turn
2. Subsequent assistant text blocks are accumulated
3. Combined: "User: {user_text}\\n\\nAssistant: {assistant_text}"
4. Summary entries become standalone chunks

Incremental reading:
- index_state.json tracks last_byte_offset per transcript file
- On each invocation, seek to offset, read only new lines
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from provider_adapters import build_source_id, canonicalize_path


def detect_project_root(transcript_path: str, max_lines: int = 30) -> str:
    """Peek at a transcript's first lines to extract the working directory.

    Claude Code transcripts embed ``cwd`` on user/assistant/progress entries.
    We grab the first non-trivial ``cwd`` (i.e. not ``"/"`` or empty) and
    resolve it to a git toplevel when possible.  This lets us tag transcripts
    living in the generic ``-/`` bucket with their real project root.

    Returns "" if no usable cwd is found.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cwd = entry.get("cwd", "")
                if not cwd or cwd == "/":
                    continue
                # Accept paths with at least two components (e.g. /Users/x)
                if cwd.count("/") >= 2:
                    return _resolve_git_root(cwd)
    except (IOError, OSError):
        pass
    return ""


_git_root_cache: Dict[str, str] = {}


def _resolve_git_root(cwd: str) -> str:
    """Resolve *cwd* to its git repo toplevel using pure Python, with caching.

    Walks up from *cwd* looking for a ``.git`` entry (directory or file — git
    worktrees use a .git *file*).  Results are cached so repeated calls for
    the same cwd (common during backfill of 10K+ transcripts) never fork a
    subprocess.  This avoids the macOS fork+exec memory-pressure problem where
    each subprocess.run() transiently duplicates the parent's VM pages (~600 MB
    when the embedding model is loaded).
    """
    if cwd in _git_root_cache:
        return _git_root_cache[cwd]

    result = cwd  # fallback
    current = cwd
    while True:
        candidate = os.path.join(current, ".git")
        if os.path.exists(candidate):
            result = current
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    _git_root_cache[cwd] = result
    return result


def parse_transcript(
    transcript_path: str,
    session_id: str,
    start_offset: int = 0,
    max_turn_chars: int = 8000,
) -> Tuple[List[Dict], int]:
    """Parse a Claude Code JSONL transcript into indexable turns.

    Args:
        transcript_path: Path to the .jsonl file
        session_id: Session UUID for this transcript
        start_offset: Byte offset to resume reading from
        max_turn_chars: Maximum characters per turn text (truncate longer)

    Returns:
        (turns, new_offset) where turns is a list of dicts ready for rag_engine.add_turns()
        and new_offset is the byte position after the last line read.

        Only newline-terminated lines are consumed. If the file ends in a
        half-written line (Claude Code mid-write), parsing stops there and
        new_offset points at its start so the next call re-reads it whole. A
        user turn still pending at that point is not emitted; new_offset
        rewinds to its user line so the turn is rebuilt with its reply.
    """
    turns = []
    current_user_text = None
    current_user_start_byte = 0
    current_assistant_texts = []
    current_timestamp = ""
    current_git_branch = ""

    file_size = os.path.getsize(transcript_path)
    if start_offset >= file_size:
        return [], file_size

    transcript_file = os.path.basename(transcript_path)

    # Binary mode: offsets are true byte positions, and a tail cut inside a
    # multi-byte character can't raise before we see the line is incomplete.
    with open(transcript_path, "rb") as f:
        f.seek(start_offset)
        current_offset = start_offset
        truncated_tail = False

        for raw_line in f:
            if not raw_line.endswith(b"\n"):
                truncated_tail = True
                break

            line_bytes = len(raw_line)
            current_offset += line_bytes

            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Track git branch from any entry that has it
            entry_branch = entry.get("gitBranch", "")
            if entry_branch:
                current_git_branch = entry_branch

            # Track timestamp from any entry that has it
            entry_ts = entry.get("timestamp", "")
            if entry_ts:
                current_timestamp = entry_ts

            # Skip non-conversation entries
            if entry_type in ("file-history-snapshot", "progress", "system", "queue-operation"):
                continue

            # Handle summary entries (compaction summaries)
            if entry_type == "summary":
                # Flush any pending turn first
                if current_user_text is not None:
                    turn = _build_turn(
                        current_user_text, current_assistant_texts,
                        session_id, transcript_file,
                        current_user_start_byte,
                        current_timestamp, current_git_branch,
                        max_turn_chars, "turn",
                    )
                    if turn:
                        turns.append(turn)
                    current_user_text = None
                    current_assistant_texts = []

                summary_text = entry.get("summary", "")
                if summary_text:
                    summary_full = f"Session Summary: {summary_text}"
                    # Use current byte offset as turn_index for summaries
                    summary_byte = current_offset - len(line.encode("utf-8")) if line else current_offset
                    content_hash = hashlib.sha256(
                        f"{summary_byte}:{summary_full}".encode()
                    ).hexdigest()[:16]
                    turns.append({
                        "text": summary_full,
                        "doc_id": f"{session_id}::{content_hash}",
                        "session_id": session_id,
                        "transcript_file": transcript_file,
                        "turn_index": summary_byte,
                        "timestamp": current_timestamp,
                        "git_branch": current_git_branch,
                        "chunk_type": "summary",
                    })
                continue

            # Handle user messages
            if entry_type == "user":
                message = entry.get("message", {})
                content = message.get("content", "")

                # Skip tool_result messages (content is a list)
                if isinstance(content, list):
                    continue

                # Skip isMeta messages
                if entry.get("isMeta"):
                    continue

                # Skip empty content
                if not isinstance(content, str) or not content.strip():
                    continue

                # Flush previous turn if we have one
                if current_user_text is not None:
                    turn = _build_turn(
                        current_user_text, current_assistant_texts,
                        session_id, transcript_file,
                        current_user_start_byte,
                        current_timestamp, current_git_branch,
                        max_turn_chars, "turn",
                    )
                    if turn:
                        turns.append(turn)

                # Start new turn — record byte position for content-based doc_id
                current_user_text = content.strip()
                current_user_start_byte = current_offset - line_bytes
                current_assistant_texts = []
                continue

            # Handle assistant messages
            if entry_type == "assistant":
                message = entry.get("message", {})
                content = message.get("content", [])

                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            current_assistant_texts.append(text)

        # Half-written tail with a turn still open: its reply may be the line
        # being written. Leave the turn unflushed and resume at its user line.
        if truncated_tail and current_user_text is not None:
            return turns, current_user_start_byte

        # Flush final pending turn
        if current_user_text is not None:
            turn = _build_turn(
                current_user_text, current_assistant_texts,
                session_id, transcript_file,
                current_user_start_byte,
                current_timestamp, current_git_branch,
                max_turn_chars, "turn",
            )
            if turn:
                turns.append(turn)

    return turns, current_offset


def _build_turn(
    user_text: str,
    assistant_texts: List[str],
    session_id: str,
    transcript_file: str,
    start_byte: int,
    timestamp: str,
    git_branch: str,
    max_chars: int,
    chunk_type: str,
) -> Optional[Dict]:
    """Build a turn dict from user + assistant text.

    doc_id uses a content hash (SHA-256 of byte position + text) to guarantee
    uniqueness across incremental parses. This avoids the turn_index reset bug
    where subsequent parse batches would collide with earlier ones.

    turn_index uses the byte offset where the user message starts. This is
    naturally monotonic across incremental parses (later turns always have
    higher byte offsets), making get_turns context browsing work correctly.
    """
    parts = [f"User: {user_text}"]
    if assistant_texts:
        combined_assistant = "\n\n".join(assistant_texts)
        parts.append(f"Assistant: {combined_assistant}")

    text = "\n\n".join(parts)

    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"

    # Skip very short turns (likely just whitespace or newlines)
    if len(text.strip()) < 20:
        return None

    # Content-addressed doc_id: hash of byte position + text.
    # - Byte position ensures identical text at different positions gets unique IDs
    # - Content ensures re-indexing the same position produces the same ID (idempotent)
    content_hash = hashlib.sha256(
        f"{start_byte}:{text}".encode()
    ).hexdigest()[:16]

    return {
        "text": text,
        "doc_id": f"{session_id}::{content_hash}",
        "session_id": session_id,
        "transcript_file": transcript_file,
        "turn_index": start_byte,
        "timestamp": timestamp,
        "git_branch": git_branch,
        "chunk_type": chunk_type,
    }


# --- Index state management ---

_STATE_DIR = Path.home() / ".sessionflow"
_STATE_PATH = _STATE_DIR / "index_state.json"
_migrated = False


def _migrate_per_project_states(state: Dict):
    """One-time migration: merge per-project index_state.json files into the global state."""
    global _migrated
    if _migrated:
        return
    _migrated = True

    # Known per-project state locations
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.is_dir():
        return

    # Scan for any .sessionflow/index_state.json under common project roots
    home = Path.home()
    candidates = []
    for idea_dir in [home / "IdeaProjects", home / "git-repos"]:
        if idea_dir.is_dir():
            for project_dir in idea_dir.iterdir():
                state_file = project_dir / ".sessionflow" / "index_state.json"
                if state_file.exists():
                    candidates.append((str(project_dir), state_file))

    if not candidates:
        return

    transcripts = state.setdefault("transcripts", {})
    merged_count = 0

    for project_root, state_file in candidates:
        try:
            with open(state_file) as f:
                old_state = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        old_transcripts = old_state.get("transcripts", {})
        for tpath, tdata in old_transcripts.items():
            if tpath not in transcripts:
                # Add project_root to the migrated entry
                entry = dict(tdata)
                entry["project_root"] = project_root
                transcripts[tpath] = entry
                merged_count += 1

        # Preserve the latest expire check
        old_expire = old_state.get("last_expire_check", 0)
        if old_expire > state.get("last_expire_check", 0):
            state["last_expire_check"] = old_expire

    if merged_count:
        import sys
        print(f"[state] Migrated {merged_count} transcript entries from "
              f"{len(candidates)} per-project states", file=sys.stderr)


def load_index_state() -> Dict:
    """Load centralized index state from ~/.sessionflow/index_state.json."""
    state = {}
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            state = {}

    _migrate_per_project_states(state)
    _ensure_provider_cursor_state(state)
    return state


def _ensure_provider_cursor_state(state: Dict):
    """Ensure state has provider-scoped cursor storage plus legacy mirror.

    Existing Claude rows remain under ``transcripts`` for compatibility, while
    SESF-6 provider-aware code can use ``providers.<provider>.<source_id>``.
    """
    state.setdefault("version", 2)
    state.setdefault("providers", {})
    state.setdefault("legacy_transcripts", {})

    transcripts = state.get("transcripts", {})
    claude_cursors = state["providers"].setdefault("claude_code_cli", {})
    for transcript_path, entry in transcripts.items():
        if not isinstance(entry, dict):
            continue
        legacy_entry = dict(entry)
        state["legacy_transcripts"].setdefault(transcript_path, legacy_entry)
        path = Path(transcript_path)
        logical_session_id = path.stem
        canonical_path = canonicalize_path(transcript_path)
        source_id = build_source_id("claude_code_cli", logical_session_id, canonical_path)
        claude_cursors.setdefault(source_id, {
            "cursor_type": "byte_offset",
            "logical_session_id": logical_session_id,
            "known_paths": [transcript_path],
            "last_byte_offset": entry.get("last_byte_offset", 0),
            "last_step_index": 0,
            "emitted_ids": [],
            "last_mtime_ns": 0,
            "project_root": entry.get("project_root", ""),
            "updated_at": "",
        })


def get_provider_cursor(state: Dict, provider: str, source_id: str) -> Dict:
    """Return provider cursor state for a source id."""
    _ensure_provider_cursor_state(state)
    return state.get("providers", {}).get(provider, {}).get(source_id, {})


def set_provider_cursor(state: Dict, provider: str, source_id: str, cursor: Dict):
    """Set provider cursor state for a source id."""
    _ensure_provider_cursor_state(state)
    state["providers"].setdefault(provider, {})[source_id] = cursor


def save_index_state(state: Dict):
    """Atomically save centralized index state to ~/.sessionflow/index_state.json.

    Writes a temp file in the same directory, then ``os.replace``s it over the
    target, so a crash mid-write never truncates the resume cursors. On
    failure the temp file is removed and the original is left untouched.
    """
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATE_PATH.with_name(f"{_STATE_PATH.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, _STATE_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_transcript_offset(state: Dict, transcript_path: str) -> int:
    """Get the last indexed byte offset for a transcript file."""
    return state.get("transcripts", {}).get(transcript_path, {}).get("last_byte_offset", 0)


def set_transcript_offset(state: Dict, transcript_path: str, offset: int,
                          project_root: str = ""):
    """Update the byte offset for a transcript file."""
    if "transcripts" not in state:
        state["transcripts"] = {}
    if transcript_path not in state["transcripts"]:
        state["transcripts"][transcript_path] = {}
    state["transcripts"][transcript_path]["last_byte_offset"] = offset
    if project_root:
        state["transcripts"][transcript_path]["project_root"] = project_root
