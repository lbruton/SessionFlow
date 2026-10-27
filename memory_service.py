"""Memory validation, write orchestration, search and related-conversation lookups.

Memory records are curated notes written by agents (session summaries,
decisions, handoffs, facts). They are stored verbatim with fixed metadata and
no LLM step. The agent vocabulary here is separate from the transcript
provider allowlist in ``provider_adapters``: one names who wrote a memory, the
other names where a transcript came from.
"""

from __future__ import annotations

LEGAL_MEMORY_TYPES = frozenset({
    "session-summary",
    "decision",
    "handoff",
    "fact",
    "workflow",
    "discovery",
    "contact",
    "blocker",
    "preference",
    "open-item",
    "milestone",
    "guardrail",
})

LEGAL_MEMORY_AGENTS = frozenset({
    "claude-code",
    "codex",
    "gemini",
    "opencode",
    "antigravity",
})

# Accepted only on the import path, for mem0 records with no usable writer.
# Later updates that omit agent_id keep it.
IMPORT_ONLY_AGENT = "unknown"

MAX_MEMORY_BYTES = 65536
"""Largest accepted memory text, counted in UTF-8 bytes."""

DEFAULT_USER_ID = "lbruton"
