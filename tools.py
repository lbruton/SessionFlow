"""
MCP tool definitions and project context for SessionFlow.
"""

import asyncio
import contextvars
import json
import os
from pathlib import Path
from mcp.server import Server
from mcp import types

import rag_engine
import sanitize
from project_scope import ProjectScope, resolve_project_scope
from provider_adapters import (
    LEGAL_PROVIDERS,
    LEGAL_SORT_BY,
    LEGAL_SOURCE_KINDS,
    is_valid_issue_token,
)


# --- Project context ---

_current_project_root: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_project_root", default=None
)


def set_current_project_root(root: str | None):
    """Set the per-request project root in the context var."""
    _current_project_root.set(root)


def get_current_project_root() -> str | None:
    """Return the per-request project root from the context var."""
    return _current_project_root.get()


SCOPE_INSTRUCTION = (
    "Pass your current working directory as 'project_root', or '*' to search "
    "every project. Without either, the call is refused unless the client names "
    "the project."
)

PROJECT_ROOT_PROPERTY = {
    "type": "string",
    "description": (
        "Pass your current working directory. A path covers its subfolders and "
        "worktrees, a folder name such as 'CoWork' matches by name, and '*' searches "
        "every project. A name or path with no indexed turns searches every project, "
        "and the Scope line says so."
    ),
}

UNNAMED_SCOPE_ERROR = (
    "No project named. Pass project_root as your current working directory, "
    "or '*' to search every project."
)


def get_db_path() -> str:
    """Milvus URI — remote Standalone if SESSIONFLOW_MILVUS_URI is set, else local Lite."""
    return os.getenv("SESSIONFLOW_MILVUS_URI", str(Path.home() / ".sessionflow" / "milvus.db"))


async def resolve_scope(arg: str | None, header_root: str | None, db_path: str) -> ProjectScope:
    """Resolve a tool's project argument against the indexed roots.

    The root lookup reads SQLite, so it runs off the event loop. ``'*'`` and an
    unnamed scope need no lookup.
    """
    if arg == "*":
        return ProjectScope(roots=None)
    if not (arg or "").strip() and not (header_root or "").strip():
        return resolve_project_scope(None, None, ())
    known = await asyncio.to_thread(rag_engine.known_project_roots, db_path)
    return resolve_project_scope(arg, header_root, known)


def _scope_line(scope: ProjectScope) -> str:
    line = f"Scope: {scope.label()}"
    return f"{line} ({scope.note})" if scope.note else line


def _validate_enum_arg(name: str, value, legal_values) -> "types.TextContent | None":
    """Return an error TextContent if `value` is non-None and outside `legal_values`.

    Shared by the search handlers so provider/source_kind/sort_by validation stays
    consistent in one place rather than duplicated per tool.
    """
    if value is not None and value not in legal_values:
        allowed = ", ".join(sorted(legal_values))
        return types.TextContent(
            type="text",
            text=f"Invalid {name}: {value!r}; expected one of: {allowed}",
        )
    return None


# --- Formatting helpers ---

def _format_signals(r: dict) -> str:
    """Name the signals that ranked a row.

    ``distance`` is the vector distance (None when the vector search did not
    return the row); ``keyword_rank`` is the 1-based FTS position. A row with
    neither comes from a query-less listing.
    """
    distance = r.get("distance")
    keyword_rank = r.get("keyword_rank")
    parts = []
    if distance is not None:
        parts.append(f"Semantic: {1 - distance:.2f}")
    if keyword_rank is not None:
        parts.append(f"Keyword: #{keyword_rank}")
    return " · ".join(parts) if parts else "Listing"


def format_results(results: list[dict], scope: ProjectScope | None = None) -> str:
    """Format search results as markdown, opening with the scope line when ``scope`` is set."""
    output = [_scope_line(scope), ""] if scope is not None else []
    if not results:
        return "\n".join(output + ["No results found."])

    for i, r in enumerate(results, 1):
        # Header with metadata
        session_id = r.get("session_id", "")
        branch = r.get("git_branch", "")
        ts = r.get("timestamp", "")[:19]  # trim to readable
        chunk_type = r.get("chunk_type", "turn")
        score = _format_signals(r)

        turn_index = r.get("turn_index", 0)

        project = r.get("project_root", "")
        provider = r.get("provider", "")
        source_kind = r.get("source_kind", "")

        header_parts = [f"**Result {i}**"]
        if ts:
            header_parts.append(f"({ts})")
        if branch:
            header_parts.append(f"[{branch}]")
        if project:
            header_parts.append(f"project:{Path(project).name}")
        if provider:
            header_parts.append(f"provider:{provider}")
        if source_kind:
            header_parts.append(f"source:{source_kind}")
        if session_id:
            header_parts.append(f"session:{session_id}")

        output.append(" ".join(header_parts))

        meta = f"*Turn: {turn_index} | Type: {chunk_type} | {score}*"
        output.append(meta)
        output.append("")
        output.append(r.get("content", ""))
        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


def format_turns(results: list[dict]) -> str:
    """Format get_turns results as markdown."""
    if not results:
        return "No turns found."

    output = []
    for r in results:
        turn_index = r.get("turn_index", 0)
        ts = r.get("timestamp", "")[:19]
        chunk_type = r.get("chunk_type", "turn")
        branch = r.get("git_branch", "")

        header_parts = [f"**Turn {turn_index}**"]
        if ts:
            header_parts.append(f"({ts})")
        if branch:
            header_parts.append(f"[{branch}]")

        output.append(" ".join(header_parts))
        output.append(f"*Type: {chunk_type}*")
        output.append("")
        output.append(r.get("content", ""))
        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


def format_stats(stats: dict, db_path: str, scope: ProjectScope | None = None) -> str:
    """Format index statistics, opening with the scope line when ``scope`` is set."""
    lines = [_scope_line(scope), ""] if scope is not None else []
    lines += [
        f"**Total Turns:** {stats['total_turns']}",
        f"**Sessions:** {stats['sessions']}",
    ]

    if stats.get("branches"):
        lines.append(f"**Branches:** {', '.join(stats['branches'])}")

    if stats.get("by_type"):
        lines.append("\n### By Type")
        for t, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {t}: {count}")

    if stats.get("providers"):
        lines.append("\n### Providers")
        for provider, count in sorted(stats["providers"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {provider}: {count}")

    lines.append(f"\n**Index Location:** {db_path}")
    return "\n".join(lines)


def format_sanitize_report(report: "sanitize.SanitizeReport") -> str:
    """Format a sanitize :class:`SanitizeReport` as value-free markdown.

    Renders the run mode, per-rule detection counts, affected/processed turn
    counts, the FTS-incomplete count, run status, and the audit-file path. On an
    apply run the rotate-the-key warning is appended. The report carries only rule
    names, integer counts, and paths — never a secret value — so the rendered text
    is safe to surface to the operator.

    Args:
        report: The outcome of :func:`sanitize.scan` or :func:`sanitize.apply`.

    Returns:
        A markdown string summarizing the run.
    """
    lines = [f"**Sanitize ({report.mode})** — status: {report.status}"]

    if report.counts:
        lines.append("\n### Detections by rule")
        for rule, count in sorted(report.counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {rule}: {count}")
    else:
        lines.append("\nNo secrets detected in scope.")

    lines.append(f"\n**Affected turns:** {report.affected_count}")
    if report.mode != "dry-run":
        lines.append(f"**Processed turns:** {report.processed_count}")
        if report.incomplete_fts:
            lines.append(
                f"**FTS-incomplete turns (retry needed):** {report.incomplete_fts}"
            )

    if report.audit_path:
        lines.append(f"**Audit file:** {report.audit_path}")

    if report.rotate_warning:
        lines.append(
            "\nWARNING: redaction is not key rotation. Rotate any exposed "
            "credential now — removing it from the index does not invalidate it."
        )

    return "\n".join(lines)


def format_timeline(entries: list[dict]) -> str:
    """Format an issue timeline feed as markdown (oldest first).

    Renders each entry's matched ``doc_id`` alongside provider/session/timestamp
    metadata so the rendered text references every turn in the feed (the MCP
    transport contract; mirrors the HTTP route's structured feed).
    """
    if not entries:
        return "No turns reference that issue."

    output = []
    for i, e in enumerate(entries, 1):
        ts = (e.get("timestamp", "") or "")[:19]
        provider = e.get("provider", "")
        session_id = e.get("session_id", "")
        role = e.get("role", e.get("chunk_type", ""))
        doc_id = e.get("doc_id", "")

        header_parts = [f"**{i}.**"]
        if ts:
            header_parts.append(f"({ts})")
        if provider:
            header_parts.append(f"provider:{provider}")
        if session_id:
            header_parts.append(f"session:{session_id}")
        if role:
            header_parts.append(f"role:{role}")
        if doc_id:
            header_parts.append(f"doc_id:{doc_id}")

        output.append(" ".join(header_parts))
        output.append("")
        output.append(e.get("text", "") or e.get("content", ""))
        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


# --- Tool registration ---


def build_search_all_sessions_schema() -> dict:
    """Return the JSON-schema for the ``search_all_sessions`` tool input."""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query. Omit it or pass an empty string to list the most recent turns, newest first.",
            },
            "n": {
                "type": "integer",
                "description": "Number of turns to return (default: 10).",
                "default": 10,
            },
            "git_branch": {
                "type": "string",
                "description": "Return only turns from this git branch, for example 'develop'.",
            },
            "project_root": PROJECT_ROOT_PROPERTY,
            "provider": {
                "type": "string",
                "description": "Return only turns from this provider, for example 'codex', 'opencode' or 'antigravity_cli'.",
            },
            "source_kind": {
                "type": "string",
                "description": "Return only turns from this source kind, for example 'codex_rollout_jsonl'.",
            },
            "sort_by": {
                "type": "string",
                "enum": sorted(LEGAL_SORT_BY),
                "description": "Ranking: 'hybrid' blends relevance and recency (default), 'relevance' ranks by match only, 'recency' lists newest first.",
                "default": "hybrid",
            },
            "date_from": {
                "type": "string",
                "description": "Earliest date to include, as an ISO date, for example '2026-04-02'.",
            },
            "date_to": {
                "type": "string",
                "description": "Latest date to include, as an ISO date, for example '2026-04-02'.",
            },
            "issue_id": {
                "type": "string",
                "description": "Return only turns tagged with this issue ID, for example 'SESF-25'. Case-insensitive.",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "json"],
                "default": "markdown",
                "description": "Output format: 'markdown' (default) or 'json'.",
            },
        },
        "required": [],
    }

# --- Memory tools ---

# Set by the HTTP server once the memory store has started; None until then.
_memory_service = None

MEMORY_AGENTS = ["antigravity", "claude-code", "codex", "gemini", "opencode"]
MEMORY_TYPES = ["blocker", "contact", "decision", "discovery", "fact", "guardrail", "handoff",
                "milestone", "open-item", "preference", "session-summary", "workflow"]
MEMORY_TOOL_NAMES = frozenset({"add_memory", "search_memories", "list_memories", "get_memory",
                               "update_memory", "delete_memory", "get_memory_context"})

_FORMAT_PROP = {"type": "string", "enum": ["markdown", "json"], "default": "markdown",
                "description": "Output format: 'markdown' (default) or 'json'."}
_PROJECT_PROP = {"type": "string",
                 "description": "Project slug, for example 'sessionflow'. Reads accept '*' for every project."}
_ID_PROP = {"type": "string", "description": "Memory ID returned by add_memory or a listing."}
_PROVENANCE_PROPS = {
    "source_session_id": {"type": "string", "description": "Session ID that produced the memory."},
    "source_provider": {"type": "string", "description": "Provider of that session, for example 'claude_code_cli'."},
    "source_project_root": {"type": "string", "description": "Project root of that session."},
    "source_started_at": {"type": "string", "description": "Session start, ISO 8601 with offset."},
    "source_ended_at": {"type": "string", "description": "Session end, ISO 8601 with offset."},
}
_FILTER_PROPS = {
    "type": {"type": "string", "enum": MEMORY_TYPES, "description": "Only this memory type."},
    "agent_id": {"type": "string", "description": "Only memories written by this agent."},
    "status": {"type": "string", "enum": ["active", "superseded", "any"],
               "description": "Record status; default 'active'."},
    "issue_id": {"type": "string", "description": "Only memories that mention this issue ID."},
    "source": {"type": "string", "description": "Only memories whose extra.source matches."},
    "category": {"type": "string", "description": "Only memories whose extra.category matches."},
    "created_from": {"type": "string", "description": "First America/Chicago day, YYYY-MM-DD."},
    "created_to": {"type": "string", "description": "Last America/Chicago day, YYYY-MM-DD."},
    "user_id": {"type": "string", "description": "Owner; default 'lbruton'."},
}
_FILTER_KEYS = tuple(_FILTER_PROPS) + ("source_session_id",)


def set_memory_service(service) -> None:
    """Install the memory service the memory tools call, or None to mark memory unavailable."""
    global _memory_service
    _memory_service = service


def build_memory_tools() -> list[types.Tool]:
    """Return the seven memory tool definitions."""
    def tool(name, description, props, required):
        return types.Tool(name=name, description=description, inputSchema={
            "type": "object", "properties": {**props, "format": _FORMAT_PROP}, "required": required,
        })

    return [
        tool("add_memory",
             "Store one memory verbatim: a session summary, decision, handoff or fact. A record "
             "with the same topic_key in the project is replaced; the same text is deduplicated.",
             {"text": {"type": "string", "description": "Memory text, at most 65,536 UTF-8 bytes."},
              "agent_id": {"type": "string", "enum": MEMORY_AGENTS, "description": "Harness writing the memory."},
              "project": _PROJECT_PROP,
              "type": {"type": "string", "enum": MEMORY_TYPES, "description": "Memory type."},
              "user_id": {"type": "string", "description": "Owner; default 'lbruton'."},
              "topic_key": {"type": "string", "description": "Stable key; a later add with it replaces the record."},
              "extra": {"type": "object", "description": "Free-form metadata, kept as given."},
              "supersedes": {"type": "array", "items": {"type": "string"},
                             "description": "IDs in the same project that this memory replaces."},
              **_PROVENANCE_PROPS},
             ["text", "agent_id", "project", "type"]),
        tool("search_memories",
             "Search memories by meaning and keywords, validated against the canonical store.",
             {"query": {"type": "string", "description": "What to find."},
              "project": _PROJECT_PROP,
              "n": {"type": "integer", "default": 10, "description": "Number of results."},
              "sort_by": {"type": "string", "enum": sorted(LEGAL_SORT_BY), "default": "hybrid",
                          "description": "Ranking strategy."},
              **_FILTER_PROPS},
             ["query", "project"]),
        tool("list_memories",
             "List memories newest first. Pass the returned cursor to read the next page.",
             {"project": _PROJECT_PROP,
              "limit": {"type": "integer", "default": 20, "description": "Page size."},
              "cursor": {"type": "string", "description": "Cursor from the previous page."},
              "source_session_id": {"type": "string", "description": "Only memories from this session."},
              **_FILTER_PROPS},
             ["project"]),
        tool("get_memory", "Return one memory in full.",
             {"memory_id": _ID_PROP, "project": _PROJECT_PROP}, ["memory_id"]),
        tool("update_memory",
             "Change fields of one memory. Omitted fields keep their values; a new text replaces the old one.",
             {"memory_id": _ID_PROP, "project": _PROJECT_PROP,
              "text": {"type": "string", "description": "New full text."},
              "agent_id": {"type": "string", "enum": MEMORY_AGENTS, "description": "New writer."},
              "type": {"type": "string", "enum": MEMORY_TYPES, "description": "New type."},
              "topic_key": {"type": "string", "description": "New topic key; empty clears it."},
              "extra": {"type": "object", "description": "New metadata object (replaces the old one)."},
              **_PROVENANCE_PROPS},
             ["memory_id"]),
        tool("delete_memory", "Delete one memory. Its history is kept.",
             {"memory_id": _ID_PROP, "project": _PROJECT_PROP}, ["memory_id"]),
        tool("get_memory_context",
             "Return a memory with the conversation that produced it and other sessions about its issues.",
             {"memory_id": _ID_PROP, "project": _PROJECT_PROP,
              "limit": {"type": "integer", "default": 10, "description": "Items per section."}},
             ["memory_id"]),
    ]


def _json_text(payload) -> str:
    return json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2)


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _memory_error(code: str, message: str, fmt: str, details: dict | None = None) -> types.CallToolResult:
    if fmt == "json":
        body = {"error": {"code": code, "message": message, **({"details": details} if details else {})}}
        text = _json_text(body)
    else:
        text = f"Error ({code}): {message}"
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def _when(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _format_memory(row: dict, heading: str = "") -> str:
    meta = " · ".join(str(x) for x in (row.get("type"), row.get("project"), row.get("agent_id"),
                                        row.get("status"), _when(row.get("created_at"))) if x)
    lines = [f"{heading}`{row['memory_id']}`" + (f" · {meta}" if meta else "")]
    if row.get("issue_ids"):
        lines.append(f"Issues: {', '.join(row['issue_ids'])}")
    if row.get("text"):
        lines.append("")
        lines.append(row["text"])
    return "\n".join(lines)


def _format_write(result: dict) -> str:
    lines = [f"Memory `{result['memory_id']}` {result['outcome']} (version {result.get('version')}). "
             f"Index: {result.get('index', 'ok')}."]
    similar = result.get("similar") or []
    if similar:
        lines.append("")
        lines.append("Similar memories:")
        for s in similar:
            lines.append(f"- `{s['memory_id']}` ({s.get('score', 0):.2f}): {s.get('text', '')}")
    if result.get("similar_state") == "pending":
        lines.append("Similarity check is pending: some vectors are not indexed yet.")
    return "\n".join(lines)


def _format_memory_list(payload: dict) -> str:
    rows = payload.get("results") or []
    if not rows:
        parts = ["No memories found."]
    else:
        parts = [_format_memory(r, f"### {i}. ") for i, r in enumerate(rows, 1)]
    if payload.get("next_cursor"):
        parts.append(f"Next page cursor: `{payload['next_cursor']}`")
    if payload.get("channels") is not None:
        parts.append(f"Channels: {', '.join(payload['channels']) or 'none'}")
    if payload.get("incomplete"):
        parts.append("Results may be incomplete: the indexes are catching up.")
    return "\n\n".join(parts)


def _format_context(payload: dict) -> str:
    parts = [_format_memory(payload["memory"], "## Memory ")]
    source = payload.get("source_session")
    if source:
        parts.append(f"## Source session `{source.get('session_id', '')}`")
        for t in source.get("turns") or []:
            parts.append(f"- turn {t.get('turn_index')}: {t.get('content', '')}")
    related = payload.get("related_sessions") or []
    parts.append("## Related sessions")
    parts.extend(f"- `{s.get('session_id')}` ({s.get('provider', '')}, {s.get('last_timestamp', '')})"
                 for s in related)
    if not related:
        parts.append("None found.")
    if payload.get("note"):
        parts.append(payload["note"])
    return "\n\n".join(parts)


async def _call_memory_tool(name: str, arguments: dict):
    args = dict(arguments or {})
    fmt = args.pop("format", "markdown")
    service = _memory_service
    if service is None:
        return _memory_error("memory_unavailable",
                             "The memory store has not started; see /health for the reason.", fmt)
    project = args.get("project")
    try:
        if name == "add_memory":
            payload = await service.add(args)
            text = _format_write(payload)
        elif name == "get_memory":
            payload = await service.get(args["memory_id"], project)
            text = _format_memory(payload)
        elif name == "update_memory":
            memory_id = args.pop("memory_id")
            payload = await service.update(memory_id, args)
            text = _format_write(payload)
        elif name == "delete_memory":
            payload = await service.delete(args["memory_id"], project)
            text = _format_write(payload)
        elif name == "list_memories":
            filters = {k: args[k] for k in ("project", *_FILTER_KEYS) if k in args}
            payload = await service.list(filters, args.get("limit", 20), args.get("cursor"))
            text = _format_memory_list(payload)
        elif name == "search_memories":
            filters = {k: args[k] for k in ("project", *_FILTER_KEYS) if k in args}
            if args.get("sort_by") not in (None, "hybrid"):
                payload = await service.search(args.get("query") or "", filters, args.get("n", 10),
                                               sort_by=args["sort_by"])
            else:
                payload = await service.search(args.get("query") or "", filters, args.get("n", 10))
            text = _format_memory_list(payload)
        else:  # get_memory_context
            payload = await service.context(args["memory_id"], project=project,
                                            limit=args.get("limit", 10))
            text = _format_context(payload)
    except KeyError as exc:
        return _memory_error("missing_field", f"{exc.args[0]} is required", fmt)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a structured tool error
        code = getattr(exc, "code", None) or "internal_error"
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        details = getattr(exc, "details", None)
        return _memory_error(code, message, fmt, details if isinstance(details, dict) else None)
    if fmt == "json":
        text = _json_text(payload)
    return [types.TextContent(type="text", text=text)]


def register_tools(server: Server):
    """Register SessionFlow MCP tools."""

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return build_memory_tools() + [
            types.Tool(
                name="search_session",
                description=(
                    "Return past turns (discussions, decisions, code, errors) that match "
                    "'query'. " + SCOPE_INSTRUCTION + " Omit 'query' to list the most "
                    "recent turns, newest first. Call search_all_sessions for branch, "
                    "provider, date or issue filters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query, for example 'error in deploy script'. Omit it or pass an empty string to list the most recent turns, newest first.",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of turns to return (default: 5).",
                            "default": 5,
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from a prior search result. Restricts the search to that one session, so no project is needed. The server cannot see the live session ID, so there is no 'current session' value.",
                        },
                        "project_root": PROJECT_ROOT_PROPERTY,
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "hybrid"],
                            "description": "Ranking: 'hybrid' blends relevance and recency (default), 'relevance' ranks by match only, 'recency' lists newest first.",
                            "default": "hybrid",
                        },
                        "issue_id": {
                            "type": "string",
                            "description": "Return only turns tagged with this issue ID, for example 'SESF-25'. Case-insensitive.",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="search_all_sessions",
                description=(
                    "Return past turns that match 'query', filtered by project, git "
                    "branch, provider, date range or issue ID. " + SCOPE_INSTRUCTION + " "
                    "Omit 'query' to list the most recent turns, newest first, to recall "
                    "recent context. Call get_turns with a result's session_id and "
                    "turn_index to read a hit in context."
                ),
                inputSchema=build_search_all_sessions_schema(),
            ),
            types.Tool(
                name="get_turns",
                description=(
                    "Return the turns before and after one turn of a session. Requires "
                    "'session_id' and 'turn_index', both from a search result. Call it after "
                    "search_session or search_all_sessions to read a hit in context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from a search result.",
                        },
                        "turn_index": {
                            "type": "integer",
                            "description": "Turn index from a search result. The window is centered on this turn.",
                        },
                        "context": {
                            "type": "integer",
                            "description": "Number of turns to include before and after the center turn (default: 2).",
                            "default": 2,
                        },
                    },
                    "required": ["session_id", "turn_index"],
                },
            ),
            types.Tool(
                name="get_session_stats",
                description=(
                    "Return turn, session, branch and provider counts for a project, "
                    "and the index location. " + SCOPE_INSTRUCTION
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"project_root": PROJECT_ROOT_PROPERTY},
                    "required": [],
                },
            ),
            types.Tool(
                name="get_issue_timeline",
                description=(
                    "Return every turn that mentions an issue ID, oldest first, across all "
                    "providers and without duplicates. Requires 'issue_id', for example "
                    "'SESF-25'. Turns match on their issue tag or on a keyword search, so "
                    "untagged mentions appear too. Pass 'limit', 'provider' or a date range "
                    "to narrow the feed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "issue_id": {
                            "type": "string",
                            "description": "Issue ID to build the timeline for, for example 'SESF-25'. Case-insensitive.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of turns to return (default: 50).",
                            "default": 50,
                        },
                        "provider": {
                            "type": "string",
                            "description": "Return only turns from this provider, for example 'codex', 'opencode' or 'antigravity_cli'.",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Earliest date to include, as an ISO date, for example '2026-04-02'.",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "Latest date to include, as an ISO date, for example '2026-04-30'.",
                        },
                    },
                    "required": ["issue_id"],
                },
            ),
            types.Tool(
                name="cleanup_sessions",
                description=(
                    "Delete turns from the index and return the number deleted. Pass at "
                    "least one of 'max_age_days', 'session_id' or 'git_branch'. Deletion "
                    "covers every project and cannot be undone."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_age_days": {
                            "type": "integer",
                            "description": "Delete turns older than this many days.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Delete every turn of this session.",
                        },
                        "git_branch": {
                            "type": "string",
                            "description": "Delete every turn from this git branch, in every project.",
                        },
                        "project_root": {
                            "type": "string",
                            "description": "Accepted but not applied. Deletion is not scoped to a project.",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="sanitize_index",
                description=(
                    "Return a dry-run report of secrets already in the index: per-rule "
                    "counts, the number of affected turns and an audit-file path. The "
                    "report holds counts and offsets, never secret values. Pass apply=true "
                    "and confirm=true to redact and re-embed the affected turns, or add "
                    "drop=true to delete them. Rotate every exposed credential after an "
                    "apply, because redaction does not revoke it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "apply": {
                            "type": "boolean",
                            "description": "Change the index. Default false: report only.",
                            "default": False,
                        },
                        "drop": {
                            "type": "boolean",
                            "description": "With apply, delete the affected turns instead of redacting them. Default false.",
                            "default": False,
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Required with apply. Without it, apply refuses and changes nothing.",
                            "default": False,
                        },
                        "project_root": {
                            "type": "string",
                            "description": "Scan only this project path.",
                        },
                        "provider": {
                            "type": "string",
                            "description": "Scan only this provider, for example 'claude_code_cli' or 'codex'.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Scan only this session.",
                        },
                        "since": {
                            "type": "string",
                            "description": "Scan only turns at or after this ISO date or timestamp, for example '2026-04-02'.",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name in MEMORY_TOOL_NAMES:
            return await _call_memory_tool(name, arguments)
        db = get_db_path()
        current_project = get_current_project_root()

        try:
            if name == "search_session":
                session_id = arguments.get("session_id")
                sort_by_arg = arguments.get("sort_by", "hybrid")
                err = _validate_enum_arg("sort_by", sort_by_arg, LEGAL_SORT_BY)
                if err:
                    return [err]
                issue_id_arg = arguments.get("issue_id")
                if issue_id_arg is not None and not is_valid_issue_token(issue_id_arg):
                    return [types.TextContent(
                        type="text",
                        text="issue_id must be a valid issue token like SESF-25")]
                scope = await resolve_scope(arguments.get("project_root"), current_project, db)
                if scope.unnamed and not session_id:
                    return [types.TextContent(type="text", text=UNNAMED_SCOPE_ERROR)]
                # search_async runs on the mlx-embed thread; a blocking search()
                # here would run MLX on the event loop and race backfill.
                results = await rag_engine.search_async(
                    arguments.get("query") or "",
                    arguments.get("n", 5),
                    session_id=session_id,
                    project_root=scope.filter_value(),
                    sort_by=sort_by_arg,
                    issue_id=issue_id_arg,
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_results(results, scope))]

            elif name == "search_all_sessions":
                provider_arg = arguments.get("provider")
                source_kind_arg = arguments.get("source_kind")
                sort_by_arg = arguments.get("sort_by", "hybrid")
                for err in (
                    _validate_enum_arg("provider", provider_arg, LEGAL_PROVIDERS),
                    _validate_enum_arg("source_kind", source_kind_arg, LEGAL_SOURCE_KINDS),
                    _validate_enum_arg("sort_by", sort_by_arg, LEGAL_SORT_BY),
                ):
                    if err:
                        return [err]

                issue_id_arg = arguments.get("issue_id")
                if issue_id_arg is not None and not is_valid_issue_token(issue_id_arg):
                    return [types.TextContent(
                        type="text",
                        text="issue_id must be a valid issue token like SESF-25")]

                scope = await resolve_scope(arguments.get("project_root"), current_project, db)
                if scope.unnamed:
                    return [types.TextContent(type="text", text=UNNAMED_SCOPE_ERROR)]
                results = await rag_engine.search_async(
                    arguments.get("query") or "",
                    arguments.get("n", 10),
                    git_branch=arguments.get("git_branch"),
                    project_root=scope.filter_value(),
                    sort_by=sort_by_arg,
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to"),
                    provider=arguments.get("provider"),
                    source_kind=arguments.get("source_kind"),
                    issue_id=issue_id_arg,
                    db_path=db,
                )
                if arguments.get("format") == "json":
                    payload = {
                        "scope": {"roots": scope.filter_value(), "note": scope.note},
                        "results": results,
                    }
                    return [types.TextContent(type="text", text=_json_text(payload))]
                return [types.TextContent(type="text", text=format_results(results, scope))]

            elif name == "get_turns":
                results = rag_engine.get_turns(
                    arguments["session_id"],
                    arguments["turn_index"],
                    context=arguments.get("context", 2),
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_turns(results))]

            elif name == "get_session_stats":
                scope = await resolve_scope(arguments.get("project_root"), current_project, db)
                if scope.unnamed:
                    return [types.TextContent(type="text", text=UNNAMED_SCOPE_ERROR)]
                stats = rag_engine.get_stats(
                    project_root=scope.filter_value(),
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_stats(stats, db, scope))]

            elif name == "get_issue_timeline":
                if not is_valid_issue_token(arguments.get("issue_id")):
                    return [types.TextContent(
                        type="text",
                        text="issue_id must be a valid issue token like SESF-25")]
                provider = arguments.get("provider")
                err = _validate_enum_arg("provider", provider, LEGAL_PROVIDERS)
                if err:
                    return [err]
                limit = arguments.get("limit", 50)
                if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                    return [types.TextContent(
                        type="text", text="limit must be a positive integer")]
                entries = await rag_engine.get_issue_timeline_async(
                    arguments["issue_id"],
                    limit=limit,
                    providers=[provider] if provider else None,
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to"),
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_timeline(entries))]

            elif name == "cleanup_sessions":
                max_age = arguments.get("max_age_days")
                sid = arguments.get("session_id")
                branch = arguments.get("git_branch")

                if not any([max_age, sid, branch]):
                    return [types.TextContent(
                        type="text",
                        text="Specify at least one of: max_age_days, session_id, git_branch",
                    )]

                parts = []
                if max_age:
                    count = rag_engine.delete_older_than(max_age, db_path=db)
                    parts.append(f"Deleted {count} turns older than {max_age} days")
                if sid:
                    count = rag_engine.delete_by_session(sid, db_path=db)
                    parts.append(f"Deleted {count} turns for session {sid[:12]}")
                if branch:
                    count = rag_engine.delete_by_branch(branch, db_path=db)
                    parts.append(f"Deleted {count} turns for branch '{branch}'")

                stats = rag_engine.get_stats(
                    project_root=current_project,
                    db_path=db,
                )
                parts.append(f"\nRemaining: {stats['total_turns']} turns across {stats['sessions']} sessions")
                return [types.TextContent(type="text", text="\n".join(parts))]

            elif name == "sanitize_index":
                do_apply = bool(arguments.get("apply", False))
                do_drop = bool(arguments.get("drop", False))
                do_confirm = bool(arguments.get("confirm", False))

                scope = sanitize.Scope(
                    project_root=arguments.get("project_root"),
                    provider=arguments.get("provider"),
                    session_id=arguments.get("session_id"),
                    since=arguments.get("since"),
                )

                if not do_apply:
                    # Offload to a thread: scan reads/iterates many turns and
                    # would otherwise block the asyncio event loop.
                    report = await asyncio.to_thread(sanitize.scan, scope)
                    return [types.TextContent(
                        type="text", text=format_sanitize_report(report))]

                if not do_confirm:
                    # Refuse before any read or write: apply requires explicit confirm.
                    action = "delete" if do_drop else "redact"
                    alt = "" if do_drop else " (drop=true to delete instead)"
                    return [types.TextContent(
                        type="text",
                        text=(
                            "Refusing to apply: confirmation required. Re-run with "
                            f"confirm=true to {action}{alt} the affected turns. "
                            "No changes were made."
                        ),
                    )]

                # Offload to a thread: apply re-embeds many turns and would
                # otherwise block the asyncio event loop for the whole run.
                report = await asyncio.to_thread(
                    sanitize.apply, scope, drop=do_drop, confirmed=True)
                return [types.TextContent(
                    type="text", text=format_sanitize_report(report))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except (Exception, asyncio.CancelledError) as e:
            return [types.TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
