#!/usr/bin/env python3
"""SessionStart hook: newest session summaries and recent turns from SessionFlow.

One file serves Claude Code and Codex. It reads the hook JSON on stdin, finds
the project from ``.claude/project.json`` (``memory.project``, else ``tag``),
calls the SessionFlow MCP server, and prints the context as hook JSON. When the
server cannot answer, the last cached summaries are shown with their time.
Standard library only; one deadline covers the whole run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEADLINE_S = 8.0
SERVER_URL = os.environ.get("SESSIONFLOW_URL", "http://127.0.0.1:7102/mcp/")
SUMMARY_COUNT = 3
RECALL_COUNT = 3
RECALL_AFTER_COMPACT = 5
# Context byte budgets per harness; summaries are cut first to fit.
BUDGET = {"claude_code_cli": 9000, "codex": 7000, None: 7000}
RECALL_TURN_CHARS = 400
CUT_MARK = "\n[cut to fit; read it in full with get_memory {id}]"

_started = time.monotonic()


def _remaining() -> float:
    return max(0.0, DEADLINE_S - (time.monotonic() - _started))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit(context: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                             "additionalContext": context}}))


# --- Inputs -----------------------------------------------------------------


def _harness(payload: dict) -> tuple:
    """Return (provider, agent_id) from the transcript path, or (None, None).

    A Codex session started from a Claude Code shell inherits
    CLAUDE_CODE_SESSION_ID, so the transcript location decides.
    """
    path = str(payload.get("transcript_path") or "")
    home = str(Path.home())
    session_id = str(payload.get("session_id") or "")
    if path.startswith(os.path.join(home, ".codex") + os.sep):
        return "codex", "codex"
    if path.startswith(os.path.join(home, ".claude") + os.sep) and \
            os.environ.get("CLAUDE_CODE_SESSION_ID") == session_id and session_id:
        return "claude_code_cli", "claude-code"
    return None, None


def _git_root(cwd: str) -> Path:
    try:
        out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=max(0.5, min(2.0, _remaining())))
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path(cwd)


def _project(root: Path) -> tuple:
    """Return (project slug, user id) from project.json, or (None, None)."""
    try:
        data = json.loads((root / ".claude" / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    block = data.get("memory") or {}
    project = block.get("project") or str(data.get("tag") or "").lower() or None
    return project, block.get("userId") or "lbruton"


# --- MCP --------------------------------------------------------------------


class ToolFailure(Exception):
    """A tool call that produced no usable result; the message says why."""


PROTOCOL_VERSION = "2025-06-18"


def _headers(session_id: str | None) -> dict:
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
        headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return headers


def _parse_reply(raw: bytes, content_type: str) -> dict:
    """Return the JSON-RPC message from a JSON body or the last SSE data event."""
    if not raw:
        return {}
    text = raw.decode("utf-8", "replace")
    if content_type.startswith("text/event-stream"):
        events = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        text = next((e for e in reversed(events) if e), "")
    try:
        return json.loads(text)
    except ValueError:
        raise ToolFailure("response is not JSON") from None


def _post(body: dict, session_id: str | None = None) -> tuple:
    """POST one JSON-RPC message; return the reply and the server's session ID."""
    timeout = _remaining()
    if timeout <= 0:
        raise ToolFailure("timed out")
    req = urllib.request.Request(SERVER_URL, data=json.dumps(body).encode(), method="POST",
                                 headers=_headers(session_id))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            sid = resp.headers.get("Mcp-Session-Id")
    except urllib.error.HTTPError as exc:
        raise ToolFailure(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ToolFailure(f"server unreachable ({reason})") from None
    return _parse_reply(raw, ctype), sid


def _close_session(session_id: str | None) -> None:
    """Ask the server to drop the session; the server expires it anyway on failure."""
    if not session_id:
        return
    req = urllib.request.Request(SERVER_URL, method="DELETE", headers=_headers(session_id))
    try:
        with urllib.request.urlopen(req, timeout=max(_remaining(), 0.5)):
            pass
    except (urllib.error.URLError, OSError):
        pass


def _call_tool(name: str, arguments: dict) -> dict:
    _, sid = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
        "clientInfo": {"name": "memory-session-start", "version": "1"}}})
    try:
        _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
        reply, _ = _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}}, sid)
    finally:
        _close_session(sid)
    return _tool_result(reply)


def _tool_result(reply: dict) -> dict:
    if "error" in reply:
        raise ToolFailure(f"JSON-RPC error {reply['error'].get('code')}")
    result = reply.get("result") or {}
    content = result.get("content") or []
    text = content[0].get("text", "") if content else ""
    if result.get("isError"):
        try:
            err = json.loads(text).get("error", {})
            raise ToolFailure(f"{err.get('code')}: {err.get('message')}")
        except ValueError:
            raise ToolFailure("tool error") from None
    try:
        data = json.loads(text)
    except ValueError:
        raise ToolFailure("result is not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ToolFailure("unexpected result shape")
    return data


def _in_thread(fn, *args):
    box: dict = {}

    def run():
        try:
            box["value"] = fn(*args)
        except ToolFailure as exc:
            box["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            box["error"] = type(exc).__name__

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


# --- Cache ------------------------------------------------------------------


def _cache_path(project: str) -> Path:
    return Path.home() / ".sessionflow" / "memory-cache" / f"{project}.json"


def _read_cache(path: Path) -> tuple:
    """Return (cache, problem); problem is missing, unreadable or damaged."""
    if not path.exists():
        return None, "missing"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, "unreadable"
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "damaged"
    if not isinstance(data, dict) or not isinstance(data.get("records"), list) \
            or not data.get("fetched_at"):
        return None, "damaged"
    return data, None


def _write_cache(path: Path, records: list, fetched_at: str) -> None:
    current, _ = _read_cache(path)
    if current and str(current.get("fetched_at")) > fetched_at:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": fetched_at, "records": records}, fh)
    os.replace(tmp, path)


# --- Rendering --------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:max(0, limit)].decode("utf-8", "ignore")


def _render(header: list, records: list, tail: list, budget: int) -> str:
    fixed = "\n".join(header + [""] + tail)
    room = budget - len(fixed.encode("utf-8")) - 64
    blocks = []
    share = max(0, room // max(1, len(records)))
    for rec in records:
        title = f"### {rec.get('created_at', '')[:10]} · {rec.get('memory_id')}"
        body = str(rec.get("text") or "")
        block = f"{title}\n{body}"
        if len(block.encode("utf-8")) > share:
            mark = CUT_MARK.format(id=rec.get("memory_id"))
            keep = share - len(mark.encode("utf-8"))
            block = _clip(block, max(len(title.encode("utf-8")) + 1, keep)) + mark
        blocks.append(block)
    text = "\n".join(header + blocks + [""] + tail)
    return _clip(text, budget)


def main() -> int:
    """Build and print the SessionStart context; always exits 0."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    provider, agent = _harness(payload)
    root = _git_root(str(payload.get("cwd") or os.getcwd()))
    project, user = _project(root)
    lines = ["## SessionFlow memory"]
    if provider:
        lines += [f"session_id: {payload.get('session_id')}", f"provider: {provider}",
                  f"agent_id: {agent}",
                  "Run /wrap at the end of the session; it saves its notes with this provenance."]
    else:
        lines.append("Unknown harness: no session provenance is recorded.")
    if not project:
        lines.append("No memory project: add a memory block or tag to .claude/project.json.")
        _emit("\n".join(lines))
        return 0
    lines.append(f"project: {project}")

    recall_n = RECALL_AFTER_COMPACT if payload.get("source") == "compact" else RECALL_COUNT
    list_thread, listed = _in_thread(_call_tool, "list_memories", {
        "project": project, "user_id": user, "type": "session-summary", "status": "active",
        "limit": SUMMARY_COUNT, "format": "json"})
    recall_thread, recalled = _in_thread(_call_tool, "search_all_sessions", {
        "project_root": str(root), "n": recall_n, "format": "json"})
    list_thread.join(_remaining())
    recall_thread.join(_remaining())

    tail: list = []
    records: list = []
    if "value" in listed:
        data = listed["value"]
        records = data["results"][:SUMMARY_COUNT]
        try:
            _write_cache(_cache_path(project), records, _now_iso())
        except OSError:
            tail.append("Warning: the memory cache could not be written.")
        dump = data.get("dump") or {}
        if dump.get("state") and dump["state"] != "ok":
            tail.append(f"Warning: nightly memory dump {dump['state']} "
                        f"(scheduled run {dump.get('scheduled_for', 'unknown')}).")
        if not records:
            lines.append("No session summaries yet for this project.")
        else:
            lines.append("Latest session notes, newest first:")
    else:
        reason = listed.get("error", "timed out")
        cache, problem = _read_cache(_cache_path(project))
        if cache is None:
            lines.append(f"Warning: memory store unavailable ({reason}) and the cache is "
                         f"{problem}; no memory context.")
        else:
            records = cache["records"][:SUMMARY_COUNT]
            lines.append(f"Memory store unavailable ({reason}); showing the cache written "
                         f"{cache['fetched_at']}:")

    if "value" in recalled:
        turns = recalled["value"]["results"]
        if turns:
            tail.append("Recent turns in this project:")
            for t in turns:
                content = " ".join(str(t.get("content") or "").split())[:RECALL_TURN_CHARS]
                tail.append(f"- [{t.get('timestamp', '')} {t.get('provider', '')}] {content}")
    else:
        tail.append(f"Transcript recall failed ({recalled.get('error', 'timed out')}).")

    _emit(_render(lines, records, tail, BUDGET.get(provider, BUDGET[None])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
