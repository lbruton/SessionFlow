"""Session-start hook: latest summaries, transcript recall, cache fallback, byte budget.

The hook runs as a subprocess, as both harnesses run it, against a stdlib fake MCP
server. HOME points at a temp directory so the cache lives there.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "memory-session-start.py"


def _summary(i, size=200, created="2026-10-2{}T12:00:00+00:00"):
    return {"memory_id": f"m-{i}", "text": f"summary {i} " + "x" * size, "type": "session-summary",
            "project": "demo", "created_at": created.format(i), "status": "active"}


class FakeMCP:
    """JSON-RPC over HTTP, scripted per tool name."""

    def __init__(self):
        self.calls: list[dict] = []
        self.scripts: dict[str, dict] = {
            "list_memories": {"kind": "ok", "payload": {
                "results": [_summary(3), _summary(2), _summary(1)], "next_cursor": None,
                "dump": {"state": "ok"}}},
            "search_all_sessions": {"kind": "ok", "payload": {"results": [
                {"doc_id": "d1", "content": "recalled turn one", "timestamp": "2026-10-25T10:00:00Z",
                 "session_id": "s", "provider": "codex", "project_root": "/p"}]}},
        }
        self.status_code = 200
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return None

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append(body)
                if fake.status_code != 200:
                    self.send_response(fake.status_code)
                    self.end_headers()
                    return
                method = body.get("method")
                if method == "initialize":
                    return self._send({"jsonrpc": "2.0", "id": body["id"], "result": {
                        "protocolVersion": "2025-06-18", "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "0"}}})
                if method and method.startswith("notifications/"):
                    self.send_response(202)
                    self.end_headers()
                    return
                name = body["params"]["name"]
                script = fake.scripts[name]
                if script.get("delay"):
                    time.sleep(script["delay"])
                kind = script["kind"]
                if kind == "rpc_error":
                    return self._send({"jsonrpc": "2.0", "id": body["id"],
                                       "error": {"code": -32603, "message": "boom"}})
                if kind == "is_error":
                    text = json.dumps({"error": {"code": "memory_unavailable", "message": "starting"}})
                    return self._send({"jsonrpc": "2.0", "id": body["id"], "result": {
                        "content": [{"type": "text", "text": text}], "isError": True}})
                if kind == "bad_json":
                    text = "No results found."
                elif kind == "wrong_shape":
                    text = json.dumps({"unexpected": True})
                else:
                    text = json.dumps(script["payload"])
                self._send({"jsonrpc": "2.0", "id": body["id"], "result": {
                    "content": [{"type": "text", "text": text}], "isError": False}})

            def _send(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tool_calls(self, name):
        return [c["params"]["arguments"] for c in self.calls
                if c.get("method") == "tools/call" and c["params"]["name"] == name]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def mcp():
    server = FakeMCP()
    yield server
    server.close()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "demo-repo"
    (root / ".claude").mkdir(parents=True)
    (root / ".claude" / "project.json").write_text(json.dumps(
        {"tag": "demo", "memory": {"userId": "lbruton", "project": "demo"}}))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _run(mcp_server, project, home, *, source="startup", session_id="sess-123",
         harness="claude", env_extra=None, timeout=20):
    home = Path(home)
    if harness == "claude":
        transcript = home / ".claude" / "projects" / "demo" / f"{session_id}.jsonl"
    else:
        transcript = home / ".codex" / "sessions" / "2026" / "10" / "26" / f"rollout-x-{session_id}.jsonl"
    payload = {"session_id": session_id, "cwd": str(project), "source": source,
               "hook_event_name": "SessionStart", "transcript_path": str(transcript)}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "CODEX"))}
    env.update(HOME=str(home), SESSIONFLOW_URL=mcp_server.url if mcp_server else "http://127.0.0.1:9/mcp/")
    if harness == "claude":
        env["CLAUDE_CODE_SESSION_ID"] = session_id
    env.update(env_extra or {})
    started = time.monotonic()
    proc = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), text=True,
                          capture_output=True, env=env, cwd=str(project), timeout=timeout)
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    return context, elapsed


def _cache(home):
    return Path(home) / ".sessionflow" / "memory-cache" / "demo.json"


# --- Provenance (AC-40) ------------------------------------------------------


def test_claude_code_session_is_detected(mcp, project, tmp_path):
    ctx, _ = _run(mcp, project, tmp_path)
    assert "session_id: sess-123" in ctx
    assert "provider: claude_code_cli" in ctx and "agent_id: claude-code" in ctx
    assert "/wrap" in ctx


def test_codex_session_is_detected_even_with_an_inherited_claude_id(mcp, project, tmp_path):
    ctx, _ = _run(mcp, project, tmp_path, harness="codex", session_id="019a-codex",
                  env_extra={"CLAUDE_CODE_SESSION_ID": "parent-claude-session", "CLAUDECODE": "1"})
    assert "session_id: 019a-codex" in ctx
    assert "provider: codex" in ctx and "agent_id: codex" in ctx


def test_unknown_harness_emits_no_provenance(mcp, project, tmp_path):
    ctx, _ = _run(mcp, project, tmp_path, env_extra={"CLAUDE_CODE_SESSION_ID": "someone-else"})
    assert "unknown harness" in ctx.lower()
    assert "session_id: sess-123" not in ctx


# --- Summaries first, then recall, then cache (AC-51) ------------------------


def test_summaries_come_first_then_recall_and_the_cache_is_written(mcp, project, tmp_path):
    ctx, _ = _run(mcp, project, tmp_path)

    (call,) = mcp.tool_calls("list_memories")
    assert call["project"] == "demo" and call["type"] == "session-summary"
    assert call["status"] == "active" and call["limit"] == 3 and call["format"] == "json"
    assert ctx.index("summary 3") < ctx.index("summary 1") < ctx.index("recalled turn one")
    (recall,) = mcp.tool_calls("search_all_sessions")
    assert recall["n"] == 3 and recall["format"] == "json"

    cache = json.loads(_cache(tmp_path).read_text())
    assert [r["memory_id"] for r in cache["records"]] == ["m-3", "m-2", "m-1"]
    assert cache["records"][0]["text"].startswith("summary 3")
    assert "fetched_at" in cache


def test_compaction_recalls_five(mcp, project, tmp_path):
    _run(mcp, project, tmp_path, source="compact")
    (recall,) = mcp.tool_calls("search_all_sessions")
    assert recall["n"] == 5


def test_empty_store_says_so(mcp, project, tmp_path):
    mcp.scripts["list_memories"]["payload"]["results"] = []
    ctx, _ = _run(mcp, project, tmp_path)
    assert "no session summaries yet" in ctx.lower()


# --- Failure fallbacks (AC-52, AC-53, D-15, D-21) ----------------------------


@pytest.mark.parametrize("kind", ["rpc_error", "is_error", "bad_json", "wrong_shape", "http_503"])
def test_tool_failures_fall_back_to_the_cache(mcp, project, tmp_path, kind):
    _run(mcp, project, tmp_path)  # writes the cache
    written = json.loads(_cache(tmp_path).read_text())["fetched_at"]
    if kind == "http_503":
        mcp.status_code = 503
    else:
        mcp.scripts["list_memories"]["kind"] = kind

    ctx, _ = _run(mcp, project, tmp_path)

    assert "cache" in ctx.lower() and written in ctx
    assert "summary 3" in ctx


@pytest.mark.parametrize("problem,setup", [
    ("missing", lambda p: None),
    ("unreadable", lambda p: (p.parent.mkdir(parents=True, exist_ok=True), p.write_text("{}"), p.chmod(0))),
    ("damaged", lambda p: (p.parent.mkdir(parents=True, exist_ok=True), p.write_text('{"records": [tru'))),
])
def test_bad_cache_prints_a_warning_and_no_memory(project, tmp_path, problem, setup):
    setup(_cache(tmp_path))
    try:
        ctx, _ = _run(None, project, tmp_path)
    finally:
        if _cache(tmp_path).exists():
            _cache(tmp_path).chmod(0o600)
    assert "warning" in ctx.lower() and problem in ctx.lower()
    assert "summary" not in ctx


def test_older_results_never_replace_a_newer_cache(mcp, project, tmp_path):
    path = _cache(tmp_path)
    path.parent.mkdir(parents=True)
    newer = {"fetched_at": "2999-01-01T00:00:00Z", "records": [_summary(9)]}
    path.write_text(json.dumps(newer))
    _run(mcp, project, tmp_path)
    assert json.loads(path.read_text()) == newer


@pytest.mark.parametrize("harness,limit", [("claude", 9000), ("codex", 7000)])
def test_output_fits_the_harness_budget_and_cuts_summaries_first(mcp, project, tmp_path, harness, limit):
    mcp.scripts["list_memories"]["payload"]["results"] = [_summary(i, size=20000) for i in (3, 2, 1)]
    ctx, _ = _run(mcp, project, tmp_path, harness=harness, session_id="019a-x")
    assert len(ctx.encode("utf-8")) <= limit
    assert "recalled turn one" in ctx
    assert "get_memory m-3" in ctx


def test_one_deadline_covers_the_whole_hook(mcp, project, tmp_path):
    _run(mcp, project, tmp_path)
    mcp.scripts["list_memories"]["delay"] = 30
    mcp.scripts["search_all_sessions"]["delay"] = 30
    ctx, elapsed = _run(mcp, project, tmp_path, timeout=30)
    assert elapsed < 10
    assert "cache" in ctx.lower()


def test_partial_failure_names_what_failed(mcp, project, tmp_path):
    mcp.scripts["search_all_sessions"]["kind"] = "rpc_error"
    ctx, _ = _run(mcp, project, tmp_path)
    assert "summary 3" in ctx
    assert "transcript recall" in ctx.lower() and "failed" in ctx.lower()


@pytest.mark.parametrize("state", ["failed", "missing"])
def test_dump_problems_are_reported(mcp, project, tmp_path, state):
    mcp.scripts["list_memories"]["payload"]["dump"] = {"state": state,
                                                       "scheduled_for": "2026-10-26T07:30:00Z"}
    ctx, _ = _run(mcp, project, tmp_path)
    assert "dump" in ctx.lower() and state in ctx.lower()


# --- Hook source (AC-54) -----------------------------------------------------


def test_hook_is_stdlib_only_and_never_calls_mem0():
    source = HOOK.read_text()
    assert "mem0" not in source.lower()
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= set(sys.stdlib_module_names)
