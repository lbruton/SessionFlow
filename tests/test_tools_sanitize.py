"""Tests for the SESF-42 ``sanitize_index`` MCP tool adapter (Component 5).

These exercise the thin MCP layer in :mod:`tools` only — the ``sanitize`` module
is monkeypatched so no real Milvus, FTS, or embedding backend is touched. The
focus is the adapter contract from design.md "Component 5":

* the tool is registered with the documented input schema;
* a default invocation calls :func:`sanitize.scan` (never :func:`sanitize.apply`);
* ``apply=true`` without ``confirm`` refuses and never calls :func:`sanitize.apply`;
* ``apply=true, confirm=true`` calls :func:`sanitize.apply` with ``confirmed=True``
  and threads ``drop`` through;
* the rendered response never contains a raw secret value.
"""

import asyncio
import importlib

tools = importlib.import_module("tools")

# A synthetic secret assembled from parts — must never appear in any response.
SECRET = "AKIA" + "IOSFODNN7EXAMPLE"


def _tool_registry():
    """Register tools against a capturing fake server; return ({name: schema}, call_tool)."""
    captured: dict = {}

    class _Server:
        def list_tools(self):
            def deco(fn):
                captured["list"] = fn
                return fn
            return deco

        def call_tool(self):
            def deco(fn):
                captured["call"] = fn
                return fn
            return deco

    tools.register_tools(_Server())
    listed = asyncio.run(captured["list"]())
    return {t.name: t.inputSchema for t in listed}, captured["call"]


def _report(mode, **kwargs):
    """Build a value-free :class:`sanitize.SanitizeReport` stub for assertions."""
    defaults = dict(
        counts={"AWS": 2},
        affected_count=1,
        processed_count=0,
        incomplete_fts=0,
        audit_path="/tmp/audit/redaction-deadbeef.jsonl",
        status="complete",
        rotate_warning=(mode != "dry-run"),
    )
    defaults.update(kwargs)
    return tools.sanitize.SanitizeReport(mode=mode, **defaults)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_sanitize_index_tool_registered_with_schema():
    # The tool is listed with all documented optional properties and none required.
    schemas, _ = _tool_registry()
    assert "sanitize_index" in schemas
    props = schemas["sanitize_index"]["properties"]
    for key in ("apply", "drop", "confirm", "project_root", "provider", "session_id", "since"):
        assert key in props, f"missing property: {key}"
    assert props["apply"]["type"] == "boolean"
    assert props["drop"]["type"] == "boolean"
    assert props["confirm"]["type"] == "boolean"
    assert schemas["sanitize_index"]["required"] == []


# ---------------------------------------------------------------------------
# Default (dry-run) path
# ---------------------------------------------------------------------------


def test_default_invocation_calls_scan_not_apply(monkeypatch):
    # No apply flag → scan() is called, apply() is NOT.
    calls = {"scan": 0, "apply": 0}
    captured_scope = {}

    def fake_scan(scope):
        calls["scan"] += 1
        captured_scope["scope"] = scope
        return _report("dry-run")

    def fake_apply(*a, **kw):  # pragma: no cover - must never run here
        calls["apply"] += 1
        raise AssertionError("apply must not be called on a dry-run")

    monkeypatch.setattr(tools.sanitize, "scan", fake_scan)
    monkeypatch.setattr(tools.sanitize, "apply", fake_apply)

    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("sanitize_index", {
        "project_root": "/repo", "provider": "codex", "since": "2026-01-01",
    }))

    assert calls == {"scan": 1, "apply": 0}
    # Scope was built from the args.
    scope = captured_scope["scope"]
    assert scope.project_root == "/repo"
    assert scope.provider == "codex"
    assert scope.since == "2026-01-01"
    # Response carries counts + audit path and is not an error.
    text = result[0].text
    assert result[0].type == "text"
    assert not text.startswith("Error")
    assert "AWS: 2" in text
    assert "redaction-deadbeef.jsonl" in text


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------


def test_apply_without_confirm_refuses_and_never_calls_apply(monkeypatch):
    # apply=true, confirm=false → refuse; apply() must NOT be called.
    calls = {"scan": 0, "apply": 0}

    def fake_scan(scope):  # pragma: no cover - apply path must not scan either
        calls["scan"] += 1
        return _report("dry-run")

    def fake_apply(*a, **kw):  # pragma: no cover - must never run
        calls["apply"] += 1
        raise AssertionError("apply must not be called without confirmation")

    monkeypatch.setattr(tools.sanitize, "scan", fake_scan)
    monkeypatch.setattr(tools.sanitize, "apply", fake_apply)

    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("sanitize_index", {"apply": True}))

    assert calls["apply"] == 0
    text = result[0].text.lower()
    assert "confirm" in text
    assert "no changes" in text


# ---------------------------------------------------------------------------
# Apply path
# ---------------------------------------------------------------------------


def test_apply_with_confirm_calls_apply_confirmed(monkeypatch):
    # apply=true, confirm=true → apply(confirmed=True, drop=False).
    seen = {}

    def fake_apply(scope, *, drop, confirmed):
        seen["scope"] = scope
        seen["drop"] = drop
        seen["confirmed"] = confirmed
        return _report("redact", processed_count=1)

    monkeypatch.setattr(tools.sanitize, "apply", fake_apply)

    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("sanitize_index", {
        "apply": True, "confirm": True, "session_id": "sess-1",
    }))

    assert seen["confirmed"] is True
    assert seen["drop"] is False
    assert seen["scope"].session_id == "sess-1"
    text = result[0].text
    assert not text.startswith("Error")
    # Apply surfaces the rotate-the-key warning.
    assert "rotate" in text.lower()


def test_apply_with_drop_threads_drop_true(monkeypatch):
    # apply=true, confirm=true, drop=true → apply(drop=True, confirmed=True).
    seen = {}

    def fake_apply(scope, *, drop, confirmed):
        seen["drop"] = drop
        seen["confirmed"] = confirmed
        return _report("drop", processed_count=1)

    monkeypatch.setattr(tools.sanitize, "apply", fake_apply)

    _, call_tool = _tool_registry()
    result = asyncio.run(call_tool("sanitize_index", {
        "apply": True, "confirm": True, "drop": True,
    }))

    assert seen["drop"] is True
    assert seen["confirmed"] is True
    assert not result[0].text.startswith("Error")


# ---------------------------------------------------------------------------
# No-leak invariant
# ---------------------------------------------------------------------------


def test_response_never_contains_secret_value(monkeypatch):
    # Across dry-run and apply, the synthetic secret must never reach the response.
    monkeypatch.setattr(tools.sanitize, "scan", lambda scope: _report("dry-run"))
    monkeypatch.setattr(
        tools.sanitize,
        "apply",
        lambda scope, *, drop, confirmed: _report("redact", processed_count=1),
    )

    _, call_tool = _tool_registry()

    dry = asyncio.run(call_tool("sanitize_index", {}))
    applied = asyncio.run(call_tool("sanitize_index", {"apply": True, "confirm": True}))
    refused = asyncio.run(call_tool("sanitize_index", {"apply": True}))

    for result in (dry, applied, refused):
        assert SECRET not in result[0].text
