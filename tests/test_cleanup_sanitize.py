"""Tests for the ``cleanup.py sanitize`` CLI adapter (SESF-42 Component 4).

These exercise the thin CLI over ``sanitize.py`` only: the argparse wiring, the
refuse-before-writes confirmation gate, the scope-flag → ``Scope`` mapping, the
dry-run/apply/drop dispatch, and the value-free output contract. The ``sanitize``
module is monkeypatched so no real Milvus, FTS, or embedding work is touched —
matching the conftest stubbing style used elsewhere in the suite.

Every token here is **synthetic and non-functional** (AC-18 / Requirement 5.3),
assembled from string parts so no contiguous secret literal appears in this
source, and no raw value is ever fed into an assertion sink.
"""

from __future__ import annotations

import importlib
import types

import pytest

# Synthetic, non-functional secret value — assembled from parts so no contiguous
# secret literal sits in this source. It is only ever placed in a *stubbed*
# report's fields to prove the CLI never echoes it; the CLI emits counts only.
SYNTHETIC_SECRET = "AKIA" + "IOSFODNN7EXAMPLE"

PROJECT = "/Volumes/DATA/GitHub/SessionFlow"
PROVIDER = "claude_code_cli"
SESSION = "synthetic-session-id"
SINCE = "2026-05-01"


@pytest.fixture
def cleanup_mod():
    """Import the CLI module under test."""
    return importlib.import_module("cleanup")


@pytest.fixture
def stub_sanitize(monkeypatch):
    """Replace the ``sanitize`` module with a recording stub.

    Records every ``scan``/``apply`` call (and the ``Scope`` built for it) so a
    test can assert which path the CLI took and what scope it passed, without any
    real store access. ``scan``/``apply`` return a canned :class:`SanitizeReport`
    whose counts are deliberately the only secret-adjacent surface — the report
    fields never contain the raw value.
    """
    calls: dict[str, list] = {"scan": [], "apply": []}

    class _Scope:
        """Mirror of ``sanitize.Scope`` — records the dimensions it was built with."""

        def __init__(
            self,
            project_root=None,
            provider=None,
            session_id=None,
            since=None,
        ):
            self.project_root = project_root
            self.provider = provider
            self.session_id = session_id
            self.since = since

    class _Report:
        """Minimal stand-in for ``sanitize.SanitizeReport`` (value-free)."""

        def __init__(self, *, mode, status):
            self.mode = mode
            self.counts = {"AWS": 2}
            self.affected_count = 1
            self.processed_count = 1
            self.incomplete_fts = 0
            self.audit_path = "/synthetic/audit/redaction-deadbeef.jsonl"
            self.status = status
            self.rotate_warning = mode != "dry-run"

    def fake_scan(scope):
        calls["scan"].append(scope)
        return _Report(mode="dry-run", status="complete")

    def fake_apply(scope, *, drop, confirmed):
        calls["apply"].append({"scope": scope, "drop": drop, "confirmed": confirmed})
        return _Report(mode="drop" if drop else "redact", status="complete")

    module = types.SimpleNamespace(
        Scope=_Scope,
        SanitizeReport=_Report,
        scan=fake_scan,
        apply=fake_apply,
        calls=calls,
    )
    import sys

    monkeypatch.setitem(sys.modules, "sanitize", module)
    return module


def _args(cleanup_mod, argv):
    """Parse ``argv`` through the real parser and return the namespace."""
    return cleanup_mod.build_parser().parse_args(argv)


# --- Argparse wiring --------------------------------------------------------


def test_parser_registers_sanitize_subcommand(cleanup_mod):
    args = _args(cleanup_mod, ["sanitize"])
    assert args.command == "sanitize"
    # Dry-run is the default posture.
    assert args.apply is False
    assert args.drop is False
    assert args.yes is False


def test_parser_accepts_all_scope_and_action_flags(cleanup_mod):
    args = _args(
        cleanup_mod,
        [
            "sanitize",
            "--apply",
            "--drop",
            "--yes",
            "--project",
            PROJECT,
            "--provider",
            PROVIDER,
            "--session",
            SESSION,
            "--since",
            SINCE,
        ],
    )
    assert args.apply is True
    assert args.drop is True
    assert args.yes is True
    assert args.project == PROJECT
    assert args.provider == PROVIDER
    assert args.session == SESSION
    assert args.since == SINCE


# --- Confirmation gate (Requirement 2.4) ------------------------------------


def test_apply_without_yes_refuses_before_any_apply(cleanup_mod, stub_sanitize, capsys):
    """``--apply`` sans ``--yes`` must refuse with a non-zero code and never call apply."""
    args = _args(cleanup_mod, ["sanitize", "--apply"])
    rc = cleanup_mod.cmd_sanitize(args)

    out = capsys.readouterr()
    assert rc != 0
    # The gate fires before any read or write: apply is NEVER invoked.
    assert stub_sanitize.calls["apply"] == []
    assert stub_sanitize.calls["scan"] == []
    # Clear instruction to re-run with confirmation.
    combined = out.out + out.err
    assert "--yes" in combined
    assert "confirmation" in combined.lower()


def test_apply_drop_without_yes_also_refuses(cleanup_mod, stub_sanitize, capsys):
    args = _args(cleanup_mod, ["sanitize", "--apply", "--drop"])
    rc = cleanup_mod.cmd_sanitize(args)

    assert rc != 0
    assert stub_sanitize.calls["apply"] == []


# --- Dry-run default --------------------------------------------------------


def test_default_runs_scan_not_apply(cleanup_mod, stub_sanitize, capsys):
    args = _args(cleanup_mod, ["sanitize"])
    rc = cleanup_mod.cmd_sanitize(args)

    out = capsys.readouterr().out
    assert rc in (0, None)
    assert len(stub_sanitize.calls["scan"]) == 1
    assert stub_sanitize.calls["apply"] == []
    # Dry-run reports per-rule counts, affected count, and the audit path.
    assert "AWS" in out
    assert "redaction-deadbeef.jsonl" in out


def test_dry_run_flag_is_equivalent_to_default(cleanup_mod, stub_sanitize):
    args = _args(cleanup_mod, ["sanitize", "--dry-run"])
    cleanup_mod.cmd_sanitize(args)
    assert len(stub_sanitize.calls["scan"]) == 1
    assert stub_sanitize.calls["apply"] == []


# --- Apply paths ------------------------------------------------------------


def test_apply_yes_calls_apply_confirmed_redact(cleanup_mod, stub_sanitize, capsys):
    args = _args(cleanup_mod, ["sanitize", "--apply", "--yes"])
    rc = cleanup_mod.cmd_sanitize(args)

    out = capsys.readouterr().out
    assert rc in (0, None)
    assert len(stub_sanitize.calls["apply"]) == 1
    call = stub_sanitize.calls["apply"][0]
    assert call["confirmed"] is True
    assert call["drop"] is False
    # Apply output carries the rotate-the-key warning.
    assert "rotate" in out.lower()


def test_apply_yes_drop_passes_drop_true(cleanup_mod, stub_sanitize):
    args = _args(cleanup_mod, ["sanitize", "--apply", "--yes", "--drop"])
    cleanup_mod.cmd_sanitize(args)

    assert len(stub_sanitize.calls["apply"]) == 1
    call = stub_sanitize.calls["apply"][0]
    assert call["confirmed"] is True
    assert call["drop"] is True


# --- Scope mapping ----------------------------------------------------------


def test_scope_flags_map_into_scope(cleanup_mod, stub_sanitize):
    args = _args(
        cleanup_mod,
        [
            "sanitize",
            "--project",
            PROJECT,
            "--provider",
            PROVIDER,
            "--session",
            SESSION,
            "--since",
            SINCE,
        ],
    )
    cleanup_mod.cmd_sanitize(args)

    scope = stub_sanitize.calls["scan"][0]
    assert scope.project_root == PROJECT
    assert scope.provider == PROVIDER
    assert scope.session_id == SESSION
    assert scope.since == SINCE


def test_apply_scope_flags_map_into_scope(cleanup_mod, stub_sanitize):
    args = _args(
        cleanup_mod,
        ["sanitize", "--apply", "--yes", "--provider", PROVIDER],
    )
    cleanup_mod.cmd_sanitize(args)

    scope = stub_sanitize.calls["apply"][0]["scope"]
    assert scope.provider == PROVIDER
    assert scope.project_root is None


# --- No-leak contract (cross-cutting) ---------------------------------------


def test_output_never_contains_a_secret_value(cleanup_mod, monkeypatch, capsys):
    """Even if a report somehow carried a raw value, the CLI emits counts only.

    The stub report here deliberately seeds the synthetic secret into fields the
    CLI must NOT print (it should print rule names, counts, and the audit path —
    never document text or a raw value).
    """
    import sys

    leaky_report = types.SimpleNamespace(
        mode="dry-run",
        counts={"AWS": 1},
        affected_count=1,
        processed_count=0,
        incomplete_fts=0,
        audit_path="/synthetic/audit/redaction-cafef00d.jsonl",
        status="complete",
        rotate_warning=False,
        # Fields the CLI must never echo:
        secret_value=SYNTHETIC_SECRET,
        document=f"here is a key {SYNTHETIC_SECRET} in text",
    )

    class _Scope:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    module = types.SimpleNamespace(
        Scope=_Scope,
        scan=lambda scope: leaky_report,
        apply=lambda scope, **kw: leaky_report,
    )
    monkeypatch.setitem(sys.modules, "sanitize", module)

    args = _args(cleanup_mod, ["sanitize"])
    cleanup_mod.cmd_sanitize(args)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SYNTHETIC_SECRET not in combined
