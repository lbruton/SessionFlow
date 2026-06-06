"""SESF-11: schema drift detection in _ensure_collection."""

import re

from unittest.mock import MagicMock

import pytest


def _fields_from(client_mock):
    """Helper: pull the field-name set the mock describe_collection returns."""
    info = client_mock.describe_collection.return_value
    return {f["name"] for f in info["fields"]}


def test_no_drift_when_field_names_match():
    import rag_engine

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": name} for name in expected],
    }

    assert rag_engine.detect_schema_drift(client) == []


def test_expected_schema_includes_issue_ids():
    """SESF-25 Req 2.1: issue_ids is a first-class field of the collection schema."""
    import rag_engine

    expected = {f.name for f in rag_engine._expected_schema_fields()}
    assert "issue_ids" in expected


def test_drift_reports_missing_issue_ids():
    """SESF-25 Req 2.2: a collection lacking issue_ids reports missing:issue_ids."""
    import rag_engine

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    # issue_ids must be in the expected set for this test to be meaningful.
    assert "issue_ids" in expected
    client = MagicMock()
    client.has_collection.return_value = True
    fields = [{"name": name} for name in expected if name != "issue_ids"]
    client.describe_collection.return_value = {"fields": fields}

    drift = rag_engine.detect_schema_drift(client)
    assert drift == ["missing:issue_ids"]


def test_drift_reports_missing_field():
    import rag_engine

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    client = MagicMock()
    client.has_collection.return_value = True
    # Pretend an old collection is missing logical_session_id.
    fields = [{"name": name} for name in expected if name != "logical_session_id"]
    client.describe_collection.return_value = {"fields": fields}

    drift = rag_engine.detect_schema_drift(client)
    assert drift == ["missing:logical_session_id"]


def test_drift_reports_extra_field():
    import rag_engine

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": name} for name in expected] + [{"name": "obsolete_legacy"}],
    }

    drift = rag_engine.detect_schema_drift(client)
    assert drift == ["extra:obsolete_legacy"]


def test_ensure_collection_raises_on_drift(monkeypatch):
    """Default behavior: refuse to start, telling the operator to migrate."""
    import rag_engine

    monkeypatch.delenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", raising=False)

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": name} for name in expected if name != "provider"],
    }

    with pytest.raises(RuntimeError, match="schema is out of date"):
        rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")


def test_ensure_collection_auto_migrates_when_env_set(monkeypatch):
    """SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 opts into drop+recreate."""
    import rag_engine

    monkeypatch.setenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", "1")

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": name} for name in expected if name != "provider"],
    }

    called: dict[str, bool] = {}

    def fake_migrate(c, db_path=""):
        called["migrated"] = True

    monkeypatch.setattr(rag_engine, "migrate_schema", fake_migrate)
    rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")
    assert called.get("migrated") is True


def test_ensure_collection_creates_when_missing(monkeypatch):
    """Brand-new install: no collection yet, should create rather than diff."""
    import rag_engine

    client = MagicMock()
    client.has_collection.return_value = False

    called: dict[str, bool] = {}

    def fake_create(c, db_path=""):
        called["created"] = True

    monkeypatch.setattr(rag_engine, "_create_collection", fake_create)
    rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")
    assert called.get("created") is True


# ---------------------------------------------------------------------------
# SESF-38 AC-3: _ensure_collection re-describes ONCE before gating either
# branch. detect_schema_drift issues a cache-free describe_collection, so a
# stale first describe followed by a clean second describe must NOT raise and
# must NOT migrate. This guards BOTH the default (raise) branch and the
# auto-migrate (drop+recreate) branch against a transient/cached drift read.
# ---------------------------------------------------------------------------


def _stale_then_clean_client():
    """Build a client whose describe_collection drifts on call 1, is clean on call 2.

    detect_schema_drift calls describe_collection exactly once per invocation,
    so the side_effect list maps 1:1 to the re-describe sequence inside
    _ensure_collection.
    """
    import rag_engine

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    drifted = {"fields": [{"name": name} for name in expected if name != "provider"]}
    clean = {"fields": [{"name": name} for name in expected]}

    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.side_effect = [drifted, clean]
    return client


@pytest.mark.parametrize(
    ("auto_migrate", "summary"),
    [
        (None, "default path: no raise"),
        ("1", "auto-migrate path: no destructive drop+recreate"),
    ],
    ids=["default", "auto_migrate"],
)
def test_ensure_collection_redescribes_clean_skips_destructive_path(
    monkeypatch, auto_migrate, summary
):
    """AC-3 {summary}: stale-then-clean re-describe → no raise, no migrate.

    Both env states (SESSIONFLOW_AUTO_MIGRATE_SCHEMA unset vs "1") must treat a
    transient/cached first describe followed by a clean second describe as clean:
    _ensure_collection must not raise and must not call migrate_schema, and the
    cache-free re-describe must actually have happened (>= 2 describes).
    """
    import rag_engine

    if auto_migrate is None:
        monkeypatch.delenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", raising=False)
    else:
        monkeypatch.setenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", auto_migrate)

    client = _stale_then_clean_client()

    migrate_called: dict[str, bool] = {}

    def fake_migrate(c, db_path=""):
        migrate_called["migrated"] = True

    monkeypatch.setattr(rag_engine, "migrate_schema", fake_migrate)

    # Must not raise: the second (cache-free) describe is clean.
    rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")

    # Even under AUTO_MIGRATE, a clean re-check must skip the destructive path.
    assert migrate_called.get("migrated") is not True
    # The re-describe must actually have happened (2 cache-free describes).
    assert client.describe_collection.call_count >= 2


# ---------------------------------------------------------------------------
# SESF-38 AC-4: when drift is PERSISTENT (both describes drift) and AUTO_MIGRATE
# is unset, the RuntimeError must be honest: both recovery paths are flagged
# destructive, and a non-destructive restart is offered FIRST.
# ---------------------------------------------------------------------------


def test_ensure_collection_persistent_drift_message_is_honest(monkeypatch):
    """AC-4: persistent drift message marks BOTH paths destructive, restart first."""
    import rag_engine

    monkeypatch.delenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", raising=False)

    expected = [f.name for f in rag_engine._expected_schema_fields()]
    drifted = {"fields": [{"name": name} for name in expected if name != "provider"]}

    client = MagicMock()
    client.has_collection.return_value = True
    # Both re-describes drift → genuinely persistent drift, must raise.
    client.describe_collection.side_effect = [drifted, drifted]

    with pytest.raises(RuntimeError) as excinfo:
        rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")

    msg = str(excinfo.value)

    # (a) the explicit recovery command is still named.
    assert "cleanup.py migrate-schema" in msg

    # (b) BOTH recovery paths are flagged destructive (all turns lost).
    #     The migrate-schema command and the AUTO_MIGRATE env both wipe data.
    assert "SESSIONFLOW_AUTO_MIGRATE_SCHEMA" in msg
    destructive_markers = re.findall(r"all turns lost|all turns will be lost|destructive", msg, re.IGNORECASE)
    assert len(destructive_markers) >= 2, (
        "expected BOTH cleanup.py migrate-schema and SESSIONFLOW_AUTO_MIGRATE_SCHEMA "
        f"flagged as destructive; message was: {msg!r}"
    )

    # (c) a non-destructive restart is offered, positioned BEFORE any
    #     destructive step (so operators see the safe option first).
    lowered = msg.lower()
    assert "restart" in lowered, f"expected a restart suggestion; message was: {msg!r}"
    assert lowered.index("restart") < lowered.index("cleanup.py migrate-schema"), (
        "restart (non-destructive) must be offered before the destructive "
        f"migrate-schema step; message was: {msg!r}"
    )
