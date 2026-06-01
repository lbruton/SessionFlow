"""SESF-11: schema drift detection in _ensure_collection."""

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
