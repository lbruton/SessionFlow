"""Memory DSN configuration: the SESSIONFLOW_MEMORY_DSN environment variable only."""

from __future__ import annotations

import pytest

from memory_store import MemoryStoreError, fetch_dsn

DSN = "postgresql://user@db.example/memory"


def test_environment_value_is_used(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN", f"  {DSN}\n")
    assert fetch_dsn() == DSN


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_is_not_configured(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SESSIONFLOW_MEMORY_DSN", raising=False)
    else:
        monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN", value)
    with pytest.raises(MemoryStoreError) as exc:
        fetch_dsn()
    assert exc.value.code == "dsn_unavailable"
    assert "not configured" in str(exc.value) and "SESSIONFLOW_MEMORY_DSN" in str(exc.value)


def test_no_file_is_read(monkeypatch, tmp_path):
    monkeypatch.delenv("SESSIONFLOW_MEMORY_DSN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".sessionflow").mkdir()
    (tmp_path / ".sessionflow" / "memory_dsn").write_text(DSN)
    with pytest.raises(MemoryStoreError):
        fetch_dsn()
