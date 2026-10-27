"""Memory DSN configuration: the environment or a private file, nothing else."""

from __future__ import annotations

import os

import pytest

import memory_store
from memory_store import MemoryStoreError, fetch_dsn

DSN = "postgresql://user@db.example/memory"


def test_environment_wins(monkeypatch, tmp_path):
    path = tmp_path / "dsn"
    path.write_text("postgresql://other@db.example/x")
    os.chmod(path, 0o600)
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN_FILE", str(path))
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN", DSN)
    assert fetch_dsn() == DSN


def test_private_file_is_read(monkeypatch, tmp_path):
    path = tmp_path / "dsn"
    path.write_text(DSN + "\n")
    os.chmod(path, 0o600)
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN_FILE", str(path))
    assert fetch_dsn() == DSN


def test_readable_file_is_refused_without_its_value(monkeypatch, tmp_path):
    path = tmp_path / "dsn"
    path.write_text(DSN)
    os.chmod(path, 0o644)
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN_FILE", str(path))
    with pytest.raises(MemoryStoreError) as exc:
        fetch_dsn()
    assert exc.value.code == "dsn_unavailable" and "chmod 600" in str(exc.value)
    assert DSN not in str(exc.value)


def test_unconfigured_names_both_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN_FILE", str(tmp_path / "missing"))
    with pytest.raises(MemoryStoreError) as exc:
        fetch_dsn()
    assert "not configured" in str(exc.value)
    assert "SESSIONFLOW_MEMORY_DSN" in str(exc.value) and str(tmp_path / "missing") in str(exc.value)


def test_default_file_lives_under_sessionflow():
    assert memory_store.DSN_FILE_DEFAULT.parent.name == ".sessionflow"
    assert memory_store.DSN_FILE_DEFAULT.name == "memory_dsn"
