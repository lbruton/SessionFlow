"""known_project_roots reads Milvus whenever the FTS sidecar may be incomplete."""

from __future__ import annotations

import importlib

import pytest

rag_engine = importlib.import_module("rag_engine")

FTS_ROOTS = ["/a", "/a/b"]
MILVUS_ROOTS = ["/a", "/a/b", "/c"]


class _FakeFts:
    def __init__(self, roots=None, rows=2, raises=False):
        self.roots, self.rows, self.raises = roots or FTS_ROOTS, rows, raises

    def connection(self, db_path=None):
        return object()

    def close_ephemeral(self, conn):
        pass

    def distinct_values(self, conn, column):
        if self.raises:
            raise RuntimeError("fts down")
        return list(self.roots)

    def count_rows(self, conn, project_root=None):
        return self.rows


@pytest.fixture
def setup(monkeypatch):
    calls = {"scan": 0}

    def _scan(fields, db_path=None, **kw):
        calls["scan"] += 1
        yield [{"project_root": r} for r in MILVUS_ROOTS]

    def _install(fts, sentinel=False, milvus_count=2):
        monkeypatch.setattr(rag_engine, "_fts", fts)
        monkeypatch.setattr(rag_engine, "fts_backfill_required", lambda: sentinel)
        monkeypatch.setattr(rag_engine, "_count_milvus_turns", lambda **kw: milvus_count)
        monkeypatch.setattr(rag_engine, "_query_batches", _scan)
        return calls

    rag_engine._known_roots_cache.clear()
    yield _install
    rag_engine._known_roots_cache.clear()


def test_sentinel_set_reads_milvus(setup):
    calls = setup(_FakeFts(), sentinel=True)
    assert rag_engine.known_project_roots("db") == MILVUS_ROOTS
    assert calls["scan"] == 1


def test_fts_row_count_below_milvus_reads_milvus(setup):
    calls = setup(_FakeFts(rows=2), milvus_count=5)
    assert rag_engine.known_project_roots("db") == MILVUS_ROOTS
    assert calls["scan"] == 1


def test_equal_counts_and_no_sentinel_use_fts_only(setup):
    calls = setup(_FakeFts(rows=2), milvus_count=2)
    assert rag_engine.known_project_roots("db") == FTS_ROOTS
    assert calls["scan"] == 0


def test_fts_read_failure_reads_milvus(setup):
    calls = setup(_FakeFts(raises=True))
    assert rag_engine.known_project_roots("db") == MILVUS_ROOTS
    assert calls["scan"] == 1


def test_milvus_count_failure_keeps_fts_list(setup, monkeypatch):
    calls = setup(_FakeFts())

    def _boom(**kw):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(rag_engine, "_count_milvus_turns", _boom)
    assert rag_engine.known_project_roots("db") == FTS_ROOTS
    assert calls["scan"] == 0
