"""known_project_roots reads Milvus whenever the FTS sidecar may be incomplete."""

from __future__ import annotations

import importlib
import threading
import time

import pytest

rag_engine = importlib.import_module("rag_engine")
fts_hybrid = importlib.import_module("fts_hybrid")

FTS_ROOTS = ["/a", "/a/b"]
MILVUS_ROOTS = ["/a", "/a/b", "/c"]


class _FakeFts:
    def __init__(self, roots=None, rows=2, raises=False):
        self.roots, self.rows, self.raises = roots or FTS_ROOTS, rows, raises

    def connection(self, db_path=None):
        return object()

    def close_ephemeral(self, conn):
        pass

    def close_all(self):
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


class _SlowFts(_FakeFts):
    """FTS fake whose scan waits for ``release`` and returns ``self.roots``."""

    def __init__(self, roots):
        super().__init__(roots=roots)
        self.release = threading.Event()
        self.scans = 0
        self.closed = 0

    def distinct_values(self, conn, column):
        self.scans += 1
        self.release.wait(5)
        return super().distinct_values(conn, column)

    def close_all(self):
        self.closed += 1


def _expire(db="db", roots=("/old",)):
    rag_engine._known_roots_cache[db] = (
        time.monotonic() - rag_engine.KNOWN_ROOTS_TTL_S - 1, list(roots),
    )


def _wait_for_refresh(db="db"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with rag_engine._known_roots_lock:
            done = db not in rag_engine._known_roots_refreshing
        if done:
            return
        time.sleep(0.01)
    raise AssertionError("the roots refresh did not finish")


def test_stale_cache_answers_at_once_and_refreshes_in_the_background(setup):
    fts = _SlowFts(["/new"])
    setup(fts)
    _expire()

    assert rag_engine.known_project_roots("db") == ["/old"]
    assert rag_engine.known_project_roots("db") == ["/old"]

    fts.release.set()
    _wait_for_refresh()
    assert fts.scans == 1
    assert fts.closed == 1
    assert rag_engine.known_project_roots("db") == ["/new"]


def test_fresh_cache_starts_no_refresh(setup):
    fts = _SlowFts(["/new"])
    setup(fts)
    rag_engine._known_roots_cache["db"] = (time.monotonic(), ["/old"])
    assert rag_engine.known_project_roots("db") == ["/old"]
    assert fts.scans == 0


def test_failed_refresh_keeps_the_list_and_retries(setup, monkeypatch):
    fts = _SlowFts(["/new"])
    fts.release.set()
    setup(fts)
    _expire()

    def _boom(*args, **kwargs):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(rag_engine, "_scan_project_roots", _boom)
    assert rag_engine.known_project_roots("db") == ["/old"]
    _wait_for_refresh()
    assert rag_engine._known_roots_cache["db"][1] == ["/old"]

    monkeypatch.setattr(rag_engine, "_scan_project_roots", lambda db_path: ["/new"])
    rag_engine.known_project_roots("db")
    _wait_for_refresh()
    assert rag_engine._known_roots_cache["db"][1] == ["/new"]


def test_refresh_fills_an_empty_cache_without_blocking(setup):
    fts = _SlowFts(["/new"])
    setup(fts)

    rag_engine.refresh_known_project_roots("db")
    assert "db" not in rag_engine._known_roots_cache

    fts.release.set()
    _wait_for_refresh()
    assert rag_engine._known_roots_cache["db"][1] == ["/new"]


def test_empty_cache_waits_for_the_scan_in_flight(setup):
    fts = _SlowFts(["/new"])
    setup(fts)
    rag_engine.refresh_known_project_roots("db")
    seen = []
    caller = threading.Thread(target=lambda: seen.append(rag_engine.known_project_roots("db")))
    caller.start()
    time.sleep(0.05)
    assert seen == []

    fts.release.set()
    caller.join(5)
    assert seen == [["/new"]]
    assert fts.scans == 1


def test_empty_cache_scans_itself_when_the_shared_scan_fails(setup, monkeypatch):
    fts = _SlowFts(["/new"])
    setup(fts)
    results = iter([RuntimeError("scan failed"), ["/retry"]])

    def _scan(db_path):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(rag_engine, "_scan_project_roots", _scan)
    assert rag_engine.known_project_roots("db") == ["/retry"]


def test_unfiltered_count_reads_docsize_not_the_table(tmp_path):
    fts = fts_hybrid.FTSIndex("t_fts", ["project_root"])
    conn = fts.connection(str(tmp_path / "milvus.db"))
    try:
        fts.insert(conn, [{"doc_id": f"d{i}", "content": "x", "project_root": "/a"} for i in range(3)])
        conn.commit()
        conn.execute("DELETE FROM t_fts WHERE doc_id = 'd0'")
        conn.commit()
        statements = []
        conn.set_trace_callback(statements.append)
        assert fts.count_rows(conn) == 2
        conn.set_trace_callback(None)
        assert statements == ["SELECT count(*) FROM t_fts_docsize"]
        assert fts.count_rows(conn, project_root="/a") == 2
    finally:
        conn.close()
