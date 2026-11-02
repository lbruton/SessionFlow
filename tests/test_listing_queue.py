"""Query-less listings stay off the MLX thread and scan only what ordering needs.

The session-start hook has one short deadline, and a listing uses no MLX. Waiting
behind searches and embeds on the single ``mlx-embed`` thread spends that deadline.
Streaming ``document`` for every matching row costs most of the listing's own time.
"""

import asyncio
import importlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tests.test_recent_listing import DB, _FakeIterator, _qrow

rag_engine = importlib.import_module("rag_engine")


@pytest.fixture
def mlx_executor(monkeypatch):
    """Install a real single-worker mlx-embed executor, as init_server_mode does."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-embed")
    monkeypatch.setattr(rag_engine, "_embed_executor", executor)
    yield executor
    executor.shutdown(wait=True)


def _search_on_busy_mlx(monkeypatch, mlx_executor, query):
    """Run ``search_async`` with the MLX thread blocked and the semaphore held."""
    captured = {}

    def fake_search(q, n=5, **kwargs):
        captured.update(thread=threading.current_thread().name, n=n, kwargs=kwargs)
        return []

    monkeypatch.setattr(rag_engine, "search", fake_search)
    release = threading.Event()
    mlx_executor.submit(release.wait, 10)

    async def run():
        semaphore = asyncio.Semaphore(1)
        monkeypatch.setattr(rag_engine, "_embed_semaphore", semaphore)
        async with semaphore:
            await asyncio.wait_for(
                rag_engine.search_async(query, 3, provider="codex", db_path=DB), 2,
            )

    try:
        asyncio.run(run())
    finally:
        release.set()
    return captured


@pytest.mark.parametrize("query", ["", "   "])
def test_listing_returns_while_the_mlx_thread_is_busy(monkeypatch, mlx_executor, query):
    captured = _search_on_busy_mlx(monkeypatch, mlx_executor, query)
    assert not captured["thread"].startswith("mlx-embed")
    assert captured["n"] == 3
    assert captured["kwargs"]["provider"] == "codex"
    assert captured["kwargs"]["db_path"] == DB


def test_query_search_still_waits_for_the_mlx_thread(monkeypatch, mlx_executor):
    with pytest.raises(asyncio.TimeoutError):
        _search_on_busy_mlx(monkeypatch, mlx_executor, "real query")


class _ProjectingClient:
    """Fake Milvus client that honors ``output_fields`` and a ``doc_id in`` filter."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def has_collection(self, name):
        return True

    def query_iterator(self, *args, **kwargs):
        self.calls.append(kwargs)
        rows = self._rows
        match = re.search(r"doc_id in (\[.*\])", kwargs.get("filter") or "")
        if match:
            wanted = set(json.loads(match.group(1)))
            rows = [r for r in rows if r["doc_id"] in wanted]
        fields = kwargs["output_fields"]
        return _FakeIterator([{k: r[k] for k in fields if k in r} for r in rows],
                             kwargs.get("batch_size", 1000))


def _patch_client(monkeypatch, rows):
    client = _ProjectingClient(rows)

    @contextmanager
    def _fake_client(db_path=None):
        yield client

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    return client


def test_scan_reads_only_doc_id_and_timestamp(monkeypatch):
    client = _patch_client(monkeypatch, [_qrow("a", 1), _qrow("b", 2)])
    rag_engine._recent_listing(5, db_path=DB)
    assert set(client.calls[0]["output_fields"]) == {"doc_id", "timestamp"}


def test_full_rows_are_fetched_for_the_newest_only(monkeypatch):
    rows = [_qrow(f"d{i}", days_ago=i) for i in range(6)]
    client = _patch_client(monkeypatch, rows)
    out = rag_engine._recent_listing(2, provider="codex", db_path=DB)

    assert [r["doc_id"] for r in out] == ["d0", "d1"]
    assert out[0]["content"] == "content d0"
    fetch = client.calls[1]
    assert "document" in fetch["output_fields"]
    assert 'provider == "codex"' in fetch["filter"]
    assert set(json.loads(re.search(r"doc_id in (\[.*\])", fetch["filter"]).group(1))) == {"d0", "d1"}


def test_fetch_splits_a_long_doc_id_list(monkeypatch):
    rows = [_qrow(f"d{i:03d}", days_ago=i) for i in range(450)]
    client = _patch_client(monkeypatch, rows)
    out = rag_engine._recent_listing(450, db_path=DB)

    assert [r["doc_id"] for r in out] == [f"d{i:03d}" for i in range(450)]
    assert len(client.calls) == 1 + 3


def test_empty_scan_skips_the_fetch(monkeypatch):
    client = _patch_client(monkeypatch, [])
    assert rag_engine._recent_listing(5, db_path=DB) == []
    assert len(client.calls) == 1


def test_row_deleted_between_scan_and_fetch_is_dropped(monkeypatch):
    rows = [_qrow("a", 1), _qrow("b", 2)]
    client = _patch_client(monkeypatch, rows)
    original = client.query_iterator

    def delete_after_scan(*args, **kwargs):
        iterator = original(*args, **kwargs)
        if "doc_id in" not in (kwargs.get("filter") or ""):
            client._rows = rows[1:]
        return iterator

    client.query_iterator = delete_after_scan
    out = rag_engine._recent_listing(5, db_path=DB)
    assert [r["doc_id"] for r in out] == ["b"]
