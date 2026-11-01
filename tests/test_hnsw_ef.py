"""HNSW ef covers the vector search limit: Milvus rejects a limit above ef."""

from __future__ import annotations

from contextlib import contextmanager

import rag_engine


def _patch_search(monkeypatch) -> dict:
    seen: dict = {}
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, limit=None, search_params=None, **kw):
            seen["limit"] = limit
            seen["search_params"] = search_params
            return [[]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [])
    return seen


def test_ef_covers_a_large_limit(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("anything", n=50, sort_by="relevance",
                      db_path="http://milvus.test:19530")
    assert seen["search_params"]["params"]["ef"] >= seen["limit"] == 150


def test_ef_keeps_its_floor_for_small_limits(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("anything", n=5, sort_by="relevance",
                      db_path="http://milvus.test:19530")
    assert seen["search_params"]["params"]["ef"] == 128
