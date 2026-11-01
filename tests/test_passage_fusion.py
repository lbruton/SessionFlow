"""Passage fusion: each channel counts one rank per turn before RRF."""

from __future__ import annotations

from contextlib import contextmanager

import rag_engine
from passages import PASSAGE_CHUNK_TYPE


def _entity(doc_id, chunk_type="turn", session="s", turn_index=7, content="c"):
    return {"doc_id": doc_id, "chunk_type": chunk_type, "session_id": session,
            "turn_index": turn_index, "document": content,
            "timestamp": "2026-10-01T00:00:00"}


def _fts_row(doc_id, chunk_type="turn", session="s", turn_index=7, content="c"):
    row = rag_engine._row_to_result(
        _entity(doc_id, chunk_type, session, turn_index, content),
        rag_engine.default_provider_metadata())
    return row


def _patch_search(monkeypatch, vector_hits, fts_rows):
    """Stub embedding, Milvus and FTS; return the limits each channel was asked for."""
    seen: dict = {}
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, limit=None, search_params=None, **kw):
            seen["vector_limit"] = limit
            seen["search_params"] = search_params
            return [[{"entity": e, "distance": 0.9 - i * 0.01}
                     for i, e in enumerate(vector_hits)]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    def _fts_search(query, n=10, **kw):
        seen["fts_n"] = n
        return list(fts_rows)

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", _fts_search)
    return seen


def test_channel_collapse_keeps_the_best_row_per_turn():
    rows = [
        _fts_row("s::a::p2", PASSAGE_CHUNK_TYPE, content="best passage"),
        _fts_row("s::a::p0", PASSAGE_CHUNK_TYPE),
        _fts_row("s::a"),
        _fts_row("s::b", turn_index=9),
    ]
    out = rag_engine._collapse_channel(rows)
    assert len(out) == 2
    assert out[0]["content"] == "best passage"
    assert out[1]["turn_index"] == 9


def test_turn_found_by_both_channels_through_different_rows_ranks_first(monkeypatch):
    # Vector: passage of turn a at rank 2. FTS: parent row of turn a at rank 2.
    # Turns b (vector rank 1) and c (FTS rank 1) are found by one channel only.
    vector_hits = [
        _entity("s::b", turn_index=9),
        _entity("s::a::p3", PASSAGE_CHUNK_TYPE, content="passage of a"),
    ]
    fts_rows = [
        _fts_row("s::c", turn_index=11),
        _fts_row("s::a", content="whole turn a"),
    ]
    _patch_search(monkeypatch, vector_hits, fts_rows)
    out = rag_engine.search("swappiness", n=3, sort_by="relevance")
    assert [r["turn_index"] for r in out][0] == 7
    assert len([r for r in out if r["turn_index"] == 7]) == 1


def test_sibling_passages_take_one_slot_in_a_channel(monkeypatch):
    vector_hits = [
        _entity("s::a::p0", PASSAGE_CHUNK_TYPE),
        _entity("s::a::p1", PASSAGE_CHUNK_TYPE),
        _entity("s::a::p2", PASSAGE_CHUNK_TYPE),
        _entity("s::b", turn_index=9),
    ]
    _patch_search(monkeypatch, vector_hits, [])
    out = rag_engine.search("anything", n=2, sort_by="relevance")
    assert [r["turn_index"] for r in out] == [7, 9]


def test_display_row_is_the_best_ranked_row_of_the_turn(monkeypatch):
    vector_hits = [
        _entity("s::a::p2", PASSAGE_CHUNK_TYPE, content="passage two"),
        _entity("s::a", content="whole turn"),
    ]
    _patch_search(monkeypatch, vector_hits, [])
    out = rag_engine.search("anything", n=1, sort_by="relevance")
    assert out[0]["doc_id"] == "s::a::p2"
    assert out[0]["content"] == "passage two"
    assert out[0]["passage_index"] == 2


def test_hnsw_ef_covers_the_vector_limit(monkeypatch):
    # HNSW rejects a search whose limit exceeds ef.
    seen = _patch_search(monkeypatch, [], [])
    rag_engine.search("anything", n=30, sort_by="relevance",
                      db_path="http://milvus.test:19530")
    assert seen["search_params"]["params"]["ef"] >= seen["vector_limit"] == 180


def test_hnsw_ef_keeps_its_floor_for_small_limits(monkeypatch):
    seen = _patch_search(monkeypatch, [], [])
    rag_engine.search("anything", n=5, sort_by="relevance",
                      db_path="http://milvus.test:19530")
    assert seen["search_params"]["params"]["ef"] == 128


def test_each_channel_fetches_extra_rows_before_collapse(monkeypatch):
    seen = _patch_search(monkeypatch, [], [])
    rag_engine.search("anything", n=5, sort_by="relevance")
    assert seen["vector_limit"] == 5 * 3 * 2
    assert seen["fts_n"] == 5 * 3 * 2
