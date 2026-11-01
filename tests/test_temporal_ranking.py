"""SESF-101: queries that ask for the latest state rank with a stronger recency weight."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import rag_engine
import reranker
import tools


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("SESSIONFLOW_RECENCY_WEIGHT", "SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT",
                 "SESSIONFLOW_RERANK_TOP"):
        monkeypatch.delenv(name, raising=False)
    reranker.reset()
    yield
    reranker.reset()


# --- cue detection --------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "latest change to the spot cards",
    "most recent goldback chip contrast problem",
    "Most  Recent dry run",
    "what is left of the mem0 migration now",
    "newest reranker numbers",
    "what did we change recently",
    "recent eval drift",
    "current version numbering of SessionFlow releases",
    "what is currently blocking the deploy",
    "what did I fix today",
    "the Plane outage yesterday",
    "what happened this week",
    "what did we decide last time about the hook",
    "LATEST?",
])
def test_temporal_cues_match(query):
    assert rag_engine.is_temporal_query(query)


@pytest.mark.parametrize("query", [
    "nowhere in the logs",
    "currency conversion for the spot price",
    "recentness score",
    "snow day schedule",
    "last week of the sprint",
    "how does the reranker load",
    "SESF-57 padding mask",
    "",
])
def test_other_queries_do_not_match(query):
    assert not rag_engine.is_temporal_query(query)


# --- settings -------------------------------------------------------------------


def test_temporal_weight_default():
    settings = rag_engine.ranking_settings()
    assert settings["temporal_recency_weight"] == rag_engine.TEMPORAL_RECENCY_WEIGHT_DEFAULT


def test_temporal_weight_from_env(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT", "0.4")
    assert rag_engine.ranking_settings()["temporal_recency_weight"] == 0.4


@pytest.mark.parametrize("value", ["2", "-1", "nan", "x"])
def test_out_of_range_temporal_weight_falls_back(monkeypatch, value):
    monkeypatch.setenv("SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT", value)
    assert (rag_engine.ranking_settings()["temporal_recency_weight"]
            == rag_engine.TEMPORAL_RECENCY_WEIGHT_DEFAULT)


# --- _rank_results override -----------------------------------------------------


def _pair():
    return [{"_rrf_score": 1.0, "timestamp": "2020-01-01T00:00:00Z", "doc_id": "old"},
            {"_rrf_score": 0.5, "timestamp": "2026-10-30T00:00:00Z", "doc_id": "new"}]


def test_rank_results_weight_override():
    from datetime import datetime, timezone

    now = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    default = rag_engine._rank_results(_pair(), "hybrid", 2, now=now)
    override = rag_engine._rank_results(_pair(), "hybrid", 2, now=now, recency_weight=1.0)
    assert [r["doc_id"] for r in default] == ["old", "new"]
    assert [r["doc_id"] for r in override] == ["new", "old"]


# --- search ---------------------------------------------------------------------


def _rows(count: int) -> list[dict]:
    return [{"doc_id": f"d{i}", "content": f"text {i}", "distance": None,
             "session_id": "s", "turn_index": i, "chunk_type": "turn",
             "timestamp": f"2026-10-{i + 1:02d}T00:00:00Z"} for i in range(count)]


@pytest.fixture
def stub_search(monkeypatch):
    """Serve 15 keyword rows in order d0..d14 and no vector rows."""
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, **kw):
            return [[]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: _rows(15))


@pytest.fixture
def rank_spy(stub_search, monkeypatch):
    calls = []
    orig = rag_engine._rank_results

    def _spy(results, sort_by, n, now=None, **kw):
        calls.append((sort_by, kw.get("recency_weight")))
        return orig(results, sort_by, n, now=now, **kw)

    monkeypatch.setattr(rag_engine, "_rank_results", _spy)
    return calls


def test_temporal_hybrid_search_passes_the_temporal_weight(rank_spy, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT", "0.6")
    rows = rag_engine.search("latest reranker change", n=3, sort_by="hybrid")
    assert rank_spy == [("hybrid", 0.6)]
    assert all(r.get("temporal") is True for r in rows)


def test_plain_hybrid_search_keeps_the_default_weight(rank_spy):
    rows = rag_engine.search("reranker change", n=3, sort_by="hybrid")
    assert rank_spy == [("hybrid", None)]
    assert all("temporal" not in r for r in rows)


@pytest.mark.parametrize("sort_by", ["relevance", "recency"])
def test_other_sorts_ignore_temporal_cues(rank_spy, sort_by):
    rows = rag_engine.search("latest reranker change", n=3, sort_by=sort_by)
    assert rank_spy == [(sort_by, None)]
    assert all("temporal" not in r for r in rows)


class _Scorer:
    def __init__(self, scores):
        self._scores = scores

    def __call__(self, query, docs):
        return list(self._scores)[:len(docs)]


def _fused(hybrid_ids, rerank_order):
    k = reranker.QWEN3_RRF_K
    hybrid = {d: i + 1 for i, d in enumerate(hybrid_ids)}
    rerank = {d: i + 1 for i, d in enumerate(rerank_order)}
    return sorted(hybrid, key=lambda d: (-(1 / (k + hybrid[d]) + 1 / (k + rerank[d])),
                                         hybrid[d]))


@pytest.mark.parametrize("sort_by", ["relevance", "hybrid"])
def test_jina_fuses_with_the_hybrid_rank_for_temporal_queries(stub_search, monkeypatch,
                                                              sort_by):
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "4")
    hybrid_ids = [r["doc_id"] for r in rag_engine.search("latest change", n=4,
                                                         sort_by=sort_by)]
    # The reranker reverses the incoming order.
    reranker.install("jina", _Scorer([0.1, 0.5, 0.8, 0.9]))
    rows = rag_engine.search("latest change", n=4, sort_by=sort_by)
    reversed_ids = list(reversed(hybrid_ids))
    if sort_by == "hybrid":
        assert [r["doc_id"] for r in rows] == _fused(hybrid_ids, reversed_ids)
    else:
        # Relevance sort is not temporal: jina sorts by its score.
        assert [r["doc_id"] for r in rows] == reversed_ids


def test_rerank_fuse_flag():
    reranker.install("jina", _Scorer([0.1, 0.9, 0.5]))
    rows = [{"doc_id": f"d{i}", "content": ""} for i in range(3)]
    lock = _NoLock()
    by_score = reranker.rerank("q", [dict(r) for r in rows], lock=lock)
    fused = reranker.rerank("q", [dict(r) for r in rows], lock=lock, fuse=True)
    assert [r["doc_id"] for r in by_score] == ["d1", "d2", "d0"]
    assert [r["doc_id"] for r in fused] == _fused(["d0", "d1", "d2"], ["d1", "d2", "d0"])


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- signals --------------------------------------------------------------------


def test_signals_line_names_a_temporal_query():
    row = {"distance": 0.25, "keyword_rank": 3, "rerank_rank": 2, "temporal": True}
    assert tools._format_signals(row) == ("Semantic: 0.75 · Keyword: #3 · Rerank: #2 · "
                                          "Temporal: yes")
