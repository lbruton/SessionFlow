"""SESF-100: the optional reranker reorders the top hybrid candidates of a query search."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import rag_engine
import reranker
import tools


def _rows(count: int) -> list[dict]:
    return [{"doc_id": f"d{i}", "content": f"text {i}", "distance": None,
             "session_id": "s", "turn_index": i, "chunk_type": "turn",
             "timestamp": f"2026-10-{i + 1:02d}T00:00:00Z"} for i in range(count)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("SESSIONFLOW_RERANKER", "SESSIONFLOW_RERANK_TOP",
                 "SESSIONFLOW_RERANK_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    reranker.reset()
    yield
    reranker.reset()


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


class _Scorer:
    def __init__(self, scores=None, error=None):
        self.calls = []
        self._scores = scores
        self._error = error

    def __call__(self, query, docs):
        self.calls.append((query, list(docs)))
        if self._error:
            raise self._error
        if self._scores is not None:
            return list(self._scores)[:len(docs)]
        # Default: the last document scores highest.
        return [float(i) for i in range(len(docs))]


def _ids(rows):
    return [r["doc_id"] for r in rows]


def _search(n=5, sort_by="relevance", query="what broke"):
    return rag_engine.search(query, n=n, sort_by=sort_by)


# --- settings -------------------------------------------------------------------


def test_settings_defaults():
    assert reranker.settings() == {"reranker": "off", "rerank_top": 10,
                                   "rerank_max_tokens": 512}


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANKER", "Jina")
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "20")
    monkeypatch.setenv("SESSIONFLOW_RERANK_MAX_TOKENS", "256")
    assert reranker.settings() == {"reranker": "jina", "rerank_top": 20,
                                   "rerank_max_tokens": 256}


def test_unknown_reranker_means_off(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANKER", "bge")
    assert reranker.settings()["reranker"] == "off"


def test_ranking_settings_report_the_reranker(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANKER", "qwen3")
    settings = rag_engine.ranking_settings()
    assert settings["reranker"] == "qwen3"
    assert settings["rerank_top"] == 10
    assert settings["rerank_max_tokens"] == 512


def test_status_is_off_until_loaded():
    assert reranker.status() == "off"
    assert not reranker.is_ready()


# --- search ---------------------------------------------------------------------


def test_off_leaves_results_unchanged(stub_search):
    assert _ids(_search()) == ["d0", "d1", "d2", "d3", "d4"]


def test_jina_sorts_only_the_top_rows_by_score(stub_search, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "3")
    scorer = _Scorer()
    reranker.install("jina", scorer)
    rows = _search(n=5)
    assert _ids(rows) == ["d2", "d1", "d0", "d3", "d4"]
    assert [r.get("rerank_rank") for r in rows] == [1, 2, 3, None, None]
    assert scorer.calls == [("what broke", ["text 0", "text 1", "text 2"])]


def test_qwen3_fuses_hybrid_and_reranker_ranks(stub_search, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "4")
    # Reranker ranks: d3 first, then d1, d2, d0.
    reranker.install("qwen3", _Scorer(scores=[0.1, 0.8, 0.5, 0.9]))
    rows = _search(n=4)
    k = reranker.QWEN3_RRF_K
    hybrid = {"d0": 1, "d1": 2, "d2": 3, "d3": 4}
    rerank = {"d3": 1, "d1": 2, "d2": 3, "d0": 4}
    fused = sorted(hybrid, key=lambda d: (-(1 / (k + hybrid[d]) + 1 / (k + rerank[d])),
                                          hybrid[d]))
    assert _ids(rows) == fused
    assert {r["doc_id"]: r["rerank_rank"] for r in rows} == rerank


def test_retrieval_covers_the_rerank_count_when_n_is_small(stub_search, monkeypatch):
    seen = {}

    def _fts(*a, n=10, **kw):
        seen["fts_n"] = n
        return _rows(15)

    monkeypatch.setattr(rag_engine._fts, "search", _fts)
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "10")
    scorer = _Scorer()
    reranker.install("jina", scorer)
    rows = _search(n=1)
    assert seen["fts_n"] >= 10
    assert len(scorer.calls[0][1]) == 10
    assert _ids(rows) == ["d9"]


def test_retrieval_pool_is_unchanged_when_reranking_is_off(stub_search, monkeypatch):
    seen = {}

    def _fts(*a, n=10, **kw):
        seen["fts_n"] = n
        return _rows(15)

    monkeypatch.setattr(rag_engine._fts, "search", _fts)
    _search(n=1)
    assert seen["fts_n"] == 3


def test_reranks_more_candidates_than_n(stub_search, monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RERANK_TOP", "4")
    scorer = _Scorer()
    reranker.install("jina", scorer)
    rows = _search(n=2)
    assert len(scorer.calls[0][1]) == 4
    assert _ids(rows) == ["d3", "d2"]


def test_recency_sort_never_reranks(stub_search):
    scorer = _Scorer()
    reranker.install("jina", scorer)
    rows = _search(n=3, sort_by="recency")
    assert scorer.calls == []
    assert all("rerank_rank" not in r for r in rows)


def test_hybrid_sort_reranks(stub_search):
    scorer = _Scorer()
    reranker.install("jina", scorer)
    _search(n=3, sort_by="hybrid")
    assert len(scorer.calls) == 1


def test_query_less_listing_never_reranks(monkeypatch):
    scorer = _Scorer()
    reranker.install("jina", scorer)
    monkeypatch.setattr(rag_engine, "_recent_listing", lambda n, **kw: _rows(n))
    rag_engine.search(None, n=3)
    rag_engine.search("   ", n=3)
    assert scorer.calls == []


def test_scorer_error_returns_the_unreranked_order(stub_search):
    reranker.install("jina", _Scorer(error=RuntimeError("metal")))
    rows = _search()
    assert _ids(rows) == ["d0", "d1", "d2", "d3", "d4"]
    assert all("rerank_rank" not in r for r in rows)


def test_reranker_not_ready_is_skipped(stub_search, monkeypatch):
    scorer = _Scorer()
    reranker.install("jina", scorer)
    monkeypatch.setitem(reranker._state, "status", "loading")
    assert _ids(_search()) == ["d0", "d1", "d2", "d3", "d4"]
    assert scorer.calls == []


def test_failed_load_reports_the_reason(stub_search, monkeypatch):
    def _broken(max_tokens):
        raise OSError("no snapshot")

    monkeypatch.setitem(reranker._BUILDERS, "jina", _broken)
    reranker.load("jina")
    assert reranker.status() == "failed: OSError: no snapshot"
    assert not reranker.is_ready()
    assert _ids(_search()) == ["d0", "d1", "d2", "d3", "d4"]


def test_load_installs_the_built_scorer(monkeypatch):
    scorer = _Scorer()
    monkeypatch.setitem(reranker._BUILDERS, "qwen3", lambda max_tokens: scorer)
    reranker.load("qwen3")
    assert reranker.status() == "ready"
    assert reranker.is_ready()


def test_load_off_does_nothing():
    reranker.load("off")
    assert reranker.status() == "off"


def test_scoring_stamps_the_embed_throttle(stub_search, monkeypatch):
    stamps = []
    throttle = reranker.embedding_control.get_embed_throttle()
    monkeypatch.setattr(throttle, "done", lambda: stamps.append(1))
    reranker.install("jina", _Scorer())
    _search()
    assert stamps == [1]


# --- output ---------------------------------------------------------------------


def test_signals_line_names_the_rerank_rank():
    row = {"distance": 0.25, "keyword_rank": 3, "rerank_rank": 2}
    assert tools._format_signals(row) == "Semantic: 0.75 · Keyword: #3 · Rerank: #2"


# --- real model -----------------------------------------------------------------


def _cached(repo, revision):
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_cached(reranker.QWEN3_REPO, reranker.QWEN3_REVISION) is None,
                    reason="Qwen3-Reranker-0.6B not cached locally")
def test_qwen3_last_position_head_matches_full_logits():
    import mlx.core as mx

    scorer = reranker._build_qwen3(512)
    docs = ["The HNSW ef must cover the search limit.", "Lunch was soup.",
            "Raise ef to max(128, fetch_n) so Milvus accepts the limit."]
    query = "why did Milvus reject the search limit"
    fast = scorer(query, docs)

    model, tok = scorer.model, scorer.tokenizer
    full = []
    for seq in scorer.sequences(query, docs):
        logits = model(mx.array([seq]))[0, -1, [scorer.no_id, scorer.yes_id]]
        full.append(mx.softmax(logits.astype(mx.float32), axis=-1)[1].item())
    assert fast == pytest.approx(full, abs=1e-3)
    assert fast[1] < min(fast[0], fast[2])


@pytest.mark.skipif(_cached(reranker.JINA_REPO, reranker.JINA_REVISION) is None,
                    reason="jina-reranker-v3.5-mlx not cached locally")
def test_jina_loads_from_its_snapshot_and_ranks_the_answer_first():
    import sys

    before = set(sys.modules)
    scorer = reranker._build_jina(512)
    assert "modeling" not in set(sys.modules) - before
    docs = ["Lunch was soup.", "Raise ef to max(128, fetch_n) so Milvus accepts the limit."]
    scores = scorer("why did Milvus reject the search limit", docs)
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_missing_snapshot_names_the_download_command(monkeypatch):
    import huggingface_hub

    calls = []

    def _offline(repo, revision=None, local_files_only=False, **kw):
        calls.append(local_files_only)
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _offline)
    reranker.load("qwen3")
    assert reranker.status().startswith("failed: ")
    assert "./download-model.sh qwen3" in reranker.status()
    assert calls == [True]
