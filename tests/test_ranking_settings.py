"""The hybrid ranking settings are read from the environment and reported by /health."""

import pytest

import rag_engine


_RERANK_ENV = ("SESSIONFLOW_RERANKER", "SESSIONFLOW_RERANK_TOP", "SESSIONFLOW_RERANK_MAX_TOKENS")
_RERANK_DEFAULTS = {"reranker": "qwen3", "rerank_top": 10, "rerank_max_tokens": 512}


def test_defaults(monkeypatch):
    for name in ("SESSIONFLOW_RECENCY_WEIGHT", "SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT",
                 "SESSIONFLOW_RECENCY_DECAY_DAYS", *_RERANK_ENV):
        monkeypatch.delenv(name, raising=False)
    assert rag_engine.ranking_settings() == {
        "recency_weight": rag_engine.RECENCY_WEIGHT_DEFAULT,
        "temporal_recency_weight": rag_engine.TEMPORAL_RECENCY_WEIGHT_DEFAULT,
        "recency_decay_days": rag_engine.RECENCY_DECAY_DAYS_DEFAULT,
        **_RERANK_DEFAULTS,
    }


def test_env_values(monkeypatch):
    for name in _RERANK_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "0.15")
    monkeypatch.setenv("SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT", "0.4")
    monkeypatch.setenv("SESSIONFLOW_RECENCY_DECAY_DAYS", "14")
    assert rag_engine.ranking_settings() == {"recency_weight": 0.15,
                                             "temporal_recency_weight": 0.4,
                                             "recency_decay_days": 14, **_RERANK_DEFAULTS}


@pytest.mark.parametrize("value", ["2", "-1", "nan", "x"])
def test_out_of_range_weight_falls_back(monkeypatch, value):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", value)
    assert rag_engine.ranking_settings()["recency_weight"] == rag_engine.RECENCY_WEIGHT_DEFAULT


@pytest.mark.parametrize("weight,order", [("1", ["new", "old"]), (None, ["old", "new"])])
def test_hybrid_rank_uses_the_reported_weight(monkeypatch, weight, order):
    from datetime import datetime, timezone

    if weight is None:
        monkeypatch.delenv("SESSIONFLOW_RECENCY_WEIGHT", raising=False)
    else:
        monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", weight)
    rows = [{"_rrf_score": 1.0, "timestamp": "2020-01-01T00:00:00Z", "doc_id": "old"},
            {"_rrf_score": 0.5, "timestamp": "2026-10-30T00:00:00Z", "doc_id": "new"}]
    now = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    ranked = rag_engine._rank_results(rows, "hybrid", 2, now=now)
    assert [r["doc_id"] for r in ranked] == order


@pytest.mark.anyio
async def test_health_reports_ranking(stub_rag_engine, monkeypatch):
    import importlib
    import json
    import sys

    monkeypatch.delitem(sys.modules, "http_server", raising=False)
    http_server = importlib.import_module("http_server")
    response = await http_server.health(None)
    assert json.loads(response.body)["ranking"] == {"recency_weight": 0.15,
                                                    "temporal_recency_weight": 0.7,
                                                    "recency_decay_days": 7,
                                                    "reranker_state": "off"}
