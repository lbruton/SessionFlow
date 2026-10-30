"""The hybrid ranking settings are read from the environment and reported by /health."""

import pytest

import rag_engine


def test_defaults(monkeypatch):
    monkeypatch.delenv("SESSIONFLOW_RECENCY_WEIGHT", raising=False)
    monkeypatch.delenv("SESSIONFLOW_RECENCY_DECAY_DAYS", raising=False)
    assert rag_engine.ranking_settings() == {
        "recency_weight": rag_engine.RECENCY_WEIGHT_DEFAULT,
        "recency_decay_days": rag_engine.RECENCY_DECAY_DAYS_DEFAULT,
    }


def test_env_values(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "0.15")
    monkeypatch.setenv("SESSIONFLOW_RECENCY_DECAY_DAYS", "14")
    assert rag_engine.ranking_settings() == {"recency_weight": 0.15, "recency_decay_days": 14}


@pytest.mark.parametrize("value", ["2", "-1", "nan", "x"])
def test_out_of_range_weight_falls_back(monkeypatch, value):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", value)
    assert rag_engine.ranking_settings()["recency_weight"] == rag_engine.RECENCY_WEIGHT_DEFAULT


def test_hybrid_rank_uses_the_reported_weight(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "0")
    rows = [{"_rrf_score": 1.0, "timestamp": "2020-01-01T00:00:00Z", "doc_id": "old"},
            {"_rrf_score": 0.5, "timestamp": "2026-10-30T00:00:00Z", "doc_id": "new"}]
    ranked = rag_engine._rank_results(rows, "hybrid", 2)
    assert [r["doc_id"] for r in ranked] == ["old", "new"]


@pytest.mark.anyio
async def test_health_reports_ranking(stub_rag_engine):
    import importlib
    import json
    import sys

    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    response = await http_server.health(None)
    assert json.loads(response.body)["ranking"] == {"recency_weight": 0.3, "recency_decay_days": 7}
