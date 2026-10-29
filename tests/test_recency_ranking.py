"""SESF-24 red-phase tests for hybrid recency-aware ranking + sort_by.

Covers EARS-1 … EARS-13 from the sketch requirements. These exercise:
  - the engine-level strategy dispatcher (`_rank_results`) and age-aware
    `recency_score` scorer in `rag_engine`,
  - two-layer `sort_by` validation (engine `ValueError` + tool `TextContent`),
  - the `hybrid` default and `relevance` parity across candidate-set shapes,
  - env tunable fallback/clamp semantics.

System under test = the SessionFlow search engine (`rag_engine`) and its two
MCP tools (`tools`).
"""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

rag_engine = importlib.import_module("rag_engine")
embedding_control = importlib.import_module("embedding_control")
provider_adapters = importlib.import_module("provider_adapters")


NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _row(doc_id: str, rrf: float, days_ago: float | None = None, **extra) -> dict:
    """A merged-pool row as it exists just before ranking (carries _rrf_score)."""
    row = {
        "doc_id": doc_id,
        "content": f"content {doc_id}",
        "session_id": "s",
        "distance": 0.1,
        "_rrf_score": rrf,
        "timestamp": _iso(days_ago) if days_ago is not None else "",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Engine-level helpers must exist (red until C.1)
# ---------------------------------------------------------------------------

def test_legal_sort_by_frozenset_exists():
    # A.1 — landed already; importable from provider_adapters.
    assert provider_adapters.LEGAL_SORT_BY == frozenset({"relevance", "recency", "hybrid"})


# --- EARS-5 / EARS-12: validation, both layers ---

def test_engine_rejects_invalid_sort_by_before_milvus(monkeypatch):
    """EARS-5: rag_engine.search(sort_by='bogus') raises ValueError naming the
    allowed values, before any embed/Milvus work happens."""
    def _explode(*_a, **_kw):
        raise AssertionError("milvus_client must not be reached for invalid sort_by")

    monkeypatch.setattr(rag_engine, "milvus_client", _explode)
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    with pytest.raises(ValueError, match="sort_by"):
        rag_engine.search("hello", sort_by="bogus")


def test_tool_search_session_rejects_invalid_sort_by_directly():
    """EARS-12: search_session handler returns the DIRECT 'Invalid sort_by'
    TextContent branch — not the global 'Error executing ...' wrapper."""
    tools = importlib.import_module("tools")
    text = _call_tool(tools, "search_session", {"query": "x", "sort_by": "nope"})
    assert "Invalid sort_by" in text
    assert "expected one of" in text
    assert "Error executing" not in text


def test_tool_search_all_sessions_rejects_invalid_sort_by_directly():
    """EARS-12: same direct-branch contract for search_all_sessions."""
    tools = importlib.import_module("tools")
    text = _call_tool(tools, "search_all_sessions", {"query": "x", "sort_by": "nope"})
    assert "Invalid sort_by" in text
    assert "expected one of" in text
    assert "Error executing" not in text


# --- EARS-4 / EARS-13: default = hybrid ---

def test_engine_default_sort_by_is_hybrid(monkeypatch):
    """EARS-4: omitting sort_by drives the hybrid strategy. We capture the
    sort_by that reaches the dispatcher."""
    captured = {}

    def _spy(results, sort_by, n, now=None):
        captured["sort_by"] = sort_by
        return results[:n]

    _patch_pipeline(monkeypatch, vector=[_row("a", 0.9, 1)], fts=[])
    monkeypatch.setattr(rag_engine, "_rank_results", _spy)
    rag_engine.search("hello")
    assert captured["sort_by"] == "hybrid"


@pytest.mark.parametrize("tool_name", ["search_session", "search_all_sessions"])
def test_tool_default_sort_by_is_hybrid(monkeypatch, tool_name):
    """EARS-13: both tools apply hybrid when sort_by is omitted."""
    tools = importlib.import_module("tools")
    captured = {}

    def _spy(*args, **kwargs):
        captured["sort_by"] = kwargs.get("sort_by")
        return []

    monkeypatch.setattr(tools.rag_engine, "search", _spy)
    _call_tool(tools, tool_name, {"query": "x", "project_root": "*"})
    assert captured["sort_by"] == "hybrid"


# --- EARS-3: relevance parity across candidate-set shapes ---

def test_relevance_parity_vector_only(monkeypatch):
    vector = [_row("a", 0.0, 1), _row("b", 0.0, 10), _row("c", 0.0, 5)]
    _patch_pipeline(monkeypatch, vector=vector, fts=[])
    out = rag_engine.search("q", n=5, sort_by="relevance")
    assert [r["doc_id"] for r in out] == ["a", "b", "c"]


def test_relevance_parity_fts_only(monkeypatch):
    fts = [_row("x", 0.0, 1), _row("y", 0.0, 10), _row("z", 0.0, 5)]
    _patch_pipeline(monkeypatch, vector=[], fts=fts)
    out = rag_engine.search("q", n=5, sort_by="relevance")
    assert [r["doc_id"] for r in out] == ["x", "y", "z"]


def test_relevance_parity_mixed_matches_rrf_oracle(monkeypatch):
    vector = [_row("a", 0.0, 1), _row("b", 0.0, 10)]
    fts = [_row("b", 0.0, 10), _row("c", 0.0, 5)]
    expected = [r["doc_id"] for r in rag_engine.rrf_merge(
        [dict(r) for r in vector], [dict(r) for r in fts], n=15)]
    _patch_pipeline(monkeypatch, vector=vector, fts=fts)
    out = rag_engine.search("q", n=5, sort_by="relevance")
    assert [r["doc_id"] for r in out] == expected[:len(out)]


def test_relevance_order_is_timestamp_independent(monkeypatch):
    """EARS-3: relevance must NOT re-rank by recency."""
    vector = [_row("old-but-relevant", 0.0, 30), _row("new-but-less", 0.0, 0)]
    _patch_pipeline(monkeypatch, vector=vector, fts=[])
    out = rag_engine.search("q", n=5, sort_by="relevance")
    assert [r["doc_id"] for r in out] == ["old-but-relevant", "new-but-less"]


# --- EARS-2: recency re-rank of the existing pool ---

def test_recency_orders_by_timestamp_descending():
    rows = [_row("old", 0.9, 10), _row("new", 0.1, 1), _row("mid", 0.5, 5)]
    out = rag_engine._rank_results([dict(r) for r in rows], "recency", n=5, now=NOW)
    assert [r["doc_id"] for r in out] == ["new", "mid", "old"]


def test_recency_missing_timestamp_sorts_last_deterministically():
    rows = [_row("missing", 0.9), _row("new", 0.1, 1), _row("old", 0.5, 9)]
    out = rag_engine._rank_results([dict(r) for r in rows], "recency", n=5, now=NOW)
    assert out[-1]["doc_id"] == "missing"
    assert [r["doc_id"] for r in out[:2]] == ["new", "old"]


# --- EARS-1 / EARS-7: hybrid ordering + normalized terms ---

def test_hybrid_ranks_newer_above_older_for_equal_relevance():
    rows = [_row("older", 0.5, 14), _row("newer", 0.5, 0)]
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    assert [r["doc_id"] for r in out] == ["newer", "older"]


def test_hybrid_blend_uses_normalized_semantic_and_recency(monkeypatch):
    """EARS-1/7: final = (1-w)*semantic + w*recency, both terms in [0,1].

    With w=0.3, a strongly-relevant old row should still beat a weakly-relevant
    fresh row when the semantic gap dominates the recency gap.
    """
    monkeypatch.delenv("SESSIONFLOW_RECENCY_WEIGHT", raising=False)
    monkeypatch.delenv("SESSIONFLOW_RECENCY_DECAY_DAYS", raising=False)
    # semantic: top normalizes to 1.0, bottom to 0.0
    # recency:  exp(-2/7)=0.751 vs exp(0)=1.0  -> gap 0.249
    # final_top = 0.7*1.0 + 0.3*0.751 = 0.925
    # final_bot = 0.7*0.0 + 0.3*1.0   = 0.300
    rows = [_row("relevant_old", 1.0, 2), _row("fresh_irrelevant", 0.0, 0)]
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    assert out[0]["doc_id"] == "relevant_old"


def test_semantic_scores_are_min_max_normalized():
    rows = [_row("a", 0.1), _row("b", 0.5), _row("c", 0.9)]
    scores = rag_engine._semantic_scores([dict(r) for r in rows])
    assert scores[0] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(1.0)
    assert 0.0 <= scores[1] <= 1.0


# --- EARS-6: exponential decay math ---

def test_recency_score_decay_constant():
    """A result exactly decay_days old scores ≈ exp(-1) ≈ 0.3679."""
    ts = _iso(7)
    assert rag_engine._recency_score(ts, NOW, decay_days=7) == pytest.approx(math.exp(-1), rel=1e-3)


def test_recency_score_now_is_one():
    assert rag_engine._recency_score(_iso(0), NOW, decay_days=7) == pytest.approx(1.0, rel=1e-3)


# --- timezone-safety + future clamp ---

def test_recency_score_handles_naive_and_aware_timestamps():
    naive = (NOW - timedelta(days=3)).replace(tzinfo=None).isoformat()
    aware = _iso(3)
    # neither should raise; both treated as ~3 days old
    s_naive = rag_engine._recency_score(naive, NOW, decay_days=7)
    s_aware = rag_engine._recency_score(aware, NOW, decay_days=7)
    assert s_naive == pytest.approx(s_aware, rel=1e-2)


def test_recency_score_future_timestamp_clamps_to_one():
    future = _iso(-5)  # 5 days in the future
    assert rag_engine._recency_score(future, NOW, decay_days=7) == pytest.approx(1.0)


def test_recency_score_parses_z_suffix_timestamp():
    """Real Claude transcripts emit a trailing 'Z'; it must parse as UTC and not
    fall back to the neutral 0.5 (the bug fixed in PR #20 review)."""
    z_ts = "2026-05-25T12:00:00Z"  # exactly 3 days before NOW, Zulu suffix
    plus_ts = "2026-05-25T12:00:00+00:00"
    s_z = rag_engine._recency_score(z_ts, NOW, decay_days=7)
    s_plus = rag_engine._recency_score(plus_ts, NOW, decay_days=7)
    assert s_z == pytest.approx(s_plus, rel=1e-6)
    assert s_z != pytest.approx(rag_engine.MISSING_TIMESTAMP_RECENCY)


# --- EARS-8: missing-timestamp fallback ---

def test_recency_score_missing_timestamp_is_neutral():
    assert rag_engine._recency_score("", NOW, decay_days=7) == pytest.approx(0.5)
    assert rag_engine._recency_score("not-a-date", NOW, decay_days=7) == pytest.approx(0.5)


def test_hybrid_does_not_drop_rows_with_missing_timestamp():
    rows = [_row("a", 0.9, 1), _row("b", 0.5)]
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    assert {r["doc_id"] for r in out} == {"a", "b"}


# --- EARS-9 / EARS-10 / EARS-11: env tunables ---

def test_env_float_uses_value_in_range(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "0.7")
    assert embedding_control._env_float("SESSIONFLOW_RECENCY_WEIGHT", 0.3, 0.0, 1.0) == pytest.approx(0.7)


def test_env_float_falls_back_on_unparseable(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "abc")
    assert embedding_control._env_float("SESSIONFLOW_RECENCY_WEIGHT", 0.3, 0.0, 1.0) == pytest.approx(0.3)


def test_env_float_falls_back_when_out_of_range(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "1.5")
    assert embedding_control._env_float("SESSIONFLOW_RECENCY_WEIGHT", 0.3, 0.0, 1.0) == pytest.approx(0.3)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "Infinity"])
def test_env_float_falls_back_on_non_finite(monkeypatch, bad):
    """float('nan'/'inf') parse cleanly and slip past min/max bounds (every NaN
    comparison is False), so they must be rejected explicitly — otherwise NaN
    propagates into hybrid scores and corrupts sort order (PR #20 review)."""
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", bad)
    assert embedding_control._env_float("SESSIONFLOW_RECENCY_WEIGHT", 0.3, 0.0, 1.0) == pytest.approx(0.3)


def test_recency_weight_env_changes_hybrid_order(monkeypatch):
    """EARS-9: a weight of 1.0 collapses hybrid to pure recency, flipping the
    order vs a relevance-dominated default."""
    rows = [_row("relevant_old", 1.0, 20), _row("fresh_irrelevant", 0.0, 0)]
    monkeypatch.setenv("SESSIONFLOW_RECENCY_WEIGHT", "1.0")
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    assert out[0]["doc_id"] == "fresh_irrelevant"


def test_decay_days_env_clamps_zero_to_minimum_one(monkeypatch):
    """EARS-11: decay_days of 0/negative must not raise ZeroDivisionError —
    _env_int clamps to minimum=1."""
    assert embedding_control._env_int("SESSIONFLOW_RECENCY_DECAY_DAYS", 7, minimum=1) == 7
    monkeypatch.setenv("SESSIONFLOW_RECENCY_DECAY_DAYS", "0")
    assert embedding_control._env_int("SESSIONFLOW_RECENCY_DECAY_DAYS", 7, minimum=1) == 1


def test_decay_days_env_zero_does_not_crash_hybrid(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_RECENCY_DECAY_DAYS", "0")
    rows = [_row("a", 0.9, 1), _row("b", 0.5, 9)]
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    assert len(out) == 2


# --- div-by-zero guard on degenerate pools ---

def test_semantic_scores_single_row_neutral():
    out = rag_engine._semantic_scores([_row("only", 0.42)])
    assert out == [pytest.approx(1.0)]


def test_semantic_scores_all_equal_neutral():
    rows = [_row("a", 0.5), _row("b", 0.5), _row("c", 0.5)]
    out = rag_engine._semantic_scores([dict(r) for r in rows])
    assert all(s == pytest.approx(1.0) for s in out)


def test_hybrid_single_row_does_not_crash():
    out = rag_engine._rank_results([_row("only", 0.5, 3)], "hybrid", n=5, now=NOW)
    assert out[0]["doc_id"] == "only"


# --- non-score metadata preservation ---

def test_rank_results_strips_only_ranking_scratch_keys():
    """_fts_warning (non-score engine metadata) must survive; ranking scratch
    keys (_rrf_score, _score, _semantic_score, _recency_score) must not leak."""
    rows = [_row("a", 0.9, 1, _fts_warning="keyword index rebuilding")]
    out = rag_engine._rank_results([dict(r) for r in rows], "hybrid", n=5, now=NOW)
    row = out[0]
    assert row.get("_fts_warning") == "keyword index rebuilding"
    for scratch in ("_rrf_score", "_score", "_semantic_score", "_recency_score"):
        assert scratch not in row


# ---------------------------------------------------------------------------
# Test plumbing
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, hits):
        self._hits = hits

    def search(self, *a, **kw):
        return [self._hits]


def _dict_to_hit(d: dict) -> dict:
    entity = {
        "document": d.get("content", ""),
        "doc_id": d.get("doc_id", ""),
        "session_id": d.get("session_id", ""),
        "transcript_file": "",
        "turn_index": 0,
        "timestamp": d.get("timestamp", ""),
        "git_branch": "",
        "chunk_type": "turn",
        "project_root": "",
        "logical_session_id": d.get("session_id", ""),
        "provider": "claude_code_cli",
        "source_kind": "claude_code_jsonl",
        "source_class": "native",
        "source_id": "",
        "source_path": "",
    }
    return {"entity": entity, "distance": d.get("distance", 0.1)}


def _patch_pipeline(monkeypatch, vector, fts):
    """Patch embed/milvus/fts so search() runs without live services.

    vector/fts are merged-pool-style dicts; vector dicts are wrapped into
    Milvus hit shape, fts dicts are returned as-is from the FTS index.
    """
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    @contextmanager
    def _fake_client(db_path=None):
        yield _FakeClient([_dict_to_hit(d) for d in vector])

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [dict(d) for d in fts])

    import fts_hybrid
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda *a, **kw: False)


def _call_tool(tools_module, name, arguments):
    """Invoke a registered MCP tool handler and return its text payload."""
    import asyncio

    captured = {}

    class _Server:
        def list_tools(self):
            def deco(fn):
                captured["list"] = fn
                return fn
            return deco

        def call_tool(self):
            def deco(fn):
                captured["call"] = fn
                return fn
            return deco

    tools_module.register_tools(_Server())
    result = asyncio.run(captured["call"](name, arguments))
    return result[0].text


# ---------------------------------------------------------------------------
# Memory search ranks on created_at and breaks exact ties by memory_id
# (descending, matching the listing order). Transcript ranking is unchanged:
# without tie_break, equal rows keep their incoming order.
# ---------------------------------------------------------------------------


def test_recency_ties_keep_input_order_without_tie_break():
    rows = [_row("b", 0.5, 1.0), _row("c", 0.5, 1.0), _row("a", 0.5, 1.0)]
    ranked = rag_engine._rank_results(rows, "recency", 3, now=NOW)
    assert [r["doc_id"] for r in ranked] == ["b", "c", "a"]


def test_recency_ties_break_by_id_when_asked():
    stamp = _iso(1.0)
    rows = [
        {"memory_id": "b", "_rrf_score": 0.5, "created_at": stamp},
        {"memory_id": "c", "_rrf_score": 0.5, "created_at": stamp},
        {"memory_id": "a", "_rrf_score": 0.5, "created_at": stamp},
    ]
    ranked = rag_engine._rank_results(
        rows, "recency", 3, now=NOW, timestamp_key="created_at", tie_break="memory_id"
    )
    assert [r["memory_id"] for r in ranked] == ["c", "b", "a"]


def test_hybrid_uses_the_given_timestamp_key():
    rows = [
        {"memory_id": "old", "_rrf_score": 0.5, "created_at": _iso(60.0)},
        {"memory_id": "new", "_rrf_score": 0.5, "created_at": _iso(0.0)},
    ]
    ranked = rag_engine._rank_results(
        rows, "hybrid", 2, now=NOW, timestamp_key="created_at", tie_break="memory_id"
    )
    assert [r["memory_id"] for r in ranked] == ["new", "old"]
