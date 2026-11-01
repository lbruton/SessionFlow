"""SESF-77 red-phase tests: identifier ranking and per-signal display.

The SESF-65 measurement showed two faults. RRF scored vector rank k and FTS
rank k equally and broke ties toward the vector list, so for rare identifiers
(``INC1262779``) unrelated vector hits outranked exact keyword hits. The
displayed ``Relevance`` was not the ranking signal either: vector rows showed
their similarity and FTS-only rows a constant 1.00 (``distance=0.0``).

Identifier-shaped queries now weight FTS ranks by ``IDENTIFIER_FTS_WEIGHT``,
FTS rows carry ``distance=None`` plus ``keyword_rank``, and ``format_results``
prints the signals that ranked each row.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager

import pytest

from fts_hybrid import FTSIndex, rrf_merge

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")


def _v(doc_id: str, distance: float = 0.5) -> dict:
    return {"doc_id": doc_id, "content": doc_id, "distance": distance}


def _f(doc_id: str) -> dict:
    return {"doc_id": doc_id, "content": doc_id, "distance": None}


# ---------------------------------------------------------------------------
# rrf_merge: fts_weight and keyword_rank
# ---------------------------------------------------------------------------

def test_default_weight_keeps_equal_rank_scores():
    merged = rrf_merge([_v("vec")], [_f("kw")], n=5)
    scores = {r["doc_id"]: r["_rrf_score"] for r in merged}
    assert scores["vec"] == pytest.approx(scores["kw"])


def test_fts_weight_puts_keyword_rank1_above_vector_rank1():
    merged = rrf_merge([_v("vec")], [_f("kw")], n=5, fts_weight=2.0)
    assert [r["doc_id"] for r in merged] == ["kw", "vec"]


def test_row_in_both_lists_still_ranks_first_with_weight():
    merged = rrf_merge(
        [_v("vec"), _v("both")], [_f("kw"), _f("both")], n=5, fts_weight=2.0,
    )
    assert merged[0]["doc_id"] == "both"


def test_keyword_rank_set_on_fts_rows_only():
    merged = rrf_merge([_v("vec"), _v("both")], [_f("kw"), _f("both")], n=5)
    ranks = {r["doc_id"]: r.get("keyword_rank") for r in merged}
    assert ranks == {"vec": None, "both": 2, "kw": 1}


def test_row_in_both_lists_keeps_vector_distance():
    merged = rrf_merge([_v("both", distance=0.4)], [_f("both")], n=5)
    assert merged[0]["distance"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _is_identifier_query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "INC1262779", "PATCH-13", "cc.lbruton.sessionflow", "_key_padding_mask",
    "  galenanswan70  ",
])
def test_identifier_shaped_queries(query):
    assert rag_engine._is_identifier_query(query) is True


@pytest.mark.parametrize("query", [
    "Wrenshall", "retag project folders", "", "   ", "SESF-41 handoff",
])
def test_non_identifier_queries(query):
    assert rag_engine._is_identifier_query(query) is False


# ---------------------------------------------------------------------------
# search passes the weight
# ---------------------------------------------------------------------------

def _patch_search(monkeypatch) -> dict:
    seen: dict = {}

    def _spy(vector_results, fts_results, n, k=60, fts_weight=1.0, **kw):
        seen["fts_weight"] = fts_weight
        return []

    monkeypatch.setattr(rag_engine, "rrf_merge", _spy)
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, **kw):
            return [[]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [])
    return seen


def test_search_weights_fts_for_identifier_query(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("INC1262779", n=5)
    assert seen["fts_weight"] == rag_engine.IDENTIFIER_FTS_WEIGHT == 2.0


def test_search_keeps_equal_weight_for_phrase(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("retag project folders", n=5)
    assert seen["fts_weight"] == 1.0


# ---------------------------------------------------------------------------
# FTSIndex.search rows carry no vector distance
# ---------------------------------------------------------------------------

def test_fts_rows_have_no_distance(tmp_path):
    fts = FTSIndex("turns_fts", ["session_id", "project_root"])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    try:
        fts.insert(conn, [{"doc_id": "d1", "content": "ticket INC1262779 closed"}])
    finally:
        fts.close_ephemeral(conn)

    rows = fts.search("INC1262779", n=5, db_path=db_path)
    assert [r["doc_id"] for r in rows] == ["d1"]
    assert rows[0]["distance"] is None


# ---------------------------------------------------------------------------
# format_results: the signals that ranked each row
# ---------------------------------------------------------------------------

def _row(**extra) -> dict:
    row = {"content": "body", "session_id": "s", "turn_index": 3, "chunk_type": "turn"}
    row.update(extra)
    return row


@pytest.mark.parametrize("row, label", [
    (_row(distance=0.56, keyword_rank=2), "Semantic: 0.44 · Keyword: #2"),
    (_row(distance=0.47), "Semantic: 0.53"),
    (_row(distance=None, keyword_rank=1), "Keyword: #1"),
    (_row(distance=None), "Listing"),
])
def test_format_results_signal_labels(row, label):
    rendered = tools.format_results([row])
    assert f"*Turn: 3 | Type: turn | {label}*" in rendered
    assert "Relevance:" not in rendered
