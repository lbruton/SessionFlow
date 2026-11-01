"""SESF-92 red-phase tests: identifier boost for multi-word queries.

A multi-word query that names a hostname or ticket ID ("galenanswan100 East
Houston port names") kept equal FTS weight, so older notes that were closer
in vector space outranked the one row that holds the identifier. A query with
identifier tokens now runs one extra FTS query for those tokens only and fuses
it as a third list with weight ``IDENTIFIER_FTS_WEIGHT``. A single-token
identifier query keeps its weighted FTS list, and a query with no identifier
token ranks as before.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager

import pytest

import fts_hybrid
from fts_hybrid import FTSIndex, identifier_search_query, identifier_tokens, rrf_merge

rag_engine = importlib.import_module("rag_engine")

COWORK_QUERY = "galenanswan100 East Houston port names Ken Nutanix"


# ---------------------------------------------------------------------------
# identifier_tokens / identifier_search_query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, tokens", [
    (COWORK_QUERY, ["galenanswan100"]),
    ("SESF-41 handoff", ["SESF-41"]),
    ("what closed INC1262779?", ["INC1262779"]),
    ("rag_engine.py and cc.lbruton.sessionflow", ["rag_engine.py", "cc.lbruton.sessionflow"]),
    ("SESF-57 padding sesf-57", ["SESF-57"]),
    ("fix SESF-57,SESF-58; galenanswan100", ["SESF-57", "SESF-58", "galenanswan100"]),
    ("retag project folders", []),
    ("v1 of the bridge", []),
    ("", []),
    (None, []),
])
def test_identifier_tokens(query, tokens):
    assert identifier_tokens(query) == tokens


@pytest.mark.parametrize("query, expected", [
    (COWORK_QUERY, "galenanswan100"),
    ("SESF-41 SESF-57 handoff", "SESF-41 SESF-57"),
    ("INC1262779", ""),
    ("retag project folders", ""),
])
def test_identifier_search_query(query, expected):
    assert identifier_search_query(query) == expected


def test_identifier_search_query_matches_only_identifier_rows(tmp_path):
    fts = FTSIndex("turns_fts", ["session_id", "project_root"])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    try:
        fts.insert(conn, [
            {"doc_id": "host", "content": "capture from galenanswan100 port 12"},
            {"doc_id": "words", "content": "East Houston port names for Ken"},
        ])
    finally:
        fts.close_ephemeral(conn)

    rows = fts.search(identifier_search_query(COWORK_QUERY), n=5, db_path=db_path)
    assert [r["doc_id"] for r in rows] == ["host"]


def test_comma_separated_identifiers_match_rows_with_one_of_them(tmp_path):
    fts = FTSIndex("turns_fts", ["session_id", "project_root"])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    try:
        fts.insert(conn, [{"doc_id": "one", "content": "only SESF-57 here"}])
    finally:
        fts.close_ephemeral(conn)

    for query in ("fix SESF-57,SESF-58", identifier_search_query("fix SESF-57,SESF-58")):
        assert [r["doc_id"] for r in fts.search(query, n=5, db_path=db_path)] == ["one"]


@pytest.mark.parametrize("query", ["v1", "a.b", "INC1262779", "  galenanswan70  "])
def test_single_token_rule_shares_the_token_rule(query):
    assert rag_engine._is_identifier_query(query) is fts_hybrid.is_identifier_token(query.strip())


# ---------------------------------------------------------------------------
# rrf_merge: extra keyword lists
# ---------------------------------------------------------------------------

def _v(doc_id: str) -> dict:
    return {"doc_id": doc_id, "content": doc_id, "distance": 0.5}


def _f(doc_id: str) -> dict:
    return {"doc_id": doc_id, "content": doc_id, "distance": None}


def test_extra_keyword_list_adds_its_weight():
    merged = rrf_merge(
        [_v("near"), _v("other")], [_f("near"), _f("other"), _f("host")], n=5,
        extra_keyword_lists=[([_f("host")], fts_hybrid.IDENTIFIER_FTS_WEIGHT)],
    )
    assert merged[0]["doc_id"] == "host"
    assert merged[0]["_rrf_score"] == pytest.approx(1 / 63 + 2 / 61)


def test_row_only_in_extra_list_carries_a_keyword_rank():
    merged = rrf_merge([_v("near")], [], n=5, extra_keyword_lists=[([_f("host")], 2.0)])
    ranks = {r["doc_id"]: r.get("keyword_rank") for r in merged}
    assert ranks == {"host": 1, "near": None}


def test_main_keyword_rank_wins_over_extra_list_rank():
    merged = rrf_merge([], [_f("a"), _f("host")], n=5, extra_keyword_lists=[([_f("host")], 2.0)])
    assert {r["doc_id"]: r["keyword_rank"] for r in merged}["host"] == 2


def test_no_extra_lists_matches_two_list_merge():
    vec, kw = [_v("a"), _v("b")], [_f("b"), _f("c")]
    assert rrf_merge(vec, kw, n=5) == rrf_merge(vec, kw, n=5, extra_keyword_lists=[])


# ---------------------------------------------------------------------------
# rag_engine.search: the identifier list reaches the fusion
# ---------------------------------------------------------------------------

def _hit(doc_id: str, distance: float) -> dict:
    entity = {
        "document": doc_id, "doc_id": doc_id, "session_id": "s", "transcript_file": "",
        "turn_index": 0, "timestamp": "2026-10-01T00:00:00", "git_branch": "",
        "chunk_type": "turn", "project_root": "", "logical_session_id": "s",
        "provider": "claude_code_cli", "source_kind": "claude_code_jsonl",
        "source_class": "native", "source_id": "", "source_path": "",
    }
    return {"entity": entity, "distance": 1.0 - distance}


def _fts_row(doc_id: str) -> dict:
    row = _hit(doc_id, 0.0)["entity"]
    row = {("content" if k == "document" else k): v for k, v in row.items()}
    row["distance"] = None
    return row


def _patch(monkeypatch, vector, fts_by_query):
    calls: list = []
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, **kw):
            return [[_hit(d, dist) for d, dist in vector]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    def _fts(query, n=15, filters=None, db_path=None):
        calls.append((query, filters))
        return [_fts_row(d) for d in fts_by_query.get(query, [])]

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", _fts)
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda *a, **kw: False)
    return calls


def test_multi_word_identifier_query_ranks_the_identifier_row_first(monkeypatch):
    _patch(monkeypatch, [("near", 0.1), ("other", 0.2)], {
        COWORK_QUERY: ["near", "other", "host"],
        "galenanswan100": ["host"],
    })
    rows = rag_engine.search(COWORK_QUERY, n=3, sort_by="relevance")
    assert rows[0]["doc_id"] == "host"


def test_identifier_query_reuses_the_main_filters(monkeypatch):
    calls = _patch(monkeypatch, [], {})
    rag_engine.search(COWORK_QUERY, n=3, project_root="/p", provider="codex")
    assert [q for q, _ in calls] == [COWORK_QUERY, "galenanswan100"]
    assert calls[0][1] == calls[1][1]


@pytest.mark.parametrize("query", ["retag project folders", "INC1262779"])
def test_no_extra_fts_call_without_a_multi_word_identifier(monkeypatch, query):
    calls = _patch(monkeypatch, [], {})
    rag_engine.search(query, n=3)
    assert [q for q, _ in calls] == [query]


def test_plain_query_ranks_as_before(monkeypatch):
    query = "retag project folders"
    vector = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
    _patch(monkeypatch, vector, {query: ["c", "b", "d"]})
    rows = rag_engine.search(query, n=4, sort_by="relevance")
    expected = rrf_merge(
        [{"doc_id": d} for d, _ in vector], [{"doc_id": d} for d in ["c", "b", "d"]], n=4)
    assert [r["doc_id"] for r in rows] == [r["doc_id"] for r in expected]
