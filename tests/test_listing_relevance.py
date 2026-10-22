"""SESF-68 red-phase tests: query-less listings must not print a relevance score.

Listing rows have no query vector, so there is no score to show. They used to
carry a ``distance=1.0`` placeholder, which ``tools.format_results`` printed as
``Relevance: 0.00`` and which read like a failed match. Listing rows now carry
``distance=None`` and render a ``Listing`` label instead. Scored search rows
are unchanged.
"""

from __future__ import annotations

import importlib

from tests.test_recent_listing import DB, _patch_listing, _qrow

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")


def _result(**extra) -> dict:
    row = {
        "content": "body",
        "session_id": "s",
        "turn_index": 7,
        "chunk_type": "turn",
        "timestamp": "2026-10-21T12:00:00+00:00",
    }
    row.update(extra)
    return row


def test_listing_rows_carry_no_distance(monkeypatch):
    _patch_listing(monkeypatch, [_qrow("a", 1), _qrow("b", 2)])
    out = rag_engine.search("", n=5, db_path=DB)
    assert out
    assert all(r["distance"] is None for r in out)


def test_listing_output_has_no_relevance(monkeypatch):
    _patch_listing(monkeypatch, [_qrow("a", 1), _qrow("b", 2)])
    rendered = tools.format_results(rag_engine.search("", n=5, db_path=DB))
    assert "Relevance:" not in rendered
    assert "*Turn: 0 | Type: turn | Listing*" in rendered


def test_scored_search_row_still_shows_relevance():
    rendered = tools.format_results([_result(distance=0.1)])
    assert "*Turn: 7 | Type: turn | Relevance: 0.90*" in rendered
    assert "Listing" not in rendered


def test_mixed_rows_render_independently():
    rendered = tools.format_results([_result(distance=0.25), _result(distance=None)])
    assert rendered.count("Relevance: 0.75") == 1
    assert rendered.count("| Listing*") == 1
