"""SESF-25 red-phase tests for the optional ``issue_id`` search filter.

Per design.md Component 4, ``search(..., issue_id=...)`` threads the value into
``_build_milvus_filter``, which appends (when set) the delimiter-wrapped LIKE
clause::

    issue_ids like "%,<UPPER-ID>,%"

The wrap (``,ID,``) gives exact-token containment so ``SESF-42`` cannot match a
stored ``SESF-420``. It is AND-combined with provider/project/date filters
(Req 3.3); when unset the filter expression is byte-for-byte unchanged (Req 3.4);
a no-match yields an empty list without error (Req 3.5); and the value is
uppercased so case-insensitive lookups match (Req 3.6).

These tests capture the filter expression the engine sends to Milvus by driving
a real (non-empty) query through ``rag_engine.search`` with a fake client.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from contextlib import contextmanager

import pytest

rag_engine = importlib.import_module("rag_engine")

DB = os.path.join(tempfile.gettempdir(), "sessionflow-issueid-test.db")


class _CapturingClient:
    """Fake Milvus client capturing the vector-search filter expression."""

    def __init__(self):
        self.search_calls: list[dict] = []

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return [[]]  # no hits → empty merged result

    def query(self, *args, **kwargs):
        raise AssertionError("query path must not run for a real query")


def _patch_search(monkeypatch, client):
    """Patch embed/milvus/fts so the vector search path runs offline."""
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    @contextmanager
    def _fake_client(db_path=None):
        yield client

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [])

    import fts_hybrid
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda *a, **kw: False)


def _filter_for(monkeypatch, **search_kwargs) -> str | None:
    client = _CapturingClient()
    _patch_search(monkeypatch, client)
    rag_engine.search("real query", db_path=DB, **search_kwargs)
    assert client.search_calls, "expected the vector search path to run"
    return client.search_calls[0].get("filter")


# ---------------------------------------------------------------------------
# Filter-expression shape
# ---------------------------------------------------------------------------

def test_issue_id_builds_delimiter_wrapped_like(monkeypatch):
    # Req 3.1 — issue_id set → comma-wrapped LIKE clause.
    expr = _filter_for(monkeypatch, issue_id="SESF-25")
    assert expr is not None
    assert 'issue_ids like "%,SESF-25,%"' in expr


def test_issue_id_case_insensitive_uppercases_value(monkeypatch):
    # Req 3.6 — a lowercased value is uppercased to match the canonical store.
    expr = _filter_for(monkeypatch, issue_id="sesf-25")
    assert expr is not None
    assert 'issue_ids like "%,SESF-25,%"' in expr


def test_issue_id_token_boundary_wrap_excludes_superstring(monkeypatch):
    # Boundary — SESF-42 wraps as ",SESF-42," so it can't match a SESF-420 row.
    expr = _filter_for(monkeypatch, issue_id="SESF-42")
    assert expr is not None
    assert 'issue_ids like "%,SESF-42,%"' in expr
    assert "SESF-420" not in expr


def test_issue_id_ands_with_provider_and_date(monkeypatch):
    # Req 3.3 — conjunctive (&&) combine with provider + date_from.
    expr = _filter_for(
        monkeypatch,
        issue_id="SESF-25",
        provider="codex",
        date_from="2026-05-01",
    )
    assert expr is not None
    assert "&&" in expr
    assert 'issue_ids like "%,SESF-25,%"' in expr
    assert 'provider == "codex"' in expr
    assert 'timestamp >= "2026-05-01"' in expr


def test_unset_issue_id_leaves_expr_byte_identical(monkeypatch):
    # Req 3.4 — omitting issue_id must not perturb the existing filter string.
    with_provider = _filter_for(monkeypatch, provider="codex")
    again = _filter_for(monkeypatch, provider="codex", issue_id=None)
    assert with_provider == again
    if with_provider is not None:
        assert "issue_ids" not in with_provider


def test_unset_issue_id_no_filter_when_no_other_filters(monkeypatch):
    # Req 3.4 — with no filters at all the expression is unchanged (None / no issue_ids).
    expr = _filter_for(monkeypatch)
    if expr is not None:
        assert "issue_ids" not in expr


# ---------------------------------------------------------------------------
# No-match behavior
# ---------------------------------------------------------------------------

def test_issue_id_no_match_returns_empty_without_error(monkeypatch):
    # Req 3.5 — a filter matching no rows yields [] and does not raise.
    client = _CapturingClient()
    _patch_search(monkeypatch, client)
    out = rag_engine.search("real query", issue_id="SESF-99999", db_path=DB)
    assert out == []


# ---------------------------------------------------------------------------
# SESF-32 — hybrid FTS half must honor issue_id too (not just the Milvus half)
# ---------------------------------------------------------------------------

class _CapturingFTS:
    """Records the filters handed to ``_fts.search`` and returns canned rows.

    The pre-SESF-32 tests mock ``_fts.search`` to ``[]``, which never exercises
    the leak: with no FTS rows there is nothing to (mis)merge. This double both
    captures the filter dict and can replay tagged/untagged rows.
    """

    def __init__(self, rows=None):
        self.calls: list[dict] = []
        self._rows = rows or []

    def __call__(self, query, n=15, filters=None, db_path=None):
        self.calls.append({"query": query, "n": n, "filters": filters, "db_path": db_path})
        return self._rows


def _patch_search_capturing_fts(monkeypatch, client, fts_rows=None) -> _CapturingFTS:
    """Like ``_patch_search`` but swaps in a filter-capturing FTS double."""
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    @contextmanager
    def _fake_client(db_path=None):
        yield client

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    cap = _CapturingFTS(fts_rows)
    monkeypatch.setattr(rag_engine._fts, "search", cap)

    import fts_hybrid
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda *a, **kw: False)
    return cap


def test_fts_path_receives_issue_id_containment_filter(monkeypatch):
    # SESF-32 — issue_id must reach the FTS filter as a comma-wrapped LIKE clause
    # mirroring the Milvus side, so the hybrid FTS half can't leak untagged rows.
    client = _CapturingClient()
    cap = _patch_search_capturing_fts(monkeypatch, client)
    rag_engine.search("real query", issue_id="sesf-25", db_path=DB)
    assert cap.calls, "expected the FTS search path to run"
    fts_filters = cap.calls[0]["filters"] or {}
    assert fts_filters.get("issue_ids") == ("like", "%,SESF-25,%")


def test_fts_path_omits_issue_id_filter_when_unset(monkeypatch):
    # SESF-32 — with no issue_id, no issue_ids clause should appear in the FTS
    # filter (mirrors Req 3.4 on the Milvus side).
    client = _CapturingClient()
    cap = _patch_search_capturing_fts(monkeypatch, client)
    rag_engine.search("real query", provider="codex", db_path=DB)
    assert cap.calls
    fts_filters = cap.calls[0]["filters"] or {}
    assert "issue_ids" not in fts_filters


def test_fts_index_like_filter_excludes_untagged_rows(tmp_path):
    # SESF-32 — the containment filter the fix plumbs through actually excludes
    # untagged rows at the SQLite layer (guards the (op, operand) mechanism).
    from fts_hybrid import FTSIndex

    fts = FTSIndex("turns_fts", [
        "session_id", "git_branch", "turn_index", "timestamp", "chunk_type",
        "project_root", "logical_session_id", "provider", "source_kind",
        "source_class", "source_id", "source_path", "issue_ids",
    ])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    fts.insert(conn, [
        {"doc_id": "tagged", "content": "auth token refresh", "issue_ids": ",SESF-25,"},
        {"doc_id": "untagged", "content": "auth token refresh", "issue_ids": ""},
    ])
    hits = fts.search("auth", filters={"issue_ids": ("like", "%,SESF-25,%")}, db_path=db_path)
    assert {h["doc_id"] for h in hits} == {"tagged"}
