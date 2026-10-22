"""SESF-55 tests — COSINE score orientation and the Milvus count(*) query.

Milvus COSINE returns *similarity* (higher is better) in ``hit["distance"]``;
the result dict's ``distance`` key is a true distance so ``tools.format_results``
can render ``1 - distance`` as relevance. ``count(*)`` queries must not carry a
``limit`` (Milvus rejects pagination on count).

The REAL ``rag_engine`` module is used throughout.
"""

import importlib
from contextlib import contextmanager

rag_engine = importlib.import_module("rag_engine")


def _hit(similarity: float) -> dict:
    entity = {
        "document": "synthetic", "doc_id": "d1", "session_id": "s1",
        "transcript_file": "", "turn_index": 0, "timestamp": "2026-10-18T00:00:00",
        "git_branch": "", "chunk_type": "turn", "project_root": "",
        "logical_session_id": "s1", "provider": "claude_code_cli",
        "source_kind": "claude_code_jsonl", "source_class": "native",
        "source_id": "", "source_path": "",
    }
    return {"entity": entity, "distance": similarity}


def test_search_converts_milvus_cosine_similarity_to_distance(monkeypatch):
    """An exact match (similarity ~1.0) must render as high relevance."""
    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])

    class _Client:
        def search(self, *a, **kw):
            return [[_hit(0.91)]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", lambda *a, **kw: [])
    import fts_hybrid
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda *a, **kw: False)

    results = rag_engine.search("anything", n=1, sort_by="relevance")

    assert results, "expected the stubbed vector hit"
    assert abs(results[0]["distance"] - 0.09) < 1e-6

    tools = importlib.import_module("tools")
    rendered = tools.format_results(results)
    # SESF-77 renamed the scored-row label from "Relevance" to "Semantic".
    assert "Semantic: 0.91" in rendered


def test_count_milvus_turns_does_not_pass_limit(monkeypatch):
    """Milvus rejects count(*) with pagination (code 1100): no ``limit`` kwarg."""
    seen = {}

    class _Client:
        def has_collection(self, name):
            return True

        def query(self, **kw):
            seen.update(kw)
            if "limit" in kw:
                raise RuntimeError("count entities with pagination is not allowed")
            return [{"count(*)": 42}]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client_for_migration", _fake_client)

    assert rag_engine._count_milvus_turns() == 42
    assert "limit" not in seen
    assert seen["output_fields"] == ["count(*)"]
