"""SESF-38 AC-6 RED tests — FTS-vs-Milvus lag observability + project scoping.

These tests encode the target behavior Cohort C will implement. They MUST fail
against the current skeletons:

  * ``rag_engine.fts_lag_status`` returns an all-zero/False stub (rag_engine.py).
  * ``FTSIndex.count_rows`` returns a stub ``0`` (fts_hybrid.py).
  * ``rag_engine.get_stats`` does not yet carry the additive FTS keys.
  * ``/health`` JSON does not yet carry a top-level ``fts`` key (http_server.py).

The REAL ``rag_engine`` module is used throughout (never the conftest
``stub_rag_engine`` namespace), so the stub can't mask the missing behavior.
"""

import importlib
from contextlib import contextmanager

import pytest

rag_engine = importlib.import_module("rag_engine")


# ---------------------------------------------------------------------------
# AC-6 (1): fts_lag_status assembles the four lag keys from real counts.
# ---------------------------------------------------------------------------

def test_fts_lag_status_computes_lag_from_milvus_and_fts_counts(monkeypatch):
    """fts_lag_status must subtract the FTS row count from the Milvus turn count.

    Today the function is a skeleton returning all zeros/False, so the computed
    ``fts_lag`` (12) and ``fts_backfill_required`` (True) assertions fail RED.
    """
    milvus_count = 20
    fts_count = 8

    # Stub the Milvus turn-count path. C.4 wires fts_lag_status to count Milvus
    # rows; we patch get_stats (its natural count source) to return a known total.
    monkeypatch.setattr(
        rag_engine,
        "get_stats",
        lambda project_root=None, db_path=None: {"total_turns": milvus_count},
    )

    # Stub the FTS count path: connection + count_rows feed the FTS side.
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: object())
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)
    monkeypatch.setattr(
        rag_engine._fts, "count_rows", lambda conn, project_root=None: fts_count
    )

    status = rag_engine.fts_lag_status(db_path="/tmp/sessionflow-lag-test.db")

    assert set(status) >= {
        "milvus_turn_count",
        "fts_row_count",
        "fts_lag",
        "fts_backfill_required",
    }
    assert status["milvus_turn_count"] == milvus_count
    assert status["fts_row_count"] == fts_count
    assert status["fts_lag"] == milvus_count - fts_count
    assert status["fts_backfill_required"] is True


# ---------------------------------------------------------------------------
# AC-6 (2): FTSIndex.count_rows filters by project_root.
# ---------------------------------------------------------------------------

def _build_fts(tmp_path):
    """Build a real FTSIndex (ephemeral mode) backed by a temp sqlite db."""
    from fts_hybrid import FTSIndex

    fts = FTSIndex(
        "turns_fts",
        [
            "session_id", "git_branch", "turn_index", "timestamp", "chunk_type",
            "project_root", "logical_session_id", "provider", "source_kind",
            "source_class", "source_id", "source_path", "issue_ids",
        ],
    )
    db_path = str(tmp_path / "milvus.db")
    return fts, db_path


def test_count_rows_project_scoped_differs_from_global(tmp_path):
    """count_rows(project_root=X) must count only that project's rows.

    Two rows live under project A and one under project B. The global count is 3
    and the project-A count is 2. Today the stub returns 0 for both, so the
    inequality (project-scoped != global) and the exact-count assertions fail RED.
    """
    fts, db_path = _build_fts(tmp_path)
    conn = fts.connection(db_path)
    try:
        fts.insert(conn, [
            {"doc_id": "a1", "content": "alpha one", "project_root": "/proj/A"},
            {"doc_id": "a2", "content": "alpha two", "project_root": "/proj/A"},
            {"doc_id": "b1", "content": "bravo one", "project_root": "/proj/B"},
        ])

        global_count = fts.count_rows(conn)
        project_a_count = fts.count_rows(conn, project_root="/proj/A")

        assert global_count == 3
        assert project_a_count == 2
        assert project_a_count != global_count
    finally:
        fts.close_ephemeral(conn)


# ---------------------------------------------------------------------------
# AC-6 (3): get_stats gains additive, project-scoped FTS keys.
# ---------------------------------------------------------------------------

def test_get_stats_carries_fts_lag_keys(monkeypatch, tmp_path):
    """get_stats(project_root=X) must add fts_row_count / fts_lag / fts_backfill_required.

    The keys are additive and project-scoped (consistent with get_stats' existing
    Milvus project_root filter). Today get_stats returns only total/sessions/
    by_type/providers, so the membership assertions fail RED.
    """
    # Make the Milvus side of get_stats deterministic and cheap: no live cluster.
    @contextmanager
    def _fake_client(db_path=None):
        class _Client:
            def has_collection(self, name):
                return True
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(
        rag_engine, "_query_all",
        lambda fields, filter_expr=None, db_path=None: [
            {"session_id": "s1", "chunk_type": "turn",
             "git_branch": "main", "project_root": "/proj/A", "provider": "claude_code_cli"},
        ],
    )

    # FTS side feeds the new keys.
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: object())
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)
    monkeypatch.setattr(
        rag_engine._fts, "count_rows", lambda conn, project_root=None: 1
    )

    stats = rag_engine.get_stats(
        project_root="/proj/A", db_path=str(tmp_path / "milvus.db")
    )

    assert "fts_row_count" in stats
    assert "fts_lag" in stats
    assert "fts_backfill_required" in stats


# ---------------------------------------------------------------------------
# AC-6 (4): /health JSON carries a top-level `fts` key.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health_payload_carries_fts_lag(monkeypatch):
    """The /health route payload must gain a top-level `fts` key (AC-6).

    Uses the REAL rag_engine; only the heavy, side-effecting payload helpers
    (provider scan, backfill manager, embedding budget) are stubbed so the route
    runs offline. The route itself is exercised end to end. Today the payload has
    no `fts` key, so the membership assertion fails RED.
    """
    import json

    http_server = importlib.import_module("http_server")

    # Neutralize the heavy, filesystem-touching payload builders so the route
    # runs offline. None of these are the assertion target.
    monkeypatch.setattr(http_server, "_provider_health", lambda deep=False: {})
    monkeypatch.setattr(http_server, "_backfill_status_payload", lambda: {})
    monkeypatch.setattr(http_server, "_embedding_status_payload", lambda: {})
    monkeypatch.setattr(
        http_server.file_watcher, "get_watcher_status", lambda: {}
    )

    response = await http_server.health(None)
    payload = json.loads(response.body.decode("utf-8"))

    assert "fts" in payload
