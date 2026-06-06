"""Tests for SESF-38 AC-1 (self-heal cadence) and D-4 (typed transient raise).

These tests verify the FTS heal worker seam ``http_server._fts_heal_run_once``
and ``rag_engine.backfill_fts``'s conversion of transient Milvus / schema-drift
failures into ``rag_engine.FtsBackfillTransientError``, per the contract below.

Worker seam contract verified here:

    async def _fts_heal_run_once(state, *, first_tick: bool) -> None

A single heal decision + attempt that the heal loop calls once per cadence
tick, passing ``first_tick=True`` only on the very first tick:

  * ``first_tick=True``  -> run ``rag_engine.backfill_fts`` UNCONDITIONALLY
    (even when ``fts_hybrid.fts_backfill_required()`` is False), preserving the
    startup catch-up behavior of the old one-shot ``_fts_backfill``.
  * ``first_tick=False`` -> run ``rag_engine.backfill_fts`` ONLY when the
    sentinel ``fts_hybrid.fts_backfill_required()`` is True; otherwise skip.
  * On a successful ``backfill_fts`` run the sentinel is cleared (today
    ``backfill_fts`` itself clears it via ``clear_fts_backfill_sentinel``).
  * No restart is involved.

``backfill_fts`` is invoked off the event loop (executor) in production, but the
seam is exercised directly here; the worker must call ``rag_engine.backfill_fts``
by attribute (so monkeypatching ``rag_engine.backfill_fts`` intercepts it).
"""

import asyncio
import importlib
import os
import tempfile
from contextlib import contextmanager

import pytest
from pymilvus.exceptions import MilvusException

rag_engine = importlib.import_module("rag_engine")
http_server = importlib.import_module("http_server")
fts_hybrid = importlib.import_module("fts_hybrid")


def _make_state():
    """Build a fresh FtsHealState the worker owns across ticks."""
    return http_server.FtsHealState()


# --- AC-1: self-heal cadence (first tick unconditional) ---


def test_first_tick_runs_backfill_even_when_sentinel_unset(monkeypatch):
    """first_tick=True with sentinel UNSET -> backfill_fts IS called."""
    calls = []
    monkeypatch.setattr(
        rag_engine, "backfill_fts", lambda *a, **k: calls.append(("backfill", a, k)) or 0
    )
    # Sentinel reports NOT required; first tick must still run.
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda: False)
    monkeypatch.setattr(http_server, "fts_backfill_required", lambda: False, raising=False)

    state = _make_state()
    asyncio.run(http_server._fts_heal_run_once(state, first_tick=True))

    assert calls, "first tick must run backfill_fts unconditionally"


def test_subsequent_tick_skips_backfill_when_sentinel_unset(monkeypatch):
    """first_tick=False with sentinel UNSET -> backfill_fts is NOT called."""
    calls = []
    monkeypatch.setattr(
        rag_engine, "backfill_fts", lambda *a, **k: calls.append(("backfill", a, k)) or 0
    )
    monkeypatch.setattr(fts_hybrid, "fts_backfill_required", lambda: False)
    monkeypatch.setattr(http_server, "fts_backfill_required", lambda: False, raising=False)

    state = _make_state()
    asyncio.run(http_server._fts_heal_run_once(state, first_tick=False))

    assert not calls, "later ticks must skip backfill when sentinel is unset"


def test_subsequent_tick_runs_and_clears_sentinel_when_set(monkeypatch):
    """first_tick=False with sentinel SET -> backfill_fts IS called.

    On its success the sentinel is cleared (real backfill_fts clears it via
    clear_fts_backfill_sentinel; here we model the sentinel as a mutable flag
    and have the stubbed backfill clear it so the worker's contract that a
    successful run leaves the sentinel cleared is exercised end-to-end).
    """
    sentinel = {"required": True}

    def _stub_backfill(*a, **k):
        # Real backfill_fts clears the sentinel on success.
        sentinel["required"] = False
        return 1

    monkeypatch.setattr(rag_engine, "backfill_fts", _stub_backfill)
    monkeypatch.setattr(
        fts_hybrid, "fts_backfill_required", lambda: sentinel["required"]
    )
    monkeypatch.setattr(
        http_server, "fts_backfill_required", lambda: sentinel["required"], raising=False
    )

    state = _make_state()
    asyncio.run(http_server._fts_heal_run_once(state, first_tick=False))

    assert sentinel["required"] is False, (
        "a successful heal run must leave the sentinel cleared"
    )


# --- D-4: backfill_fts raises a typed transient error ---


def test_backfill_fts_raises_typed_error_on_client_open_runtimeerror(monkeypatch):
    """Drift-guard RuntimeError from milvus_client.__enter__ -> FtsBackfillTransientError.

    The guarded ``with milvus_client(...)`` open raising RuntimeError (the
    schema-drift guard) must surface as the typed transient error so the heal
    worker retries on a later tick instead of treating it as terminal. It must
    NOT be swallowed to stderr / returned.
    """

    @contextmanager
    def _raising_client(db_path=None):
        raise RuntimeError("schema drift guard tripped")
        yield  # pragma: no cover - unreachable

    monkeypatch.setattr(rag_engine, "milvus_client", _raising_client)
    # Ensure the batch generator yields at least one chunk so the body reaches
    # the guarded client open. (backfill_fts opens the client around the diff
    # loop; this keeps the test independent of internal ordering.)
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *a, **k: iter([[{"doc_id": "doc-1"}]]),
    )
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: object())
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)

    db_path = os.path.join(tempfile.gettempdir(), "sessionflow-heal-d4-open.db")
    with pytest.raises(rag_engine.FtsBackfillTransientError):
        rag_engine.backfill_fts(db_path=db_path)


def test_backfill_fts_raises_typed_error_on_query_milvusexception(monkeypatch):
    """A MilvusException from a query during backfill -> FtsBackfillTransientError.

    The client opens fine, but a query raises MilvusException (transient Milvus
    fault). It must surface as the typed transient error, not propagate raw or
    be swallowed.
    """

    class _ExplodingMilvus:
        def query(self, *a, **k):
            raise MilvusException(message="transient milvus fault")

    @contextmanager
    def _client(db_path=None):
        yield _ExplodingMilvus()

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *a, **k: iter([[{"doc_id": "doc-1"}]]),
    )

    class _FTSConn:
        def execute(self, *a, **k):
            class _Cursor:
                def fetchall(self_inner):
                    return []  # nothing in FTS -> doc is "missing" -> hydrate -> query

            return _Cursor()

    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: _FTSConn())
    monkeypatch.setattr(rag_engine._fts, "insert", lambda conn, records: None)
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)

    db_path = os.path.join(tempfile.gettempdir(), "sessionflow-heal-d4-query.db")
    with pytest.raises(rag_engine.FtsBackfillTransientError):
        rag_engine.backfill_fts(db_path=db_path)
