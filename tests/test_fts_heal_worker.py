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
        raise RuntimeError(
            "Milvus collection 'sessions' schema is out of date "
            "(drift=['missing:issue_ids'])"
        )
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


def test_backfill_fts_propagates_non_drift_runtimeerror(monkeypatch):
    """A non-drift RuntimeError from milvus_client.__enter__ must NOT be reclassified.

    Only the schema-drift guard is transient. Model-mismatch / model-not-cached /
    genuine logic-bug RuntimeErrors must surface as RuntimeError so the operator
    sees a config error instead of an infinite retry loop (SESF-38 Copilot review).
    """

    @contextmanager
    def _raising_client(db_path=None):
        raise RuntimeError(
            "Model mismatch: index was built with 'X' but SESSIONFLOW_MODEL is 'Y'."
        )
        yield  # pragma: no cover - unreachable

    monkeypatch.setattr(rag_engine, "milvus_client", _raising_client)
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *a, **k: iter([[{"doc_id": "doc-1"}]]),
    )
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: object())
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)

    db_path = os.path.join(tempfile.gettempdir(), "sessionflow-heal-d4-nondrift.db")
    with pytest.raises(RuntimeError) as excinfo:
        rag_engine.backfill_fts(db_path=db_path)
    assert not isinstance(excinfo.value, rag_engine.FtsBackfillTransientError)


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


# ---------------------------------------------------------------------------
# Memory sync loop: every tick applies pending acknowledgements for this
# machine's consumers; the full reconcile runs on the first tick and then at
# most every 6 hours.
# ---------------------------------------------------------------------------


class _FakeMemoryIndex:
    def __init__(self):
        self.calls = []

    async def tick(self):
        self.calls.append("tick")

    async def reconcile(self):
        self.calls.append("reconcile")
        return 0


def test_memory_sync_reconciles_on_first_tick_then_every_six_hours():
    index = _FakeMemoryIndex()
    clock = [1000.0]
    state = http_server.MemorySyncState(index=index, clock=lambda: clock[0])

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))
    clock[0] += 6 * 3600 + 1
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert index.calls == ["reconcile", "tick", "tick", "reconcile", "tick"]


def test_memory_sync_tick_failure_does_not_stop_the_loop():
    class _Broken(_FakeMemoryIndex):
        async def tick(self):
            raise RuntimeError("postgres unreachable")

    state = http_server.MemorySyncState(index=_Broken(), clock=lambda: 0.0)
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))
    assert "postgres unreachable" in state.last_error


def test_memory_sync_reconcile_failure_still_ticks_and_stays_due():
    class _ReconcileBroken(_FakeMemoryIndex):
        async def reconcile(self):
            self.calls.append("reconcile")
            raise RuntimeError("milvus scan failed")

    index = _ReconcileBroken()
    clock = [0.0]
    state = http_server.MemorySyncState(index=index, clock=lambda: clock[0])

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))
    assert index.calls == ["reconcile", "tick"]
    assert state.last_reconcile is None
    assert "milvus scan failed" in state.last_error

    # Inside the retry delay the pass only ticks; after it, the reconcile runs again.
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))
    assert index.calls == ["reconcile", "tick", "tick"]
    clock[0] += http_server.MEMORY_RECONCILE_RETRY_INTERVAL
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))
    assert index.calls == ["reconcile", "tick", "tick", "reconcile", "tick"]


def test_memory_sync_cancellation_propagates_from_reconcile():
    class _Cancelled(_FakeMemoryIndex):
        async def reconcile(self):
            raise asyncio.CancelledError

    state = http_server.MemorySyncState(index=_Cancelled(), clock=lambda: 0.0)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))


# ---------------------------------------------------------------------------
# A tick or reconcile that never returns is bounded, reported in /health, and
# leaves a log of where it was waiting. The loop goes on.
# ---------------------------------------------------------------------------


async def _waits_forever_inner():
    await asyncio.Event().wait()


class _StuckTick(_FakeMemoryIndex):
    async def tick(self):
        self.calls.append("tick")
        await _waits_forever_inner()


def test_memory_sync_tick_timeout_is_recorded_and_the_loop_goes_on():
    index = _StuckTick()
    state = http_server.MemorySyncState(index=index, clock=lambda: 0.0, tick_timeout=0.05)

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert "timed out" in state.last_error
    assert "tick" in state.last_error


def test_memory_sync_clears_the_timeout_error_after_a_good_tick():
    index = _StuckTick()
    state = http_server.MemorySyncState(index=index, clock=lambda: 0.0, tick_timeout=0.05)
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))
    state.index = _FakeMemoryIndex()

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert state.last_error is None


def test_memory_sync_retry_delay_starts_when_a_slow_reconcile_ends():
    clock = [0.0]

    class _SlowBroken(_FakeMemoryIndex):
        async def reconcile(self):
            self.calls.append("reconcile")
            clock[0] += http_server.MEMORY_RECONCILE_RETRY_INTERVAL + 1
            raise RuntimeError("reconcile gave up")

    index = _SlowBroken()
    state = http_server.MemorySyncState(index=index, clock=lambda: clock[0])

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert index.calls == ["reconcile", "tick", "tick"]


def test_memory_sync_success_marks_attempt_and_reconcile_together():
    clock = [100.0]

    class _Slow(_FakeMemoryIndex):
        async def reconcile(self):
            self.calls.append("reconcile")
            clock[0] += 50
            return 0

    state = http_server.MemorySyncState(index=_Slow(), clock=lambda: clock[0])
    asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))

    assert state.last_reconcile == state.last_reconcile_attempt == 150.0


def test_memory_sync_pauses_while_store_workers_are_still_busy(monkeypatch):
    # A timed-out store call keeps its memory-db thread; the loop must not keep
    # adding work until search has no worker left.
    memory_store = importlib.import_module("memory_store")
    index = _FakeMemoryIndex()
    state = http_server.MemorySyncState(index=index, clock=lambda: 0.0)
    monkeypatch.setattr(memory_store, "db_calls_in_flight",
                        lambda: memory_store.DB_WORKERS - 1)

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert index.calls == []
    assert "paused" in state.last_error


def test_memory_sync_runs_while_one_store_call_is_in_flight(monkeypatch):
    memory_store = importlib.import_module("memory_store")
    index = _FakeMemoryIndex()
    state = http_server.MemorySyncState(index=index, clock=lambda: 0.0, last_reconcile=0.0)
    monkeypatch.setattr(memory_store, "db_calls_in_flight", lambda: 1)

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    assert index.calls == ["tick"]
    assert state.last_error is None


def test_run_db_counts_calls_running_on_the_store_threads():
    import threading

    memory_store = importlib.import_module("memory_store")
    release = threading.Event()
    started = threading.Event()

    def blocking():
        started.set()
        release.wait(5)
        return "done"

    async def run():
        task = asyncio.ensure_future(memory_store.run_db(blocking))
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        during = memory_store.db_calls_in_flight()
        release.set()
        result = await task
        return during, result

    during, result = asyncio.run(run())
    assert during == 1 and result == "done"
    assert memory_store.db_calls_in_flight() == 0


def test_memory_sync_timeout_logs_where_the_tick_was_waiting(caplog):
    import logging

    state = http_server.MemorySyncState(index=_StuckTick(), clock=lambda: 0.0, tick_timeout=0.05)

    with caplog.at_level(logging.WARNING, logger="sessionflow"):
        asyncio.run(http_server._memory_sync_run_once(state, first_tick=False))

    text = caplog.text
    assert "_waits_forever_inner" in text, text
    assert "Event.wait" in text or "wait" in text


def test_memory_sync_reconcile_timeout_still_ticks_and_stays_due():
    class _StuckReconcile(_FakeMemoryIndex):
        async def reconcile(self):
            self.calls.append("reconcile")
            await _waits_forever_inner()

    index = _StuckReconcile()
    state = http_server.MemorySyncState(index=index, clock=lambda: 0.0, tick_timeout=0.05)

    asyncio.run(http_server._memory_sync_run_once(state, first_tick=True))

    assert index.calls == ["reconcile", "tick"]
    assert state.last_reconcile is None
    assert "reconcile timed out" in state.last_error


def test_format_async_tasks_follows_the_await_chain():
    async def middle():
        await _waits_forever_inner()

    async def run():
        task = asyncio.create_task(middle(), name="probe-task")
        await asyncio.sleep(0)
        text = http_server.format_async_tasks()
        task.cancel()
        return text

    text = asyncio.run(run())
    assert "probe-task" in text
    assert "middle" in text and "_waits_forever_inner" in text


def test_sigusr1_logs_the_async_tasks(caplog):
    import logging
    import signal

    async def run():
        assert http_server.install_task_dump_handler()
        try:
            os.kill(os.getpid(), signal.SIGUSR1)
            await asyncio.sleep(0.05)
        finally:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGUSR1)

    with caplog.at_level(logging.WARNING, logger="sessionflow"):
        asyncio.run(run())
    assert "SIGUSR1" in caplog.text


class _HealthIndex:
    def __init__(self, last_tick):
        self.last_tick = last_tick

    def health(self):
        return {"host": "minitron", "consumers": ["milvus", "fts:minitron"],
                "milvus_blocked": None,
                "last_tick": self.last_tick.isoformat() if self.last_tick else None}


@pytest.mark.parametrize("age_s, stale", [(5.0, False), (10_000.0, True)])
def test_memory_health_flags_a_stale_sync_loop(monkeypatch, age_s, stale):
    from datetime import datetime, timedelta, timezone

    index = _HealthIndex(datetime.now(timezone.utc) - timedelta(seconds=age_s))
    state = http_server.MemorySyncState(index=index, tick_timeout=60.0)
    monkeypatch.setitem(http_server._memory_runtime, "index", index)
    monkeypatch.setitem(http_server._memory_runtime, "sync", state)
    monkeypatch.setitem(http_server._memory_status, "ready", False)

    payload = http_server._memory_health_payload()

    assert payload["index"]["stale"] is stale
    assert payload["index"]["last_tick_age_s"] == pytest.approx(age_s, abs=5.0)


def test_memory_health_is_not_stale_before_the_first_tick(monkeypatch):
    index = _HealthIndex(None)
    monkeypatch.setitem(http_server._memory_runtime, "index", index)
    monkeypatch.setitem(http_server._memory_runtime, "sync",
                        http_server.MemorySyncState(index=index, tick_timeout=60.0))
    monkeypatch.setitem(http_server._memory_status, "ready", False)

    payload = http_server._memory_health_payload()

    assert payload["index"]["stale"] is False
    assert payload["index"]["last_tick_age_s"] is None
