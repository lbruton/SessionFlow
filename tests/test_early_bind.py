"""SESF-70: the HTTP port binds before the slow startup work finishes.

The model preload and Milvus init used to run inside ``lifespan`` before it
yielded, so uvicorn kept port 7102 closed until they finished and MCP clients
that connected at login got ECONNREFUSED. These tests pin the new contract:
lifespan yields immediately, ``/health`` reports 503 ``starting`` until the
background startup completes, and shutdown during startup is clean.
"""

import importlib
import sys
import threading
import time
from contextlib import contextmanager

import pytest


@contextmanager
def _server(stub_rag_engine, tmp_path, monkeypatch, release: threading.Event):
    """Import a fresh http_server whose model load blocks on ``release``."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("HOME", str(tmp_path))

    calls = {"model": 0, "server_mode": 0}

    def slow_get_model(*args, **kwargs):
        calls["model"] += 1
        release.wait(timeout=10)
        return object()

    def init_server_mode(*args, **kwargs):
        calls["server_mode"] += 1

    stub_rag_engine.get_model = slow_get_model
    stub_rag_engine.init_server_mode = init_server_mode

    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")

    async def no_watcher(*args, **kwargs):
        return None

    async def idle_worker(*args, **kwargs):
        return None

    monkeypatch.setattr(http_server.file_watcher, "start_global_watcher", no_watcher)
    monkeypatch.setattr(http_server.file_watcher, "stop_global_watcher", no_watcher)
    monkeypatch.setattr(http_server, "_backfill_drain_worker", idle_worker)
    monkeypatch.setattr(http_server, "_fts_heal_worker", idle_worker)
    monkeypatch.setattr(http_server, "_fts_lag_payload", lambda: {})
    monkeypatch.setattr(http_server, "_provider_health", lambda deep=False: {})

    with TestClient(http_server.app) as client:
        yield http_server, client, calls


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_port_serves_requests_while_model_is_still_loading(
    stub_rag_engine, tmp_path, monkeypatch
):
    release = threading.Event()
    try:
        with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (
            http_server,
            client,
            calls,
        ):
            resp = client.get("/health")
            assert resp.status_code == 503
            body = resp.json()
            assert body["status"] == "starting"
            assert body["startup"] == "starting"
            assert body["server"] == "sessionflow"
    finally:
        release.set()


def test_health_reports_ok_once_background_startup_completes(
    stub_rag_engine, tmp_path, monkeypatch
):
    release = threading.Event()
    with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (
        http_server,
        client,
        calls,
    ):
        release.set()
        assert _wait_for(lambda: client.get("/health").status_code == 200)
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["startup"] == "ready"
        assert body["model_loaded"] is True
        assert body["milvus"] is True
        assert calls == {"model": 1, "server_mode": 1}


def test_startup_failure_still_finishes_and_reports_ready(
    stub_rag_engine, tmp_path, monkeypatch
):
    """A failed preload keeps the old warn-and-continue behaviour."""
    def broken_get_model(*args, **kwargs):
        raise RuntimeError("model missing")

    monkeypatch.setenv("HOME", str(tmp_path))
    stub_rag_engine.get_model = broken_get_model
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(http_server.file_watcher, "start_global_watcher", noop)
    monkeypatch.setattr(http_server.file_watcher, "stop_global_watcher", noop)
    monkeypatch.setattr(http_server, "_backfill_drain_worker", noop)
    monkeypatch.setattr(http_server, "_fts_heal_worker", noop)
    monkeypatch.setattr(http_server, "_fts_lag_payload", lambda: {})
    monkeypatch.setattr(http_server, "_provider_health", lambda deep=False: {})

    from starlette.testclient import TestClient

    with TestClient(http_server.app) as client:
        assert _wait_for(lambda: client.get("/health").status_code == 200)
        body = client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["startup"] == "ready"


def test_shutdown_during_startup_is_clean(stub_rag_engine, tmp_path, monkeypatch):
    release = threading.Event()
    try:
        with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (
            http_server,
            client,
            calls,
        ):
            assert client.get("/health").status_code == 503
        # Leaving the TestClient context ran lifespan shutdown while the model
        # was still loading; the PID file must be gone and nothing must hang.
        assert not http_server.PID_FILE.exists()
    finally:
        release.set()


@pytest.mark.anyio
async def test_health_handler_direct_call_reports_starting_before_startup(stub_rag_engine):
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")

    response = await http_server.health(None)

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Memory start-up (DSN fetch, Postgres migration, memories collection check)
# runs after the bind and reports its own readiness: a slow or failed memory
# start never delays or breaks transcript search.
# ---------------------------------------------------------------------------


def test_slow_memory_startup_does_not_delay_transcript_readiness(
    stub_rag_engine, tmp_path, monkeypatch
):
    memory_release = threading.Event()

    def blocked_fetch(*args, **kwargs):
        memory_release.wait(timeout=10)
        raise RuntimeError("released")

    # memory_store is not re-imported by _server, so this patch reaches the server.
    monkeypatch.setattr("memory_store.fetch_dsn", blocked_fetch, raising=False)
    release = threading.Event()
    release.set()
    try:
        with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (http_server, client, calls):
            assert _wait_for(lambda: client.get("/health").status_code == 200)
            body = client.get("/health").json()
            assert body["startup"] == "ready" and body["milvus"] is True
            assert set(body["memory"]) >= {"ready", "error", "ack_lag", "dump"}
            assert body["memory"]["ready"] is False
    finally:
        memory_release.set()


def test_failed_dsn_fetch_leaves_transcripts_ready(stub_rag_engine, tmp_path, monkeypatch):
    def broken_fetch(*args, **kwargs):
        raise RuntimeError("memory store not configured: set SESSIONFLOW_MEMORY_DSN")

    monkeypatch.setattr("memory_store.fetch_dsn", broken_fetch, raising=False)
    release = threading.Event()
    release.set()
    with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (http_server, client, calls):
        assert _wait_for(lambda: client.get("/health").status_code == 200)
        assert _wait_for(lambda: client.get("/health").json()["memory"]["error"] is not None)
        body = client.get("/health").json()
        assert body["startup"] == "ready" and body["milvus"] is True
        assert body["memory"]["ready"] is False
        assert "memory store not configured" in body["memory"]["error"]



def test_shutdown_cancels_the_reranker_load_before_closing_server_mode(
    stub_rag_engine, tmp_path, monkeypatch
):
    import asyncio

    monkeypatch.setenv("SESSIONFLOW_RERANKER", "jina")
    events = []
    stub_rag_engine.close_server_mode = lambda *a, **kw: events.append("close_server_mode")
    release = threading.Event()
    started = threading.Event()

    async def hanging_load():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("reranker_cancelled")
            raise

    try:
        with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (http_server, _, _):
            monkeypatch.setattr(http_server, "_load_reranker", hanging_load)
            release.set()
            assert _wait_for(started.is_set)
    finally:
        release.set()
    assert events[:2] == ["reranker_cancelled", "close_server_mode"]


def test_default_reranker_without_an_mlx_executor_reports_failed(
    stub_rag_engine, tmp_path, monkeypatch
):
    import reranker

    monkeypatch.delenv("SESSIONFLOW_RERANKER", raising=False)
    monkeypatch.delattr(stub_rag_engine, "_embed_executor", raising=False)
    release = threading.Event()
    release.set()
    with _server(stub_rag_engine, tmp_path, monkeypatch, release) as (_, client, _calls):
        assert _wait_for(lambda: client.get("/health").status_code == 200)
        assert _wait_for(lambda: reranker.status().startswith("failed: "))
        state = client.get("/health").json()["ranking"]["reranker_state"]
        assert state == "failed: RuntimeError: the mlx-embed executor is not running"
