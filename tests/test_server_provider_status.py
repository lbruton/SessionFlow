"""SESF-6 red-phase tests for HTTP health/backfill status surfaces.

Requirements: 6.6, 6.7, 7, 8.
"""

import importlib
import json
import asyncio
from contextlib import contextmanager

import pytest


@pytest.mark.anyio
async def test_health_payload_reports_providers_queue_and_local_embedding(stub_rag_engine):
    http_server = importlib.import_module("http_server")

    response = await http_server.health(None)
    payload = response.body.decode("utf-8")

    assert "providers" in payload
    assert "backfill" in payload
    assert "embedding" in payload
    assert "local_mlx" in payload


def test_local_backfill_control_endpoint_exists(stub_rag_engine):
    http_server = importlib.import_module("http_server")

    routes = {getattr(route, "path", "") for route in http_server.app.routes}

    assert "/backfill" in routes or "/backfill/control" in routes


def test_backfill_drain_interval_rejects_non_numeric_value(stub_rag_engine, monkeypatch):
    http_server = importlib.import_module("http_server")

    monkeypatch.setenv("SESSIONFLOW_BACKFILL_DRAIN_INTERVAL_SECONDS", "bad")
    with pytest.raises(ValueError, match="must be a number"):
        http_server._parse_backfill_drain_interval()


@contextmanager
def _backfill_client(stub_rag_engine, tmp_path, monkeypatch):
    """Reload http_server with the rag_engine stub in place and wrap it in a
    Starlette TestClient. Each test gets a fresh import so /backfill state
    can't leak across tests."""
    import sys

    from starlette.testclient import TestClient

    monkeypatch.setenv("HOME", str(tmp_path))
    # Force a fresh module so the in-process _backfill_manager is reset.
    sys.modules.pop("http_server", None)
    http_server = importlib.import_module("http_server")
    client = TestClient(http_server.app)
    try:
        yield http_server, client
    finally:
        client.close()


def test_backfill_pause_endpoint_pauses_provider(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, client):
        resp = client.post("/backfill", json={"action": "pause", "provider": "codex"})
        assert resp.status_code == 200
        assert "codex" in http_server._backfill_manager.paused_providers


def test_backfill_enqueue_endpoint_queues_recent_mode(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, client):
        resp = client.post(
            "/backfill",
            json={"action": "enqueue", "provider": "opencode", "mode": "recent"},
        )
        assert resp.status_code == 200
        queued = http_server._backfill_manager.status().jobs
        assert any(j.provider == "opencode" and j.mode == "recent" for j in queued)


def test_backfill_enqueue_endpoint_wakes_drain_worker(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, client):
        woke = []
        monkeypatch.setattr(http_server, "_wake_backfill_drain", lambda: woke.append(True))

        resp = client.post(
            "/backfill",
            json={"action": "enqueue", "provider": "opencode", "mode": "recent"},
        )

        assert resp.status_code == 200
        assert woke == [True]


@pytest.mark.anyio
async def test_backfill_drain_processes_queued_job(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        class FakeIngestionService:
            def __init__(self, manager, db_path):
                self.manager = manager

            async def process_queued_jobs(self):
                for job in list(self.manager.status().jobs):
                    self.manager.complete_job(job.job_id, processed_sources=1, indexed_turns=2)
                return {"jobs": 1, "processed_sources": 1, "indexed_turns": 2, "errors": 0}

        monkeypatch.setattr(http_server, "ProviderIngestionService", FakeIngestionService)
        http_server._backfill_manager.enqueue_provider_backfill(provider="opencode", mode="recent")

        result = await http_server._drain_backfill_once()

        assert result["jobs"] == 1
        assert result["indexed_turns"] == 2
        assert http_server._backfill_manager.status().jobs == []


@pytest.mark.anyio
async def test_backfill_drain_respects_global_pause(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        class FailingIngestionService:
            def __init__(self, manager, db_path):
                pass

            async def process_queued_jobs(self):
                raise AssertionError("paused queue should not drain")

        monkeypatch.setattr(http_server, "ProviderIngestionService", FailingIngestionService)
        http_server._backfill_manager.enqueue_provider_backfill(provider="opencode", mode="recent")
        http_server._backfill_manager.pause()

        result = await http_server._drain_backfill_once()

        assert result["jobs"] == 0
        assert len(http_server._backfill_manager.status().jobs) == 1


@pytest.mark.anyio
async def test_backfill_drain_does_not_overlap(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        entered = asyncio.Event()
        release = asyncio.Event()

        class SlowIngestionService:
            def __init__(self, manager, db_path):
                pass

            async def process_queued_jobs(self):
                entered.set()
                await release.wait()
                return {"jobs": 1, "processed_sources": 0, "indexed_turns": 0, "errors": 0}

        monkeypatch.setattr(http_server, "ProviderIngestionService", SlowIngestionService)
        http_server._backfill_manager.enqueue_provider_backfill(provider="opencode", mode="recent")

        first = asyncio.create_task(http_server._drain_backfill_once())
        await entered.wait()
        second = await http_server._drain_backfill_once()
        release.set()

        assert second["skipped"] == 1
        await first


def test_backfill_unknown_action_returns_400(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (_, client):
        resp = client.post("/backfill", json={"action": "garbage"})
        assert resp.status_code == 400


def test_backfill_enqueue_rejects_non_integer_priority(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (_, client):
        resp = client.post(
            "/backfill",
            json={
                "action": "enqueue",
                "provider": "codex",
                "priority": "not-a-number",
            },
        )
        assert resp.status_code == 400


@pytest.mark.anyio
async def test_backfill_run_action_uses_shared_drain_path(stub_rag_engine, tmp_path, monkeypatch):
    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        class Request:
            method = "POST"

            async def json(self):
                return {
                    "action": "run",
                    "mode": "incremental",
                    "providers": ["opencode"],
                }

        drain_kwargs = []

        async def fake_drain(*args, **kwargs):
            drain_kwargs.append(kwargs)
            return {"jobs": 1, "processed_sources": 0, "indexed_turns": 0, "errors": 0}

        monkeypatch.setattr(http_server, "_drain_backfill_once", fake_drain)

        response = await http_server.backfill_control_endpoint(Request())
        payload = response.body.decode("utf-8")

        assert response.status_code == 200
        assert '"jobs":1' in payload.replace(" ", "")
        assert drain_kwargs == [{"skip_if_locked": False}]


def test_provider_health_covers_every_legal_provider_without_cowork_probe(
    stub_rag_engine, tmp_path, monkeypatch
):
    from provider_adapters import LEGAL_PROVIDERS

    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        health = http_server._provider_health(deep=True)

    assert "claude_desktop_cowork" not in health
    assert set(health) == set(LEGAL_PROVIDERS)


@pytest.mark.anyio
async def test_backfill_run_defaults_to_every_legal_provider(stub_rag_engine, tmp_path, monkeypatch):
    from provider_adapters import LEGAL_PROVIDERS

    with _backfill_client(stub_rag_engine, tmp_path, monkeypatch) as (http_server, _):
        class Request:
            method = "POST"

            async def json(self):
                return {"action": "run", "mode": "incremental"}

        async def fake_drain(*args, **kwargs):
            return {"jobs": 0, "processed_sources": 0, "indexed_turns": 0, "errors": 0}

        monkeypatch.setattr(http_server, "_drain_backfill_once", fake_drain)

        response = await http_server.backfill_control_endpoint(Request())
        payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["run"]["providers"] == sorted(LEGAL_PROVIDERS)
