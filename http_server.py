#!/usr/bin/env python3
"""
Persistent HTTP server for SessionFlow MCP system.

Runs as a long-lived process serving MCP via StreamableHTTP.
Projects are identified by the X-Project-Root header in each request.
All projects share a single global DB at ~/.sessionflow/milvus.db.

Start: ./sessionflow-server.sh
Health: curl http://127.0.0.1:7102/health
"""

import asyncio
import concurrent.futures
import contextlib
from dataclasses import asdict, dataclass
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import rag_engine
import transcript_parser
import file_watcher
import fts_hybrid
from provider_adapters import LEGAL_PROVIDERS, is_valid_issue_token
from backfill_manager import BackfillManager
from embedding_control import EmbeddingIdentity, get_embedding_budget
from provider_antigravity import AntigravityAdapter
from provider_claude import ClaudeCodeCliAdapter, ClaudeDesktopCoworkProbe
from provider_codex import CodexAdapter
from provider_opencode import OpenCodeAdapter
from provider_ingestion import ProviderIngestionService, process_startup_provider_backfill
from file_watcher import register_project, get_global_watcher
from tools import register_tools, set_current_project_root

logger = logging.getLogger("sessionflow")


# --- Configuration ---

HOST = os.getenv("SESSIONFLOW_HOST", "127.0.0.1")
PORT = int(os.getenv("SESSIONFLOW_PORT", "7102"))
AUTO_EXPIRE_DAYS = int(os.getenv("SESSIONFLOW_EXPIRE_DAYS", "365"))
_EXPIRE_CHECK_INTERVAL = 86400  # Check once per day

_SERVER_DIR = Path.home() / ".sessionflow"
PID_FILE = _SERVER_DIR / "server.pid"
LOG_FILE = _SERVER_DIR / "server.log"


def _parse_backfill_drain_interval() -> float:
    raw_interval = os.getenv("SESSIONFLOW_BACKFILL_DRAIN_INTERVAL_SECONDS", "30")
    try:
        interval = float(raw_interval)
    except ValueError as exc:
        raise ValueError(
            "SESSIONFLOW_BACKFILL_DRAIN_INTERVAL_SECONDS must be a number"
        ) from exc
    return interval


BACKFILL_DRAIN_INTERVAL = _parse_backfill_drain_interval()
if BACKFILL_DRAIN_INTERVAL <= 0:
    raise ValueError("SESSIONFLOW_BACKFILL_DRAIN_INTERVAL_SECONDS must be > 0")

# Upper bound on the FTS heal exponential backoff (SESF-38 AC-2). The schedule
# escalates as ``BACKFILL_DRAIN_INTERVAL * 2**(n-1)`` per consecutive failure and
# is held at this cap so the heal loop retries forever without runaway delays.
FTS_HEAL_BACKOFF_CAP = 300.0

# Milvus backend: remote Standalone URI or local Lite file path.
MILVUS_URI = os.getenv("SESSIONFLOW_MILVUS_URI", str(_SERVER_DIR / "milvus.db"))
HEARTBEAT_FILE = _SERVER_DIR / "heartbeat"


# --- Heartbeat ---

class HeartbeatThread:
    """Daemon thread that writes a JSON heartbeat file at fixed intervals.

    File I/O releases the GIL, so this runs even when MLX Metal holds
    the GIL during embedding computation.
    """

    def __init__(self, path: Path, interval: float = 30.0):
        """
        Initialize a HeartbeatThread that periodically writes a JSON heartbeat to the given file.
        
        Parameters:
            path (Path): Destination file path where heartbeat JSON will be written.
            interval (float): Seconds between heartbeat writes (defaults to 30.0). The thread will attempt a write at this interval.
        
        Detailed behavior:
            - Creates an internal stop event used to signal the background thread to exit.
            - Initializes the thread placeholder; the background thread is started by calling start().
        """
        self._path = path
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def _get_activity(self) -> str:
        """
        Return the current file-watcher activity label.
        
        Queries the global file watcher status and returns "processing" if the watcher reports active processing; on any error or if not processing, returns "idle".
        
        Returns:
            activity (str): "processing" if the watcher is processing, "idle" otherwise.
        """
        try:
            status = file_watcher.get_watcher_status()
            if status.get('global', {}).get('processing', False):
                return "processing"
        except Exception:
            pass
        return "idle"

    def _write_heartbeat(self):
        """
        Write the heartbeat JSON file to the configured path using an atomic replace.
        
        The file contains `timestamp` (current epoch time), `pid` (current process id), and `activity` (value from `_get_activity()`). Data is written to a temporary file in the target directory and then moved into place with `os.replace`. Failures are caught and logged as a warning.
        """
        data = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "activity": self._get_activity(),
        }
        tmp_path = self._path.parent / f".heartbeat.{os.getpid()}.tmp"
        try:
            tmp_path.write_text(json.dumps(data))
            os.replace(str(tmp_path), str(self._path))
        except Exception as e:
            logger.warning("Heartbeat write failed: %s", e)

    def _run(self):
        """
        Run the thread's main loop, writing the heartbeat file periodically until stopped.
        
        This method repeatedly writes a heartbeat and then waits for the configured interval (or until a stop is requested). It exits when the thread's stop event is set.
        """
        while not self._stop_event.is_set():
            self._write_heartbeat()
            self._stop_event.wait(self._interval)

    def start(self):
        """
        Start the heartbeat background thread.
        
        Initiates a daemon thread that runs the heartbeat loop until the thread is stopped via stop().
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Signal the heartbeat thread to stop and wait briefly for it to finish.
        
        If the heartbeat thread was started, this sets the stop event and blocks up to 2 seconds for the thread to join. If the thread was not started, this returns immediately.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# --- FTS heal state ---

@dataclass
class FtsHealState:
    """Worker-owned backoff + log-once state for the FTS heal loop (SESF-38 AC-2/AC-5).

    Implements the bounded-backoff schedule (``BACKFILL_DRAIN_INTERVAL * 2**(n-1)``
    capped at ``FTS_HEAL_BACKOFF_CAP``) and the log-once latch (signature =
    exception type + stable message prefix) so a persistent failure warns once per
    distinct signature. Owned by the heal worker in http_server. This worker state
    (``consecutive_failures`` / ``last_error``) is exposed only via ``/health``;
    ``rag_engine.get_stats`` carries the static lag fields (fts_row_count, fts_lag,
    fts_backfill_required) and never this worker state.
    """

    consecutive_failures: int = 0
    last_error_signature: str | None = None
    current_backoff_seconds: float = BACKFILL_DRAIN_INTERVAL

    @staticmethod
    def _signature(exc) -> str:
        """Derive a stable log-once signature from an exception.

        Combines the exception type name with a stable message prefix. The clean
        message payload is preferred (Milvus exceptions expose ``.message``), with
        ``str(exc)`` as the fallback. A trailing parenthetical suffix (request id,
        timestamp) is stripped so it does not churn the latch, while distinct
        message bodies and exception types remain distinct signatures.

        Args:
            exc: The exception raised by a heal attempt.

        Returns:
            A ``"<TypeName>:<stable message prefix>"`` signature string.
        """
        message = getattr(exc, "message", None)
        if not isinstance(message, str):
            message = str(exc)
        stable_prefix = re.sub(r"\s*\([^()]*\)\s*$", "", message).strip()
        return f"{type(exc).__name__}:{stable_prefix}"

    def record_failure(self, exc):
        """Record a transient heal failure and escalate the bounded backoff.

        Increments the consecutive-failure streak and sets the current backoff to
        ``BACKFILL_DRAIN_INTERVAL * 2**(n-1)`` capped at ``FTS_HEAL_BACKOFF_CAP``.
        Never raises and never signals a terminal "give up": the heal loop keeps
        retrying at the capped delay.

        Args:
            exc: The exception raised by the failed heal attempt.
        """
        self.consecutive_failures += 1
        raw = BACKFILL_DRAIN_INTERVAL * (2 ** (self.consecutive_failures - 1))
        self.current_backoff_seconds = min(raw, FTS_HEAL_BACKOFF_CAP)

    def record_success(self):
        """Reset backoff and log-once state after a successful heal.

        Zeroes the failure streak, restores the backoff to the base drain
        interval, and clears the error signature so a later failure warns again.
        """
        self.consecutive_failures = 0
        self.current_backoff_seconds = BACKFILL_DRAIN_INTERVAL
        self.last_error_signature = None

    def next_delay(self):
        """Return the next backoff delay in seconds.

        Yields the base drain interval when there have been no failures since the
        last reset, otherwise the current (escalated, capped) backoff.

        Returns:
            The delay in seconds before the next heal attempt.
        """
        if self.consecutive_failures == 0:
            return BACKFILL_DRAIN_INTERVAL
        return self.current_backoff_seconds

    def should_warn(self, exc):
        """Decide whether to emit a warning for ``exc`` under log-once dedup.

        Returns ``True`` on the first sighting of a signature and whenever the
        signature changes, ``False`` while the same signature repeats. Records the
        signature as a side effect so a repeat is suppressed.

        Args:
            exc: The exception raised by the failed heal attempt.

        Returns:
            ``True`` to log a warning, ``False`` to suppress a duplicate.
        """
        signature = self._signature(exc)
        if signature == self.last_error_signature:
            return False
        self.last_error_signature = signature
        return True


# --- Project middleware ---

class ProjectMiddleware:
    """ASGI middleware that extracts X-Project-Root header and sets ContextVar."""

    def __init__(self, app):
        """Wrap the downstream ASGI ``app``."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Set the project root from the X-Project-Root header, then delegate."""
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()
            set_current_project_root(project_root if project_root else None)

        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            logger.error("ASGI handler error: %s\n%s", exc, traceback.format_exc())
            if scope["type"] == "http":
                body = json.dumps({"error": "internal_server_error", "detail": str(exc)}).encode()
                await send({"type": "http.response.start", "status": 500, "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ]})
                await send({"type": "http.response.body", "body": body})


# --- Health endpoint ---

_model_loaded = False
_server_mode_ready = False
_BACKFILL_STATE = _SERVER_DIR / "backfill_state.json"
_backfill_manager = BackfillManager(_BACKFILL_STATE)
_backfill_drain_event: asyncio.Event | None = None
_backfill_drain_lock: asyncio.Lock | None = None

# FTS heal worker primitives (SESF-38). The heal loop reuses the drain cadence
# but owns a dedicated single-thread executor so a long blocking backfill never
# competes with MLX embedding work on rag_engine's _embed_executor.
_fts_heal_event: asyncio.Event | None = None
_fts_heal_state: "FtsHealState | None" = None
_fts_heal_executor: concurrent.futures.ThreadPoolExecutor | None = None

# TTL cache for provider health — avoid expensive I/O on every /health probe.
_PROVIDER_HEALTH_TTL = 30  # seconds
_provider_health_cache: dict | None = None
_provider_health_cache_ts: float = 0.0

# TTL cache for the FTS-lag block — a full Milvus count on every /health probe
# would be wasteful, so cache it for a few seconds (mirrors provider health).
_FTS_LAG_TTL = 5  # seconds
_fts_lag_cache: dict | None = None
_fts_lag_cache_ts: float = 0.0


def _ensure_backfill_drain_primitives() -> None:
    """Create loop-bound primitives lazily inside the running event loop."""
    global _backfill_drain_event, _backfill_drain_lock
    if _backfill_drain_event is None:
        _backfill_drain_event = asyncio.Event()
    if _backfill_drain_lock is None:
        _backfill_drain_lock = asyncio.Lock()


def _wake_backfill_drain() -> None:
    """Wake the background drain worker after queue-changing HTTP actions."""
    if _backfill_drain_event is not None:
        # Current callers are async HTTP handlers on the event-loop thread.
        _backfill_drain_event.set()


async def _drain_backfill_once(skip_if_locked: bool = True) -> dict:
    """Drain queued provider jobs once without overlapping another drain."""
    _ensure_backfill_drain_primitives()
    lock = _backfill_drain_lock
    if lock is None:
        return {"jobs": 0, "processed_sources": 0, "indexed_turns": 0, "errors": 0}
    if skip_if_locked and lock.locked():
        return {"jobs": 0, "processed_sources": 0, "indexed_turns": 0, "errors": 0, "skipped": 1}

    async with lock:
        status = _backfill_manager.status()
        if status.paused or not status.jobs:
            return {"jobs": 0, "processed_sources": 0, "indexed_turns": 0, "errors": 0}
        return await ProviderIngestionService(_backfill_manager, MILVUS_URI).process_queued_jobs()


async def _backfill_drain_worker(interval: float = BACKFILL_DRAIN_INTERVAL) -> None:
    """Poll and drain durable backfill jobs while the HTTP server is alive."""
    _ensure_backfill_drain_primitives()
    event = _backfill_drain_event
    if event is None:
        return
    while True:
        event.clear()
        try:
            await _drain_backfill_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Backfill drain worker failed")

        try:
            await asyncio.wait_for(event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _ensure_fts_heal_primitives() -> None:
    """Create the heal Event, state, and dedicated executor lazily.

    Mirrors ``_ensure_backfill_drain_primitives`` so the loop-bound ``asyncio``
    primitive and the heal worker's single-thread ThreadPoolExecutor are built
    inside the running event loop on first use. The executor is kept SEPARATE
    from rag_engine's MLX ``_embed_executor`` (SESF-38 D-8) so a blocking FTS
    backfill never contends with embedding work.
    """
    global _fts_heal_event, _fts_heal_state, _fts_heal_executor
    if _fts_heal_event is None:
        _fts_heal_event = asyncio.Event()
    if _fts_heal_state is None:
        _fts_heal_state = FtsHealState()
    if _fts_heal_executor is None:
        _fts_heal_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fts-heal"
        )


async def _fts_heal_run_once(state: "FtsHealState", *, first_tick: bool) -> None:
    """Make and attempt one FTS heal decision for a single cadence tick.

    On the first tick the backfill runs unconditionally (preserving the startup
    catch-up of the old one-shot ``_fts_backfill``); on later ticks it runs only
    when ``fts_hybrid.fts_backfill_required()`` reports the sentinel is set. The
    backfill itself is dispatched on the dedicated heal executor so it never
    blocks the event loop. A ``FtsBackfillTransientError`` is recorded on
    ``state`` (with a log-once WARN) instead of crashing the loop.

    Args:
        state: The worker-owned ``FtsHealState`` carrying backoff + log-once state.
        first_tick: ``True`` only on the very first tick of the heal loop.
    """
    if not first_tick and not fts_hybrid.fts_backfill_required():
        return

    _ensure_fts_heal_primitives()
    executor = _fts_heal_executor
    db_path = MILVUS_URI
    loop = asyncio.get_running_loop()
    try:
        backfilled = await loop.run_in_executor(
            executor, lambda: rag_engine.backfill_fts(db_path=db_path)
        )
    except rag_engine.FtsBackfillTransientError as exc:
        if state.should_warn(exc):
            logger.warning("FTS heal backfill failed (transient): %s", exc)
        state.record_failure(exc)
        return

    state.record_success()
    if backfilled:
        logger.info("FTS heal backfill: %s records", backfilled)


async def _fts_heal_worker() -> None:
    """Run the recurring FTS heal loop for the life of the HTTP server.

    Calls ``_fts_heal_run_once`` once per cadence tick — ``first_tick=True`` on
    the very first iteration only — then waits on the heal Event with a bounded
    timeout of ``state.next_delay()`` (the bounded-backoff schedule). The loop
    only ever exits on cancellation; transient failures are absorbed by
    ``_fts_heal_run_once`` and surfaced through the backoff schedule.
    """
    _ensure_fts_heal_primitives()
    event = _fts_heal_event
    state = _fts_heal_state
    if event is None or state is None:
        return
    first_tick = True
    while True:
        event.clear()
        try:
            await _fts_heal_run_once(state, first_tick=first_tick)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("FTS heal worker tick failed")
        first_tick = False

        try:
            await asyncio.wait_for(event.wait(), timeout=state.next_delay())
        except asyncio.TimeoutError:
            pass


async def _process_startup_provider_backfill_locked(
    db_path: str,
    enabled_providers: list[str],
    mode: str,
    startup_delay: float,
) -> dict:
    """Run startup provider backfill under the shared drain lock."""
    _ensure_backfill_drain_primitives()
    lock = _backfill_drain_lock
    if lock is None:
        return {}
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)
    async with lock:
        return await process_startup_provider_backfill(
            _backfill_manager,
            db_path,
            enabled_providers=enabled_providers,
            mode=mode,
            startup_delay=0,
        )


def _provider_health(deep: bool = False) -> dict:
    global _provider_health_cache, _provider_health_cache_ts
    now = time.monotonic()
    if not deep and _provider_health_cache is not None and (now - _provider_health_cache_ts) < _PROVIDER_HEALTH_TTL:
        return _provider_health_cache
    providers = [
        ClaudeCodeCliAdapter(),
        ClaudeDesktopCoworkProbe(),
        CodexAdapter(),
        OpenCodeAdapter(),
        AntigravityAdapter(source_kind="cli"),
        AntigravityAdapter(source_kind="desktop"),
    ]
    health = {}
    for provider in providers:
        try:
            status = provider.health()
            health[status.provider] = asdict(status)
        except Exception as exc:
            name = getattr(provider, "provider", provider.__class__.__name__)
            health[name] = {"provider": name, "status": "error", "error": str(exc)}
    _provider_health_cache = health
    _provider_health_cache_ts = now
    return health


def _backfill_status_payload() -> dict:
    status = _backfill_manager.status()
    return {
        "paused": status.paused,
        "jobs": [asdict(job) for job in status.jobs],
        "providers": {
            provider: asdict(provider_status)
            for provider, provider_status in status.providers.items()
        },
    }


def _embedding_status_payload() -> dict:
    try:
        identity = EmbeddingIdentity.current_local()
        identity_data = asdict(identity)
    except Exception as exc:
        identity_data = {"status": "error", "message": str(exc)}
    return {
        "identity": identity_data,
        "budget": get_embedding_budget().status(),
    }


def _fts_lag_payload() -> dict:
    """Return the FTS-vs-Milvus lag block for ``/health`` (SESF-38 AC-6).

    Combines the static, server-wide lag from ``rag_engine.fts_lag_status`` with
    the heal worker's standing-failure state (``consecutive_failures`` and the
    last error signature) from the module-global ``_fts_heal_state``. The whole
    computation is wrapped in try/except so a Milvus/FTS count failure under
    persistent drift returns an ``{"status": "error", ...}`` marker instead of
    raising — ``/health`` must never 500. The result is cached for a few seconds
    to avoid a full Milvus scan on every probe.

    Returns:
        dict: static lag keys (``milvus_turn_count``, ``fts_row_count``,
        ``fts_lag``, ``fts_backfill_required``) plus ``consecutive_failures`` and
        ``last_error``; or ``{"status": "error", "message": ...}`` on failure.
    """
    global _fts_lag_cache, _fts_lag_cache_ts
    now = time.monotonic()
    if _fts_lag_cache is not None and (now - _fts_lag_cache_ts) < _FTS_LAG_TTL:
        return _fts_lag_cache
    try:
        payload = dict(rag_engine.fts_lag_status(db_path=MILVUS_URI, project_root=None))
        state = _fts_heal_state
        payload["consecutive_failures"] = (
            state.consecutive_failures if state is not None else 0
        )
        payload["last_error"] = (
            state.last_error_signature if state is not None else None
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    _fts_lag_cache = payload
    _fts_lag_cache_ts = now
    return payload


async def health(request: Request) -> JSONResponse:
    """Return server health JSON; ``?deep=1`` bypasses the provider-status cache."""
    deep = request is not None and request.query_params.get("deep") == "1"
    watchers = file_watcher.get_watcher_status()
    # _fts_lag_payload does synchronous Milvus/SQLite I/O on a cache miss; offload
    # it to the loop's default executor so the scan never blocks the event loop.
    loop = asyncio.get_running_loop()
    fts_payload = await loop.run_in_executor(None, _fts_lag_payload)
    return JSONResponse({
        "status": "ok",
        "server": "sessionflow",
        "port": PORT,
        "model_name": rag_engine.get_model_name(),
        "model_loaded": _model_loaded,
        "milvus": _server_mode_ready,
        "milvus_backend": "standalone" if MILVUS_URI.startswith("http") else "lite",
        "watchers": {k: v for k, v in watchers.items()},
        "providers": _provider_health(deep=deep),
        "backfill": _backfill_status_payload(),
        "embedding": _embedding_status_payload(),
        "fts": fts_payload,
    })


async def backfill_control_endpoint(request: Request) -> JSONResponse:
    """Local provider-scoped backfill queue/status controls."""
    if request.method == "GET":
        return JSONResponse(_backfill_status_payload())

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    action = body.get("action", "status")
    provider = body.get("provider")
    if action == "pause":
        _backfill_manager.pause(provider=provider)
    elif action == "resume":
        _backfill_manager.resume(provider=provider)
    elif action == "enqueue":
        if not provider:
            return JSONResponse({"error": "provider is required for enqueue"}, status_code=400)
        mode = body.get("mode", "recent")
        _VALID_MODES = {"recent", "incremental", "full"}
        if mode not in _VALID_MODES:
            return JSONResponse(
                {"detail": f"Invalid mode {mode!r}. Must be one of: {sorted(_VALID_MODES)}"},
                status_code=400,
            )
        raw_limit = body.get("limit")
        limit: int | None = None
        if raw_limit is not None:
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"detail": "limit must be an integer"},
                    status_code=400,
                )
        try:
            priority = int(body.get("priority", 0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "priority must be an integer"}, status_code=400)
        _backfill_manager.enqueue_provider_backfill(
            provider=provider,
            mode=mode,
            limit=limit,
            since=body.get("since", ""),
            priority=priority,
        )
        _wake_backfill_drain()
    elif action == "run":
        # Hourly LaunchAgent entrypoint: enqueue every enabled provider in the
        # requested mode and drain the queue inline using the server's already
        # warmed-up MLX executor. Avoids the multi-process Metal risk of
        # spinning up a parallel MLX context from cleanup.py.
        from provider_adapters import LEGAL_PROVIDERS

        mode = body.get("mode", "incremental")
        _VALID_MODES = {"recent", "incremental", "full"}
        if mode not in _VALID_MODES:
            return JSONResponse(
                {"detail": f"Invalid mode {mode!r}. Must be one of: {sorted(_VALID_MODES)}"},
                status_code=400,
            )
        providers_arg = body.get("providers")
        if isinstance(providers_arg, str):
            providers = [p.strip() for p in providers_arg.split(",") if p.strip()]
        elif isinstance(providers_arg, list):
            providers = [str(p).strip() for p in providers_arg if str(p).strip()]
        else:
            providers = sorted(LEGAL_PROVIDERS - {"claude_desktop_cowork"})
        skipped: list[str] = []
        for provider in providers:
            try:
                _backfill_manager.enqueue_provider_backfill(provider=provider, mode=mode)
            except ValueError:
                skipped.append(provider)
        totals = await _drain_backfill_once(skip_if_locked=False)
        payload = _backfill_status_payload()
        payload["run"] = {
            "mode": mode,
            "providers": [p for p in providers if p not in skipped],
            "skipped": skipped,
            "totals": totals,
        }
        return JSONResponse(payload)
    elif action != "status":
        return JSONResponse({"error": f"Unknown backfill action: {action}"}, status_code=400)

    return JSONResponse(_backfill_status_payload())


# --- Index endpoint (called by hooks) ---

async def index_endpoint(request: Request) -> JSONResponse:
    """Index new turns from a transcript file.

    Expected JSON body:
        {
            "transcript_path": "/path/to/session.jsonl",
            "session_id": "uuid",
            "cwd": "/path/to/project"   (optional, fallback for project root)
        }

    Project root comes from X-Project-Root header (preferred) or cwd in body.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    transcript_path = body.get("transcript_path", "")
    session_id = body.get("session_id", "")

    if not transcript_path or not session_id:
        return JSONResponse(
            {"error": "transcript_path and session_id are required"},
            status_code=400,
        )

    if not os.path.exists(transcript_path):
        return JSONResponse(
            {"error": f"Transcript not found: {transcript_path}"},
            status_code=404,
        )

    # Determine project root from header or body
    headers = dict(request.scope.get("headers", []))
    project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()
    if not project_root:
        project_root = body.get("cwd", "")
    if not project_root:
        return JSONResponse(
            {"error": "Project root required (X-Project-Root header or cwd in body)"},
            status_code=400,
        )

    db_path = MILVUS_URI

    # Register slug→root mapping for the global watcher
    register_project(project_root)

    # Load centralized incremental state
    state = transcript_parser.load_index_state()
    offset = transcript_parser.get_transcript_offset(state, transcript_path)

    # Parse new turns
    turns, new_offset = transcript_parser.parse_transcript(
        transcript_path, session_id, start_offset=offset
    )

    if not turns:
        # Update offset even if no turns (e.g., only tool_result messages)
        transcript_parser.set_transcript_offset(
            state, transcript_path, new_offset, project_root=project_root)
        transcript_parser.save_index_state(state)
        return JSONResponse({"indexed": 0, "message": "No new turns to index"})

    # Inject project_root into each turn
    for t in turns:
        t["project_root"] = project_root

    # Index turns
    count = await rag_engine.add_turns_async(turns, db_path=db_path)

    # Save state
    transcript_parser.set_transcript_offset(
        state, transcript_path, new_offset, project_root=project_root)
    transcript_parser.save_index_state(state)

    print(f"[index] Indexed {count} turns from {os.path.basename(transcript_path)} "
          f"(session {session_id[:8]})", file=sys.stderr)

    # Auto-expiry: prune old turns once per day
    expired = 0
    if AUTO_EXPIRE_DAYS > 0:
        last_expire = state.get("last_expire_check", 0)
        now = time.time()
        if now - last_expire > _EXPIRE_CHECK_INTERVAL:
            expired = rag_engine.delete_older_than(AUTO_EXPIRE_DAYS, db_path=db_path)
            state["last_expire_check"] = now
            transcript_parser.save_index_state(state)
            if expired > 0:
                print(f"[expire] Pruned {expired} turns older than {AUTO_EXPIRE_DAYS} days",
                      file=sys.stderr)

    return JSONResponse({"indexed": count, "expired": expired, "session_id": session_id})


# --- Watch endpoint (register project + backfill) ---

async def watch_endpoint(request: Request) -> JSONResponse:
    """Register a project for file watching and trigger backfill.

    Called by SessionStart hook to ensure the watcher is running and
    any missed sessions are indexed.

    Project root comes from X-Project-Root header or JSON body.
    """
    # Determine project root from header
    headers = dict(request.scope.get("headers", []))
    project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()

    if not project_root:
        try:
            body = await request.json()
            project_root = body.get("project_root", "") or body.get("cwd", "")
        except Exception:
            pass

    if not project_root:
        return JSONResponse(
            {"error": "Project root required (X-Project-Root header or project_root in body)"},
            status_code=400,
        )

    # Register slug→root mapping
    register_project(project_root)

    # Trigger backfill for this project's slug dir
    backfilled = 0
    watcher = get_global_watcher()
    if watcher is not None:
        from file_watcher import _project_root_to_slug
        slug = _project_root_to_slug(project_root)
        backfilled = await watcher.backfill(slug_filter=slug)

    # Auto-expiry check
    expired = 0
    if AUTO_EXPIRE_DAYS > 0:
        db_path = MILVUS_URI
        state = transcript_parser.load_index_state()
        last_expire = state.get("last_expire_check", 0)
        now = time.time()
        if now - last_expire > _EXPIRE_CHECK_INTERVAL:
            expired = rag_engine.delete_older_than(AUTO_EXPIRE_DAYS, db_path=db_path)
            state["last_expire_check"] = now
            transcript_parser.save_index_state(state)
            if expired > 0:
                print(f"[expire] Pruned {expired} turns older than {AUTO_EXPIRE_DAYS} days",
                      file=sys.stderr)

    return JSONResponse({
        "watching": project_root,
        "backfilled": backfilled,
        "expired": expired,
    })


# --- Timeline endpoint ---

async def timeline_endpoint(request: Request) -> JSONResponse:
    """Cross-harness issue timeline over HTTP (mirrors the MCP get_issue_timeline tool).

    Reads the same query params as the MCP tool (``issue_id`` required, plus
    optional ``limit``, ``provider``, ``date_from``, ``date_to``) and returns the
    engine's deduplicated, chronological feed as a JSON ``{"timeline": [...]}``
    envelope. The singular ``provider`` param maps to the engine's ``providers``
    list. Same matching/sort/limit logic as the MCP transport (Req 5.3).
    """
    params = request.query_params
    issue_id = params.get("issue_id")
    if not issue_id or not issue_id.strip():
        return JSONResponse({"error": "issue_id is required"}, status_code=400)
    if not is_valid_issue_token(issue_id):
        return JSONResponse(
            {"error": "issue_id must be a valid issue token like SESF-25"},
            status_code=400,
        )

    raw_limit = params.get("limit")
    limit = 50
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)
        if limit < 1:
            return JSONResponse({"error": "limit must be a positive integer"}, status_code=400)

    provider = params.get("provider")
    if provider and provider not in LEGAL_PROVIDERS:
        allowed = ", ".join(sorted(LEGAL_PROVIDERS))
        return JSONResponse(
            {"error": f"invalid provider: {provider}; expected one of: {allowed}"},
            status_code=400,
        )
    entries = await rag_engine.get_issue_timeline_async(
        issue_id,
        limit=limit,
        providers=[provider] if provider else None,
        date_from=params.get("date_from"),
        date_to=params.get("date_to"),
        db_path=MILVUS_URI,
    )
    return JSONResponse({"timeline": entries})


# --- Lifespan ---

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """
    Manage server startup and shutdown tasks for the Starlette application.
    
    On startup: ensure the server directory exists, write the PID file, attempt to preload the embedding model and initialize server mode for the shared Milvus backend, start a periodic heartbeat thread, schedule a background full-text-search backfill, and start the global file watcher (which may schedule its own backfill). The context yields control while the server is running.
    
    On shutdown: stop the heartbeat thread and remove the heartbeat file only if it was written by this process, stop the global file watcher, close server mode, remove the PID file, and perform any remaining cleanup.
    """
    _SERVER_DIR.mkdir(parents=True, exist_ok=True)

    PID_FILE.write_text(str(os.getpid()))
    print(f"[HTTP] PID {os.getpid()} written to {PID_FILE}", file=sys.stderr)

    global _model_loaded, _server_mode_ready, _backfill_drain_event, _backfill_drain_lock
    global _fts_heal_event, _fts_heal_state, _fts_heal_executor

    # Pre-load embedding model
    try:
        model_name = rag_engine.get_model_name()
        print(f"[HTTP] Pre-loading {model_name} model...", file=sys.stderr)
        rag_engine.get_model()
        _model_loaded = True
        print(f"[HTTP] {model_name} model loaded.", file=sys.stderr)
    except Exception as e:
        print(f"[HTTP] Warning: Could not pre-load model: {e}", file=sys.stderr)

    db_path = MILVUS_URI
    try:
        rag_engine.init_server_mode(db_path=db_path)
        _server_mode_ready = True
    except Exception as e:
        print(f"[HTTP] Warning: Could not init server mode: {e}", file=sys.stderr)

    heartbeat = HeartbeatThread(HEARTBEAT_FILE)
    heartbeat.start()
    print(f"[HTTP] Heartbeat thread started ({HEARTBEAT_FILE})", file=sys.stderr)

    _ensure_backfill_drain_primitives()
    backfill_drain_task = asyncio.create_task(_backfill_drain_worker())

    # Recurring FTS heal worker (SESF-38) — replaces the old one-shot
    # _fts_backfill. The first tick runs the startup catch-up backfill
    # unconditionally; later ticks heal only when the sentinel reports drift.
    # Runs on a dedicated executor so it never blocks HTTP server binding.
    _ensure_fts_heal_primitives()
    fts_heal_task = asyncio.create_task(_fts_heal_worker())

    # Start global file watcher on ~/.claude/projects/
    try:
        watcher = await file_watcher.start_global_watcher(db_path)
        if watcher:
            enabled_providers = [
                "claude_code_cli",
                "codex",
                "opencode",
                "antigravity_cli",
                "antigravity_desktop",
            ]
            mode = get_embedding_budget().mode
            # Delay lets HTTP server finish binding before bounded provider work starts.
            asyncio.create_task(_process_startup_provider_backfill_locked(
                db_path,
                enabled_providers=enabled_providers,
                mode=mode,
                startup_delay=3,
            ))
    except Exception as e:
        print(f"[HTTP] Warning: Global watcher start failed: {e}", file=sys.stderr)

    async with session_manager.run():
        print(f"[HTTP] Server ready on http://{HOST}:{PORT}", file=sys.stderr)
        try:
            yield
        finally:
            pass

    heartbeat.stop()
    backfill_drain_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await backfill_drain_task
    _backfill_drain_event = None
    _backfill_drain_lock = None

    # Tear down the FTS heal worker: cancel the loop, await any in-flight heal
    # with a bounded timeout, then shut down the dedicated executor.
    fts_heal_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(fts_heal_task, timeout=BACKFILL_DRAIN_INTERVAL)
    if _fts_heal_executor is not None:
        _fts_heal_executor.shutdown(wait=False)
    _fts_heal_event = None
    _fts_heal_state = None
    _fts_heal_executor = None
    if HEARTBEAT_FILE.exists():
        try:
            data = json.loads(HEARTBEAT_FILE.read_text())
            if data.get("pid") == os.getpid():
                HEARTBEAT_FILE.unlink()
        except Exception:
            pass

    await file_watcher.stop_global_watcher()
    rag_engine.close_server_mode()
    if PID_FILE.exists():
        PID_FILE.unlink()
    print("[HTTP] Server stopped.", file=sys.stderr)


# --- MCP server setup ---

mcp_server = Server(
    "sessionflow",
    instructions=(
        "SessionFlow provides semantic search over conversation history across all "
        "harnesses (Claude Code, Codex, OpenCode, Antigravity). To recall recent context "
        "for the project you are working in, use search_all_sessions — when project "
        "context is available (the client's X-Project-Root header) it scopes to that "
        "project by default; pass project_root='*' to search every project. Omit "
        "'query' to list the most recent turns chronologically. Results are 'hybrid'-"
        "ranked (semantic relevance blended with recency) by default; pass sort_by="
        "'relevance' or 'recency' to change that. To expand a specific hit, call get_turns "
        "with the session_id and turn_index from a result. There is no automatic 'current "
        "session' filter — MCP clients do not expose the live session ID to the server — "
        "so pass session_id (from a prior result) only when you want to pin one conversation."
    ),
)
register_tools(mcp_server)

session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    stateless=True,
    json_response=True,
)


# --- Starlette app ---

app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/backfill", backfill_control_endpoint, methods=["GET", "POST"]),
        Route("/index", index_endpoint, methods=["POST"]),
        Route("/watch", watch_endpoint, methods=["POST"]),
        Route("/timeline", timeline_endpoint, methods=["GET"]),
        Mount("/mcp", app=ProjectMiddleware(session_manager.handle_request)),
    ],
    lifespan=lifespan,
)


# --- Signal handling ---

def _handle_signal(signum, frame):
    print(f"[HTTP] Received signal {signum}, shutting down...", file=sys.stderr)
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
