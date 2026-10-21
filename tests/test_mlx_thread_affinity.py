"""SESF-72 tests: in server mode every MLX eval runs on the one mlx-embed thread.

MLX's Metal command encoder is not thread-safe. A search tool that embedded on
the event-loop thread while backfill embedded on ``mlx-embed`` crashed the
server with SIGSEGV in ``CommandEncoder::dispatch_threads``.
"""

import asyncio
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")


@pytest.fixture
def mlx_executor(monkeypatch):
    """Install a real single-worker mlx-embed executor, as init_server_mode does."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-embed")
    monkeypatch.setattr(rag_engine, "_embed_executor", executor)
    yield executor
    executor.shutdown(wait=True)


def _record_thread(monkeypatch):
    seen = []

    def fake_local(texts, is_query=False):
        seen.append(threading.current_thread().name)
        return [[0.0] for _ in texts]

    monkeypatch.setattr(rag_engine, "_embed_texts_local", fake_local)
    return seen


def test_embed_from_other_thread_runs_on_mlx_embed(monkeypatch, mlx_executor):
    seen = _record_thread(monkeypatch)
    worker = threading.Thread(target=rag_engine.embed_texts, args=(["q"],), name="caller")
    worker.start()
    worker.join()
    rag_engine.embed_texts(["q"], is_query=True)  # main thread
    assert len(seen) == 2
    assert all(name.startswith("mlx-embed") for name in seen)


def test_embed_on_mlx_embed_thread_runs_inline(monkeypatch, mlx_executor):
    """No self-submit deadlock when backfill embeds from the executor itself."""
    seen = _record_thread(monkeypatch)
    out = mlx_executor.submit(rag_engine.embed_texts, ["d"]).result(timeout=5)
    assert out == [[0.0]]
    assert seen and seen[0].startswith("mlx-embed")


def test_embed_without_server_mode_runs_inline(monkeypatch):
    monkeypatch.setattr(rag_engine, "_embed_executor", None)
    seen = _record_thread(monkeypatch)
    rag_engine.embed_texts(["q"])
    assert seen == [threading.current_thread().name]


def test_search_async_forwards_all_filters(monkeypatch, mlx_executor):
    captured = {}

    def fake_search(query, n=5, **kwargs):
        captured["thread"] = threading.current_thread().name
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(rag_engine, "search", fake_search)

    async def run():
        monkeypatch.setattr(rag_engine, "_embed_semaphore", asyncio.Semaphore(1))
        return await rag_engine.search_async(
            "q", 3, date_from="2026-10-01", date_to="2026-10-20", issue_id="SESF-72",
        )

    asyncio.run(run())
    assert captured["thread"].startswith("mlx-embed")
    assert captured["kwargs"]["date_from"] == "2026-10-01"
    assert captured["kwargs"]["date_to"] == "2026-10-20"
    assert captured["kwargs"]["issue_id"] == "SESF-72"


def _call_tool():
    captured = {}

    class _Server:
        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            def deco(fn):
                captured["call"] = fn
                return fn
            return deco

    tools.register_tools(_Server())
    return captured["call"]


@pytest.mark.parametrize("tool_name", ["search_session", "search_all_sessions"])
def test_search_tools_do_not_search_on_event_loop(monkeypatch, tool_name):
    calls = {}

    def sync_search(*args, **kwargs):
        raise AssertionError("tool called the blocking search() on the event loop")

    async def fake_search_async(query, n=5, **kwargs):
        calls["kwargs"] = kwargs
        return []

    monkeypatch.setattr(tools.rag_engine, "search", sync_search)
    monkeypatch.setattr(tools.rag_engine, "search_async", fake_search_async)

    call = _call_tool()
    args = {"query": "q", "session_id": "s1", "issue_id": "SESF-72"}
    if tool_name == "search_all_sessions":
        args.update(date_from="2026-10-01", project_root="*")
    result = asyncio.run(call(tool_name, args))

    assert "Error executing" not in result[0].text
    assert calls["kwargs"]["issue_id"] == "SESF-72"
    if tool_name == "search_all_sessions":
        assert calls["kwargs"]["date_from"] == "2026-10-01"
