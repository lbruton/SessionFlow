"""The persistent Milvus client cache stays consistent across threads.

Query-less listings run on the default executor, so several threads can reach
``_get_persistent_client`` at once.
"""

import importlib
import threading

rag_engine = importlib.import_module("rag_engine")

PATH = "http://milvus.test:19530"


class _FakeClient:
    created = []

    def __init__(self, *args, **kwargs):
        type(self).created.append(self)
        self.closed = 0

    def has_collection(self, name):
        return True

    def close(self):
        self.closed += 1


def _patch(monkeypatch, clients=None):
    _FakeClient.created = []
    monkeypatch.setattr(rag_engine, "MilvusClient", _FakeClient)
    monkeypatch.setattr(rag_engine, "_persistent_clients", dict(clients or {}))


def _in_threads(count):
    results, errors = [], []

    def run():
        try:
            results.append(rag_engine._get_persistent_client(PATH))
        except Exception as exc:  # noqa: BLE001 - a race surfaces as any exception
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    return results, errors


class _StaleClient(_FakeClient):
    def __init__(self, barrier):
        self.barrier = barrier
        self.closed = 0

    def has_collection(self, name):
        self.barrier.wait(5)
        raise ConnectionError("stale")


def test_two_threads_evict_a_stale_client_once(monkeypatch):
    stale = _StaleClient(threading.Barrier(2))
    _patch(monkeypatch, {PATH: stale})

    results, errors = _in_threads(2)

    assert errors == []
    assert stale.closed == 1
    assert len(_FakeClient.created) == 1
    assert results == [_FakeClient.created[0]] * 2
    assert rag_engine._persistent_clients[PATH] is _FakeClient.created[0]


def test_concurrent_first_opens_create_one_client(monkeypatch):
    _patch(monkeypatch)
    barrier = threading.Barrier(4)
    built = []

    class _SlowClient(_FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(rag_engine, "MilvusClient", _SlowClient)
    real_is_remote = rag_engine._is_remote_uri

    def gate(path):
        try:
            barrier.wait(0.2)
        except threading.BrokenBarrierError:
            pass
        return real_is_remote(path)

    monkeypatch.setattr(rag_engine, "_is_remote_uri", gate)

    results, errors = _in_threads(4)

    assert errors == []
    assert len(built) == 1
    assert results == built * 4


def test_healthy_client_is_reused(monkeypatch):
    client = _FakeClient()
    _FakeClient.created = []
    _patch(monkeypatch, {PATH: client})
    assert rag_engine._get_persistent_client(PATH) is client
    assert _FakeClient.created == []


def test_failed_open_leaves_no_entry(monkeypatch):
    _patch(monkeypatch)

    def refuse(*args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(rag_engine, "MilvusClient", refuse)
    results, errors = _in_threads(1)
    assert results == []
    assert isinstance(errors[0], ConnectionError)
    assert PATH not in rag_engine._persistent_clients
