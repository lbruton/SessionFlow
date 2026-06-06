"""SESF-40: pymilvus data-path schema cache invalidation on schema mutation.

pymilvus 2.6 caches collection schemas in a process-global, class-level LRU
(GlobalCache.schema) that drop_collection invalidates but add_collection_field
does not. These tests exercise rag_engine._invalidate_schema_cache against the
real GlobalCache (no live Milvus) and verify migrate_schema funnels through it.
"""

from unittest.mock import MagicMock

import pytest

# These tests reach into pymilvus internals, but rag_engine._invalidate_schema_cache
# is designed to degrade gracefully when those internals move. requirements.txt
# pins no pymilvus upper bound, so guard the import + the test-only reset hook and
# skip the whole module (rather than erroring at collection) if either is gone.
try:
    from pymilvus.client.cache import GlobalCache

    if not hasattr(GlobalCache, "_reset_for_testing"):
        raise ImportError("GlobalCache._reset_for_testing unavailable")
except Exception as exc:  # pragma: no cover - pymilvus internals moved
    pytest.skip(
        f"pymilvus schema-cache internals unavailable ({exc})",
        allow_module_level=True,
    )

_ENDPOINT = "test-endpoint:19530"


@pytest.fixture(autouse=True)
def _reset_global_cache():
    """Isolate the process-global schema cache around each test."""
    GlobalCache._reset_for_testing()
    yield
    GlobalCache._reset_for_testing()


def _client_with_endpoint(endpoint: str) -> MagicMock:
    """Mock client whose _get_connection().server_address returns endpoint."""
    client = MagicMock()
    client._get_connection.return_value.server_address = endpoint
    return client


def test_invalidate_clears_warm_entry():
    """A schema cached under the client's endpoint is evicted by the helper."""
    import rag_engine

    GlobalCache.schema.set(_ENDPOINT, "", rag_engine.COLLECTION_NAME, {"fields": []})
    assert GlobalCache.schema.get(_ENDPOINT, "", rag_engine.COLLECTION_NAME) is not None

    rag_engine._invalidate_schema_cache(_client_with_endpoint(_ENDPOINT))

    assert GlobalCache.schema.get(_ENDPOINT, "", rag_engine.COLLECTION_NAME) is None


def test_invalidate_degrades_gracefully():
    """Defense-in-depth: a broken client connection must not raise out of the helper."""
    import rag_engine

    client = MagicMock()
    client._get_connection.side_effect = RuntimeError("no connection")

    # Must not raise — the helper is best-effort and never breaks a real migration.
    rag_engine._invalidate_schema_cache(client)


def test_migrate_schema_invalidates_cache(monkeypatch):
    """add-field-then-insert analog: migrate_schema funnels through invalidation.

    Seeds a stale schema entry (simulating a warm post-mutation cache), runs
    migrate_schema with create/drop stubbed out, and asserts the entry is gone.
    """
    import rag_engine

    GlobalCache.schema.set(_ENDPOINT, "", rag_engine.COLLECTION_NAME, {"fields": []})

    client = _client_with_endpoint(_ENDPOINT)
    client.has_collection.return_value = False  # skip the drop branch

    monkeypatch.setattr(rag_engine, "_create_collection", lambda c, db_path="": None)

    rag_engine.migrate_schema(client, db_path="/tmp/whatever.db")

    assert GlobalCache.schema.get(_ENDPOINT, "", rag_engine.COLLECTION_NAME) is None
