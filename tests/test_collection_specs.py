"""Generic collection plumbing: every Milvus helper works from a CollectionSpec."""

import json
from unittest.mock import MagicMock

import collection_specs


def _sig(field):
    return (field.name, field.dtype, dict(field.params), field.is_primary, field.auto_id)


def test_sessions_spec_matches_current_schema():
    """The sessions spec reproduces today's schema, so transcripts see no change."""
    import rag_engine

    spec = collection_specs.sessions_spec(rag_engine._EMBED_DIM)
    assert [_sig(f) for f in spec.field_schemas()] == [
        _sig(f) for f in rag_engine._expected_schema_fields()
    ]
    assert spec.index_spec(remote=True) == {
        "field_name": "vector",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256},
    }
    assert spec.consistency_level is None


def test_rag_engine_exposes_both_specs():
    import rag_engine

    assert rag_engine.SESSIONS_SPEC == collection_specs.sessions_spec(rag_engine._EMBED_DIM)
    assert rag_engine.MEMORIES_SPEC == collection_specs.memories_spec(rag_engine._EMBED_DIM)


def test_memories_spec_holds_no_text():
    spec = collection_specs.memories_spec(768)
    names = spec.field_names()
    assert "document" not in names and "text" not in names
    assert {"memory_id", "version", "generation", "project", "type", "status"} <= set(names)
    assert spec.consistency_level == "Strong"


def test_ensure_collection_creates_memories_with_hnsw_and_strong(monkeypatch):
    import rag_engine

    client = MagicMock()
    client.has_collection.return_value = False
    index_params = MagicMock()
    client.prepare_index_params.return_value = index_params

    rag_engine._ensure_collection(
        client, db_path="http://milvus.test:19530", spec=rag_engine.MEMORIES_SPEC
    )

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "memories"
    assert [f.name for f in kwargs["schema"].fields] == rag_engine.MEMORIES_SPEC.field_names()
    assert kwargs["consistency_level"] == "Strong"
    index_kwargs = index_params.add_index.call_args.kwargs
    assert index_kwargs["index_type"] == "HNSW"
    assert index_kwargs["metric_type"] == "COSINE"
    client.load_collection.assert_called_once_with(collection_name="memories")
    client.has_collection.assert_called_with("memories")


def test_ensure_collection_default_still_targets_sessions():
    """The existing call form keeps meaning the sessions collection."""
    import rag_engine

    client = MagicMock()
    client.has_collection.return_value = False
    client.prepare_index_params.return_value = MagicMock()

    rag_engine._ensure_collection(client, db_path="/tmp/whatever.db")

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "sessions"
    assert "consistency_level" not in kwargs or kwargs["consistency_level"] is None


def test_migrate_to_standalone_builds_the_sessions_spec():
    """The Lite-to-Standalone script creates the full current schema, not its own copy."""
    import migrate_to_standalone
    import rag_engine

    client = MagicMock()
    client.has_collection.return_value = False
    client.prepare_index_params.return_value = MagicMock()

    migrate_to_standalone.create_collection(client)

    schema = client.create_collection.call_args.kwargs["schema"]
    assert [f.name for f in schema.fields] == rag_engine.SESSIONS_SPEC.field_names()


def test_model_identity_stamp_records_the_collection(tmp_path, monkeypatch):
    import rag_engine

    stamp = tmp_path / "model_identity.json"
    monkeypatch.setattr(rag_engine, "_IDENTITY_FILE", stamp)

    rag_engine._check_model_identity()

    data = json.loads(stamp.read_text())
    assert data["model_name"] == rag_engine._MODEL_NAME
    assert data["collection"] == "sessions"
