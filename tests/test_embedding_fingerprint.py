"""Embedding fingerprint and the transcript stamp that records it.

The fingerprint names everything that shapes a stored vector. A fingerprint
change warns and shows in /health; only a model or dimension change refuses.
Milvus Standalone keeps the stamp in collection properties so every host
sharing it sees one stamp; Milvus Lite keeps it in the local identity file.
"""

from __future__ import annotations

import json
import logging

import pytest

import rag_engine


def _cache(tmp_path, revision="rev1", tokenizer=b'{"vocab": 1}'):
    cache = tmp_path / "models--x--y"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(revision)
    snap = cache / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "tokenizer.json").write_bytes(tokenizer)
    return cache


def _fp(tmp_path, cfg_over=None, impl_version=1, **cache_kw):
    cfg = dict(rag_engine._MODEL_REGISTRY["embeddinggemma"], **(cfg_over or {}))
    return rag_engine.embedding_fingerprint(cfg=cfg, cache=_cache(tmp_path, **cache_kw),
                                            impl_version=impl_version)


# --- The fingerprint ----------------------------------------------------------


def test_fingerprint_names_every_input(tmp_path):
    fp = _fp(tmp_path)

    assert fp["model_id"] == "mlx-community/embeddinggemma-300m-bf16"
    assert fp["revision"] == "rev1"
    assert len(fp["tokenizer_sha256"]) == 64
    assert fp["search_prefix"] == "task: search result | query: "
    assert fp["document_prefix"] == "title: none | text: "
    assert fp["pooling"] == "mean"
    assert fp["max_tokens"] == 2048
    assert fp["embed_dim"] == 768
    assert fp["impl_version"] == 1


def test_fingerprint_is_stable(tmp_path):
    a = _fp(tmp_path / "a")
    b = _fp(tmp_path / "b")

    assert a == b
    assert rag_engine.fingerprint_hash(a) == rag_engine.fingerprint_hash(b)


@pytest.mark.parametrize("change, field", [
    ({"cache_kw": {"revision": "rev2"}}, "revision"),
    ({"cache_kw": {"tokenizer": b'{"vocab": 2}'}}, "tokenizer_sha256"),
    ({"cfg_over": {"search_prefix": "query: "}}, "search_prefix"),
    ({"cfg_over": {"document_prefix": "text: "}}, "document_prefix"),
    ({"cfg_over": {"pooling": "cls"}}, "pooling"),
    ({"cfg_over": {"max_tokens": 1024}}, "max_tokens"),
    ({"cfg_over": {"model_id": "google/embeddinggemma-300m"}}, "model_id"),
    ({"cfg_over": {"embed_dim": 512}}, "embed_dim"),
    ({"impl_version": 2}, "impl_version"),
])
def test_fingerprint_changes_with_each_input(tmp_path, change, field):
    base = _fp(tmp_path / "base")
    other = _fp(tmp_path / "other", cfg_over=change.get("cfg_over"),
                impl_version=change.get("impl_version", 1), **change.get("cache_kw", {}))

    assert rag_engine.fingerprint_hash(base) != rag_engine.fingerprint_hash(other)
    assert rag_engine.fingerprint_drift(base, other) == [field]


def test_missing_cache_gives_empty_revision_and_tokenizer(tmp_path):
    cfg = rag_engine._MODEL_REGISTRY["embeddinggemma"]
    fp = rag_engine.embedding_fingerprint(cfg=cfg, cache=tmp_path / "absent", impl_version=1)

    assert fp["revision"] is None
    assert fp["tokenizer_sha256"] is None


def test_drift_is_empty_without_a_stored_fingerprint(tmp_path):
    assert rag_engine.fingerprint_drift(None, _fp(tmp_path)) == []


def test_every_registry_model_names_its_pooling():
    assert all(cfg.get("pooling") for cfg in rag_engine._MODEL_REGISTRY.values())


# --- The transcript stamp -----------------------------------------------------


CURRENT = {
    "model_id": "mlx-community/embeddinggemma-300m-bf16", "revision": "new",
    "tokenizer_sha256": "a" * 64, "search_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: ", "pooling": "mean", "max_tokens": 2048,
    "embed_dim": 768, "impl_version": 1,
}


class FakeStandalone:
    """Just enough of MilvusClient for the sessions collection properties."""

    def __init__(self, properties=None, exists=True):
        self.properties = dict(properties or {})
        self.exists = exists
        self.altered = []

    def has_collection(self, name):
        return self.exists

    def describe_collection(self, name):
        return {"collection_name": name, "properties": dict(self.properties)}

    def alter_collection_properties(self, collection_name, properties):
        self.altered.append(properties)
        self.properties.update(properties)

    def close(self):
        pass


@pytest.fixture
def stamp_env(tmp_path, monkeypatch):
    identity = tmp_path / "model_identity.json"
    monkeypatch.setattr(rag_engine, "_IDENTITY_FILE", identity)
    monkeypatch.setattr(rag_engine, "current_fingerprint", lambda: dict(CURRENT))
    monkeypatch.setattr(rag_engine, "_index_has_data", lambda db_path: True)
    monkeypatch.setattr(rag_engine, "_stamp_status", None)
    monkeypatch.setattr(rag_engine, "_stamp_db_path", None)
    return identity


def _stamp(fingerprint=None, model_name="embeddinggemma"):
    stamp = {"model_name": model_name, "collection": "sessions"}
    if fingerprint is not None:
        stamp["fingerprint"] = fingerprint
    return stamp


def test_lite_stamp_writes_the_identity_file(stamp_env, tmp_path):
    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    stored = json.loads(stamp_env.read_text())
    assert stored["model_name"] == "embeddinggemma"
    assert stored["fingerprint"] == CURRENT
    assert rag_engine.embedding_stamp_status()["drift"] == []


def test_fingerprint_mismatch_warns_and_does_not_refuse(stamp_env, tmp_path, caplog):
    old = dict(CURRENT, revision="old", impl_version=0)
    stamp_env.write_text(json.dumps(_stamp(old)))

    with caplog.at_level(logging.WARNING, logger=rag_engine.logger.name):
        rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    status = rag_engine.embedding_stamp_status()
    assert status["drift"] == ["revision", "impl_version"]
    assert status["stored_hash"] == rag_engine.fingerprint_hash(old)
    assert status["hash"] == rag_engine.fingerprint_hash(CURRENT)
    assert "revision" in caplog.text
    # The stamp describes the stored vectors, so it keeps the old fingerprint.
    assert json.loads(stamp_env.read_text())["fingerprint"] == old


def test_drift_on_an_empty_index_restamps_the_current_fingerprint(stamp_env, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "_index_has_data", lambda db_path: False)
    stamp_env.write_text(json.dumps(_stamp(dict(CURRENT, revision="old"))))

    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert json.loads(stamp_env.read_text()) == _stamp(CURRENT)
    assert rag_engine.embedding_stamp_status()["drift"] == []


def test_model_name_mismatch_still_refuses(stamp_env, tmp_path):
    stamp_env.write_text(json.dumps(_stamp(CURRENT, model_name="modernbert")))

    with pytest.raises(RuntimeError, match="Model mismatch"):
        rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))


@pytest.mark.parametrize("field, value", [
    ("model_id", "google/embeddinggemma-300m"),
    ("embed_dim", 512),
])
def test_model_id_or_dimension_mismatch_refuses(stamp_env, tmp_path, field, value):
    stamp_env.write_text(json.dumps(_stamp(dict(CURRENT, **{field: value}))))

    with pytest.raises(RuntimeError, match="Model mismatch"):
        rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))


def test_model_mismatch_on_an_empty_index_restamps(stamp_env, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "_index_has_data", lambda db_path: False)
    stamp_env.write_text(json.dumps(_stamp(CURRENT, model_name="modernbert")))

    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert json.loads(stamp_env.read_text()) == _stamp(CURRENT)


def test_legacy_stamp_without_fingerprint_adopts_the_current_one(stamp_env, tmp_path):
    stamp_env.write_text(json.dumps(_stamp()))

    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert json.loads(stamp_env.read_text())["fingerprint"] == CURRENT
    assert rag_engine.embedding_stamp_status()["drift"] == []


def test_standalone_stamp_lives_in_collection_properties(stamp_env, monkeypatch):
    client = FakeStandalone()
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)

    rag_engine._check_model_identity(db_path="http://milvus:19530")

    stored = json.loads(client.properties[rag_engine.STAMP_PROPERTY])
    assert stored == _stamp(CURRENT)
    assert not stamp_env.exists()


def test_standalone_reads_the_shared_stamp_not_the_local_file(stamp_env, monkeypatch):
    old = dict(CURRENT, tokenizer_sha256="b" * 64)
    client = FakeStandalone({rag_engine.STAMP_PROPERTY: json.dumps(_stamp(old))})
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)
    stamp_env.write_text(json.dumps(_stamp(CURRENT, model_name="modernbert")))

    rag_engine._check_model_identity(db_path="http://milvus:19530")

    assert rag_engine.embedding_stamp_status()["drift"] == ["tokenizer_sha256"]
    assert client.altered == []


def test_standalone_without_a_property_falls_back_to_the_local_file(stamp_env, monkeypatch):
    client = FakeStandalone()
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)
    stamp_env.write_text(json.dumps(_stamp(model_name="modernbert")))

    with pytest.raises(RuntimeError, match="Model mismatch"):
        rag_engine._check_model_identity(db_path="http://milvus:19530")


def test_standalone_without_the_collection_leaves_the_stamp_to_creation(stamp_env, monkeypatch):
    client = FakeStandalone(exists=False)
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)

    rag_engine._check_model_identity(db_path="http://milvus:19530")

    assert client.altered == []
    assert rag_engine.embedding_stamp_status()["drift"] == []


def test_write_embedding_stamp_records_the_current_fingerprint(stamp_env, monkeypatch):
    client = FakeStandalone({rag_engine.STAMP_PROPERTY: json.dumps(_stamp(dict(CURRENT, revision="old")))})
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)
    rag_engine._check_model_identity(db_path="http://milvus:19530")
    assert rag_engine.embedding_stamp_status()["drift"] == ["revision"]

    rag_engine.write_embedding_stamp(db_path="http://milvus:19530")

    assert json.loads(client.properties[rag_engine.STAMP_PROPERTY]) == _stamp(CURRENT)
    assert rag_engine.embedding_stamp_status()["drift"] == []


def test_creating_the_standalone_sessions_collection_stamps_it(stamp_env, monkeypatch):
    class Creating(FakeStandalone):
        def prepare_index_params(self):
            class P:
                def add_index(self, **kwargs):
                    pass
            return P()

        def create_collection(self, **kwargs):
            self.exists = True

        def load_collection(self, collection_name):
            pass

    client = Creating(exists=False)
    rag_engine._create_collection(client, "http://milvus:19530")

    assert json.loads(client.properties[rag_engine.STAMP_PROPERTY]) == _stamp(CURRENT)


def test_standalone_refuses_when_another_host_wins_the_first_stamp(stamp_env, monkeypatch):
    other = json.dumps(_stamp(dict(CURRENT, model_id="google/embeddinggemma-300m"), model_name="modernbert"))

    class Racing(FakeStandalone):
        def alter_collection_properties(self, collection_name, properties):
            self.altered.append(properties)
            self.properties[rag_engine.STAMP_PROPERTY] = other

    client = Racing()
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)

    with pytest.raises(RuntimeError, match="Model mismatch"):
        rag_engine._check_model_identity(db_path="http://milvus:19530")


def test_status_refresh_reads_a_stamp_written_by_another_process(stamp_env, monkeypatch):
    client = FakeStandalone({rag_engine.STAMP_PROPERTY: json.dumps(_stamp(dict(CURRENT, revision="old")))})
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)
    rag_engine._check_model_identity(db_path="http://milvus:19530")
    assert rag_engine.embedding_stamp_status()["drift"] == ["revision"]

    client.properties[rag_engine.STAMP_PROPERTY] = json.dumps(_stamp(CURRENT))
    rag_engine._refresh_stamp_status()

    assert rag_engine.embedding_stamp_status()["drift"] == []
    assert client.altered == []


def test_stale_status_schedules_a_refresh_and_answers_at_once(stamp_env, monkeypatch):
    client = FakeStandalone({rag_engine.STAMP_PROPERTY: json.dumps(_stamp(dict(CURRENT, revision="old")))})
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)
    rag_engine._check_model_identity(db_path="http://milvus:19530")
    scheduled = []
    monkeypatch.setattr(rag_engine, "_schedule_stamp_refresh", lambda: scheduled.append(1))

    assert rag_engine.embedding_stamp_status()["drift"] == ["revision"]
    assert scheduled == []

    rag_engine._stamp_status["checked_at"] -= rag_engine.STAMP_STATUS_TTL_S + 1
    assert rag_engine.embedding_stamp_status()["drift"] == ["revision"]
    assert scheduled == [1]


def test_malformed_identity_file_refuses_over_stored_vectors(stamp_env, tmp_path):
    stamp_env.write_text("{not json")

    with pytest.raises(RuntimeError, match="model_identity.json"):
        rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert stamp_env.read_text() == "{not json"


def test_malformed_identity_file_on_an_empty_index_restamps(stamp_env, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "_index_has_data", lambda db_path: False)
    stamp_env.write_text("{not json")

    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert json.loads(stamp_env.read_text()) == _stamp(CURRENT)


def test_index_check_error_counts_as_data(monkeypatch):
    def unreachable(uri):
        raise ConnectionError("milvus down")

    monkeypatch.setattr(rag_engine, "MilvusClient", unreachable)

    assert rag_engine._index_has_data("http://milvus:19530") is True


def test_unresolved_fingerprint_is_not_cached(tmp_path, monkeypatch):
    cache = tmp_path / "models--x--y"
    monkeypatch.setattr(rag_engine, "_MODEL_CACHE", cache)
    monkeypatch.setattr(rag_engine, "_current_fingerprint", None)

    assert rag_engine.current_fingerprint()["revision"] is None

    _cache(tmp_path)
    assert rag_engine.current_fingerprint()["revision"] == "rev1"


UNRESOLVED = dict(CURRENT, revision=None, tokenizer_sha256=None)


def test_unresolved_fingerprint_is_not_stamped_on_lite(stamp_env, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "current_fingerprint", lambda: dict(UNRESOLVED))

    rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))

    assert not stamp_env.exists()


def test_unresolved_fingerprint_is_not_stamped_on_standalone(stamp_env, monkeypatch):
    monkeypatch.setattr(rag_engine, "current_fingerprint", lambda: dict(UNRESOLVED))
    client = FakeStandalone()
    monkeypatch.setattr(rag_engine, "MilvusClient", lambda uri: client)

    rag_engine._check_model_identity(db_path="http://milvus:19530")

    assert client.altered == []


def test_unresolved_fingerprint_still_refuses_a_model_change(stamp_env, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "current_fingerprint", lambda: dict(UNRESOLVED))
    stamp_env.write_text(json.dumps(_stamp(CURRENT, model_name="modernbert")))

    with pytest.raises(RuntimeError, match="Model mismatch"):
        rag_engine._check_model_identity(db_path=str(tmp_path / "milvus.db"))


def test_health_embedding_block_reports_the_fingerprint(monkeypatch):
    import http_server

    # Another test can leave http_server bound to an earlier rag_engine module.
    status = {"hash": "abc", "fingerprint": dict(CURRENT), "stored_hash": "def", "drift": ["pooling"]}
    monkeypatch.setattr(http_server.rag_engine, "embedding_stamp_status", lambda: status, raising=False)

    assert http_server._embedding_status_payload()["fingerprint"] == status


def test_creating_the_sessions_collection_skips_an_unresolved_stamp(stamp_env, monkeypatch):
    monkeypatch.setattr(rag_engine, "current_fingerprint", lambda: dict(UNRESOLVED))

    class Creating(FakeStandalone):
        def prepare_index_params(self):
            class P:
                def add_index(self, **kwargs):
                    pass
            return P()

        def create_collection(self, **kwargs):
            self.exists = True

        def load_collection(self, collection_name):
            pass

    client = Creating(exists=False)
    rag_engine._create_collection(client, "http://milvus:19530")

    assert client.altered == []


def test_creating_the_memories_collection_does_not_stamp(stamp_env, monkeypatch):
    class Creating(FakeStandalone):
        def prepare_index_params(self):
            class P:
                def add_index(self, **kwargs):
                    pass
            return P()

        def create_collection(self, **kwargs):
            self.exists = True

        def load_collection(self, collection_name):
            pass

    client = Creating(exists=False)
    rag_engine._create_collection(client, "http://milvus:19530", spec=rag_engine.MEMORIES_SPEC)

    assert client.altered == []
