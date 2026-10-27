"""SESF-57: in-place re-embed of the existing index after the padding-mask fix.

Covers the ``reembed`` orchestrator (snapshot -> chunked, checkpointed, batched
upsert of new vectors), the two ``rag_engine`` primitives it needs
(``get_rows_by_ids`` / ``upsert_rows``), and the ``cleanup.py reembed`` CLI gate
and chunk driver. Milvus, MLX and the embedding budget are all stubbed: nothing
here touches a real index or the GPU.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import stat
import struct
import types

import pytest

import rag_engine

DIM = 4


def _row(pk: int, text: str) -> dict:
    """A synthetic Milvus row with every field a real ``output_fields=["*"]`` row carries."""
    return {
        "id": pk,
        "vector": [0.5, 0.5, 0.5, 0.5],
        "document": text,
        "doc_id": f"doc-{pk}",
        "session_id": "s1",
        "logical_session_id": "s1",
        "provider": "claude_code_cli",
        "source_kind": "cli",
        "source_class": "transcript",
        "source_id": "",
        "source_path": "/synthetic/t.jsonl",
        "transcript_file": "/synthetic/t.jsonl",
        "turn_index": pk,
        "timestamp": "2026-10-01T00:00:00Z",
        "git_branch": "main",
        "chunk_type": "turn",
        "project_root": "/synthetic",
        "issue_ids": "",
    }


def _vec_for(text: str) -> list:
    """Deterministic fake embedding so tests can assert which text produced a vector."""
    n = float(len(text))
    return [n, n + 1, n + 2, n + 3]


class _Budget:
    """Budget stub: always allows, records batch sizes, honors batch_size splitting."""

    def __init__(self, batch_size=2, paused=False):
        self.batch_size = batch_size
        self.paused = paused
        self.before = []
        self.after = []

    def split_batches(self, items):
        return [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    def before_batch(self, batch_size, estimated_chars=0):
        self.before.append(batch_size)
        if self.paused:
            return types.SimpleNamespace(allowed=False, reason="paused", retry_after_seconds=0.0)
        return types.SimpleNamespace(allowed=True, reason="", retry_after_seconds=0.0)

    def after_batch(self, duration, turns, error=None):
        self.after.append((turns, error))


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated HOME plus stubbed rag_engine/budget seams; returns a capture dict."""
    monkeypatch.setenv("HOME", str(tmp_path))
    reembed = importlib.import_module("reembed")

    rows = {pk: _row(pk, "t" * pk) for pk in range(1, 8)}  # 7 rows
    cap = {
        "rows": rows,
        "snapshots": 0,
        "embed_inputs": [],
        "embed_is_query": [],
        "upserts": [],
        "embed_error": None,
        "budget": _Budget(batch_size=2),
        "clock": [0.0],
    }

    def fake_query_batches(output_fields, *args, **kwargs):
        cap["snapshots"] += 1
        assert output_fields == ["id"]
        ids = sorted(cap["rows"])
        yield [{"id": i} for i in ids[:4]]
        yield [{"id": i} for i in ids[4:]]

    def fake_get_rows_by_ids(ids, db_path=None):
        return [dict(cap["rows"][i], vector=list(cap["rows"][i]["vector"]))
                for i in ids if i in cap["rows"]]

    def fake_upsert_rows(batch_rows, db_path=None):
        cap["upserts"].append([dict(r) for r in batch_rows])
        for r in batch_rows:
            cap["rows"][r["id"]] = dict(r)
        return len(batch_rows)

    def fake_embed(texts, is_query=False):
        cap["embed_inputs"].append(list(texts))
        cap["embed_is_query"].append(is_query)
        if cap["embed_error"] is not None:
            raise cap["embed_error"]
        return [_vec_for(t) for t in texts]

    def forbidden(*a, **k):
        raise AssertionError("re-embed must not rewrite FTS or go through upsert_document")

    monkeypatch.setattr(rag_engine, "_query_batches", fake_query_batches)
    monkeypatch.setattr(rag_engine, "get_rows_by_ids", fake_get_rows_by_ids, raising=False)
    monkeypatch.setattr(rag_engine, "upsert_rows", fake_upsert_rows, raising=False)
    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed)
    monkeypatch.setattr(rag_engine, "upsert_document", forbidden)
    monkeypatch.setattr(reembed, "get_embedding_budget", lambda: cap["budget"])
    monkeypatch.setattr(reembed, "_monotonic", lambda: cap["clock"][0])
    monkeypatch.setattr(reembed, "_sleep", lambda s: None)
    cap["module"] = reembed
    cap["home"] = tmp_path
    return cap


def _state(env) -> dict:
    return json.loads((env["home"] / ".sessionflow" / "reembed_state.json").read_text())


# --- Orchestrator ------------------------------------------------------------


def test_run_refuses_without_confirmation(env):
    report = env["module"].run(confirmed=False)
    assert report.status == "refused"
    assert env["snapshots"] == 0
    assert env["upserts"] == []
    assert not (env["home"] / ".sessionflow" / "reembed_state.json").exists()


def test_plan_is_read_only(env):
    report = env["module"].plan()
    assert report.status == "dry-run"
    assert report.total == 7
    assert report.estimated_seconds > 0
    assert env["upserts"] == []
    assert env["embed_inputs"] == []
    assert not (env["home"] / ".sessionflow" / "reembed_state.json").exists()


def test_full_run_reembeds_every_row_in_place(env):
    before = {pk: dict(r) for pk, r in env["rows"].items()}
    report = env["module"].run(confirmed=True)

    assert report.status == "done"
    assert report.done == 7
    assert env["snapshots"] == 1
    # Embedded the stored document text, as documents (not queries).
    flat = [t for batch in env["embed_inputs"] for t in batch]
    assert flat == [before[pk]["document"] for pk in sorted(before)]
    assert set(env["embed_is_query"]) == {False}
    # Every row upserted exactly once, only the vector changed.
    upserted = [r for batch in env["upserts"] for r in batch]
    assert sorted(r["id"] for r in upserted) == sorted(before)
    for r in upserted:
        old = before[r["id"]]
        assert r["vector"] == _vec_for(old["document"])
        assert {k: v for k, v in r.items() if k != "vector"} == \
            {k: v for k, v in old.items() if k != "vector"}
    # Batched per the budget, through the budget.
    assert env["budget"].before == [2, 2, 2, 1]
    assert [t for t, _ in env["budget"].after] == [2, 2, 2, 1]


def test_state_file_is_private_and_complete(env):
    env["module"].run(confirmed=True)
    path = env["home"] / ".sessionflow" / "reembed_state.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    state = _state(env)
    assert state["status"] == "done"
    assert state["snapshot_count"] == 7
    assert state["cursor"] == 7
    assert state["model_name"] == rag_engine.get_model_name()
    assert state["chunks"] and state["chunks"][-1]["turns"] == 7


def test_max_turns_pauses_then_resume_finishes_without_resnapshot(env):
    first = env["module"].run(confirmed=True, max_turns=4)
    assert first.status == "paused"
    assert _state(env)["cursor"] == 4

    second = env["module"].run(confirmed=True)
    assert second.status == "done"
    assert env["snapshots"] == 1
    upserted = [r["id"] for batch in env["upserts"] for r in batch]
    assert sorted(upserted) == list(range(1, 8))
    assert len(upserted) == len(set(upserted))
    assert len(_state(env)["chunks"]) == 2


def test_max_minutes_pauses(env):
    clock = env["clock"]
    real_embed = rag_engine.embed_texts

    def slow_embed(texts, is_query=False):
        clock[0] += 61.0
        return real_embed(texts, is_query=is_query)

    rag_engine.embed_texts = slow_embed
    try:
        report = env["module"].run(confirmed=True, max_minutes=2)
    finally:
        rag_engine.embed_texts = real_embed
    assert report.status == "paused"
    # 61s per batch: after batch 2 elapsed=122s >= 120s, so stop there.
    assert report.done == 4
    assert _state(env)["cursor"] == 4


def test_rows_deleted_mid_run_count_as_missing(env):
    env["module"].run(confirmed=True, max_turns=2)
    del env["rows"][5]
    report = env["module"].run(confirmed=True)
    assert report.status == "done"
    state = _state(env)
    assert state["missing"] == 1
    assert state["done"] + state["missing"] == state["snapshot_count"]


def test_embed_failure_checkpoints_without_advancing(env):
    env["module"].run(confirmed=True, max_turns=2)
    env["embed_error"] = RuntimeError("metal went away")
    report = env["module"].run(confirmed=True)
    assert report.status == "error"
    state = _state(env)
    assert state["cursor"] == 2
    assert state["errors"] == 1
    assert state["status"] == "paused"

    env["embed_error"] = None
    assert env["module"].run(confirmed=True).status == "done"


def test_budget_denial_pauses_cleanly(env):
    env["budget"].paused = True
    report = env["module"].run(confirmed=True)
    assert report.status == "paused"
    assert env["upserts"] == []
    assert _state(env)["cursor"] == 0


def test_old_vectors_backed_up_before_overwrite(env):
    report = env["module"].run(confirmed=True)
    path = report.backup_path
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    record = struct.calcsize(f"<q{DIM}e")
    data = open(path, "rb").read()
    assert len(data) == 7 * record
    pk, *vec = struct.unpack_from(f"<q{DIM}e", data, 0)
    assert pk == 1
    assert vec == [0.5, 0.5, 0.5, 0.5]


def test_resume_refuses_on_model_change(env, monkeypatch):
    env["module"].run(confirmed=True, max_turns=2)
    monkeypatch.setattr(rag_engine, "get_model_name", lambda: "modernbert")
    report = env["module"].run(confirmed=True)
    assert report.status == "refused"
    assert _state(env)["cursor"] == 2


def test_restart_flag_takes_a_fresh_snapshot(env):
    env["module"].run(confirmed=True)
    report = env["module"].run(confirmed=True, restart=True)
    assert report.status == "done"
    assert env["snapshots"] == 2


def test_verify_reports_cosine_against_single_embeds(env):
    env["module"].run(confirmed=True)
    report = env["module"].verify(sample=5, threshold=0.9999)
    assert report.checked == 5
    assert report.passed == 5
    assert report.min_cos == pytest.approx(1.0)

    # A drifted stored vector fails the check.
    env["rows"][3]["vector"] = [1.0, 0.0, 0.0, 0.0]
    report = env["module"].verify(sample=7, threshold=0.9999)
    assert report.checked == 7
    assert report.passed == 6
    assert report.min_cos < 0.9999
    # verify never writes.
    assert len(env["upserts"]) == 4


# --- rag_engine primitives ---------------------------------------------------


@pytest.fixture
def fake_client(monkeypatch):
    calls = {"query": [], "upsert": []}

    class Client:
        def has_collection(self, name):
            return True

        def query(self, **kwargs):
            calls["query"].append(kwargs)
            return [{"id": 1, "vector": [0.1] * DIM, "document": "x"}]

        def upsert(self, **kwargs):
            calls["upsert"].append(kwargs)
            return {"upsert_count": len(kwargs["data"])}

    @contextlib.contextmanager
    def fake_milvus_client(db_path=None):
        yield Client()

    monkeypatch.setattr(rag_engine, "milvus_client", fake_milvus_client)
    return calls


def test_get_rows_by_ids_fetches_full_rows_by_pk(fake_client):
    rows = rag_engine.get_rows_by_ids([1, 22, 333])
    assert rows == [{"id": 1, "vector": [0.1] * DIM, "document": "x"}]
    q = fake_client["query"][0]
    assert q["filter"] == "id in [1, 22, 333]"
    assert q["output_fields"] == ["*"]
    assert q["collection_name"] == rag_engine.COLLECTION_NAME


def test_get_rows_by_ids_empty_is_noop(fake_client):
    assert rag_engine.get_rows_by_ids([]) == []
    assert fake_client["query"] == []


def test_upsert_rows_single_call(fake_client):
    data = [{"id": 1}, {"id": 2}]
    assert rag_engine.upsert_rows(data) == 2
    assert len(fake_client["upsert"]) == 1
    assert fake_client["upsert"][0]["data"] == data
    assert rag_engine.upsert_rows([]) == 0
    assert len(fake_client["upsert"]) == 1


# --- CLI ---------------------------------------------------------------------


def _parse(argv):
    cleanup = importlib.import_module("cleanup")
    return cleanup, cleanup.build_parser().parse_args(argv)


def test_cli_run_requires_yes(env, monkeypatch, capsys):
    cleanup, args = _parse(["reembed", "run"])
    called = []
    monkeypatch.setattr(env["module"], "run", lambda **k: called.append(k))
    assert cleanup.cmd_reembed(args) == 1
    assert called == []
    assert "--yes" in capsys.readouterr().err


def test_cli_drive_requires_yes(env, monkeypatch):
    cleanup, args = _parse(["reembed", "drive"])
    monkeypatch.setattr(cleanup.subprocess, "run",
                        lambda *a, **k: pytest.fail("must not spawn without --yes"))
    assert cleanup.cmd_reembed(args) == 1


def test_cli_drive_spawns_one_process_per_chunk_until_done(env, monkeypatch):
    cleanup, args = _parse(["reembed", "drive", "--yes", "--chunk-minutes", "35",
                            "--gap-minutes", "10"])
    statuses = iter(["paused", "paused", "done"])
    spawned, slept = [], []

    def fake_run(cmd, **kwargs):
        spawned.append(cmd)
        env["module"]._save_state({"status": next(statuses)})
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(cleanup.subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup.time, "sleep", lambda s: slept.append(s))
    assert cleanup.cmd_reembed(args) == 0
    assert len(spawned) == 3
    assert all("--max-minutes" in c and "35.0" in c and "--yes" in c for c in spawned)
    assert slept == [600.0, 600.0]


def test_cli_drive_gives_up_after_repeated_crashes(env, monkeypatch):
    cleanup, args = _parse(["reembed", "drive", "--yes", "--max-failures", "2"])
    spawned = []

    def crash(cmd, **kwargs):
        spawned.append(cmd)
        return types.SimpleNamespace(returncode=-11)  # SIGSEGV

    monkeypatch.setattr(cleanup.subprocess, "run", crash)
    monkeypatch.setattr(cleanup.time, "sleep", lambda s: None)
    assert cleanup.cmd_reembed(args) == 1
    assert len(spawned) == 2


# ---------------------------------------------------------------------------
# One state and worklist file per collection; a re-embed run also bumps the
# memories embedding generation once, and leaves memory embedding to the
# server's sync loop.
# ---------------------------------------------------------------------------


def test_state_files_are_per_collection(env):
    reembed = env["module"]
    assert reembed._state_path().name == "reembed_state.json"
    assert reembed._state_path("memories") != reembed._state_path()
    assert reembed._worklist_path("memories") != reembed._worklist_path()


def test_fresh_run_bumps_memory_generation_once(env, monkeypatch):
    reembed = env["module"]
    bumps = []
    monkeypatch.setattr(reembed, "_reembed_memories", lambda: bumps.append(1) or 2)

    first = reembed.run(confirmed=True, max_turns=2)
    assert first.status == "paused"
    reembed.run(confirmed=True)

    assert bumps == [1]


def _memory_record(memory_store, text):
    import hashlib

    return memory_store.NewMemory(
        text=text,
        agent_id="claude-code",
        project="sessionflow",
        type="fact",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_cli_reembed_covers_memories_without_embedding_them(env, monkeypatch, memory_pg):
    import memory_store

    reembed = env["module"]
    store = memory_store.MemoryStore(memory_pg)
    try:
        store.migrate()
        written = store.add(_memory_record(memory_store, "memory text that must not embed here"))
        store.ack("milvus", written.memory_id, version=1, generation=1, deleted=False)
        assert store.pending("milvus") == []

        monkeypatch.setattr(reembed, "_open_memory_store", lambda: store)
        cleanup, args = _parse(["reembed", "run", "--yes"])
        assert cleanup.cmd_reembed(args) == 0

        assert store.index_meta()["generation"] == 2
        assert [r["memory_id"] for r in store.pending("milvus")] == [written.memory_id]
        flat = [t for batch in env["embed_inputs"] for t in batch]
        assert all("memory text" not in t for t in flat)
    finally:
        store.close()
