"""SESF-62 / SESF-66: in-place rewrite of stored rows (strip boilerplate, re-embed long rows).

Covers the ``rewrite`` orchestrator (worklist selection, strip + re-embed + upsert,
drops, FTS retry, backups, chunking, verify) and the ``cleanup.py rewrite`` CLI gate.
Milvus, FTS, MLX, the tokenizer and the embedding budget are stubbed.
"""

from __future__ import annotations

import importlib
import json
import os
import stat
import struct
import types

import pytest

import rag_engine

DIM = 4
WINDOW = 100

TASK = "<task-notification>\n<task-id>x</task-id>\n</task-notification>\n"
PLAIN = "User: how does the scope line work\n\nAssistant: it names the roots"
BOILER = "User: " + TASK + "what changed in SESF-66?\n\nAssistant: the strip rules"
ONLY_BOILER = "User: " + TASK.rstrip("\n")
LONG = "User: " + "word " * 20 + "\n\nAssistant: SESF-62 " + "long " * 20


def _row(pk: int, text: str) -> dict:
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
        "issue_ids": ",OLD-1,",
    }


def _vec_for(text: str) -> list:
    n = float(len(text))
    return [n, n + 1, n + 2, n + 3]


class _Budget:
    def __init__(self, batch_size=2, paused=False):
        self.batch_size = batch_size
        self.paused = paused
        self.before = []

    def before_batch(self, batch_size, estimated_chars=0):
        self.before.append(batch_size)
        if self.paused:
            return types.SimpleNamespace(allowed=False, reason="paused", retry_after_seconds=0.0)
        return types.SimpleNamespace(allowed=True, reason="", retry_after_seconds=0.0)

    def after_batch(self, duration, turns, error=None):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated HOME plus stubbed engine seams; returns a capture dict."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SESSIONFLOW_REDACT", "off")
    rewrite = importlib.import_module("rewrite")
    reembed = importlib.import_module("reembed")

    rows = {
        1: _row(1, PLAIN),
        2: _row(2, BOILER),
        3: _row(3, ONLY_BOILER),
        4: _row(4, LONG),
        5: _row(5, PLAIN + " again"),
    }
    cap = {
        "rows": rows,
        "fts": {r["doc_id"]: r["document"] for r in rows.values()},
        "scans": [],
        "embed_inputs": [],
        "upserts": [],
        "deletes": [],
        "fts_fail": set(),
        "delete_fts_fail": set(),
        "embed_error": None,
        "budget": _Budget(batch_size=2),
        "clock": [0.0],
        "memory_bumps": 0,
    }

    def fake_query_batches(output_fields, *args, **kwargs):
        cap["scans"].append(list(output_fields))
        ids = sorted(cap["rows"])
        yield [{f: cap["rows"][i][f] for f in output_fields} for i in ids]

    def fake_get_rows_by_ids(ids, db_path=None):
        return [dict(cap["rows"][i], vector=list(cap["rows"][i]["vector"]))
                for i in ids if i in cap["rows"]]

    def fake_upsert_rows(batch_rows, db_path=None):
        cap["upserts"].append([dict(r) for r in batch_rows])
        for r in batch_rows:
            cap["rows"][r["id"]] = dict(r)
        return len(batch_rows)

    def fake_embed(texts, is_query=False):
        assert is_query is False
        cap["embed_inputs"].append(list(texts))
        if cap["embed_error"] is not None:
            raise cap["embed_error"]
        return [_vec_for(t) for t in texts]

    def fake_rewrite_fts(doc_id, row, db_path):
        if doc_id in cap["fts_fail"]:
            return False
        cap["fts"][doc_id] = row["document"]
        return True

    def fake_delete(doc_id, db_path=None):
        cap["deletes"].append(doc_id)
        pk = int(doc_id.split("-")[1])
        deleted = 1 if cap["rows"].pop(pk, None) is not None else 0
        if doc_id in cap["delete_fts_fail"]:
            return rag_engine.DeleteResult(deleted, False)
        cap["fts"].pop(doc_id, None)
        return rag_engine.DeleteResult(deleted, True)

    def fake_bump():
        cap["memory_bumps"] += 1
        return 7

    monkeypatch.setattr(rag_engine, "_query_batches", fake_query_batches)
    monkeypatch.setattr(rag_engine, "get_rows_by_ids", fake_get_rows_by_ids)
    monkeypatch.setattr(rag_engine, "upsert_rows", fake_upsert_rows)
    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed)
    monkeypatch.setattr(rag_engine, "_rewrite_fts_row", fake_rewrite_fts)
    monkeypatch.setattr(rag_engine, "delete_by_doc_id", fake_delete)
    monkeypatch.setattr(reembed, "_reembed_memories", fake_bump)
    monkeypatch.setattr(rewrite, "get_embedding_budget", lambda: cap["budget"])
    monkeypatch.setattr(rewrite, "_monotonic", lambda: cap["clock"][0])
    monkeypatch.setattr(reembed, "_sleep", lambda s: None)
    monkeypatch.setattr(rewrite, "_token_lengths", lambda texts: [len(t) for t in texts])
    monkeypatch.setattr(rewrite, "_window", lambda: WINDOW)
    cap["module"] = rewrite
    cap["home"] = tmp_path
    return cap


def _state(env) -> dict:
    return json.loads((env["home"] / ".sessionflow" / "rewrite_state.json").read_text())


# --- Selection ----------------------------------------------------------------


def test_needs_rewrite_selects_stripped_and_long_rows(env):
    rw = env["module"]
    assert rw.needs_rewrite(BOILER, token_length=10, window=WINDOW)
    assert rw.needs_rewrite(PLAIN, token_length=WINDOW + 2, window=WINDOW)
    assert not rw.needs_rewrite(PLAIN, token_length=WINDOW + 1, window=WINDOW)
    assert not rw.needs_rewrite(PLAIN, token_length=10_000, window=None)


def test_long_rows_are_measured_after_stripping(env):
    rw = env["module"]
    padded = "User: " + TASK * 20 + "short question here"
    clean = rw.clean_text(padded)
    assert len(clean) <= WINDOW + 1
    # Selected for the strip, not for its length.
    assert rw.needs_rewrite(padded, token_length=len(clean), window=WINDOW)


# --- Plan ---------------------------------------------------------------------


def test_run_refuses_without_confirmation(env):
    report = env["module"].run(confirmed=False)
    assert report.status == "refused"
    assert env["scans"] == []
    assert not (env["home"] / ".sessionflow" / "rewrite_state.json").exists()


def test_plan_counts_without_writing(env):
    report = env["module"].plan()
    assert report.status == "dry-run"
    assert report.scanned == 5
    assert report.stripped == 2          # rows 2 and 3
    assert report.long == 1              # row 4: over WINDOW + 1, nothing to strip
    assert report.dropped == 1           # row 3 strips to nothing
    assert report.total == 3
    assert env["upserts"] == [] and env["deletes"] == [] and env["embed_inputs"] == []
    assert not (env["home"] / ".sessionflow" / "rewrite_state.json").exists()


# --- Run ----------------------------------------------------------------------


def test_full_run_strips_reembeds_and_drops(env):
    report = env["module"].run(confirmed=True)
    assert report.status == "done"
    assert (report.total, report.done, report.deleted, report.missing) == (3, 2, 1, 0)

    stripped = env["rows"][2]
    assert "<task-notification>" not in stripped["document"]
    assert stripped["document"].startswith("User: what changed in SESF-66?")
    assert stripped["vector"] == _vec_for(stripped["document"])
    assert stripped["issue_ids"] == ",SESF-66,"
    assert env["fts"]["doc-2"] == stripped["document"]

    long_row = env["rows"][4]
    assert long_row["document"] == LONG
    assert long_row["vector"] == _vec_for(LONG)
    assert long_row["issue_ids"] == ",SESF-62,"

    assert 3 not in env["rows"] and "doc-3" not in env["fts"]
    assert env["rows"][1]["vector"] == [0.5, 0.5, 0.5, 0.5]   # untouched
    assert env["memory_bumps"] == 1


def test_metadata_is_preserved(env):
    env["module"].run(confirmed=True)
    row = env["rows"][2]
    for field in ("doc_id", "session_id", "timestamp", "git_branch", "project_root", "turn_index"):
        assert row[field] == _row(2, BOILER)[field]


def test_state_and_backups_are_private(env):
    report = env["module"].run(confirmed=True)
    state_path = env["home"] / ".sessionflow" / "rewrite_state.json"
    assert stat.S_IMODE(os.stat(state_path).st_mode) == 0o600
    docs = env["home"] / ".sessionflow" / "audit" / f"rewrite-{report.run_id}.jsonl"
    vecs = env["home"] / ".sessionflow" / "audit" / f"rewrite-{report.run_id}.f16"
    for path in (docs, vecs):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    saved = [json.loads(line) for line in docs.read_text().splitlines()]
    by_id = {s["id"]: s for s in saved}
    assert set(by_id) == {2, 3, 4}
    assert by_id[2]["document"] == BOILER
    assert by_id[2]["issue_ids"] == ",OLD-1,"
    assert by_id[3]["doc_id"] == "doc-3"

    record = struct.calcsize(f"<q{DIM}e")
    assert len(vecs.read_bytes()) == record * 3


def test_max_turns_pauses_then_resume_finishes(env):
    rw = env["module"]
    first = rw.run(confirmed=True, max_turns=2)
    assert first.status == "paused"
    assert first.cursor == 2
    scans = len(env["scans"])
    second = rw.run(confirmed=True)
    assert second.status == "done"
    assert second.cursor == 3
    assert len(env["scans"]) == scans          # no new snapshot on resume
    assert env["memory_bumps"] == 1


def test_max_minutes_pauses(env, monkeypatch):
    inner = rag_engine.embed_texts

    def slow_embed(texts, is_query=False):
        env["clock"][0] += 120.0
        return inner(texts, is_query)

    monkeypatch.setattr(rag_engine, "embed_texts", slow_embed)
    report = env["module"].run(confirmed=True, max_minutes=1)
    assert report.status == "paused"
    assert report.cursor == 2


def test_embed_failure_does_not_advance(env):
    env["embed_error"] = RuntimeError("metal")
    report = env["module"].run(confirmed=True)
    assert report.status == "error"
    assert report.cursor == 0
    assert report.errors == 1
    assert env["rows"][2]["document"] == BOILER
    assert env["deletes"] == []


def test_budget_denial_pauses(env):
    env["budget"].paused = True
    report = env["module"].run(confirmed=True)
    assert report.status == "paused"
    assert report.cursor == 0


def test_rows_gone_mid_run_count_as_missing(env):
    rw = env["module"]
    rw.run(confirmed=True, max_turns=0)        # snapshot only
    del env["rows"][4]
    report = rw.run(confirmed=True)
    assert report.status == "done"
    assert report.missing == 1


def test_delete_interrupted_after_milvus_is_finished_next_run(env, monkeypatch):
    rw = env["module"]
    inner = rag_engine.delete_by_doc_id

    def crash_after_milvus(doc_id, db_path=None):
        env["rows"].pop(int(doc_id.split("-")[1]), None)
        raise RuntimeError("process died before FTS delete")

    monkeypatch.setattr(rag_engine, "delete_by_doc_id", crash_after_milvus)
    first = rw.run(confirmed=True)
    assert first.status == "error"
    assert "doc-3" in env["fts"]
    assert first.fts_pending == 1

    monkeypatch.setattr(rag_engine, "delete_by_doc_id", inner)
    second = rw.run(confirmed=True)
    assert second.status == "done"
    assert second.fts_pending == 0
    assert "doc-3" not in env["fts"]


def test_fts_rewrite_failure_is_retried_next_run(env):
    rw = env["module"]
    env["fts_fail"].add("doc-2")
    first = rw.run(confirmed=True)
    assert first.status == "paused"
    assert first.cursor == 3
    assert first.fts_pending == 1
    assert "<task-notification>" in env["fts"]["doc-2"]
    assert _state(env)["status"] == "paused"

    env["fts_fail"].clear()
    embeds = len(env["embed_inputs"])
    second = rw.run(confirmed=True)
    assert second.status == "done"
    assert second.fts_pending == 0
    assert env["fts"]["doc-2"] == env["rows"][2]["document"]
    assert len(env["embed_inputs"]) == embeds  # FTS retry does not re-embed
    assert _state(env)["status"] == "done"


def test_fts_delete_failure_is_retried_next_run(env):
    rw = env["module"]
    env["delete_fts_fail"].add("doc-3")
    first = rw.run(confirmed=True)
    assert first.status == "paused"
    assert first.fts_pending == 1
    assert "doc-3" in env["fts"]
    env["delete_fts_fail"].clear()
    second = rw.run(confirmed=True)
    assert second.fts_pending == 0
    assert "doc-3" not in env["fts"]


def test_rerun_after_done_is_refused_without_restart(env):
    rw = env["module"]
    rw.run(confirmed=True)
    again = rw.run(confirmed=True)
    assert again.status == "done"
    assert "restart" in again.message


def test_resume_refuses_on_model_change(env, monkeypatch):
    rw = env["module"]
    rw.run(confirmed=True, max_turns=1)
    monkeypatch.setattr(rag_engine, "get_model_name", lambda: "other-model")
    report = rw.run(confirmed=True)
    assert report.status == "refused"


def test_restart_takes_a_fresh_snapshot(env):
    rw = env["module"]
    rw.run(confirmed=True, max_turns=1)
    scans = len(env["scans"])
    rw.run(confirmed=True, restart=True)
    assert len(env["scans"]) == scans + 1
    assert env["memory_bumps"] == 2


def test_redaction_runs_on_the_stripped_text(env, monkeypatch):
    seen = []

    def fake_redaction(turns):
        seen.extend(t["text"] for t in turns)
        for t in turns:
            t["text"] = t["text"].replace("strip rules", "[REDACTED]")

    monkeypatch.setattr(rag_engine, "_apply_redaction", fake_redaction)
    env["module"].run(confirmed=True)
    assert all("<task-notification>" not in s for s in seen)
    assert env["rows"][2]["document"].endswith("[REDACTED]")


# --- Verify -------------------------------------------------------------------


def test_verify_checks_strip_and_vectors(env):
    rw = env["module"]
    rw.run(confirmed=True)
    result = rw.verify(sample=10, seed=1)
    assert result.checked == 2               # row 3 was deleted
    assert result.unstripped == 0
    assert result.passed == 2
    env["rows"][4]["vector"] = [9.0, 0.0, 0.0, 0.0]
    env["rows"][2]["document"] = BOILER
    result = rw.verify(sample=10, seed=1)
    assert result.unstripped == 1
    assert result.passed == 0


def test_verify_reports_incomplete_when_budget_stops_it(env):
    rw = env["module"]
    rw.run(confirmed=True)
    assert rw.verify(sample=10, seed=1).complete is True
    env["budget"].paused = True
    result = rw.verify(sample=10, seed=1)
    assert result.complete is False


def test_backups_are_synced_before_the_upsert(env, monkeypatch):
    events = []
    real_fsync = os.fsync
    inner_upsert = rag_engine.upsert_rows

    def fsync(fd):
        events.append("fsync")
        real_fsync(fd)

    def upsert(rows, db_path=None):
        events.append("upsert")
        return inner_upsert(rows, db_path)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(rag_engine, "upsert_rows", upsert)
    env["module"].run(confirmed=True, max_turns=2)
    assert events[:3] == ["fsync", "fsync", "upsert"]


# --- CLI ----------------------------------------------------------------------


def _parse(argv):
    cleanup = importlib.import_module("cleanup")
    return cleanup, cleanup.build_parser().parse_args(argv)


def test_cli_run_requires_yes(env, capsys):
    cleanup, args = _parse(["rewrite", "run"])
    assert cleanup.cmd_rewrite(args) == 1
    assert "--yes" in capsys.readouterr().err
    assert env["scans"] == []


def test_cli_drive_requires_yes(env, monkeypatch):
    cleanup, args = _parse(["rewrite", "drive"])
    monkeypatch.setattr(cleanup.subprocess, "run",
                        lambda *a, **k: pytest.fail("drive must not spawn without --yes"))
    assert cleanup.cmd_rewrite(args) == 1


def test_cli_drive_spawns_rewrite_chunks_until_done(env, monkeypatch):
    cleanup, args = _parse(["rewrite", "drive", "--yes", "--chunk-minutes", "30",
                            "--gap-minutes", "0"])
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        state = {"status": "done" if len(calls) == 2 else "paused"}
        path = env["home"] / ".sessionflow" / "rewrite_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(cleanup.subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup.time, "sleep", lambda s: None)
    assert cleanup.cmd_rewrite(args) == 0
    assert len(calls) == 2
    assert calls[0][2:6] == ["rewrite", "run", "--yes", "--max-minutes"]


def test_cli_verify_fails_when_incomplete(env):
    env["module"].run(confirmed=True)
    env["budget"].paused = True
    cleanup, args = _parse(["rewrite", "verify"])
    assert cleanup.cmd_rewrite(args) == 1


def test_cli_plan_prints_counts(env, capsys):
    cleanup, args = _parse(["rewrite", "plan"])
    assert cleanup.cmd_rewrite(args) == 0
    out = capsys.readouterr().out
    assert "Stripped:" in out and "Long:" in out and "Dropped:" in out
