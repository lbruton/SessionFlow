"""Tests for the SESF-42 retroactive secret sanitizer.

Two cohorts live here:

* **Orchestrator** — ``sanitize.Scope`` / ``sanitize.scan`` / ``sanitize.apply``
  and the ``SanitizeReport`` it returns. The orchestrator drives the dry-run/apply
  flow, throttling, checkpointing, and the secret-free audit trail (design.md
  Component 3). It owns policy only — detection routes through
  ``secret_redaction.scan_spans`` and data access through the
  ``rag_engine`` primitives (``upsert_document`` / ``delete_by_doc_id`` /
  ``get_row_by_doc_id``). The worklist holds only doc_ids + value-free audit
  metadata; affected rows are fetched just-in-time at apply time so memory stays
  bounded on large indices, and a resumed run loads the worklist + run_id from
  the checkpoint without a re-scan.
* **Primitives** — ``rag_engine.upsert_document`` / ``rag_engine.delete_by_doc_id``
  (design.md Component 2): the in-place Milvus overwrite + FTS rewrite, and the
  doc-id-scoped dual delete.

The implementation is present; this suite is the green contract that guards it.

Every token in this file is **synthetic and non-functional** (AC-18 / Requirement
5.3) — assembled from string parts so no contiguous secret literal appears, and no
raw value ever reaches stdout, the audit file, or a Milvus/FTS sink in the
assertions below.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import types
from pathlib import Path

import pytest

secret_redaction = importlib.import_module("secret_redaction")


# --- Synthetic, non-functional fixtures (AC-18) -----------------------------
# Assembled from parts so no contiguous secret literal sits in this source file.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
AWS_KEY2 = "AKIA" + "1234567890ABCDEF"
GH_KEY = "ghp" + "_" + "0123456789abcdefABCDEF0123456789abcd"

PROJECT = "/Volumes/DATA/GitHub/SessionFlow"
PROVIDER = "claude_code_cli"
SESSION = "synthetic-session-id"
SINCE = "2026-05-01"


def _require(module_name: str):
    """Import a target module (sanitize / rag_engine); RED until it exists."""
    return importlib.import_module(module_name)


def _milvus_row(doc_id: str, text: str) -> dict:
    """One synthetic Milvus scan row carrying all fields needed to rebuild + audit."""
    return {
        "id": int.from_bytes(doc_id.encode()[:7].ljust(7, b"0"), "big"),
        "doc_id": doc_id,
        "document": text,
        "provider": PROVIDER,
        "source_path": "/synthetic/transcript.jsonl",
        "turn_index": 1,
        "timestamp": "2026-05-21T10:00:00Z",
        "session_id": SESSION,
        "project_root": PROJECT,
    }


@pytest.fixture
def sanitize_env(monkeypatch, tmp_path):
    """Point the sanitizer's audit dir + checkpoint at a tmp HOME (no real writes).

    The audit dir (``~/.sessionflow/audit/``) and checkpoint
    (``~/.sessionflow/sanitize_state.json``) follow the existing state-file
    convention, so redirecting HOME isolates them per test.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # Keep detection config deterministic (enforce so masking is observable).
    monkeypatch.setenv("SESSIONFLOW_REDACT", "on")
    monkeypatch.setenv("SESSIONFLOW_REDACT_MODE", "enforce")
    monkeypatch.delenv("SESSIONFLOW_REDACT_ALLOWLIST", raising=False)
    return tmp_path


@pytest.fixture
def stubbed_engine(monkeypatch):
    """Stub the rag_engine seams the orchestrator calls; record every interaction.

    Records scan batches it serves, embed inputs, and upsert/delete/insert calls so
    a test can assert a dry-run writes nothing and an apply writes the redacted text.
    """
    import rag_engine

    cap = {
        "rows": [],            # rows the scan iterator yields
        "embed_inputs": [],    # texts passed to embed_texts (re-embed path)
        "upserts": [],         # (doc_id, new_document, new_vector)
        "deletes": [],         # doc_ids passed to delete_by_doc_id
        "fts_calls": [],       # ordered ("delete"|"insert", doc_id) for ordering checks
        "budget_calls": [],    # ("before"|"after", ...) budget interactions
        "upsert_result": None,  # override the UpsertResult the stub returns
        "delete_result": None,  # override the DeleteResult the stub returns
        "before_decisions": None,  # optional list of _Decision to serve in order
    }

    def fake_query_batches(output_fields, *args, **kwargs):
        if cap["rows"]:
            yield list(cap["rows"])

    monkeypatch.setattr(rag_engine, "_query_batches", fake_query_batches, raising=False)

    def fake_get_row_by_doc_id(doc_id, db_path=None):
        # JIT fetch (SESF-42 Fix A): apply pulls each affected row's full payload
        # by doc_id at write time instead of from an all-rows cache. Serve from
        # the same synthetic rows the scan iterator yields.
        for row in cap["rows"]:
            if row.get("doc_id") == doc_id:
                return dict(row)
        return None

    monkeypatch.setattr(
        rag_engine, "get_row_by_doc_id", fake_get_row_by_doc_id, raising=False
    )

    def fake_embed_texts(texts, is_query=False):
        cap["embed_inputs"].append(list(texts))
        return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed_texts)

    # UpsertResult is a NamedTuple introduced by SESF-42; build via the module so
    # the test binds to the real type once it exists.
    def fake_upsert_document(doc_id, *, new_document, new_vector=None, db_path=None):
        # Record the vector as-passed (None for the resume/FTS-only path) so a test
        # can assert the resume branch never sends a zero-length vector.
        cap["upserts"].append(
            (doc_id, new_document, None if new_vector is None else list(new_vector))
        )
        cap["fts_calls"].append(("delete", doc_id))
        cap["fts_calls"].append(("insert", doc_id))
        if cap["upsert_result"] is not None:
            return cap["upsert_result"]
        return rag_engine.UpsertResult(milvus_ok=True, fts_ok=True)

    monkeypatch.setattr(
        rag_engine, "upsert_document", fake_upsert_document, raising=False
    )

    def fake_delete_by_doc_id(doc_id, db_path=None):
        cap["deletes"].append(doc_id)
        if cap["delete_result"] is not None:
            return cap["delete_result"]
        return rag_engine.DeleteResult(deleted=1, fts_ok=True)

    monkeypatch.setattr(
        rag_engine, "delete_by_doc_id", fake_delete_by_doc_id, raising=False
    )

    class _Decision:
        allowed = True
        retry_after_seconds = 0.0
        reason = ""

    class _Budget:
        def split_batches(self, turns):
            return [list(turns)] if turns else []

        def before_batch(self, *a, **k):
            cap["budget_calls"].append(("before", a, k))
            # When a test seeds a deny decision queue, serve those in order; the
            # last one is reused once the queue drains.
            queue = cap["before_decisions"]
            if queue:
                return queue.pop(0) if len(queue) > 1 else queue[0]
            return _Decision()

        def after_batch(self, *a, **k):
            cap["budget_calls"].append(("after", a, k))

    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _Budget())

    return rag_engine, cap


# === Scope.to_filter() ======================================================


def test_scope_to_filter_project_provider_session_since():
    """Scope.to_filter() builds escaped Milvus clauses for each scope dimension."""
    sanitize = _require("sanitize")
    flt = sanitize.Scope(
        project_root=PROJECT, provider=PROVIDER, session_id=SESSION, since=SINCE
    ).to_filter()
    assert flt is not None
    assert f'project_root == "{PROJECT}"' in flt
    assert f'provider == "{PROVIDER}"' in flt
    assert f'session_id == "{SESSION}"' in flt
    # `since` maps to a timestamp lower bound.
    assert "timestamp >=" in flt and SINCE in flt


def test_scope_to_filter_escapes_injection():
    """A scope value with a quote/backslash is escaped, not concatenated raw."""
    sanitize = _require("sanitize")
    flt = sanitize.Scope(project_root='/a"b\\').to_filter()
    # The closing quote of the literal is not escaped away by a trailing backslash:
    # the escaper doubles backslashes and C-escapes the quote.
    assert flt.endswith('"')
    assert '/a\\"b\\\\' in flt


def test_scope_empty_is_match_all():
    """An empty scope produces no filter (match-all)."""
    sanitize = _require("sanitize")
    flt = sanitize.Scope().to_filter()
    assert flt in (None, "")


# === Dry-run does NO writes =================================================


def test_scan_dry_run_lists_affected_and_writes_nothing(
    sanitize_env, stubbed_engine
):
    """scan() reports affected doc_ids + per-rule counts and performs no writes."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [
        _milvus_row("d1", f"cred {AWS_KEY} end"),
        _milvus_row("d2", f"token {GH_KEY} end"),
        _milvus_row("d3", "totally benign text"),
    ]
    report = sanitize.scan(sanitize.Scope(project_root=PROJECT))

    # Affected = the two secret-bearing turns; the benign one is not listed.
    assert report.affected_count == 2
    assert report.counts.get("AWS", 0) >= 1
    assert report.counts.get("GITHUB", 0) >= 1
    assert report.mode in ("dry-run", "report")
    # No destructive calls of any kind.
    assert cap["upserts"] == []
    assert cap["deletes"] == []
    assert cap["embed_inputs"] == []
    assert cap["fts_calls"] == []


# === Apply (redact + re-embed) ==============================================


def test_apply_redact_upserts_redacted_doc_and_reembeds_redacted_text(
    sanitize_env, stubbed_engine
):
    """apply() upserts a redacted document + a vector from re-embedding the redacted text."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )

    assert cap["upserts"], "apply(redact) must call upsert_document"
    doc_id, new_document, _vector = cap["upserts"][0]
    assert doc_id == "d1"
    # The stored document is redacted, not the original secret.
    assert AWS_KEY not in new_document
    assert "[REDACTED:AWS]" in new_document
    # The re-embed ran over the REDACTED text (not the original).
    assert cap["embed_inputs"], "redact path must re-embed"
    embedded = cap["embed_inputs"][0][0]
    assert AWS_KEY not in embedded
    assert "[REDACTED:AWS]" in embedded
    # FTS rewrite is delete-then-insert for the doc.
    assert ("delete", "d1") in cap["fts_calls"]
    assert ("insert", "d1") in cap["fts_calls"]
    assert cap["fts_calls"].index(("delete", "d1")) < cap["fts_calls"].index(
        ("insert", "d1")
    )
    assert report.status == "complete"


def test_apply_redact_audit_action_is_redact_and_value_free(
    sanitize_env, stubbed_engine
):
    """The audit file records action='redact' and never the raw secret value."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )

    audit_text = Path(report.audit_path).read_text()
    assert AWS_KEY not in audit_text
    actions = {json.loads(line)["action"] for line in audit_text.splitlines() if line}
    assert actions == {"redact"}


# === Apply (drop fast path) =================================================


def test_apply_drop_deletes_without_reembed(sanitize_env, stubbed_engine):
    """drop mode calls delete_by_doc_id, performs NO re-embed, audit action='drop'."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=True, confirmed=True
    )

    assert cap["deletes"] == ["d1"]
    assert cap["upserts"] == []
    assert cap["embed_inputs"] == []  # no re-embed on the drop path
    audit_text = Path(report.audit_path).read_text()
    assert AWS_KEY not in audit_text
    actions = {json.loads(line)["action"] for line in audit_text.splitlines() if line}
    assert actions == {"drop"}


# === Confirmation gate ======================================================


def test_apply_without_confirmation_makes_no_calls(
    sanitize_env, stubbed_engine, monkeypatch
):
    """apply(confirmed=False) performs no reads or writes.

    The confirmation gate must short-circuit *before* any read path runs, not just
    before the writes — so spy both the scan read (``_query_batches``) and the
    just-in-time row fetch (``get_row_by_doc_id``) and assert neither fired.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    reads = {"query_batches": 0, "get_row": 0}
    real_query_batches = rag_engine._query_batches
    real_get_row = rag_engine.get_row_by_doc_id

    def spy_query_batches(*args, **kwargs):
        reads["query_batches"] += 1
        yield from real_query_batches(*args, **kwargs)

    def spy_get_row(doc_id, db_path=None):
        reads["get_row"] += 1
        return real_get_row(doc_id, db_path=db_path)

    monkeypatch.setattr(rag_engine, "_query_batches", spy_query_batches, raising=False)
    monkeypatch.setattr(rag_engine, "get_row_by_doc_id", spy_get_row, raising=False)

    sanitize.apply(sanitize.Scope(project_root=PROJECT), drop=False, confirmed=False)

    # No writes.
    assert cap["upserts"] == []
    assert cap["deletes"] == []
    assert cap["embed_inputs"] == []
    # No reads either: the scan iterator and the JIT row fetch never fired.
    assert reads["query_batches"] == 0
    assert reads["get_row"] == 0


# === FTS-failure incompletion ===============================================


@pytest.mark.parametrize("drop", [False, True], ids=["redact", "drop"])
def test_apply_fts_failure_leaves_row_incomplete(sanitize_env, stubbed_engine, drop):
    """fts_ok=False (either path) -> doc not 'done', incomplete_fts, status='incomplete'.

    Parameterized over redact (FTS rewrite fails) and drop (FTS delete fails): in
    both, the Milvus side succeeded but the FTS row may still carry the secret, so
    the doc_id must stay on the worklist, off the ``done`` set.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]
    if drop:
        cap["delete_result"] = rag_engine.DeleteResult(deleted=1, fts_ok=False)
    else:
        cap["upsert_result"] = rag_engine.UpsertResult(milvus_ok=True, fts_ok=False)

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=drop, confirmed=True
    )

    assert report.incomplete_fts >= 1
    assert report.status == "incomplete"  # NOT complete while FTS-failed rows remain

    # The doc_id stays in the checkpoint worklist, off the `done` set.
    state_path = Path(sanitize_env) / ".sessionflow" / "sanitize_state.json"
    state = json.loads(state_path.read_text())
    assert "d1" not in state.get("done", [])
    assert "d1" in state.get("worklist", [])


# === Checkpoint / resume ====================================================


def test_apply_resume_skips_already_done_doc_ids(
    sanitize_env, stubbed_engine, monkeypatch
):
    """A run with a pre-seeded `done` set skips those doc_ids (no re-process)."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [
        _milvus_row("d1", f"cred {AWS_KEY} end"),
        _milvus_row("d2", f"token {GH_KEY} end"),
    ]

    # Pre-seed the durable checkpoint with d1 already done.
    state_dir = Path(sanitize_env) / ".sessionflow"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sanitize_state.json").write_text(
        json.dumps(
            {
                "run_id": "prior",
                "mode": "redact",
                "scope": {"project_root": PROJECT},
                "worklist": ["d1", "d2"],
                "done": ["d1"],
                "counts": {},
                "status": "applying",
            }
        )
    )

    sanitize.apply(
        sanitize.Scope(project_root=PROJECT),
        drop=False,
        confirmed=True,
        resume=True,
    )

    # Only d2 is (re)processed; d1 was already done.
    upserted_ids = [u[0] for u in cap["upserts"]]
    assert "d1" not in upserted_ids
    assert "d2" in upserted_ids


def test_apply_resume_no_spans_passes_none_vector(
    sanitize_env, stubbed_engine
):
    """A resumed row already clean in Milvus (no spans) upserts with new_vector=None.

    The resume/FTS-only converge path must never send a 0-length vector — that
    would corrupt the fixed-dim HNSW row. It also performs no re-embed.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    # The row's Milvus document is already redacted (scan_spans yields no spans),
    # but the doc_id is still on the durable worklist from a prior crash — so the
    # resume pass must converge FTS only.
    cap["rows"] = [_milvus_row("d1", "totally benign already redacted text")]

    state_dir = Path(sanitize_env) / ".sessionflow"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sanitize_state.json").write_text(
        json.dumps(
            {
                "run_id": "prior",
                "mode": "redact",
                "scope": {"project_root": PROJECT},
                "worklist": ["d1"],
                "done": [],
                "counts": {},
                "status": "applying",
            }
        )
    )

    sanitize.apply(
        sanitize.Scope(project_root=PROJECT),
        drop=False,
        confirmed=True,
        resume=True,
    )

    assert cap["upserts"], "the FTS-only converge still calls upsert_document"
    doc_id, _new_doc, vector = cap["upserts"][0]
    assert doc_id == "d1"
    # None — not [] — so the stored vector is preserved, not zeroed.
    assert vector is None
    # No re-embed on the clean/resume path.
    assert cap["embed_inputs"] == []


# === Budget throttle ========================================================


def test_apply_reembed_invokes_embedding_budget(sanitize_env, stubbed_engine):
    """The re-embed path goes through the budget's before_batch/after_batch."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    sanitize.apply(sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True)

    kinds = [c[0] for c in cap["budget_calls"]]
    assert "before" in kinds, "re-embed must request budget via before_batch"
    assert "after" in kinds, "re-embed must report completion via after_batch"


def test_apply_budget_hard_deny_aborts_without_embed(sanitize_env, stubbed_engine):
    """A budget hard-deny (allowed=False, no retry) -> no embed, status='paused'.

    Mirrors rag_engine.add_turns: when the budget denies with no retry delay (a
    pause/cap), the run must NOT bypass the gate and embed — it aborts cleanly,
    checkpoints the un-processed turn, and reports status='paused'.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    class _HardDeny:
        allowed = False
        retry_after_seconds = 0.0
        reason = "paused"

    cap["before_decisions"] = [_HardDeny()]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )

    # The gate was respected: no embed, no upsert for the denied turn.
    assert cap["embed_inputs"] == [], "hard-deny must not embed past the gate"
    assert cap["upserts"] == []
    assert report.status == "paused"
    # The un-processed turn stays on the worklist for the next run.
    state_path = Path(sanitize_env) / ".sessionflow" / "sanitize_state.json"
    state = json.loads(state_path.read_text())
    assert "d1" in state.get("worklist", [])
    assert "d1" not in state.get("done", [])


def test_apply_budget_soft_deny_then_retry_embeds(sanitize_env, stubbed_engine):
    """A soft-deny (retry_after>0) sleeps then retries; the second decision allows."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    class _SoftDeny:
        allowed = False
        retry_after_seconds = 0.001
        reason = "cooldown"

    class _Allow:
        allowed = True
        retry_after_seconds = 0.0
        reason = ""

    cap["before_decisions"] = [_SoftDeny(), _Allow()]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )

    # After the cooldown the embed runs and the turn completes.
    assert cap["embed_inputs"], "soft-deny must retry and then embed"
    assert report.status == "complete"


# === Worklist is bounded (doc_ids only, JIT fetch) ==========================


def test_build_worklist_holds_doc_ids_not_full_rows(sanitize_env, stubbed_engine):
    """_build_worklist returns doc_ids + value-free metadata, never document text.

    Guards SESF-42 Fix A: the worklist must not cache every scanned row's payload
    (which OOMs on large indices). The per-turn metadata it does keep must never
    carry the ``document`` field.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [
        _milvus_row("d1", f"cred {AWS_KEY} end"),
        _milvus_row("d2", "totally benign text"),
    ]

    worklist, meta_by_id, _counts = sanitize._build_worklist(
        sanitize.Scope(project_root=PROJECT), [], None
    )

    # Only the secret-bearing turn is listed; the benign one is dropped.
    assert worklist == ["d1"]
    # Metadata is value-free: no document text is cached for any doc_id.
    for meta in meta_by_id.values():
        assert "document" not in meta
    assert set(meta_by_id) == {"d1"}


def test_apply_jit_fetches_document_via_get_row_by_doc_id(
    sanitize_env, stubbed_engine
):
    """apply() reads each affected row's document just-in-time, not from a cache."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    seen = []
    real = rag_engine.get_row_by_doc_id

    def spy(doc_id, db_path=None):
        seen.append(doc_id)
        return real(doc_id, db_path=db_path)

    import unittest.mock as _mock

    with _mock.patch.object(rag_engine, "get_row_by_doc_id", spy):
        sanitize.apply(
            sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
        )

    assert seen == ["d1"], "apply must JIT-fetch the row by doc_id"


def test_apply_resume_loads_worklist_and_run_id_without_rescan(
    sanitize_env, stubbed_engine, monkeypatch
):
    """resume=True loads worklist + run_id from the checkpoint, skipping _build_worklist.

    Guards SESF-42 Fix B/D: a resumed run must not re-scan (worklist comes from the
    checkpoint) and must reuse the prior run_id so the audit appends to the same
    redaction-<runid>.jsonl rather than starting a fresh trail.
    """
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d2", f"token {GH_KEY} end")]

    # Fail loudly if apply re-scans instead of trusting the checkpoint worklist.
    def _no_rescan(*a, **k):
        raise AssertionError("resume must not call _build_worklist")

    monkeypatch.setattr(sanitize, "_build_worklist", _no_rescan)

    state_dir = Path(sanitize_env) / ".sessionflow"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sanitize_state.json").write_text(
        json.dumps(
            {
                "run_id": "priorrun01",
                "mode": "redact",
                "scope": {"project_root": PROJECT},
                "worklist": ["d2"],
                "done": [],
                "counts": {"GITHUB": 1},
                "status": "applying",
            }
        )
    )

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT),
        drop=False,
        confirmed=True,
        resume=True,
    )

    # The worklist came from the checkpoint (no re-scan), d2 was processed.
    assert [u[0] for u in cap["upserts"]] == ["d2"]
    # Same run_id reused -> audit appends to redaction-priorrun01.jsonl.
    assert report.audit_path.endswith("redaction-priorrun01.jsonl")


def test_audit_writer_appends_on_resume(sanitize_env):
    """A second _AuditWriter for the same run_id appends, not truncates (Fix D)."""
    sanitize = _require("sanitize")

    class _Span:
        rule_name = "AWS"
        tier = 2
        start = 0
        end = 4
        masked_snippet = "AK..."

    row = {"doc_id": "d1", "provider": PROVIDER}

    w1 = sanitize._AuditWriter("sharedrun")
    w1.write(row=row, spans=[_Span()], action="redact")
    w1.close()

    # A resumed run reuses the same run_id; the prior line must survive.
    w2 = sanitize._AuditWriter("sharedrun")
    w2.write(row=row, spans=[_Span()], action="redact")
    w2.close()

    text = Path(w2.path).read_text()
    assert len(text.splitlines()) == 2, "append mode must preserve the prior line"
    # File is owner-only (0600) from creation.
    import stat

    mode = stat.S_IMODE(Path(w2.path).stat().st_mode)
    assert mode == 0o600


# === No-leak (cross-cutting) ================================================


def test_no_secret_leaks_to_stdout_or_audit_across_all_modes(
    sanitize_env, stubbed_engine, capsys
):
    """Across dry-run, redact-apply, and drop-apply: no synthetic secret anywhere."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine

    audit_paths = []

    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]
    audit_paths.append(sanitize.scan(sanitize.Scope(project_root=PROJECT)).audit_path)

    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]
    audit_paths.append(
        sanitize.apply(
            sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
        ).audit_path
    )

    cap["rows"] = [_milvus_row("d2", f"cred {AWS_KEY2} end")]
    audit_paths.append(
        sanitize.apply(
            sanitize.Scope(project_root=PROJECT), drop=True, confirmed=True
        ).audit_path
    )

    captured = capsys.readouterr()
    assert AWS_KEY not in captured.out and AWS_KEY not in captured.err
    assert AWS_KEY2 not in captured.out and AWS_KEY2 not in captured.err
    for path in audit_paths:
        text = Path(path).read_text()
        assert AWS_KEY not in text
        assert AWS_KEY2 not in text


# === rotate-the-key warning =================================================


def test_apply_report_carries_rotate_warning(sanitize_env, stubbed_engine):
    """An apply run flags rotate_warning (redaction != rotation; Requirement 7.1)."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )
    assert report.rotate_warning is True


# === rag_engine.upsert_document primitive ===================================


@pytest.fixture
def primitive_harness(monkeypatch):
    """Stub the Milvus client + FTS for the new write primitives (no real stores)."""
    import rag_engine

    cap = {
        "milvus_upserts": None,
        "milvus_deletes": [],
        "fts_ops": [],  # ordered ("delete"|"insert", payload)
        "existing_row": None,
    }

    class _Client:
        def has_collection(self, name):
            return True

        def query(self, **kw):
            return [cap["existing_row"]] if cap["existing_row"] is not None else []

        def upsert(self, collection_name, data):
            cap["milvus_upserts"] = data

        def delete(self, collection_name, filter):
            cap["milvus_deletes"].append(filter)

    @contextlib.contextmanager
    def _fake_milvus_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_milvus_client)

    def _fts_delete(conn, column, value):
        cap["fts_ops"].append(("delete", (column, value)))

    def _fts_insert(conn, records):
        cap["fts_ops"].append(("insert", records))

    fake_fts = types.SimpleNamespace(
        connection=lambda db_path: object(),
        delete=_fts_delete,
        insert=_fts_insert,
        close_ephemeral=lambda conn: None,
    )
    monkeypatch.setattr(rag_engine, "_fts", fake_fts)
    return rag_engine, cap


def test_upsert_document_milvus_upsert_and_fts_rewrite(primitive_harness):
    """upsert_document does a Milvus upsert + FTS delete-then-insert, returns ok/ok."""
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {
        "id": 123,
        "doc_id": "d1",
        "document": f"old {AWS_KEY}",
        "vector": [0.5] * 8,
        "provider": PROVIDER,
        "session_id": SESSION,
        "project_root": PROJECT,
    }
    result = rag_engine.upsert_document(
        "d1",
        new_document="redacted [REDACTED:AWS]",
        new_vector=[0.0] * 8,
        db_path="/tmp/sf-test.db",
    )

    assert result.milvus_ok is True
    assert result.fts_ok is True
    # Milvus row swapped: document + vector overwritten, metadata preserved.
    assert cap["milvus_upserts"] is not None
    row = cap["milvus_upserts"][0]
    assert row["document"] == "redacted [REDACTED:AWS]"
    assert row["vector"] == [0.0] * 8
    assert row["doc_id"] == "d1"
    assert AWS_KEY not in row["document"]
    # FTS rewrite is delete-then-insert (ordering matters).
    kinds = [op[0] for op in cap["fts_ops"]]
    assert kinds == ["delete", "insert"]
    inserted = cap["fts_ops"][1][1][0]
    assert inserted["content"] == "redacted [REDACTED:AWS]"


def test_upsert_document_none_vector_preserves_stored_vector(primitive_harness):
    """new_vector=None keeps the existing fixed-dim vector and still rewrites FTS.

    The resume/no-spans path passes None so a 0-length list can never clobber the
    HNSW/COSINE row; the document + FTS sidecar still converge.
    """
    rag_engine, cap = primitive_harness
    original_vector = [0.5] * 8
    cap["existing_row"] = {
        "id": 123,
        "doc_id": "d1",
        "document": "already redacted [REDACTED:AWS]",
        "vector": list(original_vector),
        "provider": PROVIDER,
        "session_id": SESSION,
        "project_root": PROJECT,
    }
    result = rag_engine.upsert_document(
        "d1",
        new_document="already redacted [REDACTED:AWS]",
        new_vector=None,
        db_path="/tmp/sf-test.db",
    )

    assert result.milvus_ok is True
    assert result.fts_ok is True
    row = cap["milvus_upserts"][0]
    # The stored vector is untouched — not zero-length, not replaced.
    assert row["vector"] == original_vector
    assert len(row["vector"]) == 8
    # FTS still converges (delete-then-insert of the redacted content).
    kinds = [op[0] for op in cap["fts_ops"]]
    assert kinds == ["delete", "insert"]
    assert cap["fts_ops"][1][1][0]["content"] == "already redacted [REDACTED:AWS]"


def test_upsert_document_fts_record_carries_source_metadata(primitive_harness):
    """The FTS rewrite copies every metadata column from the fetched row.

    Matches the normal-ingest FTS record shape so filtered/BM25 search survives a
    sanitize — content is the redacted document, metadata is the source row's.
    """
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {
        "id": 7,
        "doc_id": "d1",
        "document": f"old {AWS_KEY}",
        "vector": [0.1] * 8,
        "session_id": SESSION,
        "logical_session_id": "logical-xyz",
        "provider": PROVIDER,
        "source_kind": "transcript",
        "source_class": "cli",
        "source_id": "src-42",
        "source_path": "/synthetic/transcript.jsonl",
        "git_branch": "main",
        "turn_index": 5,
        "timestamp": "2026-05-21T10:00:00Z",
        "chunk_type": "turn",
        "project_root": PROJECT,
        "issue_ids": "SESF-42",
    }
    rag_engine.upsert_document(
        "d1",
        new_document="redacted [REDACTED:AWS]",
        new_vector=[0.0] * 8,
        db_path="/tmp/sf-test.db",
    )

    inserted = cap["fts_ops"][1][1][0]
    assert inserted["content"] == "redacted [REDACTED:AWS]"
    assert inserted["session_id"] == SESSION
    assert inserted["logical_session_id"] == "logical-xyz"
    assert inserted["provider"] == PROVIDER
    assert inserted["source_kind"] == "transcript"
    assert inserted["source_class"] == "cli"
    assert inserted["source_id"] == "src-42"
    assert inserted["source_path"] == "/synthetic/transcript.jsonl"
    assert inserted["git_branch"] == "main"
    assert inserted["turn_index"] == 5
    assert inserted["timestamp"] == "2026-05-21T10:00:00Z"
    assert inserted["chunk_type"] == "turn"
    assert inserted["project_root"] == PROJECT
    assert inserted["issue_ids"] == "SESF-42"


def test_upsert_document_fts_content_matches_truncated_milvus_document(primitive_harness):
    """FTS content uses the UTF-8-truncated row['document'], not raw new_document.

    A redacted payload can expand past Milvus's 65535-byte VARCHAR cap; both stores
    must index identical content or the dual-write contract breaks (CodeRabbit).
    """
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {
        "id": 9,
        "doc_id": "d1",
        "document": "old",
        "vector": [0.1] * 8,
        "session_id": SESSION,
        "provider": PROVIDER,
    }
    oversized = "x" * 70000  # > 65535 bytes -> Milvus truncates
    rag_engine.upsert_document(
        "d1", new_document=oversized, new_vector=[0.0] * 8, db_path="/tmp/sf-test.db"
    )

    milvus_doc = cap["milvus_upserts"][0]["document"]
    fts_content = cap["fts_ops"][1][1][0]["content"]
    assert fts_content == milvus_doc          # the two stores agree
    assert fts_content != oversized           # truncated, not the raw payload
    assert len(fts_content.encode("utf-8")) <= 65535


def test_upsert_document_fts_failure_reported_distinctly(primitive_harness):
    """An FTS-insert failure yields fts_ok=False (not swallowed) with milvus_ok=True."""
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {"id": 1, "doc_id": "d1", "document": "x"}

    def boom(conn, records):
        raise RuntimeError("fts insert failed")

    rag_engine._fts.insert = boom
    result = rag_engine.upsert_document(
        "d1", new_document="clean", new_vector=[0.0] * 8, db_path="/tmp/sf-test.db"
    )
    assert result.milvus_ok is True
    assert result.fts_ok is False


def test_delete_by_doc_id_dual_delete(primitive_harness):
    """delete_by_doc_id deletes from BOTH stores and reports the outcome distinctly."""
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {"id": 999, "doc_id": "d1", "document": "x"}

    result = rag_engine.delete_by_doc_id("d1", db_path="/tmp/sf-test.db")

    # Contract: returns a DeleteResult carrying the Milvus deleted count + fts_ok.
    assert isinstance(result, rag_engine.DeleteResult)
    assert result.deleted == 1
    assert result.fts_ok is True
    assert cap["milvus_deletes"], "Milvus delete must run"
    assert any(("delete", ("doc_id", "d1")) == op for op in cap["fts_ops"])


def test_delete_by_doc_id_fts_failure_reported_distinctly(primitive_harness):
    """An FTS-delete failure yields fts_ok=False even when the Milvus delete ran."""
    rag_engine, cap = primitive_harness
    cap["existing_row"] = {"id": 999, "doc_id": "d1", "document": "x"}

    def boom(conn, column, value):
        raise RuntimeError("fts delete failed")

    rag_engine._fts.delete = boom
    result = rag_engine.delete_by_doc_id("d1", db_path="/tmp/sf-test.db")

    assert result.deleted == 1  # Milvus delete still happened
    assert result.fts_ok is False  # but the FTS row may survive -> not done
    assert cap["milvus_deletes"], "Milvus delete must run"
