"""TDD red-phase tests for the SESF-42 retroactive secret sanitizer.

Two cohorts live here:

* **Orchestrator** — ``sanitize.Scope`` / ``sanitize.scan`` / ``sanitize.apply``
  and the ``SanitizeReport`` it returns. The orchestrator drives the dry-run/apply
  flow, throttling, checkpointing, and the secret-free audit trail (design.md
  Component 3). It owns policy only — detection routes through
  ``secret_redaction.scan_spans`` and data access through the two new
  ``rag_engine`` primitives.
* **Primitives** — ``rag_engine.upsert_document`` / ``rag_engine.delete_by_doc_id``
  (design.md Component 2): the in-place Milvus overwrite + FTS rewrite, and the
  doc-id-scoped dual delete.

None of the target code exists yet; every test here is RED until SESF-42 lands.

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
    }

    def fake_query_batches(output_fields, *args, **kwargs):
        if cap["rows"]:
            yield list(cap["rows"])

    monkeypatch.setattr(rag_engine, "_query_batches", fake_query_batches, raising=False)

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


def test_apply_without_confirmation_makes_no_calls(sanitize_env, stubbed_engine):
    """apply(confirmed=False) performs no reads or writes."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]

    sanitize.apply(sanitize.Scope(project_root=PROJECT), drop=False, confirmed=False)

    assert cap["upserts"] == []
    assert cap["deletes"] == []
    assert cap["embed_inputs"] == []


# === FTS-failure incompletion ===============================================


def test_apply_fts_failure_leaves_row_incomplete(sanitize_env, stubbed_engine):
    """fts_ok=False -> doc not 'done', incomplete_fts increments, status='incomplete'."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]
    # Milvus succeeded, but the FTS rewrite failed.
    cap["upsert_result"] = rag_engine.UpsertResult(milvus_ok=True, fts_ok=False)

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=False, confirmed=True
    )

    assert report.incomplete_fts >= 1
    assert report.status == "incomplete"  # NOT complete while FTS-failed rows remain

    # The doc_id stays in the checkpoint worklist, off the `done` set.
    state_path = Path(sanitize_env) / ".sessionflow" / "sanitize_state.json"
    state = json.loads(state_path.read_text())
    assert "d1" not in state.get("done", [])
    assert "d1" in state.get("worklist", [])


def test_apply_drop_fts_failure_leaves_row_incomplete(sanitize_env, stubbed_engine):
    """drop path: an FTS-delete failure must NOT mark the doc done (mirrors redact)."""
    sanitize = _require("sanitize")
    rag_engine, cap = stubbed_engine
    cap["rows"] = [_milvus_row("d1", f"cred {AWS_KEY} end")]
    # Milvus delete ran, but the FTS row may still carry the secret.
    cap["delete_result"] = rag_engine.DeleteResult(deleted=1, fts_ok=False)

    report = sanitize.apply(
        sanitize.Scope(project_root=PROJECT), drop=True, confirmed=True
    )

    assert report.incomplete_fts >= 1
    assert report.status == "incomplete"

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
