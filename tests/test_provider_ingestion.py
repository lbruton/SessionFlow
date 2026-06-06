"""SESF-10 regression tests for provider parse-to-index execution.

Also carries SESF-25 (Req 1.1 / 1.5 / 2.3) red-phase tests asserting that
``add_turns`` writes an ``issue_ids`` field into BOTH the Milvus insert dict and
the FTS record dict, uniformly across every provider adapter's turns.
"""

import importlib
import os
import tempfile
from contextlib import contextmanager

import pytest

rag_engine = importlib.import_module("rag_engine")


class _FakeDecision:
    """Stand-in for the embedding-budget decision object."""

    allowed = True
    retry_after_seconds = 0.0
    reason = ""


class _FakeBudget:
    """Minimal embedding budget that passes a single batch straight through."""

    def split_batches(self, turns):
        return [turns]

    def before_batch(self, batch_size, estimated_chars):
        return _FakeDecision()

    def after_batch(self, elapsed, count, error=None):
        return None


class _CapturingMilvus:
    """Fake Milvus client capturing dedup queries and insert payloads."""

    def __init__(self):
        self.inserted: list[dict] = []

    def query(self, *args, **kwargs):
        return []  # nothing pre-exists → every turn is new

    def insert(self, *args, **kwargs):
        self.inserted.extend(kwargs.get("data", []))


def _capture_add_turns(monkeypatch, turns):
    """Run ``add_turns`` offline, returning (milvus_rows, fts_records)."""
    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _FakeBudget())
    monkeypatch.setattr(
        rag_engine, "embed_texts", lambda texts, is_query=False: [[0.0] * 768 for _ in texts]
    )

    client = _CapturingMilvus()

    @contextmanager
    def _fake_client(db_path=None):
        yield client

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)

    fts_records: list[dict] = []
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: object())
    monkeypatch.setattr(
        rag_engine._fts, "insert", lambda conn, records: fts_records.extend(records)
    )
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)

    rag_engine.add_turns(turns, db_path=os.path.join(tempfile.gettempdir(), "sessionflow-ingest-test.db"))
    return client.inserted, fts_records


# All five provider identities the multi-harness pipeline tags turns with.
_PROVIDERS = [
    "claude_code_cli",
    "codex",
    "opencode",
    "antigravity_cli",
    "antigravity_desktop",
]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_add_turns_milvus_dict_carries_issue_ids(provider, monkeypatch):
    # Req 1.1 / 1.5 / 2.3 — the Milvus insert row gets issue_ids from the text.
    turns = [{
        "text": f"Working on SESF-25 via {provider}",
        "doc_id": f"doc-{provider}",
        "session_id": "s1",
        "timestamp": "2026-05-01T10:00:00",
        "provider": provider,
    }]
    milvus_rows, _ = _capture_add_turns(monkeypatch, turns)
    assert milvus_rows
    row = milvus_rows[0]
    assert "issue_ids" in row
    assert row["issue_ids"] == ",SESF-25,"


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_add_turns_fts_dict_carries_issue_ids(provider, monkeypatch):
    # Req 1.1 / 1.5 / 2.3 — the FTS record gets issue_ids from the text.
    turns = [{
        "text": f"Working on SESF-25 via {provider}",
        "doc_id": f"doc-fts-{provider}",
        "session_id": "s1",
        "timestamp": "2026-05-01T10:00:00",
        "provider": provider,
    }]
    _, fts_records = _capture_add_turns(monkeypatch, turns)
    assert fts_records
    rec = fts_records[0]
    assert "issue_ids" in rec
    assert rec["issue_ids"] == ",SESF-25,"


def test_add_turns_empty_issue_ids_when_no_token(monkeypatch):
    # Req 1.3 — a turn with no issue token still ingests, with an empty field.
    turns = [{
        "text": "no tracker reference here",
        "doc_id": "doc-none",
        "session_id": "s1",
        "timestamp": "2026-05-01T10:00:00",
    }]
    milvus_rows, fts_records = _capture_add_turns(monkeypatch, turns)
    assert milvus_rows and fts_records
    assert milvus_rows[0]["issue_ids"] == ""
    assert fts_records[0]["issue_ids"] == ""


def test_backfill_fts_rehydrates_issue_ids(monkeypatch):
    # Req 1.5 — issue_ids fetched from Milvus must survive the FTS re-hydration
    # round-trip in backfill_fts(); a dropped key silently zeroes the field.
    doc_id = "doc-backfill"

    class _FTSConn:
        def execute(self, *args, **kwargs):
            class _Cursor:
                def fetchall(self):
                    return []  # nothing in FTS yet → the doc is "missing"

            return _Cursor()

    class _BackfillMilvus:
        def query(self, *args, **kwargs):
            return [{
                "doc_id": doc_id,
                "document": "Working on SESF-25",
                "session_id": "s1",
                "timestamp": "2026-05-01T10:00:00",
                "issue_ids": ",SESF-25,",
            }]

    @contextmanager
    def _fake_client(db_path=None):
        yield _BackfillMilvus()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *args, **kwargs: iter([[{"doc_id": doc_id}]]),
    )

    fts_records: list[dict] = []
    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: _FTSConn())
    monkeypatch.setattr(
        rag_engine._fts, "insert", lambda conn, records: fts_records.extend(records)
    )
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)

    inserted = rag_engine.backfill_fts(db_path=os.path.join(tempfile.gettempdir(), "sessionflow-backfill-test.db"))

    assert inserted == 1
    assert fts_records
    rec = fts_records[0]
    assert "issue_ids" in rec
    assert rec["issue_ids"] == ",SESF-25,"


def _run_backfill_with_counts(monkeypatch, *, milvus_count, fts_count_after):
    """Drive backfill_fts() offline with controllable Milvus + post-hydrate FTS counts.

    Returns the number of ``clear_fts_backfill_sentinel`` invocations observed so
    the D-5 sentinel-clear gate can be asserted in both directions.

    Arrangement:
      * Milvus exposes ``milvus_count`` distinct doc_ids (the "truth" row count).
      * FTS starts empty, so every doc is treated as missing and re-hydrated.
      * After hydration, ``_fts.count_rows`` reports ``fts_count_after`` — the lever
        that decides whether the sentinel may be cleared.
    """
    import fts_hybrid

    doc_ids = [f"doc-{i}" for i in range(milvus_count)]

    class _FTSConn:
        def execute(self, *args, **kwargs):
            class _Cursor:
                def fetchall(self):
                    return []  # FTS empty → every Milvus doc is "missing"

            return _Cursor()

    class _BackfillMilvus:
        def query(self, *args, **kwargs):
            return [
                {
                    "doc_id": d,
                    "document": "Working on SESF-38",
                    "session_id": "s1",
                    "timestamp": "2026-05-01T10:00:00",
                    "issue_ids": ",SESF-38,",
                }
                for d in doc_ids
            ]

    @contextmanager
    def _fake_client(db_path=None):
        yield _BackfillMilvus()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(
        rag_engine, "_query_batches",
        lambda *args, **kwargs: iter([[{"doc_id": d} for d in doc_ids]]),
    )

    monkeypatch.setattr(rag_engine._fts, "connection", lambda db_path: _FTSConn())
    monkeypatch.setattr(rag_engine._fts, "insert", lambda conn, records: None)
    monkeypatch.setattr(rag_engine._fts, "close_ephemeral", lambda conn: None)
    # Post-hydrate FTS row count is the gate input for clearing the sentinel.
    monkeypatch.setattr(
        rag_engine._fts, "count_rows", lambda conn, project_root=None: fts_count_after
    )

    cleared = {"count": 0}

    def _spy_clear():
        cleared["count"] += 1

    # backfill_fts imports clear_fts_backfill_sentinel from fts_hybrid at call time.
    monkeypatch.setattr(fts_hybrid, "clear_fts_backfill_sentinel", _spy_clear)

    rag_engine.backfill_fts(
        db_path=os.path.join(tempfile.gettempdir(), "sessionflow-d5-test.db")
    )
    return cleared["count"]


def test_backfill_fts_keeps_sentinel_when_hydration_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the sentinel set when post-hydration FTS count is below Milvus count."""
    # D-5 — if the post-hydrate FTS count is still BELOW the Milvus count, the
    # sentinel MUST remain set (keyword search is still degraded): backfill_fts
    # only clears once FTS catches up, so the sentinel stays set while FTS lags
    # Milvus and this asserts 0 clears.
    clears = _run_backfill_with_counts(monkeypatch, milvus_count=10, fts_count_after=4)
    assert clears == 0


def test_backfill_fts_clears_sentinel_when_hydration_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the sentinel when post-hydration FTS count reaches Milvus count."""
    # D-5 — once the FTS count reaches (or exceeds) the Milvus count, the sentinel
    # IS cleared. This direction passes today (unconditional clear) and must keep
    # passing after C gates the clear, pinning the positive branch.
    clears = _run_backfill_with_counts(monkeypatch, milvus_count=10, fts_count_after=10)
    assert clears == 1


@pytest.mark.anyio
async def test_provider_ingestion_drains_codex_job_to_index_and_cursor(
    tmp_path,
    synthetic_codex_home,
    monkeypatch,
):
    import provider_ingestion
    import transcript_parser
    from backfill_manager import BackfillManager
    from provider_codex import CodexAdapter

    state_path = tmp_path / ".sessionflow" / "index_state.json"
    monkeypatch.setattr(transcript_parser, "_STATE_DIR", state_path.parent)
    monkeypatch.setattr(transcript_parser, "_STATE_PATH", state_path)

    indexed_batches = []

    async def fake_add_turns(turns, db_path=None):
        indexed_batches.append((turns, db_path))
        return len(turns)

    monkeypatch.setattr(provider_ingestion.rag_engine, "add_turns_async", fake_add_turns)

    manager = BackfillManager(tmp_path / ".sessionflow" / "backfill_state.json")
    manager.enqueue_provider_backfill(provider="codex", mode="recent")
    service = provider_ingestion.ProviderIngestionService(
        manager,
        db_path=str(tmp_path / "milvus.db"),
        adapters={"codex": CodexAdapter(home=synthetic_codex_home)},
    )

    result = await service.process_queued_jobs()

    assert result["jobs"] == 1
    assert result["processed_sources"] == 1
    assert result["indexed_turns"] == 1
    assert indexed_batches
    assert indexed_batches[0][0][0]["provider"] == "codex"
    assert "synthetic codex question" in indexed_batches[0][0][0]["text"]
    state = transcript_parser.load_index_state()
    assert state["providers"]["codex"]
    assert manager.status().jobs == []


def test_provider_ingestion_startup_uses_recent_not_full(tmp_path):
    from backfill_manager import BackfillManager

    manager = BackfillManager(tmp_path / "backfill_state.json")
    manager.enqueue_startup_defaults(
        enabled_providers=["claude_code_cli", "codex", "opencode"],
        mode="full",
    )

    assert {job.mode for job in manager.status().jobs} == {"recent"}
