"""Memory lineage at read time: the newer record a memory points to, and the turns that match it.

The store is real Postgres (memory_pg); the index channels and transcript search are stubs.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    import memory_service
    from memory_service import MemoryService
    from memory_store import MemoryStore
except ImportError:
    memory_service = MemoryService = MemoryStore = None


class StubIndex:
    """Keyword channel under test control; vectors return nothing."""

    def __init__(self):
        self.fts_hits: list[dict] = []

    async def apply(self, memory_id):
        return "ok"

    async def similar(self, text, *, project, user_id, threshold):
        return [], "ok"

    async def vector_search(self, vector, *, filters, limit):
        return []

    async def fts_search(self, query, *, filters, limit):
        return self.fts_hits[:limit]

    def lagging(self):
        return False


async def _embed_query(text):
    return [0.0] * 768


@pytest.fixture
def store(memory_pg):
    s = MemoryStore(memory_pg)
    s.migrate()
    yield s
    s.close()


@pytest.fixture
def index():
    return StubIndex()


@pytest.fixture
def svc(store, index):
    return MemoryService(store=store, index=index, embed_query=_embed_query, allowlist=[],
                         similar_threshold=0.9, audit_age_days=30,
                         project_inventory={"sessionflow", "devops"})


def _add(svc, text, **over):
    args = {"text": text, "agent_id": "claude-code", "project": "sessionflow",
            "type": "session-summary"}
    args.update(over)
    return asyncio.run(svc.add(args))


def _hit(svc, written):
    row = asyncio.run(svc.get(written["memory_id"], project="sessionflow"))
    return {"memory_id": row["memory_id"], "version": row["version"], "generation": 1}


def _search(svc, index, hits, **filters):
    index.fts_hits = hits
    out = asyncio.run(svc.search("lineage", {"project": "sessionflow", **filters}, 10))
    return {r["memory_id"]: r for r in out["results"]}


# --- Newer record ------------------------------------------------------------


def _targets(row):
    return [(e["memory_id"], e["reason"], e["issues"]) for e in row.get("newer", [])]


def test_each_worked_issue_names_its_newest_later_record_rarest_first(svc, index):
    old = _add(svc, "triage wrap", extra={"issues": ["SESF-1", "SESF-2"]})
    _add(svc, "first follow-up on SESF-1", extra={"issues": ["SESF-1"]})
    on_two = _add(svc, "follow-up on SESF-2", type="decision", extra={"issues": ["SESF-2"]})
    on_one = _add(svc, "latest on SESF-1", type="handoff", extra={"issues": ["SESF-1"]})

    rows = _search(svc, index, [_hit(svc, old), _hit(svc, on_one)])

    # SESF-2 is in two records and SESF-1 in three, so SESF-2 comes first.
    assert _targets(rows[old["memory_id"]]) == [
        (on_two["memory_id"], "issues", ["SESF-2"]),
        (on_one["memory_id"], "issues", ["SESF-1"]),
    ]
    entry = rows[old["memory_id"]]["newer"][1]
    assert entry["type"] == "handoff" and entry["created_at"] is not None
    assert "newer" not in rows[on_one["memory_id"]]


def test_one_record_on_several_issues_is_named_once(svc, index):
    old = _add(svc, "wrap", extra={"issues": ["SESF-1", "SESF-2"]})
    both = _add(svc, "later wrap on both", extra={"issues": ["SESF-2", "SESF-1"]})

    rows = _search(svc, index, [_hit(svc, old)])

    assert _targets(rows[old["memory_id"]]) == [(both["memory_id"], "issues", ["SESF-1", "SESF-2"])]


def test_umbrella_issues_are_skipped(svc, index, monkeypatch):
    import memory_store

    monkeypatch.setattr(memory_store, "UMBRELLA_MIN_RECORDS", 4)
    old = _add(svc, "wrap on the epic and SESF-9", extra={"issues": ["SESF-61", "SESF-9"]})
    on_nine = _add(svc, "SESF-9 follow-up", extra={"issues": ["SESF-61", "SESF-9"]})
    for i in range(3):
        _add(svc, f"other child {i}", extra={"issues": ["SESF-61", f"SESF-{20 + i}"]})
        _add(svc, f"other work {i}", extra={"issues": [f"SESF-{30 + i}"]})

    rows = _search(svc, index, [_hit(svc, old)])

    # Of eight records, SESF-61 is in five (over UMBRELLA_SHARE) and SESF-9 in two.
    assert _targets(rows[old["memory_id"]]) == [(on_nine["memory_id"], "issues", ["SESF-9"])]


def test_no_umbrella_check_below_the_record_floor(svc, index):
    old = _add(svc, "wrap", extra={"issues": ["SESF-61"]})
    later = _add(svc, "later wrap", extra={"issues": ["SESF-61"]})

    rows = _search(svc, index, [_hit(svc, old)])

    assert _targets(rows[old["memory_id"]]) == [(later["memory_id"], "issues", ["SESF-61"])]


def test_at_most_max_newer_records_are_named(svc, index):
    import memory_store

    issues = [f"SESF-{i}" for i in range(1, memory_store.MAX_NEWER + 2)]
    old = _add(svc, "wide wrap", extra={"issues": issues})
    for issue in issues:
        _add(svc, f"follow-up on {issue}", extra={"issues": [issue]})

    rows = _search(svc, index, [_hit(svc, old)])

    named = [e["issues"] for e in rows[old["memory_id"]]["newer"]]
    assert named == [[i] for i in sorted(issues)[:memory_store.MAX_NEWER]]


def test_a_supersedes_pointer_comes_first(svc, index):
    old = _add(svc, "old decision", type="decision", extra={"issues": ["SESF-7"]})
    replacement = _add(svc, "replacement decision", type="decision",
                       supersedes=[old["memory_id"]])
    later = _add(svc, "later wrap on the same issue", extra={"issues": ["SESF-7"]})

    rows = _search(svc, index, [_hit(svc, old)], status="any")

    assert rows[old["memory_id"]]["status"] == "superseded"
    assert _targets(rows[old["memory_id"]]) == [
        (replacement["memory_id"], "supersedes", []),
        (later["memory_id"], "issues", ["SESF-7"]),
    ]


def test_records_without_worked_issues_get_no_newer_record(svc, index):
    # mem0 imports carry issue_ids from their text but no extra.issues.
    old = _add(svc, "imported note about SESF-3", type="decision")
    _add(svc, "later wrap about SESF-3", extra={"issues": ["SESF-3"]})

    rows = _search(svc, index, [_hit(svc, old)])

    assert "newer" not in rows[old["memory_id"]]


@pytest.mark.parametrize("over", [
    {"project": "devops"},
    {"type": "fact"},
    {"type": "open-item"},
])
def test_other_projects_and_types_are_not_newer_records(svc, index, over):
    old = _add(svc, "wrap on SESF-4", extra={"issues": ["SESF-4"]})
    _add(svc, f"later record {over}", extra={"issues": ["SESF-4"]}, **over)

    rows = _search(svc, index, [_hit(svc, old)])

    assert "newer" not in rows[old["memory_id"]]


def test_a_superseded_newer_record_is_skipped(svc, index):
    old = _add(svc, "wrap one", extra={"issues": ["SESF-5"]})
    later = _add(svc, "wrap two", extra={"issues": ["SESF-5"]})
    _add(svc, "retires wrap two", type="decision", supersedes=[later["memory_id"]])

    rows = _search(svc, index, [_hit(svc, old)])

    assert "newer" not in rows[old["memory_id"]]


def test_get_names_the_newer_record(svc):
    old = _add(svc, "wrap one", extra={"issues": ["SESF-6"]})
    new = _add(svc, "wrap two", type="handoff", extra={"issues": ["SESF-6"]})

    row = asyncio.run(svc.get(old["memory_id"], project="sessionflow"))

    assert _targets(row) == [(new["memory_id"], "issues", ["SESF-6"])]
    assert row["newer"][0]["type"] == "handoff"


def test_a_failed_lineage_read_keeps_the_search_result(svc, index, store, monkeypatch):
    old = _add(svc, "wrap one", extra={"issues": ["SESF-8"]})
    _add(svc, "wrap two", extra={"issues": ["SESF-8"]})

    def boom(rows):
        raise RuntimeError("lineage query failed")

    monkeypatch.setattr(store, "newer_records", boom)
    rows = _search(svc, index, [_hit(svc, old)])

    assert list(rows) == [old["memory_id"]]
    assert "newer" not in rows[old["memory_id"]]


# --- Matching turns ----------------------------------------------------------


def _stub_context(monkeypatch, search):
    import rag_engine

    monkeypatch.setattr(rag_engine, "session_turns",
                        lambda session_id, provider, *, limit: [{"turn_index": 40, "content": "last"}])
    monkeypatch.setattr(rag_engine, "sessions_by_issue", lambda *a, **k: [])
    monkeypatch.setattr(rag_engine, "search_async", search)


def test_context_returns_the_source_turns_that_match_the_memory(svc, monkeypatch):
    head = "Chose read-time links for memory lineage. " * 60
    written = _add(svc, head + "\n\nSecond paragraph with other detail.",
                   source_session_id="sess-src", source_provider="claude_code_cli")
    calls = []

    async def fake_search(query, n=5, **kwargs):
        calls.append((query, n, kwargs))
        return [{"turn_index": 12, "timestamp": "2026-11-03T02:00:00Z", "content": "we chose it",
                 "session_id": "sess-src", "doc_id": "d12"}]

    _stub_context(monkeypatch, fake_search)
    out = asyncio.run(svc.context(written["memory_id"], project="sessionflow", limit=4))

    assert out["source_session"]["matched_turns"] == [
        {"turn_index": 12, "timestamp": "2026-11-03T02:00:00Z", "content": "we chose it"}]
    assert out["source_session"]["turns"] == [{"turn_index": 40, "content": "last"}]
    query, n, kwargs = calls[0]
    assert n == 4
    assert kwargs["session_id"] == "sess-src"
    assert kwargs["provider"] == "claude_code_cli"
    assert kwargs["sort_by"] == "relevance"
    assert "Second paragraph" not in query
    assert len(query) <= memory_service.CONTEXT_QUERY_CHARS
    assert query.startswith("Chose read-time links")


def test_context_skips_matching_for_codex_sessions(svc, monkeypatch):
    written = _add(svc, "codex wrap", source_session_id="rollout-1", source_provider="codex")
    called = []

    async def fake_search(*a, **k):
        called.append(1)
        return []

    _stub_context(monkeypatch, fake_search)
    out = asyncio.run(svc.context(written["memory_id"], project="sessionflow"))

    assert out["source_session"]["matched_turns"] == []
    assert called == []


def test_a_failed_turn_search_keeps_the_context(svc, monkeypatch):
    written = _add(svc, "wrap", source_session_id="sess-src", source_provider="claude_code_cli")

    async def fake_search(*a, **k):
        raise RuntimeError("Milvus down")

    _stub_context(monkeypatch, fake_search)
    out = asyncio.run(svc.context(written["memory_id"], project="sessionflow"))

    assert out["source_session"]["matched_turns"] == []
    assert out["source_session"]["turns"] == [{"turn_index": 40, "content": "last"}]
