"""Memory service: validation, write orchestration, search pipeline, audit and context.

The store is real Postgres (memory_pg); the index channels and session lookups are
stubs, so each test controls exactly which hits the search pipeline sees.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest

# A collection-time ImportError would stop the whole test session.
try:
    import memory_service
    from memory_service import MemoryInputError, MemoryService
    from memory_store import MemoryStore, MemoryStoreError
except ImportError:
    memory_service = MemoryInputError = MemoryService = MemoryStore = MemoryStoreError = None

AWS_KEY = "AKIA" + "1234567890ABCDEF"
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class StubIndex:
    """Index channels under test control. Hits are dicts of memory_id, version, generation."""

    def __init__(self):
        self.vector_hits: list[dict] = []
        self.fts_hits: list[dict] = []
        self.vector_error: Exception | None = None
        self.lag = False
        self.applied: list[str] = []
        self.calls: list[tuple[str, int]] = []
        self.similar_result: tuple[list, str] = ([], "ok")
        self.neighbour_map: dict[str, list[str]] = {}

    async def apply(self, memory_id):
        self.applied.append(memory_id)
        return "ok"

    async def similar(self, text, *, project, user_id, threshold):
        return self.similar_result

    async def vector_search(self, vector, *, filters, limit):
        self.calls.append(("vector", limit))
        if self.vector_error:
            raise self.vector_error
        return self.vector_hits[:limit]

    async def fts_search(self, query, *, filters, limit):
        self.calls.append(("fts", limit))
        return self.fts_hits[:limit]

    def lagging(self):
        return self.lag

    async def neighbours(self, memory_id, *, threshold):
        return self.neighbour_map.get(memory_id, [])


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
    args = {"text": text, "agent_id": "claude-code", "project": "sessionflow", "type": "fact"}
    args.update(over)
    return asyncio.run(svc.add(args))


def _err(fn, *args):
    with pytest.raises((MemoryInputError, MemoryStoreError)) as exc:
        asyncio.run(fn(*args))
    return exc.value


# --- Size, bytes and secrets (AC-3, AC-4, AC-5) ------------------------------


def test_text_at_the_byte_limit_round_trips_exactly(svc):
    unit = "é漢😀"  # 2 + 3 + 4 bytes
    text = unit * (65536 // 9) + "a" * (65536 % 9)
    assert len(text.encode("utf-8")) == 65536
    written = _add(svc, text)
    got = asyncio.run(svc.get(written["memory_id"], project="sessionflow"))
    assert got["text"] == text
    listed = asyncio.run(svc.list({"project": "sessionflow"}, limit=5))
    assert listed["results"][0]["text"] == text


def test_text_over_the_limit_is_rejected_with_both_sizes(svc, memory_pg):
    err = _err(svc.add, {"text": "a" * 65537, "agent_id": "codex", "project": "sessionflow",
                         "type": "fact"})
    assert err.code == "too_large"
    assert "65536" in str(err) and "65537" in str(err)
    with psycopg.connect(memory_pg) as conn:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0


def test_secret_rejects_the_write_and_names_the_rule(svc, memory_pg):
    err = _err(svc.add, {"text": f"creds {AWS_KEY}", "agent_id": "codex",
                         "project": "sessionflow", "type": "fact"})
    assert err.code == "secret_detected" and "AWS" in str(err)
    assert AWS_KEY not in str(err)
    with psycopg.connect(memory_pg) as conn:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0


def test_allowlisted_and_unforced_entropy_text_is_stored(store, index):
    svc = MemoryService(store=store, index=index, embed_query=_embed_query,
                        allowlist=[re.compile(re.escape(AWS_KEY))], similar_threshold=0.9,
                        audit_age_days=30, project_inventory={"sessionflow"})
    assert _add(svc, f"documented example key {AWS_KEY}")["outcome"] == "created"
    assert _add(svc, "entropy-ish Zx9Qw3Er7Ty1Ui5Op2As8Df4Gh6Jk0Lz")["outcome"] == "created"


# --- Required fields and vocabularies (AC-8 to AC-10) ------------------------


@pytest.mark.parametrize("field", ["agent_id", "project", "type"])
def test_missing_required_field_is_named(svc, field):
    args = {"text": "x", "agent_id": "codex", "project": "sessionflow", "type": "fact"}
    del args[field]
    err = _err(svc.add, args)
    assert err.code == "missing_field" and field in str(err)


def test_user_id_defaults_to_lbruton(svc):
    written = _add(svc, "default user")
    assert asyncio.run(svc.get(written["memory_id"], project="sessionflow"))["user_id"] == "lbruton"


def test_bad_type_lists_the_allowed_values(svc):
    err = _err(svc.add, {"text": "x", "agent_id": "codex", "project": "sessionflow", "type": "note"})
    assert err.code == "invalid_type"
    for allowed in ("session-summary", "decision", "guardrail"):
        assert allowed in str(err)


@pytest.mark.parametrize("agent", ["claude_code_cli", "unknown", "cowork"])
def test_bad_agent_lists_the_allowed_values(svc, agent):
    err = _err(svc.add, {"text": "x", "agent_id": agent, "project": "sessionflow", "type": "fact"})
    assert err.code == "invalid_agent"
    assert "claude-code" in str(err) and "antigravity" in str(err)


# --- Issue ids (AC-11) -------------------------------------------------------


def test_issue_ids_come_from_the_transcript_extractor(svc):
    written = _add(svc, "Fixed devs-145 and STRK-12; SHA-256 unchanged")
    got = asyncio.run(svc.get(written["memory_id"], project="sessionflow"))
    assert got["issue_ids"] == ["DEVS-145", "STRK-12"]


# --- Write orchestration (AC-20, AC-26, AC-36) -------------------------------


def test_add_indexes_after_commit_and_reports_similar(svc, index):
    index.similar_result = ([{"memory_id": "m-1", "score": 0.97, "text": "near"}], "ok")
    written = _add(svc, "indexed")
    assert index.applied == [written["memory_id"]]
    assert written["index"] == "ok"
    assert written["similar"] == [{"memory_id": "m-1", "score": 0.97, "text": "near"}]
    assert written["similar_state"] == "ok"


def test_failed_commit_touches_no_index(svc, store, index):
    def boom(stage, conn):
        if stage == "before_commit":
            raise RuntimeError("injected")

    store._fault_hook = boom
    with pytest.raises(Exception):
        _add(svc, "never committed")
    store._fault_hook = None
    assert index.applied == []


def test_update_recomputes_derived_fields(svc):
    first = _add(svc, "about DEVS-1", topic_key="t1", extra={"source": "wrap"})
    asyncio.run(svc.update(first["memory_id"], {"text": "about DEVS-2", "project": "sessionflow"}))
    got = asyncio.run(svc.get(first["memory_id"], project="sessionflow"))
    assert got["issue_ids"] == ["DEVS-2"]
    assert got["topic_key"] == "t1" and got["extra"] == {"source": "wrap"}
    assert got["version"] == 2


# --- Project scope (D-11) ----------------------------------------------------


def test_scoped_calls_need_a_project(svc):
    assert _err(svc.search, "anything", {}, 5).code == "project_required"
    assert _err(svc.list, {}, 5).code == "project_required"
    star = _err(svc.add, {"text": "x", "agent_id": "codex", "project": "*", "type": "fact"})
    assert star.code == "invalid_project"


def test_star_lists_every_project(svc):
    _add(svc, "here")
    _add(svc, "there", project="devops")
    rows = asyncio.run(svc.list({"project": "*"}, limit=10))["results"]
    assert {r["project"] for r in rows} == {"sessionflow", "devops"}


def test_id_calls_reject_a_mismatched_project(svc):
    written = _add(svc, "mine")
    err = _err(svc.get, written["memory_id"], "devops")
    assert err.code == "project_mismatch"


# --- Search pipeline (AC-17, AC-18, AC-37, D-3) -----------------------------


def _hit(row, **over):
    hit = {"memory_id": row["memory_id"], "version": row["version"], "generation": 1}
    hit.update(over)
    return hit


def test_stale_vector_hit_does_not_hide_a_current_fts_hit(svc, index):
    written = _add(svc, "first wording")
    asyncio.run(svc.update(written["memory_id"], {"text": "second wording", "project": "sessionflow"}))
    index.vector_hits = [{"memory_id": written["memory_id"], "version": 1, "generation": 1}]
    index.fts_hits = [{"memory_id": written["memory_id"], "version": 2, "generation": 1}]

    out = asyncio.run(svc.search("wording", {"project": "sessionflow"}, 5))

    assert [r["memory_id"] for r in out["results"]] == [written["memory_id"]]
    assert out["results"][0]["text"] == "second wording"
    assert out["channels"] == ["vector", "fts"]


def test_deleted_hits_are_dropped(svc, index):
    gone = _add(svc, "deleted later")
    kept = _add(svc, "kept")
    asyncio.run(svc.delete(gone["memory_id"], project="sessionflow"))
    index.fts_hits = [{"memory_id": gone["memory_id"], "version": 1, "generation": 1},
                      {"memory_id": kept["memory_id"], "version": 1, "generation": 1}]

    out = asyncio.run(svc.search("x", {"project": "sessionflow"}, 5))
    assert [r["memory_id"] for r in out["results"]] == [kept["memory_id"]]


def test_short_result_refills_once_with_a_larger_fetch(svc, index):
    stale = [{"memory_id": "00000000-0000-0000-0000-00000000000%d" % i, "version": 1,
              "generation": 1} for i in range(6)]
    real = [_add(svc, f"real {i}") for i in range(2)]
    index.fts_hits = stale + [{"memory_id": r["memory_id"], "version": 1, "generation": 1} for r in real]

    out = asyncio.run(svc.search("real", {"project": "sessionflow"}, 2))

    fts_limits = [limit for kind, limit in index.calls if kind == "fts"]
    assert len(fts_limits) == 2 and fts_limits[0] == 6 and fts_limits[1] > fts_limits[0]
    assert len(out["results"]) == 2 and out["incomplete"] is False


def test_still_short_under_index_lag_is_incomplete(svc, index):
    written = _add(svc, "only one")
    index.fts_hits = [{"memory_id": written["memory_id"], "version": 1, "generation": 1}]
    index.lag = True
    out = asyncio.run(svc.search("only", {"project": "sessionflow"}, 3))
    assert len(out["results"]) == 1 and out["incomplete"] is True


def test_milvus_error_falls_back_to_keyword_search(svc, index):
    written = _add(svc, "keyword only")
    index.vector_error = RuntimeError("milvus down")
    index.fts_hits = [{"memory_id": written["memory_id"], "version": 1, "generation": 1}]
    out = asyncio.run(svc.search("keyword", {"project": "sessionflow"}, 5))
    assert out["channels"] == ["fts"]
    assert [r["memory_id"] for r in out["results"]] == [written["memory_id"]]


def test_canonical_filters_are_rechecked(svc, index):
    decision = _add(svc, "a decision", type="decision", agent_id="codex",
                    extra={"source": "wrap", "category": "infra"})
    fact = _add(svc, "a fact about DEVS-7", extra={"source": "discover"})
    other = _add(svc, "another project", project="devops")
    index.fts_hits = [{"memory_id": r["memory_id"], "version": 1, "generation": 1}
                      for r in (decision, fact, other)]

    def ids(**f):
        out = asyncio.run(svc.search("x", dict(project="sessionflow", **f), 10))
        return {r["memory_id"] for r in out["results"]}

    assert ids() == {decision["memory_id"], fact["memory_id"]}
    assert ids(type="decision") == {decision["memory_id"]}
    assert ids(agent_id="codex") == {decision["memory_id"]}
    assert ids(issue_id="DEVS-7") == {fact["memory_id"]}
    assert ids(source="discover") == {fact["memory_id"]}
    assert ids(category="infra") == {decision["memory_id"]}
    assert ids(user_id="someone-else") == set()


def _backdated(store, text, created_at, **over):
    """Write through the store: only the import path may set created_at."""
    import hashlib

    from memory_store import NewMemory

    fields = dict(text=text, agent_id="claude-code", project="sessionflow", type="fact",
                  content_hash=hashlib.sha256(text.encode()).hexdigest(),
                  created_at=created_at, updated_at=created_at)
    fields.update(over)
    return store.add(NewMemory(**fields))


def test_search_date_filter_uses_chicago_days(svc, store, index):
    late = _backdated(store, "late on the 30th", datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc))
    early = _backdated(store, "early on the 1st", datetime(2026, 10, 1, 5, 30, tzinfo=timezone.utc))
    index.fts_hits = [{"memory_id": r.memory_id, "version": 1, "generation": 1} for r in (late, early)]
    out = asyncio.run(svc.search("x", {"project": "sessionflow", "created_from": "2026-10-01"}, 10))
    assert [r["memory_id"] for r in out["results"]] == [early.memory_id]


def test_default_search_returns_active_records_only(svc, index):
    old = _add(svc, "old version")
    new = _add(svc, "new version", supersedes=[old["memory_id"]])
    index.fts_hits = [{"memory_id": old["memory_id"], "version": 2, "generation": 1},
                      {"memory_id": new["memory_id"], "version": 1, "generation": 1}]
    active = asyncio.run(svc.search("version", {"project": "sessionflow"}, 10))
    assert [r["memory_id"] for r in active["results"]] == [new["memory_id"]]
    sup = asyncio.run(svc.search("version", {"project": "sessionflow", "status": "superseded"}, 10))
    assert [r["memory_id"] for r in sup["results"]] == [old["memory_id"]]


# --- Audit (AC-38) -----------------------------------------------------------


def test_audit_reports_without_changing_anything(svc, store, index, memory_pg):
    a = _add(svc, "near one")
    b = _add(svc, "near two")
    stale = _backdated(store, "old open item", datetime.now(timezone.utc) - timedelta(days=45),
                       type="open-item")
    _add(svc, "fresh blocker", type="blocker")
    _add(svc, "stray", project="mystery")
    index.neighbour_map = {a["memory_id"]: [b["memory_id"]], b["memory_id"]: [a["memory_id"]]}

    with psycopg.connect(memory_pg) as conn:
        before = conn.execute("SELECT memory_id, version, status FROM memories ORDER BY 1").fetchall()

    report = asyncio.run(svc.audit())

    sf = report["projects"]["sessionflow"]
    assert [sorted(c) for c in sf["clusters"]] == [sorted([a["memory_id"], b["memory_id"]])]
    assert sf["stale"] == [stale.memory_id]
    assert sf["counts"]["fact"]["active"] == 2
    assert report["unknown_projects"] == ["mystery"]
    with psycopg.connect(memory_pg) as conn:
        after = conn.execute("SELECT memory_id, version, status FROM memories ORDER BY 1").fetchall()
    assert after == before


# --- Related conversations (AC-41, AC-42, D-22) ------------------------------


def test_context_returns_source_turns_and_issue_sessions(svc, monkeypatch):
    import rag_engine

    written = _add(svc, "wrap for DEVS-145", type="session-summary",
                   source_session_id="sess-src", source_provider="claude_code_cli",
                   source_project_root="/Volumes/DATA/GitHub/SessionFlow")
    calls = {}

    def fake_turns(session_id, provider, *, limit):
        calls["turns"] = (session_id, provider, limit)
        return [{"turn_index": 0, "content": "hello"}]

    def fake_by_issue(issue_ids, *, limit, project_roots=None, day=None, exclude_session=None):
        calls["by_issue"] = (issue_ids, project_roots, exclude_session)
        return [{"session_id": "s-new", "provider": "codex", "last_timestamp": "2026-10-20T00:00:00Z"},
                {"session_id": "s-old", "provider": "claude_code_cli", "last_timestamp": "2026-10-10T00:00:00Z"}]

    monkeypatch.setattr(rag_engine, "session_turns", fake_turns)
    monkeypatch.setattr(rag_engine, "sessions_by_issue", fake_by_issue)

    out = asyncio.run(svc.context(written["memory_id"], project="sessionflow", limit=7))

    assert out["memory"]["memory_id"] == written["memory_id"]
    assert out["source_session"]["turns"] == [{"turn_index": 0, "content": "hello"}]
    assert calls["turns"] == ("sess-src", "claude_code_cli", 7)
    assert [s["session_id"] for s in out["related_sessions"]] == ["s-new", "s-old"]
    assert calls["by_issue"] == (["DEVS-145"], None, "sess-src")
    assert out["note"] is None


def test_context_without_a_source_uses_issues_and_the_day(svc, monkeypatch):
    import rag_engine

    written = _add(svc, "fact about DEVS-9", extra={"date": "2026-10-20"})
    seen = []

    def fake_by_issue(issue_ids, *, limit, project_roots=None, day=None, exclude_session=None):
        seen.append({"issue_ids": issue_ids, "project_roots": project_roots, "day": day})
        if day is None:
            return [{"session_id": "by-issue", "provider": "codex", "last_timestamp": "2026-10-19T00:00:00Z"}]
        return [{"session_id": "same-day", "provider": "claude_code_cli",
                 "last_timestamp": "2026-10-20T15:00:00Z"}]

    monkeypatch.setattr(rag_engine, "sessions_by_issue", fake_by_issue)
    monkeypatch.setattr(memory_service, "resolve_project_roots",
                        lambda project: ["/Volumes/DATA/GitHub/SessionFlow"])

    out = asyncio.run(svc.context(written["memory_id"], project="sessionflow", limit=5))

    assert out["source_session"] is None
    assert "no source session" in out["note"].lower()
    assert {s["session_id"] for s in out["related_sessions"]} == {"by-issue", "same-day"}
    day_call = next(c for c in seen if c["day"] is not None)
    assert day_call["day"] == date(2026, 10, 20)
    assert day_call["project_roots"] == ["/Volumes/DATA/GitHub/SessionFlow"]
