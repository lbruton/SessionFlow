"""Canonical memory store on Postgres: transactions, uniqueness, history, acknowledgements.

Runs against SESSIONFLOW_TEST_PG_DSN through the memory_pg fixture (one fresh
schema per test). Skips with a reason when no test database is configured.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest

# A collection-time ImportError would stop the whole test session, so a missing
# module fails each test here instead.
try:
    import memory_store
    from memory_store import (
        InvalidSupersede,
        MemoryConflict,
        MemoryFilters,
        MemoryNotFound,
        MemoryStore,
        MemoryStoreError,
        NewMemory,
    )
except ImportError:
    memory_store = InvalidSupersede = MemoryConflict = MemoryFilters = None
    MemoryNotFound = MemoryStore = MemoryStoreError = NewMemory = None

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rec(text: str, **over) -> NewMemory:
    fields = dict(
        text=text,
        agent_id="claude-code",
        project="sessionflow",
        type="fact",
        content_hash=_hash(text),
    )
    fields.update(over)
    return NewMemory(**fields)


@pytest.fixture
def store(memory_pg):
    s = MemoryStore(memory_pg)
    s.migrate()
    yield s
    s.close()


def _count(dsn: str, sql: str, *params) -> int:
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql, params).fetchone()[0]


# --- Migrations -------------------------------------------------------------


def test_migrate_applies_once_and_records_it(memory_pg):
    s = MemoryStore(memory_pg)
    try:
        assert s.migrate() == ["001_initial"]
        assert s.migrate() == []
    finally:
        s.close()
    assert _count(memory_pg, "SELECT count(*) FROM schema_migrations WHERE version = '001_initial'") == 1
    assert _count(memory_pg, "SELECT count(*) FROM memory_index_meta") == 1


# --- Fields round trip (AC-6, AC-7, AC-39) -----------------------------------


def test_every_field_round_trips(store):
    started = T0 - timedelta(hours=1)
    extra = {"source": "wrap", "date": "2026-10-01", "handoff": {"next": ["a", "b"], "n": 3},
             "issue_url": "https://plane.example/x", "topics": ["x", "y"]}
    written = store.add(_rec(
        "round trip", topic_key="session:abc", extra=extra, issue_ids=["DEVS-145"],
        source_session_id="abc", source_provider="claude_code_cli",
        source_project_root="/Volumes/DATA/GitHub/SessionFlow",
        source_started_at=started, source_ended_at=T0,
    ))
    row = store.get(written.memory_id)

    assert row["memory_id"] == written.memory_id
    assert row["text"] == "round trip"
    assert row["user_id"] == "lbruton"
    assert row["agent_id"] == "claude-code"
    assert row["project"] == "sessionflow"
    assert row["type"] == "fact"
    assert row["status"] == "active"
    assert row["topic_key"] == "session:abc"
    assert row["content_hash"] == _hash("round trip")
    assert row["issue_ids"] == ["DEVS-145"]
    assert row["supersedes"] == []
    assert row["extra"] == extra
    assert row["source_session_id"] == "abc"
    assert row["source_provider"] == "claude_code_cli"
    assert row["source_project_root"] == "/Volumes/DATA/GitHub/SessionFlow"
    assert row["source_started_at"] == started
    assert row["source_ended_at"] == T0
    assert row["version"] == 1
    assert row["created_at"].tzinfo is not None and row["updated_at"] == row["created_at"]


def test_provenance_is_empty_without_a_session(store):
    row = store.get(store.add(_rec("no session")).memory_id)
    for key in ("source_session_id", "source_provider", "source_project_root",
                "source_started_at", "source_ended_at"):
        assert row[key] is None


# --- Uniqueness and resolution (AC-12 to AC-15) ------------------------------


def test_uniqueness_is_scoped_to_user_and_project(store):
    a = store.add(_rec("same words", project="sessionflow"))
    b = store.add(_rec("same words", project="devops"))
    c = store.add(_rec("same words", project="sessionflow", user_id="someone-else"))
    assert len({a.memory_id, b.memory_id, c.memory_id}) == 3
    assert {a.outcome, b.outcome, c.outcome} == {"created"}


def test_empty_topic_key_is_no_key(store):
    a = store.add(_rec("first", topic_key=""))
    b = store.add(_rec("second", topic_key=""))
    assert a.memory_id != b.memory_id
    assert store.get(a.memory_id)["topic_key"] is None


def test_topic_match_replaces_in_place(store):
    first = store.add(_rec("old text", topic_key="session:1", extra={"source": "wrap"}))
    second = store.add(_rec("new text", topic_key="session:1", extra={"source": "wrap"}))

    assert second.outcome == "replaced"
    assert second.memory_id == first.memory_id
    row = store.get(first.memory_id)
    assert row["text"] == "new text"
    assert row["created_at"] == first.created_at
    assert row["updated_at"] > first.updated_at
    assert row["version"] == 2


def test_hash_match_returns_the_existing_record(memory_pg, store):
    first = store.add(_rec("duplicate"))
    second = store.add(_rec("duplicate"))
    assert second.outcome == "deduplicated"
    assert second.memory_id == first.memory_id
    assert _count(memory_pg, "SELECT count(*) FROM memories") == 1


def test_topic_and_hash_on_different_records_conflict(memory_pg, store):
    a = store.add(_rec("record a", topic_key="topic-a"))
    b = store.add(_rec("record b"))
    before = _count(memory_pg, "SELECT count(*) FROM memory_versions")

    with pytest.raises(MemoryConflict) as exc:
        store.add(_rec("record b", topic_key="topic-a"))
    assert set(exc.value.memory_ids) == {a.memory_id, b.memory_id}
    assert exc.value.code == "conflict"

    with pytest.raises(MemoryConflict) as exc:
        store.update(a.memory_id, {"text": "record b", "content_hash": _hash("record b")})
    assert set(exc.value.memory_ids) == {a.memory_id, b.memory_id}

    assert _count(memory_pg, "SELECT count(*) FROM memory_versions") == before
    assert store.get(a.memory_id)["text"] == "record a"


# --- Concurrency (AC-16) -----------------------------------------------------


def _race(memory_pg, records):
    """Run one add per record on its own store and connection, released together."""
    barrier = threading.Barrier(len(records))
    results, errors = [None] * len(records), []

    def worker(i, rec):
        s = MemoryStore(memory_pg)
        try:
            barrier.wait()
            results[i] = s.add(rec)
        except Exception as exc:  # noqa: BLE001 - surfaced through the assertion below
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=worker, args=(i, r)) for i, r in enumerate(records)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    return results


def test_concurrent_same_hash_adds_make_one_record(memory_pg, store):
    results = _race(memory_pg, [_rec("raced text") for _ in range(4)])
    assert len({r.memory_id for r in results}) == 1
    assert sorted(r.outcome for r in results) == ["created", "deduplicated", "deduplicated", "deduplicated"]
    assert _count(memory_pg, "SELECT count(*) FROM memories") == 1


def test_concurrent_same_topic_adds_make_one_record(memory_pg, store):
    results = _race(memory_pg, [_rec(f"topic text {i}", topic_key="session:race") for i in range(4)])
    assert len({r.memory_id for r in results}) == 1
    assert sorted(r.outcome for r in results) == ["created", "replaced", "replaced", "replaced"]
    assert _count(memory_pg, "SELECT count(*) FROM memories") == 1
    assert _count(memory_pg, "SELECT count(*) FROM memory_versions") == 4


def test_interleaved_update_and_delete_apply_in_one_order(memory_pg, store):
    target = store.add(_rec("contended"))
    barrier = threading.Barrier(3)

    def updater(i):
        s = MemoryStore(memory_pg)
        try:
            barrier.wait()
            try:
                s.update(target.memory_id, {"text": f"edit {i}", "content_hash": _hash(f"edit {i}")})
            except MemoryNotFound:
                pass
        finally:
            s.close()

    def deleter():
        s = MemoryStore(memory_pg)
        try:
            barrier.wait()
            s.delete(target.memory_id)
        finally:
            s.close()

    threads = [threading.Thread(target=updater, args=(i,)) for i in range(2)]
    threads.append(threading.Thread(target=deleter))
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    history = store.versions(target.memory_id)
    assert [h["version"] for h in history] == list(range(1, len(history) + 1))
    assert history[-1]["deleted_at"] is not None
    assert sum(1 for h in history if h["deleted_at"] is not None) == 1
    assert store.get(target.memory_id) is None


# --- Visibility, listing and cursor (AC-17, AC-19) ---------------------------


def test_committed_write_is_visible_at_once(memory_pg, store):
    written = store.add(_rec("visible"))
    other = MemoryStore(memory_pg)
    try:
        assert other.get(written.memory_id)["text"] == "visible"
        rows, _ = other.list(MemoryFilters(project="sessionflow"), limit=10)
        assert [r["memory_id"] for r in rows] == [written.memory_id]
    finally:
        other.close()


def test_cursor_pages_through_every_match_newest_first(store):
    ids = []
    for i in range(5):
        ids.append(store.add(_rec(f"tied {i}", created_at=T0, updated_at=T0)).memory_id)
    for i in range(3):
        ids.append(store.add(_rec(f"older {i}", created_at=T0 - timedelta(days=i + 1),
                                  updated_at=T0 - timedelta(days=i + 1))).memory_id)
    store.add(_rec("other project", project="devops"))

    seen, cursor = [], None
    while True:
        page, cursor = store.list(MemoryFilters(project="sessionflow"), limit=2, cursor=cursor)
        seen.extend(r["memory_id"] for r in page)
        if cursor is None:
            break

    tied = sorted(ids[:5], reverse=True)
    assert seen == tied + ids[5:]
    assert len(seen) == len(set(seen)) == 8


def test_cursor_is_bound_to_its_filters(store):
    for i in range(3):
        store.add(_rec(f"page {i}"))
    _, cursor = store.list(MemoryFilters(project="sessionflow"), limit=1)
    with pytest.raises(MemoryStoreError) as exc:
        store.list(MemoryFilters(project="sessionflow", type="decision"), limit=1, cursor=cursor)
    assert exc.value.code == "invalid_cursor"


def test_list_filters(store):
    store.add(_rec("a decision", type="decision", agent_id="codex", issue_ids=["DEVS-1"],
                   extra={"source": "wrap", "category": "infra"}))
    store.add(_rec("a fact", issue_ids=["DEVS-2"], extra={"source": "discover"}))
    store.add(_rec("old one", created_at=datetime(2026, 9, 30, 4, 59, tzinfo=timezone.utc),
                   updated_at=datetime(2026, 9, 30, 4, 59, tzinfo=timezone.utc)))
    store.add(_rec("first chicago minute", created_at=datetime(2026, 9, 30, 5, 0, tzinfo=timezone.utc),
                   updated_at=datetime(2026, 9, 30, 5, 0, tzinfo=timezone.utc)))

    def texts(**f):
        rows, _ = store.list(MemoryFilters(project="sessionflow", **f), limit=50)
        return {r["text"] for r in rows}

    assert texts(type="decision") == {"a decision"}
    assert texts(agent_id="codex") == {"a decision"}
    assert texts(issue_id="devs-2") == {"a fact"}
    assert texts(source="discover") == {"a fact"}
    assert texts(category="infra") == {"a decision"}
    # 2026-09-30 in America/Chicago starts at 05:00 UTC (CDT).
    assert texts(created_from=date(2026, 9, 30), created_to=date(2026, 9, 30)) == {"first chicago minute"}
    assert texts(created_to=date(2026, 9, 29)) == {"old one"}


def test_list_all_projects_with_star(store):
    store.add(_rec("here", project="sessionflow"))
    store.add(_rec("there", project="devops"))
    rows, _ = store.list(MemoryFilters(project="*"), limit=10)
    assert {r["project"] for r in rows} == {"sessionflow", "devops"}


# --- Update and delete (AC-20, AC-21, AC-28) ---------------------------------


def test_update_replaces_text_and_keeps_omitted_fields(store):
    first = store.add(_rec("before", topic_key="t", extra={"source": "wrap"}, issue_ids=["DEVS-1"]))
    result = store.update(first.memory_id, {"text": "after", "content_hash": _hash("after"),
                                            "issue_ids": ["DEVS-9"]})
    row = store.get(first.memory_id)
    assert result.outcome == "updated" and result.version == 2
    assert row["text"] == "after"
    assert row["issue_ids"] == ["DEVS-9"]
    assert row["topic_key"] == "t" and row["extra"] == {"source": "wrap"}
    assert row["agent_id"] == "claude-code"
    assert row["created_at"] == first.created_at and row["updated_at"] > first.updated_at


def test_delete_leaves_a_tombstone_and_a_final_version(memory_pg, store):
    first = store.add(_rec("to delete"))
    result = store.delete(first.memory_id)

    assert result.outcome == "deleted" and result.version == 2
    assert store.get(first.memory_id) is None
    assert _count(memory_pg, "SELECT count(*) FROM memories WHERE deleted_at IS NOT NULL") == 1
    assert store.versions(first.memory_id)[-1]["deleted_at"] is not None
    again = store.add(_rec("to delete"))
    assert again.outcome == "created" and again.memory_id != first.memory_id


def test_unknown_ids_raise_not_found(store):
    missing = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(MemoryNotFound):
        store.delete(missing)
    with pytest.raises(MemoryNotFound):
        store.update(missing, {"text": "x", "content_hash": _hash("x")})
    assert store.get(missing) is None


def test_id_operations_reject_a_project_mismatch(store):
    written = store.add(_rec("scoped"))
    with pytest.raises(MemoryStoreError) as exc:
        store.get(written.memory_id, expect_project="devops")
    assert exc.value.code == "project_mismatch"


# --- Commit boundary and crashes (AC-25, AC-26, AC-31) -----------------------


def test_failed_commit_changes_nothing(memory_pg, store):
    written = store.add(_rec("stable"))
    store.ack("milvus", written.memory_id, version=1, generation=1, deleted=False)

    def boom(stage, conn):
        if stage == "before_commit":
            raise RuntimeError("injected")

    store._fault_hook = boom
    with pytest.raises(Exception):
        store.update(written.memory_id, {"text": "never", "content_hash": _hash("never")})
    with pytest.raises(Exception):
        store.add(_rec("never added"))
    store._fault_hook = None

    assert store.get(written.memory_id)["text"] == "stable"
    assert store.pending("milvus") == []
    assert _count(memory_pg, "SELECT count(*) FROM memories") == 1
    assert _count(memory_pg, "SELECT count(*) FROM memory_versions") == 1


def test_backend_killed_mid_write_leaves_the_before_state(memory_pg, store):
    written = store.add(_rec("before the crash"))

    def kill(stage, conn):
        if stage == "before_commit":
            pid = conn.info.backend_pid
            with psycopg.connect(memory_pg, autocommit=True) as admin:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))

    store._fault_hook = kill
    with pytest.raises(Exception):
        store.update(written.memory_id, {"text": "after the crash", "content_hash": _hash("after the crash")})
    store._fault_hook = None

    row = store.get(written.memory_id)
    assert row["text"] == "before the crash" and row["version"] == 1
    assert len(store.versions(written.memory_id)) == 1


# --- Supersession (AC-37) ----------------------------------------------------


def test_supersede_marks_targets_and_keeps_history(store):
    old_a = store.add(_rec("old a"))
    old_b = store.add(_rec("old b"))
    new = store.add(_rec("new", supersedes=[old_a.memory_id, old_b.memory_id]))

    for old in (old_a, old_b):
        row = store.get(old.memory_id)
        assert row["status"] == "superseded" and row["version"] == 2
        assert [h["status"] for h in store.versions(old.memory_id)] == ["active", "superseded"]
    assert store.get(new.memory_id)["supersedes"] == sorted([old_a.memory_id, old_b.memory_id])

    active, _ = store.list(MemoryFilters(project="sessionflow"), limit=10)
    assert [r["memory_id"] for r in active] == [new.memory_id]
    superseded, _ = store.list(MemoryFilters(project="sessionflow", status="superseded"), limit=10)
    assert {r["memory_id"] for r in superseded} == {old_a.memory_id, old_b.memory_id}


def test_supersede_rejects_another_project(memory_pg, store):
    other = store.add(_rec("elsewhere", project="devops"))
    with pytest.raises(InvalidSupersede):
        store.add(_rec("here", supersedes=[other.memory_id]))
    assert store.get(other.memory_id)["status"] == "active"
    assert _count(memory_pg, "SELECT count(*) FROM memories WHERE project = 'sessionflow'") == 0


# --- Acknowledgements (D-24) -------------------------------------------------


def test_write_makes_every_consumer_pending_again(store):
    written = store.add(_rec("acked"))
    assert {r["memory_id"] for r in store.pending("milvus")} == {written.memory_id}
    store.ack("milvus", written.memory_id, version=1, generation=1, deleted=False)
    store.ack("fts:minitron", written.memory_id, version=1, generation=1, deleted=False)
    assert store.pending("milvus") == [] and store.pending("fts:minitron") == []

    store.update(written.memory_id, {"text": "changed", "content_hash": _hash("changed")})

    assert [r["version"] for r in store.pending("milvus")] == [2]
    assert [r["version"] for r in store.pending("fts:minitron")] == [2]


def test_ack_never_moves_backwards(store):
    written = store.add(_rec("v1"))
    store.update(written.memory_id, {"text": "v2", "content_hash": _hash("v2")})
    assert store.ack("milvus", written.memory_id, version=2, generation=1, deleted=False) is True
    assert store.ack("milvus", written.memory_id, version=1, generation=1, deleted=False) is False
    assert store.pending("milvus") == []


# --- Lookups (AC-55, D-25) ---------------------------------------------------


def test_source_session_lookup(memory_pg, store):
    store.add(_rec("wrap one", type="session-summary", source_session_id="sess-1"))
    store.add(_rec("wrap two", type="session-summary", source_session_id="sess-2"))
    rows, _ = store.list(MemoryFilters(project="sessionflow", type="session-summary",
                                       source_session_id="sess-1"), limit=10)
    assert [r["text"] for r in rows] == ["wrap one"]
    assert _count(memory_pg, "SELECT count(*) FROM pg_indexes WHERE indexname = 'memories_source_session_idx'") == 1


def test_find_by_mem0_id(store):
    written = store.add(_rec("imported", extra={"mem0_ids": ["m-1", "m-2"]}))
    assert store.find_by_mem0_id("m-2")["memory_id"] == written.memory_id
    assert store.find_by_mem0_id("m-3") is None


# --- Nightly dump status (AC-32, D-16) ---------------------------------------


def _run(started: datetime, ok: bool | None, phase: str = "result") -> dict:
    return {"job": "pg-dump", "phase": phase, "started_at": started, "ok": ok}


def test_dump_status_ok_when_the_latest_due_run_succeeded():
    # 03:00 CDT: today's 02:30 run is only 30 minutes old, so yesterday's slot is due.
    now = datetime(2026, 10, 27, 8, 0, tzinfo=timezone.utc)
    rows = [_run(datetime(2026, 10, 26, 7, 31, tzinfo=timezone.utc), True)]
    status = memory_store.evaluate_dump_status(rows, now)
    assert status["state"] == "ok"
    assert status["scheduled_for"] == datetime(2026, 10, 26, 7, 30, tzinfo=timezone.utc)


def test_dump_status_missing_when_the_due_run_never_ran():
    now = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)
    rows = [_run(datetime(2026, 10, 26, 7, 31, tzinfo=timezone.utc), True)]
    status = memory_store.evaluate_dump_status(rows, now)
    assert status["state"] == "missing"
    assert status["last_ok"] == datetime(2026, 10, 26, 7, 31, tzinfo=timezone.utc)


def test_dump_status_failed_when_the_due_run_failed():
    now = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)
    rows = [
        _run(datetime(2026, 10, 26, 7, 31, tzinfo=timezone.utc), True),
        _run(datetime(2026, 10, 27, 7, 31, tzinfo=timezone.utc), None, phase="attempt"),
        _run(datetime(2026, 10, 27, 7, 31, tzinfo=timezone.utc), False),
    ]
    assert memory_store.evaluate_dump_status(rows, now)["state"] == "failed"


def test_dump_status_follows_chicago_across_dst_end():
    # 2026-11-01: CDT ends, so the 02:30 run is 08:30 UTC, not 07:30.
    now = datetime(2026, 11, 1, 9, 20, tzinfo=timezone.utc)
    early = [_run(datetime(2026, 11, 1, 7, 31, tzinfo=timezone.utc), True)]
    on_time = [_run(datetime(2026, 11, 1, 8, 31, tzinfo=timezone.utc), True)]
    assert memory_store.evaluate_dump_status(early, now)["state"] == "missing"
    assert memory_store.evaluate_dump_status(on_time, now)["state"] == "ok"


def test_store_dump_status_reads_maintenance_runs(memory_pg, store):
    with psycopg.connect(memory_pg) as conn:
        conn.execute(
            "INSERT INTO maintenance_runs (job, phase, scheduled_for, started_at, finished_at, ok, detail) "
            "VALUES ('pg-dump', 'result', %s, %s, %s, true, 'memory-x.dump 10 bytes')",
            (datetime(2026, 10, 26, 7, 30, tzinfo=timezone.utc),
             datetime(2026, 10, 26, 7, 31, tzinfo=timezone.utc),
             datetime(2026, 10, 26, 7, 32, tzinfo=timezone.utc)),
        )
    status = store.dump_status(now=datetime(2026, 10, 27, 8, 0, tzinfo=timezone.utc))
    assert status["state"] == "ok"

# --- DSN configuration: the environment variable only ----------------------

DSN = "postgresql://user@db.example/memory"


def test_environment_value_is_used(monkeypatch):
    monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN", f"  {DSN}\n")
    assert memory_store.fetch_dsn() == DSN


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_is_not_configured(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SESSIONFLOW_MEMORY_DSN", raising=False)
    else:
        monkeypatch.setenv("SESSIONFLOW_MEMORY_DSN", value)
    with pytest.raises(MemoryStoreError) as exc:
        memory_store.fetch_dsn()
    assert exc.value.code == "dsn_unavailable"
    assert "not configured" in str(exc.value) and "SESSIONFLOW_MEMORY_DSN" in str(exc.value)


def test_no_file_is_read(monkeypatch, tmp_path):
    monkeypatch.delenv("SESSIONFLOW_MEMORY_DSN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".sessionflow").mkdir()
    (tmp_path / ".sessionflow" / "memory_dsn").write_text(DSN)
    with pytest.raises(MemoryStoreError):
        memory_store.fetch_dsn()
