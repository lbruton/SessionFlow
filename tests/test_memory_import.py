"""mem0 import: field mapping, identity by mem0 id, dedupe, skips and the report.

Input is the import JSON shape that Devops `mem0-export.py convert` writes: raw
mem0 values, with every mapping done here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import psycopg
import pytest

# A collection-time ImportError would stop the whole test session.
try:
    import memory_import
    from memory_service import MemoryInputError, MemoryService
    from memory_store import MemoryFilters, MemoryStore
except ImportError:
    memory_import = MemoryInputError = MemoryService = MemoryFilters = MemoryStore = None

AWS_KEY = "AKIA" + "1234567890ABCDEF"


class _Index:
    async def apply(self, memory_id):
        return "ok"

    async def similar(self, text, *, project, user_id, threshold):
        return [], "ok"


async def _embed(text):
    return [0.0] * 768


def _record(mem0_id, text, *, project="DevOps", type_="session-summary", agent="claude-code",
            user="lbruton", created="2026-10-01T09:00:00-07:00", updated=None, **meta):
    metadata = {"project": project, "type": type_, "source": "wrap", **meta}
    if project is None:
        metadata.pop("project")
    if type_ is None:
        metadata.pop("type")
    return {"mem0_id": mem0_id, "text": text, "created_at": created,
            "updated_at": updated or created, "user_id": user, "agent_id": agent,
            "metadata": metadata, "categories": ["infra"]}


def _export(*records):
    return {"source": "mem0", "cutoff": "2026-09-27T00:00:00-05:00",
            "converted_at": "2026-10-26T00:00:00Z", "csv_ids": len(records),
            "records": list(records)}


@pytest.fixture
def store(memory_pg):
    s = MemoryStore(memory_pg)
    s.migrate()
    yield s
    s.close()


@pytest.fixture
def svc(store):
    return MemoryService(store=store, index=_Index(), embed_query=_embed, allowlist=[],
                         similar_threshold=0.9, audit_age_days=30,
                         project_inventory={"devops", "staktrakr"})


def _run(svc, data, tmp_path, name="report.json"):
    report_path = tmp_path / name
    report = asyncio.run(memory_import.import_records(data, service=svc, report_path=report_path))
    assert json.loads(report_path.read_text()) == report
    return report


def _all(store, project="*", status=None):
    rows, _ = store.list(MemoryFilters(project=project, status=status), limit=500)
    return rows


def _outcome(report, mem0_id):
    return next(r for r in report["records"] if r["mem0_id"] == mem0_id)


# --- Mapping (AC-43, AC-44) --------------------------------------------------


def test_records_keep_timestamps_text_and_metadata(svc, store, tmp_path):
    rec = _record("m-1", "verbatim text, kept as is", issue_url="https://plane/x",
                  handoff={"next": "a"}, updated="2026-10-02T10:00:00-07:00")
    _run(svc, _export(rec), tmp_path)

    (row,) = _all(store)
    assert row["text"] == "verbatim text, kept as is"
    assert row["created_at"] == datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
    assert row["updated_at"] == datetime(2026, 10, 2, 17, 0, tzinfo=timezone.utc)
    assert row["project"] == "devops"
    assert row["type"] == "session-summary"
    assert row["extra"]["source"] == "wrap"
    assert row["extra"]["issue_url"] == "https://plane/x"
    assert row["extra"]["handoff"] == {"next": "a"}
    assert row["extra"]["mem0_ids"] == ["m-1"]
    assert "project" not in row["extra"] and "type" not in row["extra"]


def test_retro_learning_maps_to_discovery(svc, store, tmp_path):
    _run(svc, _export(_record("m-r", "a lesson", project="staktrakr", type_="retro-learning")), tmp_path)
    assert _all(store)[0]["type"] == "discovery"


# --- Agents (AC-45, AC-46) ---------------------------------------------------


@pytest.mark.parametrize("raw,stored", [
    ("cowork", "claude-code"), ("codex", "codex"), ("claude-code", "claude-code"),
    (None, "unknown"), ("", "unknown"), ("False", "unknown"),
])
def test_agent_mapping(svc, store, tmp_path, raw, stored):
    _run(svc, _export(_record("m-a", f"agent {raw!r}", agent=raw)), tmp_path)
    assert _all(store)[0]["agent_id"] == stored


def test_unknown_agent_stays_import_only_and_survives_updates(svc, store, tmp_path):
    _run(svc, _export(_record("m-u", "no writer", agent=None)), tmp_path)
    (row,) = _all(store)

    with pytest.raises(MemoryInputError):
        asyncio.run(svc.add({"text": "new", "agent_id": "unknown", "project": "devops",
                             "type": "fact"}))

    filtered, _ = store.list(MemoryFilters(project="devops", agent_id="unknown"), limit=10)
    assert [r["memory_id"] for r in filtered] == [row["memory_id"]]

    asyncio.run(svc.update(row["memory_id"], {"text": "edited, still no writer", "project": "devops"}))
    assert store.get(row["memory_id"])["agent_id"] == "unknown"


def test_missing_user_defaults_to_lbruton(svc, store, tmp_path):
    _run(svc, _export(_record("m-n", "no user", user=None)), tmp_path)
    assert _all(store)[0]["user_id"] == "lbruton"


# --- Dedupe and skips (AC-47, AC-48) -----------------------------------------


def test_duplicate_text_adds_its_mem0_id_and_creates_nothing(svc, store, tmp_path):
    report = _run(svc, _export(_record("m-a", "same words"), _record("m-b", "same words")), tmp_path)
    (row,) = _all(store)
    assert sorted(row["extra"]["mem0_ids"]) == ["m-a", "m-b"]
    assert _outcome(report, "m-a")["outcome"] == "imported"
    assert _outcome(report, "m-b")["outcome"] == "deduplicated"
    assert _outcome(report, "m-b")["memory_id"] == row["memory_id"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"project": None}, "project"),
    ({"created": None}, "created"),
    ({"type_": None}, "type"),
    ({"type_": "musing"}, "type"),
])
def test_unplaceable_records_are_skipped_with_a_reason(svc, store, tmp_path, kwargs, reason):
    rec = _record("m-s", "cannot place", **kwargs)
    report = _run(svc, _export(rec), tmp_path)
    assert _all(store) == []
    entry = _outcome(report, "m-s")
    assert entry["outcome"] == "skipped" and reason in entry["reason"]


def test_validation_failures_are_skipped_not_fatal(svc, store, tmp_path):
    report = _run(svc, _export(_record("m-secret", f"leaked {AWS_KEY}"),
                               _record("m-ok", "fine")), tmp_path)
    assert [r["text"] for r in _all(store)] == ["fine"]
    entry = _outcome(report, "m-secret")
    assert entry["outcome"] == "skipped" and "secret" in entry["reason"]
    assert AWS_KEY not in json.dumps(report)


# --- Re-runs (AC-49, D-25) ---------------------------------------------------


def _state(dsn):
    with psycopg.connect(dsn) as conn:
        rows = conn.execute("SELECT memory_id, version, change_seq, updated_at, extra "
                            "FROM memories ORDER BY memory_id").fetchall()
        versions = conn.execute("SELECT count(*) FROM memory_versions").fetchone()[0]
    return rows, versions


def test_second_run_on_the_same_export_changes_nothing(svc, memory_pg, tmp_path):
    data = _export(_record("m-1", "one"), _record("m-2", "two", project="staktrakr", type_="fact"),
                   _record("m-3", "one"))
    _run(svc, data, tmp_path, "first.json")
    before = _state(memory_pg)

    report = _run(svc, data, tmp_path, "second.json")

    assert _state(memory_pg) == before
    assert {r["outcome"] for r in report["records"]} == {"unchanged"}


def test_merged_duplicates_with_different_metadata_stay_put_on_reruns(svc, memory_pg, store, tmp_path):
    data = _export(_record("m-a", "same words", source="wrap"),
                   _record("m-b", "same words", source="manual"))
    _run(svc, data, tmp_path, "first.json")
    (row,) = _all(store)
    assert sorted(row["extra"]["mem0_ids"]) == ["m-a", "m-b"]
    before = _state(memory_pg)

    for name in ("second.json", "third.json"):
        report = _run(svc, data, tmp_path, name)
        assert _state(memory_pg) == before
        assert _outcome(report, "m-a")["outcome"] == "unchanged"
        assert _outcome(report, "m-b")["outcome"] == "unchanged"
    assert len(_all(store)) == 1


def test_changed_record_writes_one_new_version(svc, store, tmp_path):
    _run(svc, _export(_record("m-1", "first wording")), tmp_path, "first.json")
    (row,) = _all(store)

    report = _run(svc, _export(_record("m-1", "second wording", updated="2026-10-05T09:00:00-07:00")),
                  tmp_path, "second.json")

    after = store.get(row["memory_id"])
    assert after["text"] == "second wording" and after["version"] == 2
    assert after["created_at"] == row["created_at"]
    assert after["extra"]["mem0_ids"] == ["m-1"]
    assert _outcome(report, "m-1")["outcome"] == "updated"


# --- Report (AC-50) ----------------------------------------------------------


def test_report_counts_per_project_and_type(svc, tmp_path):
    data = _export(
        _record("m-1", "a"), _record("m-2", "a"),
        _record("m-3", "b", project="StakTrakr", type_="fact"),
        _record("m-4", "c", project=None),
    )
    report = _run(svc, data, tmp_path)

    devops = report["counts"]["devops"]["session-summary"]
    assert devops == {"read": 2, "imported": 1, "deduplicated": 1, "skipped": 0,
                      "updated": 0, "unchanged": 0}
    assert report["counts"]["staktrakr"]["fact"]["imported"] == 1
    assert report["counts"]["(none)"]["session-summary"]["skipped"] == 1
    assert sorted(r["mem0_id"] for r in report["records"]) == ["m-1", "m-2", "m-3", "m-4"]
    assert report["totals"]["read"] == 4


@pytest.mark.parametrize("field", ["project", "user"])
def test_oversized_scope_values_are_skipped_not_stored(svc, store, tmp_path, field):
    kwargs = {"project": "p" * 129} if field == "project" else {"user": "u" * 129}
    report = _run(svc, _export(_record("m-big", "too long a scope", **kwargs)), tmp_path)
    assert _all(store) == []
    entry = _outcome(report, "m-big")
    assert entry["outcome"] == "skipped" and "128" in entry["reason"]
