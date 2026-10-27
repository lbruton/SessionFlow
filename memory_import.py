"""Import a converted mem0 export into the memory store.

Input is the JSON that ``mem0-export.py convert`` writes, with raw mem0 values::

    {"source": "mem0", "cutoff": "...", "converted_at": "...", "csv_ids": 470,
     "records": [{"mem0_id": "<uuid>", "text": "...", "created_at": "<ISO with offset>",
                  "updated_at": "<ISO with offset>", "user_id": "lbruton" | null,
                  "agent_id": "cowork" | "claude-code" | "codex" | null | "False",
                  "metadata": {"project": "...", "type": "...", ...}, "categories": [...]}]}

Every mapping happens here. A record is found again by its mem0 ID, so a
second run of the same export writes nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_service import (
    IMPORT_ONLY_AGENT,
    LEGAL_MEMORY_AGENTS,
    LEGAL_MEMORY_TYPES,
    MAX_PROJECT_BYTES,
    MemoryService,
    _issue_ids,
)
from memory_store import MemoryStoreError, NewMemory, content_hash, run_db

OUTCOMES = ("imported", "deduplicated", "updated", "unchanged", "skipped")
TYPE_MAP = {"retro-learning": "discovery"}
AGENT_MAP = {"cowork": "claude-code"}
# mem0 values that mean "no writer recorded".
NO_AGENT = {None, "", "False", "false", "None", "null"}
NO_PROJECT = "(none)"


class _Skip(Exception):
    """A record that cannot be placed; the message is the report reason."""


def _parse_time(value: Any, field: str) -> datetime:
    if not value:
        raise _Skip(f"no {field} timestamp")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise _Skip(f"unparseable {field} timestamp") from exc
    if parsed.tzinfo is None:
        raise _Skip(f"{field} timestamp has no offset")
    return parsed


def _map(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map one raw export record to store fields, or raise ``_Skip``."""
    metadata = dict(record.get("metadata") or {})
    project = str(metadata.pop("project", "") or "").strip().lower()
    raw_type = metadata.pop("type", None)
    type_ = TYPE_MAP.get(raw_type, raw_type)
    if not project:
        raise _Skip("no project in the mem0 metadata")
    if not type_:
        raise _Skip("no type in the mem0 metadata")
    if type_ not in LEGAL_MEMORY_TYPES:
        raise _Skip(f"type {raw_type!r} has no mapping")
    created = _parse_time(record.get("created_at"), "created")
    updated = _parse_time(record.get("updated_at") or record.get("created_at"), "updated")
    raw_agent = record.get("agent_id")
    if raw_agent in NO_AGENT:
        agent = IMPORT_ONLY_AGENT
    else:
        agent = AGENT_MAP.get(raw_agent, raw_agent)
        if agent not in LEGAL_MEMORY_AGENTS:
            raise _Skip(f"agent {raw_agent!r} has no mapping")
    extra = metadata
    if record.get("categories"):
        extra["mem0_categories"] = list(record["categories"])
    # The add path bypasses MemoryService.add, so apply its index-key limits here.
    MemoryService._check_index_key("project", project, MAX_PROJECT_BYTES)
    user_id = MemoryService._write_user_id(record.get("user_id"))
    return {"project": project, "type": type_, "agent_id": agent, "created_at": created,
            "updated_at": updated, "user_id": user_id, "extra": extra}


def _with_id(extra: Dict[str, Any], ids: List[str]) -> Dict[str, Any]:
    return {**extra, "mem0_ids": sorted(set(ids))}


async def _import_one(record: Dict[str, Any], service: MemoryService) -> Dict[str, Any]:
    mem0_id = str(record.get("mem0_id") or "")
    entry: Dict[str, Any] = {"mem0_id": mem0_id, "project": NO_PROJECT,
                             "type": (record.get("metadata") or {}).get("type") or NO_PROJECT}
    try:
        fields = _map(record)
        entry.update(project=fields["project"], type=fields["type"])
        text = service._check_text(record.get("text"))
        store = service.store
        existing = await run_db(store.find_by_mem0_id, mem0_id)
        if existing is not None:
            if existing.get("deleted_at") is not None:
                raise _Skip("deleted in SessionFlow after an earlier import")
            ids = list((existing.get("extra") or {}).get("mem0_ids") or [])
            if len(set(ids)) > 1:
                # Merged duplicates share one memory; comparing against either
                # record's metadata would rewrite it on every run.
                entry.update(outcome="unchanged", memory_id=existing["memory_id"])
                return entry
            changes = {
                "text": text, "type": fields["type"], "agent_id": fields["agent_id"],
                "extra": _with_id(fields["extra"], ids + [mem0_id]),
            }
            current = {"text": existing["text"], "type": existing["type"],
                       "agent_id": existing["agent_id"], "extra": existing["extra"]}
            if json.dumps(changes, sort_keys=True) == json.dumps(current, sort_keys=True):
                entry.update(outcome="unchanged", memory_id=existing["memory_id"])
                return entry
            changes.update(content_hash=content_hash(text), issue_ids=_issue_ids(text),
                           updated_at=fields["updated_at"])
            result = await run_db(store.update, existing["memory_id"], changes)
            await service._apply(result.memory_id)
            entry.update(outcome="updated", memory_id=result.memory_id)
            return entry

        result = await run_db(store.add, NewMemory(
            text=text, agent_id=fields["agent_id"], project=fields["project"],
            type=fields["type"], content_hash=content_hash(text), user_id=fields["user_id"],
            extra=_with_id(fields["extra"], [mem0_id]), issue_ids=_issue_ids(text),
            created_at=fields["created_at"], updated_at=fields["updated_at"],
        ))
        if result.outcome == "created":
            entry.update(outcome="imported", memory_id=result.memory_id)
        else:
            row = await run_db(store.get, result.memory_id)
            ids = list((row.get("extra") or {}).get("mem0_ids") or [])
            await run_db(store.update, result.memory_id, {
                "extra": _with_id(row.get("extra") or {}, ids + [mem0_id]),
                "updated_at": row["updated_at"],
            })
            entry.update(outcome="deduplicated", memory_id=result.memory_id)
        await service._apply(result.memory_id)
        return entry
    except _Skip as exc:
        entry.update(outcome="skipped", reason=str(exc))
    except MemoryStoreError as exc:
        entry.update(outcome="skipped", reason=f"{exc.code}: {exc.message}")
    return entry


def _report(data: Dict[str, Any], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"read": 0, **{o: 0 for o in OUTCOMES}}))
    totals = {"read": 0, **{o: 0 for o in OUTCOMES}}
    for e in entries:
        bucket = counts[e["project"]][e["type"]]
        bucket["read"] += 1
        bucket[e["outcome"]] += 1
        totals["read"] += 1
        totals[e["outcome"]] += 1
    return {
        "source": data.get("source"),
        "cutoff": data.get("cutoff"),
        "csv_ids": data.get("csv_ids"),
        "totals": totals,
        "counts": {p: dict(t) for p, t in counts.items()},
        "records": entries,
    }


async def import_records(data: Dict[str, Any], *, service: MemoryService,
                         report_path: Optional[Path]) -> Dict[str, Any]:
    """Import every record and write the per-record report to ``report_path``.

    Returns the report: totals and per-project, per-type counts by outcome,
    and every mem0 ID with its outcome, memory ID or skip reason.
    """
    entries = [await _import_one(r, service) for r in data.get("records") or []]
    report = _report(data, entries)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

