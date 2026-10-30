"""Opt-in usage log of search and follow-up tool calls, for the retrieval eval.

``SESSIONFLOW_SEARCH_LOG=1`` turns it on. Each call appends one JSON line to
``~/.sessionflow/search-log.jsonl``. Queries pass through the secret redactor in
enforce mode, and a write failure is logged and never fails the tool call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import secret_redaction

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_lock = threading.Lock()


def enabled() -> bool:
    """Return whether ``SESSIONFLOW_SEARCH_LOG`` turns the log on."""
    return os.getenv("SESSIONFLOW_SEARCH_LOG", "").strip().lower() in _TRUE


def log_path() -> Path:
    """Return the log file path under the current home folder."""
    return Path.home() / ".sessionflow" / "search-log.jsonl"


def _append(entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    path = log_path()
    try:
        path.parent.mkdir(mode=0o700, exist_ok=True)
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Search log write failed: %s", exc)


def record_search(tool: str, *, client: Optional[str], project, sort_by: Optional[str],
                  query: str, ids: Iterable[str]) -> None:
    """Append one search call with its result IDs in rank order.

    The query is redacted with the built-in rules only; the operator allowlist
    is not applied, so an allowlisted path may be masked in the log.
    """
    if not enabled():
        return
    redacted, _ = secret_redaction.redact(query or "", mode="enforce")
    _append({"kind": "search", "tool": tool, "client": client, "project": project,
             "sort_by": sort_by, "query": redacted, "ids": list(ids)})


def record_follow_up(tool: str, *, client: Optional[str], item_id: str) -> None:
    """Append one ``get_memory`` or ``get_turns`` call with the ID it opened."""
    if not enabled():
        return
    _append({"kind": "follow_up", "tool": tool, "client": client, "id": item_id})
