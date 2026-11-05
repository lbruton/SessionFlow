"""One row per indexed session, with a title and whether a session summary covers it.

``list_sessions`` joins the Milvus session aggregates from ``rag_engine`` with the
wrapped flag from the memory store. Milvus reads run on worker threads; the
memory lookup goes through the memory service, which uses ``memory_store.run_db``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import rag_engine
from provider_adapters import is_subagent_session

WRAPPED_CHUNK = 50
"""Sessions checked against the memory store per query while filling a page."""


class WrappedUnavailable(Exception):
    """The ``wrapped`` filter was asked for, but the memory store cannot answer."""


async def _wrapped(memory: Any, ids: List[str]) -> Optional[set]:
    """Return the wrapped IDs, or None when the memory store cannot answer."""
    if memory is None:
        return None
    try:
        return set(await memory.wrapped_sessions(ids))
    except Exception:  # noqa: BLE001 - the listing stays useful without the flag
        return None


async def list_sessions(project_root: Optional[Sequence[str]], *, n: int = 20,
                        date_from: Optional[str] = None, date_to: Optional[str] = None,
                        provider: Optional[str] = None, wrapped: Optional[bool] = None,
                        include_subagents: bool = False, db_path: Optional[str] = None,
                        memory: Any = None) -> Dict[str, Any]:
    """Return the newest ``n`` sessions with their title and wrapped flag.

    A session is wrapped when a session-summary memory names it as its source.
    Without a memory answer, ``wrapped`` is None on every row and ``wrapped_known``
    is False; a ``wrapped`` filter then raises ``WrappedUnavailable``.

    Args:
        project_root: Indexed roots to cover, or None for every project.
        n: Maximum number of sessions.
        date_from: Earliest ISO date of a counted turn.
        date_to: Latest ISO date of a counted turn.
        provider: Only this provider.
        wrapped: True or False keeps only wrapped or only unwrapped sessions.
        include_subagents: Keep subagent transcripts (``is_subagent_session``).
        db_path: Milvus URI.
        memory: An object with an async ``wrapped_sessions(ids)``, or None.

    Returns:
        ``{"sessions": [...], "wrapped_known": bool}``.
    """
    rows = await asyncio.to_thread(
        rag_engine.list_sessions, list(project_root) if project_root else None,
        db_path, date_from=date_from, date_to=date_to, provider=provider)
    if not include_subagents:
        rows = [r for r in rows if not is_subagent_session(r["session_id"])]

    picked: List[Dict[str, Any]] = []
    known = memory is not None
    step = max(n, WRAPPED_CHUNK) if wrapped is not None else n
    for start in range(0, len(rows), max(step, 1)):
        if len(picked) >= n:
            break
        chunk = rows[start:start + step]
        found = await _wrapped(memory, [r["session_id"] for r in chunk]) if known else None
        if found is None:
            known = False
            if wrapped is not None:
                raise WrappedUnavailable(
                    "The wrapped filter needs the memory store, which has not started or "
                    "did not answer. See /health, or call list_sessions without 'wrapped'.")
        for row in chunk:
            row["wrapped"] = None if found is None else row["session_id"] in found
            if wrapped is None or row["wrapped"] is wrapped:
                picked.append(row)
        if wrapped is None:
            break
    picked = picked[:n]

    titles = await asyncio.to_thread(rag_engine.session_titles, picked, db_path)
    for row in picked:
        row["title"] = titles.get((row["session_id"], row.get("provider") or ""), "")
        row.pop("first_doc_ids", None)
    if not known:
        for row in picked:
            row["wrapped"] = None
    return {"sessions": picked, "wrapped_known": known}
