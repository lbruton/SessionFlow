"""Re-tag stored rows with the issue IDs the active patterns extract.

``cleanup.py retag-issues`` drives this. It re-runs extraction on each stored
``document`` and rewrites ``issue_ids`` through ``rag_engine.update_row_metadata``,
which keeps the stored vector, so nothing is embedded.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

import passages
import rag_engine

logger = logging.getLogger("sessionflow.issue_retag")

_SCAN_BATCH = 200


@dataclass
class RetagPlan:
    """Rows whose stored issue IDs differ from what extraction gives.

    Attributes:
        changes: ``{doc_id: new issue_ids}`` for every row to rewrite.
        scanned: Milvus rows read.
        fts_stale: Rows whose Milvus value is right but whose FTS row is wrong or missing.
        added: Per issue ID, the rows that gain it.
        removed: Per issue ID, the rows that lose it.
        fts_error: Why the FTS sidecar could not be read, or None.
        document_hashes: ``{doc_id: document_sha256}`` of each changed row as scanned.
    """

    changes: Dict[str, str] = field(default_factory=dict)
    document_hashes: Dict[str, str] = field(default_factory=dict)
    scanned: int = 0
    fts_stale: int = 0
    added: Counter = field(default_factory=Counter)
    removed: Counter = field(default_factory=Counter)
    fts_error: Optional[str] = None


@dataclass
class RetagResult:
    """Counts of an applied plan.

    ``changed`` counts rows whose document was rewritten after the scan and were left alone.
    """

    written: int = 0
    milvus_failed: int = 0
    fts_failed: int = 0
    changed: int = 0


def _ids(value: Optional[str]) -> set:
    return {token for token in (value or "").split(",") if token}


def _fts_issue_ids(db_path: Optional[str]) -> Dict[str, str]:
    """Read ``{doc_id: issue_ids}`` from the FTS sidecar; raises when it cannot be read."""
    if not db_path:
        return {}
    conn = rag_engine._fts.connection(db_path)
    try:
        return {doc_id: value or "" for doc_id, value in rag_engine._fts.select_where(
            conn, ["doc_id", "issue_ids"], "1 = 1", ())}
    finally:
        rag_engine._fts.close_ephemeral(conn)


def plan(*, db_path: Optional[str] = None) -> RetagPlan:
    """Scan every row and list the ones whose ``issue_ids`` would change. Writes nothing.

    A turn's tags come from its own ``document``. A passage takes its parent's
    tags, as at ingest; a passage without a parent row keeps its value. When
    the FTS sidecar can be read, a row whose FTS value differs or is missing is
    listed too.

    Args:
        db_path: Milvus DB path; also derives the FTS sidecar path.

    Returns:
        The plan, with per-ID counts of tags gained and lost.
    """
    result = RetagPlan()
    stored: Dict[str, str] = {}
    computed: Dict[str, str] = {}
    passage_rows: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    for batch in rag_engine._query_batches(
            ["doc_id", "document", "issue_ids", "chunk_type"],
            batch_size=_SCAN_BATCH, db_path=db_path):
        for row in batch:
            doc_id = row["doc_id"]
            result.scanned += 1
            stored[doc_id] = row.get("issue_ids") or ""
            hashes[doc_id] = rag_engine.document_sha256(row.get("document"))
            if row.get("chunk_type") == passages.PASSAGE_CHUNK_TYPE:
                passage_rows[doc_id] = passages.passage_parent(doc_id) or ""
            else:
                computed[doc_id] = rag_engine._extract_issue_ids(row.get("document") or "")
    for doc_id, parent in passage_rows.items():
        computed[doc_id] = computed[parent] if parent in computed else stored[doc_id]

    fts: Optional[Dict[str, str]] = None
    if db_path:
        try:
            fts = _fts_issue_ids(db_path)
        except Exception as exc:
            result.fts_error = rag_engine._scrub_exception(exc)
            logger.warning("FTS issue_ids read failed: %s", result.fts_error)

    for doc_id, new in computed.items():
        old = stored[doc_id]
        if old != new:
            result.changes[doc_id] = new
            for token in _ids(new) - _ids(old):
                result.added[token] += 1
            for token in _ids(old) - _ids(new):
                result.removed[token] += 1
        # A failed FTS insert leaves no row behind its delete, so a missing row is stale too.
        elif fts is not None and fts.get(doc_id) != new:
            result.changes[doc_id] = new
            result.fts_stale += 1
    result.document_hashes = {doc_id: hashes[doc_id] for doc_id in result.changes}
    return result


def apply(retag: RetagPlan, *, db_path: Optional[str] = None,
          progress_every: int = 1000) -> RetagResult:
    """Rewrite ``issue_ids`` of each planned row in Milvus and FTS, keeping its vector.

    Args:
        retag: The plan from :func:`plan`.
        db_path: Milvus DB path; also derives the FTS sidecar path.
        progress_every: Log a progress line after this many rows.

    Returns:
        Rows written, rows whose Milvus or FTS write failed, and rows skipped
        because their document changed after the scan. A re-run of :func:`plan`
        lists a row again while either store is stale.
    """
    outcome = RetagResult()
    for count, (doc_id, new) in enumerate(retag.changes.items(), start=1):
        try:
            written = rag_engine.update_row_metadata(
                doc_id, db_path=db_path, issue_ids=new,
                expect_document_sha256=retag.document_hashes.get(doc_id))
        except rag_engine.RowChanged:
            outcome.changed += 1
            continue
        if not written.milvus_ok:
            outcome.milvus_failed += 1
        elif not written.fts_ok:
            outcome.fts_failed += 1
        else:
            outcome.written += 1
        if progress_every and count % progress_every == 0:
            logger.info("retag-issues: %d of %d rows", count, len(retag.changes))
    return outcome
