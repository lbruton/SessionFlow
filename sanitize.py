"""Retroactive secret sanitizer orchestration for SESF-42.

This module drives the dry-run/apply flow that retroactively removes secrets
already embedded into the SessionFlow stores. It holds *policy only*: detection
routes through :func:`secret_redaction.scan_spans`, and every store mutation goes
through the two SESF-42 :mod:`rag_engine` primitives
(:func:`rag_engine.upsert_document` / :func:`rag_engine.delete_by_doc_id`).

The flow has two entry points:

* :func:`scan` — a dry-run that iterates the in-scope turns, detects secrets, and
  reports per-rule counts plus the affected ``doc_id`` set without writing to any
  store. It still persists a value-free audit trail and a worklist checkpoint.
* :func:`apply` — the destructive pass. It refuses to read or write unless
  ``confirmed`` is set, then for each affected turn either re-embeds the redacted
  text and overwrites the row (``drop=False``) or deletes the row outright
  (``drop=True``). Progress is checkpointed after each batch so an interrupted run
  resumes from the durable worklist + done-set.

Hard invariants (design.md "Security Considerations", AC-18):

* No raw secret value ever reaches a log line, the audit file, stdout, or a return
  value — only rule names, tiers, integer offsets, and pre-masked snippets.
* Re-embedding always flows through the shared embedding budget, never below the
  200ms MLX cooldown floor.
* Milvus filters are built via :func:`rag_engine._escape_filter_scalar`, never raw
  f-string interpolation of operator/scope data.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Iterator, Optional

import rag_engine
import secret_redaction

# Output fields pulled from each Milvus scan row — everything needed to rebuild the
# row on upsert and to emit a value-free audit record. ``document`` is the only
# field that may carry a secret; it never leaves this module un-redacted.
_SCAN_FIELDS = [
    "doc_id",
    "document",
    "provider",
    "source_path",
    "turn_index",
    "timestamp",
    "session_id",
    "project_root",
]

# State-file layout under ``~/.sessionflow`` (mirrors the existing convention so a
# redirected HOME isolates them per test/run).
_STATE_DIRNAME = ".sessionflow"
_CHECKPOINT_NAME = "sanitize_state.json"
_AUDIT_DIRNAME = "audit"

# 0600 — owner read/write only. The audit + checkpoint may reference doc ids and
# masked snippets, so they are never world-readable.
_FILE_MODE = 0o600


@dataclass
class Scope:
    """A selection of turns to sanitize, expressed as Milvus filter dimensions.

    Each populated dimension contributes one escaped equality clause to the Milvus
    filter; ``since`` contributes a timestamp lower bound. An empty scope matches
    every turn in the collection.

    Attributes:
        project_root: Restrict to turns whose ``project_root`` equals this value.
        provider: Restrict to a single harness provider (e.g. ``claude_code_cli``).
        session_id: Restrict to a single session.
        since: ISO date/timestamp lower bound; maps to ``timestamp >= "<since>"``.
    """

    project_root: Optional[str] = None
    provider: Optional[str] = None
    session_id: Optional[str] = None
    since: Optional[str] = None

    def to_filter(self) -> str:
        """Build an escaped Milvus boolean filter for this scope.

        Returns:
            A conjunction of escaped clauses, or an empty string when no dimension
            is set (match-all). Every string scalar is escaped via
            :func:`rag_engine._escape_filter_scalar` so a quote or backslash in
            operator/scope data cannot break out of the filter literal.
        """
        clauses: list[str] = []
        if self.project_root:
            esc = rag_engine._escape_filter_scalar(self.project_root)
            clauses.append(f'project_root == "{esc}"')
        if self.provider:
            esc = rag_engine._escape_filter_scalar(self.provider)
            clauses.append(f'provider == "{esc}"')
        if self.session_id:
            esc = rag_engine._escape_filter_scalar(self.session_id)
            clauses.append(f'session_id == "{esc}"')
        if self.since:
            esc = rag_engine._escape_filter_scalar(self.since)
            clauses.append(f'timestamp >= "{esc}"')
        return " and ".join(clauses)

    def as_dict(self) -> dict:
        """Return the populated scope dimensions as a JSON-serializable dict."""
        out: dict = {}
        if self.project_root:
            out["project_root"] = self.project_root
        if self.provider:
            out["provider"] = self.provider
        if self.session_id:
            out["session_id"] = self.session_id
        if self.since:
            out["since"] = self.since
        return out


@dataclass
class SanitizeReport:
    """Outcome of a :func:`scan` or :func:`apply` run (value-free).

    Attributes:
        mode: ``"dry-run"`` for a scan, or ``"redact"`` / ``"drop"`` for an apply.
        counts: Per-rule detection histogram (e.g. ``{"AWS": 2}``); rule names only.
        affected_count: Number of distinct secret-bearing turns in scope.
        processed_count: Number of turns this run actually wrote (or deleted).
        incomplete_fts: Count of turns whose Milvus row was cleaned but whose FTS
            rewrite failed — these stay unfinished/retryable.
        audit_path: Filesystem path to the JSONL audit trail for this run.
        status: ``"complete"`` | ``"incomplete"`` | ``"paused"``.
        rotate_warning: True on any apply run — redaction is not rotation, so the
            operator must still rotate the exposed credential.
    """

    mode: str
    counts: dict = field(default_factory=dict)
    affected_count: int = 0
    processed_count: int = 0
    incomplete_fts: int = 0
    audit_path: Optional[str] = None
    status: str = "complete"
    rotate_warning: bool = False


def _ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) and force owner-only 0700 permissions.

    ``mkdir(mode=...)`` is masked by the process umask and is a no-op on an
    already-existing directory, so an explicit ``chmod`` is what actually
    guarantees the operator-private mode required by SESF-42 (Req 5.1) — audit
    and checkpoint state may enumerate which turns held secrets.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def _state_dir() -> Path:
    """Return ``~/.sessionflow`` (resolved against the live HOME), owner-only 0700."""
    return _ensure_private_dir(Path.home() / _STATE_DIRNAME)


def _checkpoint_path() -> Path:
    """Return the durable checkpoint path (``~/.sessionflow/sanitize_state.json``)."""
    return _state_dir() / _CHECKPOINT_NAME


def _audit_path(run_id: str) -> Path:
    """Return the audit JSONL path for ``run_id``, creating the audit dir 0700."""
    audit_dir = _ensure_private_dir(_state_dir() / _AUDIT_DIRNAME)
    return audit_dir / f"redaction-{run_id}.jsonl"


def _new_run_id() -> str:
    """Return a short, filesystem-safe run identifier."""
    return uuid.uuid4().hex[:12]


def _load_checkpoint() -> Optional[dict]:
    """Load the durable checkpoint, or ``None`` when absent/unreadable."""
    path = _checkpoint_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_checkpoint(state: dict) -> None:
    """Persist ``state`` to the checkpoint with 0600 permissions."""
    path = _checkpoint_path()
    path.write_text(json.dumps(state))
    try:
        os.chmod(path, _FILE_MODE)
    except OSError:
        pass


def _allowlist() -> list:
    """Load the operator allowlist via rag_engine settings (impure by design)."""
    _enabled, _mode, allowlist_path = rag_engine._redaction_settings()
    return rag_engine.load_allowlist(allowlist_path)


def _db_path() -> Optional[str]:
    """Return the configured Milvus DB path (used to derive the FTS sidecar)."""
    try:
        import cleanup

        return cleanup.get_db_path()
    except Exception:  # pragma: no cover - cleanup import/lookup never fails in tests
        return None


def _iter_rows(scope: Scope, db_path: Optional[str]) -> Iterator[dict]:
    """Yield each in-scope Milvus row, one at a time, via the keyset scan."""
    filter_expr = scope.to_filter()
    for batch in rag_engine._query_batches(
        _SCAN_FIELDS, filter_expr=filter_expr or None, db_path=db_path
    ):
        for row in batch:
            yield dict(row)


class _AuditWriter:
    """Append-only JSONL audit writer; emits one value-free record per span.

    Records carry only rule/tier/offset/length/masked_snippet (from the Span) plus
    non-sensitive row metadata and the action. A raw secret value never reaches a
    record because the snippet is pre-masked and the value itself is never copied.
    """

    def __init__(self, run_id: str):
        """Open (truncate) the audit file for ``run_id`` with 0600 permissions."""
        self.run_id = run_id
        self.path = _audit_path(run_id)
        # Truncate any prior file for this run id, then lock down permissions.
        self._handle = open(self.path, "w", encoding="utf-8")
        try:
            os.chmod(self.path, _FILE_MODE)
        except OSError:
            pass

    def write(self, *, row: dict, spans: list, action: str) -> None:
        """Append one audit line per span for ``row`` under ``action``."""
        for span in spans:
            record = {
                "run_id": self.run_id,
                "doc_id": row.get("doc_id"),
                "provider": row.get("provider"),
                "source_path": row.get("source_path"),
                "turn_index": row.get("turn_index"),
                "timestamp": row.get("timestamp"),
                "rule": span.rule_name,
                "tier": span.tier,
                "offset": span.start,
                "length": span.end - span.start,
                "masked_snippet": span.masked_snippet,
                "action": action,
            }
            self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        try:
            self._handle.close()
        except OSError:
            pass


def _accumulate_counts(counts: dict, spans: list) -> None:
    """Fold ``spans`` rule names into the ``counts`` histogram in place."""
    for span in spans:
        counts[span.rule_name] = counts.get(span.rule_name, 0) + 1


def scan(scope: Scope) -> SanitizeReport:
    """Dry-run the sanitizer over ``scope`` and report findings without writing.

    Iterates every in-scope turn, detects secrets via
    :func:`secret_redaction.scan_spans` in ``report`` mode, and accumulates per-rule
    counts plus the affected ``doc_id`` set. A value-free audit JSONL and a worklist
    checkpoint are written, but no Milvus/FTS/embedding write occurs.

    Args:
        scope: The selection of turns to inspect.

    Returns:
        A :class:`SanitizeReport` with ``mode="dry-run"`` summarizing the findings.
    """
    run_id = _new_run_id()
    allowlist = _allowlist()
    db_path = _db_path()

    counts: dict = {}
    worklist: list[str] = []
    audit = _AuditWriter(run_id)
    try:
        for row in _iter_rows(scope, db_path):
            _scanned, spans = secret_redaction.scan_spans(
                row.get("document", ""), mode="report", allowlist=allowlist
            )
            if not spans:
                continue
            doc_id = row.get("doc_id")
            if doc_id not in worklist:
                worklist.append(doc_id)
            _accumulate_counts(counts, spans)
            audit.write(row=row, spans=spans, action="dry-run")
    finally:
        audit.close()

    _save_checkpoint(
        {
            "run_id": run_id,
            "created_at": _now_iso(),
            "mode": "scan",
            "scope": scope.as_dict(),
            "worklist": list(worklist),
            "done": [],
            "counts": counts,
            "status": "scanning",
        }
    )

    return SanitizeReport(
        mode="dry-run",
        counts=counts,
        affected_count=len(worklist),
        processed_count=0,
        incomplete_fts=0,
        audit_path=str(audit.path),
        status="complete",
        rotate_warning=False,
    )


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _throttled_embed(redacted: str) -> list:
    """Embed ``redacted`` through the shared budget, honoring the cooldown floor.

    Routes the single-text re-embed through the process-wide embedding budget's
    ``split_batches`` / ``before_batch`` / ``after_batch`` so the MLX cooldown floor
    is never undercut. Returns the embedding vector for ``redacted``.
    """
    budget = rag_engine.get_embedding_budget()
    batches = budget.split_batches([redacted])
    vector: Optional[list] = None
    for batch in batches:
        # Wait out any cooldown the budget asks for; never busy-spin below the floor.
        while True:
            decision = budget.before_batch(len(batch), 0)
            if getattr(decision, "allowed", True):
                break
            delay = getattr(decision, "retry_after_seconds", 0.0) or 0.0
            if delay <= 0:
                break
            time.sleep(delay)
        started = time.monotonic()
        error: Optional[BaseException] = None
        try:
            vectors = rag_engine.embed_texts(batch, is_query=False)
            vector = vectors[0]
        except BaseException as exc:  # noqa: BLE001 - scrub before re-raise
            error = exc
            rag_engine._scrub_exception_args(exc)
            raise
        finally:
            budget.after_batch(time.monotonic() - started, len(batch), error)
    return vector if vector is not None else []


def _build_worklist(
    scope: Scope,
    allowlist: list,
    db_path: Optional[str],
) -> tuple[list[str], dict[str, dict], dict]:
    """Scan ``scope`` to build the affected worklist, row cache, and counts.

    Returns:
        ``(worklist, rows_by_id, counts)`` where ``worklist`` is the ordered list of
        affected ``doc_id``s, ``rows_by_id`` caches each affected row so apply needs
        no extra Milvus round-trip, and ``counts`` is the per-rule histogram.
    """
    worklist: list[str] = []
    rows_by_id: dict[str, dict] = {}
    counts: dict = {}
    for row in _iter_rows(scope, db_path):
        doc_id = row.get("doc_id")
        # Cache every scanned row's payload so a resumed worklist entry whose
        # Milvus document is already redacted (no spans) is still recoverable for
        # an FTS-only converge — the worklist/counts only track secret-bearing turns.
        if doc_id not in rows_by_id:
            rows_by_id[doc_id] = row
        _scanned, spans = secret_redaction.scan_spans(
            row.get("document", ""), mode="report", allowlist=allowlist
        )
        if not spans:
            continue
        if doc_id not in worklist:
            worklist.append(doc_id)
            _accumulate_counts(counts, spans)
    return worklist, rows_by_id, counts


def apply(
    scope: Scope,
    *,
    drop: bool,
    confirmed: bool,
    resume: bool = False,
    pause_event: Optional[Event] = None,
) -> SanitizeReport:
    """Apply the sanitizer destructively over ``scope`` (guarded by ``confirmed``).

    Refuses to perform any read or write when ``confirmed`` is False, returning an
    empty paused report. Otherwise builds (or resumes) the affected worklist and, per
    turn, either redacts + re-embeds + upserts the row (``drop=False``) or deletes it
    (``drop=True``). Progress is checkpointed after each turn so an interrupted run
    resumes from the durable worklist + done-set.

    A turn is marked ``done`` only when both stores succeed. When Milvus succeeds but
    the FTS rewrite fails (``fts_ok is False``), the ``doc_id`` stays in the worklist,
    ``incomplete_fts`` increments, and the run reports ``status="incomplete"``.

    Args:
        scope: The selection of turns to sanitize.
        drop: When True, delete affected turns instead of redacting them.
        confirmed: Must be True for any read or write to occur.
        resume: When True, resume from the durable checkpoint's worklist + done-set.
        pause_event: Optional ``threading.Event``; when set, the run pauses (status
            ``"paused"``) and checkpoints before processing further turns.

    Returns:
        A :class:`SanitizeReport` describing the run; ``rotate_warning`` is always
        True (redaction is not rotation).
    """
    mode = "drop" if drop else "redact"

    # Confirmation gate (Requirement 2.4): no read, no write, no audit file.
    if not confirmed:
        return SanitizeReport(
            mode=mode,
            counts={},
            affected_count=0,
            processed_count=0,
            incomplete_fts=0,
            audit_path=None,
            status="paused",
            rotate_warning=True,
        )

    allowlist = _allowlist()
    db_path = _db_path()

    checkpoint = _load_checkpoint() if resume else None
    if checkpoint is not None:
        run_id = checkpoint.get("run_id") or _new_run_id()
        done: set[str] = set(checkpoint.get("done", []))
        counts: dict = dict(checkpoint.get("counts", {}))
        # Re-scan to recover the row payloads (the checkpoint stores ids, not rows).
        _scan_worklist, rows_by_id, scan_counts = _build_worklist(
            scope, allowlist, db_path
        )
        worklist: list[str] = list(checkpoint.get("worklist", _scan_worklist))
        if not counts:
            counts = scan_counts
    else:
        run_id = _new_run_id()
        done = set()
        worklist, rows_by_id, counts = _build_worklist(scope, allowlist, db_path)

    audit = _AuditWriter(run_id)
    processed = 0
    incomplete_fts = 0
    paused = False

    def _checkpoint(status: str) -> None:
        _save_checkpoint(
            {
                "run_id": run_id,
                "created_at": _now_iso(),
                "mode": mode,
                "scope": scope.as_dict(),
                "worklist": [d for d in worklist if d not in done],
                "done": sorted(done),
                "counts": counts,
                "status": status,
            }
        )

    try:
        for doc_id in list(worklist):
            if doc_id in done:
                continue
            if pause_event is not None and pause_event.is_set():
                paused = True
                break

            row = rows_by_id.get(doc_id)
            if row is None:
                # Affected id with no recoverable row payload — leave for a re-run.
                continue

            if drop:
                fts_ok = _apply_drop(doc_id, row, allowlist, audit, db_path)
            else:
                fts_ok = _apply_redact(doc_id, row, allowlist, audit, db_path)
            if fts_ok:
                done.add(doc_id)
                processed += 1
            else:
                incomplete_fts += 1
            _checkpoint("applying")
    finally:
        audit.close()

    remaining = [d for d in worklist if d not in done]
    if paused:
        status = "paused"
    elif incomplete_fts or remaining:
        status = "incomplete"
    else:
        status = "complete"
    _checkpoint(status)

    return SanitizeReport(
        mode=mode,
        counts=counts,
        affected_count=len(worklist),
        processed_count=processed,
        incomplete_fts=incomplete_fts,
        audit_path=str(audit.path),
        status=status,
        rotate_warning=True,
    )


def _apply_redact(
    doc_id: str,
    row: dict,
    allowlist: list,
    audit: "_AuditWriter",
    db_path: Optional[str],
) -> bool:
    """Redact + re-embed + upsert one turn; return whether the FTS rewrite succeeded.

    On a resumed row already clean in Milvus (``scan_spans`` yields no spans) the
    re-embed + upsert are skipped and only the FTS delete-then-insert of the already
    redacted content is retried, keeping the operation idempotent.

    Returns:
        ``True`` when the row is fully sanitized (Milvus + FTS both ok), ``False``
        when Milvus is clean but the FTS rewrite still needs a retry.
    """
    document = row.get("document", "")
    redacted, spans = secret_redaction.scan_spans(
        document, mode="enforce", allowlist=allowlist
    )

    if not spans:
        # Already redacted in Milvus (resume case): converge FTS only, no re-embed.
        # Pass new_vector=None so the existing fixed-dim vector is preserved — a
        # 0-length vector would corrupt the HNSW/COSINE row.
        result = rag_engine.upsert_document(
            doc_id, new_document=redacted, new_vector=None, db_path=db_path
        )
        return bool(result.milvus_ok and result.fts_ok)

    audit.write(row=row, spans=spans, action="redact")
    vector = _throttled_embed(redacted)
    result = rag_engine.upsert_document(
        doc_id, new_document=redacted, new_vector=vector, db_path=db_path
    )
    return bool(result.milvus_ok and result.fts_ok)


def _apply_drop(
    doc_id: str,
    row: dict,
    allowlist: list,
    audit: "_AuditWriter",
    db_path: Optional[str],
) -> bool:
    """Delete one turn from both stores and audit it (``action="drop"``, no re-embed).

    Returns:
        ``True`` when the turn is fully removed (Milvus + FTS both ok), ``False``
        when the FTS delete failed and the row is still keyword-searchable — the
        caller must keep the doc_id on the worklist, mirroring the redact path.
    """
    _scanned, spans = secret_redaction.scan_spans(
        row.get("document", ""), mode="report", allowlist=allowlist
    )
    audit.write(row=row, spans=spans, action="drop")
    result = rag_engine.delete_by_doc_id(doc_id, db_path=db_path)
    return bool(result.fts_ok)
