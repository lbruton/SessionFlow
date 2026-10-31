#!/usr/bin/env python3
"""
CLI tool to manage and clean up SessionFlow data.

All data lives in a single global DB at ~/.sessionflow/milvus.db.
Use --project to filter to a specific project.

Usage:
    cleanup.py list        [--project <root>]                # List all sessions
    cleanup.py expire      [--days N]                        # Delete turns older than N days (default: 365)
    cleanup.py delete      --session <id>                    # Delete a specific session
    cleanup.py delete      --branch <name>                   # Delete all turns for a branch
    cleanup.py reset                                         # Drop everything and start fresh
    cleanup.py stats       [--project <root>]                # Show index statistics
    cleanup.py reembed     {plan|run|drive|status|verify}    # Re-embed every stored vector in place
    cleanup.py rewrite     {plan|run|drive|status|verify}    # Strip boilerplate and re-embed long rows in place
    cleanup.py retag-project --from <old> --to <new> [--provider P] [--apply --yes]
                                                             # Move turns to a new project root
    cleanup.py restamp-codex [--apply --yes]                 # Rewrite Codex turn times and branches
    cleanup.py prune-codex   [--apply --yes]                 # Delete Codex turns their rollout no longer yields
    cleanup.py dedupe-turns  [--apply --yes]                 # Repair Claude turns indexed mid-write
    cleanup.py dedupe-codex  [--apply --yes]                 # Delete Codex rows of the old turn keys
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request

import rag_engine
from embedding_control import EmbeddingIdentity, get_embedding_budget


def get_db_path() -> str:
    """Milvus URI — remote Standalone if SESSIONFLOW_MILVUS_URI is set, else local Lite."""
    return os.getenv("SESSIONFLOW_MILVUS_URI", str(Path.home() / ".sessionflow" / "milvus.db"))


def get_server_url() -> str:
    """Loopback SessionFlow server URL for local control endpoints."""
    default_port = 7102
    raw_port = os.getenv("SESSIONFLOW_PORT", str(default_port))
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = default_port
    if not 1 <= port <= 65535:
        port = default_port
    return f"http://127.0.0.1:{port}"


def post_backfill_action(payload: dict, timeout: float = 2.0) -> dict:
    """POST a backfill action to the running HTTP server."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{get_server_url()}/backfill",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body) if response_body else {}


def _http_error_message(exc: error.HTTPError) -> str:
    """Extract a clear server rejection message from an HTTPError body."""
    try:
        response_body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc.reason)
    if not response_body:
        return str(exc.reason)
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body
    if isinstance(parsed, dict):
        detail = parsed.get("error") or parsed.get("detail")
        if detail:
            return str(detail)
    return response_body


def _print_server_rejected(action: str, exc: error.HTTPError) -> None:
    print(
        f"Server rejected backfill {action} request: {_http_error_message(exc)}",
        file=sys.stderr,
    )


def cmd_list(args):
    """List indexed sessions with per-session turn and branch counts."""
    db = get_db_path()
    project = getattr(args, 'project', None)
    sessions = rag_engine.list_sessions(project_root=project, db_path=db)

    if not sessions:
        print("No sessions indexed.")
        return

    print(f"{'Session ID':<40} {'Turns':>6} {'Branch':<30} {'First':>20} {'Last':>20}")
    print("-" * 120)
    for s in sessions:
        branches = ", ".join(s["branches"]) if s["branches"] else "(none)"
        first = s["min_ts"][:19] if s["min_ts"] else ""
        last = s["max_ts"][:19] if s["max_ts"] else ""
        print(f"{s['session_id']:<40} {s['turns']:>6} {branches:<30} {first:>20} {last:>20}")

    print(f"\nTotal: {len(sessions)} sessions, {sum(s['turns'] for s in sessions)} turns")


def cmd_expire(args):
    """Delete turns older than ``args.days`` and report before/after counts."""
    db = get_db_path()
    days = args.days

    # Show what would be deleted
    stats_before = rag_engine.get_stats(db_path=db)
    print(f"Current index: {stats_before['total_turns']} turns across {stats_before['sessions']} sessions")
    print(f"Deleting turns older than {days} days...")

    count = rag_engine.delete_older_than(days, db_path=db)
    print(f"Deleted {count} turns.")

    stats_after = rag_engine.get_stats(db_path=db)
    print(f"Remaining: {stats_after['total_turns']} turns across {stats_after['sessions']} sessions")


def cmd_delete(args):
    """Delete indexed turns by ``--session`` or ``--branch``."""
    db = get_db_path()

    if args.session:
        count = rag_engine.delete_by_session(args.session, db_path=db)
        print(f"Deleted {count} turns for session {args.session}")
    elif args.branch:
        count = rag_engine.delete_by_branch(args.branch, db_path=db)
        print(f"Deleted {count} turns for branch '{args.branch}'")
    else:
        print("Error: specify --session or --branch", file=sys.stderr)
        sys.exit(1)


MEMORY_UNTOUCHED = (
    "Memory data is untouched: the memories collection, memories_fts.db and the "
    "Postgres memory store are not changed by transcript cleanup."
)


def cmd_reset(args):
    """Delete all indexed transcript data across every project (with confirmation)."""
    db = get_db_path()

    if not args.yes:
        answer = input("This will delete ALL indexed transcript data (all projects). Continue? [y/N] ")
        if answer.lower() != "y":
            print("Cancelled.")
            return

    rag_engine.clear_collection(db_path=db)

    # Clean up WAL/SHM files left by SQLite FTS
    from fts_hybrid import FTSIndex
    fts_path = Path(FTSIndex.db_path(db))
    for suffix in ("-wal", "-shm"):
        f = Path(str(fts_path) + suffix)
        if f.exists():
            f.unlink()

    print("Reset complete. All session data deleted.")
    print(MEMORY_UNTOUCHED)


def cmd_stats(args):
    """Print index statistics, optionally scoped to ``--project``."""
    db = get_db_path()
    project = getattr(args, 'project', None)
    stats = rag_engine.get_stats(project_root=project, db_path=db)

    if project:
        print(f"Project:      {project}")
    print(f"Total turns:  {stats['total_turns']}")
    print(f"Sessions:     {stats['sessions']}")

    if stats.get("branches"):
        print(f"Branches:     {', '.join(stats['branches'])}")

    if stats.get("by_type"):
        print("\nBy type:")
        for t, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {count}")

    print(f"\nDB location:  {db}")


def cmd_migrate_schema(args):
    """Drop and recreate the Milvus sessions collection to match current code.

    Use it when `_expected_schema_fields()` gains a field: inserts into the old
    collection fail with DataNotMatchException. All turns are lost.
    """
    db = get_db_path()

    if not args.yes:
        answer = input(
            "This drops the Milvus 'sessions' collection (all indexed turns lost). "
            "Continue? [y/N] "
        )
        if answer.lower() != "y":
            print("Cancelled.")
            return

    with rag_engine.milvus_client_for_migration(db_path=db) as client:
        drift = rag_engine.detect_schema_drift(client)
        if drift:
            print(f"Detected schema drift: {drift}")
        rag_engine.migrate_schema(client, db_path=db)
    print("Schema migrated. Backfill the FTS sidecar via the server watcher.")
    print(MEMORY_UNTOUCHED)


def cmd_status(args):
    """Show provider-aware index, embedding, and backfill status."""
    db = get_db_path()
    project = getattr(args, "project", None)
    provider_filter = getattr(args, "provider", None)
    stats = rag_engine.get_stats(project_root=project, db_path=db)
    identity = EmbeddingIdentity.current_local()
    budget = get_embedding_budget().status()

    print("Provider Status")
    providers = stats.get("providers", {})
    if provider_filter:
        providers = {
            provider: count
            for provider, count in providers.items()
            if provider == provider_filter
        }
    if providers:
        for provider, count in sorted(providers.items()):
            print(f"  Provider {provider}: {count} turns")
    else:
        print("  Provider counts unavailable or empty")

    print("\nEmbedding")
    print(f"  Provider: {identity.embedding_provider}")
    print(f"  Model:    {identity.model_name}")
    print(f"  Dim:      {identity.dimension}")
    print(f"  Paused:   {budget.get('paused')}")

    print("\nBackfill")
    print(f"  Mode:       {budget.get('mode')}")
    print(f"  Batch size: {budget.get('batch_size')}")
    print(f"  Cooldown:   {budget.get('cooldown_ms')} ms")
    print(f"\nDB location:  {db}")


def cmd_backfill(args) -> int:
    """Control the provider-aware backfill queue."""
    from backfill_manager import BackfillManager

    manager = None

    def get_manager() -> BackfillManager:
        nonlocal manager
        if manager is None:
            state_path = Path.home() / ".sessionflow" / "backfill_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            manager = BackfillManager(state_path)
        return manager

    provider_arg = getattr(args, "provider", None)
    provider_label = provider_arg or "all"
    provider_kw = None if not provider_arg else provider_arg
    action = args.action

    try:
        if action == "status":
            status = get_manager().status()
            print(f"Backfill status (provider={provider_label}):")
            print(f"  Global paused: {status.paused}")
            print(f"  Queued jobs:   {len(status.jobs)}")
            for job in status.jobs:
                if provider_kw and job.provider != provider_kw:
                    continue
                print(f"    [{job.provider}] {job.mode} job={job.job_id} priority={job.priority}")
            for provider, pstatus in sorted(status.providers.items()):
                if provider_kw and provider != provider_kw:
                    continue
                print(
                    f"  Provider {provider}: queued={pstatus.queued_jobs} "
                    f"paused={pstatus.paused} processed_sources={pstatus.processed_sources} "
                    f"indexed_turns={pstatus.indexed_turns} errors={pstatus.error_count}"
                )
        elif action == "pause":
            get_manager().pause(provider=provider_kw)
            print(f"Backfill paused: provider={provider_label}")
        elif action == "resume":
            get_manager().resume(provider=provider_kw)
            print(f"Backfill resumed: provider={provider_label}")
        elif action == "enqueue":
            if not provider_kw:
                print("Backfill enqueue requires --provider <name>", file=sys.stderr)
                return 2
            mode = getattr(args, "mode", None) or "recent"
            try:
                post_backfill_action({
                    "action": "enqueue",
                    "provider": provider_kw,
                    "mode": mode,
                })
                print(
                    f"Backfill enqueued via running server: provider={provider_kw} mode={mode}"
                )
                return 0
            except error.HTTPError as exc:
                _print_server_rejected("enqueue", exc)
                return 1
            except (OSError, error.URLError, TimeoutError) as exc:
                print(
                    f"Running server unavailable for enqueue ({exc}); "
                    "falling back to local state file.",
                    file=sys.stderr,
                )
            job = get_manager().enqueue_provider_backfill(provider=provider_kw, mode=mode)
            print(f"Backfill enqueued locally: provider={provider_kw} mode={mode} job_id={job.job_id}")
        elif action == "run":
            mode = getattr(args, "mode", None) or "incremental"
            providers_arg = getattr(args, "providers", None)
            payload = {"action": "run", "mode": mode}
            if providers_arg:
                payload["providers"] = providers_arg
            try:
                server_result = post_backfill_action(payload)
                run_result = server_result.get("run", {}) if isinstance(server_result, dict) else {}
                totals = run_result.get("totals", {}) if isinstance(run_result, dict) else {}
                skipped = run_result.get("skipped", []) if isinstance(run_result, dict) else []
                skipped_summary = f" skipped={','.join(skipped)}" if skipped else ""
                print(
                    f"Backfill run complete via running server (mode={mode}): "
                    f"jobs={totals.get('jobs', 0)} "
                    f"processed_sources={totals.get('processed_sources', 0)} "
                    f"indexed_turns={totals.get('indexed_turns', 0)} "
                    f"errors={totals.get('errors', 0)}"
                    f"{skipped_summary}"
                )
                return 0
            except error.HTTPError as exc:
                _print_server_rejected("run", exc)
                return 1
            except (OSError, error.URLError, TimeoutError) as exc:
                print(
                    f"Running server unavailable for run ({exc}); "
                    "falling back to local enqueue and drain.",
                    file=sys.stderr,
                )
            if providers_arg:
                providers = [p.strip() for p in providers_arg.split(",") if p.strip()]
            else:
                from provider_adapters import LEGAL_PROVIDERS
                providers = sorted(LEGAL_PROVIDERS)
            from provider_ingestion import ProviderIngestionService
            import asyncio

            db = get_db_path()
            for provider in providers:
                try:
                    get_manager().enqueue_provider_backfill(provider=provider, mode=mode)
                except ValueError as exc:
                    print(f"Skipping {provider}: {exc}", file=sys.stderr)
            # add_turns_async requires the dedicated MLX executor.
            # Init for the duration of the drain, then tear down so we don't
            # leave Metal contexts hanging in a short-lived CLI process.
            rag_engine.init_server_mode(db_path=db)
            try:
                totals = asyncio.run(
                    ProviderIngestionService(get_manager(), db).process_queued_jobs()
                )
            finally:
                rag_engine.close_server_mode()
            print(
                f"Backfill run complete (mode={mode}): "
                f"jobs={totals.get('jobs', 0)} "
                f"processed_sources={totals.get('processed_sources', 0)} "
                f"indexed_turns={totals.get('indexed_turns', 0)} "
                f"errors={totals.get('errors', 0)}"
            )
        else:
            print(f"Unknown backfill action: {action}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"Backfill {action} failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_sanitize(args) -> int:
    """Retroactively scan or remove secrets already persisted in the index.

    Thin CLI adapter over :mod:`sanitize`. Default posture
    is **dry-run**: report per-rule counts, the affected-turn count, and the audit
    path, writing nothing. ``--apply`` rewrites (redact + re-embed) or, with
    ``--drop``, deletes the affected turns — but only after an explicit ``--yes``.

    The confirmation gate is **refuse-before-writes**: ``--apply`` without ``--yes``
    prints that confirmation is required, makes no read or write (``sanitize.apply``
    is never called), and returns a non-zero exit code. This intentionally avoids
    the interactive ``[y/N]`` prompt that ``reset``/``migrate-schema`` use, matching
    the MCP ``apply && !confirm`` behavior (Requirement 2.4).

    Args:
        args: Parsed argparse namespace with ``apply``, ``drop``, ``yes``, and the
            scope flags (``project``, ``provider``, ``session``, ``since``).

    Returns:
        A process exit code: ``0`` on success, non-zero when the confirmation gate
        refuses an ``--apply`` run.
    """
    import sanitize

    if args.drop and not args.apply:
        print("--drop is only valid with --apply.", file=sys.stderr)
        return 2

    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "rewrite or drop the affected turns.",
            file=sys.stderr,
        )
        return 1

    scope = sanitize.Scope(
        project_root=getattr(args, "project", None),
        provider=getattr(args, "provider", None),
        session_id=getattr(args, "session", None),
        since=getattr(args, "since", None),
    )

    if not args.apply:
        report = sanitize.scan(scope)
        _print_sanitize_report(report)
        return 0

    report = sanitize.apply(scope, drop=args.drop, confirmed=True)
    _print_sanitize_report(report)
    print(
        "\nWARNING: redaction is not rotation. The detected secret was already "
        "exposed — rotate the affected key/credential at its source.",
    )
    return 0


def _print_sanitize_report(report) -> None:
    """Print a value-free summary of a sanitize ``report`` (counts only, no values).

    Emits the mode, per-rule detection counts, affected/processed/incomplete-FTS
    tallies, and the audit-file path. It deliberately reads only the report's
    aggregate fields, never any document text or raw secret value.
    """
    print(f"Mode:           {report.mode}")
    print(f"Affected turns: {report.affected_count}")
    if report.mode != "dry-run":
        print(f"Processed:      {report.processed_count}")
        if report.incomplete_fts:
            print(f"Incomplete FTS: {report.incomplete_fts} (re-run to converge)")
    print(f"Status:         {report.status}")

    counts = report.counts or {}
    if counts:
        print("\nDetected secrets by rule:")
        for rule, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {rule}: {count}")
    else:
        print("\nNo secrets found in scope.")

    if report.audit_path:
        print(f"\nAudit log: {report.audit_path}")


def cmd_retag_project(args) -> int:
    """Move indexed turns from an old project root to a new one, without re-embedding.

    Finds every turn whose ``project_root`` is ``--from`` or lies under it
    (in Milvus and the FTS sidecar), and rewrites it to the same relative path
    under ``--to`` via ``rag_engine.update_row_metadata``. Dry run by default:
    it prints the counts and writes nothing. ``--apply`` refuses without
    ``--yes`` before touching either store. Re-running is safe: moved turns no
    longer match, and a turn whose FTS rewrite failed is found again.

    Args:
        args: Parsed argparse namespace with ``from_root``, ``to_root``,
            ``provider``, ``apply`` and ``yes``.

    Returns:
        ``0`` on success, ``1`` when the confirmation gate refuses or any turn
        failed to rewrite, ``2`` for invalid paths.
    """
    from provider_adapters import path_has_prefix

    old, new = args.from_root.rstrip("/"), args.to_root.rstrip("/")
    if not old.startswith("/") or not new.startswith("/"):
        print("--from and --to must be absolute paths.", file=sys.stderr)
        return 2
    if path_has_prefix(new, old):
        print("--to must not be --from or lie under it.", file=sys.stderr)
        return 2

    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "rewrite the affected turns.",
            file=sys.stderr,
        )
        return 1

    db_path = get_db_path()
    found = rag_engine.find_doc_ids_by_project_root(
        old, provider=args.provider, db_path=db_path,
    )
    by_root: dict = {}
    for root in found.values():
        by_root[root] = by_root.get(root, 0) + 1

    scope = f" (provider {args.provider})" if args.provider else ""
    print(f"{len(found)} turns under {old}{scope}:")
    for root, count in sorted(by_root.items()):
        print(f"  {count:>7}  {root}  ->  {new + root[len(old):]}")

    if not args.apply:
        print("\nDry run: nothing written. Re-run with --apply --yes to retag.")
        return 0

    milvus_failed = fts_failed = 0
    for doc_id, root in found.items():
        result = rag_engine.update_row_metadata(
            doc_id, db_path=db_path, project_root=new + root[len(old):],
        )
        if not result.milvus_ok:
            milvus_failed += 1
        elif not result.fts_ok:
            fts_failed += 1

    print(f"\nRetagged: {len(found) - milvus_failed - fts_failed}")
    if milvus_failed or fts_failed:
        print(f"Milvus failures:     {milvus_failed}")
        print(f"FTS-only failures:   {fts_failed} (re-run to converge)")
        return 1
    return 0


def _codex_adapter():
    from provider_codex import CodexAdapter

    return CodexAdapter()


def cmd_restamp_codex(args) -> int:
    """Rewrite ``timestamp`` and ``git_branch`` on indexed Codex turns, without re-embedding.

    Re-parses every Codex rollout on disk from the start, compares each turn's
    values with Milvus and the FTS sidecar, and rewrites the turns that differ
    via ``rag_engine.update_row_metadata``. Turns whose rollout is gone are left
    alone. Dry run by default; ``--apply`` refuses without ``--yes`` before any read.

    Args:
        args: Parsed argparse namespace with ``apply`` and ``yes``.

    Returns:
        ``0`` on success, ``1`` when the confirmation gate refuses or any turn
        failed to rewrite.
    """
    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "rewrite the affected turns.",
            file=sys.stderr,
        )
        return 1

    adapter = _codex_adapter()
    wanted: dict = {}
    for source in adapter.discover_sources():
        for turn in adapter.parse_source(source, cursor=None).turns:
            values = {
                field: turn.get(field) or ""
                for field in ("timestamp", "git_branch") if turn.get(field)
            }
            if values:
                wanted[turn["doc_id"]] = values

    db_path = get_db_path()
    try:
        stored = rag_engine.stored_metadata_by_provider(
            "codex", ["timestamp", "git_branch"], db_path=db_path,
        )
    except RuntimeError as e:
        print(f"{e}. Nothing written; re-run once the FTS sidecar is readable.",
              file=sys.stderr)
        return 1
    changes = {
        doc_id: values for doc_id, values in wanted.items()
        if doc_id in stored and any(
            store.get(field) != value
            for store in stored[doc_id] for field, value in values.items()
        )
    }

    print(f"Codex turns parsed from disk: {len(wanted)}")
    print(f"Indexed Codex turns:          {len(stored)}")
    print(f"Indexed turns to rewrite:     {len(changes)}")

    if not args.apply:
        print("\nDry run: nothing written. Re-run with --apply --yes to rewrite.")
        return 0

    milvus_failed = fts_failed = 0
    for count, (doc_id, values) in enumerate(sorted(changes.items()), start=1):
        result = rag_engine.update_row_metadata(doc_id, db_path=db_path, **values)
        if not result.milvus_ok:
            milvus_failed += 1
        elif not result.fts_ok:
            fts_failed += 1
        if count % 1000 == 0:
            print(f"  {count}/{len(changes)}", flush=True)

    print(f"\nRewritten: {len(changes) - milvus_failed - fts_failed}")
    if milvus_failed or fts_failed:
        print(f"Milvus failures:     {milvus_failed}")
        print(f"FTS-only failures:   {fts_failed} (re-run to converge)")
        return 1
    return 0


def cmd_prune_codex(args) -> int:
    """Delete indexed Codex turns that a full parse of their rollout no longer produces.

    Only sessions whose rollout is still on disk are touched; turns of a deleted
    rollout stay searchable. The index is read before the rollouts, so a turn
    indexed during the run is always in the parse. Dry run by default;
    ``--apply`` refuses without ``--yes`` before any read.

    Args:
        args: Parsed argparse namespace with ``apply`` and ``yes``.

    Returns:
        ``0`` on success, ``1`` when the confirmation gate refuses, the FTS read
        fails, or any delete failed in either store.
    """
    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "delete the orphan turns.",
            file=sys.stderr,
        )
        return 1

    db_path = get_db_path()
    try:
        stored = rag_engine.stored_metadata_by_provider("codex", ["timestamp"], db_path=db_path)
    except RuntimeError as e:
        print(f"{e}. Nothing deleted; re-run once the FTS sidecar is readable.",
              file=sys.stderr)
        return 1

    adapter = _codex_adapter()
    parsed: set = set()
    sessions: set = set()
    for source in adapter.discover_sources():
        result = adapter.parse_source(source, cursor=None)
        # A rollout that failed to read is not an empty one; keep its rows.
        if result.errors:
            continue
        sessions.add(source.logical_session_id)
        for turn in result.turns:
            parsed.add(turn["doc_id"])

    orphans = sorted(
        doc_id for doc_id in stored
        if doc_id not in parsed
        and doc_id.removeprefix("codex:").rsplit(":", 1)[0] in sessions
    )

    print(f"Codex turns parsed from disk: {len(parsed)}")
    print(f"Indexed Codex turns:          {len(stored)}")
    print(f"Orphan turns to delete:       {len(orphans)}")

    if not args.apply:
        print("\nDry run: nothing deleted. Re-run with --apply --yes to delete.")
        return 0

    fts_failed = 0
    for count, doc_id in enumerate(orphans, start=1):
        if not rag_engine.delete_by_doc_id(doc_id, db_path=db_path).fts_ok:
            fts_failed += 1
        if count % 1000 == 0:
            print(f"  {count}/{len(orphans)}", flush=True)

    print(f"\nDeleted: {len(orphans) - fts_failed}")
    if fts_failed:
        print(f"FTS delete failures: {fts_failed} (re-run to converge)")
        return 1
    return 0


def cmd_dedupe_turns(args) -> int:
    """Repair Claude Code turns indexed mid-write (see :mod:`dedupe_turns`).

    Dry run by default: prints the affected groups and writes nothing. ``--apply``
    refuses without ``--yes`` before any read. Rebuilt turns are embedded in this
    process, through the same embedding budget as ingestion.

    Args:
        args: Parsed argparse namespace with ``apply`` and ``yes``.

    Returns:
        ``0`` on success, ``1`` when the confirmation gate refuses or any delete
        failed in FTS.
    """
    import dedupe_turns

    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "rebuild and delete the affected turns.",
            file=sys.stderr,
        )
        return 1

    db_path = get_db_path()
    plan = dedupe_turns.plan(db_path=db_path)
    print(f"Duplicate turn groups: {plan.duplicate_groups}")
    print(f"User-only turns:       {plan.user_only_rows}")
    print(f"Affected sessions:     {len(plan.affected_sessions)}")

    if not args.apply:
        print("\nDry run: nothing written. Re-run with --apply --yes to repair.")
        return 0

    result = dedupe_turns.apply(plan, db_path=db_path)
    print(f"\nSessions with a transcript: {result.sessions_on_disk}")
    print(f"Sessions without one:       {result.sessions_missing}")
    print(f"Turns rebuilt:              {result.rebuilt_turns}")
    print(f"Rows deleted:               {result.deleted_rows}")
    failed = False
    if result.deferred_turns:
        print(f"Rebuilds deferred by the embedding budget: {result.deferred_turns} "
              "(old rows kept; re-run to finish)")
        failed = True
    if result.fts_failed:
        print(f"FTS delete failures: {result.fts_failed} (re-run to converge)")
        failed = True
    return 1 if failed else 0


def cmd_dedupe_codex(args) -> int:
    """Delete Codex rows that the task-keyed parse replaces (see :mod:`dedupe_codex`).

    Dry run by default. ``--apply`` refuses without ``--yes`` before any read. It
    never embeds: a session on disk waits until the backfill has indexed its turns.

    Args:
        args: Parsed argparse namespace with ``apply`` and ``yes``.

    Returns:
        ``0`` on success, ``1`` when the confirmation gate refuses, a session still
        waits for the backfill on ``--apply``, or any delete failed in FTS.
    """
    import dedupe_codex

    if args.apply and not args.yes:
        print(
            "Refusing to apply: confirmation required. Re-run with --yes to "
            "delete the replaced Codex rows.",
            file=sys.stderr,
        )
        return 1

    db_path = get_db_path()
    plan = dedupe_codex.plan(db_path=db_path)
    stale = sum(len(ids) for ids in plan.stale.values())
    duplicates = sum(len(ids) for ids in plan.duplicates.values())
    print(f"Sessions with a rollout:   {plan.sessions_on_disk}")
    print(f"Sessions without one:      {plan.sessions_missing}")
    print(f"Replaced rows to delete:   {stale}")
    print(f"Duplicate rows to delete:  {duplicates}")
    if plan.unreadable_sessions:
        print(f"Sessions whose rollout failed to read: {plan.unreadable_sessions} "
              "(their rows are kept)")
    if plan.waiting_sessions:
        print(f"Sessions waiting for the backfill: {plan.waiting_sessions} "
              f"({plan.waiting_turns} turns not indexed yet; their rows are kept). Run "
              "`cleanup.py backfill enqueue --provider codex --mode full`, then re-run.")

    if not args.apply:
        print("\nDry run: nothing deleted. Re-run with --apply --yes to delete.")
        return 0

    result = dedupe_codex.apply(plan, db_path=db_path)
    print(f"\nRows deleted: {result.deleted_rows}")
    print(f"Audit file:   {result.audit_path}")
    failed = bool(plan.waiting_sessions)
    if result.fts_failed:
        print(f"FTS delete failures: {result.fts_failed} (re-run to converge)")
        failed = True
    return 1 if failed else 0


def cmd_reembed(args) -> int:
    """Drive the in-place re-embed (see :mod:`reembed`).

    ``plan`` is read-only and ``status`` prints the checkpoint. ``run`` processes
    one chunk in this process, and ``drive`` spawns one ``run`` subprocess per chunk
    with idle gaps, so no process outlives the MLX Metal crash window and this
    parent never loads MLX itself. ``run`` and ``drive`` refuse without ``--yes``,
    before any read or write. ``verify`` compares stored vectors with fresh
    single-text embeddings and writes nothing.

    Args:
        args: Parsed argparse namespace for the ``reembed`` subcommand.

    Returns:
        A process exit code: ``0`` on success, ``1`` on refusal, error, or a
        failed verify.
    """
    import reembed

    if args.action != "verify":
        return _chunked_action("reembed", reembed, args, _print_reembed_report)
    result = reembed.verify(sample=args.sample, threshold=args.threshold,
                            db_path=get_db_path())
    print(f"Checked:   {result.checked}")
    print(f"Passed:    {result.passed} (cos >= {result.threshold})")
    print(f"Min cos:   {result.min_cos:.6f}")
    print(f"Mean cos:  {result.mean_cos:.6f}")
    return 0 if result.checked and result.passed == result.checked else 1


def _chunked_action(command: str, module, args, print_report) -> int:
    """Run the ``plan``, ``status``, ``run`` or ``drive`` action of a chunked command.

    ``run`` and ``drive`` refuse without ``--yes``, before any read or write.
    """
    action = args.action
    if action in ("run", "drive") and not args.yes:
        print(f"Refusing to {action}: confirmation required. Re-run with --yes.",
              file=sys.stderr)
        return 1

    if action == "status":
        print(json.dumps(module.status(), indent=2))
        return 0

    if action == "plan":
        print_report(module.plan(db_path=get_db_path()))
        return 0

    if action == "run":
        report = module.run(
            confirmed=True,
            max_minutes=args.max_minutes,
            max_turns=args.max_turns,
            restart=args.restart,
            db_path=get_db_path(),
        )
        print_report(report)
        return 0 if report.status in ("done", "paused") else 1

    return _drive_chunks(command, module.status, args)


def _drive_chunks(command: str, status, args) -> int:
    """Run ``cleanup.py <command> run`` chunks in fresh processes until the checkpoint is done.

    Each chunk is capped at ``--chunk-minutes`` and followed by a ``--gap-minutes``
    idle, so no process outlives the MLX Metal crash window and this parent never
    loads MLX. Gives up after ``--max-failures`` consecutive chunk crashes.
    """
    cmd_base = [sys.executable, str(Path(__file__).resolve()), command, "run", "--yes",
                "--max-minutes", str(float(args.chunk_minutes))]
    failures = 0
    first = True
    while True:
        cmd = list(cmd_base)
        if first and args.restart:
            cmd.append("--restart")
        first = False
        print(f"[{command} drive] starting chunk: {' '.join(cmd[2:])}", flush=True)
        result = subprocess.run(cmd, check=False)
        state_status = status().get("status")
        if result.returncode == 0:
            failures = 0
            if state_status == "done":
                print(f"[{command} drive] complete.")
                return 0
        else:
            failures += 1
            print(f"[{command} drive] chunk exited with {result.returncode} "
                  f"({failures}/{args.max_failures} consecutive failures)", file=sys.stderr)
            if failures >= args.max_failures:
                print(f"[{command} drive] giving up; resume later with the same command.",
                      file=sys.stderr)
                return 1
        gap = float(args.gap_minutes) * 60
        print(f"[{command} drive] idling {gap:.0f}s before the next chunk", flush=True)
        time.sleep(gap)


def cmd_rewrite(args) -> int:
    """Drive the in-place rewrite of stale rows (see :mod:`rewrite`).

    The actions match ``reembed``: ``plan`` scans and writes nothing, ``run`` and
    ``drive`` refuse without ``--yes``, and ``verify`` writes nothing.

    Args:
        args: Parsed argparse namespace for the ``rewrite`` subcommand.

    Returns:
        A process exit code: ``0`` on success, ``1`` on refusal, error, or a
        failed verify.
    """
    import rewrite

    if args.action != "verify":
        return _chunked_action("rewrite", rewrite, args, _print_rewrite_report)
    result = rewrite.verify(sample=args.sample, threshold=args.threshold,
                            db_path=get_db_path())
    print(f"Checked:    {result.checked}")
    print(f"Passed:     {result.passed} (stripped, cos >= {result.threshold})")
    print(f"Unstripped: {result.unstripped}")
    print(f"Min cos:    {result.min_cos:.6f}")
    print(f"Mean cos:   {result.mean_cos:.6f}")
    if not result.complete:
        print("Note:       the embedding budget stopped the check; the sample is incomplete.",
              file=sys.stderr)
    ok = result.complete and result.checked and result.passed == result.checked
    return 0 if ok else 1


def _print_rewrite_report(report) -> None:
    """Print a rewrite report's counters."""
    print(f"Status:    {report.status}")
    print(f"Scanned:   {report.scanned}")
    print(f"Stripped:  {report.stripped}")
    print(f"Long:      {report.long}")
    print(f"Dropped:   {report.dropped}")
    print(f"Total:     {report.total}")
    print(f"Done:      {report.done}")
    print(f"Deleted:   {report.deleted}")
    print(f"Missing:   {report.missing}")
    print(f"Errors:    {report.errors}")
    print(f"FTS retry: {report.fts_pending}")
    print(f"Cursor:    {report.cursor}")
    if report.status == "dry-run":
        print(f"Estimate:  {report.estimated_seconds / 3600:.1f} h of compute")
    if report.backup_path:
        print(f"Backup:    {report.backup_path}")
    if report.message:
        print(f"Note:      {report.message}")


def _print_reembed_report(report) -> None:
    """Print a re-embed report's counters."""
    print(f"Status:    {report.status}")
    print(f"Total:     {report.total}")
    print(f"Done:      {report.done}")
    print(f"Missing:   {report.missing}")
    print(f"Errors:    {report.errors}")
    print(f"Cursor:    {report.cursor}")
    if report.status == "dry-run":
        print(f"Estimate:  {report.estimated_seconds / 3600:.1f} h of compute")
    if report.backup_path:
        print(f"Backup:    {report.backup_path}")
    if report.message:
        print(f"Note:      {report.message}")


def build_parser():
    """Build the argparse parser for the cleanup CLI subcommands."""
    parser = argparse.ArgumentParser(
        description="Manage SessionFlow index data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all indexed sessions")
    p_list.add_argument("--project", help="Filter to a specific project root")

    # expire
    p_expire = subparsers.add_parser("expire", help="Delete turns older than N days")
    p_expire.add_argument("--days", type=int, default=365, help="Max age in days (default: 365)")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete by session or branch")
    p_delete.add_argument("--session", help="Session ID to delete")
    p_delete.add_argument("--branch", help="Git branch to delete")

    # reset
    p_reset = subparsers.add_parser("reset", help="Delete all data (full reset)")
    p_reset.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("--project", help="Filter to a specific project root")

    # migrate-schema
    p_migrate = subparsers.add_parser(
        "migrate-schema",
        help="Drop+recreate Milvus collection to match current schema (destructive)",
    )
    p_migrate.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # sanitize
    p_sanitize = subparsers.add_parser(
        "sanitize",
        help="Retroactively scan/remove secrets already indexed (dry-run by default)",
    )
    sanitize_mode = p_sanitize.add_mutually_exclusive_group()
    sanitize_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Report findings without writing (default)",
    )
    sanitize_mode.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite (redact + re-embed) or drop affected turns; requires --yes",
    )
    p_sanitize.add_argument(
        "--drop",
        action="store_true",
        help="With --apply, delete affected turns instead of redacting them",
    )
    p_sanitize.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirm an --apply run (required; no interactive prompt)",
    )
    p_sanitize.add_argument("--project", help="Restrict to a project root")
    p_sanitize.add_argument("--provider", help="Restrict to a provider")
    p_sanitize.add_argument("--session", help="Restrict to a session id")
    p_sanitize.add_argument("--since", help="Restrict to turns at/after an ISO date")

    # retag-project
    p_retag = subparsers.add_parser(
        "retag-project",
        help="Move turns from an old project root to a new one (dry-run by default)",
    )
    p_retag.add_argument("--from", dest="from_root", required=True,
                         help="Old project root; turns under it move too")
    p_retag.add_argument("--to", dest="to_root", required=True, help="New project root")
    p_retag.add_argument("--provider", help="Restrict to a provider")
    p_retag.add_argument("--apply", action="store_true",
                         help="Rewrite the turns; requires --yes")
    p_retag.add_argument("--yes", "-y", action="store_true",
                         help="Confirm an --apply run (required; no interactive prompt)")

    # restamp-codex
    p_restamp = subparsers.add_parser(
        "restamp-codex",
        help="Rewrite Codex turn times and branches from the rollouts (dry-run by default)",
    )
    p_restamp.add_argument("--apply", action="store_true",
                           help="Rewrite the turns; requires --yes")
    p_restamp.add_argument("--yes", "-y", action="store_true",
                           help="Confirm an --apply run (required; no interactive prompt)")

    # prune-codex
    p_prune = subparsers.add_parser(
        "prune-codex",
        help="Delete Codex turns their rollout no longer yields (dry-run by default)",
    )
    p_prune.add_argument("--apply", action="store_true",
                         help="Delete the turns; requires --yes")
    p_prune.add_argument("--yes", "-y", action="store_true",
                         help="Confirm an --apply run (required; no interactive prompt)")

    # dedupe-codex
    p_dedupe_codex = subparsers.add_parser(
        "dedupe-codex",
        help="Delete Codex rows of the old line-number turn keys (dry-run by default)",
    )
    p_dedupe_codex.add_argument("--apply", action="store_true",
                                help="Delete the rows; requires --yes")
    p_dedupe_codex.add_argument("--yes", "-y", action="store_true",
                                help="Confirm an --apply run (required; no interactive prompt)")

    # dedupe-turns
    p_dedupe = subparsers.add_parser(
        "dedupe-turns",
        help="Repair Claude turns indexed mid-write (dry-run by default)",
    )
    p_dedupe.add_argument("--apply", action="store_true",
                          help="Rebuild and delete the turns; requires --yes")
    p_dedupe.add_argument("--yes", "-y", action="store_true",
                          help="Confirm an --apply run (required; no interactive prompt)")

    # status
    p_status = subparsers.add_parser("status", help="Show provider/backfill/embedding status")
    p_status.add_argument("--project", help="Filter to a specific project root")
    p_status.add_argument("--provider", help="Filter status to a provider")

    # backfill controls
    p_backfill = subparsers.add_parser("backfill", help="Control provider backfill")
    backfill_sub = p_backfill.add_subparsers(dest="action", required=True)
    for action in ("status", "pause", "resume"):
        p_action = backfill_sub.add_parser(action, help=f"{action} provider backfill")
        p_action.add_argument("--provider", help="Provider to control")
    p_enqueue = backfill_sub.add_parser("enqueue", help="Enqueue provider backfill")
    p_enqueue.add_argument("--provider", required=True, help="Provider to enqueue")
    p_enqueue.add_argument("--mode", default="recent", choices=("recent", "incremental", "full"))
    p_run = backfill_sub.add_parser(
        "run",
        help=(
            "Enqueue + drain backfill for one or more providers (defaults to all). "
            "Designed for the hourly LaunchAgent: pairs enqueue with an immediate "
            "synchronous drain so jobs don't sit waiting for the next server restart."
        ),
    )
    p_run.add_argument("--mode", default="incremental", choices=("recent", "incremental", "full"))
    p_run.add_argument(
        "--providers",
        help="Comma-separated provider list (default: all enabled providers).",
    )

    _add_chunked_command(
        subparsers, "reembed",
        help="Re-embed every stored vector in place with the current padding mask",
        plan_help="Show what a run would do (read-only)",
        run_help="Re-embed one chunk in this process",
        verify_help="Compare stored vectors with fresh single-text embeddings",
    )
    _add_chunked_command(
        subparsers, "rewrite",
        help="Strip boilerplate from stored rows and re-embed rows over the sliding window",
        plan_help="Scan and count what a run would do (read-only)",
        run_help="Rewrite one chunk in this process",
        verify_help="Check rewritten rows for boilerplate and stale vectors",
    )

    return parser


def _add_chunked_command(subparsers, name: str, *, help: str, plan_help: str,
                         run_help: str, verify_help: str) -> None:
    """Add a ``{plan|run|drive|status|verify}`` subcommand driven by ``_drive_chunks``."""
    sub = subparsers.add_parser(name, help=help).add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help=plan_help)
    sub.add_parser("status", help=f"Print the {name} checkpoint")
    p_run = sub.add_parser("run", help=run_help)
    p_run.add_argument("--yes", "-y", action="store_true", help="Confirm (required)")
    p_run.add_argument("--max-minutes", type=float, default=None,
                       help="Stop this chunk after N minutes")
    p_run.add_argument("--max-turns", type=int, default=None,
                       help="Stop this chunk after N turns")
    p_run.add_argument("--restart", action="store_true",
                       help="Discard the checkpoint and take a fresh snapshot")
    p_drive = sub.add_parser(
        "drive", help="Run chunks in fresh processes with idle gaps until done",
    )
    p_drive.add_argument("--yes", "-y", action="store_true", help="Confirm (required)")
    p_drive.add_argument("--chunk-minutes", type=float, default=35.0,
                         help="Wall-time cap per chunk process (default 35)")
    p_drive.add_argument("--gap-minutes", type=float, default=10.0,
                         help="Idle time between chunks (default 10)")
    p_drive.add_argument("--max-failures", type=int, default=3,
                         help="Consecutive chunk crashes before giving up (default 3)")
    p_drive.add_argument("--restart", action="store_true",
                         help="Discard the checkpoint on the first chunk")
    p_verify = sub.add_parser("verify", help=verify_help)
    p_verify.add_argument("--sample", type=int, default=200, help="Rows to check")
    p_verify.add_argument("--threshold", type=float, default=0.9999,
                          help="Minimum cosine to pass")


def main():
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "expire": cmd_expire,
        "delete": cmd_delete,
        "reset": cmd_reset,
        "stats": cmd_stats,
        "migrate-schema": cmd_migrate_schema,
        "sanitize": cmd_sanitize,
        "retag-project": cmd_retag_project,
        "restamp-codex": cmd_restamp_codex,
        "prune-codex": cmd_prune_codex,
        "dedupe-turns": cmd_dedupe_turns,
        "dedupe-codex": cmd_dedupe_codex,
        "status": cmd_status,
        "backfill": cmd_backfill,
        "reembed": cmd_reembed,
        "rewrite": cmd_rewrite,
    }
    return_code = commands[args.command](args)
    if return_code is not None:
        sys.exit(return_code)


if __name__ == "__main__":
    main()
