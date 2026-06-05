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
"""

import argparse
import json
import os
import sys
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


def cmd_reset(args):
    """Delete all indexed data across every project (with confirmation)."""
    db = get_db_path()

    if not args.yes:
        answer = input("This will delete ALL indexed data (all projects). Continue? [y/N] ")
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

    SESF-11: explicit recovery path for the silent-DataNotMatchException class
    of failures when fields are added to `_expected_schema_fields()`.
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
                providers = sorted(LEGAL_PROVIDERS - {"claude_desktop_cowork"})
            from provider_ingestion import ProviderIngestionService
            import asyncio

            db = get_db_path()
            for provider in providers:
                try:
                    get_manager().enqueue_provider_backfill(provider=provider, mode=mode)
                except ValueError as exc:
                    print(f"Skipping {provider}: {exc}", file=sys.stderr)
            # SESF-8: add_turns_async requires the dedicated MLX executor.
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

    return parser


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
        "status": cmd_status,
        "backfill": cmd_backfill,
    }
    return_code = commands[args.command](args)
    if return_code is not None:
        sys.exit(return_code)


if __name__ == "__main__":
    main()
