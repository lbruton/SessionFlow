#!/usr/bin/env python3
"""Memory maintenance commands: rebuild, reindex, audit, import and restore-check.

The CLI never embeds. It writes Postgres or resets acknowledgements, and the
running server's sync loop does the indexing. ``restore-check`` restores a
dump into an empty database and asks the server's loopback route to index and
compare it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

import memory_store

SERVER_URL = os.getenv("SESSIONFLOW_URL", "http://127.0.0.1:7102")
RESTORE_TIMEOUT_S = 600.0
_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")


class _DeferredIndex:
    """Index stand-in for CLI writes: every record stays pending for the server's sync loop."""

    async def apply(self, memory_id: str) -> str:
        return "pending"

    async def similar(self, text: str, **kwargs: Any) -> tuple[list, str]:
        return [], "pending"

    async def neighbours(self, memory_id: str, **kwargs: Any) -> list:
        return []

    def lagging(self) -> bool:
        return True


class _StoredVectorIndex(_DeferredIndex):
    """Deferred index whose audit neighbour lookup reads the stored vectors.

    ``neighbours`` queries the vector already in Milvus and never embeds. A failed
    lookup sets ``unavailable`` and returns no neighbours.
    """

    def __init__(self, memory_index: Any) -> None:
        """Wrap a ``MemoryIndex`` that was never started."""
        self._index = memory_index
        self.unavailable: Optional[str] = None

    async def neighbours(self, memory_id: str, **kwargs: Any) -> list:
        """Return the stored-vector neighbours of one memory, or none when Milvus fails."""
        try:
            return await self._index.neighbours(memory_id, **kwargs)
        except Exception as exc:  # Milvus Lite raises plain exceptions when the file is locked.
            self.unavailable = f"{type(exc).__name__}: {exc}"
            return []


def _open_neighbour_index() -> Optional[_StoredVectorIndex]:
    """Open Milvus read-only for neighbour lookups; return None when it cannot be opened."""
    import memory_index
    import rag_engine

    uri = os.getenv("SESSIONFLOW_MILVUS_URI", str(Path.home() / ".sessionflow" / "milvus.db"))
    try:
        if not rag_engine._is_remote_uri(uri) and not Path(uri).exists():
            return None
        client = rag_engine._get_persistent_client(uri)
        index = memory_index.MemoryIndex(
            store=_open_store(), milvus=client, fts_path=Path(uri).with_name("memories_fts.db"),
            host="cli", milvus_uri=uri,
        )
    except Exception:
        return None
    return _StoredVectorIndex(index)


def _attach_neighbours(service: Any) -> Optional[_StoredVectorIndex]:
    """Give the service a stored-vector neighbour lookup; leave it deferred if Milvus is closed."""
    if not hasattr(service, "index"):
        return None
    index = _open_neighbour_index()
    if index is not None:
        service.index = index
    return index


def _open_store() -> memory_store.MemoryStore:
    store = memory_store.MemoryStore(memory_store.fetch_dsn())
    store.migrate()
    return store


def _open_service():
    import memory_service
    import rag_engine

    async def no_query_embedding(text: str) -> list:
        raise RuntimeError("memory_cli does not embed; run search through the server")

    _, _, allowlist_path = rag_engine._redaction_settings()
    return memory_service.MemoryService(
        store=_open_store(), index=_DeferredIndex(), embed_query=no_query_embedding,
        allowlist=rag_engine.load_allowlist(allowlist_path),
        project_inventory=memory_service.project_inventory(),
    )


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False,
                      default=lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v))


# --- rebuild, reindex -------------------------------------------------------


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Reset every acknowledgement so the server re-indexes every record."""
    if not args.yes:
        _err("rebuild resets every index acknowledgement; the server then re-indexes every "
             "memory. Re-run with --yes to continue.")
        return 2
    reset = _open_store().reset_acks()
    print(f"Reset {reset} acknowledgements. The server sync loop re-indexes them.")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Reset the acknowledgements of one record."""
    reset = _open_store().reset_acks([args.memory_id])
    print(f"Reset {reset} acknowledgements for {args.memory_id}.")
    return 0


# --- audit, import ----------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    """Print the audit report; nothing is changed."""
    service = _open_service()
    index = _attach_neighbours(service)
    report = asyncio.run(service.audit())
    if index is None or index.unavailable:
        _err("Similar-memory clusters are unavailable: Milvus could not be read. "
             "Stale items and counts are still reported.")
    if report.get("inventory_checked") is False:
        _err("Unknown-project check skipped: no known project root has a readable "
             ".claude/project.json. Open one session in each project so its "
             "SessionStart hook registers the root, then run the audit again.")
    if args.format == "json":
        print(_json(report))
        return 0
    for project, section in sorted(report.get("projects", {}).items()):
        print(f"## {project}")
        for cluster in section.get("clusters", []):
            print(f"- similar: {', '.join(cluster)}")
        for mid in section.get("stale", []):
            print(f"- stale: {mid}")
        for type_, by_status in sorted(section.get("counts", {}).items()):
            print(f"- {type_}: " + ", ".join(f"{s} {n}" for s, n in sorted(by_status.items())))
    if report.get("unknown_projects"):
        print(f"Unknown projects: {', '.join(report['unknown_projects'])}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Import a converted mem0 export and write the report."""
    import memory_import

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    report = asyncio.run(memory_import.import_records(
        data, service=_open_service(), report_path=Path(args.report)))
    print(_json(report.get("totals", {})))
    return 0


# --- restore-check ----------------------------------------------------------


def _user_tables(conn: psycopg.Connection) -> int:
    return conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema <> ALL(%s)",
        (list(_SYSTEM_SCHEMAS),),
    ).fetchone()[0]


def _memories_schema(conn: psycopg.Connection) -> str:
    row = conn.execute(
        "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memories' "
        "AND table_schema <> ALL(%s) ORDER BY table_schema LIMIT 1",
        (list(_SYSTEM_SCHEMAS),),
    ).fetchone()
    if row is None:
        raise RuntimeError("the dump holds no memories table")
    return row[0]


def _manifest(conn: psycopg.Connection, schema: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        sql.SQL("SELECT memory_id, version, content_hash, deleted_at IS NOT NULL "
                "FROM {}.memories ORDER BY memory_id").format(sql.Identifier(schema))
    ).fetchall()
    return [{"memory_id": str(r[0]), "version": r[1], "content_hash": r[2], "deleted": r[3]}
            for r in rows]


def _clear_target(target_dsn: str) -> None:
    """Drop everything the restore created, leaving the target database empty."""
    with psycopg.connect(target_dsn, autocommit=True) as conn:
        schemas = [r[0] for r in conn.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname <> ALL(%s) "
            "AND nspname NOT LIKE 'pg_%%' AND nspname <> 'public'",
            (list(_SYSTEM_SCHEMAS),),
        )]
        for schema in schemas:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        for (table,) in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall():
            conn.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(
                sql.Identifier(table)))
        for (seq,) in conn.execute(
            "SELECT sequencename FROM pg_sequences WHERE schemaname = 'public'"
        ).fetchall():
            conn.execute(sql.SQL("DROP SEQUENCE IF EXISTS public.{} CASCADE").format(
                sql.Identifier(seq)))


def _post_restore_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the running server to index the restored rows and compare them."""
    request = urllib.request.Request(
        f"{SERVER_URL}/memory/restore-check", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    timeout = float(payload.get("timeout_s", RESTORE_TIMEOUT_S)) + 60
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except ValueError:
            return {"ok": False, "error": f"server returned HTTP {exc.code}"}
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {"ok": False, "error": str(getattr(exc, "reason", exc))}


def cmd_restore_check(args: argparse.Namespace) -> int:
    """Restore a dump into an empty database and check the server rebuilds the same state."""
    if shutil.which("pg_restore") is None:
        _err("pg_restore is not on PATH; install the PostgreSQL 17 client tools.")
        return 2
    target = args.target_dsn
    with psycopg.connect(target) as conn:
        if _user_tables(conn):
            _err("The restore target is not empty. Use an empty database; nothing was changed.")
            return 2
    try:
        proc = subprocess.run(
            ["pg_restore", "--no-owner", "--no-privileges", "--exit-on-error", "-d", target,
             str(args.dump)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            _err(f"pg_restore failed: {lines[-1] if lines else 'unknown error'}")
            return 1
        with psycopg.connect(target) as conn:
            schema = _memories_schema(conn)
            manifest = _manifest(conn, schema)
        payload = {
            "target_dsn": make_conninfo(target, options=f"-c search_path={schema}"),
            "manifest": manifest,
            "timeout_s": args.timeout,
        }
        result = _post_restore_check(payload)
    finally:
        _clear_target(target)
    print(_json(result))
    if not result.get("ok"):
        _err("Restore check failed.")
        return 1
    print(f"Restore check passed: {result.get('expected')} live records rebuilt and matched.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_cli", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rebuild", help="Re-index every memory from Postgres.")
    p.add_argument("--yes", action="store_true", help="Confirm the rebuild.")
    p.set_defaults(func=cmd_rebuild)
    p = sub.add_parser("reindex", help="Re-index one memory.")
    p.add_argument("--memory-id", required=True)
    p.set_defaults(func=cmd_reindex)
    p = sub.add_parser("audit", help="Report clusters, stale items and counts; changes nothing.")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("import", help="Import a converted mem0 export.")
    p.add_argument("path")
    p.add_argument("--report", required=True, help="Where to write the import report.")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("restore-check", help="Prove a dump rebuilds the same committed state.")
    p.add_argument("dump")
    p.add_argument("--target-dsn", required=True, help="An empty database to restore into.")
    p.add_argument("--timeout", type=float, default=RESTORE_TIMEOUT_S,
                   help="Seconds the server may spend indexing.")
    p.set_defaults(func=cmd_restore_check)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one memory command and return its exit code."""
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except memory_store.MemoryStoreError as exc:
        _err(f"{exc.code}: {exc.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
