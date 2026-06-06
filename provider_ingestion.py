"""Provider ingestion execution for queued backfill jobs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

import rag_engine
import transcript_parser
from backfill_manager import BackfillJob, BackfillManager
from embedding_control import get_embedding_budget
from provider_antigravity import AntigravityAdapter
from provider_claude import ClaudeCodeCliAdapter
from provider_codex import CodexAdapter
from provider_opencode import OpenCodeAdapter


def default_provider_adapters() -> Dict[str, object]:
    """Build the provider adapter registry used by server backfill."""
    antigravity_cli = AntigravityAdapter(source_kind="cli")
    antigravity_desktop = AntigravityAdapter(source_kind="desktop")
    return {
        "claude_code_cli": ClaudeCodeCliAdapter(),
        "codex": CodexAdapter(),
        "opencode": OpenCodeAdapter(),
        antigravity_cli.provider: antigravity_cli,
        antigravity_desktop.provider: antigravity_desktop,
    }


class ProviderIngestionService:
    """Drain BackfillManager jobs into provider parsers and the RAG index."""

    def __init__(
        self,
        manager: BackfillManager,
        db_path: str,
        adapters: Optional[Dict[str, object]] = None,
    ):
        """Wire the backfill manager, target DB path, and provider adapters."""
        self.manager = manager
        self.db_path = db_path
        self.adapters = adapters if adapters is not None else default_provider_adapters()

    async def process_queued_jobs(self, max_jobs: Optional[int] = None) -> dict:
        """Process up to ``max_jobs`` queued jobs and return aggregate totals."""
        status = self.manager.status()
        jobs = status.jobs[:max_jobs] if max_jobs is not None else status.jobs
        totals = {"jobs": 0, "processed_sources": 0, "indexed_turns": 0, "errors": 0}
        if status.paused:
            return totals

        for job in jobs:
            provider_status = status.providers.get(job.provider)
            if provider_status and provider_status.paused:
                continue
            result = await self.process_job(job)
            for key in totals:
                totals[key] += result.get(key, 0)
        return totals

    async def process_job(self, job: BackfillJob) -> dict:
        """Run one backfill job through its provider adapter into the RAG index."""
        adapter = self.adapters.get(job.provider)
        if adapter is None:
            self.manager.complete_job(job.job_id, errors=1)
            return {"jobs": 1, "processed_sources": 0, "indexed_turns": 0, "errors": 1}

        state = transcript_parser.load_index_state()
        processed_sources = 0
        indexed_turns = 0
        errors = 0
        sources = self._select_sources(adapter.discover_sources(), job)

        try:
            for source in sources:
                cursor = transcript_parser.get_provider_cursor(state, source.provider, source.source_id)
                try:
                    result = adapter.parse_source(source, cursor=cursor)
                    processed_sources += 1
                    if result.errors:
                        errors += len(result.errors)
                    if result.turns:
                        indexed_turns += await rag_engine.add_turns_async(result.turns, db_path=self.db_path)
                    transcript_parser.set_provider_cursor(state, source.provider, source.source_id, result.cursor)
                except Exception as exc:
                    errors += 1
                    logger.error(
                        "Error processing source %s/%s: %s",
                        source.provider,
                        source.source_id,
                        exc,
                        exc_info=True,
                    )

            transcript_parser.save_index_state(state)
        finally:
            self.manager.complete_job(
                job.job_id,
                processed_sources=processed_sources,
                indexed_turns=indexed_turns,
                errors=errors,
            )
        return {
            "jobs": 1,
            "processed_sources": processed_sources,
            "indexed_turns": indexed_turns,
            "errors": errors,
        }

    def _select_sources(self, sources: Iterable, job: BackfillJob) -> list:
        selected = [source for source in sources if source.status == "eligible"]
        selected.sort(
            key=lambda source: Path(source.path).stat().st_mtime if Path(source.path).exists() else 0,
            reverse=True,
        )
        if job.mode == "recent":
            import time
            cutoff = time.time() - (get_embedding_budget().recent_days * 86400)
            selected = [
                source for source in selected
                if Path(source.path).exists() and Path(source.path).stat().st_mtime >= cutoff
            ]
        if job.source_id:
            selected = [source for source in selected if source.source_id == job.source_id]
        if job.limit is not None:
            selected = selected[:job.limit]
        return selected


async def process_startup_provider_backfill(
    manager: BackfillManager,
    db_path: str,
    enabled_providers: list[str],
    mode: str = "recent",
    startup_delay: float = 0.0,
) -> dict:
    """Queue and process startup-safe provider jobs after the HTTP server binds."""
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)
    limit = get_embedding_budget().max_files_per_run
    for provider in enabled_providers:
        startup_mode = mode if mode in {"recent", "incremental"} else "recent"
        manager.enqueue_provider_backfill(provider=provider, mode=startup_mode, limit=limit, priority=1)
    return await ProviderIngestionService(manager, db_path).process_queued_jobs()
