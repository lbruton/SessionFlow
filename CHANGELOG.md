# Changelog

All notable changes to this fork are documented here.
For upstream changes, see [UPSTREAM-README.md](UPSTREAM-README.md).

## [Unreleased]

### Added

- Issue-ID extraction at ingestion: every turn is scanned for issue references (`[A-Z][A-Z0-9]+-\d+`, with a technical-standard prefix denylist) and tagged into a new `issue_ids` Milvus `VARCHAR(4096)` field + FTS5 metadata column (SESF-25)
- Optional `issue_id` filter on the `search_all_sessions` and `search_session` MCP tools — structured exact-token pre-filter, combinable with `provider` / `project_root` / `date_from` / `date_to` (SESF-25)
- `get_issue_timeline` MCP tool and `GET /timeline` HTTP endpoint — cross-harness chronological feed of all turns referencing an issue, merged from the structured field + FTS fallback, deduped by `doc_id`, sorted oldest-first, with `limit` / `provider` / `date_from` / `date_to` (SESF-25, SESF-26)
- `detect_project_root()` in `transcript_parser.py` — peeks first 30 lines of a transcript for `cwd`, resolves to git root (SR-2)
- Auto-derive project root from Claude Code slug names via greedy path reconstruction with filesystem verification (SR-2)
- `date_from` / `date_to` optional parameters on `search_all_sessions` tool for date-range filtering (SR-3)
- Date filters applied to both Milvus vector search and FTS5 keyword search (SR-3)
- Progress checkpoints every 100 files during backfill with progress logging (SR-1)
- Milvus error recovery pause (5s) on GOAWAY/readonly/connection errors during backfill (SR-1)
- Startup delay (3s) for backfill task so HTTP server can bind before embedding work begins (SR-1)
- `.claude/project.json` for project onboarding

### Fixed
- Backfill silently dying after Milvus-lite gRPC `GOAWAY` / `too_many_pings` under heavy insert load (SR-1)
- `project_root` empty on 99%+ of indexed rows — slug map wasn't populated during initial bulk index, and unregistered slugs defaulted to `""` (SR-2)
- Bare `-` slug incorrectly mapping to `/` as project root (SR-2)
- Server startup health check timeout caused by backfill monopolizing the async event loop (SR-1)

### Changed
- Backfill loop now throttles with 50ms `asyncio.sleep` between transcripts instead of `sleep(0)` (SR-1)
- `_get_project_root_for_slug()` now attempts path derivation from slug name before returning empty string (SR-2)

## [0.1.0] — 2026-04-16

### Added
- `date_from` / `date_to` filters on `search_all_sessions` (initial implementation before reliability fixes)
- Fork from upstream at `637e6f4`
