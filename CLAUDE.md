# session-rag

Semantic search over Claude Code session transcripts. Fork of [mwgreen/claude-code-session-rag](https://github.com/mwgreen/claude-code-session-rag) with local modifications.

## Upstream Relationship

| | |
|---|---|
| **Upstream** | `https://github.com/mwgreen/claude-code-session-rag.git` (remote: `upstream`) |
| **Fork** | `git@github.com:lbruton/claude-code-session-rag.git` (remote: `origin`) |
| **Relationship** | Detached clone (not a GitHub fork). Upstream is read-only reference. |

### Local Modifications (divergences from upstream)

- Global MCP server with `headersHelper` (replaces per-project `.mcp.json`)
- EmbeddingGemma-300M as default model (configurable, was ModernBERT)
- pymilvus 2.6.10 exclusion (breaks unix socket for milvus-lite)
- Global file watcher on `~/.claude/projects/` (single observer for all projects)
- Date range filtering on `search_all_sessions` (`date_from`/`date_to` params)
- Reliable backfill: 200ms throttle, gRPC keepalive 120s, progress checkpoints every 100 files
- Auto-derive `project_root` from slug names + transcript CWD peek fallback
- Fork documentation: README.md, UPSTREAM-README.md, CHANGELOG.md

When pulling from upstream, review changes against this list to avoid regressing local fixes.

## Operational Gotchas

- **milvus-lite gRPC keepalive** — pymilvus defaults 10s, milvus-lite rejects as `too_many_pings`. Fixed in `rag_engine.py` with `grpc_options: keepalive_time_ms: 120000`.
- **MLX Metal SIGSEGV under sustained load** — EmbeddingGemma crashes GPU driver after ~50min continuous compute. Backfill throttle is 200ms between inserts. Do not reduce below 100ms.
- **milvus-lite single-process lock** — only one process can open `milvus.db`. Can't query DB externally while server holds it. Long-term fix: migrate to containerized Milvus (SRAG-1).
- **Backfill checkpoints every 100 files** — `index_state.json` saves progress. If server crashes mid-backfill, restart picks up from last checkpoint (not from zero).
- **`project_root` for `-/` transcripts** — generic bucket sessions have `cwd="/"`. No project tagging possible. Project-dir transcripts get project_root via slug map + `detect_project_root()` fallback.
- **Never create GitHub issues** — all issues go to DocVault via `/issue`. No `gh issue create`, no GitHub Issues UI. Ever.

## Issue Tracking

Issues use `SRAG-` prefix, stored in `DocVault/Projects/session-rag/Issues/`.

## Tech Stack

- Python 3.13, venv at `./venv/`
- Milvus Lite (vector DB at `~/.session-rag/milvus.db`)
- mlx-embeddings (Apple Silicon only)
- HTTP MCP server on port 7102
- FTS5 (SQLite full-text search, hybrid with vector)

## Git Rules

- `main` is the default branch
- Direct commits OK for now (solo project, no CI)
- Keep upstream syncs as merge commits for clear history
