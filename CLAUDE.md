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

When pulling from upstream, review changes against this list to avoid regressing local fixes.

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
