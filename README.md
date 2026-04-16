# SessionFlow

Semantic search over Claude Code conversation history. Persistent HTTP server with hybrid vector + keyword search, real-time transcript indexing, and multi-project support.

## What it does

SessionFlow watches your Claude Code session transcripts, embeds them with a local MLX model (EmbeddingGemma-300M on Apple Silicon), and makes them searchable via MCP tools. Every conversation turn is indexed — search by keyword, semantic meaning, date range, project, or git branch.

## Architecture

```
Claude Code terminals (6-8 concurrent)
    ↓ POST /mcp (MCP StreamableHTTP)
SessionFlow HTTP server (port 7102)
    ├── Embedding: EmbeddingGemma-300M via MLX Metal (local, ~600 MB)
    ├── Vectors: Milvus Standalone (remote, HNSW index)
    ├── Keywords: SQLite FTS5 (local sidecar)
    ├── Search: Hybrid RRF merge (vector + keyword)
    └── Watcher: FSEvents → debounce → incremental parse → index
```

- **Embedding model** runs locally on Apple Silicon GPU (Metal) — no API calls
- **Vector storage** on Milvus Standalone (Portainer) via `SESSIONFLOW_MILVUS_URI` — or embedded Milvus Lite fallback
- **One server process** serves all terminals — model loaded once, connections pooled
- **Real-time indexing** via file watcher on `~/.claude/projects/`

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_session` | Search current session with recency bias |
| `search_all_sessions` | Cross-session semantic search. Optional `git_branch`, `date_from`, `date_to`, `project_root` filters |
| `get_turns` | Retrieve context around a specific turn |
| `get_session_stats` | Index statistics: turn count, session count, branches |
| `cleanup_sessions` | Delete session data by age, session ID, or branch |

### Search examples

```
search_all_sessions("deployment decisions", date_from="2026-04-08", date_to="2026-04-08")
search_all_sessions("what broke in production", date_from="2026-04-01")
search_session("the milvus migration", session_id="<CLAUDE_SESSION_ID>")
```

## Setup

```bash
./setup.sh                    # venv, deps, model download, hooks
# Restart Claude Code to activate
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSIONFLOW_MILVUS_URI` | `~/.sessionflow/milvus.db` | Milvus URI — `http://host:port` for Standalone, file path for Lite |
| `SESSIONFLOW_HOST` | `127.0.0.1` | HTTP server bind address |
| `SESSIONFLOW_PORT` | `7102` | HTTP server port |
| `SESSIONFLOW_MODEL` | `embeddinggemma` | Embedding model (`embeddinggemma` or `modernbert`) |
| `SESSIONFLOW_EXPIRE_DAYS` | `365` | Auto-prune turns older than N days |
| `SESSIONFLOW_WATCH` | `true` | Enable real-time file watcher |
| `SESSIONFLOW_URL` | `http://127.0.0.1:7102` | Server URL (used by hooks) |

## Running

```bash
./sessionflow-server.sh start     # Start server + watchdog
./sessionflow-server.sh stop      # Stop server
./sessionflow-server.sh restart   # Restart
./sessionflow-server.sh status    # Check health

curl http://127.0.0.1:7102/health # Health check
```

## Key features

- **Hybrid search** — vector similarity (Milvus) + keyword matching (FTS5), merged with Reciprocal Rank Fusion
- **HNSW index** on Milvus Standalone — O(log n) search over 21K+ turns
- **Pure Python git root resolver** — no subprocess forks, cached lookups (0.4ms for 10K calls)
- **Non-blocking startup** — FTS backfill and transcript backfill run as background tasks
- **Multi-project** — 30+ projects indexed, searchable by `project_root` filter
- **Date-range filtering** — `date_from` / `date_to` on all search tools
- **Incremental indexing** — byte-offset tracking per transcript, checkpoint every 100 files

## Origins

Originally forked from [mwgreen/claude-code-session-rag](https://github.com/mwgreen/claude-code-session-rag). Diverged significantly with Milvus Standalone migration, subprocess fork elimination, HNSW indexing, background startup, and multi-terminal hardening. Now an independent project.

## License

MIT
