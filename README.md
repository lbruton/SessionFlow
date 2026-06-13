# SessionFlow

Semantic search over Claude Code conversation history. Persistent HTTP server with hybrid vector + keyword search, real-time transcript indexing, and multi-project support.

## What it does

SessionFlow watches your Claude Code session transcripts, embeds them with a local MLX model (EmbeddingGemma-300M on Apple Silicon), and makes them searchable via MCP tools. Every conversation turn is indexed — search by keyword, semantic meaning, date range, project, or git branch.

## Architecture

```text
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

```text
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
~/.sessionflow/sessionflow-launcher.sh start # Hook-safe start via LaunchAgent

curl http://127.0.0.1:7102/health # Health check
```

### Optional macOS LaunchAgent (autostart at login)

When several harnesses (Claude Code, Codex, OpenCode, Antigravity CLI) launch
at the same time, their `SessionStart` hooks race to start the server. The
optional user LaunchAgent starts SessionFlow at login *before* any hook fires,
so every harness simply attaches to an already-running server.

```bash
./sessionflow-server.sh install-agent     # write & bootstrap ~/Library/LaunchAgents/cc.lbruton.sessionflow.plist
./sessionflow-server.sh agent-status      # show plist + launchctl state
./sessionflow-server.sh uninstall-agent   # bootout + remove plist
```

The LaunchAgent is OPTIONAL — `start`/`stop`/`restart`/`status` behavior is
unchanged whether it is installed or not. `setup.sh` will offer to install it
interactively, or non-interactively when `SESSIONFLOW_INSTALL_AGENT=1`.

## Provider support matrix

SessionFlow ingests sessions from multiple coding agents. Native structured
sources are preferred; terminal/log fallbacks are not used in this release.

| Provider | Status | Source kind | Notes |
|----------|--------|-------------|-------|
| `claude_code_cli` | Searchable | `claude_code_jsonl` (`~/.claude/projects/**/*.jsonl`) | Original SessionFlow source. Watcher + backfill both supported. |
| `claude_desktop_cowork` | Probe only | `claude_desktop_sessions` (`~/Library/Application Support/Claude/claude-code-sessions/**/local_*.json`) | Discovery surfaces files in health/status output. Full turn-content ingestion deferred pending a parser spike. |
| `codex` | Searchable | `codex_rollout_jsonl` | Native rollout JSONL; provider-tagged. |
| `opencode` | Searchable | `opencode_storage` | Native storage; provider-tagged. |
| `antigravity_cli` | Searchable | `antigravity_cli_transcript_jsonl` (`brain/<id>/.system_generated/logs/transcript.jsonl`) | Authoritative source per discovery. Sibling `.pb` artifacts are NOT parsed in this release. |
| `antigravity_desktop` | Searchable | `antigravity_desktop_transcript_jsonl` | Desktop/IDE transcript JSONL. Source kind is distinguished from `antigravity_cli` in diagnostics. |

### Antigravity migration paths

Antigravity is the successor to Gemini CLI. SessionFlow treats legacy Gemini
CLI artifacts as migration context only:

- `antigravity_cli` ingests `brain/<id>/.system_generated/logs/transcript.jsonl`.
  Sibling `.pb` (protobuf) artifacts are opaque without a stable schema and are
  not parsed in this release.
- `antigravity_desktop` ingests desktop/IDE transcript JSONL.
- Legacy Gemini CLI history (`legacy_gemini_history`) is recognized as a
  source-kind constant for one-time import work but is not auto-ingested.

### Claude Desktop / CoWork

`~/Library/Application Support/Claude/claude-code-sessions/**/local_*.json`
files are discovered and reported in health/status output, but full turn
content is not yet indexed — a parser spike is required before claiming
searchable support. Treat it as **probe-only** for now.

## Local resource controls (embedding budget)

All embedding work in SessionFlow is local MLX (EmbeddingGemma-300M). Backfill
respects a configurable budget so long historical imports cannot saturate the
GPU. See `embedding_control.py` for the authoritative list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SESSIONFLOW_EMBED_BATCH_SIZE` | `16` | Max turns embedded per batch. |
| `SESSIONFLOW_EMBED_COOLDOWN_MS` | `200` | Sleep between batches. **Hard floor of 200ms** — MLX Metal driver SIGSEGVs under sustained load below this. |
| `SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN` | `200` | Max turns embedded per backfill invocation. |
| `SESSIONFLOW_BACKFILL_MAX_FILES_PER_RUN` | `100` | Max source files visited per backfill invocation. |
| `SESSIONFLOW_BACKFILL_RECENT_DAYS` | `14` | Window for `recent` mode. |
| `SESSIONFLOW_BACKFILL_MODE` | `recent` | One of `recent`, `incremental`, `full`. |
| `SESSIONFLOW_BACKFILL_PAUSED` | unset | If truthy at startup, backfill begins paused. |

### Backfill modes and pause/resume

Backfill is provider-aware and durable across restarts (queue state lives in
SessionFlow's index state directory). Modes:

- `recent` — only sources modified in the last `SESSIONFLOW_BACKFILL_RECENT_DAYS`.
  This is the default so the most useful recall lands first.
- `incremental` — pick up from the last cursor for each source; no rescans.
- `full` — exhaustive backfill across all sources for the provider. Use
  sparingly; runs are still bounded by the per-run turn/file caps above.

Maintenance commands (see `cleanup.py`):

```bash
python cleanup.py status                          # provider + embedding + backfill snapshot
python cleanup.py status --provider codex         # per-provider view
python cleanup.py backfill status                 # queue status
python cleanup.py backfill pause                  # pause all providers
python cleanup.py backfill pause --provider codex # pause one provider
python cleanup.py backfill resume                 # resume all
python cleanup.py backfill enqueue --provider antigravity_cli --mode recent
```

Pause state and queued jobs persist on disk, so a restart (or LaunchAgent
re-launch) resumes the same plan.

## Retroactive secret sanitizer

If a secret was indexed before the ingestion-time redaction guard caught it,
`cleanup.py sanitize` finds and removes it from the Milvus `document` field, the
FTS5 `content` column, and the embedding vector derived from them. Detection
reuses the same engine as the ingestion guard, so what the sanitizer flags is
exactly what live ingestion would now redact.

**Dry-run is the default** — it reports per-rule counts, the affected-turn count,
and an audit path, and writes nothing to the index:

```bash
python cleanup.py sanitize                              # dry-run over the whole index
python cleanup.py sanitize --provider claude_code_cli   # scope by provider
python cleanup.py sanitize --project /path/to/repo --since 2026-05-01
```

`--apply` rewrites the affected turns (redact the text, re-embed, overwrite the
row). With `--drop` it deletes the affected turns instead. Both require an
explicit `--yes` — there is **no interactive prompt**. `--apply` without `--yes`
refuses before any read or write and exits non-zero:

```bash
python cleanup.py sanitize --apply --yes                # redact + re-embed in place
python cleanup.py sanitize --apply --yes --drop         # delete affected turns
```

Scope flags (`--project`, `--provider`, `--session`, `--since`) apply to both
dry-run and apply. `--drop` is only valid with `--apply`.

Every run writes a **value-free** JSONL audit trail under `~/.sessionflow/audit/`
(0600) — rule names, tiers, integer offsets, and pre-masked snippets only, never
a raw secret value. Output to stdout is likewise counts-only.

> **Redaction is not safety — rotate the key.** Removing a secret from the index
> does not un-expose it. Once a credential has been written anywhere, treat it as
> compromised and rotate it at the source; the sanitizer warns about this on every
> apply but cannot perform the rotation for you.

## Hosted embeddings — deferred

Hosted/OpenAI embeddings are **deferred and not implemented in SESF-6**.
SessionFlow remains self-hosted: all embedding runs through local MLX. The
provider/identity layer leaves room for a future opt-in hosted path, but no
hosted setup steps, credentials, or collections are created by this release.
If local resource controls prove insufficient for your workload, hosted
embeddings will be tracked as a separate future issue.

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
