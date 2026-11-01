# SessionFlow

Semantic search over coding-agent conversation history. One persistent HTTP server indexes transcripts as they change and serves hybrid vector and keyword search over MCP.

`version.py` holds the version, and [CHANGELOG.md](CHANGELOG.md) lists the changes in each version. `curl http://127.0.0.1:7102/health` reports the version a server runs.

## What it does

SessionFlow watches session transcripts from Claude Code and other providers. It embeds every turn with a local MLX model (EmbeddingGemma-300M on Apple Silicon) and exposes the index through MCP tools. Search by meaning, keyword, date range, project, git branch or issue ID.

## Architecture

```text
MCP clients (Claude Code, Codex, OpenCode, Antigravity)
    ↓ POST /mcp (MCP StreamableHTTP)
SessionFlow HTTP server (SESSIONFLOW_PORT, default 7102)
    ├── Embedding: EmbeddingGemma-300M via MLX Metal (local)
    ├── Vectors: Milvus Standalone (HNSW index), or Milvus Lite
    ├── Keywords: SQLite FTS5 sidecar
    ├── Search: Reciprocal Rank Fusion of vector and keyword results
    └── Watcher: FSEvents → debounce → incremental parse → index
```

- The embedding model runs on the local GPU through Metal. SessionFlow makes no embedding API calls.
- `SESSIONFLOW_MILVUS_URI` selects Milvus Standalone. When it is unset, SessionFlow uses embedded Milvus Lite.
- One server process serves every client. It loads the model once and pools connections.
- A file watcher on `~/.claude/projects/` indexes new turns as they are written.

## MCP tools

| Tool | Returns |
|------|---------|
| `search_session` | Turns that match a query in one project, or in one session with `session_id`. Omit `query` to list recent turns. |
| `search_all_sessions` | Turns that match a query, with `project_root`, `git_branch`, `provider`, `date_from`, `date_to` and `issue_id` filters. |
| `get_turns` | The turns around one turn of a session. Requires `session_id` and `turn_index` from a search result. |
| `get_session_stats` | Turn, session, branch and provider counts for one project, or for the index with `"*"`. |
| `get_issue_timeline` | Every turn that mentions an issue ID, oldest first. Requires `issue_id`. |
| `cleanup_sessions` | Deletes turns by age, session ID or git branch, across every project. |
| `sanitize_index` | A dry-run report of secrets in the index. Removes them with `apply` and `confirm`. |

`search_session`, `search_all_sessions` and `get_session_stats` take `project_root`: a path, a folder name such as `CoWork`, or `"*"` for every project. The tool descriptions ask the model to pass its working directory. Without the argument, the server asks the client for its MCP roots and uses the first folder. Claude Code answers with its working directory. A client can also name the project in the `X-Project-Root` header. When none of these names a project, the tool refuses the call and asks for `project_root`.

### Search examples

```text
search_all_sessions("deployment decisions", date_from="2026-04-08", date_to="2026-04-08")
search_all_sessions("what broke in production", date_from="2026-04-01")
search_all_sessions(project_root="*")
get_turns(session_id="<session_id from a result>", turn_index=12)
```

## Setup

```bash
./setup.sh                    # venv, dependencies, model download, hooks
```

Restart Claude Code after setup to load the MCP server.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSIONFLOW_MILVUS_URI` | `~/.sessionflow/milvus.db` | Milvus URI: `http://host:port` for Standalone, a file path for Lite. |
| `SESSIONFLOW_HOST` | `127.0.0.1` | HTTP server bind address. |
| `SESSIONFLOW_PORT` | `7102` | HTTP server port. |
| `SESSIONFLOW_MODEL` | `embeddinggemma` | Embedding model: `embeddinggemma` or `modernbert`. |
| `SESSIONFLOW_EXPIRE_DAYS` | `365` | Prune turns older than this many days. |
| `SESSIONFLOW_WATCH` | `true` | Turn the transcript file watcher on or off. |
| `SESSIONFLOW_URL` | `http://127.0.0.1:7102` | Server URL that the hooks call. |
| `SESSIONFLOW_SEARCH_LOG` | off | Set to `1` to log each search and follow-up call to `~/.sessionflow/search-log.jsonl`. Queries are redacted. |
| `SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT` | `0.7` | Recency weight of a search whose query asks for the latest state, with words such as "latest", "most recent" or "now". Other searches use `SESSIONFLOW_RECENCY_WEIGHT`. |
| `SESSIONFLOW_RERANKER` | `qwen3` | Reranks the top results of a search with a query: `qwen3`, `jina` or `off`. See [Reranking](#reranking). |
| `SESSIONFLOW_RERANK_TOP` | `10` | Number of top results the reranker reorders. |
| `SESSIONFLOW_RERANK_MAX_TOKENS` | `512` | Tokens of each result the reranker reads. |

## Running

```bash
./sessionflow-server.sh start     # Start the server and watchdog
./sessionflow-server.sh stop      # Stop the server
./sessionflow-server.sh restart   # Restart the server
./sessionflow-server.sh status    # Check health
~/.sessionflow/sessionflow-launcher.sh start # Hook-safe start through the LaunchAgent

curl http://127.0.0.1:7102/health # Health check
```

### Optional macOS LaunchAgent

Several harnesses can start at the same time, and their `SessionStart` hooks then race to start the server. The optional LaunchAgent starts SessionFlow at login, before any hook runs. Each harness then connects to the running server.

```bash
./sessionflow-server.sh install-agent     # Write and bootstrap ~/Library/LaunchAgents/cc.lbruton.sessionflow.plist
./sessionflow-server.sh agent-status      # Show the plist and launchctl state
./sessionflow-server.sh uninstall-agent   # Boot out and remove the plist
```

The `start`, `stop`, `restart` and `status` commands work the same with or without the LaunchAgent. `setup.sh` offers to install it. Set `SESSIONFLOW_INSTALL_AGENT=1` to install it without a prompt.

### Reranking

A reranker reads the top results of a search and reorders them by how well each one answers the query. The default is `qwen3`, which `setup.sh` downloads. It adds a second model of about 1.2 GB, and each search with a query takes about 2 to 3 seconds longer. Searches without a query, such as the session-start listing, never use it. Until the model loads, or when it is missing, search runs without it.

The project's retrieval eval ran each setting on version 0.14.0 on 2026-11-01. It used one running server on an M-series Mac, 77 queries, and reranked the top 10 results:

| `SESSIONFLOW_RERANKER` | Model | License | MRR | nDCG@10 | Median search |
| --- | --- | --- | --- | --- | --- |
| `off` | none | | 0.766 | 0.730 | 0.15 s |
| `qwen3` (default) | Qwen3-Reranker-0.6B | Apache 2.0 | 0.828 | 0.783 | 1.99 s |
| `jina` | jina-reranker-v3.5-mlx | CC BY-NC 4.0 | 0.871 | 0.806 | 2.88 s |

Reranking helps most when the answer names an identifier or sits deep in a long turn. Those queries rose from MRR 0.82 and 0.76 with reranking off to 1.00 and 0.96 with `jina`.

`jina` ranks best, and `qwen3` is faster. To switch to `jina`, download it, then set the variable for the server:

```bash
./download-model.sh jina
SESSIONFLOW_RERANKER=jina ./sessionflow-server.sh restart
```

Set `SESSIONFLOW_RERANKER=off` the same way to turn reranking off. launchd does not pass shell variables, so under the LaunchAgent add the variable to the plist and reload it:

```bash
PLIST=~/Library/LaunchAgents/cc.lbruton.sessionflow.plist
/usr/libexec/PlistBuddy -c "Add :EnvironmentVariables:SESSIONFLOW_RERANKER string jina" "$PLIST"
launchctl bootout "gui/$(id -u)/cc.lbruton.sessionflow"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
```

To change the value later, use `Set` in place of `Add`, and reload. `install-agent` rewrites the plist, so add the variable again after it. `curl -s http://127.0.0.1:7102/health` reports the state in `ranking.reranker_state`. It is `ready`, `loading`, or `failed:` with the reason. Each reranked search result shows `Rerank: #k`.

### Retrieval eval

`eval_retrieval.py` scores a labelled query set against the running server. Each JSONL line names an `id`, a `group`, a `tool`, a `query`, optional `args` and the `expected` IDs (`memory_id` or `session_id:turn_index`). `tests/eval/fixture_queries.jsonl` shows the format.

```bash
./venv/bin/python eval_retrieval.py run --queries queries.jsonl --sort-by hybrid --sort-by relevance --out reports/
./venv/bin/python eval_retrieval.py sweep --queries queries.jsonl --var SESSIONFLOW_RECENCY_WEIGHT --values 0,0.15,0.3,0.5 --out reports/ --yes
./venv/bin/python eval_retrieval.py freshness     # Time a canary memory until search finds it
./venv/bin/python eval_retrieval.py usage --days 7 # Roll up the search usage log
```

`sweep` restarts the LaunchAgent once per value. `SESSIONFLOW_EVAL_QUERIES=<file> SESSIONFLOW_EVAL_OUT=<folder> ./sessionflow-server.sh install-eval-agent` runs `nightly` every day at 03:30.

## Memories

SessionFlow also stores memories: notes an agent writes on purpose, such as session summaries, decisions and handoffs. Postgres holds them verbatim. The Milvus `memories` collection and a local `memories_fts.db` file are search indexes that SessionFlow rebuilds from Postgres. Memory is optional. Without `SESSIONFLOW_MEMORY_DSN`, the server serves transcripts only and `/health` reports memory as not configured.

| Tool | Returns |
|------|---------|
| `add_memory` | The stored memory, or the existing one with the same topic key or text, and a list of similar memories. |
| `search_memories` | Memories that match a query in one project, or in every project with `project: "*"`. |
| `list_memories` | Memories newest first, with a cursor for the next page. |
| `get_memory` / `update_memory` / `delete_memory` | One memory, read, changed or deleted by ID. |
| `get_memory_context` | A memory, the session that produced it and other sessions about its issue IDs. |

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSIONFLOW_MEMORY_DSN` | unset | Postgres connection string. SessionFlow reads it from the environment only. |
| `SESSIONFLOW_MEMORY_SYNC_INTERVAL` | `5` | Seconds between index sync passes. |
| `SESSIONFLOW_MEMORY_TICK_TIMEOUT` | `600` | Seconds one sync pass may run. A pass that runs longer is cancelled, its waiting tasks are logged, and `/health` reports it under `memory.sync_error`. `memory.index.stale` is true when the last pass is older than this plus three intervals. |
| `SESSIONFLOW_MEMORY_SIMILAR_THRESHOLD` | `0.88` | Similarity at which `add_memory` lists a memory as similar. |
| `SESSIONFLOW_MEMORY_AUDIT_AGE_DAYS` | `21` | Age at which the audit lists an open item or blocker. |
| `SESSIONFLOW_HOST_NAME` | short host name | Names this machine's keyword index in the acknowledgements. |

`setup.sh` copies `hooks/memory-session-start.py` into the Claude Code and Codex hook folders. At session start it shows the three newest session summaries and recent turns, and falls back to a local cache when the server cannot answer. Setup registers the hook for Codex and prints the Claude Code `~/.claude/settings.json` SessionStart entry for you to add by hand.

```bash
python memory_cli.py audit                        # Likely duplicates, aged open items, counts, unknown projects
python memory_cli.py rebuild --yes                # Re-index every memory from Postgres
python memory_cli.py import export.json --report report.json
python memory_cli.py restore-check memory.dump --target-dsn "$EMPTY_DSN"
```

## Provider support

SessionFlow ingests native structured transcripts. It does not read terminal logs.

| Provider | Status | Source kind | Notes |
|----------|--------|-------------|-------|
| `claude_code_cli` | Searchable | `claude_code_jsonl` (`~/.claude/projects/**/*.jsonl`) | Watcher and backfill. Includes Claude Desktop Code-tab sessions. |
| `claude_cowork_legacy` | Searchable, one-time | `claude_cowork_legacy_jsonl` (`~/Library/Application Support/Claude/local-agent-mode-sessions/*/*/local_*/.claude/projects/**/*.jsonl`) | Legacy Cowork agent-mode sessions. Backfill only, enqueued by hand. |
| `codex` | Searchable | `codex_rollout_jsonl` | Native rollout JSONL. |
| `opencode` | Searchable | `opencode_storage` | Native storage. |
| `antigravity_cli` | Searchable | `antigravity_cli_transcript_jsonl` (`brain/<id>/.system_generated/logs/transcript.jsonl`) | Sibling `.pb` artifacts are not parsed. |
| `antigravity_desktop` | Searchable | `antigravity_desktop_transcript_jsonl` | Desktop and IDE transcript JSONL. |

### Antigravity and Gemini CLI

Antigravity replaces Gemini CLI. The `.pb` (protobuf) artifacts beside Antigravity transcripts have no stable schema, so SessionFlow does not parse them. Legacy Gemini CLI history (`legacy_gemini_history`) is a source-kind constant for one-time import work. SessionFlow does not ingest it automatically.

### Claude Desktop and Cowork

Claude Desktop Code-tab sessions, former Cowork projects included, are ordinary Claude Code transcripts under `~/.claude/projects/`. They are searchable as `claude_code_cli`. The `claude-code-sessions/**/local_*.json` files beside them hold only metadata (title, `cliSessionId`).

Legacy Cowork agent-mode sessions under `~/Library/Application Support/Claude/local-agent-mode-sessions/` ran Claude Code with a sandboxed config directory. The `claude_cowork_legacy` provider ingests them. It tags each session with the first folder the user selected (from the sibling `local_<id>.json`), or `unknown`. This provider is not in the startup backfill. Run it once:

```bash
python cleanup.py backfill enqueue --provider claude_cowork_legacy --mode full
```

If Desktop has pruned or moved that folder, set `SESSIONFLOW_COWORK_LEGACY_ROOT` to a backup copy first.

## Embedding budget

All embedding is local MLX. Backfill runs under a budget so a long import cannot saturate the GPU. `embedding_control.py` holds the full list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SESSIONFLOW_EMBED_BATCH_SIZE` | `16` | Maximum turns per embedding batch. |
| `SESSIONFLOW_EMBED_COOLDOWN_MS` | `200` | Sleep between batches. Values below 200 are raised to 200, because the MLX Metal driver crashes under sustained load below it. |
| `SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN` | `100000` | Maximum turns embedded per backfill run. |
| `SESSIONFLOW_BACKFILL_MAX_FILES_PER_RUN` | `100` | Maximum source files visited per backfill run. |
| `SESSIONFLOW_BACKFILL_RECENT_DAYS` | `14` | Window for `recent` mode. |
| `SESSIONFLOW_BACKFILL_MODE` | `recent` | One of `recent`, `incremental` or `full`. |
| `SESSIONFLOW_BACKFILL_PAUSED` | unset | When true at startup, backfill starts paused. |

### Backfill modes

Backfill is per provider. Its queue lives in the index state directory and survives a restart.

- `recent` indexes sources modified in the last `SESSIONFLOW_BACKFILL_RECENT_DAYS`. It is the default, so recent turns become searchable first.
- `incremental` resumes each source from its last cursor and does not rescan.
- `full` visits every source for the provider. The per-run turn and file caps still apply.

Maintenance commands (see `cleanup.py`):

```bash
python cleanup.py status                          # Provider, embedding and backfill snapshot
python cleanup.py status --provider codex         # One provider
python cleanup.py backfill status                 # Queue status
python cleanup.py backfill pause                  # Pause all providers
python cleanup.py backfill pause --provider codex # Pause one provider
python cleanup.py backfill resume                 # Resume all providers
python cleanup.py backfill enqueue --provider antigravity_cli --mode recent
```

Pause state and queued jobs persist on disk. A restart or a LaunchAgent relaunch resumes the same plan.

## Retroactive secret sanitizer

The ingestion guard redacts secrets in new turns. `cleanup.py sanitize` finds secrets that were indexed before the guard caught them. It removes them from the Milvus `document` field, the FTS `content` column and the embedding vector. It uses the same detector as the ingestion guard.

A dry run is the default. It reports per-rule counts, the number of affected turns and an audit path, and writes nothing to the index:

```bash
python cleanup.py sanitize                              # Dry run over the whole index
python cleanup.py sanitize --provider claude_code_cli   # Scope by provider
python cleanup.py sanitize --project /path/to/repo --since 2026-05-01
```

`--apply` redacts the text of each affected turn, re-embeds it and overwrites the row. `--apply --drop` deletes the affected turns instead. Both need `--yes`; there is no interactive prompt. Without `--yes`, `--apply` refuses before any read or write and exits non-zero:

```bash
python cleanup.py sanitize --apply --yes                # Redact and re-embed in place
python cleanup.py sanitize --apply --yes --drop         # Delete affected turns
```

The scope flags (`--project`, `--provider`, `--session`, `--since`) apply to a dry run and to `--apply`. `--drop` is valid only with `--apply`.

Every run writes a JSONL audit trail under `~/.sessionflow/audit/` with mode 0600. It holds rule names, tiers, integer offsets and masked snippets, and never a secret value. Standard output holds counts only.

> Removing a secret from the index does not revoke it. Rotate any credential that was indexed. The sanitizer prints this warning on every apply run.

## Hosted embeddings

SessionFlow embeds locally only. It has no hosted or OpenAI embedding path, credentials or collections. A hosted path would need its own embedding identity and collection, so that vectors from two models never mix.

## Features

- Hybrid search: Milvus vector similarity and FTS5 keyword matches, merged with Reciprocal Rank Fusion.
- HNSW index on Milvus Standalone.
- A pure-Python git root resolver with a cache, so indexing forks no subprocesses.
- Non-blocking startup: the server binds its port first, then loads the model and runs backfill in the background.
- Multi-project search through the `project_root` filter.
- `date_from` and `date_to` filters on `search_all_sessions` and `get_issue_timeline`.
- Incremental indexing: byte offsets per transcript, with a checkpoint every 100 files.

## Origins

SessionFlow started as a fork of [mwgreen/claude-code-session-rag](https://github.com/mwgreen/claude-code-session-rag). It is an independent project with Milvus Standalone, HNSW indexing, background startup and multi-client hardening.

## License

MIT
