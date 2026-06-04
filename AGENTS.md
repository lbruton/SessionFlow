# SessionFlow

Semantic search over agent session transcripts. Independent project, originally forked from mwgreen/claude-code-session-rag.

## Origin

| | |
|---|---|
| **Repo** | `git@github.com:lbruton/SessionFlow.git` |
| **Original upstream** | `mwgreen/claude-code-session-rag` (detached, no longer tracked) |
| **Fork point** | `637e6f4` — diverged significantly, now independent |

## Tech Stack

- Python 3.13, venv at `./venv/`
- Milvus Standalone at `192.168.1.83:19530` via `SESSIONFLOW_MILVUS_URI` (fallback: Milvus Lite at `~/.sessionflow/milvus.db`)
- HNSW index (COSINE, M=16, efConstruction=256) on Standalone
- mlx-embeddings with EmbeddingGemma-300M (Apple Silicon Metal)
- HTTP MCP server on port 7102 (Starlette + Uvicorn)
- SQLite FTS5 sidecar for hybrid vector + keyword search
- Reciprocal Rank Fusion (RRF) merge

## Operational Gotchas

- **MLX Metal SIGSEGV under sustained load** — EmbeddingGemma crashes GPU driver after ~50min continuous compute. Backfill throttle is 200ms between inserts. Do not reduce below 100ms.
- **Milvus Lite gRPC keepalive** — only applies when `SESSIONFLOW_MILVUS_URI` is unset (Lite fallback). Standalone doesn't need the workaround.
- **Backfill checkpoints every 100 files** — `index_state.json` saves progress. Restart picks up from last checkpoint.
- **`project_root` for `-/` transcripts** — generic bucket sessions have `cwd="/"`. No project tagging possible.
- **Never create GitHub issues** — all issues go to Plane via `/issue` (which dispatches on `.specflow/config.json` `issue_backend`).
- **Provider backfill controls (SESF-6)** — multi-harness ingestion (`codex`, `opencode`, `antigravity_cli`, `antigravity_desktop`) shares one embedding budget. Tune via `SESSIONFLOW_EMBED_BATCH_SIZE`, `SESSIONFLOW_EMBED_COOLDOWN_MS` (floor 200ms — MLX Metal SIGSEGVs lower), `SESSIONFLOW_BACKFILL_MODE` (`recent`|`incremental`|`full`), `SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN`, `SESSIONFLOW_BACKFILL_MAX_FILES_PER_RUN`, `SESSIONFLOW_BACKFILL_RECENT_DAYS`, `SESSIONFLOW_BACKFILL_PAUSED`. Pause/resume + per-provider control via `python cleanup.py backfill {status|pause|resume|enqueue} [--provider <name>]`. Queue state is durable across restarts.
- **Claude Desktop / CoWork is probe-only** — `claude-code-sessions/**/local_*.json` is discovered and surfaced in status output, but full turn content is not yet indexed. Do not claim searchable support until the parser spike lands.
- **Hosted embeddings deferred** — SESF-6 keeps embedding fully local (MLX). No hosted/OpenAI setup steps, credentials, or collections exist. Future hosted path would require a separate identity/collection to avoid vector mixing.
- **Optional LaunchAgent for multi-harness startup** — `./sessionflow-server.sh install-agent` writes `~/Library/LaunchAgents/cc.lbruton.sessionflow.plist` so the server starts at login before any harness hook races. OPTIONAL — `start/stop/status/restart` behavior is unchanged when not installed. `agent-status` / `uninstall-agent` manage it. `setup.sh` honors `SESSIONFLOW_INSTALL_AGENT=1` for non-interactive installs.
- **Optional hourly backfill LaunchAgent (SESF-7)** — `./sessionflow-server.sh install-backfill-agent` writes `cc.lbruton.sessionflow-backfill.plist` with a `StartInterval` (defaults to 3600s; override via `SESSIONFLOW_BACKFILL_INTERVAL_SECONDS`). It runs `python cleanup.py backfill enqueue --provider claude_code_cli --mode incremental`, decoupling catch-up indexing from MCP session activity. MCP-client SessionStart hooks should connect to the already-running server rather than racing to start it.
- **Schema drift (SESF-11)** — `_ensure_collection` refuses to start when the live Milvus schema differs from `_expected_schema_fields()`. Drop+recreate is opt-in via `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1`, or run `python cleanup.py migrate-schema` explicitly (destructive).
- **`SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN` (SESF-12)** — default raised from 200 to 100,000 so a long-running server isn't silently capped after a few minutes. The budget logs a single WARN the first time the cap is hit so operators don't have to grep for "0 turns" lines.
- **FTS5 thread affinity (SESF-13)** — `FTSIndex` keeps per-thread persistent connections (`threading.local`). Server-mode connections opened on the embed executor and request threads are isolated, and cross-thread `close_all()` is a no-op rather than a noisy WARN.
- **OpenCode timestamps (SESF-14)** — `provider_adapters.normalize_timestamp()` coerces int-ms epochs (and any other numeric/datetime input) to ISO-8601 strings before they hit Milvus's `VARCHAR(64)` timestamp field. All four provider adapters route timestamps through it.

## Code Style

See CLAUDE.md "Code Style" for the docstring policy and the pre-PR `ruff` gate (`ruff.toml`).

## Issue Tracking

Issues use the `SESF-` prefix and are tracked in Plane: <https://plane.lbruton.cc/lbruton/projects/3835ead1-4cc4-4f89-8145-4923068f7403/>.

Renamed from `SRAG-` (originally `SF-` in `.claude/project.json` post-rebrand) on 2026-04-26 with the Plane migration. Pre-migration markdown archived at `DocVault/Archive/Issues-Pre-Plane/SessionFlow/`. New issues are created via `/issue` (which dispatches on `.specflow/config.json` `issue_backend`) or directly via `mcp__plane__create_issue`.

## Git Rules

- `main` is the default branch
- Branch protections: signed commits + PR required
- Worktree branches for all changes

## Project-Local Agent Assets

- Claude project metadata lives in `.claude/project.json`.
- No project-local Claude skills, commands, or agents are currently present under `.claude/`.
- Codex-compatible project-local mirrors belong under `.agents/` if future `.claude/skills`, `.claude/commands`, or `.claude/agents` assets are added.
