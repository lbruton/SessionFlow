---
tags: []
doc_type: overview
project: SessionFlow
source: manual
created: "2026-04-26"
updated: "2026-06-08"
---

# SessionFlow

> **Public copy.** Deployment and infra details (hosts, IPs, ports) are in the private companion: `Devops/DocVault/Projects/SessionFlow/Overview.md` (`vault-path private`).

Semantic search over Claude Code session transcripts, expanded in SESF-6 to ingest sessions from multiple AI coding harnesses. HTTP MCP server with hybrid retrieval (Milvus vector + FTS5 keyword) over EmbeddingGemma vectors. Projectized fork of `mwgreen/claude-code-session-rag`.

## At a Glance

| Field | Value |
|-------|-------|
| Repo | [lbruton/SessionFlow](https://github.com/lbruton/SessionFlow) |
| Language | Python (venv + `requirements.txt`) |
| Branches | `main` |
| Issue Tracker | [Plane SESF](https://plane.lbruton.cc/lbruton/projects/3835ead1-4cc4-4f89-8145-4923068f7403/) |
| PR Target | `main` |
| Upstream | [mwgreen/claude-code-session-rag](https://github.com/mwgreen/claude-code-session-rag) |

## Architecture

| Component | File | Role |
|-----------|------|------|
| HTTP MCP server | `http_server.py` | Exposes search tools over MCP/HTTP; provider/source_kind filters and backfill control endpoints |
| RAG engine | `rag_engine.py` | Embedding + Milvus vector search; provider + source_kind columns on every row |
| FTS hybrid | `fts_hybrid.py` | SQLite FTS5 keyword pass, fused via RRF with vectors |
| Transcript parser | `transcript_parser.py` | Reads `~/.claude/projects/**/*.jsonl` (Claude Code CLI) |
| Provider adapters | `provider_adapters.py`, `provider_claude.py`, `provider_codex.py`, `provider_opencode.py`, `provider_antigravity.py` | Per-harness discovery, parsing, and cursor models |
| Backfill manager | `backfill_manager.py` | Provider-aware job queue with pause/resume + recent/incremental/full modes |
| File watcher | `file_watcher.py` | Live ingest of new turns (per-provider `watch_roots()`) |
| Embedding control | `embedding_control.py` | Local-only `EmbeddingIdentity`; hosted provider field reserved but disabled |
| Index hook | `index_hook.py`, `session_start_hook.sh` | SessionStart hook entry |

## Provider Coverage (SESF-6)

| Provider | Source kind | Status |
|----------|-------------|--------|
| Claude Code CLI | `claude_cli_jsonl` | Native, default |
| Codex | `codex_rollout_jsonl` | Native; active + archive rollouts merged by logical session |
| OpenCode | `opencode_storage` | Native; session/message/part records reconstructed; settled-window gate |
| Antigravity CLI | `antigravity_cli` | Native; JSONL transcripts (sibling `.pb`/`.db` flagged but not parsed) |
| Antigravity Desktop | `antigravity_desktop` | Native; distinct provider tag from CLI variant |
| Claude Desktop / CoWork | n/a | **Probe-only** — discovered, not indexed |
| Legacy Gemini CLI | `legacy_gemini_history` | Reserved; out of scope (use Antigravity instead) |

Hosted/OpenAI embeddings are **explicitly deferred**. Embedding stays local MLX (EmbeddingGemma) by policy.

## Operational Notes

- Embedding model is downloaded via `download-model.sh` — EmbeddingGemma local on Apple Silicon (MLX Metal).
- **Gotcha:** `search_all_sessions` silently returns empty when a `project_root` filesystem path is passed. Omit the filter or pass `"*"`.
- **MLX Metal SIGSEGV under sustained load** — EmbeddingGemma crashes the GPU driver after ~50min continuous compute. Backfill throttle is 200ms between inserts; do not drop below 100ms.
- **FTS schema migration** writes a sentinel and emits a `WARNING` log line while the keyword index rebuilds — expected, not an error.
- **LaunchAgent (optional):** `cc.lbruton.sessionflow` user agent autostarts the server at login. Disable via `launchctl unload` if running ad-hoc.

## Environment Variables (SESF-6)

| Var | Purpose |
|-----|---------|
| `SESSIONFLOW_MILVUS_URI` | Standalone Milvus endpoint (else Lite fallback) |
| `SESSIONFLOW_EMBED_BATCH_SIZE` | Embedding batch size for backfill throttling |
| `SESSIONFLOW_EMBED_COOLDOWN_MS` | Inter-batch cooldown to protect MLX Metal driver |
| `SESSIONFLOW_BACKFILL_MODE` | `recent` \| `incremental` \| `full` |
| `SESSIONFLOW_RECENCY_WEIGHT` | Hybrid blend weight `w` in `[0,1]` (SESF-24); default `0.3`, silent fallback if unparseable/out-of-range |
| `SESSIONFLOW_RECENCY_DECAY_DAYS` | Exponential decay constant for `recency_score = exp(-days_old/decay_days)` (SESF-24); default `7`, min `1` |
| `SESSIONFLOW_REDACT` | Secret-redaction guard on/off (SESF-41); default **on** when unset |
| `SESSIONFLOW_REDACT_MODE` | `enforce` (mask) \| `report` (detect + count, store raw) (SESF-41); default `report` when unset |
| `SESSIONFLOW_REDACT_ALLOWLIST` | Path to an operator regex allowlist file, one pattern per line (SESF-41) |

## MCP Tools

`mcp__sessionflow__search_all_sessions`, `search_session`, `get_turns`, `get_session_stats`, `cleanup_sessions`, `get_issue_timeline`. All search tools accept optional `provider` and `source_kind` filters; legacy callers without those args continue to work (defaults preserve Claude behavior). Always pass `session_id` from `CLAUDE_SESSION_ID` env when scoping to the current session — auto-resolution is not supported.

Both search tools also accept an optional `sort_by` (SESF-24): `relevance` (pure RRF), `recency` (timestamp-desc re-rank of the candidate pool), or `hybrid` (blended semantic + recency). Omitting it defaults to `hybrid` — this changed `search_all_sessions`' prior pure-relevance default. Tune the blend via `SESSIONFLOW_RECENCY_WEIGHT` / `SESSIONFLOW_RECENCY_DECAY_DAYS`.

### Issue-ID search & timeline (SESF-25 / SESF-26)

Turn text is scanned at ingestion for tracker issue tokens (`[A-Z][A-Z0-9]+-\d+`, boundary-anchored, with a technical-standard prefix denylist) and the deduplicated, uppercased set is stored in a structured `issue_ids` field — Milvus `VARCHAR(4096)` plus an FTS5 `issue_ids` metadata column.

- **`issue_id` filter** on `search_all_sessions` / `search_session` — optional structured exact-token pre-filter (`issue_ids like "%,SESF-25,%"`), case-insensitive, combinable conjunctively with `provider` / `project_root` / `date_from`. Unset leaves search behavior byte-identical.
- **`get_issue_timeline(issue_id, limit=50, provider, date_from, date_to)`** MCP tool + `GET /timeline` HTTP endpoint — single deduplicated, oldest-first cross-harness feed of every turn referencing an issue. Unions the structured field with an FTS keyword fallback (covering un-tagged turns), deduped by `doc_id`, sorted by `(timestamp, doc_id)`.
- **DEVS-39 dependency:** `issue_ids` is populated only on ingest. Pre-existing rows stay empty until the post-merge DEVS-39 dedicated-Milvus 2.6.x re-index runs, so the structured filter returns empty for historical data until then; the timeline's FTS fallback still surfaces those turns. The shared claude-context Milvus is left untouched.

## Specs

| Spec | Summary |
|------|---------|
| [[../../specflow/SessionFlow/specs/SESF-6-multi-harness-session-ingestion-codex-opencode-antigravity/requirements\|SESF-6 Multi-Harness Session Ingestion]] | Codex, OpenCode, Antigravity CLI + desktop ingestion; provider-aware search; local-only embeddings; 45/45 tests passing |
| [[../../specflow/SessionFlow/specs/SESF-25-issue-id-search-and-timeline/requirements\|SESF-25 Issue-ID Search and Timeline]] | Issue-ID extraction at ingestion (`issue_ids` field), `issue_id` search filter, `get_issue_timeline` MCP tool + `/timeline` HTTP endpoint; 189 passed (50 new tests). Live coverage of historical rows depends on the DEVS-39 re-index |

## Pages

_(none yet — add deep dives as documentation lands)_

Pre-Plane issues: [[../../Archive/Issues-Pre-Plane/SessionFlow/SessionFlow|Archive/Issues-Pre-Plane/SessionFlow]].
