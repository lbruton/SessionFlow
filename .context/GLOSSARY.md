# SessionFlow — Domain Glossary

Session transcript search and retrieval MCP server.

## Ingestion & Providers

**Turn**:
A single conversation message/exchange — the atomic record SessionFlow indexes and returns.
_Avoid_: message, chunk

**Provider**:
A source harness whose transcripts are ingested: `claude_code_cli`, `codex`, `opencode`, `antigravity_cli`, `antigravity_desktop`, `claude_desktop_cowork`.
_Avoid_: harness (informal), tool

**Source Kind**:
The on-disk transcript format for a provider (`claude_code_jsonl`, `codex_rollout_jsonl`, `opencode_storage`, etc.).
_Avoid_: file type, format

**Source ID**:
A stable provider-scoped identifier, shaped `provider:logical_session_id:digest`.
_Avoid_: session id

**Logical Session ID**:
The provider-agnostic session identifier normalized across harnesses.
_Avoid_: session id

**Provider Adapter**:
A module implementing the discover/parse/watch/health Protocol for one harness.
_Avoid_: parser, connector

**Transcript**:
The raw session log file (JSONL/SQLite) a provider emits, before parsing into Turns.
_Avoid_: log, session file

## Storage & Indexing

**Milvus Standalone**:
The primary vector store at `192.168.1.83:19530`; Milvus Lite is the local fallback.
_Avoid_: the database, vector db

**FTS Sidecar**:
The SQLite FTS5 keyword index maintained alongside the vectors for hybrid search.
_Avoid_: keyword index, sqlite

**Embedding Identity**:
The model fingerprint stored to guard against mixing vectors from different embedding models.
_Avoid_: model name

**Schema Drift**:
A mismatch between the live Milvus schema and expected fields; refuses startup unless migrated.
_Avoid_: schema mismatch

**Chunk Type**:
The classification of an indexed record (e.g. `turn`).
_Avoid_: record type

## Search & Retrieval

**RRF (Reciprocal Rank Fusion)**:
The merge algorithm combining vector and FTS result sets (k=60) into one ranking.
_Avoid_: fusion, merge

**Hybrid Search**:
Retrieval that combines vector similarity with FTS keyword matching via RRF.
_Avoid_: search

**Recency Boost**:
A post-ranking adjustment that favors more recent Turns in results.
_Avoid_: freshness

**Project Root**:
The `cwd`-derived project tag on a session; generic `/` sessions are untaggable.
_Avoid_: project, cwd

## Backfill & Operations

**Backfill**:
Catch-up indexing of historical transcripts, decoupled from live MCP activity.
_Avoid_: reindex, import

**Backfill Job**:
A queued, durable unit of backfill work scoped to a source or provider.
_Avoid_: task

**Embedding Budget**:
The shared throttle/cooldown governing embedding across all providers (200ms floor — MLX Metal SIGSEGVs below it).
_Avoid_: rate limit

**Backfill Mode**:
The catch-up scope selector: `recent` | `incremental` | `full`.

**Checkpoint**:
Backfill progress saved to `index_state.json` every 100 files; restart resumes from it.
_Avoid_: save state

## Relationships

- A **Provider** has exactly one **Source Kind** per transcript format and exactly one **Provider Adapter**.
- A **Transcript** is parsed into many **Turns**; each Turn carries a **Source ID** and **Logical Session ID**.
- **Hybrid Search** merges vector results from **Milvus Standalone** and keyword results from the **FTS Sidecar** via **RRF**, then applies **Recency Boost**.
- A **Backfill Job** runs under the shared **Embedding Budget** in a given **Backfill Mode**, recording progress to a **Checkpoint**.

## Flagged Ambiguities

- "session id" — ambiguous between the raw provider value and the normalized one. Use **Logical Session ID** for the normalized identifier and **Source ID** for the stable composite key.
- "Claude Desktop" — its work is searchable today under **Provider** `claude_code_cli` (via linked `cliSessionId` JSONL), not under a distinct `claude_desktop_cowork` provider. The latter source kind is probe-only until the parser spike lands.
