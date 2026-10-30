# Changelog

All notable changes to SessionFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions 0.1.0 to 0.8.0 were reconstructed from
the git history in SESF-87: each one summarizes a phase of work and carries the date of its last merge. They
have no git tags or GitHub Releases. For changes before the fork, see [UPSTREAM-README.md](UPSTREAM-README.md).

## [Unreleased]

## [0.8.0] - 2026-10-30

Project scope and index cleanup.

### Added

- `version.py` holds the version. `/health` and the MCP `serverInfo` report it, and this changelog is
  restored (SESF-87).
- Project scope resolves `project_root` to indexed roots: a path covers its subfolders and worktrees, a bare
  name matches a root by folder name, and search and stats output opens with a `Scope:` line (SESF-64).
- The server takes the current project from the client's MCP roots when a call names none. MCP sessions are
  stateful and streamed so a tool call can send `roots/list` (SESF-86).
- Harness boilerplate (task notifications, system reminders, compaction summaries, command output) is
  stripped from every turn at ingest (SESF-66).
- `cleanup.py rewrite` strips and re-embeds stored rows in place, with backups, a checkpoint and `verify`
  (SESF-62, SESF-66). `cleanup.py dedupe-turns` repairs duplicate partial turns (SESF-74).

### Changed

- `search_all_sessions`, `search_session` and `get_session_stats` refuse a call that names no project. Pass
  the working directory or `'*'` (SESF-85).
- EmbeddingGemma applies its 512-token sliding window on local layers, so long texts embed like the
  reference model (SESF-62).

### Fixed

- The headers helper sends the project root instead of `~/.claude`, and sends no project from Desktop spare
  folders (SESF-83, SESF-84).
- Codex Desktop external-import turns are skipped, because their original Claude Code transcripts are
  already indexed (SESF-66).
- The open last turn of a transcript is rebuilt with its reply instead of staying indexed as a user-only
  turn (SESF-74).

## [0.7.0] - 2026-10-28

Memory store.

### Added

- Memories backed by Postgres, with version history, topic-key replace, supersession and tombstone deletes.
  A Milvus `memories` collection and a local `memories_fts.db` are rebuilt from Postgres by a sync loop
  (DEVS-145).
- Seven MCP memory tools: `add_memory`, `search_memories`, `list_memories`, `get_memory`, `update_memory`,
  `delete_memory` and `get_memory_context`. `search_all_sessions` also takes `format: "json"` (DEVS-145).
- `memory_cli.py` with `rebuild`, `reindex`, `audit`, `import` and `restore-check`, and a session-start hook
  that prints the newest session summaries for Claude Code and Codex (DEVS-145).
- CI runs ruff, pytest against Postgres 17, and Vale (DEVS-147).
- `cleanup.py prune-codex` deletes orphan Codex turns (SESF-80).

### Fixed

- The memory sync loop is bounded by a timeout and reports a stall in `/health` (SESF-79).
- Codex turns carry the task start time and the session branch, and resumed parses no longer create orphan
  turns (SESF-80).

## [0.6.0] - 2026-10-22

Index quality and Desktop reliability.

### Added

- `cleanup.py reembed` rewrites stored vectors in place, chunked under the MLX crash limit, with float16
  backups (SESF-57).
- The `claude_cowork_legacy` provider ingests legacy Cowork agent-mode transcripts (SESF-60).
- `SESSIONFLOW_PROJECT_PATH_MAP` remaps moved project folders at ingest, and `cleanup.py retag-project`
  fixes indexed rows (SESF-73).
- Search results print the signals that ranked each row, and identifier queries weight keyword ranks
  (SESF-77).

### Removed

- The Claude Desktop Cowork probe. Desktop Code-tab sessions are indexed as `claude_code_cli` (SESF-59).

### Fixed

- Batched EmbeddingGemma documents mask their padding, so a batched vector matches the single one (SESF-57).
- The server binds port 7102 before the model and Milvus finish loading, and the Desktop MCP bridge runs
  from the internal disk and waits for the port (SESF-70, SESF-71).
- Every MLX evaluation runs on the `mlx-embed` thread, which stops a SIGSEGV when a search runs during
  backfill (SESF-72).
- The relevance score is no longer inverted, half-written transcript lines are not skipped, and index state
  is saved atomically (SESF-55, SESF-56, SESF-58).
- Query-less listings print "Listing" instead of "Relevance: 0.00" (SESF-68).

## [0.5.0] - 2026-06-14

Secret safety.

### Added

- Secret redaction at ingest covers the embedding, the Milvus `document` and the FTS `content`. It runs in
  `report` or `enforce` mode, with an operator allowlist and per-rule counts in `get_stats` (SESF-41).
- `cleanup.py sanitize` and the `sanitize_index` MCP tool remove secrets that are already indexed. A dry run
  is the default, and `apply` refuses without confirmation (SESF-42).

### Security

- Redaction is irreversible and is not a substitute for rotation. Rotate any key that was ever indexed.

### Fixed

- The recency listing streams through `query_iterator` instead of hitting the Milvus result limit (SESF-36).
- Timeline pagination orders by parsed UTC timestamps (SESF-43).
- The Tier-2 auth-keyword rule rejects non-literal assignment values (SESF-44).

## [0.4.0] - 2026-06-07

Issue IDs and timeline.

### Added

- Every turn is tagged with the issue IDs it mentions, in a new `issue_ids` Milvus field and FTS column
  (SESF-25).
- An `issue_id` filter on `search_all_sessions` and `search_session` (SESF-25).
- The `get_issue_timeline` MCP tool and `GET /timeline` give a cross-harness, oldest-first feed of the
  turns that reference an issue (SESF-25, SESF-26).

### Fixed

- The hybrid FTS path applies the `issue_id` filter (SESF-32).
- Filter values use C-style escaping, and timeline pagination is memory-bounded (SESF-33, SESF-34).
- FTS backfill heals after a transient schema drift, and the schema cache is invalidated on change
  (SESF-38, SESF-40).

## [0.3.0] - 2026-05-31

Ranking and recall.

### Added

- Hybrid recency-aware ranking and a `sort_by` parameter (`hybrid`, `relevance`, `recency`) on the search
  tools (SESF-24).
- `search_all_sessions` without a `query` lists the newest turns first (SESF-16).
- Antigravity Desktop turns take their `project_root` from the summaries metadata (SESF-17, SESF-29).

## [0.2.0] - 2026-05-28

Multi-harness ingestion.

### Added

- Provider adapters ingest Codex, OpenCode and Antigravity sessions next to Claude Code, sharing one
  embedding budget (SESF-6, SESF-19).
- `cleanup.py backfill` pauses, resumes and enqueues provider backfill, and its queue survives a restart
  (SESF-6).
- A home-directory launcher for hooks, an optional login LaunchAgent and an hourly backfill LaunchAgent
  (SESF-7, SESF-20).

### Fixed

- Schema drift is detected at startup, and a destructive migration is opt-in (SESF-11).
- The backfill turn budget defaults to 100,000, FTS connections are per thread, and OpenCode timestamps are
  ISO-8601 (SESF-12, SESF-13, SESF-14).
- A running server drains backfill jobs enqueued from the CLI (SESF-15).

## [0.1.0] - 2026-05-20

Fork baseline.

### Added

- Fork from `mwgreen/claude-code-session-rag` at `637e6f4`.
- `date_from` and `date_to` filters on `search_all_sessions`, applied to Milvus and FTS (SR-3).
- Milvus Standalone through an environment variable, with Milvus Lite as the fallback.
- `detect_project_root()` and project roots derived from Claude Code slug names (SR-2).
- Backfill checkpoints every 100 files and pauses after Milvus connection errors (SR-1).

### Fixed

- Backfill no longer dies after a Milvus Lite gRPC `GOAWAY` (SR-1).
- FTS backfill runs in the background, so the server binds without waiting for it (SR-1).
- MLX embedding calls run on one dedicated thread, and the watchdog uses a heartbeat file (SESF-3, SESF-8).
- A new collection is loaded after it is created (SESF-2).
- Milvus drains through `query_iterator` and FTS inserts are streamed (SESF-4, SESF-5).
