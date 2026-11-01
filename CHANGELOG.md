# Changelog

All notable changes to SessionFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions 0.1.0 to 0.8.0 were reconstructed from
the git history in SESF-87: each one summarizes a phase of work and carries the date of its last merge. They
have no git tags or GitHub Releases. For changes before the fork, see [UPSTREAM-README.md](UPSTREAM-README.md).

## [Unreleased]

### Added

- Temporal queries. A hybrid search whose query asks for the latest state ("latest", "most recent", "newest", "recent", "recently", "last time", "current", "currently", "now", "today", "yesterday", "this week", matched as whole words) ranks with `SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT` (default 0.7) instead of `SESSIONFLOW_RECENCY_WEIGHT`. The reranker fuses its order with that hybrid order by RRF for both `jina` and `qwen3`, so the recency signal survives reranking. `/health` reports the weight under `ranking`, and each result of such a query prints `Temporal: yes`. Relevance and recency sorts are unchanged. In-process on the eval set with `jina`, temporal MRR rose from 0.738 to 0.757 and nDCG@10 from 0.477 to 0.528; other groups do not change (SESF-101).

## [0.14.1] - 2026-11-01

### Fixed

- `add_memory` with a `topic_key` that matches a superseded record and different text sets the record's status from the new write, so the replacement shows in default reads. Identical text stays deduplicated and superseded (SESF-103).

## [0.14.0] - 2026-11-01

### Added

- MLX reranker for `search_all_sessions` and `search_session` calls with a query. `SESSIONFLOW_RERANKER` selects `qwen3` (default; Qwen3-Reranker-0.6B, Apache 2.0, fused with the hybrid rank by RRF), `jina` (jina-reranker-v3.5-mlx, CC BY-NC 4.0, sorts by its score) or `off`. `setup.sh` downloads the selected reranker. `SESSIONFLOW_RERANK_TOP` (default 10) and `SESSIONFLOW_RERANK_MAX_TOKENS` (default 512) bound the work. `./download-model.sh jina` or `./download-model.sh qwen3` downloads the pinned snapshot, and the server loads it from the cache on the MLX thread after startup; until it is ready, or when it fails, search skips reranking. With fewer results requested than `SESSIONFLOW_RERANK_TOP`, retrieval still fetches enough candidates for the reranker. The README has a Reranking section with the eval MRR of each model and how to switch or turn it off. `/health` reports the settings under `ranking` and the load state as `ranking.reranker_state`, and each reranked result prints `Rerank: #k`. Recency-sorted searches and query-less listings are not reranked. In-process on the eval set, MRR rose from 0.720 to 0.808 with Qwen3 and 0.870 with Jina (SESF-100).

## [0.13.2] - 2026-10-31

### Fixed

- Reverted the per-turn rank fusion of 0.13.1. Adding a turn's best keyword rank and best vector rank favored long turns, whose many passages place a row in both channels, over short turns that match one channel at rank 1; overall eval MRR fell from 0.790 to 0.738. Search again fuses rows by doc_id and collapses passages after fusion (SESF-99).
- `search_all_sessions` and `search_session` with `n` above 42 failed on Milvus Standalone, because HNSW rejects a search whose limit exceeds `ef`. The `ef` search parameter is at least the vector search limit (SESF-99).

## [0.13.1] - 2026-10-31

### Fixed

- Search counts one vector rank and one keyword rank per turn. Passages were collapsed only after RRF fusion, so a turn found by keyword on its full row and by vector on one of its passages never added the two ranks, and sibling passages filled the candidate pool. Each channel now keeps the best-ranked row of each turn before fusion and reads twice the candidate count before the cut. The HNSW `ef` search parameter rises with that count, because Milvus rejects a search whose limit exceeds `ef` (SESF-99).

## [0.13.0] - 2026-10-31

### Added

- `cleanup.py passages {plan|run|drive|status|verify}` backfills turns indexed before passages. The uncut phase re-parses, from disk with no cursor, each Claude, Cowork legacy and Codex session that holds a turn cut at the old 8,000-character limit, and re-adds those turns whole with their passages. The passages phase then adds passages to every long turn still without them, from the stored text. Sessions gone from disk or unreadable keep their cut row, which gets passages too, and so does a re-parsed turn that boilerplate stripping drops or that does not hold the whole stored cut text. Passages embed in budget-sized batches. A failed FTS write of a passage or an uncut turn stays in the checkpoint and is retried, without re-embedding, before the run is done. `run` and `drive` refuse without `--yes`, and the checkpoint is `~/.sessionflow/passages_state.json` (SESF-69).
- `rag_engine.replace_passages` replaces the stored passages of one turn in Milvus and FTS, and `rag_engine.write_passage_fts` rewrites only their FTS rows (SESF-69).

## [0.12.0] - 2026-10-31

### Added

- A turn longer than 450 tokens also gets `turn_passage` rows: windows of 450 tokens with 64 tokens of overlap, so each passage fits inside the 512-token sliding window. Passages embed with the project name and the user question in EmbeddingGemma's title slot, and they carry the parent's session, turn index and issue IDs. A deferred embedding batch drops the whole turn, so no parent lands without its passages (SESF-69).

### Changed

- Turns are stored whole up to the 65,535-byte Milvus `document` cap instead of being cut at 8,000 characters. Search output still prints at most 8,000 characters of a turn, and `get_turns` returns all of it (SESF-69).
- Search keeps one row per turn: the best of the parent and its passages. A passage hit prints its part number (SESF-69).
- `cleanup.py reembed`, `cleanup.py rewrite` and `cleanup.py sanitize` re-embed passages with the title rebuilt from their parent turn (`rag_engine.document_titles`); `sanitize` redacts that title too (SESF-69).
- Deleting a turn by doc_id also deletes its passages from Milvus and FTS. The recency listing, `get_turns`, the issue timeline, session listings and the stats turn count skip passages (SESF-69).

## [0.11.0] - 2026-10-31

### Changed

- `cleanup.py dedupe-codex` deletes an old Codex row of a session on disk only when a new task row holds its whole reply. Import turns, text past the 8,000-character task cut and text from an older rollout layout stay, and the command reports them as uncovered rows kept (SESF-98).
- `cleanup.py prune-codex` runs `dedupe-codex` instead of deleting every row the parse does not yield (SESF-98).

## [0.10.0] - 2026-10-30

### Added

- `cleanup.py dedupe-codex` deletes Codex rows of the old line-number keys. A session whose rollout is on disk keeps only the rows its parse yields, and only once the backfill has indexed every parsed turn. A session whose rollout is gone keeps one row per distinct turn, and one whose rollout fails to read is left alone. It is a dry run by default, `--apply` refuses without `--yes`, and the deleted doc_ids go to `~/.sessionflow/audit/dedupe-codex-<run>.jsonl` (SESF-98).

### Changed

- Codex sessions are indexed as one row per task: the user message and every reply up to `task_complete`. `turn_index` is the task's ordinal in the rollout instead of the line number, so a rollout rewrite that shifts lines keeps each doc_id. Codex turns replace their earlier row as a task grows, and a task is cut at 8,000 characters with `[truncated]`, like a Claude turn (SESF-98).
- Codex rollouts compressed to `.jsonl.zst` are read, through the new `zstandard` dependency. A plain copy next to its compressed twin wins (SESF-98).

## [0.9.2] - 2026-10-30

### Fixed

- Transcript keyword search builds its FTS5 query from the quoted terms joined with OR, so natural-language questions and queries with an issue ID such as `SESF-57 padding` get keyword hits in hybrid search. Identifiers stay one phrase, stopwords are dropped, and memory search uses the same builder (SESF-93).

## [0.9.1] - 2026-10-30

### Changed

- The default hybrid recency weight (`SESSIONFLOW_RECENCY_WEIGHT` unset) is 0.15 instead of 0.3. The retrieval eval sweep scored 0.15 best overall (MRR 0.654 against 0.630) (SESF-97).

## [0.9.0] - 2026-10-30

### Added

- `eval_retrieval.py` scores a labelled JSONL query set against the running server: recall@5, MRR and nDCG@10 per query group, plus a current-first rate for superseded decisions and the top hit score for no-answer queries. `run`, `sweep` (restarts the LaunchAgent once per env value; needs `--yes`), `freshness` (a canary memory, deleted after the probe), `usage` and `nightly` subcommands (SESF-82).
- `SESSIONFLOW_SEARCH_LOG=1` appends each search, `get_memory` and `get_turns` call to `~/.sessionflow/search-log.jsonl` with redacted queries and result IDs in rank order. It is off by default (SESF-82).
- `/health` reports the hybrid ranking settings under `ranking` (SESF-82).
- `sessionflow-server.sh install-eval-agent`, `uninstall-eval-agent` and `eval-agent-status` manage a nightly LaunchAgent that runs `eval_retrieval.py nightly` (SESF-82).

## [0.8.1] - 2026-10-30

### Fixed

- `list_memories`, `search_memories` and `get_memory` print a `Source: session <id> (<provider>)` line for a memory that names the session that wrote it (SESF-90).

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
