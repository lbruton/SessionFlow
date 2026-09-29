---
sketch: "SESF-38-fts-backfill-hardening"
phase: discovery
created: 2026-06-05
---

# SESF-38 — Discovery

_Research the existing system and prior art. **Don't propose solutions** — that's the next phase._

> **Method note:** findings came from a read-only Workflow fan-out (5 angles: adversarial verification, CGC structural, semantic/reuse, prior decisions, pymilvus-source external). All 7 seam claims from the plan brief were **confirmed against live source** with current `path:line`. CGC is indexed but its line numbers are **stale (~+543 offset)** and it under-reports cross-module callers (reported "0 callers" for `backfill_fts`/`get_stats`) — line numbers below are from live-file reads, not CGC.

## Existing Code

### The fix touches these (verified live)

| Path | Role | Notes |
|------|------|-------|
| `http_server.py:704-720` | `_fts_backfill` — the **only** automatic FTS Backfill trigger | One-shot: single `asyncio.create_task(_fts_backfill())` at `:720`; `sleep(1)` → `backfill_fts` in `run_in_executor` once → swallows all errors to a stderr WARN. No loop, no retry, no sentinel re-trigger. **Self-heal seam.** |
| `http_server.py:251-269` | `_backfill_drain_worker` — periodic loop (drains **provider** jobs only) | `asyncio.Event` + `await asyncio.wait_for(event.wait(), timeout=interval)`; interval `BACKFILL_DRAIN_INTERVAL` (`:75`) from env `SESSIONFLOW_BACKFILL_DRAIN_INTERVAL_SECONDS` default `30` (`:64-72`); scheduled `:700`. `_drain_backfill_once` (`:235-248`) calls `ProviderIngestionService.process_queued_jobs` — **never** `backfill_fts`. Candidate host for the self-heal re-trigger. |
| `http_server.py:228-232` | `_wake_backfill_drain` — sets the Event to run the drain loop immediately | Reusable to force an immediate FTS heal check if piggybacking on the drain loop. |
| `http_server.py:345-361` | `/health` handler (route `:803`) | Returns status/model/milvus/watchers/providers/backfill(**provider queue** via `_backfill_status_payload` `:321-330`)/embedding. **No FTS row count, lag, or sentinel state.** AC-6 insertion point. |
| `rag_engine.py:1653-1741` | `backfill_fts` — idempotent two-pass diff-then-hydrate | Streams Milvus `doc_id`s via `_query_batches`, diffs FTS, hydrates only-missing in 100-row chunks via `_fts.insert`, clears sentinel on success (`:1734-1736`), returns inserted count (0 cheaply when nothing missing). **No internal retry; no `MilvusException` catch** (the only try/finally closes the ephemeral FTS conn `:1740-1741`). Gated by `_ensure_collection` via `milvus_client` (`:1680`). `output_fields` (`:1672-1675`) + per-record dict (`:1692-1709`) must stay in lockstep with `add_turns`/`_row_to_result`. |
| `rag_engine.py:518-553` | `_ensure_collection` — Schema-Drift startup guard | On drift: auto-migrate branch (`:538-546`, opt-in `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1`) calls `migrate_schema`, else raises destructive-only `RuntimeError` (`:548-553`). **AC-3/AC-4 target.** Exact message: `"Milvus collection {COLLECTION_NAME!r} schema is out of date (drift={drift}). Run `python cleanup.py migrate-schema` to drop and recreate it (destructive), or set SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1 to migrate on startup."` |
| `rag_engine.py:472-483` | `detect_schema_drift` — NAME-only field diff | Uses `client.describe_collection` (cache-free — see External). AC-3's re-describe reuses this verbatim. Dtype/length diffing explicitly deferred (docstring `:454-461`). |
| `rag_engine.py:588-593` | `milvus_client_for_migration` — **drift-tolerant** client | Opens **without** `_ensure_collection`. Reuse candidate for letting backfill proceed under transient drift (approach decides). |
| `rag_engine.py:606-623` | `milvus_client` — the drift-**guarded** context manager | Calls `_ensure_collection` (`:611-619`); this is the gate `backfill_fts` currently inherits. |
| `rag_engine.py:501-515` | `migrate_schema` — `drop_collection` + `_create_collection` | DESTRUCTIVE ("all indexed turns are lost"). Stays the explicit escape hatch; only its framing changes. |
| `rag_engine.py:1480-1512` | `get_stats` — Milvus-only counts | Drains the collection via `_query_all` + `len` (no `num_entities`); returns `total_turns/sessions/branches/by_type/providers`; never touches FTS. CLI-only callers (`cleanup.py:113,120,166,214`), **not** `/health`. AC-6 adds FTS count + lag. |
| `rag_engine.py:1007-1017` | `search()` — **sole** sentinel reader | Stamps `r["_fts_warning"]="keyword index rebuilding, results may be vector-only"` when `fts_backfill_required()`. **Most complex function in the repo (CC 35)** — keep AC-1 changes minimal/additive; heal completion auto-clears this warning via the shared sentinel. |
| `rag_engine.py:350,400-431` | `_persistent_clients` — process-global `MilvusClient`s keyed by `db_path` | Long-lived (reused until a `has_collection()` probe fails). Relevant to the data-path stale-view window (External). |
| `fts_hybrid.py:33,36,41` | `FTS_BACKFILL_SENTINEL` + `fts_backfill_required()` + `clear_fts_backfill_sentinel()` | Sentinel = `~/.sessionflow/fts_backfill_required`. The ready-made trigger condition. |
| `fts_hybrid.py:142-187` | `_check_and_migrate` — sentinel **writer** + FTS-table DROP on schema change | Reached lazily from `connection()` (`:189-212`) on first connection per thread. Warning (`:170-174`): `"FTS schema changed for %s, dropping and rebuilding — keyword search will degrade to vector-only until FTS backfill completes"`. Table `"turns_fts"` (`rag_engine.py:351`); `_fts_schema` bookkeeping (`:149-154`). |
| `fts_hybrid.py:71,101,112,189,222` | `FTSIndex` threading.local model (SESF-13) + `_fts.insert` (`:222`) | Per-thread connection cache; `close_all()` is this-thread-only. **No row-count method exists** — AC-6 needs a new `SELECT count(*) FROM turns_fts` helper here. |
| `cleanup.py:184,202,205,414-416,464` | `cmd_migrate_schema` — the destructive path the error message points to | Reads `detect_schema_drift` then `migrate_schema`; subparser `migrate-schema`. Stays valid; AC-4 just reframes how it's recommended. |

### Reusable primitives (prefer these over new code)

| Path | Pattern | Use for |
|------|---------|---------|
| `embedding_control.py:80,175` (test: `tests/test_embedding_control.py:62-78`) | **Log-once boolean latch** — `EmbeddingBudget` emits exactly one WARN across repeated denials | **AC-5** repeated-failure collapse |
| `rag_engine.py:1653-1741` | Idempotent FTS rebuild from Milvus rows (`backfill_fts` itself) | **AC-1/AC-2** — re-invoke to heal; don't write new streaming code |
| `http_server.py:251-269` | `asyncio.Event`+timeout periodic loop | **AC-1** self-heal placement |
| `tests/test_schema_drift.py:14-52` | Mocks `MilvusClient.describe_collection`, asserts on `detect_schema_drift` | **AC-3/AC-4** drift tests |

### Test seams to preserve

- `tests/test_provider_ingestion.py:136-182` — `test_backfill_fts_rehydrates_issue_ids` monkeypatches `rag_engine.milvus_client`, `rag_engine._query_batches`, `rag_engine._fts.{connection,insert,close_ephemeral}`; asserts `inserted==1` and `issue_ids` round-trips. Any refactor of `backfill_fts` internals must keep these seams or update the test.
- `tests/conftest.py:449` — `stub_rag_engine` stubs `backfill_fts=lambda *a,**k: 0`. New SESF-38 behavior tests must exercise the real module, not this stub.

## Prior Decisions

- **2026-06-04 — SESF-38 filed from a Devops triage session** (sessionflow session `65772997-44c4-49fc-b248-f9734580e325`, `2026-06-04T18:06:11`). Framed deliberately as a **resilience gap**; data-loss and stats-desync theories explicitly ruled out (25,966 Turns intact across the restart; `get_session_stats` correctly project-scoped by `X-Project-Root`). Stated correct recovery: **RESTART or POST the backfill control endpoint — NEVER `cleanup.py migrate-schema`** (which drops the collection). _Query: `search_all_sessions("SESF-38 FTS backfill self-heal schema drift incident", project_root="*", date_from≈2026-06-01)`._
- **SESF-11 schema-drift guard is the root cause** — `_ensure_collection` (`rag_engine.py:518-553`) refuses startup on drift; `backfill_fts` inherits the hard-fail because it opens via the guarded `milvus_client` (`:1680`). The drift-tolerant `milvus_client_for_migration` (`:588`) already exists as a bypass.
- mem0 was queried (`"SessionFlow FTS backfill schema drift"`, `"SESF-11 schema drift migrate-schema"`, `"SESF-13 FTS thread affinity"`, `"SESF-25 issue_ids backfill_fts"`, `"SESF-38 FTS backfill"`) and surfaced **no curated decisions beyond** the codebase + sessionflow record above — this surface is effectively greenfield for design choices.

## External References

- **pymilvus 2.6.12 — `describe_collection` is cache-free** (`venv/.../pymilvus/milvus_client/milvus_client.py:873-888`; `client/grpc_handler.py:595-612`). It issues a fresh `DescribeCollection` RPC every call and never reads/writes the schema cache. **Consequence for AC-3:** re-calling `describe_collection` on the **same persistent client** re-reads server truth — **no new client/connection required.** `detect_schema_drift` already uses this path (`rag_engine.py:472-483`).
- **The schema cache is data-path only** — `GlobalCache.schema` (`client/cache.py:49-139`) is a process-global `ClassVar` LRU(4096) keyed by `(endpoint, db, collection)`, consumed only by insert/upsert/search/hybrid_search via `_get_schema` (`grpc_handler.py:822-848`). It returns cached entries with **no client-side timestamp re-validation**; the server rejects stale schema via a `schema_timestamp` sent on the request.
- **Latent cache-invalidation gap (possible follow-up, see Constraints)** — `drop_collection` invalidates the cache (`grpc_handler.py:423`) but `add_collection_field` does **not** (`:442-455`). After a field add (e.g. SESF-25 `issue_ids`), a long-lived persistent client can hold a stale, field-less schema on the **insert/search** path → server-side `code=65535 "field not exist"` until LRU eviction. A fresh `MilvusClient` does **not** clear it (the cache is a class-level singleton). Refs: [pymilvus v2.5.5 release note](https://github.com/milvus-io/pymilvus/releases/tag/v2.5.5), [milvus #39093](https://github.com/milvus-io/milvus/issues/39093) / [PR #39096](https://github.com/milvus-io/milvus/pull/39096).
- **Incident error codes** — the issue log shows `code=65535 field … not exist` (data-path schema mismatch) and `code=1100 Error output fields: invalid parameter` (an `output_fields` request naming a field the live schema lacked), alongside `_ensure_collection`'s `drift=[missing:…]` `RuntimeError`s. Consistent with the collection **genuinely lacking fields for a window mid-migration**, not a describe-path client cache.
- **SQLite FTS5 `count(*)`** — `SELECT count(*) FROM turns_fts` is fully supported (`count(*)` needs no column values, so contentless/external-content restrictions don't apply); avoid `count(<column>)` on contentless tables. [sqlite.org/fts5.html](https://sqlite.org/fts5.html). Confirms AC-6's FTS row count is a plain `count(*)`.

## Constraints

_The implementation must respect these. Several are **design decisions deferred to approach** — surfaced here, not decided._

- **SESF-13 FTS thread affinity** — `FTSIndex` uses per-thread `threading.local` connections; `backfill_fts` already opens an ephemeral conn and `close_ephemeral`s it. The self-heal trigger must open/close its FTS connection on its own executor thread (don't share across threads).
- **Concurrency (approach decides)** — startup `_fts_backfill` and a periodic self-heal could overlap; `_backfill_drain_lock` guards **provider** jobs only. Approach must either add a dedicated guard (asyncio lock / "already running" flag) or rely on `backfill_fts`'s idempotent diff (double-run is wasteful, not corrupting).
- **Loop placement (approach decides)** — fold the self-heal into `_backfill_drain_worker` (reuses Event/interval/wake, but serializes behind provider backfill) vs. a separate FTS-heal worker.
- **Client choice under drift (approach decides)** — `backfill_fts` could open via the drift-tolerant `milvus_client_for_migration` to proceed under transient drift, **but** if fields are genuinely absent the `output_fields` query still `65535`s — so cadence-retry (AC-2) is the durable mechanism; the drift-tolerant client is complementary, not sufficient alone.
- **`search()` complexity (CC 35)** — the sole sentinel reader; keep AC-1-related edits there minimal/additive.
- **AC-4 must not break the recovery instruction** — `cleanup.py migrate-schema` stays a valid (destructive) command the message references; only its framing/ordering changes (non-destructive-first).
- **`detect_schema_drift` is NAME-only** — extending to dtype/length is explicitly out of scope (requirements Non-Goals).
- **Embedding budget coexistence** — self-heal runs in `run_in_executor` today; keep it mindful of the shared embedding executor and the 200ms MLX floor (SESF-6) — don't introduce a tight Milvus-poll that starves embedding.
- **Possible follow-up issue (out of scope here):** invalidate `GlobalCache.schema` after schema-mutating ops to close the data-path stale-view vector. Distinct from SESF-38's FTS self-heal; flag for a separate ticket rather than expanding this sketch (per requirements Non-Goals).

## Open Questions

_None block approach._ Requirements pinned the observable behavior (AC-1…AC-6); the residual choices (loop placement, drift-tolerant client vs catch-and-retry, concurrency guard) are **approach's** to decide and are captured under Constraints. The pymilvus deep-dive resolved the load-bearing AC-3 question (re-describe is cache-free; no new client needed).

## Discovery Summary

The work lands in three well-isolated seams — `backfill_fts`/its trigger (`http_server` self-heal), `_ensure_collection` (drift re-verify + honest message), and `/health`+`get_stats`+`fts_hybrid` (FTS-lag stat) — and leans almost entirely on existing primitives (the idempotent `backfill_fts`, the `asyncio.Event` drain loop, the `EmbeddingBudget` log-once latch, the drift-tolerant `milvus_client_for_migration`, and the cache-free `describe_collection`). What's already easy: AC-3 (re-describe is a no-op-mechanism change) and AC-6 (a plain `count(*)`). What needs care: the concurrency/loop-placement choice for the self-heal trigger and keeping the additive `search()` sentinel-reader edits small. A separate latent pymilvus data-path cache gap was found and flagged as a follow-up, not in scope.

---

> **Phase complete?** Existing code mapped. Prior decisions surfaced. Open questions resolved. Then advance: `/sketch-approach SESF-38`.
