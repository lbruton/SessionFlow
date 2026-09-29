---
sketch: "SESF-38-fts-backfill-hardening"
phase: approach
created: 2026-06-05
---

# SESF-38 — Approach

_How we'll build it. **Don't write code or tests** — the tasks phase produces the work plan._

> Synthesized from a 3-architect / 3-judge design panel (reuse-minimal · isolation-robust · operational-correctness lenses). Base = operational-correctness (2/3 judges); grafted refinements noted inline. The panel's most valuable output was **7 hazards all three drafts missed** — folded into Decisions and Risk Notes below. Reconciled against Codex review 2026-06-06 (4 findings accepted — see Review Archive).

## High-Level Architecture

**Seam 1 — self-heal trigger + retry (AC-1, AC-2, AC-5).** Replace the one-shot `_fts_backfill` (`http_server.py:704-720`) with a recurring `_fts_heal_worker` asyncio task. It reuses the proven `asyncio.Event` + `await asyncio.wait_for(event.wait(), timeout=…)` shape from `_backfill_drain_worker`, but is its **own** task — **not** folded into the provider drain loop. Folding was rejected by the risk judge: that loop's single `except Exception` is scoped to the provider call, so an escaped FTS-heal error could wedge **provider ingestion** — turning a transient FTS drift into a broader outage, the exact stale→stuck regression SESF-38 exists to prevent. The worker is the **sole dispatcher** of `backfill_fts`, so its serial `await run_in_executor(...)` per tick eliminates the startup-vs-heal race intrinsically (no lock needed for that race). **The worker's first tick runs `backfill_fts` UNCONDITIONALLY** — preserving today's startup catch-up of any pre-sentinel / legacy missing FTS rows that the idempotent backfill already corrects on every boot — and **subsequent ticks run when `fts_backfill_required()` is set** (cheap: a sentinel `stat()` when clear). Each run executes on a **dedicated single-thread executor** (so a large rebuild can't starve the shared MLX embed pool, SESF-6/8); the result is classified and the next tick scheduled with bounded backoff. The loop only exits on shutdown cancellation, satisfying AC-2's "never abandon until restart."

**Seam 2 — drift re-verify + honest guidance (AC-3, AC-4).** Centralized in `_ensure_collection` (`rag_engine.py:518-553`). On the first observed drift, re-call `detect_schema_drift` once — which re-runs `client.describe_collection` (proven cache-free in pymilvus 2.6 on the same persistent client; **no new client/connection**) — **before EITHER the destructive `SESSIONFLOW_AUTO_MIGRATE_SCHEMA` auto-migrate branch (`rag_engine.py:538-546`, which drops+recreates *before* the raise today) OR the raised error**; if the re-evaluation clears, proceed (non-destructive recovery from a stale view) without touching the collection. Gating the auto-migrate branch is essential: otherwise an `AUTO_MIGRATE=1` operator loses an intact collection to a stale-view false positive — directly against US-2/AC-4. Because the recurring heal re-opens via the guarded `milvus_client` every tick, this single fix serves **both** startup and every heal tick. If drift persists, rewrite the `RuntimeError` to **lead with the non-destructive restart**, mark **both** `cleanup.py migrate-schema` **and** `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1` as destructive ("all Turns lost"), and keep the literal `cleanup.py migrate-schema` command string intact. **Key nuance (panel):** `backfill_fts` opens the guarded client at `with milvus_client(...)` (`:1680`), so under genuine drift it raises this `RuntimeError` at *open time* — before the `output_fields` query. The worker must treat **both** that `RuntimeError` and a query-time `MilvusException` as retryable transient failures.

**Seam 3 — observability that survives the incident (AC-6).** One shared `rag_engine.fts_lag_status(db_path, project_root=None)` helper feeds **both** `/health` and `get_stats` so they can't diverge — returning **only static lag data**: Milvus Turn count, FTS-Sidecar row count (new `FTSIndex.count_rows`), their delta (FTS lag), and the `fts_backfill_required` sentinel state. **The worker's standing failure (`consecutive_failures`/`last_error`) is NOT part of this helper** — it is added **only** in the `/health` payload by the worker (`http_server`), because MCP/CLI callers reach `get_stats` (`tools.py:500-505`, `cleanup.py:162-166`) with no HTTP worker in scope. Surfacing the standing failure on `/health` matters because AC-5 deliberately suppresses repeat logs, so `/health` becomes the place to see a stuck heal. **Scoping (Codex):** `get_stats` filters Milvus by `project_root` (`rag_engine.py:1480-1491`; `tools.py` passes the current root), so the lag must be apples-to-apples — `FTSIndex.count_rows(conn, project_root=None)` takes the **same** optional filter (FTS rows store `project_root`, `rag_engine.py:353`) and `fts_lag_status` threads it through both counts. `get_stats(project_root=X)` → project-scoped lag; `/health` passes `None` → global, server-wide. Two robustness grafts: the Milvus count reads through the **drift-tolerant** `milvus_client_for_migration` (the guarded client raises under the very drift we're reporting), applying the same `project_root` filter; and the `/health` fts block is wrapped in **try/except degradation** (mirroring `_embedding_status_payload` `:333-342`) behind a **short-TTL cache** (mirroring `_provider_health_cache` `:295-318`) so frequent probes don't each pay a full Milvus scan. Finally, the AC-6 counts do double duty: `backfill_fts`'s sentinel-clear is **gated on post-insert `fts_row_count >= milvus_turn_count`** so a partial hydrate can't falsely declare "consistent" (AC-1 correctness).

## Key Decisions

| # | Decision | Rationale | Tradeoff |
|---|----------|-----------|----------|
| **D-1** (F1) | Dedicated `_fts_heal_worker` task, **not** folded into `_backfill_drain_worker` | Isolates the FTS failure domain — an escaped heal error can't wedge provider ingestion (risk judge demoted the fold to 5/10 for this) | More lifecycle surface (schedule + cancel/await in `lifespan`, mirroring the drain task) than reusing the existing loop |
| **D-2** (F2) | Keep the drift-**guarded** `milvus_client` in `backfill_fts`; cadence-retry is the durable mechanism | A drift-tolerant client can't conjure absent fields — the query still `65535`s mid-migration; retry once the schema settles is the only correct durable path (unanimous across drafts + judges) | No heal progress during a genuine-absence window — correct, but FTS stays vector-only (surfaced via `_fts_warning`) until drift settles |
| **D-3** (F3) | Worker is the **sole** `backfill_fts` dispatcher (replaces the one-shot); **first tick unconditional**, sentinel-gated thereafter; defensive non-blocking `threading.Lock` in `backfill_fts` | A single serial coroutine never overlaps its own `run_in_executor`, eliminating the startup-vs-heal race without a load-bearing lock; the unconditional first tick preserves today's boot catch-up (Codex); the lock is cross-thread-correct for any future caller | Removes the startup one-shot (the worker's first tick now owns that behavior) |
| **D-4** | `backfill_fts` **catches** the open-time `RuntimeError` (drift guard) **and** query-time `MilvusException`, re-raising a typed transient signal; sentinel-clear relocated to the gated success path | Today errors are swallowed to stderr (`:717-718`) with no internal catch, so a retry worker is blind to failure vs success — the load-bearing fix the correctness+risk judges both flagged | `backfill_fts` gains minimal error-classification responsibility; its monkeypatch test seam must be updated |
| **D-5** | **Gate sentinel-clear on `fts_row_count >= milvus_turn_count`** (reusing AC-6 counts), not on any non-exception return | Hazard all 3 missed: a partial hydrate clears the sentinel while FTS still lags, violating AC-1 "clear once consistent" and producing a contradictory `/health` (lag>0, sentinel cleared) | One extra count comparison at end of heal; ties AC-1 to the AC-6 machinery (acceptable coupling) |
| **D-6** | AC-6 Milvus count via drift-**tolerant** `milvus_client_for_migration` (same `project_root` filter as `get_stats`); `/health` fts block try/except-degraded + short-TTL cached | The guarded client raises under persistent drift — unguarded, the observability breaks exactly during the incident it exists to diagnose; project-scoped filtering keeps lag apples-to-apples (Codex) | Slightly looser invariant (count taken without the startup guard); a few seconds of cache staleness on `/health` |
| **D-7** | Bounded backoff + log-once state in a small `FtsHealState` **owned by the worker** (`http_server.py`), surfaced **only via `/health`** (never `get_stats`); AC-5 signature = `type(exc).__name__` + **stable message prefix** | Keeps worker state with the worker and out of the worker-less `get_stats` path (Codex); stable-prefix signature stops volatile MilvusException request-ids/timestamps from churning the latch and re-firing the warning every tick | Signature normalization needs a small prefix-extraction rule; two genuinely-distinct faults with identical prefixes collapse to one WARN (the intended "same error = one line") |
| **D-8** | Dedicated **single-thread executor** for the heal `backfill_fts` call (separate from the MLX `_embed_executor`) | A large rebuild on the shared default/embed executor would queue behind / starve MLX embedding (the 200ms floor can't help a queued task) | One more executor object to create/shutdown in `lifespan` |
| **D-9** | AC-3 re-describe (gating **both** the auto-migrate and raise branches) + AC-4 message both live in `_ensure_collection` | Single chokepoint serves startup AND every heal tick; keeps `detect_schema_drift` NAME-only (dtype/length stays out of scope) | A genuine dtype/length drift still passes undetected (explicitly out of scope) |

## File Map

### New
- _none_ — all changes land in existing files. (New symbols — `_fts_heal_worker`, `FtsHealState`, `FtsBackfillTransientError`, `fts_lag_status`, `FTSIndex.count_rows` — are added to existing modules.)

### Modified
- `http_server.py` — replace one-shot `_fts_backfill` (`:704-720`) with the recurring `_fts_heal_worker` (**unconditional first tick**, sentinel-gated thereafter, bounded backoff via `FtsHealState`, dedicated single-thread executor, sole dispatcher); schedule it + cancel/await in `lifespan` mirroring `backfill_drain_task` (`:700`/`:752-756`); add a backoff-cap const near `:64-77` (reuse the drain interval as the base, one new cap const); add `_fts_lag_payload()` that merges `rag_engine.fts_lag_status(...)` (static lag) **plus** the worker's `FtsHealState` (`consecutive_failures`/`last_error`) — try/except-degraded, TTL-cached — into the `/health` JSON (`:345-361`) as a new `"fts"` key.
- `rag_engine.py` — `_ensure_collection` (`:518-553`): re-describe **before both** the auto-migrate branch (`:538-546`) **and** the raise (AC-3) + rewrite the `RuntimeError` message (AC-4). `backfill_fts` (`:1653-1741`): catch open-time `RuntimeError` + query-time `MilvusException`, re-raise typed `FtsBackfillTransientError`; gate `clear_fts_backfill_sentinel` on `fts_row_count >= milvus_turn_count`; add defensive `threading.Lock`. Add `fts_lag_status(db_path, project_root=None)` returning **static lag only** (Milvus count via `milvus_client_for_migration` with the `project_root` filter; FTS count via the new helper) and extend `get_stats` (`:1480-1512`) to thread its existing `project_root` into the helper and merge the static lag fields into its return. Add the `FtsBackfillTransientError` type.
- `fts_hybrid.py` — add `FTSIndex.count_rows(conn, project_root=None) -> int` running `SELECT count(*) FROM turns_fts` with an optional `project_root = ?` filter (FTS rows store `project_root`; non-contentless table → exact). Sentinel helpers (`:36-48`) reused as-is.
- `tests/` — extend `test_schema_drift.py` (AC-3 gating both branches / AC-4), `test_provider_ingestion.py` (preserve/extend the `backfill_fts` monkeypatch seam for D-4/D-5), `test_embedding_control.py`-style log-once assertions (AC-5); new tests for the worker backoff state + unconditional first tick, `fts_lag_status` (incl. project-scoped vs global), and `/health`/`get_stats` shape. (Tasks phase enumerates.)

### Deleted
- _none_ (the one-shot `_fts_backfill` coroutine is replaced in place, not a file deletion).

## Data / Schema Changes

None — no Milvus schema, FTS-table schema, or persisted-data changes. `get_stats`'s return dict gains additive keys (`fts_row_count`, `fts_lag`, `fts_backfill_required`), **project-scoped to match `get_stats`' existing `project_root` filtering**; CLI callers (`cleanup.py:113/120/166/214`) must tolerate the extra keys (verify in tasks — additive dict keys, low risk).

## Tradeoffs Surfaced for Review

- **Dedicated worker over folding (D-1)** costs more lifecycle wiring than the minimal-diff fold, bought to avoid coupling FTS-heal liveness to provider ingestion. If reviewers prefer minimal surface over isolation, the fold is the alternative — but accept the provider-outage risk.
- **Touching `backfill_fts` (D-4/D-5)** edits the repo's idempotent heal primitive (and the function `conftest.py:449` stubs). It's the load-bearing change for AC-2; the alternative (leave `backfill_fts` as-is) leaves the worker unable to distinguish failure from success — rejected by 2 judges.
- **AC-6 on `/health` adds Milvus load** even cached; the TTL mitigates frequent polls but a very large collection still pays a full doc_id scan on cache miss (Milvus has no cheap count in this codebase). Flagged, mitigated, accepted.

## UI Contract

N/A — no UI surface. Backend / async-worker / health-endpoint / CLI-stats only.

## Out of Scope (follow-up issues)

- **pymilvus data-path schema cache not invalidated on `add_collection_field`** — a separate latent stale-view vector: after a field add (e.g. SESF-25 `issue_ids`), a long-lived persistent client can keep raising `65535` on the insert/search path until LRU eviction or restart, even after Milvus settles. SESF-38's cadence-retry + AC-4 restart guidance + the `/health` standing-failure signal are the operator's cue, but the durable fix (invalidate `GlobalCache.schema` after schema-mutating ops) belongs in its own ticket. **File as a new SESF- issue.**
- Extending `detect_schema_drift` to dtype/length diffing (currently NAME-only) — out of scope per requirements Non-Goals.
- A dedicated `cleanup.py rebuild-fts` offline operator command — Non-Goal (auto self-heal covers the running-server case).

## Risk Notes

- **Cancellation during an in-flight heal** → `run_in_executor` can't interrupt a blocking thread at shutdown. Mitigation: `lifespan` teardown awaits the in-flight heal with a bounded timeout, then proceeds; the dedicated executor is shut down after. Document the chosen teardown behavior in tasks.
- **Sentinel re-arm race** → `_check_and_migrate` can re-`touch` the sentinel on another thread after a heal clears it. Mitigation: harmless — the next tick re-runs, and D-5's count-gate makes a *premature* clear impossible, so the loop converges rather than flaps.
- **Single-`db_path` assumption** → the sentinel is a single global file while lag is per-`db_path`. SessionFlow runs one configured Milvus `db_path`, so they stay consistent; state the constraint and don't generalize.
- **Milvus Lite describe caching** → AC-3 assumes `describe_collection` is cache-free; on the Lite fallback it may differ. If Lite caches, the re-describe returns the stale view and recovery degrades to a (still non-destructive) raise — acceptable, equal to today's behavior. Pin/verify the pymilvus version.
- **TOCTOU on startup** → eliminated by the single-dispatcher design (D-3); no two callers means no check-then-act window.

---

> **Phase complete?** Architecture clear, decisions logged with rationale, file map complete. Then advance: `/sketch-tasks SESF-38`.

## Review Archive — approach (2026-06-06)

### Resolution Summary

- **Accepted: 4** (all Codex findings — verified against live code before applying).
- **Rejected: 0.**
- **Resolved with your input: 0** (all were mechanical/correctness fixes; user pre-authorized application).
- Applied to: Seam 1 (unconditional first tick), Seam 2 + D-9 (re-describe gates auto-migrate too), Seam 3 + D-7 (split static lag from worker state), Seam 3 + D-6 + File Map (`project_root`-scoped `count_rows`). The "Unverified assumption" (pymilvus not re-opened by Codex) is already covered — the discovery sweep read the installed pymilvus source directly (`grpc_handler.py`/`cache.py`/`milvus_client.py`), confirming `describe_collection` is cache-free.

### Codex inline marks (verbatim)

> CODEX: This contradicts D-3's "first heal still runs ≈at startup": live `_fts_backfill` runs `rag_engine.backfill_fts` unconditionally after a 1s sleep (`http_server.py:704-720`), while the proposed worker only runs when `fts_backfill_required()` is set. Unless tasks require either an initial unconditional pass or a lag-detected trigger, replacing the one-shot will stop healing pre-sentinel or legacy missing FTS rows that today's idempotent `backfill_fts` catches.

> CODEX: Make the re-describe happen before the auto-migrate branch too, not only before raising. Live `_ensure_collection` drops/recreates immediately when `SESSIONFLOW_AUTO_MIGRATE_SCHEMA` is set after first drift (`rag_engine.py:538-546`); a stale-view drift under that opt-in would still destroy an intact collection, which cuts against US-2/AC-4's non-destructive-first intent.

> CODEX: This mixes module ownership: `FtsHealState` is intentionally owned by `http_server.py` (D-7), but `rag_engine.fts_lag_status(db_path)` is asked to return `consecutive_failures`/`last_error` for both `/health` and `get_stats`. Live MCP/CLI stats call `rag_engine.get_stats` without an HTTP worker (`tools.py:500-505`, `cleanup.py:162-166`), so the approach needs to split static lag data from HTTP worker state or pass worker state only when building the health payload.

> CODEX: Preserve `get_stats(project_root=...)` scoping for these fields. `tools.py:500-505` passes the current project root into `rag_engine.get_stats`, and `rag_engine.py:1480-1491` filters Milvus rows by that value; an unfiltered `FTSIndex.count_rows(conn)` would compare project-scoped Milvus totals to global FTS rows. Either make the count helper accept the same `project_root` filter (FTS rows already store it) or name the fields as global-only.

### CODEX Review (2026-06-06)

#### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, and the SESF-38 `requirements.md`, `discovery.md`, and `approach.md`.
- Verified the live one-shot FTS trigger in `http_server.py:704-720`, drain worker/event pattern in `http_server.py:228-269`, `/health` payload in `http_server.py:345-361`, and lifecycle teardown in `http_server.py:699-756`.
- Verified `_ensure_collection`, `milvus_client_for_migration`, `milvus_client`, `get_stats`, `search()` sentinel warning, and `backfill_fts` in `rag_engine.py:472-623`, `rag_engine.py:1007-1017`, `rag_engine.py:1480-1512`, and `rag_engine.py:1653-1741`.
- Verified FTS sentinel/thread-local/metadata behavior in `fts_hybrid.py:36-48`, `fts_hybrid.py:61-72`, `fts_hybrid.py:142-212`, and `fts_hybrid.py:254-326`; verified MCP/CLI stats scoping in `tools.py:500-505` and `cleanup.py:162-166`.

#### Top concerns

- The dedicated worker is described as sentinel-gated but also as replacing the unconditional startup `backfill_fts`; without an initial unconditional pass or lag-detected trigger, it regresses today's idempotent startup catch-up behavior.
- `_ensure_collection` needs to re-verify drift before the `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1` destructive branch, not only before the raised-error branch.
- The observability design crosses module boundaries by asking `rag_engine.fts_lag_status` to return HTTP-worker state, and the `get_stats` extension needs explicit `project_root` scoping or global-only field names.

#### Unverified assumptions

- Did not re-open pymilvus source during this review; relied on discovery's cited finding that `describe_collection` is cache-free in pymilvus 2.6.
