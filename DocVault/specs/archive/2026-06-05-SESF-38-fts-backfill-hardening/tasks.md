---
sketch: "SESF-38-fts-backfill-hardening"
phase: tasks
created: 2026-06-05
approved: 2026-06-05
---

# SESF-38 — Tasks

_Concrete checklist grouped into Sprint Cohorts. `[P]` marks tasks that can run in parallel within a cohort. Tasks reference file paths from approach.md._

> **Approval gate:** `/sketch run|workflow|dispatch` refuses to run unless the `approved:` frontmatter field above contains a date in `YYYY-MM-DD` format, ≤14 days old. Stamp it only after reviewing all four sketch files.

> **TDD discipline:** Cohort B tests are written to FAIL (RED) before any Cohort C implementation. Do not modify a test to make it pass — a failing test means the implementation is wrong. Behavior tests must import the **real** `rag_engine` module, not `conftest.py:449 stub_rag_engine` (which forces `backfill_fts → 0`).

## Sprint Cohort 0 — Setup (sequential)

- [ ] **0.1** — Ensure sketch worktree exists
  - **File(s):** _no file changes — verification only_
  - **Acceptance:** `git worktree list` shows the sketch worktree and branch. Working directory is that worktree, not the main checkout. No project `.context/sketch-conventions.md` override exists, so fall back to branch `sketch/SESF-38-fts-backfill-hardening`.
  - **If the worktree does not exist yet:** Create it via the generic `using-git-worktrees` skill.
  - **Leverage:** `using-git-worktrees` skill.

## Sprint Cohort A — Foundation (parallel-safe)

_New symbols as skeletons so Cohort B tests can import them — no behavioral logic yet. All three touch distinct files (rag_engine / fts_hybrid / http_server) with no shared symbols → genuinely `[P]`._

- [x] **A.1 [P]** — `rag_engine` foundation: typed transient error + lag-status signature _(b4690fc)_
  - **File(s):** `rag_engine.py`
  - **Acceptance:** `FtsBackfillTransientError(Exception)` exists; `fts_lag_status(db_path, project_root=None)` is defined with a docstring and a stub return (`{"milvus_turn_count": 0, "fts_row_count": 0, "fts_lag": 0, "fts_backfill_required": False}`) that B.2 will pin. Module imports cleanly; `ruff check rag_engine.py` clean on the new symbols.
  - **Maps to:** D-4, D-6, AC-6
  - **Leverage:** existing `get_stats` (`rag_engine.py:1480`) for the dict shape; `fts_hybrid.fts_backfill_required` (`:36`).

- [x] **A.2 [P]** — `FTSIndex.count_rows` skeleton with optional `project_root` filter _(0703633)_
  - **File(s):** `fts_hybrid.py`
  - **Acceptance:** `FTSIndex.count_rows(self, conn, project_root=None) -> int` is defined (stub `return 0`) with a docstring noting it runs `SELECT count(*) FROM turns_fts` and filters by `project_root` when provided. Importable; `ruff check fts_hybrid.py` clean.
  - **Maps to:** AC-6, D-6
  - **Leverage:** `FTSIndex` table name + per-thread `connection()` (`fts_hybrid.py:189`); `turns_fts` column list incl. `project_root` (`rag_engine.py:353`).

- [x] **A.3 [P]** — `FtsHealState` skeleton (worker-owned backoff + log-once state) _(4e5246f)_
  - **File(s):** `http_server.py`
  - **Acceptance:** `FtsHealState` (dataclass or small class) defined with fields `consecutive_failures: int`, `last_error_signature: str | None`, `current_backoff_seconds: float`, and stub methods `record_failure(exc)`, `record_success()`, `next_delay() -> float`, `should_warn(exc) -> bool`. No behavior yet (stubs). Importable; `ruff check http_server.py` clean.
  - **Maps to:** D-7, AC-2, AC-5
  - **Leverage:** `EmbeddingBudget._cap_warned` log-once latch pattern (`embedding_control.py:153-161`); drain const pattern (`http_server.py:64-77`).

## Sprint Cohort B — Tests · RED (sequential)

_Each EARS acceptance criterion gets ≥1 failing test. Sequential because tests share fixtures/imports. Each depends on the Cohort A skeletons it imports._

- [x] **B.1** — Failing tests for AC-3 (re-describe gates BOTH branches) + AC-4 (honest message) _(23c31ab — 3 RED)_
  - **File(s):** `tests/test_schema_drift.py` (extend)
  - **Acceptance:** (AC-3) `describe_collection` mock returns drift on first call, clean on second → assert `_ensure_collection` does NOT raise and does NOT call `migrate_schema`, on **both** the default path **and** with `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1` set (the auto-migrate branch must also re-verify). (AC-4) With persistent drift (both reads dirty, AUTO_MIGRATE unset) assert the `RuntimeError` text: contains the literal `cleanup.py migrate-schema`, marks **both** `migrate-schema` and `SESSIONFLOW_AUTO_MIGRATE_SCHEMA` destructive ("all Turns lost"), and names a restart as non-destructive **before** any destructive step. All new assertions FAIL today.
  - **Depends on:** A.1
  - **Maps to:** AC-3, AC-4
  - **Leverage:** existing `describe_collection` mock harness (`tests/test_schema_drift.py:14-52`).

- [x] **B.2** — Failing tests for AC-6 (lag observability + scoping) + D-5 (sentinel-clear gate) _(483e482 — 5 RED)_
  - **File(s):** `tests/test_fts_lag.py` (new), `tests/test_provider_ingestion.py` (extend for D-5)
  - **Acceptance:** `fts_lag_status` returns correct `milvus_turn_count`/`fts_row_count`/`fts_lag`(=delta)/`fts_backfill_required` from stubbed counts; `count_rows(project_root=X)` filters (project-scoped count ≠ global count); `get_stats(project_root=X)` carries the three additive fields **project-scoped**; the `/health` JSON carries an `fts` key (assert via Starlette test client). (D-5) a `backfill_fts` whose hydrate leaves `fts_row_count < milvus_turn_count` must NOT clear the sentinel; only `fts_row_count >= milvus_turn_count` clears it. All FAIL today.
  - **Depends on:** A.1, A.2
  - **Maps to:** AC-6, AC-1 (D-5 gate)
  - **Leverage:** `test_provider_ingestion.py:136-182` monkeypatch seam (`milvus_client`/`_query_batches`/`_fts.{connection,insert,close_ephemeral}`); real `rag_engine` module (not `conftest.py:449` stub).

- [x] **B.3** — Failing tests for AC-2 (bounded backoff, never abandon) + AC-5 (log-once) _(0e0bbc2 — 8 RED)_
  - **File(s):** `tests/test_fts_heal_state.py` (new)
  - **Acceptance:** (AC-2) feeding N consecutive transient failures into `FtsHealState`, `next_delay()` escalates 30→60→120→240→cap and stays at cap; never returns a terminal "give up"; `record_success()` resets to base. (AC-5) `should_warn` returns True on the first occurrence of a signature and on a **changed** signature, False while the **same** signature repeats; signature is normalized to `type(exc).__name__` + stable message **prefix** (a `MilvusException` with a volatile request-id suffix must NOT churn). All FAIL today.
  - **Depends on:** A.3
  - **Maps to:** AC-2, AC-5
  - **Leverage:** `test_embedding_control.py:62-78` `_cap_warned` assertion style.

- [x] **B.4** — Failing tests for AC-1 (self-heal cadence) + D-4 (typed transient raise) _(76cfc6d — 5 RED; seam `_fts_heal_run_once(state, *, first_tick)`)_
  - **File(s):** `tests/test_fts_heal_worker.py` (new)
  - **Acceptance:** (D-4) `backfill_fts` raises `FtsBackfillTransientError` when the guarded `milvus_client` open raises `RuntimeError` (drift guard) AND when the query raises `MilvusException`; it does not swallow to stderr. (AC-1) driving one `_fts_heal_worker` tick: the **first** tick calls `backfill_fts` **unconditionally** (sentinel unset) — proving startup catch-up is preserved; a later tick calls it only when `fts_backfill_required()` is set, and on success the sentinel is cleared — no restart involved. All FAIL today.
  - **Depends on:** A.1, A.3
  - **Maps to:** AC-1, AC-2, D-3, D-4
  - **Leverage:** `_backfill_drain_worker` event/wait_for shape (`http_server.py:251-269`); monkeypatch `rag_engine.backfill_fts` for the cadence assertions.

## Sprint Cohort C — Implementation · GREEN (sequential)

_Minimum code to turn every Cohort B test green. Sequential — most tasks share `rag_engine.py`/`http_server.py` and have dependency chains._

- [x] **C.1** — `_ensure_collection`: re-describe before BOTH branches + honest AC-4 message _(80435fa — 11 pass)_
  - **File(s):** `rag_engine.py`
  - **Acceptance:** B.1 green. On non-empty drift, re-call `detect_schema_drift` once and proceed if it clears — placed **before** the `SESSIONFLOW_AUTO_MIGRATE_SCHEMA` branch (`:538-546`) and the raise. Rewrite the `RuntimeError` per AC-4 (restart-first, both paths destructive, literal `cleanup.py migrate-schema` retained).
  - **Depends on:** B.1
  - **Maps to:** AC-3, AC-4, D-9
  - **Leverage:** `detect_schema_drift` (`:472-483`, cache-free describe).

- [x] **C.2** — `backfill_fts`: catch+typed-raise, gated sentinel-clear, defensive lock _(53a7705 — D-4/D-5 green)_
  - **File(s):** `rag_engine.py`
  - **Acceptance:** B.4 (D-4) + B.2 (D-5) green. Wrap the `with milvus_client(...)` body to catch open-time `RuntimeError` and query-time `MilvusException`, re-raising `FtsBackfillTransientError`. Move `clear_fts_backfill_sentinel()` to fire only when post-insert `fts_row_count >= milvus_turn_count`. Add a module-level non-blocking `threading.Lock` around the heal entry. Preserve the `test_provider_ingestion.py:136-182` monkeypatch seam (update the test there if the call shape changes).
  - **Depends on:** C.1, A.1
  - **Maps to:** AC-1, AC-2, D-2, D-3, D-4, D-5
  - **Leverage:** `backfill_fts` (`:1653-1741`); `fts_lag_status` counts (C.4) or inline `count_rows` for the gate.

- [x] **C.3** — `FTSIndex.count_rows` implementation (project_root filter) _(50ecf3d)_
  - **File(s):** `fts_hybrid.py`
  - **Acceptance:** B.2 count assertions green. `SELECT count(*) FROM turns_fts`, plus `WHERE project_root = ?` when `project_root` is provided. Runs on the caller-thread connection (SESF-13).
  - **Depends on:** B.2
  - **Maps to:** AC-6, D-6
  - **Leverage:** `turns_fts` schema incl. `project_root` (`rag_engine.py:353`).

- [x] **C.4** — `fts_lag_status` + `get_stats` extension (static lag, scoped) _(04425ca + c32ce29 correction — drift-tolerant count, sentinel-only required)_
  - **File(s):** `rag_engine.py`
  - **Acceptance:** B.2 green. `fts_lag_status(db_path, project_root=None)` returns static lag only — Milvus count via `milvus_client_for_migration` (drift-tolerant) with the same `project_root` filter `get_stats` uses, FTS count via `FTSIndex.count_rows(project_root=…)`, `fts_lag` delta, sentinel state. Thread `get_stats`' existing `project_root` into the helper and merge the additive fields into its return. **No worker state here.**
  - **Depends on:** C.3
  - **Maps to:** AC-6, D-6, D-7
  - **Leverage:** `milvus_client_for_migration` (`:588`); `get_stats` Milvus count (`:1480-1493`).

- [x] **C.5** — `FtsHealState` implementation (backoff + log-once) _(c2fdb48 — 8 pass)_
  - **File(s):** `http_server.py`
  - **Acceptance:** B.3 green. `next_delay` escalates `base*2**(n-1)` capped at the backoff cap; `record_success` resets; `should_warn`/signature = `type(exc).__name__` + stable message prefix; never abandons. Add the backoff-cap const near `:64-77` (reuse drain interval as base).
  - **Depends on:** B.3
  - **Maps to:** AC-2, AC-5, D-7
  - **Leverage:** `EmbeddingBudget._cap_warned` (`embedding_control.py:153-161`).

- [x] **C.6** — `_fts_heal_worker` + lifespan wiring (replaces the one-shot) _(490f9ab — 5/5 worker)_
  - **File(s):** `http_server.py`
  - **Acceptance:** B.4 (AC-1) green. New recurring `_fts_heal_worker`: **first tick unconditional**, sentinel-gated thereafter, runs `backfill_fts` on a **dedicated single-thread executor**, catches `FtsBackfillTransientError` → `FtsHealState` backoff + log-once, never exits except on cancel. Replace `_fts_backfill` (`:704-720`); schedule + cancel/await in `lifespan` mirroring `backfill_drain_task` (`:700`/`:752-756`); bounded-timeout await of an in-flight heal at teardown; shut the executor down after.
  - **Depends on:** C.5, C.2
  - **Maps to:** AC-1, AC-2, D-1, D-3, D-8
  - **Leverage:** `_backfill_drain_worker` (`:251-269`); `_embed_executor` pattern (`rag_engine.py:351+` comment) — but a SEPARATE executor.

- [x] **C.7** — `/health` fts payload (static lag + worker state, degraded, cached) _(d790097 — test_fts_lag 4/4)_
  - **File(s):** `http_server.py`
  - **Acceptance:** B.2 `/health` assertion green. `_fts_lag_payload()` merges `rag_engine.fts_lag_status(project_root=None)` (global) **plus** the worker's `FtsHealState` (`consecutive_failures`/`last_error`); wrapped in try/except degradation (mirror `_embedding_status_payload` `:333-342`) behind a short-TTL cache (mirror `_provider_health_cache` `:295-318`); added to `/health` JSON (`:345-361`) under a new `fts` key.
  - **Depends on:** C.4, C.5
  - **Maps to:** AC-6, D-6, D-7
  - **Leverage:** `_provider_health_cache` (`:295-318`), `_embedding_status_payload` (`:333-342`).

---

## Standard Closing Tasks

- [x] **CLOSE-1. Run full test suite** — zero regressions _(238 passed, 0 failed; the 2 test_watchdog failures were the worktree's missing venv symlink — pass with venv present)_
  - **File:** _no file changes — verification only_
  - Run `pytest tests/` (≤10-min suite timeout). All existing tests pass; all new Cohort B tests pass (green after Cohort C).
  - If anything fails: fix the implementation, not the test. Tests are the spec.

- [x] **CLOSE-2. Codacy CLI scan** — security + quality _(ruff clean on all 3 source files; Codacy CLI not bootstrapped locally — routed to the PR's required Codacy status check, triaged in CLOSE-7)_
  - **File:** _no file changes — scan only_
  - Run the `codacy-cli` skill against changed files (`http_server.py`, `rag_engine.py`, `fts_hybrid.py`, new tests). Triage: Critical/High must fix, Medium fix-or-document, Low/Info advisory. Also run `ruff check` on the changed `.py` files.

- [x] **CLOSE-3. Generate verification stamp** _(see ## Verification Stamp — all 6 ACs traced to passing tests)_
  - **File:** Append to bottom of `tasks.md` under heading `## Verification Stamp`.
  - For EACH AC (AC-1…AC-6) write one line: `- [x] AC-N — verified by <test name>` or `- [x] AC-N — verified at <path>:<line>`, or `- [ ] AC-N — gap: <reason>`. List every AC. No UI Contract → no visual-verification requirement. Refuse to proceed to CLOSE-4 if any `[ ]` remains.

- [x] **CLOSE-4. Version bump** — N/A — project opts out of version management (no `devops/version.lock`, no version-bearing files; SessionFlow is an internal MCP tool with no release workflow)
  - **File:** project version files.
  - **MUST invoke `/release patch`** as a skill. If SessionFlow opts out of version management (no `devops/version.lock` / no version-bearing files), write `N/A — project opts out of version management (<reason>)` as the acceptance line. Do not silently drop this task.

- [x] **CLOSE-5. Vault update + close issue**
  - SESF-38 marked **Done** in Plane (2026-06-05, post-merge). ✓
  - Follow-up filed: **SESF-40** (pymilvus data-path schema cache not invalidated on `add_collection_field`). ✓
  - `vault-update`: **audited — N/A (zero changes)**. SessionFlow has no DocVault `Foundation/` docs; no non-sketch vault page asserts the FTS-backfill/`/health` internals SESF-38 changed; no infra/schema/dependency facts changed. (Candidate follow-up: a project CLAUDE.md "Operational Gotchas" entry for the self-heal worker + `/health` FTS-lag stat — a repo file, separate from DocVault.)

- [x] **CLOSE-6. Open PR** _(https://github.com/lbruton/SessionFlow/pull/32)_
  - Branch `sketch/SESF-38-fts-backfill-hardening`. Title: `fix(SESF-38): self-heal FTS backfill on transient schema drift` (or `chore(SESF-38): …`).
  - Body must include: link to SESF-38, link to the sketch folder (`DocVault/Projects/SessionFlow/sketches/SESF-38-fts-backfill-hardening/`), and a test-plan checklist (the six ACs).

- [x] **CLOSE-7. Resolve PR review threads** _(18 threads — 5 Codacy + 7 Copilot + 6 CodeRabbit — all fixed across ae1db08/0674848/daf2868/c9135da, replied + resolved; 0 unresolved. Codacy gate = duplication metric only, not a required check. Re-review on c9135da in flight.)_
  - **File:** _GitHub PR threads only — code fixes land via the worktree as needed_
  - **MUST invoke `/pr-resolve`** as a skill. Scan **both** inline diff threads AND review-body findings (Codacy/Copilot often post outside the diff).
  - Critical/High: fix or mark false-positive with reasoning. Medium: fix or document waiver. Low/Info: advisory. Re-check for auto-scanner re-posts after each push.

- [x] **CLOSE-8. Archive sketch** (after PR merges)
  - PR #32 merged (squash) as commit `6cf54c1`; worktree removed, branch deleted, remote ref pruned, main fast-forwarded. Sketch archived to `archive/2026-06-05-SESF-38-fts-backfill-hardening/` + mem0 summary saved.

---

> **UI Contract Traceability:** N/A — `approach.md` UI Contract is "N/A — no UI surface." No mockup/playground/screenshot cited in any phase doc.

> **Multi-model dispatch hint:** Cohort A `[P]` tasks (A.1 rag_engine / A.2 fts_hybrid / A.3 http_server) touch disjoint files and can be sent to different models in parallel. The TDD boundary between Cohort B (RED) and Cohort C (GREEN) is a natural model-routing seam.

---

## Verification Stamp

_Suite: `pytest tests/` → **238 passed, 0 failed** (worktree, venv present). `ruff check` clean on rag_engine.py / http_server.py / fts_hybrid.py._

- [x] AC-1 (FTS Sidecar self-heals without restart) — verified by `tests/test_fts_heal_worker.py::test_first_tick_runs_backfill_even_when_sentinel_unset`, `::test_subsequent_tick_skips_backfill_when_sentinel_unset`, `::test_subsequent_tick_runs_and_clears_sentinel_when_set`; sentinel-consistency gate by `tests/test_provider_ingestion.py::test_backfill_fts_keeps_sentinel_when_hydration_incomplete`
- [x] AC-2 (transient failures retry, never abandon) — verified by `tests/test_fts_heal_state.py::test_next_delay_escalates_then_plateaus_at_cap`, `::test_next_delay_is_monotonic_and_never_terminal`, `::test_record_success_resets_streak_and_delay_to_base`
- [x] AC-3 (startup re-verifies drift before raising — both branches) — verified by `tests/test_schema_drift.py::test_ensure_collection_redescribes_clean_does_not_raise`, `::test_ensure_collection_redescribes_clean_does_not_migrate`
- [x] AC-4 (persistent-drift error gives honest, non-destructive-first guidance) — verified by `tests/test_schema_drift.py::test_ensure_collection_persistent_drift_message_is_honest`
- [x] AC-5 (repeated identical failures collapse to one log line) — verified by `tests/test_fts_heal_state.py::test_should_warn_first_true_then_false_on_repeat`, `::test_should_warn_true_when_signature_changes`, `::test_should_warn_resets_after_success`, `::test_signature_ignores_volatile_suffix`
- [x] AC-6 (FTS lag observable in health/stats, project-scoped) — verified by `tests/test_fts_lag.py::test_fts_lag_status_computes_lag_from_milvus_and_fts_counts`, `::test_count_rows_project_scoped_differs_from_global`, `::test_get_stats_carries_fts_lag_keys`, `::test_health_payload_carries_fts_lag`

_No UI Contract → no visual-verification requirement (CLOSE-3 UI clause N/A)._
