---
sketch: SESF-24-recency-ranking
phase: tasks
created: 2026-05-28
approved: 2026-05-28
---

# SESF-24 — Tasks

_Concrete checklist grouped into Sprint Cohorts. `[P]` marks tasks that can run in parallel within a cohort. Tasks reference file paths from approach.md._

> **Approval gate:** `/sketch run` refuses to run unless the `approved:` frontmatter field above contains a date in `YYYY-MM-DD` format, <=14 days old. Stamp it only after reviewing all four sketch files.

> **Skill-name discipline:** Closing tasks below name specific skills (`/release patch`, `/vault-update`, `codacy-cli`, `/sketch archive`). When generating or executing tasks.md, **invoke skills by name verbatim** --- paraphrasing the steps inline is not equivalent because the skill enforces project-specific rules the prose can't carry. If a closing task is genuinely irrelevant to the project (e.g., no version management, no foundation docs touched), mark it `N/A --- <one-line reason>` rather than dropping the task. Audible skips beat silent skips.

## Sprint Cohort 0 --- Setup (sequential)

_Cohort 0 ensures the worktree exists before implementation begins. If the worktree is missing, the executing agent creates it --- this is setup work, not a stop-the-world gate._

- [x] **0.1** --- Ensure sketch worktree exists
  - **File(s):** _no file changes --- verification only_
  - **Acceptance:** `git worktree list` shows branch `sketch/SESF-24-recency-ranking`. Working directory is that worktree, not the main checkout. If the worktree does not exist yet, create it.
  - **Leverage:** `using-git-worktrees` skill. (No project-specific setup skill — SessionFlow has no `.context/sketch-conventions.md`, so the generic `sketch/{ISSUE-ID}-{slug}` branch convention applies.)

## Sprint Cohort A --- Foundation (parallel-safe)

_Independent scaffolding in two different files — no shared symbols, no behavioral logic. Dispatchable concurrently._

- [x] **A.1 [P]** --- Add `LEGAL_SORT_BY` frozenset
  - **File(s):** `provider_adapters.py`
  - **Acceptance:** A module-level `LEGAL_SORT_BY = frozenset({"relevance", "recency", "hybrid"})` exists alongside `LEGAL_PROVIDERS` / `LEGAL_SOURCE_KINDS` (`provider_adapters.py:16`), following the same definition pattern. Importable from both `rag_engine.py` and `tools.py`.
  - **Leverage:** Mirror `LEGAL_PROVIDERS` / `LEGAL_SOURCE_KINDS` at `provider_adapters.py:16-25`. (approach D-7)
  - **Maps to:** AC EARS-5, EARS-12

- [x] **A.2 [P]** --- Add `_env_float` env helper
  - **File(s):** `embedding_control.py`
  - **Acceptance:** A new `_env_float(name, default, minimum=None, maximum=None)` helper exists next to `_env_int` / `_env_bool` (`embedding_control.py:26-40`). On unparseable input OR a value outside `[minimum, maximum]` it returns `default` (**fallback**, NOT clamp — distinct from `_env_int`). No existing caller of `_env_int`/`_env_bool` is changed.
  - **Leverage:** Existing `_env_int` at `embedding_control.py:26`; note the deliberate fallback-vs-clamp divergence (approach D-5).
  - **Maps to:** AC EARS-9, EARS-10

## Sprint Cohort B --- Tests · RED (sequential)

_TDD red phase. Encode every testable EARS criterion as a failing test before any engine/tool logic exists. All tests must fail (red) because the new behavior isn't implemented yet._

- [x] **B.1** --- Write failing tests for `sort_by` ranking
  - **File(s):** `tests/test_recency_ranking.py` (new)
  - **Acceptance:** A new test module asserts each testable EARS criterion, all currently failing (red). Coverage MUST include:
    - **Validation, both layers (EARS-5, EARS-12):** `rag_engine.search(..., sort_by="bogus")` raises `ValueError` naming the allowed values; AND `search_session` / `search_all_sessions` handlers return the **direct** `Invalid sort_by: ... expected one of ...` `TextContent` branch — assert that exact message shape, NOT merely "any error text" (the global `Error executing {name}: ...` wrapper at `tools.py:395-396` must not be what satisfies the test).
    - **Default = hybrid (EARS-4, EARS-13):** omitting `sort_by` on both engine and both tools applies `hybrid`.
    - **`relevance` parity (EARS-3):** ordering identical to the pre-SESF-24 `recency_boost=False` post-RRF order — assert across **all three candidate-set shapes: vector-only, FTS-only, and mixed** (D-1 routes single-engine searches through `rrf_merge`, changing the bypass shape at `rag_engine.py:776-779`; parity must hold on each).
    - **`recency` re-rank (EARS-2):** orders the existing candidate pool by timestamp descending; missing-timestamp rows sort last deterministically; candidate generation unchanged.
    - **`hybrid` ordering (EARS-1, EARS-7):** newer result ranks above older for the same topic; both `semantic_score` (min-max-normalized `_rrf_score`) and `recency_score` confirmed in `[0,1]` before blend; blend formula `(1-w)*semantic + w*recency`.
    - **Decay math (EARS-6):** `recency_score = exp(-days_old / decay_days)`; a `decay_days`-old result scores ≈0.367.
    - **Timezone-safe age (discovery/approach constraint):** mixed naive + aware ISO-8601 timestamps don't raise `TypeError`; future-dated timestamp clamps to `days_old = max(0.0, …)` (no inflated boost).
    - **Missing-timestamp fallback (EARS-8):** unparseable/empty timestamp → `recency_score = 0.5`, row not dropped or raised.
    - **Env tunables (EARS-9, EARS-10, EARS-11):** `SESSIONFLOW_RECENCY_WEIGHT` in `[0,1]` is used; unparseable/out-of-range falls back to `0.3`; `SESSIONFLOW_RECENCY_DECAY_DAYS` used when set, defaults to `7`, clamps to `minimum=1` on `0`/negative.
    - **Min-max div-by-zero guard (approach Risk):** a 1-row pool or all-equal `_rrf_score` assigns a neutral `semantic_score` rather than dividing by zero.
    - **`_fts_warning` preservation (non-score metadata):** when `fts_backfill_required()` is true, the `_fts_warning` field set at `rag_engine.py:799-804` survives the post-scoring score-field cleanup — the strategy dispatcher strips only ranking scratch keys (`_rrf_score`, `_score`), not all underscore keys.
  - **Depends on:** A.1 (imports `LEGAL_SORT_BY`), A.2 (imports `_env_float`)
  - **Leverage:** Validation-test pattern at `tests/test_search_provider_metadata.py:70`; fixtures at `tests/conftest.py:210-242`. No prior recency/`sort_by` test exists — building from zero.
  - **Maps to:** All AC (EARS-1 … EARS-13)

## Sprint Cohort C --- Implementation · GREEN (sequential)

_TDD green phase. Minimum code to turn Cohort B green. C.2 depends on C.1's new `search()` signature, so it is NOT parallel-safe._

- [x] **C.1** --- Engine: `sort_by` strategy dispatcher + age-aware scorer
  - **File(s):** `rag_engine.py`
  - **Acceptance:** In `search()` and `search_async()`: replace the `recency_boost: bool` param with `sort_by: str` (default `"hybrid"`); validate against `LEGAL_SORT_BY`, raising `ValueError` on bad input (EARS-5). Route **every** candidate set through `rrf_merge` (eliminate the single-engine bypass at `rag_engine.py:773-779`) so all rows carry `_rrf_score`; defer the `_rrf_score` pop (`rag_engine.py:787`) until after scoring. Replace `_apply_recency_boost` (`rag_engine.py:811`) with a strategy dispatcher: `relevance` = post-RRF order, no re-rank (EARS-3); `recency` = timestamp-desc re-rank, missing-ts last (EARS-2); `hybrid` = `final = (1-w)*semantic_score + w*recency_score` (EARS-1). `semantic_score` = min-max-normalized `_rrf_score` with a zero-range neutral guard (EARS-7); `recency_score = exp(-days_old / decay_days)` from UTC-normalized, timezone-safe age with `days_old = max(0.0, …)` clamp (EARS-6); missing/unparseable timestamp → `0.5` (EARS-8). Read `SESSIONFLOW_RECENCY_WEIGHT` via `_env_float` (default 0.3, fallback on bad input) and `SESSIONFLOW_RECENCY_DECAY_DAYS` via `_env_int` (default 7, `minimum=1`) (EARS-9/10/11). Strip **ranking score scratch fields only** (`_rrf_score`, `_score`, and any scoring scratch keys) before returning — do **NOT** blanket-strip all underscore keys: preserve non-score engine metadata like `_fts_warning` (`rag_engine.py:799-804`), which callers render as "keyword index rebuilding...". All Cohort B engine-level tests pass (green).
  - **Depends on:** A.1, A.2, B.1
  - **Leverage:** `rrf_merge` at `fts_hybrid.py:347` (handles empty lists natively, rank-order-preserving); existing validation pattern at `rag_engine.py:679-688`; `math.exp` + stdlib datetime. (approach D-1…D-6, D-8)
  - **Maps to:** AC EARS-1, EARS-2, EARS-3, EARS-4, EARS-5, EARS-6, EARS-7, EARS-8, EARS-9, EARS-10, EARS-11

- [x] **C.2** --- MCP surface: `sort_by` parameter, two-layer validation, hybrid default
  - **File(s):** `tools.py`
  - **Acceptance:** Add `sort_by` (enum `relevance`/`recency`/`hybrid`, default `hybrid`) to the inline `search_session` schema (`tools.py:195-213`) and to `build_search_all_sessions_schema()` (`tools.py:141`). In both handlers, validate `sort_by` against `LEGAL_SORT_BY` returning the direct `Invalid sort_by: ... expected one of ...` `TextContent` error (mirroring the provider/source_kind layer at `tools.py:320-331`). Pass `sort_by=` (default `"hybrid"`) instead of `recency_boost=` (`tools.py:303`, `:338`). Update both tool descriptions (`tools.py:190-194`, `:217-220`) to reflect the `hybrid` default (drop "Pure semantic search without recency bias" / "boosted by recency" wording). Ensure no internal score field leaks into `format_results` (`tools.py:38`). All Cohort B tool-level tests pass (green).
  - **Depends on:** C.1 (consumes the new `search(sort_by=...)` signature), A.1
  - **Leverage:** Two-layer validation pattern at `tools.py:320-331`; `format_results` at `tools.py:38`. (approach D-7, D-9, EARS-12, EARS-13)
  - **Maps to:** AC EARS-5, EARS-12, EARS-13

---

## Standard Closing Tasks

- [x] **CLOSE-1. Run full test suite** --- zero regressions
  - **File:** _no file changes --- verification only_
  - Run the project's complete test command. All existing tests pass; all new tests from Cohort B pass (green after Cohort C implementation).
  - If anything fails: fix the implementation, not the test. Tests are the spec --- a failing test means the implementation is wrong, not the test.

- [x] **CLOSE-2. Codacy CLI scan** --- security + quality
  - **File:** _no file changes --- scan only_
  - Run `codacy-cli` skill against changed files. Triage: Critical/High must fix, Medium fix-or-document, Low/Info advisory.

- [x] **CLOSE-3. Generate verification stamp**
  - **File:** Append to bottom of `tasks.md` (this file) under heading `## Verification Stamp`.
  - For EACH acceptance criterion in `requirements.md` (EARS-1 … EARS-13), write exactly one traceability line: `- [x] EARS-N — verified at <path>:<line>` OR `- [x] EARS-N — verified by <test name>` OR `- [ ] EARS-N — gap: <reason>`.
  - The stamp block MUST list every EARS criterion. No UI verification applies — `approach.md`'s `## UI Contract` is N/A (no UI surface).
  - Refuse to proceed to CLOSE-4 if any `[ ]` remains in the stamp block.

- [x] **CLOSE-4. Version bump**
  - N/A --- SessionFlow opts out of version management (no `devops/version.lock`; Python project with no published package).

- [x] **CLOSE-5. Vault update**
  - **MUST invoke `/vault-update`** as a skill --- even if you believe no foundation docs are affected, the skill performs the audit.
  - Do **not** close the Plane issue here. On SessionFlow's branch-protected, PR-required `main` flow, the change is still unmerged at this point. Final Done transition happens after merge (CLOSE-8).

- [x] **CLOSE-6. Open PR**
  - Use worktree branch `sketch/SESF-24-recency-ranking`. Title: `feat(SESF-24): hybrid recency-aware ranking + sort_by parameter`.
  - Body must include: link to source issue, link to sketch folder (`DocVault/Projects/SessionFlow/sketches/SESF-24-recency-ranking/`), test plan checklist.

- [x] **CLOSE-7. Resolve PR review threads**
  - **MUST invoke `/pr-resolve`** as a skill --- triages findings, fixes-or-classifies, replies to threads consistently.
  - Coverage: scan **both** inline diff threads AND review-body findings (Codacy/Copilot often post critical findings as summary prose, not inline). All Critical/High fixed or marked false-positive with reasoning.

- [x] **CLOSE-8. Close issue + archive sketch** (after PR merges)
  - Mark the source issue Done in Plane: `mcp__plane__update_issue` to state "Done". (Deferred from CLOSE-5 — final closure only after merge.)
  - **MUST invoke `/sketch archive SESF-24`** as a skill --- moves folder to `archive/YYYY-MM-DD-SESF-24-recency-ranking/` and saves mem0 summary.

---

> **Multi-model dispatch hint:** Cohort A tasks marked `[P]` (A.1 provider_adapters.py, A.2 embedding_control.py) touch independent files and can be sent to different models in parallel. Reconverge before Cohort B. The B (red) → C (green) boundary is a natural model-routing seam.

## Review Archive — tasks (2026-05-28)

### Resolution Summary

- **Accepted (2):** (1) CODEX top-concern #1 — C.1 reworded to strip ranking score scratch keys only and preserve `_fts_warning`; matching `_fts_warning`-preservation assertion added to B.1. (2) CODEX top-concern #2 — issue-Done transition removed from CLOSE-5 (retitled "Vault update") and deferred to CLOSE-8 (after PR merge).
- **Rejected (0).**
- **Resolved with your input (0).**

---

### CODEX Review (2026-05-28)

#### Verified

- Read `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, and all SESF-24 phase files (`requirements.md`, `discovery.md`, `approach.md`, `tasks.md`); no `.context/GLOSSARY.md` or `.context/sketch-conventions.md` exists in the repo.
- Checked live search code in `/Volumes/DATA/GitHub/SessionFlow/rag_engine.py`: `search()` validation and merge flow, the current single-engine bypass, `_rrf_score` cleanup, `_apply_recency_boost`, the `_fts_warning` sentinel branch, and `search_async` forwarding (`rag_engine.py:660-808`, `rag_engine.py:811-867`, `rag_engine.py:1258-1285`).
- Checked `/Volumes/DATA/GitHub/SessionFlow/tools.py` schemas, descriptions, direct handler validation, global error wrapper, and `format_results()` (`tools.py:38-80`, `tools.py:141-221`, `tools.py:296-345`, `tools.py:395-396`).
- Checked `/Volumes/DATA/GitHub/SessionFlow/fts_hybrid.py`, `/Volumes/DATA/GitHub/SessionFlow/provider_adapters.py`, `/Volumes/DATA/GitHub/SessionFlow/embedding_control.py`, and existing search/provider test fixtures for the cited task leverage points.
- Confirmed `git worktree list` currently shows only the main SessionFlow checkout, so Cohort 0 correctly needs to create `sketch/SESF-24-recency-ranking` if execution starts from the current state.

#### Top concerns

1. C.1's "strip all internal score fields / scratch keys" wording can accidentally remove the existing `_fts_warning` result metadata unless the task explicitly preserves non-score underscore fields.
2. CLOSE-5 closes the Plane issue before PR creation, review-thread resolution, and merge, which can put SESF-24 in Done while the required branch-protected delivery path is still incomplete.

#### Unverified assumptions

- I did not verify the live Plane issue body; this review used the source issue excerpt embedded in `requirements.md`.
- I did not run tests, live Milvus queries, or MCP tool calls; findings are based on static source verification and the current sketch artifacts.

---

## Verification Stamp

_Generated 2026-05-28 (CLOSE-3). Line numbers reference the `sketch/SESF-24-recency-ranking` worktree. UI Contract: N/A (no UI surface). Full suite: 104 passed (32 SESF-24 tests). Codacy: pylint/trivy clean; lizard `search` CCN pre-existing; opengrep tool-infra crash (not a code finding)._

- [x] EARS-1 — hybrid blended score `(1-w)*semantic + w*recency` — verified at `rag_engine.py:902-916` (`_rank_results` hybrid branch); test `test_hybrid_blend_uses_normalized_semantic_and_recency`
- [x] EARS-2 — `recency` re-ranks candidate pool by timestamp desc — verified at `rag_engine.py:895-901`; test `test_recency_orders_by_timestamp_descending`
- [x] EARS-3 — `relevance` = RRF passthrough, no recency re-rank — verified at `rag_engine.py:893-894`; tests `test_relevance_parity_mixed_matches_rrf_oracle`, `test_relevance_order_is_timestamp_independent`
- [x] EARS-4 — `sort_by` omitted defaults to `hybrid` (engine) — verified at `rag_engine.py:675`; test `test_engine_default_sort_by_is_hybrid`
- [x] EARS-5 — invalid `sort_by` rejected naming allowed values — verified at `rag_engine.py:693-696`; test `test_engine_rejects_invalid_sort_by_before_milvus`
- [x] EARS-6 — `recency_score = exp(-days_old/decay_days)`, decay default 7 — verified at `rag_engine.py:848-859`, `rag_engine.py:52`; test `test_recency_score_decay_constant`
- [x] EARS-7 — both terms normalized to `[0,1]` before blend — verified at `rag_engine.py:862-875` (`_semantic_scores` min-max) + `rag_engine.py:859` (recency in (0,1]); test `test_semantic_scores_are_min_max_normalized`
- [x] EARS-8 — missing timestamp gets deterministic fallback (0.5), row not dropped — verified at `rag_engine.py:53,857`; tests `test_recency_score_missing_timestamp_is_neutral`, `test_hybrid_does_not_drop_rows_with_missing_timestamp`
- [x] EARS-9 — `SESSIONFLOW_RECENCY_WEIGHT` in `[0,1]` used else default 0.3 — verified at `rag_engine.py:903-905`; test `test_recency_weight_env_changes_hybrid_order`
- [x] EARS-10 — unparseable/out-of-range weight silently falls back to 0.3 — verified at `embedding_control.py:36-58` (`_env_float` fallback); tests `test_env_float_falls_back_on_unparseable`, `test_env_float_falls_back_when_out_of_range`
- [x] EARS-11 — `SESSIONFLOW_RECENCY_DECAY_DAYS` used else default 7 — verified at `rag_engine.py:906-908`; tests `test_decay_days_env_clamps_zero_to_minimum_one`, `test_decay_days_env_zero_does_not_crash_hybrid`
- [x] EARS-12 — `search_all_sessions` + `search_session` both expose `sort_by` — verified at `tools.py:170-176`, `tools.py:218-224`; test `test_legal_sort_by_frozenset_exists`
- [x] EARS-13 — both tools default to `hybrid` when `sort_by` omitted — verified at `tools.py:313,342`; test `test_tool_default_sort_by_is_hybrid`
