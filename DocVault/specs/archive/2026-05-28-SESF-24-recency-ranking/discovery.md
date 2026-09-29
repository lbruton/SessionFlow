---
sketch: "SESF-24-recency-ranking"
phase: discovery
created: 2026-05-28
---

# SESF-24 — Discovery

_Research the existing system and prior art. **Don't propose solutions** — that's the next phase._

## Existing Code

_Files and modules already in the project that this work will touch or build on._

| Path | Role | Notes |
|------|------|-------|
| `rag_engine.py:660` | `search()` — the system under test | Builds vector + FTS candidate pools (`fetch_n = n*3`), merges via `rrf_merge`, **pops `_rrf_score` at :787**, then optionally calls `_apply_recency_boost` at :793. Provider/source_kind validation raises `ValueError` at :679-688 — the pattern to mirror for `sort_by`. |
| `rag_engine.py:811` | `_apply_recency_boost()` — the code SESF-24 replaces | Rank-local, **linear** recency: `similarity*(1 + recency_weight*recency)` with `recency_weight = 0.3` **hardcoded** (:817). `recency` is the row's *position* in the returned set (newest=1.0, oldest=0.0), not true age. `similarity = 1 - distance` (:843). Missing-timestamp rows get `recency = 0.5` (:857); a set with <2 valid timestamps is returned unsorted (:836). |
| `rag_engine.py:1258` | `search_async()` — async wrapper | Also carries `recency_boost: bool` (:1260) and threads it to `search()` (:1282). Must change in lockstep if the kwarg is superseded. (Note: it does **not** forward `date_from`/`date_to` — pre-existing gap, out of scope.) |
| `tools.py:296` | `search_session` dispatch | Calls `rag_engine.search(..., recency_boost=True)` (:303). Inline schema at :195-213 has **no** `sort_by`. Tool description (:190-194) claims "boosted by recency". |
| `tools.py:308` | `search_all_sessions` dispatch | Calls `rag_engine.search(..., recency_boost=False)` (:338). Schema from `build_search_all_sessions_schema()` (:141). Mirrors validation by **returning a `TextContent` error** (:320-331), not raising — the second validation layer to extend. Description (:217-220) says "Pure semantic search without recency bias" — will need updating for the `hybrid` default. |
| `tools.py:38` | `format_results()` — user-visible ranking surface | Every search result is rendered through here before reaching the MCP caller. Displays `Relevance` as `similarity = 1 - distance` (:50, :74) — already **fake (1.0) for FTS-only rows** (`distance = 0.0`), and can disagree with hybrid/RRF ordering even when internal fields are correct. Any new internal score fields (`_rrf_score`, `_score`) must be **stripped before formatting** (or `format_results()` taught to ignore them) to avoid leaking into MCP text output. |
| `fts_hybrid.py:314` | FTS row construction | **Every FTS row is stamped `distance = 0.0`.** So FTS-only rows that survive the merge yield `similarity = 1 - 0 = 1.0` in the current boost — they masquerade as perfect semantic matches. This is the crux of Open Question #1. |
| `fts_hybrid.py:347` | `rrf_merge()` | Produces `_rrf_score = Σ 1/(k+rank)` (k=60) keyed by `doc_id` — the **only** relevance signal that spans both vector and FTS-only origins. Currently discarded at `rag_engine.py:787` before recency runs. **Single-engine bypass:** `search()` only calls `rrf_merge` when *both* vector and FTS return hits (`rag_engine.py:773-779`); when only one engine returns, `merged` is the raw result list and carries **no `_rrf_score`**. Any `semantic_score` built on `_rrf_score` must route all searches through `rrf_merge` (it handles an empty list natively) or attach a fallback score on the single-engine paths. |
| `provider_adapters.py:16` | `LEGAL_PROVIDERS` / `LEGAL_SOURCE_KINDS` | `frozenset` validation sets — the canonical pattern to copy for a `LEGAL_SORT_BY = frozenset({"relevance","recency","hybrid"})`. |
| `embedding_control.py:26` | `_env_int(name, default, minimum)` / `_env_bool` | Existing env-var parse-with-fallback+clamp helpers. There is **no** `_env_float`; `SESSIONFLOW_RECENCY_WEIGHT` (clamp `[0,1]`, EARS-9/10) needs one. `decay_days` (EARS-11) could reuse `_env_int`, but its invalid-input contract is **undefined** — EARS-11 only specifies the unset→`7` default, and EARS-9/10 define invalid fallback for *weight* only. Whether a `0`/negative/unparseable decay clamps to `1` or falls back to `7` is an **explicit approach decision**, not an assumed `minimum=1`. |
| `tests/test_search_provider_metadata.py:70` | Validation test pattern | Asserts `ValueError` on bad `provider`/`source_kind`. The model for new `sort_by` validation tests. **No recency or `sort_by` test exists anywhere** — the current boost is entirely untested, so Cohort B starts from zero coverage. |

## Prior Decisions

_Searched mem0 (`"SessionFlow search ranking recency boost RRF semantic score decision"`) — no prior decision touches SessionFlow's own ranking internals; the hits were all about mem0's API and unrelated skills._

- _None found on the ranking surface — first time formalizing the recency/relevance blend._
- Tangential: mem0 `4328e538` (2026-05-22) — SESF-19 made the cross-harness review trail searchable; confirms multi-provider rows now share the candidate pool, so the blend must treat all providers uniformly. Not a ranking decision.

## External References

_None required._ EARS-6's exponential decay `exp(-days_old / decay_days)` is standard math implementable with `math.exp` and stdlib datetime parsing; no library, RFC, or third-party pattern is needed. RRF (k=60) is already in-tree per the original paper.

## Constraints

_Things the implementation must respect._

- **Candidate generation is frozen** (requirements Non-Goal): the vector + FTS RRF merge that fills the pool is unchanged. All three `sort_by` modes re-rank the **existing merged pool** of up to `fetch_n = n*3` rows — including `recency`, which is a re-rank, not a chronological feed (EARS-2).
- **FTS-only rows have no real cosine distance** (`distance = 0.0`). Any `semantic_score` definition must not let those rows score as 1.0 — that's the bug the current boost has.
- **Two validation layers must stay consistent**: `rag_engine.search` raises `ValueError` (EARS-5); `tools.py` returns a `TextContent` error string. New `sort_by` validation must exist in both.
- **`recency_boost` has a small blast radius**: 2 production call sites (`tools.py`), 1 async wrapper (`search_async`), 0 tests reference it. Whatever happens to the kwarg, all three sites change together.
- **Timestamps are VARCHAR ISO-8601** sorted lexicographically; some rows have empty/unparseable timestamps (EARS-8 fallback path already exercised at `rag_engine.py:857`).
- **`relevance` must reproduce today's `recency_boost=False` ordering exactly** (EARS-3) — i.e. the post-RRF order with no re-rank.
- **Timestamp-to-age must be timezone-safe** (consensus: GEMINI/DEEPSEEK/CODEX): Milvus timestamps are ISO-8601 `VARCHAR` strings with *mixed* representation — some tz-aware (from `provider_adapters.normalize_timestamp()`), some naive. A naive `now - timestamp` raises `TypeError` on aware/naive mixing. The `recency_score` / `days_old` computation must normalize to a common tz before subtraction, and clamp `days_old = max(0.0, …)` so clock skew or future-dated logs can't earn an inflated boost. This is a parsing-safety constraint, not just a decay-formula choice.

## Open Questions

_Carried from requirements, now sharpened by code findings. These are decisions for `approach.md`, not blockers on discovery._

- [ ] **`semantic_score` source (EARS-1, EARS-7).** Finding: the unified cross-origin signal is `_rrf_score` (spans vector + FTS-only rows), but `search()` currently throws it away at :787; raw cosine `1 - distance` only exists for vector hits and is fake (1.0) for FTS-only rows. → Approach must decide: **preserve and min-max-normalize `_rrf_score`** (clean, covers all rows) vs keep cosine and special-case FTS rows. Discovery's read strongly favors normalized `_rrf_score`. **Caveat (verified):** `_rrf_score` is only attached when both engines return hits — the single-engine bypass at `rag_engine.py:773-779` skips `rrf_merge`, so any `_rrf_score`-based definition must also cover those paths (route through `rrf_merge` or attach a fallback).
- [ ] **Missing-timestamp fallback value (EARS-8).** Current code uses `0.5` (neutral). Keep `0.5`, or switch to `0.0` (treat as oldest)? One-line decision for approach.
- [ ] **Fate of the `recency_boost` kwarg.** Replace with `sort_by` end-to-end (low blast radius — 3 sites, no tests), or retain `recency_boost` as a thin internal alias for programmatic callers? Approach call.

## Discovery Summary

The work lands almost entirely in `rag_engine.py` (`search()` orchestration + a rewritten `_apply_recency_boost`/new scorer), with surface changes in `tools.py` (two schemas, two dispatch sites, two descriptions) and a new `LEGAL_SORT_BY` set mirroring `provider_adapters.py`. The genuinely tricky part is defining `semantic_score` over a mixed vector/FTS pool where FTS rows carry a fake `distance = 0.0`; the latent fix is to stop discarding `rrf_merge`'s `_rrf_score` and normalize it. Easy wins: env-var tunables (reuse `_env_int`, add an `_env_float`), two-layer `sort_by` validation (established pattern), and a clean test slate since recency is currently untested.

---

> **Phase complete?** Existing code mapped. Prior decisions surfaced. Open questions resolved. Then advance: `/sketch-approach SESF-24`.

## Review Archive — discovery (2026-05-28)

_Reviewer marks from `/sketch-review`, preserved verbatim. Reconciled 2026-05-28._

### Resolution Summary

- **Accepted (4):** added `tools.py:38 format_results()` row to Existing Code table (consensus CODEX/DEEPSEEK); augmented `rrf_merge()` row + Open Question #1 with the single-engine `_rrf_score` bypass at `rag_engine.py:773-779` (consensus GEMINI/DEEPSEEK); added timezone-safe timestamp-parsing + `days_old` clamp Constraint (consensus GEMINI/DEEPSEEK/CODEX); softened the `embedding_control.py:26` row so invalid `decay_days` handling is an explicit approach decision rather than an assumed `_env_int(minimum=1)` (CODEX, verified against EARS-9/10/11).
- **Rejected / no body change:** GEMINI inline recommendations on Open Questions #1/#2/#3 — deferred to `approach.md` by design (discovery does not decide open questions); their factual content (single-engine bypass, `days_old` clamp) was folded into the accepted edits. All "Verified" confirmation blocks and "Unverified assumptions" lists confirm discovery accuracy / carry no factual error — retained here as audit trail only.
- **Resolved with your input (0).**

### GEMINI — Existing Code Review Comments

> **`rag_engine.py:660` / `search()`**:
> GEMINI: Verified. `search()` is defined at :660. The candidate pool fetches `fetch_n = n*3` at :691. `rrf_merge` is called at :775. The `_rrf_score` is popped at :787. The `_apply_recency_boost` call is at :794. Provider and source_kind validation raises `ValueError` at :679-688. This matches the target locations and serves as the pattern for `sort_by` validation.

> **`rag_engine.py:811` / `_apply_recency_boost()`**:
> GEMINI: Verified. `_apply_recency_boost` is defined at :811. The hardcoded weight of `0.3` is at :817. Relative rank-local recency position is calculated at :853 based on sorting `valid_ts`. Similarity `1 - distance` is at :843. Missing timestamps get `0.5` fallback at :857. Early return when valid timestamps count `< 2` is at :836-837.

> **`rag_engine.py:1258` / `search_async()`**:
> GEMINI: Verified. `search_async` is defined at :1258, carries `recency_boost` parameter at :1260, and passes it to `search` at :1282. Note: there are no other active callers of `search_async` in the codebase, but the signature must be updated in lockstep.

> **`tools.py:296` / `search_session` dispatch**:
> GEMINI: Verified. `search_session` handler is at :296, calling `rag_engine.search(..., recency_boost=True)` at :303. The tool's schema at :195-213 lacks `sort_by`, and the description at :190-194 claims "boosted by recency".

> **`tools.py:308` / `search_all_sessions` dispatch**:
> GEMINI: Verified. `search_all_sessions` handler is at :308, calling `rag_engine.search` at :338. Schema helper `build_search_all_sessions_schema` is at :141. Explicit `TextContent` validation error returns are at :320-331. The description at :217-220 needs updating to reflect the new `hybrid` default.

> **`fts_hybrid.py:314` / FTS row construction**:
> GEMINI: Verified. `fts_hybrid.py:314` constructs FTS results with `"distance": 0.0`. This treats keyword-only matches as having maximum similarity (`1.0`) when applying the linear recency boost.

> **`fts_hybrid.py:347` / `rrf_merge()`**:
> GEMINI: Verified. `rrf_merge` is defined at :347, scoring items at :360 and :366, and returning `_rrf_score` at :375. It is popped at `rag_engine.py:787` before recency. Note: if the approach relies on `_rrf_score` for `semantic_score`, it must account for the fact that `rag_engine.search` currently bypasses `rrf_merge` when only one engine (vector or FTS) returns hits.

> **`provider_adapters.py:16` / `LEGAL_PROVIDERS` / `LEGAL_SOURCE_KINDS`**:
> GEMINI: Verified. The `LEGAL_PROVIDERS` and `LEGAL_SOURCE_KINDS` frozensets are at :16 and :25 in `provider_adapters.py`. A local or shared `LEGAL_SORT_BY` frozenset should follow this design.

> **`embedding_control.py:26` / `_env_int` / `_env_bool`**:
> GEMINI: Verified. `_env_int` is at :26 and `_env_bool` is at :36 in `embedding_control.py`. A new `_env_float` helper must be implemented to support the `SESSIONFLOW_RECENCY_WEIGHT` environment tunable.

> **`tests/test_search_provider_metadata.py:70` / Validation test pattern**:
> GEMINI: Verified. The validation error tests are at :70 and :73 in `tests/test_search_provider_metadata.py`. Since there are no other tests for search ranking or recency boost in `tests/`, Cohort B will need to build the test harness for `sort_by` strategies from scratch.

### GEMINI Review (2026-05-28)

##### Verified
- Checked `/Volumes/DATA/GitHub/SessionFlow/rag_engine.py` search orchestration (`search()` at L660, async wrapper `search_async()` at L1258) and existing rank-local recency boost (`_apply_recency_boost()` at L811).
- Checked `/Volumes/DATA/GitHub/SessionFlow/tools.py` dispatching of `search_session` L296 and `search_all_sessions` L308, as well as the input schemas (L141 and L195).
- Checked `/Volumes/DATA/GitHub/SessionFlow/fts_hybrid.py` FTS search L300 (distance assignment at L314) and `rrf_merge()` L347.
- Checked `/Volumes/DATA/GitHub/SessionFlow/provider_adapters.py` frozenset definitions L16-35 and `embedding_control.py` env-var helper `_env_int()` L26.
- Checked `/Volumes/DATA/GitHub/SessionFlow/tests/test_search_provider_metadata.py` L53-74 to confirm no existing tests cover search ranking/recency boost.
- Verified that `search_async` has no external callers and is only defined/implemented locally, and that `recency_boost` has a very tight 3-site blast radius.

#### Top concerns
1. **Single-Engine Bypass of `_rrf_score`**: If we normalize `_rrf_score` for `semantic_score`, we must address the fact that `rag_engine.search()` currently bypasses `rrf_merge()` when only one search engine (FTS or vector) returns results. The approach must either route all searches through `rrf_merge` (which handles empty lists natively) or manually attach `_rrf_score` in the single-engine shortcuts.
2. **Timezone Mismatches and Future-Dated Logs**: Result timestamps are ISO 8601 strings (some with timezone offsets, some naive). Comparing them to `datetime.now(timezone.utc)` can trigger `TypeError` (naive vs aware subtraction). Subtraction must be timezone-safe, and `days_old` must be clamped to `max(0.0, ...)` to prevent future-dated logs from achieving infinite recency boosts due to clock skew.
3. **`decay_days` validation**: The environment variable `SESSIONFLOW_RECENCY_DECAY_DAYS` must be validated to be strictly positive (e.g. `minimum=1`) to avoid `ZeroDivisionError` (if 0) or inverted/decay-reversing behavior (if negative).

#### Unverified assumptions
- None. The codebase has been thoroughly reviewed and all referenced symbols were verified to be in-tree.

### DEEPSEEK Review (2026-05-28)

#### Verified
- All existing code table entries confirmed against live source:
  - `rag_engine.py:660` — `search()` signature, validation at :679-688, `fetch_n` at :691, RRF merge at :773-779, `_rrf_score` pop at :787, `_apply_recency_boost` call at :793-794.
  - `rag_engine.py:811` — `_apply_recency_boost()` hardcoded `0.3` at :817, rank-local position recency at :847-857, `similarity = 1 - distance` at :843, missing-timestamp fallback `0.5` at :857, `< 2` valid timestamps early-return at :836.
  - `rag_engine.py:1258` — `search_async()` with `recency_boost` at :1260, threaded through to `search()` at :1282.
  - `tools.py:296` — `search_session` dispatches with `recency_boost=True` at :303, schema lacks `sort_by` at :195-213, description at :190-194.
  - `tools.py:308` — `search_all_sessions` dispatches with `recency_boost=False` at :338, uses `build_search_all_sessions_schema()` at :141, TextContent error returns at :320-331, description at :217-220.
  - `fts_hybrid.py:314` — FTS rows stamped `distance = 0.0`.
  - `fts_hybrid.py:347` — `rrf_merge()` with k=60, produces `_rrf_score`.
  - `provider_adapters.py:16` — `LEGAL_PROVIDERS` and `LEGAL_SOURCE_KINDS` frozensets.
  - `embedding_control.py:26` — `_env_int()` and `_env_bool()` at :36; no `_env_float` exists (grep confirmed).
  - `tests/test_search_provider_metadata.py:69-73` — validation error test pattern.
- Confirmed `recency_boost` blast radius: 2 MCP call sites + 1 async wrapper, zero test references (grep for `recency_boost|_apply_recency_boost` across workspace).
- Confirmed `search_async` has no external callers (grep finds only definition and self-referencing error message).
- Confirmed `_env_float` does not exist anywhere in the codebase.
- Confirmed `rrf_merge` is bypassed at `rag_engine.py:776-779` when only one engine returns results — single-engine rows have no `_rrf_score`.

#### Top concerns
1. **Single-engine `_rrf_score` gap is under-documented.** The existing code table notes that `_rrf_score` is popped and RRF merge produces it, but does not call out that the single-engine bypass at :776-779 skips `rrf_merge` entirely. Any `semantic_score` definition built on `_rrf_score` must account for these code paths — either route all searches through `rrf_merge` or define a fallback `_rrf_score` for single-engine rows. This affects the resolution of Open Question #1.
2. **Timestamp-to-age computation is an open risk, not just a formula question.** The timestamps stored in Milvus are VARCHAR ISO-8601 strings with mixed timezone representation (some utc-aware from `provider_adapters.normalize_timestamp()`, some naive). Computing `days_old = now - timestamp` requires consistent timezone handling to avoid `TypeError` on naive-vs-aware subtraction. The discovery correctly identifies decay math as "standard," but the datetime parsing safety is a distinct concern not yet surfaced in any open question.
3. **`format_results()` in `tools.py` is missing from the existing code table.** Every search result passes through `format_results()` before reaching the MCP caller. If the approach adds or retains internal fields like `_rrf_score` or `_score` on result dicts, those fields must be stripped before formatting — or `format_results()` must be taught to ignore them. A minor but mandatory touch-point that the tasks phase will need.

#### Unverified assumptions
- That ISO-8601 timestamp strings from Milvus can be parsed without timezone normalization — providers may inject naive vs aware strings from different adapters.
- That `_rrf_score` will be present on all rows after implementation — single-engine paths currently lack it.
- That `format_results()` needs no changes to handle new internal fields (`_rrf_score`, `_score`, etc.).
- That `SESSIONFLOW_RECENCY_DECAY_DAYS` as an int is sufficient — the division `days_old / decay_days` with int denominator works in Python 3 (float result), but `sessionflow_server.sh` and `cleanup.py` env-var plumbing may pass strings that need parsing.
- That `_env_float()` can be added to `embedding_control.py` without changing other callers — existing `_env_int` and `_env_bool` have no shared mutex or side effects, and the file has no tests; this should be safe but is unverified.

### CODEX Review (2026-05-28)

#### Verified

- Read `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, the SESF-24 `requirements.md`, and the current `discovery.md`; no `.context/GLOSSARY.md` or project-local sketch conventions file exists in the repo.
- Checked live search orchestration in `/Volumes/DATA/GitHub/SessionFlow/rag_engine.py`: provider/source-kind validation, `fetch_n = n * 3`, vector + FTS candidate generation, RRF merge/bypass behavior, `_rrf_score` cleanup, existing `recency_boost`, and `search_async` forwarding (`rag_engine.py:660-808`, `rag_engine.py:811-867`, `rag_engine.py:1258-1285`).
- Checked MCP surfaces in `/Volumes/DATA/GitHub/SessionFlow/tools.py`: schemas/descriptions, `search_session`, `search_all_sessions`, explicit `TextContent` validation for provider/source_kind, global error wrapping, and `format_results()` display scoring (`tools.py:38-80`, `tools.py:141-221`, `tools.py:296-345`, `tools.py:395-396`).
- Checked `/Volumes/DATA/GitHub/SessionFlow/fts_hybrid.py` for FTS `distance = 0.0` rows and `_rrf_score` production (`fts_hybrid.py:250-322`, `fts_hybrid.py:347-376`), `/Volumes/DATA/GitHub/SessionFlow/provider_adapters.py` for legal-value sets and timestamp normalization (`provider_adapters.py:16-34`, `provider_adapters.py:82-104`), and `/Volumes/DATA/GitHub/SessionFlow/embedding_control.py` for env-var helpers (`embedding_control.py:26-40`).
- Confirmed via grep that `recency_boost` is limited to `rag_engine.py` and the two `tools.py` call sites, `_env_float` does not exist, and the current tests only cover provider/source-kind search validation and result formatting metadata, not recency or `sort_by`.

#### Top concerns

1. **Invalid decay-days behavior is under-specified.** Discovery currently assumes `_env_int(..., minimum=1)` for `SESSIONFLOW_RECENCY_DECAY_DAYS`, but requirements only define invalid fallback behavior for `SESSIONFLOW_RECENCY_WEIGHT`. Approach needs to choose clamp-vs-default explicitly so tests do not encode an accidental contract.
2. **`format_results()` is a user-visible ranking surface.** It renders `Relevance` as `1 - distance`, which is already fake for FTS-only rows and will not necessarily match normalized RRF or hybrid score ordering. Discovery should carry this into approach/tasks, not only mention internal-field leakage.
3. **Existing GEMINI/DEEPSEEK concerns are valid.** The single-engine `_rrf_score` gap, timezone-safe timestamp parsing, and future-dated timestamp clamping are real constraints verified against live code.

#### Unverified assumptions

- I did not verify the Plane issue body directly; this review used the source issue excerpt embedded in `requirements.md`.
- I did not verify the mem0 prior-decision search noted in discovery; no local memory hit for SESF-24 was found during this review.
- I did not run live Milvus queries; findings are based on static source verification and test discovery.
