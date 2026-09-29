---
sketch: "SESF-24-recency-ranking"
phase: approach
created: 2026-05-28
---

# SESF-24 — Approach

_How we'll build it. **Don't write code or tests** — the tasks phase produces the work plan._

## High-Level Architecture

The work is a query-time re-rank entirely inside `rag_engine.search()`. Candidate
generation (vector + FTS5, `fetch_n = n*3`) is unchanged. The single behavioral seam
moves one step earlier: instead of conditionally calling the rank-local
`_apply_recency_boost`, `search()` will always carry a cross-origin relevance signal
(`_rrf_score`) through the pool and hand it to a new strategy dispatcher that orders the
pool by one of three `sort_by` modes — `relevance`, `recency`, `hybrid` — before
truncating to `n`.

The crux fix is `semantic_score` over a mixed vector/FTS pool. FTS-only rows carry a
fake `distance = 0.0` (→ similarity 1.0), so cosine can't be the semantic basis. The
only signal that spans both origins is RRF's `_rrf_score`, which today is discarded at
`rag_engine.py:787` and is *absent entirely* on the single-engine bypass paths
(`rag_engine.py:773-779`). The approach routes **every** search through `rrf_merge`
(it handles an empty list natively and is rank-order-preserving, so single-engine
`relevance` ordering is unchanged), keeps `_rrf_score` alive until after scoring,
min-max-normalizes it into `[0,1]` as `semantic_score`, and strips all internal score
fields before results reach `format_results`.

Recency becomes a true age-aware decay (`exp(-days_old / decay_days)`) computed against
`now` in UTC, with timezone-safe parsing of the mixed naive/aware ISO-8601 timestamps
and a `days_old = max(0.0, …)` clamp so clock-skewed future logs can't earn an inflated
boost. Tunables (`SESSIONFLOW_RECENCY_WEIGHT`, `SESSIONFLOW_RECENCY_DECAY_DAYS`) read
through `embedding_control.py`'s env-helper family. The MCP surface in `tools.py` gains
a `sort_by` parameter on both search tools, two-layer validation mirroring the existing
`provider`/`source_kind` pattern, and a new `hybrid` default.

## Key Decisions

| # | Decision | Rationale | Tradeoff |
|---|----------|-----------|----------|
| D-1 | **`semantic_score` = min-max-normalized `_rrf_score`** over the candidate pool; route *all* searches through `rrf_merge` so every row carries `_rrf_score` (eliminate the single-engine bypass at `rag_engine.py:773-779`). | `_rrf_score` is the only relevance signal spanning vector + FTS-only rows; cosine is fake (1.0) for FTS rows. Resolves discovery Open Question #1. | Discards raw cosine distance as the semantic basis; degenerate pools (1 row or all-equal scores) need a div-by-zero guard (see Risks). |
| D-2 | **Replace `recency_boost: bool` with `sort_by: str` end-to-end** in `search()` and `search_async()`. Do not retain a `recency_boost` alias. | Single explicit knob; avoids two coexisting ranking flags. Blast radius is 3 sites + 0 tests (verified). Resolves discovery Open Question #3. | Breaks any *out-of-tree* programmatic caller passing `recency_boost=`; none exist in-repo. |
| D-3 | **`recency_score = exp(-days_old / decay_days)`** from true age vs `now` (UTC), replacing the rank-local linear `_apply_recency_boost`. | EARS-6: age-aware decay, not position-within-set. A `decay_days`-old result scores ≈0.367. | Requires robust ISO-8601 parsing of mixed naive/aware strings (timezone-safety constraint). |
| D-4 | **Blend:** `final = (1 - w) * semantic_score + w * recency_score`, both terms in `[0,1]`. | EARS-1/EARS-7: neither term dominates by scale. `recency_score` is already `(0,1]` by construction; `semantic_score` min-max'd to `[0,1]`. | Linear blend; a weight near 0 or 1 effectively collapses to one mode (acceptable — that's the point). |
| D-5 | **Env tunables:** add `_env_float(name, default, minimum, maximum)` to `embedding_control.py` for `SESSIONFLOW_RECENCY_WEIGHT` (default 0.3) with **fallback-to-default on unparseable or out-of-`[0,1]`** (EARS-10). Use existing `_env_int(SESSIONFLOW_RECENCY_DECAY_DAYS, default=7, minimum=1)` for `decay_days`. | EARS-9/10 mandate fallback semantics for weight; reuse `_env_int` for the int decay. `minimum=1` makes `ZeroDivisionError`/negative-decay impossible. | **Deliberate divergence:** weight *falls back* on bad input, decay *clamps* to 1. Documented intentionally (discovery flagged decay's invalid-input contract as an approach call). |
| D-6 | **Missing-timestamp fallback `recency_score = 0.5`** (neutral), retained from current code. | EARS-8: deterministic, doesn't bury an otherwise-relevant hit by treating a parse gap as "oldest". | A genuinely old but unparseable row gets an undeserved neutral boost. |
| D-7 | **`LEGAL_SORT_BY = frozenset({"relevance","recency","hybrid"})`** in `provider_adapters.py`; validate in **both** layers — `raise ValueError` in `rag_engine.search` (EARS-5), `return TextContent` error in both `tools.py` handlers. | Mirrors the established `LEGAL_PROVIDERS`/`LEGAL_SOURCE_KINDS` two-layer pattern. | Validation logic duplicated across engine + tool layers (existing, accepted pattern). |
| D-8 | **`recency` mode re-ranks the existing pool by timestamp descending** (EARS-2), not a chronological feed; missing-timestamp rows sort last deterministically. | Non-Goal: candidate generation is frozen — `recency` is a re-rank of the merged pool, not a new query. | A pure-recency caller still only sees rows that matched the query; not a global "newest N sessions" feed. |
| D-9 | **Strip internal score fields** (`_rrf_score`, `_score`, any scratch keys) before results reach `format_results`; leave the visible `Relevance` display (`1 - distance`) unchanged this round. | Prevents internal fields leaking into MCP text output. Changing the display is scope creep (Non-Goal on the FTS `distance` surface). | Displayed `Relevance` can disagree with `hybrid` ordering (already true today). Flagged as follow-up + risk. |

## File Map

_Every file this sketch will create, modify, or delete. Tasks.md will reference these paths._

### New
- `tests/test_recency_ranking.py` — coverage for `sort_by` validation (both layers), `hybrid` ordering (newer-above-older for same topic), `recency` re-rank, decay math, `_env_float` weight parsing/fallback, `decay_days` clamp, timezone-safe age parsing (naive + aware + future-dated), and missing-timestamp `0.5` fallback. **`relevance`-parity is an explicit obligation across all three candidate-set shapes** — vector-only, FTS-only, *and* mixed vector+FTS — because D-1 routes single-engine searches through `rrf_merge`, changing the bypass shape (`rag_engine.py:776-779`) from raw results to copied/deduped RRF results; parity tests must guard that single-engine relevance ordering is unchanged. **Tool-layer validation tests must assert the direct `Invalid sort_by: ... expected one of ...` `TextContent` branch** for *both* `search_session` and `search_all_sessions` — not merely "any error text" — since the global wrapper catches uncaught exceptions as `Error executing {name}: ...` (`tools.py:395-396`) and could mask a missing handler-level check (the intended two-layer validation in D-7).

### Modified
- `rag_engine.py` — `search()` + `search_async()`: replace `recency_boost: bool` param with `sort_by: str` (default `"hybrid"`); add `sort_by` validation against `LEGAL_SORT_BY`; route all candidate sets through `rrf_merge`; defer the `_rrf_score` pop until after scoring; replace `_apply_recency_boost` with a new strategy dispatcher + age-aware `recency_score` / min-max `semantic_score` scorer; read env tunables.
- `tools.py` — add `sort_by` to the inline `search_session` schema (`:195-213`) and to `build_search_all_sessions_schema()`; add `LEGAL_SORT_BY` two-layer validation in both `search_session` and `search_all_sessions` handlers; pass `sort_by=` (default `"hybrid"`) instead of `recency_boost=`; update the two tool descriptions (`:190-194`, `:217-220`) for the `hybrid` default; ensure internal score fields are stripped before `format_results`.
- `embedding_control.py` — add `_env_float(name, default, minimum=None, maximum=None)` with fallback-to-default on unparseable/out-of-range (distinct from `_env_int`'s clamp).
- `provider_adapters.py` — add `LEGAL_SORT_BY` frozenset alongside `LEGAL_PROVIDERS` / `LEGAL_SOURCE_KINDS`.

### Deleted
- None. `_apply_recency_boost` is rewritten in place (renamed if the dispatcher shape warrants), not removed as a separate file.

## Data / Schema Changes

None — no schema, migration, or persisted-data changes. This is a query-time ranking change only (requirements Non-Goal: ingestion / indexing / backfill untouched).

## Tradeoffs Surfaced for Review

- **D-2 removes `recency_boost` outright** rather than aliasing it. Low risk in-tree (3 sites, 0 tests), but it is a breaking change to the internal `search()` signature. Flagging in case any external automation imports `rag_engine.search` directly.
- **D-5's weight-vs-decay asymmetry** (fallback vs clamp) is intentional but is a subtle contract; tests will pin both behaviors so they don't drift.
- **D-9 leaves the visible `Relevance` number as `1 - distance`**, which is already misleading for FTS-only rows and won't match `hybrid` ordering. Honest display is deferred to a follow-up rather than smuggled into this sketch.

## UI Contract

N/A — no UI surface. This is an MCP / back-end ranking change; `requirements.md` and `discovery.md` reference no mockup, playground, screenshot, or prototype.

## Out of Scope (follow-up issues)

- Future issue: **Honest `Relevance` display in `format_results`** — render the actual ranking score (normalized RRF / hybrid `final`) instead of `1 - distance`, which is fake for FTS-only rows. (File under SessionFlow.)
- Future issue: **Per-call `recency_weight` / `decay_days` MCP parameters** — env-only tuning this round (requirements Non-Goal); add per-query overrides later. (File under SessionFlow.)
- Future issue: **`search_async` doesn't forward `date_from` / `date_to`** — pre-existing gap noted at `rag_engine.py:1258`; unrelated to ranking. (File under SessionFlow.)

## Risk Notes

- Risk: **min-max normalization div-by-zero** on a 1-row pool or all-equal `_rrf_score` → mitigation: when the score range is 0, assign a neutral `semantic_score` (e.g. 1.0) to all rows rather than dividing.
- Risk: **mixed naive/aware ISO-8601 timestamps** cause `TypeError` on subtraction → mitigation: parse to UTC-aware before computing `days_old`; treat unparseable as the EARS-8 fallback path; clamp `days_old = max(0.0, …)`.
- Risk: **`hybrid` default changes `search_all_sessions` output for all existing callers** (intended per EARS-13) → mitigation: covered by a `relevance`-parity test so callers can opt back to the old ordering explicitly; call out the default change in the tool description.

---

> **Phase complete?** Architecture clear, decisions logged with rationale, file map complete. Then advance: `/sketch-tasks SESF-24`.

## Review Archive — approach (2026-05-28)

### Resolution Summary

- **Accepted: 2** — both CODEX "Top concerns" folded into the File Map `tests/test_recency_ranking.py` entry: (1) explicit `relevance`-parity obligation across vector-only / FTS-only / mixed candidate-set shapes; (2) tool-layer tests must assert the direct `Invalid sort_by: …` `TextContent` branch for both handlers, not the global error wrapper. Both code claims (single-engine bypass at `rag_engine.py:776-779`, global wrapper at `tools.py:395-396`) verified against the live repo.
- **Rejected: 0.**
- **Resolved with your input: 0** — no judgment calls; both were mechanical test-obligation refinements.

---

## CODEX Review (2026-05-28)

### Verified

- Read `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `requirements.md`, `discovery.md`, and `approach.md`; no `.context/GLOSSARY.md` or project-local sketch conventions file exists in the repo.
- Checked live search orchestration in `/Volumes/DATA/GitHub/SessionFlow/rag_engine.py`: `search()` validation, `fetch_n`, vector + FTS candidate generation, current single-engine bypass, `_rrf_score` cleanup, existing `recency_boost`, and `search_async` forwarding (`rag_engine.py:660-808`, `rag_engine.py:811-867`, `rag_engine.py:1258-1285`).
- Checked `/Volumes/DATA/GitHub/SessionFlow/fts_hybrid.py` for FTS `distance = 0.0` rows and `rrf_merge()` score/order behavior (`fts_hybrid.py:250-322`, `fts_hybrid.py:347-376`).
- Checked MCP schema/dispatch/error behavior in `/Volumes/DATA/GitHub/SessionFlow/tools.py`, including `format_results()`, `search_session`, `search_all_sessions`, and the global error wrapper (`tools.py:38-80`, `tools.py:141-221`, `tools.py:296-345`, `tools.py:395-396`).
- Checked `/Volumes/DATA/GitHub/SessionFlow/provider_adapters.py` for legal-value sets and timestamp normalization (`provider_adapters.py:16-34`, `provider_adapters.py:82-104`), `/Volumes/DATA/GitHub/SessionFlow/embedding_control.py` for env helpers (`embedding_control.py:26-40`), and current search/provider test patterns (`tests/test_search_provider_metadata.py:43-73`, `tests/conftest.py:210-242`).

### Top concerns

1. The approach correctly routes all candidate sets through `rrf_merge`, but tasks need explicit `sort_by="relevance"` parity coverage for vector-only and FTS-only paths because this changes the current single-engine bypass shape from raw results to copied/deduped RRF results.
2. Tool-layer validation tests need to assert the intended direct `TextContent` validation branch for both search tools; otherwise the global `Error executing ...` wrapper can mask missing handler-level `sort_by` validation.

### Unverified assumptions

- I did not verify the Plane issue body directly; this review used the source issue excerpt embedded in `requirements.md`.
- I did not run live Milvus or MCP queries; findings are based on static source verification and current test/fixture inspection.
