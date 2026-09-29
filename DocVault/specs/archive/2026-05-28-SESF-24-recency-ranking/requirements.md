---
sketch: "SESF-24-recency-ranking"
phase: requirements
created: 2026-05-28
---

# SESF-24 — Requirements

> **Source Issue:** [SESF-24](https://plane.lbruton.cc/lbruton/browse/SESF-24/)
> **Title:** Hybrid recency-aware ranking + sort_by parameter for search endpoints
>
> **Problem:** Semantic search currently ranks purely by vector cosine similarity. Older sessions with high keyword overlap consistently outrank newer, more relevant sessions. The audit found that without a `date_from` filter, older hits rank above newer ones — a recency bias that hurts session-start context retrieval.
>
> **Solution:** Add a hybrid scoring mode that blends semantic similarity with temporal recency:
> - Hybrid score formula: `final = (1 - recency_weight) * semantic_score + recency_weight * recency_score`
> - Default recency_weight: 0.3 (tunable via env var `SESSIONFLOW_RECENCY_WEIGHT`)
> - Recency score: exponential decay from now, e.g. `exp(-days_old / half_life)` with configurable half-life (default 7 days)
>
> Add a `sort_by` parameter to `search_all_sessions` and `search_session`:
> - `relevance` — pure semantic (current behavior)
> - `recency` — pure timestamp sort (newest first)
> - `hybrid` — blended score (new default)
>
> **Acceptance (from issue):**
> - `search_all_sessions(query="sketch review", sort_by="hybrid")` returns today's sessions above last week's for the same topic
> - `sort_by="recency"` returns a pure chronological feed regardless of query match quality
> - Existing `sort_by="relevance"` behavior unchanged
> - Default is `hybrid` so existing callers get improved results without code changes
>
> **Related:** Audit SF-3, SF-7. Child of Search Intelligence epic (SESF-23).

## Overview

SessionFlow ranks search results by vector cosine similarity merged with FTS5 keyword
hits (RRF), with an optional rank-based recency nudge on `search_session` only. The
nudge is linear *within the returned set* (newest hit = 1.0, oldest = 0.0), so an
all-old result set still gets a "freshest" item promoted and `search_all_sessions`
gets no recency signal at all — older high-keyword-overlap sessions outrank newer,
more relevant ones at session start. SESF-24 replaces the boolean nudge with an
explicit, caller-selectable `sort_by` strategy (`relevance` / `recency` / `hybrid`)
and a true age-aware recency score (exponential decay from *now*), blended into the
final score via a tunable weight. `hybrid` becomes the default for both search tools,
so existing callers get fresher, more relevant results without code changes.

## User Stories

> Format: **As a** [role], **I want** [capability], **so that** [outcome].

- **US-1:** As a developer reorienting at session start, I want recent sessions to rank above older ones for the same topic, so that the first hits reflect what I was just working on rather than stale high-keyword-overlap matches.
- **US-2:** As an MCP caller, I want to pick the ranking strategy per query (relevance / recency / hybrid), so that I can get a pure chronological feed or pure relevance when the default blend isn't what I need.
- **US-3:** As an operator, I want to tune how strongly recency influences ranking without editing code, so that I can rebalance for my corpus's age distribution.

## Acceptance Criteria

> EARS normative statements. Each line is individually testable (→ TDD Cohort B).
> System under test = **the SessionFlow search engine** (`rag_engine.search`) and its two MCP tools.

### sort_by strategy (maps to US-1, US-2)
- **EARS-1** — WHERE `sort_by` is `hybrid`, the search engine SHALL rank results by a blended score `final = (1 - recency_weight) * semantic_score + recency_weight * recency_score`. The precise source of `semantic_score` over the mixed vector + FTS candidate pool is deferred to discovery/approach (see Open Question #1).
- **EARS-2** — WHERE `sort_by` is `recency`, the search engine SHALL order the **existing query-generated candidate pool** by timestamp descending (newest first), independent of semantic match quality. This is a re-rank of the vector + FTS candidate set, not a chronological feed over all scoped records (candidate generation is unchanged — see Non-Goals).
- **EARS-3** — WHERE `sort_by` is `relevance`, the search engine SHALL rank results by the current RRF relevance ordering with **no** recency re-rank, producing ordering identical to the pre-SESF-24 `recency_boost=False` path.
- **EARS-4** — IF `sort_by` is omitted, THEN the search engine SHALL default to `hybrid`.
- **EARS-5** — IF `sort_by` is any value other than `relevance`, `recency`, or `hybrid`, THEN the search engine SHALL reject the request with a validation error naming the allowed values (consistent with existing `provider` / `source_kind` validation).

### Recency score (maps to US-1)
- **EARS-6** — The search engine SHALL compute `recency_score` for a result as `exp(-days_old / decay_days)`, where `days_old` is the result timestamp's age relative to the current time and `decay_days` (the decay constant) defaults to 7. Note: this is an exponential decay constant, not a half-life — a result exactly `decay_days` old scores ≈0.367.
- **EARS-7** — WHILE computing a `hybrid` score, the search engine SHALL normalize both `semantic_score` and `recency_score` into the range `[0, 1]` before blending, so neither term dominates by scale.
- **EARS-8** — IF a result has no parseable timestamp, THEN the search engine SHALL assign it a deterministic fallback `recency_score` (see Open Questions) rather than raising or skipping the result.

### Tunables (maps to US-3)
- **EARS-9** — WHERE `SESSIONFLOW_RECENCY_WEIGHT` is set to a value in `[0, 1]`, the search engine SHALL use it as `recency_weight`; otherwise `recency_weight` SHALL default to `0.3`.
- **EARS-10** — IF `SESSIONFLOW_RECENCY_WEIGHT` is unparseable or outside `[0, 1]`, THEN the search engine SHALL silently fall back to the default `0.3`.
- **EARS-11** — WHERE `SESSIONFLOW_RECENCY_DECAY_DAYS` is set, the search engine SHALL use it as `decay_days`; otherwise `decay_days` SHALL default to `7`.

### MCP surface (maps to US-2)
- **EARS-12** — The `search_all_sessions` and `search_session` MCP tools SHALL each expose a `sort_by` parameter accepting `relevance`, `recency`, or `hybrid`.
- **EARS-13** — WHEN `search_all_sessions` or `search_session` is invoked without `sort_by`, the tool SHALL apply `hybrid` ranking (changing `search_all_sessions`' prior pure-relevance default).

## Non-Goals

_Explicit list of things this sketch does NOT do._

- **Not changing candidate generation** — the vector + FTS5 RRF merge that produces the candidate pool is unchanged; hybrid scoring is a re-rank on top of the existing merged set.
- **Not adding per-call `recency_weight` / `half_life` MCP parameters** — tuning is env-var only this round; per-query override is a possible follow-up.
- **Not touching ingestion, indexing, or backfill** — this is purely a query-time ranking change.
- **Not fixing the pre-existing FTS-only `distance` absence** beyond whatever the EARS-7 normalization requires — flagged in Open Questions, not a goal to fully resolve here.
- **Not introducing hosted embeddings or new providers** — out of scope (SESF-6 keeps embedding local).

## Open Questions

_Items discovery/approach must resolve. None block writing requirements._

- [ ] **`semantic_score` source for the blend (EARS-1, EARS-7):** raw cosine similarity (`1 - distance`) vs min-max-normalized RRF score? FTS-only results have no cosine `distance` after RRF merge, so a uniform normalization across mixed-origin candidates is needed.
- [ ] **Missing-timestamp fallback (EARS-8):** `recency_score = 0` (treat as oldest) vs `0.5` (neutral)? Current code uses `0.5`.
- [ ] **Legacy `recency_boost` kwarg:** does the existing internal `rag_engine.search(recency_boost: bool)` get removed/superseded by `sort_by`, or retained for programmatic callers?

---

> **Phase complete?** Acceptance criteria are concrete and verifiable. Then advance: `/sketch-discovery SESF-24`.

## Review Archive — requirements (2026-05-28)

### Resolution Summary

- **3 accepted:** EARS-3 (corrected "semantic only" → current RRF relevance ordering, no recency re-rank — verified against `rag_engine.py:775`), EARS-2 (clarified `recency` re-ranks the existing candidate pool, not a feed), EARS-1 (added pointer to Open Question #1 for the deferred `semantic_score` source).
- **0 rejected.**
- **2 resolved with operator input:** EARS-6 (renamed tunable to decay constant `decay_days` / `SESSIONFLOW_RECENCY_DECAY_DAYS`, dropped the inaccurate "half-life" term — propagated to EARS-11 and removed Open Question #4); EARS-10 (locked in silent fallback to `0.3` for invalid weight — removed Open Question #3).

---

### CODEX Review (2026-05-28)

#### Verified

- Read the shared sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, and `.context/GLOSSARY.md`.
- Reviewed `requirements.md` for SESF-24 only; no later phase was advanced.
- Verified live search behavior in `/Volumes/DATA/GitHub/SessionFlow/rag_engine.py`: `search()` builds vector and FTS candidate pools, merges them with RRF, strips `_rrf_score`, and only applies the existing rank-local `recency_boost` when requested (`rag_engine.py:660-808`, `rag_engine.py:811-867`).
- Verified the MCP surfaces in `/Volumes/DATA/GitHub/SessionFlow/tools.py`: `search_session` currently calls `rag_engine.search(..., recency_boost=True)` and `search_all_sessions` calls `rag_engine.search(..., recency_boost=False)` with provider/source-kind validation (`tools.py:141-180`, `tools.py:185-345`).
- Verified FTS rows and RRF behavior in `/Volumes/DATA/GitHub/SessionFlow/fts_hybrid.py`: FTS results carry `distance = 0.0`, and RRF is currently the cross-engine relevance signal (`fts_hybrid.py:250-322`, `fts_hybrid.py:347-376`).

#### Top concerns

- `semantic_score` is not defined tightly enough for a mixed vector + FTS candidate pool, especially because FTS-only rows have no real vector distance.
- `sort_by="recency"` needs to state whether it sorts the existing query-generated candidate pool or creates a chronological feed over all scoped records.
- `sort_by="relevance"` should preserve current RRF relevance behavior, not accidentally regress to vector-only semantic ordering.
- The recency decay formula is not a half-life formula as written.
- Invalid `SESSIONFLOW_RECENCY_WEIGHT` behavior is both specified and still listed as unresolved.

#### Unverified assumptions

- I did not verify the Plane issue body directly; this review used the source issue excerpt embedded in the requirements file.
- I did not run live Milvus queries; the review is based on static source verification of the ranking and MCP tool contracts.
