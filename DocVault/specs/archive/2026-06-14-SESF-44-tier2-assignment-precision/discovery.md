---
sketch: "SESF-44-tier2-assignment-precision"
phase: discovery
created: 2026-06-14
---

# SESF-44 — Discovery

_Research the existing system and prior art. **Don't propose solutions** — that's the next phase._

> Research run **inline** (not via Workflow fan-out): the surface is a single module + its test file, so the structural/semantic angles were covered by direct reads. Prior-decision angle drawn from the in-repo SESF-41/42 artifacts and the live module docstring.

## Existing Code

The entire change is contained in one module and its test file. The detector is pure and module-private; the two public entry points (`redact`, `scan_spans`) are consumed by three call sites, none of which reach the internals being changed — so a single detector edit propagates to both the SESF-41 guard and the SESF-42 sanitizer (this is the structural basis for AC-6).

| Path | Role | Notes |
|------|------|-------|
| `secret_redaction.py:103-105` | `_SECRET_KEYWORD` alternation | **Change target (D-A).** Bare `auth` lives here: `api[_-]?key\|secret\|token\|password\|passwd\|pwd\|auth\|credential\|access[_-]?key`. |
| `secret_redaction.py:106-110` | `_ASSIGN_RE` | Wraps the keyword with `[A-Za-z0-9_\-]*` on **both** sides → bare-substring matching is the FP root cause (`author`, and `oauth`⊂`coauthor`). Value capture `[^\"'\s,}{]+`. **Change/affected by both D-A and D-B.** |
| `secret_redaction.py:111` | `_KEYWORD_RE` | Same `_SECRET_KEYWORD` reused for Tier-3 entropy adjacency — editing the keyword set also shifts Tier-3 `_keyword_adjacent`; verify no Tier-3 fixture regresses. |
| `secret_redaction.py:189-199` | `_is_placeholder_value` | **Change target (D-B).** Today guards length / placeholder-word / placeholder-shape only; the `$(…)`/`${…}` non-literal rejection lands here (or at the `_ASSIGN_RE` match level — approach decides). |
| `secret_redaction.py:284-330` | `_collect_candidates` | Tier-2 branch at `:324-328` runs `_ASSIGN_RE.finditer(line)` then `_is_placeholder_value`. The point where a non-literal/interpolation guard would be applied. |
| `secret_redaction.py:333-369` | `_aggregate_maskable_candidates` | Shared aggregation behind **both** `redact` and `_maskable_values`/`scan_spans` — the "single fix, both paths" seam (AC-6). |
| `secret_redaction.py:372-426` | `redact()` | Public path 1 (SESF-41 guard). |
| `secret_redaction.py:455-567` | `_collect_spans` / `scan_spans()` | Public path 2 (SESF-42 sanitizer); `scan_spans` reuses `_maskable_values` → same aggregation. |
| `tests/test_secret_redaction.py:94-100` | Tier-2 fixtures | `ASSIGN_SECRET_VALUE` (20-char synthetic) + `MCP_CONFIG_DUMP` leak-class fixture. |
| `tests/test_secret_redaction.py:159-188` | Tier-2 + value-guard tests | `test_ac6_tier2_assignment_masked`, `test_ac6_mcp_config_dump_leak_class`, parametrized `test_ac7_placeholder_shape_or_short_value_not_flagged` — **the suite the new fixtures extend.** |
| `tests/test_secret_redaction.py:702-712` | scan_spans Tier-2 test | `test_scan_spans_tier2_assignment_span` — pattern for asserting the SESF-42 span path (AC-7 "both paths"). |

### Public callers (integration points — not modified)

- `rag_engine.py:767` — `redact()` in the `add_turns` ingestion hook (SESF-41 guard).
- `rag_engine.py:792`, `:805` — `redact()` inside `_scrub_exception` (SESF-45 log scrub) — also benefits from the precision fix.
- `sanitize.py:314`, `:444`, `:638`, `:678` — `scan_spans()` in the SESF-42 dry-run/apply paths.

## Prior Decisions

> Queries: SessionFlow `search_all_sessions` "SESF-44 Tier-2 assignment false positive secret_redaction" (project-scoped); mem0 "SessionFlow SESF-44 SESF-45 secret redaction sanitizer"; in-repo module docstring + git log.

- **2026-06-13** — SESF-42 retroactive sanitizer shipped (PR #40, `f0a35cd`; verified via `git log` — `c642208` is SESF-45/PR #41, **not** SESF-42); its live dry-run baseline (1,296 turns) is what surfaced the 882 Tier-2 hits / ~47% FP rate driving this issue. (mem0 `d4899432…`, `4dfdd911…`)
- **SESF-41 design (module docstring `secret_redaction.py:9-27`)** — the detector is **pure / deterministic / idempotent** and emits **no raw values**. Any change here must preserve those four invariants (tested by `test_ac14_*`, `test_ac15_*`, `test_ac16_*`, `test_ac18_*`).
- **D-1 hybrid / D-2 single-masker / Tier model** — Tier-2 is the contextual `key:value` scanner; the `auth` substring FP is purely a keyword-alternation breadth issue, not a tier-architecture issue.
- **Recall guidance (mem0 `f53f6a69…`, retro 2026-06-13):** redaction features should be verified against the **live Milvus index**. A dry-run `cleanup.py sanitize --dry-run` re-run is the natural way to confirm the FP-count drop (was: 882 Tier-2 hits / 328 author* FPs).

## External References

- Python `re` (stdlib) — alternation, `(?i)`, character classes, and the substring-vs-boundary behavior at the heart of D-A. No third-party regex/lib involved; `detect-secrets` is **not** in the Tier-2 path — it provides **Tier-1 structured-provider plugins only** here (`_PLUGINS`, `secret_redaction.py:51-59`). Tier-3 entropy is the local length-gated Shannon scanner (`_entropy_tokens` `:233-238`, forced by `_keyword_adjacent` `:241-258`); detect-secrets' own entropy plugins are intentionally unused (`:46-50`).
- _No RFC / external doc needed — this is internal detector tuning._

## Constraints

- **Four detector invariants** (pure / deterministic / idempotent / no-raw-value) must hold — covered by the existing AC-14/15/16/18 tests; do not weaken them.
- **No regression on existing green suite** — current `main` collects **360 tests** (verified `venv/bin/python -m pytest --collect-only -q`, 2026-06-14; the "357" from SESF-42 close is historical, pre-SESF-45). The no-regression gate is command-based — run the full `venv/bin/python -m pytest -q`, don't lock a count. The Tier-2 true-positive tests (`test_ac6_*`) and value-guard tests (`test_ac7_*`) must stay green.
- **Synthetic fixtures only** — every new token must be assembled from parts (`"AKIA" + "…"` style) so no contiguous literal hits GitHub push-protection (`tests/test_secret_redaction.py:30-35`).
- **`_KEYWORD_RE` coupling** — `_SECRET_KEYWORD` is reused by Tier-3 adjacency; a keyword-set edit must not silently change Tier-3 entropy forcing (check `test_ac8_*`).
- **Both public paths** — per reconciled AC-7, new cases assert through `redact()` and `scan_spans()`.

## Open Questions

_None — requirements fully resolved D-A/D-B and the four CODEX findings. Approach can proceed._

## Discovery Summary

The work lands entirely in `secret_redaction.py` (keyword alternation at `:103-105`, the `_ASSIGN_RE`/`_is_placeholder_value` value path at `:106-110`/`:189-199`, Tier-2 collection at `:324-328`) plus new fixtures in `tests/test_secret_redaction.py` extending the established `test_ac6_*`/`test_ac7_*`/`test_scan_spans_*` patterns. The tricky parts are both already pinned in requirements: the **bounded-key-segment** requirement (so `oauth` doesn't re-match `coauthor`) and the **interpolation-span** detection (so `${FOO_PASSWORD:-default}` is caught even though its captured value is `-default`). Everything else is mechanical, and the shared `_aggregate_maskable_candidates` seam means one edit covers both the SESF-41 guard and SESF-42 sanitizer with no call-site changes.

---

> **Phase complete?** Existing code mapped. Prior decisions surfaced. Open questions resolved. Then advance: `/sketch-approach SESF-44`.

## Review Archive — discovery (2026-06-14)

### Resolution Summary

- **Accepted (3):** all three CODEX findings — (1) SESF-42 commit corrected `c642208`→`f0a35cd` (the `c642208` pointer was stale-from-mem0; `c642208` is SESF-45/PR #41); (2) `detect-secrets` scope corrected to Tier-1 structured providers only (Tier-3 is the local Shannon scanner); (3) the stale "357 tests" constraint replaced with the verified current count (360) + a command-based no-regression gate.
- **Rejected (0).** All three verified against live `git log` / `pytest --collect-only` before applying.

### CODEX Review (2026-06-14)

### Verified

- Read sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`; confirmed the active sketch folder is `/Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/sketches/SESF-44-tier2-assignment-precision/`.
- Read reconciled `requirements.md`, this `discovery.md`, project `AGENTS.md`, and `.context/GLOSSARY.md`; confirmed `.context/sketch-conventions.md` is absent.
- Checked live detector structure in `secret_redaction.py`: `_SECRET_KEYWORD` / `_ASSIGN_RE` at lines 103-110, `_KEYWORD_RE` and `_keyword_adjacent()` at lines 111 and 241-258, `_is_placeholder_value()` at lines 189-199, Tier-2 collection at lines 323-328, `_aggregate_maskable_candidates()` at lines 333-369, `redact()` at lines 372-426, and `_maskable_values()` / `_collect_spans()` / `scan_spans()` at lines 429-567.
- Checked public callers with `rg`: `rag_engine.py` has three direct `secret_redaction.redact()` calls at lines 767, 792, and 805; `sanitize.py` has four direct `secret_redaction.scan_spans()` calls at lines 314, 444, 638, and 678.
- Verified current Tier-2 and span test anchors in `tests/test_secret_redaction.py`, including synthetic fixture conventions at lines 30-35, Tier-2 fixtures at lines 94-100, Tier-2/value-guard tests at lines 159-188, and scan-spans parity coverage at lines 821-845.
- Checked history: `git log --grep=SESF-42` shows `f0a35cd` is SESF-42 PR #40; `git show c642208` shows SESF-45 PR #41. Ran `venv/bin/python -m pytest --collect-only -q`; current collection is 360 tests.
- Queried SessionFlow and mem0 for the SESF-44 baseline and redaction verification context; the 882 / 328 / 1,296-turn baseline and live-Milvus verification guidance are supported, while the `c642208` memory-derived commit pointer is stale.

### Top concerns

- Discovery cites the wrong commit for SESF-42, which would send the approach/tasks phase to the SESF-45 change if copied forward.
- The external-reference section incorrectly says `detect-secrets` covers Tier-3; live code explicitly uses a local Shannon scanner for Tier-3 and only uses detect-secrets for structured Tier-1 plugins.
- The exact "357 tests pass" constraint is stale on current `main`; the no-regression gate should be current-command based rather than locked to an old count.

### Unverified assumptions

- I did not run the full test suite; I only collected tests to validate the current count.
- I did not run a live `cleanup.py sanitize --dry-run` against Milvus Standalone; I treated that as an implementation verification step for later phases.
- I did not fetch additional Plane comments beyond the embedded issue context and SessionFlow/mem0 recall.
