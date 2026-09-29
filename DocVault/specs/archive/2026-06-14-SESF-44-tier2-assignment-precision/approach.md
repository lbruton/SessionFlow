---
sketch: "SESF-44-tier2-assignment-precision"
phase: approach
created: 2026-06-14
---

# SESF-44 — Approach

_How we'll build it. **Don't write code or tests** — the tasks phase produces the work plan._

## High-Level Architecture

The whole change is a precision tune of the **Tier-2 contextual-assignment path** in `secret_redaction.py`. Two surgical edits, plus one new pure helper, all upstream of the shared `_aggregate_maskable_candidates` seam — so a single change propagates to both `redact()` (SESF-41 guard) and `scan_spans()` (SESF-42 sanitizer) with zero call-site edits (the AC-6 guarantee established in discovery).

1. **Keyword breadth (D-A):** `_SECRET_KEYWORD` (`:103-105`) drops bare `auth` and adds a **left-boundary-anchored** auth family. The boundary is what stops the residual `oauth`⊂`coauthor` substring FP that a naïve alternation would reintroduce.
2. **Non-literal value rejection (D-B):** a new pure predicate is consulted in the Tier-2 branch of `_collect_candidates` (`:324-328`) to drop command-substitution / env-interpolation matches — covering both the captured-value shape (`$(…)`) and the harder case where the secret keyword is matched *inside* a `${…}` span (so the captured value isn't itself `${…}`-shaped).

No tier-architecture change, no new module, no call-site change. The four detector invariants (pure / deterministic / idempotent / no-raw-value) are preserved — the new helper is pure (line + match offsets in, bool out).

## Key Decisions

| # | Decision | Rationale | Tradeoff |
|---|----------|-----------|----------|
| **D-1** | Drop bare `auth`; auth family becomes `(?<![A-Za-z])(?:oauth\|authorization\|auth[_-]?token)` inside `_SECRET_KEYWORD`. | A **left** negative-lookbehind is sufficient and minimal: every reported FP (`author`, `authority`, `coauthor`, `coAuthoredBy`) has the keyword either gone (bare `auth` removed) or preceded by a letter (`oauth` in `coauthor` sits behind `c`), so the lookbehind kills it. `authToken`/`auth_token`/`auth-token` still match via the existing bare `token` keyword **and** the explicit `auth[_-]?token` alt; `oauthToken` (camelCase) still matches because there is **no** right boundary. | No **right** boundary, so a contrived `oauthxyz=…` would still match `oauth` — accepted: erring toward recall on a secret detector, and no right-side FP family was observed. Other keywords (`secret`, `token`, …) keep loose substring matching — accepted: only the `auth` family produced an FP family. |
| **D-2** | Add a pure helper `_is_nonliteral_assignment(line, match) -> bool`, consulted in the Tier-2 branch alongside `_is_placeholder_value`. It rejects when **(a)** the captured value starts with `$(` or `${`, **or (b)** the match span `[match.start(), match.end())` overlaps a `${…}` interpolation span found on the line via `re.finditer(r"\$\{[^}]*\}", line)`. | (a) covers AC-3 (`KEY=$(security …)` captures value `$(security`). (b) covers AC-4's hard case: `x=${FOO_PASSWORD:-default}` matches key `FOO_PASSWORD`, value `-default` — the `${` precedes the key, so a value-shape check alone misses it; the span-overlap check catches it. | A genuine secret value that *starts* with `$(`/`${` would be skipped — accepted: `startswith` (not `contains`) keeps this conservative, and a literal secret beginning with `$(` is implausible. The `[^}]*` span regex won't match nested `${a${b}}` — accepted as a rare edge (noted in Risks). |
| **D-3** | Keep the boundary in the **shared** `_SECRET_KEYWORD` (inherited by `_KEYWORD_RE` → Tier-3 adjacency), not localized to `_ASSIGN_RE`. | One source of truth; and removing spurious `auth`-in-`author` adjacency from Tier-3 entropy forcing is a *benefit*, not a regression — we don't want `author` forcing entropy hits either. | Tier-3 `_keyword_adjacent` behavior near `author*` shifts slightly; mitigated by keeping `test_ac8_*` green (verification, not redesign). |
| **D-4** | Put (b)'s line/offset logic in the Tier-2 branch (it has `line` + `match`); leave `_is_placeholder_value` otherwise untouched. | `_is_placeholder_value` is value-only by contract; threading line/offsets through it would broaden its signature for one caller. A sibling helper keeps each predicate cohesive. | One extra predicate call in the hot Tier-2 loop — negligible (one `finditer` over a single line). |

> **Decided target pattern (the contract for tasks/impl, not final code):**
> `_SECRET_KEYWORD` → `api[_-]?key|secret|token|password|passwd|pwd|(?<![A-Za-z])(?:oauth|authorization|auth[_-]?token)|credential|access[_-]?key`

## File Map

### New
- _none_ — the new `_is_nonliteral_assignment` helper lives in the existing `secret_redaction.py`.

### Modified
- `secret_redaction.py`
  - `_SECRET_KEYWORD` (`:103-105`) — drop bare `auth`, add left-anchored auth family (D-1).
  - new pure helper `_is_nonliteral_assignment(line, match)` near the other value guards (`~:189-219`) (D-2).
  - `_collect_candidates` Tier-2 branch (`:324-328`) — consult the new helper alongside `_is_placeholder_value` before appending the `ASSIGNMENT` candidate.
- `tests/test_secret_redaction.py`
  - new synthetic fixtures (assembled-from-parts) + tests for AC-1…AC-5, each asserted through **both** `redact()` (Hit/no-Hit) and `scan_spans()` (Span/no-Span), extending the `test_ac6_*` / `test_ac7_*` / `test_scan_spans_*` patterns (AC-7).

### Deleted
- _none_

## Data / Schema Changes

None — no schema, migration, or persisted-data changes. (Detector logic only; existing indexed data is unaffected — a separate sanitizer re-run is the verification step, not part of this change.)

## Tradeoffs Surfaced for Review

- **Left-boundary-only (D-1).** I chose recall-safe (no right boundary). If you'd rather also suppress right-side substring matches (`oauthxyz`), that's a one-token change to `(?![a-z])` — but it risks missing `oauthToken`-style camelCase unless case-aware. Flagging in case you want the stricter variant.
- **Shared vs localized boundary (D-3).** I put the boundary in the shared keyword (affects Tier-3 adjacency benignly). If you'd prefer Tier-3 untouched, the boundary can be localized to `_ASSIGN_RE` at the cost of duplicating the auth-family sub-pattern.

## UI Contract

N/A — no UI surface. Pure detector/regex change in `secret_redaction.py`.

## Out of Scope (follow-up issues)

- `Authorization: Bearer <token>` header capture — pre-existing recall gap, already a Non-Goal in requirements; not filed (raise separately if recall there is wanted).
- A general key-segment tokenizer for *all* keywords — not justified; only the `auth` family produced an FP family. If `secret`/`token` substring FPs surface later, revisit.

## Risk Notes

- **Tier-3 coupling** (`_KEYWORD_RE` shares `_SECRET_KEYWORD`) → run `test_ac8_*`; expected behavior shift is *fewer* spurious forced-entropy hits near `author*`, which is benign. → mitigation: keep AC-8 tests green; if any asserts `auth`-adjacency, reconcile it as an intended change.
- **Nested interpolation** `${a${b}}` not matched by `\$\{[^}]*\}` → rare; accept, note in code comment.
- **`startswith` vs `contains`** for `$(`/`${` value-shape → deliberately `startswith` to avoid suppressing real secrets that merely contain those chars.

---

> **Phase complete?** Architecture clear, decisions logged with rationale, file map complete. Then advance: `/sketch-tasks SESF-44`.

## Review Archive — approach (2026-06-14)

### Resolution Summary

- **Accepted (0) / Rejected (0) / Your-input (0).** CODEX returned **Top concerns: None** — it independently probed the D-1 target pattern (suppresses `author`/`coauthor`/`coAuthoredBy`/`authority`; preserves `authorization`/`oauth`/`oauthToken`/`authToken`/`auth_token`/`myAuthToken`/`api_key`) and the D-2 value-start + interpolation-span checks against live `_ASSIGN_RE` captures, and found the approach consistent with reconciled requirements/discovery and live code. No changes to apply; review archived verbatim.

### CODEX Review (2026-06-14)

### Verified

- Read sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`; confirmed this review is phase-scoped to `approach.md`.
- Read reconciled `requirements.md`, reconciled `discovery.md`, this `approach.md`, project `AGENTS.md`, and `.context/GLOSSARY.md`; confirmed `.context/sketch-conventions.md` is absent.
- Checked live detector structure in `secret_redaction.py`: `_SECRET_KEYWORD` / `_ASSIGN_RE` at lines 103-110, `_KEYWORD_RE` / `_keyword_adjacent()` at lines 111 and 241-258, `_is_placeholder_value()` at lines 189-199, Tier-2 collection at lines 323-328, `_aggregate_maskable_candidates()` at lines 333-369, and `scan_spans()` routing through `_maskable_values()` at lines 429-567.
- Probed the D-1 target pattern against representative keys: it suppresses `author`, `coauthor`, `coAuthoredBy`, and `authority`; it preserves `authorization`, `oauth`, `oauthToken`, `authToken`, `auth_token`, `myAuthToken`, and `api_key`; the stated `oauthxyz` recall tradeoff also holds.
- Probed the current `_ASSIGN_RE` captures for `export FOO_API_KEY=$(security find-generic-password item)`, `x=${FOO_PASSWORD:-default}`, and `password=${FOO_PASSWORD:-default}`; the proposed value-start and interpolation-span checks cover the observed failure modes.
- Checked existing Tier-2 / value-guard / scan-spans parity test anchors in `tests/test_secret_redaction.py` so the file map and AC-7 test placement are realistic.

### Top concerns

- None. The approach is consistent with the reconciled requirements, reconciled discovery, and live code shape.

### Unverified assumptions

- I did not run the full test suite; this was a review-only pass over the approach and live code surface.
- I did not run a live `cleanup.py sanitize --dry-run` against Milvus Standalone; that remains an implementation verification step.
