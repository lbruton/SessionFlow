---
sketch: "SESF-44-tier2-assignment-precision"
phase: requirements
created: 2026-06-14
---

# SESF-44 — Requirements

> **Source Issue:** [SESF-44](https://plane.lbruton.cc/lbruton/browse/SESF-44/)
>
> **Title:** Tier-2 ASSIGNMENT false positives: `auth` keyword matches `author*` + command/interpolation values (secret_redaction precision)
>
> **Summary (from issue):** The SESF-42 retroactive-sanitizer live dry-run baseline flagged **882 Tier-2 ASSIGNMENT hits across 302 turns**; a value-free audit (key names + masked context only) shows **~47% are false positives**. The detector lives in `secret_redaction.py` and is shared by **both** the SESF-41 ingestion guard and the SESF-42 sanitizer, so this precision fix benefits both.
>
> **False-positive breakdown (of 882):**
> - **328 (37%) — `auth` keyword matches `author*`:** GitHub/git metadata — `{"author":"<username>"}` PR-review JSON, `authoredDate`, `committedDate`, `authorAssociation` (enum like OWNER), `gradingAuthority`. The `_SECRET_KEYWORD` alternation includes bare `auth`, which substring-matches `author`.
> - **83 — keychain lookups:** `export KEY=$(security find-generic-password …)` — the matched value is the `$(security…)` command substitution, not a literal secret.
> - **6 — env interpolation:** `${VAR:-default}` — placeholder/default, not a literal secret.
> - Plus scattered JS function assignments (`window.changeVaultPassword = <fn>`) and meta-discussion (transcripts *about* the redaction detector).
>
> **Genuine exposure (NOT to fix here):** the real leaks are a focused subset, dominated by an Antigravity MCP-config JSON dump with live values (MEM0/OPENAI/BRAVE/CODACY/FIRECRAWL/PERPLEXITY API keys, Infisical client secret). Those are redacted via the sanitizer and the keys rotated (operator action, separate).
>
> **Proposed fix (secret_redaction.py):**
> - **Tighten the `auth` keyword:** replace bare `auth` with word-boundary / negative-lookahead handling (e.g. `auth(?!or)`, or drop it in favor of `oauth|authorization|auth[_-]?token`) so `author`/`authority`/`authored`/`authorassociation` no longer match.
> - **Skip non-literal values** in the Tier-2 value-shape guard (`_is_placeholder_value` / `_ASSIGN_RE`): a value that is a command substitution `$(…)` or env interpolation `${…}` contains no literal secret — reject it.
>
> **Acceptance Criteria (from issue):**
> - Synthetic fixtures: `author="<name>"`, `authoredDate="…"`, `authorAssociation="OWNER"` → NOT flagged.
> - `export FOO_API_KEY=$(security find-generic-password …)` and `${FOO_PASSWORD:-default}` → NOT flagged.
> - A real `FOO_API_KEY="<synthetic-literal>"` (assembled from parts) → STILL flagged (no regression on true positives).
> - Re-running the existing `tests/test_secret_redaction.py` Tier-2 suite stays green; add the above cases.
>
> **Workflow (from issue):** Sized for `/sketch` (single-file detector change with focused tests; non-destructive). Surfaced by the SESF-42 live baseline.

## Overview

Tighten the **Tier-2 ASSIGNMENT detector** in `secret_redaction.py` so it stops emitting two classes of false positive that dominated the SESF-42 live baseline (~47% of 882 Tier-2 hits): the bare `auth` keyword substring-matching `author*` git/GitHub metadata (37%), and command-substitution / env-interpolation values being treated as literal secrets (~10%). Because `redact()` (the SESF-41 ingestion guard) and `scan_spans()` (the SESF-42 sanitizer) share `_aggregate_maskable_candidates()`, a single detector fix raises precision on both the forward guard and retroactive auditing without touching either call site. The change must preserve recall on genuine secrets — `Authorization` headers, OAuth tokens, `auth_token`, and literal `*_API_KEY="…"` assignments stay detected.

## Decisions Locked (grill)

- **D-A `auth` keyword → bounded alternation.** Replace bare `auth` in `_SECRET_KEYWORD` with the `oauth` / `authorization` / `auth_token` family. This eliminates the `author*` family and keeps OAuth / authorization / auth-token keys. **Bare standalone `auth=<value>` is intentionally no longer matched by this keyword** (accepted recall tradeoff — see Non-Goals). A naïve `auth(?!or)` lookahead is rejected because it would also break `authorization`. **Key-segment boundary is mandatory:** `_ASSIGN_RE` currently wraps every keyword with `[A-Za-z0-9_\-]*`, so a bare `oauth` alternative would still substring-match `coauthor` / `coAuthoredBy` and reintroduce the FP class. The replacement must match only as a bounded key segment; *how* (segment-boundary regex, word boundary, etc.) is an approach-phase decision.
- **D-B non-literal value rejection scope = `$(…)` and `${…}` only.** Backtick command substitution and other interpolation forms (`<%= %>`, `#{…}`) are explicitly out of scope to minimize the chance of suppressing a real literal.

## User Stories

> Format: **As a** [role], **I want** [capability], **so that** [outcome].

- **US-1:** As an operator running the secret detector, I want git/GitHub `author*` metadata to stop being flagged as Tier-2 secrets, so the audit signal isn't drowned in false positives.
- **US-2:** As an operator, I want shell command-substitution and env-interpolation values recognized as non-literal, so keychain lookups and env defaults aren't reported as leaked secrets.
- **US-3:** As a security maintainer, I want true-positive detection on genuine secret-keyed assignments (`authorization` / `oauth` / `auth_token` keys with literal values, and literal `*_API_KEY` assignments) preserved, so the precision fix does not reduce recall on genuine secrets.

## Acceptance Criteria

> EARS syntax. Each line is individually testable and becomes a TDD Cohort B assertion. `<detector>` = the Tier-2 ASSIGNMENT path in `secret_redaction.py` (`_SECRET_KEYWORD` / `_ASSIGN_RE` / `_is_placeholder_value`).

### AC-1 — author* / co-author family suppressed (maps to US-1)
- **IF** a Tier-2 assignment key is an `author*` / `authority*` / co-author identifier — `author`, `authored`, `authoredDate`, `authorAssociation`, `authority`, `gradingAuthority`, `coauthor`, `coAuthoredBy` (i.e. a secret keyword appearing only as a substring of a longer non-secret word, **including the `oauth`-in-`coauthor` overlap**) — **THEN** the `<detector>` **SHALL NOT** emit a Tier-2 ASSIGNMENT hit for that assignment.
- The replacement keyword set **SHALL** match only as a bounded key segment, so no `author*`/co-author key is reintroduced as a false positive via substring overlap (e.g. `oauth` ⊂ `coauthor`).

### AC-2 — genuine auth keywords preserved (maps to US-3)
- **WHEN** a Tier-2 assignment key is `authorization`, `oauth`, or `auth_token` / `auth-token` / `authToken` (case-insensitive) **AND** its value is a literal that passes the existing value-shape guard (`_is_placeholder_value` — ≥ 8 chars, not a placeholder word/shape), the `<detector>` **SHALL** emit a Tier-2 ASSIGNMENT hit (recall preserved). Placeholder/short values (e.g. `authorization=none`) remain correctly skipped by the existing guard — this AC does not override it.

### AC-3 — command substitution rejected (maps to US-2)
- **IF** a Tier-2 assignment value is a shell command substitution of the form `$(…)`, **THEN** the `<detector>` **SHALL NOT** emit a hit for that assignment.

### AC-4 — env interpolation rejected (maps to US-2)
- **IF** a Tier-2 assignment's key **or** value falls inside an environment-interpolation span of the form `${…}`, **THEN** the `<detector>` **SHALL NOT** emit a hit for that assignment. This covers the live failure mode where `x=${FOO_PASSWORD:-default}` matches as key `FOO_PASSWORD` with value `-default` — the `${`/`}` surround the whole match, so a guard that only rejects captured *values* literally beginning with `${` is insufficient; the interpolation span around the match must be detected.

### AC-5 — literal secrets still flagged (maps to US-3)
- **WHEN** a Tier-2 assignment pairs a secret-bearing key with a literal value (e.g. `FOO_API_KEY="<synthetic-literal>"`, value assembled from parts in tests), the `<detector>` **SHALL** emit a Tier-2 ASSIGNMENT hit (no regression on true positives).

### AC-6 — single fix benefits both paths (maps to US-1, US-2, US-3)
- The `<detector>` **SHALL** apply the AC-1…AC-5 behavior identically through both `redact()` (SESF-41 guard) and `scan_spans()` (SESF-42 sanitizer), via the shared `_aggregate_maskable_candidates()` aggregation — no duplicated or divergent logic.

### AC-7 — suite green + new fixtures through both paths (maps to US-1, US-2, US-3)
- The `<detector>` change **SHALL** keep the existing `tests/test_secret_redaction.py` Tier-2 suite green, **AND** the suite **SHALL** gain synthetic fixtures covering AC-1 through AC-5: false positives (`author="<name>"`, `authoredDate="…"`, `authorAssociation="OWNER"`, `coAuthoredBy="…"` → not flagged; `export FOO_API_KEY=$(security find-generic-password …)` and `x=${FOO_PASSWORD:-default}` → not flagged) and true positives (`authorization` / `oauth` / `auth_token` with a literal value, and a literal `FOO_API_KEY` → still flagged).
- Each new false-positive and true-positive case **SHALL** be asserted through **both** public paths — `redact()` (Hit present/absent) **and** `scan_spans()` (Span present/absent) — so the SESF-41 guard and SESF-42 sanitizer are proven aligned (per AC-6).

## Non-Goals

_Explicit list of things this sketch does NOT do. Each entry should make a future reader confident the omission was intentional._

- **Not** redacting the genuine secret exposure (the Antigravity MCP-config JSON dump with live MEM0/OPENAI/BRAVE/etc. keys) — that is operator sanitizer action + key rotation, tracked separately. This sketch is precision-only.
- **Not** preserving bare standalone `auth=<value>` detection — intentionally dropped via the D-A alternation (accepted recall tradeoff; standalone `auth=` with no `oauth`/`authorization`/`_token` context is rare and the FP cost of bare `auth` is high).
- **Not** rejecting backtick command substitution `` `…` `` or other interpolation forms (`<%= %>`, `#{…}`) — D-B keeps the non-literal guard to `$(…)` / `${…}` only.
- **Not** adding `Authorization: Bearer <token>` header capture — a pre-existing recall gap (`_ASSIGN_RE` captures `Bearer` as the value and skips it as too short); expanding recall there is a separate concern, out of scope for this precision fix.
- **Not** changing Tier-1 (structured providers / PEM) or Tier-3 (entropy) detection.
- **Not** altering the existing placeholder-word / minimum-length value guard in `_is_placeholder_value`, beyond adding the `$(…)` / `${…}` non-literal rejection.

## Open Questions

_None — both precision/recall decisions resolved in grill (D-A, D-B above)._

---

> **Phase complete?** Acceptance criteria are concrete and verifiable. Open questions list is empty. Then advance: `/sketch-discovery SESF-44`.

## Review Archive — requirements (2026-06-14)

### Resolution Summary

- **Accepted (4):** all four CODEX inline findings — (1) `oauth`-substring boundary gap (`coauthor`/`coAuthoredBy`) folded into D-A + AC-1 with a mandatory bounded-key-segment requirement; (2) AC-2 qualified with the existing literal value-shape guard; (3) AC-4 reworded to detect the `${…}` interpolation span around the match (not just captured values starting with `${`); (4) AC-7 now requires every new fixture to assert through both `redact()` and `scan_spans()`.
- **Rejected (0).**
- **Scoped out (1, your-call default):** the Bearer-header sub-point — added as a Non-Goal (pre-existing recall gap, separate concern), consistent with this sketch's precision-only scope.

### CODEX Review (2026-06-14)

### Verified

- Read sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`; resolved the single active sketch folder to `/Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/sketches/SESF-44-tier2-assignment-precision/`.
- Read project context in `AGENTS.md`; `.context/GLOSSARY.md`; confirmed `.context/sketch-conventions.md` is absent, so generic sketch conventions apply.
- Checked the live detector in `secret_redaction.py`: `_SECRET_KEYWORD` / `_ASSIGN_RE` at lines 103-110, `_is_placeholder_value()` at lines 189-199, Tier-2 collection at lines 323-328, shared aggregation at lines 333-345, `redact()` at lines 372-426, `_maskable_values()` / `_collect_spans()` at lines 429-527, and `scan_spans()` at lines 530-567.
- Checked current Tier-2 tests in `tests/test_secret_redaction.py`, including placeholder/short-value behavior at lines 177-188 and `scan_spans` Tier-2 coverage at lines 702-712.
- Probed the live regex behavior for `author`, `authorization`, `authToken`, `$(security ...)`, and `${FOO_PASSWORD:-default}` cases to confirm the review comments against the current implementation.

### Top concerns

- D-A's proposed alternation can still match co-author metadata because `_ASSIGN_RE` allows arbitrary key text around `oauth`; the requirements need a boundary or an explicit co-author false-positive fixture.
- AC-2 currently requires detection from key shape alone and does not say the value must pass the existing literal-value guard; this risks contradicting the placeholder/min-length guard or missing the intended Authorization-header shape.
- AC-4 does not capture the live `${VAR:-default}` failure mode: the regex can match inside the interpolation as `FOO_PASSWORD:-default`, so a simple captured-value guard is insufficient.
- AC-7 should explicitly require the new fixtures through both `redact()` and `scan_spans()` to prove the SESF-41 and SESF-42 paths stay aligned.

### Unverified assumptions

- I did not re-run the full `tests/test_secret_redaction.py` suite because this phase is requirements review only.
- I treated the embedded source issue text as authoritative for SESF-44 and did not fetch additional Plane comments beyond the local sketch artifact.
