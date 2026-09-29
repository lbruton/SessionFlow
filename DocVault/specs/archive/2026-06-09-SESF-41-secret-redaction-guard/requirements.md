---
sketch: "SESF-41-secret-redaction-guard"
phase: requirements
created: 2026-06-08
---

# SESF-41 — Requirements

> **Source Issue:** [SESF-41](https://plane.lbruton.cc/lbruton/browse/SESF-41/) — _Ingestion-time secret redaction guard for indexed turns_ (priority: high)
>
> **Issue summary (pasted at scaffold):**
> Prevention half of the SESF-35 research split (research note: `DocVault/Research/SessionFlow/Secret Redaction & Retroactive Sanitization (SESF-35).md`). Today raw transcript text reaches durable storage with **zero redaction**: inside `add_turns` (`rag_engine.py:689`) the raw `turn["text"]` flows into the embedding vector (`:742`), the Milvus `document` field (`:764`), and the FTS5 `content` column (`:791`). Confirmed leak class: the Antigravity Provider Adapter ingests every entry's raw text incl. tool/MCP-config output (`provider_antigravity.py:325`).
>
> **Scope:** New pure module `secret_redaction.py` exposing `redact(text) -> (redacted_text, hits)` — no I/O, no network. Tiered detector + operator-extensible allowlist + typed placeholder masker. Hook at the top of `add_turns` (covers all Providers + both entry points). Config env vars. Scrub error-log leak surfaces.
>
> **Workflow:** Sized for `/sketch`. Non-destructive. Builds the shared `secret_redaction.py` that the cleanup issue (SESF-42) depends on — do this first.

## Overview

SessionFlow currently writes raw Turn text verbatim into three durable artifacts — the embedding vector, the Milvus Standalone `document` field, and the FTS Sidecar `content` column — with no redaction anywhere. A confirmed leak (an Antigravity Turn carrying a dumped MCP config with secret-bearing fields) proves the risk is live. This sketch adds a single ingestion-time guard: a pure `secret_redaction.py` module invoked at the top of `add_turns`, the one chokepoint every Provider and entry point funnels through. It ships **report-first** (measure, don't mutate) so the operator can size the false-positive rate before enabling enforcement, and it builds the shared detector module the retroactive sanitizer (SESF-42) depends on.

## User Stories

- **US-1:** As a SessionFlow operator, I want secrets in Turn text replaced with typed placeholders before the text is embedded or stored, so that the vector store and FTS Sidecar never persist credentials in cleartext.
- **US-2:** As an operator evaluating the guard, I want a `report` mode that detects and counts secrets without mutating Turns, so that I can size the false-positive rate against my real content before switching to enforcement.
- **US-3:** As a maintainer, I want detection to be deterministic, idempotent, and side-effect-free, so that re-ingesting the same Transcript still dedups by `doc_id` and the detector never double-redacts or leaks values through logs.

## Acceptance Criteria

> EARS syntax. Each line is individually testable and becomes a TDD Cohort B assertion. No raw secret values appear anywhere — fixtures use synthetic, non-functional tokens only.

### Redaction at ingestion (maps to US-1)

- **AC-1:** WHILE redaction is enabled in `enforce` mode, WHEN a Turn is ingested through `add_turns`, the system SHALL replace each detected secret in the Turn text with a typed placeholder `[REDACTED:<RULE_NAME>]` before the text is embedded, written to the Milvus `document` field, or written to the FTS Sidecar `content` column.
- **AC-2:** WHILE redaction is enabled in `enforce` mode, the system SHALL compute the embedding vector over the redacted text, such that two Turns differing only in their secret values produce identical embeddings.
- **AC-3:** The system SHALL apply redaction at a single hook at the top of `add_turns`, such that both `add_turns` and its async wrapper `add_turns_async` and every Provider and entry point are covered without per-Provider changes.
- **AC-4:** WHERE the matched detector rule has no specific name, the system SHALL fall back to the placeholder `[REDACTED:SECRET]`.

### Detector accuracy (maps to US-1)

- **AC-5:** The detector SHALL flag, as Tier 1, structured provider secrets (OpenAI / Anthropic / GitHub / AWS / Google / Slack / Stripe / GitLab / HuggingFace / npm token classes), PEM private-key blocks, and JWTs — emitting the correct rule name per class.
- **AC-6:** The detector SHALL flag, as Tier 2, contextual assignment patterns (a secret-bearing keyword or env-var name followed by `:` or `=` and a value), including the MCP-config-dump leak class.
- **AC-7:** IF a Tier-2 candidate value is a placeholder shape (e.g. `***`, `<your-key>`, `changeme`, `xxxx`) or is shorter than the minimum value length, THEN the detector SHALL NOT flag it.
- **AC-8:** WHERE a high-entropy token is adjacent to a Tier-2 keyword or inside a fenced config block, the system SHALL redact it; otherwise the system SHALL treat the entropy hit as advisory-only (reported, never auto-redacted).
- **AC-9:** IF a token matches an allowlist entry — UUID, 40-hex git SHA, 64-hex SHA-256, a SessionFlow content-hash `doc_id`, an issue ID, or an operator-supplied pattern loaded from the configured allowlist path — THEN the detector SHALL NOT flag it.

### Report mode & configuration (maps to US-2)

- **AC-10:** WHILE redaction is enabled in `report` mode, the system SHALL store the Turn text unredacted and SHALL emit per-rule detection counts (rule names only, no secret values) without altering the embedding, `document`, or `content`.
- **AC-11:** WHEN `SESSIONFLOW_REDACT` is unset, the system SHALL default to redaction enabled in `report` mode.
- **AC-12:** IF `SESSIONFLOW_REDACT` is set off, THEN the system SHALL store Turn text exactly as before — no detection and no mutation.
- **AC-13:** The system SHALL read mode from `SESSIONFLOW_REDACT_MODE` (`enforce` | `report`) and the allowlist location from a configured allowlist-path env var.

### Determinism, idempotency & no-leak (maps to US-3)

- **AC-14:** The redaction output SHALL be deterministic (no random salt), such that re-ingesting identical source bytes yields identical redacted text and preserves `doc_id` dedup (no duplicate rows).
- **AC-15:** The redaction SHALL be idempotent — applying `redact` to already-redacted text leaves existing `[REDACTED:…]` placeholders unchanged (`redact(redact(x)) == redact(x)`).
- **AC-16:** The `secret_redaction.py` module SHALL perform no I/O and no network calls (allowlist file loading happens outside the pure `redact` function).
- **AC-17:** WHEN an FTS insert fails or an embedding error is raised, the system SHALL scrub the logged exception text so that no secret value can echo through error output.
- **AC-18:** No raw secret values SHALL appear in code, tests, fixtures, logs, or program output — only synthetic tokens and rule/class names.

## Non-Goals

- **Retroactive sanitizer (SESF-42).** Cleaning already-indexed Turns (Milvus `upsert`, `cleanup.py sanitize`, re-embed throttle, audit store) is the `/spec`-sized cleanup half. This sketch is prevention only.
- **Rewriting on-disk source Transcripts.** SessionFlow sanitizes only what it owns; external harness files stay upstream (research note §9).
- **Searchable redaction-status schema field.** No `redacted`/`redaction_rev` column in the Milvus schema for MVP — would be a gated schema add.
- **Provider-level entry-type filtering.** Skipping obvious tool/command-output entries in adapters (esp. Antigravity) is a coarser defense-in-depth layer; deferred as a follow-up (research note §7).
- **Correlation fingerprints / HMAC audit store.** Salted-fingerprint cross-Turn correlation belongs to the SESF-42 audit flow, not the guard.
- **Key rotation.** Redaction ≠ safety; rotating already-exposed keys is the operator's responsibility and is surfaced in docs, not automated here.

## Open Questions

_Empty — resolved during scaffolding and the requirements grill._

- [x] Detector engine → **adopt `detect-secrets`** (Yelp, pure-Python, no network, native allowlist/baseline). New runtime dependency accepted.
- [x] Report vs enforce semantics → **`report` = detect + count only, store raw (measurement); `enforce` = redact + placeholder** (AC-10, AC-1).
- [x] Out-of-the-box default → **on + `report`** when `SESSIONFLOW_REDACT` unset (AC-11).

---

> **Phase complete?** Acceptance criteria are concrete and verifiable. Open questions list is empty. Then advance: `/sketch-discovery SESF-41`.
