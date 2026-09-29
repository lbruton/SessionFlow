---
sketch: "SESF-38-fts-backfill-hardening"
phase: requirements
created: 2026-06-05
---

# SESF-38 — Requirements

> **Source Issue:** [SESF-38](https://plane.lbruton.cc/lbruton/browse/SESF-38/)
> **Title:** FTS backfill hard-fails on schema drift during migration; only fix offered is destructive
>
> **Summary (from issue):** During the Milvus migration (.81 → .83) the startup FTS Backfill (`rag_engine.backfill_fts`, kicked off by `http_server.py:_fts_backfill`) hard-failed for an extended window and did **not** self-heal — it required a manual server restart to recover. The only remediation surfaced in the error string is the **destructive** `cleanup.py migrate-schema` (drops & recreates the collection), which on an intact 25,966-row collection would have caused the exact data loss the migration was trying to avoid. The collection schema was actually fine by the time it was inspected (all fields present incl. `issue_ids`); the running process was holding a stale view.
>
> **Ruled out (not bugs):** no data loss (Milvus held 25,966 Turns throughout); no stats desync (`get_session_stats` correctly project-scoped via `X-Project-Root`); search works (`search_all_sessions` with `project_root='*'` retrieves fine).
>
> **Suggested hardening (from issue):** (1) retry after transient Schema Drift (re-describe + re-attempt) instead of erroring permanently until restart; (2) offer a non-destructive reconcile path in the error message and stop recommending destructive `migrate-schema` when the live schema already matches; (3) collapse repeated identical failures into a single warning + backoff; (4) optional `/health` flag or stat exposing FTS-sidecar lag vs Milvus row count.
>
> _Filed from Devops session 2026-06-04 after live incident triage; resolved in the moment by a clean `launchctl kickstart -k` restart (non-destructive). Related: SESF-37 (.81 config drift), SESF-11 (Schema-Drift guard), SESF-13 (FTS thread affinity), SESF-25 (issue_ids field add)._

## Overview

This sketch hardens the FTS Backfill so a **transient Schema Drift no longer becomes a stuck, restart-only outage with data-loss-shaped guidance**. Today the startup backfill is one-shot: when Milvus reports drift mid-migration (often a stale in-process view, not real drift), backfill aborts, Hybrid Search silently degrades to vector-only until an operator restarts the server, and the only remediation surfaced is destructive (`migrate-schema`, which drops the collection). The work makes the FTS Sidecar **self-heal on the server's existing background cadence**, makes the startup Schema-Drift guard **re-verify before it raises**, replaces the misleading destructive-only error text with **honest non-destructive-first guidance**, **collapses repeated failure spam**, and **surfaces FTS lag** in health/stats. It matters now because the failure mode is not migration-specific — any FTS-schema or Milvus-schema change (e.g. SESF-25 adding `issue_ids`) re-arms it.

## User Stories

> Format: **As a** [role], **I want** [capability], **so that** [outcome].

- **US-1:** As a SessionFlow operator, I want the FTS Backfill to recover on its own after a transient Schema Drift, so that Hybrid Search returns to full keyword fidelity without my restarting the server.
- **US-2:** As an operator who hits a Schema-Drift error, I want remediation guidance that never steers me toward dropping intact data, so that I don't cause the data loss I was trying to avoid.
- **US-3:** As an operator tailing `server.log`, I want repeated identical backfill failures collapsed, so that a transient-drift window doesn't bury other signal in spam.
- **US-4:** As an operator, I want the FTS-Sidecar lag vs the Milvus Turn count exposed in health/stats, so that I can see keyword-search degradation without log-diving.

## Acceptance Criteria

> EARS syntax. Each line is individually testable and becomes a TDD Cohort B assertion downstream.

### AC-1 — FTS Sidecar self-heals without restart (maps to US-1; issue suggestion 1)
- **WHILE** the FTS Backfill sentinel (`fts_backfill_required`) is set, the system **SHALL** re-attempt `backfill_fts` on the server's existing background cadence — without an operator restart — and clear the sentinel once the FTS Sidecar is consistent with Milvus.

### AC-2 — Transient backfill failures retry rather than abandon (maps to US-1; issue suggestion 1)
- **IF** a background FTS Backfill attempt fails with a transient error (e.g. a `MilvusException`, or a Schema Drift that has not yet settled), **THEN** the system **SHALL** retry on a later cadence tick with bounded backoff and **SHALL NOT** permanently abandon re-attempts until the next process restart.

### AC-3 — Startup re-verifies drift before raising (maps to US-2; issue suggestions 1, 2)
- **WHEN** startup Schema-Drift detection observes drift, the system **SHALL** re-describe the Milvus collection and re-evaluate drift before raising; **IF** the re-evaluation shows no drift, **THEN** startup **SHALL** proceed without raising (recovering from a stale in-process view non-destructively).

### AC-4 — Persistent-drift error gives honest, non-destructive-first guidance (maps to US-2; issue suggestion 2)
- **IF** Schema Drift persists after re-describe, **THEN** the raised error **SHALL NOT** present `SESSIONFLOW_AUTO_MIGRATE_SCHEMA=1` as a non-destructive alternative, and **SHALL** state that (a) both `cleanup.py migrate-schema` and `AUTO_MIGRATE` drop and recreate the collection (destructive, all Turns lost), and (b) if the live schema is believed correct, a server restart may clear a stale in-process view (non-destructive) before any destructive step is considered.

### AC-5 — Repeated identical failures collapse to one log line (maps to US-3; issue suggestion 3)
- **WHILE** the FTS Backfill is failing repeatedly with the same error, the system **SHALL** emit a single warning for that error rather than one per cadence tick, logging again only when the error message changes or the backfill succeeds.

### AC-6 — FTS lag is observable in health/stats (maps to US-4; issue suggestion 4)
- The system **SHALL** expose, via the `/health` endpoint and `get_stats`, the Milvus Turn count, the FTS-Sidecar row count, their delta (FTS lag), and the `fts_backfill_required` sentinel state.

## Non-Goals

- **No dedicated `cleanup.py rebuild-fts` operator command.** The automatic in-process self-heal (AC-1/AC-2) covers FTS-Sidecar recovery while the server runs; an offline/manual rebuild lever is a possible follow-up, not part of this sketch. _(Surfaced for the requirements review seam — promote if operators need an offline lever.)_
- **No change to `migrate-schema`'s destructive behavior.** It stays the explicit, opt-in escape hatch for a genuinely drifted Milvus collection; only its *framing in the error message* changes (AC-4).
- **`SESSIONFLOW_AUTO_MIGRATE_SCHEMA` stays opt-in.** The system does not start auto-dropping/recreating the collection by default.
- **No Milvus migration-mechanics changes.** The .81→.83 migration is complete (SESF-37); this is general drift hardening, not migration tooling.
- **No embedding/MLX cadence changes**, no hosted embeddings, no new Providers, no search-ranking changes.

## Open Questions

_None — the issue body and the plan-mode investigation brief resolve all blocking questions. Ready for discovery._

---

> **Phase complete?** Acceptance criteria are concrete and verifiable. Open questions list is empty. Then advance: `/sketch-discovery SESF-38`.
