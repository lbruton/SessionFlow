---
sketch: "SESF-17-desktop-project-detection"
phase: requirements
created: 2026-05-28
---

# SESF-17 — Requirements

> **Source Issue:** [SESF-17](https://plane.lbruton.cc/lbruton/browse/SESF-17/)
> **Title:** Enhance project detection heuristics in desktop logging daemon to resolve 'unknown' project tags
>
> **Description:** Some turns captured under the `antigravity_desktop` provider are currently indexed with the project tag 'unknown'. The desktop logger should be enhanced to resolve the workspace path more reliably and associate actions with their correct project names, avoiding unmapped entries.

## Overview

The `antigravity_desktop` adapter resolves a conversation's `project_root` from `~/.gemini/antigravity/history.jsonl` — but that file is **CLI-only and absent for Desktop**, so `_load_history()` always returns empty and *every* desktop conversation falls back to `project_root="unknown"` (observed at sketch time: every discovered desktop conversation tagged `unknown` — a snapshot in the low-40s, with the exact source count and `.pb` summary count drifting; treat as illustrative, not a fixed assertion). Antigravity Desktop instead records a conversation→workspace association in `~/.gemini/antigravity/agyhub_summaries_proto.pb`. This sketch makes the desktop variant resolve `project_root` from that summaries metadata, so desktop turns are tagged with their real workspace path and become discoverable by project-scoped search and `/start`. Coverage is best-effort: conversations the metadata doesn't map stay `unknown`, and the CLI path is untouched.

## User Stories

- **US-1:** As someone searching SessionFlow history, I want `antigravity_desktop` turns tagged with their real workspace path, so project-scoped search and `/start` surface them instead of burying them under `unknown`.
- **US-2:** As the SessionFlow operator, I want desktop workspace resolution to degrade gracefully when its metadata source is missing or its format changes, so ingestion never crashes and the CLI path never regresses.

## Acceptance Criteria

> EARS syntax. Each line is individually testable and becomes a downstream TDD assertion.

### AC-1 — resolve workspace from desktop summaries (maps to US-1)
- **WHEN** the desktop `AntigravityAdapter` discovers a source whose `conversation_id` is present in the Antigravity Desktop summaries metadata (`agyhub_summaries_proto.pb`), the adapter **SHALL** set that source's `project_root` to the mapped workspace path.

### AC-2 — normalize `file://` URIs (maps to US-1)
- The adapter **SHALL** normalize a `file://` workspace URI from the summaries metadata to a plain filesystem path (e.g. `file:///Volumes/DATA/GitHub/StakTrakr` → `/Volumes/DATA/GitHub/StakTrakr`) before assigning it to `project_root`. Normalization **SHALL** use proper URL-decoding and path semantics (e.g. Python's `urllib.parse.urlparse` + `unquote`) — not a bare scheme strip — so percent-encoded characters (spaces as `%20`, unicode) decode to the literal filesystem path; a `file://` URI with encoded path characters **SHALL** yield the same path the other providers would store for that workspace.

### AC-3 — resolution precedence (maps to US-1)
- **WHEN** resolving `project_root` for a desktop conversation, the adapter **SHALL** apply the precedence `history.jsonl` → summaries metadata → `"unknown"`, using the first source that yields a workspace.

### AC-4 — unmapped conversations remain explicit (maps to US-1)
- **IF** a discovered desktop `conversation_id` has no workspace in either `history.jsonl` or the summaries metadata, **THEN** the adapter **SHALL** set `project_root` to `"unknown"`.

### AC-5 — graceful degradation (maps to US-2)
- **IF** `agyhub_summaries_proto.pb` is absent, unreadable, locked by a concurrent desktop-daemon write, or truncated/partially written, **OR** binary decoding hits a boundary error, **THEN** the adapter **SHALL** catch the failure (including `OSError`, `PermissionError`, and binary-parse boundary errors such as `IndexError`/`ValueError`), fall back to existing behavior without raising, and desktop sources **SHALL** resolve to `"unknown"` (or `history.jsonl` if present).

### AC-6 — CLI path unchanged (maps to US-2)
- The `antigravity_cli` variant **SHALL** continue to resolve `project_root` solely from `history.jsonl`, with no behavioral change from this sketch.

### AC-7 — workspace value is a path, not a derived name (maps to US-1)
- The adapter **SHALL** store `project_root` as the workspace filesystem path (consistent with all other providers); it **SHALL NOT** introduce a separate project-name field — the display name remains derived downstream from the path.

### AC-8 — health/limitations messaging reflects summaries support (maps to US-2)
- **WHEN** the desktop adapter parses `agyhub_summaries_proto.pb` for workspace resolution, the provider's `health()`/limitations text **SHALL** distinguish the now-supported summaries metadata from still-opaque transcript/binary artifacts — it **SHALL NOT** continue to report the summaries file as unparsed (the current `"Protobuf/database artifacts are not parsed in SESF-6."` diagnostic), so the operator-facing status accurately reflects what is and is not parsed.

## Non-Goals

- **Not backfilling / re-tagging existing rows** — desktop turns already indexed in Milvus as `project_root="unknown"` are left as-is; this sketch fixes go-forward ingestion only. (Re-indexing those rows is a possible follow-up.)
- **Not adding a secondary fallback heuristic** — conversations the summaries metadata does not map (e.g. multi-repo sessions with no single workspace) stay `"unknown"`. No inference from tool-call file paths.
- **Not parsing the full Antigravity protobuf schema** — only the conversation→workspace association is extracted; other binary artifacts remain unparsed, consistent with the SESF-6 limitation.
- **Not changing the CLI variant** — `antigravity_cli` resolution is untouched.
- **Not touching the search/query layer** — this is ingestion-side only.

## Open Questions

_None — scope resolved._

---

> **Phase complete?** Acceptance criteria are concrete and verifiable. Open questions list is empty. Then advance: `/sketch-review SESF-17 requirements [AGENT]`.

## Review Archive — requirements (2026-05-30)

### Resolution Summary

Reconciled 2026-05-30 via `/sketch-reconcile SESF-17 requirements`. **4 accepted, 1 rejected, 1 resolved with operator input.**

- **Accepted:**
  - Overview desktop-conversation count softened to an illustrative snapshot (CODEX) — no longer asserts an exact "0 of 41".
  - **AC-2** strengthened to mandate `urllib.parse` URL-decoding + path semantics over a bare scheme strip (CODEX + GEMINI consensus).
  - **AC-5** strengthened with explicit error-class handling for locks/concurrent-writes/truncation (`OSError`/`PermissionError`/`IndexError`/`ValueError`) (GEMINI).
  - **AC-8** added — `health()`/limitations text must distinguish supported summaries metadata from still-opaque artifacts (CODEX, your-call; GEMINI corroborated the live `health()` string).
- **Rejected:**
  - GEMINI "parse `.pb` once per `discover_sources()`" — implementation-efficiency detail, not a user-observable requirement. **Carried forward to approach.md**, not dropped.
- **Forward-guidance (no requirements change):** Both reviewers' *Unverified assumptions* are Discovery-phase instructions (validate full protobuf field boundaries; check for daemon file-rotation/temp files) — preserved verbatim below for the Discovery phase to action.

---

## CODEX Review (2026-05-30)

### Verified

- Read shared sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, and `.context/GLOSSARY.md`.
- Read SessionFlow DocVault pages `/Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/SessionFlow.md` and `/Volumes/DATA/GitHub/DocVault/KnowledgeBase/Architecture/SessionFlow MCP.md`.
- Checked live adapter behavior in `/Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py`: desktop and CLI variants share `_load_history()`; `discover_sources()` maps `project_root` from history only; `health()` still reports protobuf/database artifacts as unparsed.
- Checked existing Antigravity tests in `/Volumes/DATA/GitHub/SessionFlow/tests/test_provider_antigravity.py` and fixture setup in `/Volumes/DATA/GitHub/SessionFlow/tests/conftest.py`.
- Confirmed local Desktop state: no `~/.gemini/antigravity/history.jsonl`; `~/.gemini/antigravity/agyhub_summaries_proto.pb` exists and contains `file:///Volumes/DATA/GitHub/...` workspace strings; live desktop discovery currently returns 40 sources and 40 unknown `Project Root` values.

### Top concerns

- The observed desktop source count in the overview is stale relative to the current local inventory; keep it as context, not a fixed assertion.
- AC-2 needs URL-decoding/path-normalization semantics so implementation cannot pass with a fragile `file://` string strip.
- US-2 should cover provider health/limitations messaging, because the current diagnostics will be misleading once this root summary metadata is intentionally parsed.

### Unverified assumptions

- I verified that the summaries file contains UUID and workspace strings, but did not validate a complete protobuf schema or all field boundaries. Discovery should still prove the extraction rule against real samples before Approach locks it in.

## GEMINI Review (2026-05-30)

### Verified

- Verified that `~/.gemini/antigravity/agyhub_summaries_proto.pb` exists locally (size: 41,824 bytes) and contains 41 conversation summaries.
- Prototyped a schema-free binary protobuf parser in Python that decodes the binary file and extracts conversation UUIDs from Field 1, and workspace URIs from Field 2 -> Field 9 -> Field 1. Confirmed this extracts all mappings successfully.
- Verified that some conversations in the summaries file do not map to any workspace (returning `None`), confirming that AC-4 and AC-5 are necessary and must be handled.
- Confirmed that the current `provider_antigravity.py` health check reports `"Protobuf/database artifacts are not parsed in SESF-6."` which will need to be updated.

### Top concerns

- **Parsing Robustness and Graceful Fallbacks**: The binary structure of `agyhub_summaries_proto.pb` is subject to concurrent writes by the desktop daemon or partial/corrupt states during active sessions. The requirements must mandate that any file reading or binary decoding failures (such as `OSError`, `PermissionError`, or `IndexError`/`ValueError` during varint decoding) are handled silently, falling back to empty mappings rather than blocking ingestion.
- **Percent-Encoding and Spaces in Path Normalization**: In alignment with Codex's concern, the workspace path normalization in AC-2 must mandate proper URL decoding (e.g. handling spaces encoded as `%20` or Unicode characters) utilizing standard libraries like `urllib.parse`.
- **Parsing Performance**: Ingestion discovery loops may invoke `discover_sources()` frequently. The summaries metadata file should be loaded and parsed exactly once per `discover_sources()` execution rather than per-source to prevent redundant disk I/O.

### Unverified assumptions

- We assume the desktop daemon only writes to `agyhub_summaries_proto.pb` and doesn't rotate it under a different filename during active execution. Discovery should verify if there are temporary files or alternative locations used during active desktop logger writes.
