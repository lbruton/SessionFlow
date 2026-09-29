---
sketch: "SESF-17-desktop-project-detection"
phase: approach
created: 2026-05-28
updated: 2026-05-30
---

# SESF-17 — Approach

_How we'll build it. **Don't write code or tests** — the tasks phase produces the work plan._

## High-Level Architecture

The entire change lands in `provider_antigravity.py`, behind the existing desktop/CLI variant split. Today `discover_sources()` resolves each conversation's `project_root` from a single map (`_load_history()`), which is empty for Desktop — so every desktop source falls back to `"unknown"`. We add a **second resolution layer**: a schema-free protobuf reader for `~/.gemini/antigravity/agyhub_summaries_proto.pb` that returns a `{conversation_id: workspace_path}` map, parsed **once per discovery pass** and consulted only when `history.jsonl` has no entry. The precedence chain becomes `history → summaries → "unknown"` (AC-3), with the summaries layer gated to the desktop variant so the CLI path is provably untouched (AC-6).

The new code is three small, self-contained pieces, all module-level in `provider_antigravity.py`: (1) a **varint/length-delimited walker** that decodes the `.pb` wire format with stdlib only (no `protobuf`/`google` dependency — consistent with the SESF-6 non-goal and `requirements.txt`); (2) a **`file://` → filesystem-path normalizer** built on `urllib.parse` (`urlparse` + `unquote`) so percent-encoded paths decode to the literal path other providers would store (AC-2); and (3) a **summaries loader** that ties them together and wraps all file I/O and binary decoding in defensive error handling, returning `{}` on any failure (AC-5). The only other touched surface is `health()`, whose limitations string is reworded so the operator-facing diagnostics stop claiming the summaries file is unparsed while still naming the genuinely-opaque per-conversation `brain/**/*.pb` artifacts (AC-8).

No data model, no schema, no registry, and no search/query changes: `ProviderSource.project_root` stays a plain path string (AC-7), `http_server.py` already registers the desktop adapter, and existing rows are left as-is (re-tagging is an explicit non-goal). The work is purely a richer go-forward resolution inside one method.

## Key Decisions

| # | Decision | Rationale | Tradeoff |
|---|----------|-----------|----------|
| D-1 | **Stdlib schema-free varint walker** — decode the `.pb` with a hand-rolled length-delimited reader; no `protobuf`/`google` package | Honors the "don't parse the full schema" non-goal and keeps `requirements.txt` dependency-free; mirrors the validated reviewer prototypes | Brittle to wire-format reshuffles by the daemon — accepted because any decode failure degrades to `"unknown"` (AC-5), never a crash; the varint reader caps each value at 10 bytes (the max for a 64-bit varint) and raises `ValueError` beyond that, so a malformed/truncated stream can't spin in an infinite loop |
| D-2 | **Structured field-path extraction** — read conversation UUID from Field 1 and workspace URI via the nested Field 2→9→1 submessage path, decoding nested lengths recursively | Precision: avoids grabbing an unrelated `file://` string (e.g. a recent-folders list) and mis-tagging a conversation, which pollutes search worse than `"unknown"` does. The walker iterates **top-level length-delimited records** and does not assume a single wrapper-with-repeated-field; any record whose framing doesn't decode degrades to `"unknown"`, and tasks must confirm the observed top-level layout against the live `.pb` | Couples to the observed nesting; if the daemon moves the field, that conversation silently falls to `"unknown"` rather than mis-mapping (see Tradeoffs Surfaced) |
| D-3 | **Parse `.pb` once per `discover_sources()`**, inject as a precedence layer between history and the `"unknown"` default | One disk read + decode per discovery pass instead of per-source (carried forward from the requirements reconcile); matches the existing `_load_history()` once-per-pass shape | A long-running pass won't observe a mid-pass daemon rewrite — acceptable, the next pass picks it up |
| D-4 | **Gate summaries resolution to the desktop variant** explicitly (`if self.variant == "desktop"`), not just rely on the CLI root lacking the file | Guarantees AC-6 (CLI unchanged) by construction, independent of filesystem state | One extra branch; trivial |
| D-5 | **Helpers live module-level in `provider_antigravity.py`**, not in `provider_adapters.py` | They're Antigravity-desktop-specific and small; keeps the shared adapters module (home of generic `normalize_timestamp`/`canonicalize_path`) uncluttered — minimum-code | If a second provider ever needs `file://` normalization, it must be lifted to `provider_adapters.py` then |
| D-6 | **`file://` normalizer accepts decoded `file://` paths and bare absolute paths only** — any other string drops the mapping → `"unknown"` | Workspace URIs are *assumed* but not proven to always carry the scheme (GEMINI); a bare path that is absolute (starts with `/`) passes through unchanged, but relative/malformed/non-path strings are dropped so they can't poison project-scoped search (AC-7) | A real but mis-shaped path could be dropped to `"unknown"` rather than mapped — accepted, since a missing tag is recoverable while a wrong/non-path tag corrupts search |
| D-7 | **Variant-aware `health()` limitations, keep the protobuf assertion green** — the **desktop** variant says root summaries metadata is now parsed while per-conversation `brain/**/*.pb`/`*.db` artifacts stay opaque; the **CLI** variant keeps the baseline limitation string unchanged (it never parses summaries) | Satisfies AC-8 for desktop without advertising parsing the CLI never performs; the CLI's unchanged string keeps `test_..._does_not_claim_protobuf_support` green by construction | One branch in `health()` keyed on variant; both variants need test coverage, not just the current CLI assertion |

## File Map

_Every file this sketch will create, modify, or delete. Tasks.md will reference these paths._

### New
- _None._ The new logic is module-level functions added to the existing `provider_antigravity.py`; no new source module is warranted (D-5).

### Modified
- `provider_antigravity.py` — add the three helpers (varint/length-delimited walker, `file://` normalizer, summaries loader); insert the `history → summaries → "unknown"` precedence into `discover_sources()` (currently the single `history.get(conversation_id, "unknown")` at line 77), gated to the desktop variant (D-4); reword the `health()` limitations string (line 180) per AC-8 — variant-aware, so desktop reflects parsed summaries metadata while CLI keeps the baseline string (D-7).
- `tests/conftest.py` — add a synthetic **desktop** fixture (`brain/<uuid>/…/transcript.jsonl` dirs + a hand-built `agyhub_summaries_proto.pb`), including a malformed/truncated variant and one record with a percent-encoded `file://` path. The existing `synthetic_antigravity_home` fixture is CLI-only and grows no desktop/`.pb` coverage today.
- `tests/test_provider_antigravity.py` — add cases for AC-1 (mapped → workspace path), AC-2 (`file://` + `%20`/unicode decode), AC-3 (precedence ordering), AC-4 (unmapped → `"unknown"`), AC-5 (absent/locked/truncated/boundary-error → graceful `"unknown"`), AC-6 (CLI variant unchanged), AC-8 (health text updated — assert **both variants**: desktop reflects parsed summaries metadata, CLI keeps the baseline string). Keep `test_..._does_not_claim_protobuf_support` green — the CLI wording is unchanged, so it passes without modification (D-7).

### Deleted
- _None._

## Data / Schema Changes

None — no schema, migration, or persisted-data changes. `project_root` remains a `str` path on `ProviderSource` (AC-7). Already-indexed `"unknown"` rows are not backfilled or re-tagged (explicit non-goal).

## Tradeoffs Surfaced for Review

- **D-2 (structured field-path vs. tolerant first-`file://`-string).** This is the one genuinely reversible design call. The structured path (Field 2→9→1) is precise but couples to the daemon's current nesting; the tolerant alternative ("first length-delimited `file://` string inside each conversation record") survives field reshuffles but risks mapping a conversation to the wrong workspace if the record carries more than one path-like string. I've chosen **precise + fail-to-`unknown`** on the principle that a *missing* tag is recoverable (re-tag follow-up) while a *wrong* tag silently corrupts project-scoped search. Flagging in case you'd rather bias toward maximal coverage and accept occasional mis-maps.
- **Coverage is best-effort by design.** The `39/41` local-join figure is a **discovery-time snapshot, not a success criterion** — the live desktop source count drifts as new conversations appear (CODEX re-ran pre-change discovery on 2026-05-30 and found more sources, all still `"unknown"`). A residual handful stay `"unknown"` (no workspace in the metadata), which is per AC-4, not a defect. Tasks/tests assert synthetic mapped/unmapped behavior; local counts stay illustrative only.

## UI Contract

N/A — no UI surface. Ingestion-side only; no mockup, playground, or screen referenced in requirements or discovery.

## Out of Scope (follow-up issues)

- **Backfill / re-tag existing `unknown` desktop rows** — go-forward only here; re-indexing the already-stored rows is a separate SessionFlow issue (requirements non-goal).
- **Secondary fallback heuristic for the unmapped ~2** (e.g. infer workspace from tool-call file paths in the transcript) — deliberately excluded; file separately if the residual `unknown` count proves annoying.
- **Parsing the full Antigravity protobuf schema / per-conversation `brain/**/*.pb` artifacts** — remains opaque per SESF-6.

## Risk Notes

- **Risk: the `.pb` is rewritten in place by the live daemon** (size drift observed across reads; no temp/lock siblings at rest) → a discovery read can race a truncated/short file. **Mitigation:** D-1's loader catches `OSError`/`PermissionError` and binary-boundary errors (`IndexError`/`ValueError`) plus `UnicodeDecodeError` during string decode, returning `{}` (AC-5). The malformed-fixture test exercises this path.
- **Risk: percent-decoding (AC-2) cannot be validated against the live file** — all local workspace paths are clean ASCII. **Mitigation:** the synthetic fixture must include an encoded path (`%20`/unicode) so the decode branch is actually exercised.
- **Risk: AC-8 rewording accidentally weakens the protobuf-support test.** **Mitigation:** new limitations text must still name the opaque `brain/**/*.pb`/`*.db` artifacts so `test_..._does_not_claim_protobuf_support` stays meaningful, not merely string-satisfied (D-7).

---

> **Phase complete?** Architecture clear, decisions logged with rationale, file map complete. Then advance: `/sketch-tasks SESF-17`.

---

## Review Archive — approach (2026-05-30)

### Resolution Summary

Reconciled 2026-05-30 via `/sketch-reconcile SESF-17 approach`. **5 accepted / 0 rejected / 0 resolved with your input.** CODEX and GEMINI agreed on the two consensus items (D-6, D-7); no divergence.

- **Accepted:** (A) D-1 — cap each varint at 10 bytes (max 64-bit varint), raising `ValueError` past that, to prevent infinite loops on malformed/truncated streams (GEMINI); (B) D-2 — documented that the walker iterates top-level length-delimited records and does not assume a single wrapper-with-repeated-field, with tasks to confirm the live layout (GEMINI); (C) D-6 — tightened the normalizer to accept decoded `file://` paths and bare absolute paths only, dropping any other string to `"unknown"` to protect the AC-7 path contract (CODEX + GEMINI, consensus); (D) D-7 — made `health()` wording variant-aware (desktop reflects parsed summaries, CLI keeps the baseline string), with both-variant test coverage so `test_..._does_not_claim_protobuf_support` stays green unmodified (CODEX + GEMINI, consensus); (E) reframed the `39/41` join figure as a discovery-time snapshot, not a success criterion, with tasks/tests asserting synthetic behavior and local counts illustrative only (CODEX).
- **Rejected:** none.
- **Your input:** none — all findings were consensus tightenings or factual/mechanical corrections. The author-flagged D-2 tradeoff (precise field-path vs. tolerant first-`file://`) drew no reviewer objection and was left as authored.

The verbatim reviewer sections below are preserved as the audit trail.

## CODEX Review (2026-05-30)

### Verified

- Read `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, and the reconciled SESF-17 `requirements.md` / `discovery.md`.
- Checked live `provider_antigravity.py:22-81` for the desktop/CLI root split, `_load_history()`, the current `history.get(conversation_id, "unknown")` injection point, and `provider_antigravity.py:172-190` for shared health/limitations behavior.
- Checked `provider_adapters.py:76-110`, `provider_adapters.py:122-146`, `provider_adapters.py:186-200`, `tests/test_provider_antigravity.py:23-34`, `tests/conftest.py:190-207`, and `requirements.txt`.
- Re-ran live desktop discovery with the current code: `AntigravityAdapter(source_kind="desktop").discover_sources()` returns 42 sources and 42 `project_root == "unknown"` values before this sketch is implemented.
- Confirmed the approach file map stays scoped to `provider_antigravity.py`, `tests/conftest.py`, and `tests/test_provider_antigravity.py`, matching the requirements/discovery boundary.

### Top concerns

- D-6's raw fallback needs to accept only bare absolute paths; accepting arbitrary non-`file://` strings would violate the Project Root path contract and risks wrong project-scoped search results.
- D-7 should specify variant-aware `health()` wording and tests because the method is shared by CLI and Desktop but summaries parsing is Desktop-only.
- The `39/41` join count in Tradeoffs should be framed as a discovery snapshot; current live source count already differs.

### Unverified assumptions

- I did not re-parse the live `agyhub_summaries_proto.pb`; this review relies on the reconciled discovery's structured field-path validation and only rechecked the current source count/unknown baseline.

## GEMINI Review (2026-05-30)

### Verified

- Checked [/Volumes/DATA/GitHub/DocVault/sketch/conventions.md](file:///Volumes/DATA/GitHub/DocVault/sketch/conventions.md), SessionFlow [AGENTS.md](file:///Volumes/DATA/GitHub/SessionFlow/AGENTS.md), and the reconciled SESF-17 [requirements.md](file:///Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/sketches/SESF-17-desktop-project-detection/requirements.md) / [discovery.md](file:///Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/sketches/SESF-17-desktop-project-detection/discovery.md).
- Verified current codebase files and paths listed in the File Map:
  - [provider_antigravity.py](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py)
  - [tests/conftest.py](file:///Volumes/DATA/GitHub/SessionFlow/tests/conftest.py)
  - [tests/test_provider_antigravity.py](file:///Volumes/DATA/GitHub/SessionFlow/tests/test_provider_antigravity.py)
- Verified unit test suite passes successfully.
- Confirmed that the proposed approach has no schema migrations or registry updates, keeping the code changes strictly contained within the adapter.

### Top concerns

- **Varint Loop Safety (D-1)**: Enforce a limit of 10 bytes inside the varint decoder loop to prevent CPU hang or infinite loop execution when parsing malformed/truncated protobuf streams.
- **Strict Absolute Path Validation (D-6)**: Validate that decoded path strings are absolute paths (start with `/`) to maintain the `project_root` path contract and prevent arbitrary strings from polluting project searches.
- **Top-Level Wire Framing Documentation (D-2)**: Clearly document the top-level layout of the summaries file (repeated elements vs. single parent wrapper) so the schema-free walker behaves correctly when encountering multiple conversation records.

### Unverified assumptions

- We assume that the desktop daemon does not record multiple URIs per conversation or write multi-part workspace paths within Field 2->9->1.
