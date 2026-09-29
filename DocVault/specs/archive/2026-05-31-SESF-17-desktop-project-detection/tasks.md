---
sketch: SESF-17-desktop-project-detection
phase: tasks
created: 2026-05-28
approved: 2026-05-30
---

# SESF-17 — Tasks

_Concrete checklist grouped into Sprint Cohorts. `[P]` marks tasks that can run in parallel within a cohort. Tasks reference file paths from approach.md._

> **Approval gate:** `/sketch run` refuses to run unless the `approved:` frontmatter field above contains a date in `YYYY-MM-DD` format, ≤14 days old. Stamp it only after reviewing all four sketch files.

> **Skill-name discipline:** Closing tasks below name specific skills (`/release patch`, `/vault-update`, `codacy-cli`, `/sketch archive`). When generating or executing tasks.md, **invoke skills by name verbatim** — paraphrasing the steps inline is not equivalent because the skill enforces project-specific rules the prose can't carry. If a closing task is genuinely irrelevant to the project (e.g., no version management, no foundation docs touched), mark it `N/A — <one-line reason>` rather than dropping the task. Audible skips beat silent skips.

## Sprint Cohort 0 — Setup (sequential)

_Cohort 0 ensures the worktree exists before implementation begins. If the worktree is missing, the executing agent creates it — this is setup work, not a stop-the-world gate._

- [x] **0.1** — Ensure sketch worktree exists
  - **File(s):** _no file changes — verification only_
  - **Acceptance:** `git worktree list` shows a worktree on branch `sketch/SESF-17-desktop-project-detection` off `main`. Working directory is that worktree, not the main checkout. SessionFlow has no project-specific setup skill (`.context/sketch-conventions.md` is absent; `AGENTS.md` only states "Worktree branches for all changes" + "signed commits + PR required"), so the generic branch name applies.
  - **If the worktree does not exist yet:** Create it via the `using-git-worktrees` skill. This is expected on the first run — the agent satisfies this task by doing the setup, not by stopping.
  - **Leverage:** `using-git-worktrees` skill; `AGENTS.md:48-50` (branch protections: signed commits + PR required).

## Sprint Cohort A — Foundation (parallel-safe)

_Scaffolding only — the test fixtures and importable stubs Cohort B needs. No behavioral logic yet. A.1 and A.2 touch disjoint files and create no shared symbols, so they are genuinely `[P]`._

- [x] **A.1 [P]** — Synthetic desktop Antigravity fixture (`.pb` + brain dirs)
  - **File(s):** `tests/conftest.py`
  - **Acceptance:** A new fixture (e.g. `synthetic_antigravity_desktop_home`) materializes a desktop root with: (a) several `brain/<uuid>/.system_generated/logs/transcript.jsonl` conversations; (b) a hand-built **valid** `agyhub_summaries_proto.pb` mapping a subset of those `conversation_id`s to `file://` workspace URIs via the Field 1 / Field 2→9→1 structure (per discovery); (c) at least one conversation **present in brain but absent from the `.pb`** (exercises AC-4); (d) one record carrying a **percent-encoded** `file://` path (e.g. a `%20` space and a unicode char) to exercise AC-2's decode branch; (e) a separately-addressable **malformed/truncated** `.pb` variant (e.g. a short read of the valid bytes) for AC-5. Fixture is importable and does not depend on the live `~/.gemini` tree.
  - **Leverage:** `tests/conftest.py:190-207` (`synthetic_antigravity_home`, CLI-only — mirror its shape); discovery.md "Field-boundary proof" + the Protobuf encoding notes for the hand-built `.pb` byte layout; GEMINI's validated Field 1 / Field 2→9→1 extraction path. **Wire-format tags for the hand-built bytes** (all wire type 2, length-delimited): Field 1 → `0x0A`, Field 2 (submessage) → `0x12`, Field 9 (nested submessage) → `0x4A`, nested Field 1 (workspace URI string) → `0x0A`. Specifying these lets the implementer construct the fixture bytes without guesswork.
  - **Maps to:** AC-1, AC-2, AC-4, AC-5 (provides the data these tests assert against)

- [x] **A.2 [P]** — Stub the three helpers + desktop-gated resolution branch
  - **File(s):** `provider_antigravity.py`
  - **Acceptance:** Three module-level helper stubs exist and are importable — a varint/length-delimited `.pb` walker, a `file://`→path normalizer, and a summaries loader returning `Dict[str, str]` — each currently returning the empty/no-op result (`{}` / unchanged input) with **no parsing logic**. `discover_sources()` gains the desktop-gated call site shape (e.g. `if self.variant == "desktop"`) wired to the (stub) loader so the module imports and current behavior is preserved (`project_root` still resolves to `"unknown"` for desktop). This is scaffolding so Cohort B tests fail on assertions, not on `ImportError`/`AttributeError` at collection.
  - **Leverage:** `provider_antigravity.py:61-81` (`discover_sources()` injection point, current `history.get(conversation_id, "unknown")` at line 77); `provider_antigravity.py:37-59` (`_load_history()` once-per-pass shape to mirror); `provider_adapters.py:88-110` (`normalize_timestamp` precedent for a stdlib coercion helper).
  - **Maps to:** AC-1, AC-3 (establishes the precedence call site and helper surface)

## Sprint Cohort B — Tests · RED (sequential)

_TDD red phase. One task — all cases land in the single `tests/test_provider_antigravity.py`, so they cannot be split into `[P]` siblings without colliding on the file. Tests MUST fail at this point because the helpers from A.2 are still no-op stubs._

- [x] **B.1** — Write failing tests for desktop summaries-based `project_root` resolution
  - **File(s):** `tests/test_provider_antigravity.py`
  - **Acceptance:** Each testable AC has at least one failing test, all using the A.1 fixture:
    - **AC-1** — desktop conversation whose `conversation_id` is in the `.pb` resolves `project_root` to the mapped workspace **path**.
    - **AC-2** — a `file://` URI with `%20`/unicode is normalized via `urllib.parse` (`urlparse`+`unquote`) to the literal decoded filesystem path (assert the exact decoded path, not a scheme strip).
    - **AC-3** — precedence `history.jsonl` → summaries → `"unknown"`: when both a (synthetic) history entry and a summaries entry exist for a conversation, history wins; when only summaries exists, summaries wins.
    - **AC-4** — a `conversation_id` present in brain but absent from the `.pb` (and absent from history) resolves to `"unknown"`.
    - **AC-5** — absent / unreadable (`PermissionError`/`OSError`) / truncated / boundary-error (`IndexError`/`ValueError`) / `UnicodeDecodeError` `.pb` → loader returns `{}` and desktop sources resolve to `"unknown"` **without raising** (use the malformed fixture variant + monkeypatched-error cases).
    - **AC-6** — the `antigravity_cli` variant's `discover_sources()` resolution is unchanged (no summaries consulted; behavior identical to pre-sketch).
    - **AC-7** — `ProviderSource.project_root` remains a `str` path; assert no new project-name field is introduced and the normalizer drops non-absolute / non-`file://` junk to `"unknown"` (D-6) rather than storing a derived name.
    - **AC-8** — assert **both variants** of `health()`: desktop limitations text no longer claims the summaries metadata is unparsed while still naming the opaque `brain/**/*.pb`/`*.db` artifacts; CLI keeps the baseline string. Keep `test_antigravity_adapter_does_not_claim_protobuf_support` green (CLI wording unchanged).
  - **Depends on:** A.1 (fixture), A.2 (importable stubs)
  - **Leverage:** `tests/test_provider_antigravity.py:23-34` (`test_..._does_not_claim_protobuf_support` — the AC-8 anchor to keep green); discovery.md validation notes (39/41 join is illustrative — assert synthetic mapped/unmapped behavior, **not** live counts).
  - **Maps to:** AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8

## Sprint Cohort C — Implementation · GREEN (sequential)

_TDD green phase. Minimum code to turn Cohort B green. All three tasks edit `provider_antigravity.py`, so none are `[P]` — they run in sequence._

- [x] **C.1** — Implement the three stdlib helpers
  - **File(s):** `provider_antigravity.py`
  - **Acceptance:** Replace the A.2 stubs with working logic: (1) a schema-free **varint / length-delimited walker** decoding the `.pb` wire format with stdlib only (no `protobuf`/`google` dep), capping each varint at **10 bytes** and raising `ValueError` past that to prevent infinite loops on truncated streams (D-1); iterating **top-level length-delimited records** (D-2, not assuming a single wrapper) and decoding the nested Field 2→9→1 submessage path recursively; tolerating `UnicodeDecodeError` during string decode. (2) a **`file://` normalizer** built on `urllib.parse` (`urlparse`+`unquote`) that accepts decoded `file://` paths and bare **absolute** paths only, dropping anything else (D-6). (3) a **summaries loader** tying them together, wrapping all file I/O + binary decoding so any failure (`OSError`/`PermissionError`/`IndexError`/`ValueError`/`UnicodeDecodeError`) returns `{}` (AC-5). C.1 turns **only the helper-level tests green** — Cohort B's AC-2 (normalizer decodes `file://`/percent-encoded paths) and AC-5 (loader returns `{}` on malformed/unreadable input). The source-resolution assertions AC-1/AC-4 stay RED here by design: B.1 defines them as `discover_sources()` behavior, and the summaries map is not wired into `discover_sources()` until C.2. Do not smuggle the C.2 resolver change into C.1.
  - **Depends on:** B.1
  - **Leverage:** discovery.md External References (Protobuf encoding wire format; `int.from_bytes`/manual varint loop); `provider_adapters.py:88-110` (`normalize_timestamp` style); GEMINI/CODEX validated Field 1 / Field 2→9→1 extraction.
  - **Maps to:** AC-2, AC-5

- [x] **C.2** — Wire `history → summaries → "unknown"` precedence into `discover_sources()`
  - **File(s):** `provider_antigravity.py`
  - **Acceptance:** `discover_sources()` loads the summaries map **once per pass** (mirroring `_load_history()`), gated to `self.variant == "desktop"` (D-3/D-4). Per source, `project_root` resolves history first, then summaries, then `"unknown"` (replaces the bare `history.get(conversation_id, "unknown")`). CLI variant never consults summaries. Wiring the summaries map into `discover_sources()` here is what turns the source-resolution tests green — Cohort B's AC-1 (mapped path) and AC-4 (absent → `"unknown"`) go green at this task, alongside AC-3, AC-6, AC-7.
  - **Depends on:** C.1
  - **Leverage:** `provider_antigravity.py:61-81` (line 77 is the resolution line being replaced; line 63 is where history loads once); SESF-19 precedence precedent (mem0 `9f20b491`).
  - **Maps to:** AC-1, AC-3, AC-4, AC-6, AC-7

- [x] **C.3** — Variant-aware `health()` limitations text
  - **File(s):** `provider_antigravity.py`
  - **Acceptance:** `health()` limitations become variant-aware: the **desktop** variant's text reflects that the root summaries metadata is now parsed while per-conversation `brain/**/*.pb`/`*.db` artifacts stay opaque; the **CLI** variant keeps the baseline string unchanged (D-7). `test_..._does_not_claim_protobuf_support` stays green unmodified; Cohort B's AC-8 (both variants) passes.
  - **Depends on:** B.1
  - **Leverage:** `provider_antigravity.py:178-190` (`health()`, the literal limitations string at line 180); `provider_antigravity.py:172-173` (`_has_opaque_binary_artifacts()` — the still-opaque brain artifacts to keep naming).
  - **Maps to:** AC-8

---

## Standard Closing Tasks

- [x] **CLOSE-1. Run full test suite** — zero regressions
  - **File:** _no file changes — verification only_
  - Run `./venv/bin/pytest` (the project's complete test command; 124 tests pre-sketch per the reconciled reviews). All existing tests pass; all new Cohort B tests pass (green after Cohort C).
  - If anything fails: fix the implementation, not the test. Tests are the spec — a failing test means the implementation is wrong, not the test.

- [x] **CLOSE-2. Codacy CLI scan** — security + quality
  - **File:** _no file changes — scan only_
  - Run `codacy-cli` skill against changed files (`provider_antigravity.py`, `tests/conftest.py`, `tests/test_provider_antigravity.py`). Triage: Critical/High must fix, Medium fix-or-document, Low/Info advisory.

- [x] **CLOSE-3. Generate verification stamp**
  - **File:** Append to bottom of `tasks.md` (this file) under heading `## Verification Stamp`.
  - For EACH acceptance criterion in `requirements.md` (AC-1 … AC-8), write exactly one line in one of these formats:
    - `- [x] AC-N — verified at <relative/file/path>:<line>` (cite the test or implementation line that proves it), OR
    - `- [x] AC-N — verified by <test name>` (when an entire named test case proves it), OR
    - `- [ ] AC-N — gap: <one-line reason>` (when not yet verified).
  - The stamp block MUST list every AC from requirements.md (AC-1 through AC-8). Status-only notes ("PR opened", "tests passing") are not equivalent — the gate requires per-AC traceability.
  - `approach.md`'s `## UI Contract` is `N/A — no UI surface` (ingestion-side only), so no visual-verification evidence is required on any stamp line.
  - Refuse to proceed to CLOSE-4 if any `[ ]` remains in the stamp block.

- [x] **CLOSE-4. Version bump**
  - **File:** _N/A_
  - `N/A — project opts out of version management` (no `pyproject.toml`, `VERSION`, `__version__`, or `devops/version.lock` exists in SessionFlow; confirmed during `/sketch-tasks`). Do not invoke `/release patch`.

- [x] **CLOSE-5. Vault update** — `/vault-update` audit run: zero foundation-doc changes (Overview/SessionFlow/SessionFlow MCP make no claim contradicted by this ingestion-side change). DocVault sketch-progress commit deferred to CLOSE-8 archive (direct push to DocVault `main` needs user auth).
  - **MUST invoke `/vault-update`** as a skill — even if you believe no foundation docs are affected, the skill performs the audit. If no docs need updating, the skill reports zero changes and the task is done.
  - **Do not** mark SESF-17 Done here. SessionFlow `AGENTS.md` requires signed commits + a merged PR for code changes; CLOSE-6 opens the PR, CLOSE-7 resolves review, and CLOSE-8 waits for merge. The Plane "Done" transition is deferred to CLOSE-8 so the issue isn't marked shipped while review/merge is still pending.

- [x] **CLOSE-6. Open PR** — https://github.com/lbruton/SessionFlow/pull/22
  - Use worktree branch `sketch/SESF-17-desktop-project-detection`. Title: `feat(SESF-17): resolve desktop project_root from antigravity summaries metadata`.
  - Body must include: link to source issue ([SESF-17](https://plane.lbruton.cc/lbruton/browse/SESF-17/)), link to sketch folder (`DocVault/Projects/SessionFlow/sketches/SESF-17-desktop-project-detection/`), test plan checklist. Signed commits + PR required (`AGENTS.md:49`).

- [x] **CLOSE-7. Resolve PR review threads** — all 7 findings triaged; HIGH wire-type bug fixed (`8495a21`) + regression test; 6 fixed, 1 acknowledged (Windows out-of-scope); 0 unresolved threads.
  - **File:** _GitHub PR threads only — code fixes land via the worktree as needed_
  - **MUST invoke `/pr-resolve`** as a skill — it triages findings, fix-or-classifies, and replies to threads consistently. Scan **both** inline diff threads AND review-body findings (Codacy/Copilot often post critical findings as summary-style review prose, not inline threads).
  - All Critical/High findings must be either fixed or marked false-positive with explicit reasoning. Medium: fix or document waiver. Low/Info: advisory. Re-check for new auto-scanner threads after each commit.

- [x] **CLOSE-8. Archive sketch + close issue** (after PR merges) — PR #22 merged `1d49b98` 2026-05-31; archived; SESF-17 → Done.
  - **MUST invoke `/sketch archive SESF-17`** as a skill — moves folder to `archive/YYYY-MM-DD-SESF-17-desktop-project-detection/` and saves mem0 summary.
  - **Mark the source issue Done in Plane** (deferred from CLOSE-5): `mcp__plane__update_issue` on SESF-17 to state "Done". Run this only after the PR has merged and the sketch is archived.

---

> **Multi-model dispatch hint:** The only genuine parallel seam here is Cohort A — A.1 (`tests/conftest.py` fixture) and A.2 (`provider_antigravity.py` stubs) touch disjoint files and can go to two models at once. Reconverge before Cohort B. Cohorts B (tests) and C (implementation) are single-file-bound and sequential; the B→C boundary is the natural model-routing seam (one model writes the failing tests, another makes them green).

## Review Archive — tasks (2026-05-30)

### Resolution Summary

- **Accepted: 4** — (1) A.1 protobuf wire tags folded into leverage [GEMINI]; (2) C.1/C.2 TDD boundary — C.1 narrowed to helper-level AC-2/AC-5, AC-1/AC-4 moved to C.2 [CODEX+GEMINI consensus]; (3) CLOSE-3 UI Contract premise corrected — `approach.md` does have a `## UI Contract` (`N/A`) [CODEX, verified against approach.md:56]; (4) CLOSE-5 premature closeout — Plane "Done" moved to CLOSE-8 (post-merge) [CODEX+GEMINI consensus].
- **Rejected: 0**
- **Resolved with your input: 0** — no judgment/tradeoff items; all findings were correct + mechanical.

Original reviewer sections preserved verbatim below.

### CODEX Review (2026-05-31)

#### Verified

- Read shared sketch conventions at `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, and the reconciled SESF-17 `requirements.md`, `discovery.md`, and `approach.md`.
- Read relevant DocVault context: `/Volumes/DATA/GitHub/DocVault/INDEX.md`, `/Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/SessionFlow.md`, `/Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/Overview.md`, and `/Volumes/DATA/GitHub/DocVault/KnowledgeBase/Architecture/SessionFlow MCP.md`.
- Checked live `provider_antigravity.py:37-81` for current history-only resolution and the `project_root=history.get(conversation_id, "unknown")` call site; checked `provider_antigravity.py:172-190` for current opaque-artifact health text.
- Checked `tests/conftest.py:190-207` and `tests/test_provider_antigravity.py:23-34` to confirm the tasks' fixture/test anchors are still current.
- Confirmed no SessionFlow project-specific `.context/sketch-conventions.md` exists and no version-management files were found by the requested patterns (`pyproject.toml`, `VERSION`, `version.lock`).

#### Top concerns

- C.1 claims AC-1/AC-4 source-resolution tests should pass before C.2 wires the summaries map into `discover_sources()`, which breaks the intended RED/GREEN task boundary.
- CLOSE-5 marks SESF-17 Done before the PR is opened, reviewed, and merged, contradicting the repo's PR-required code-change flow.
- CLOSE-3's visual-verification note is directionally right but cites a false premise: `approach.md` has a `## UI Contract` section with `N/A`, not no section.

#### Unverified assumptions

- I did not run the test suite because this is a review-only pass over `tasks.md`.
- I did not re-parse the live `agyhub_summaries_proto.pb`; the tasks review relies on the reconciled discovery/approach validation for the protobuf field-path facts.

### GEMINI Review (2026-05-30)

#### Verified

- Read shared sketch conventions at [conventions.md](file:///Volumes/DATA/GitHub/DocVault/sketch/conventions.md), SessionFlow [AGENTS.md](file:///Volumes/DATA/GitHub/SessionFlow/AGENTS.md), `.context/GLOSSARY.md`, and the reconciled requirements, discovery, and approach documents for SESF-17.
- Checked [provider_antigravity.py](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py) to confirm entry points for summaries loading (`discover_sources()` at line 61, `_load_history()` at line 37, and `health()` at line 178).
- Checked [tests/conftest.py](file:///Volumes/DATA/GitHub/SessionFlow/tests/conftest.py) (lines 190-207) to confirm CLI fixture structure, and [test_provider_antigravity.py](file:///Volumes/DATA/GitHub/SessionFlow/tests/test_provider_antigravity.py) (lines 1-35) to verify the test suite baseline.
- Checked [.specflow/config.json](file:///Volumes/DATA/GitHub/SessionFlow/.specflow/config.json) to confirm issue backend is set to Plane.

#### Top concerns

- **TDD Boundary Violation (C.1 vs C.2)**: As CODEX noted, expecting integration/resolution tests (AC-1/AC-4) to pass in C.1 before C.2 wires the summaries map into `discover_sources()` breaks the TDD flow. The expectations for C.1 must be restricted to helper-level unit tests.
- **Premature Issue Closeout**: Marking the issue Done in Plane in CLOSE-5 violates the PR-required workflow. We should update the task list to defer the Plane status change to CLOSE-8 (or a new CLOSE-9 task) to happen after the PR is merged and the sketch is archived.
- **Protobuf wire-tag clarity**: Task A.1 should explicitly list the Protobuf wire-format tags (Field 1: `0x0A`, Field 2: `0x12`, Field 9: `0x4A`, Field 1: `0x0A`) to simplify the manually encoded byte construction for the synthetic fixture.

#### Unverified assumptions

- I did not run the test suite or compile code because this is a read-only review task.
- I did not verify the behavior of the Plane API or construct the summaries file myself.

---

## Verification Stamp

_Generated 2026-05-30 (CLOSE-3). Full suite: 137 passed, 0 failed. `approach.md` `## UI Contract` is `N/A — no UI surface`, so no visual-verification evidence is required._

- [x] AC-1 — verified by `test_desktop_summaries_maps_conversation_to_workspace_path_ac1` (desktop conversation in `.pb` resolves `project_root` to the mapped workspace path)
- [x] AC-2 — verified by `test_desktop_summaries_normalizes_percent_encoded_file_uri_ac2` (asserts exact decoded path `/Users/lbruton/My Projects/café` via `urlparse`+`unquote`)
- [x] AC-3 — verified by `test_desktop_resolution_precedence_history_over_summaries_ac3` + `test_desktop_resolution_precedence_summaries_when_no_history_ac3` (history wins when present; summaries wins otherwise)
- [x] AC-4 — verified by `test_desktop_unmapped_conversation_resolves_unknown_ac4` (brain conversation absent from `.pb` → `"unknown"`)
- [x] AC-5 — verified by `test_desktop_absent_pb_resolves_unknown_without_raising_ac5`, `test_desktop_truncated_pb_resolves_unknown_without_raising_ac5`, `test_desktop_unreadable_pb_resolves_unknown_without_raising_ac5`, `test_desktop_decode_error_pb_resolves_unknown_without_raising_ac5` (loader returns `{}` on every failure mode, no raise)
- [x] AC-6 — verified by `test_cli_variant_resolution_unchanged_no_summaries_ac6` (CLI variant never consults summaries; `_load_summaries` monkeypatched to fail if called)
- [x] AC-7 — verified by `test_project_root_remains_str_path_no_project_name_field_ac7` (`project_root` stays a `str` path; no project-name field; D-6 absolute-path-or-`"unknown"` contract held) and `provider_antigravity.py` `_normalize_file_uri` (drops non-absolute / non-`file://` strings to `None`)
- [x] AC-8 — verified by `test_desktop_health_reflects_summaries_parsed_ac8` + `test_cli_health_keeps_baseline_protobuf_string_ac8` + `test_antigravity_adapter_does_not_claim_protobuf_support` (desktop limitations reflect parsed summaries while naming opaque `brain/**/*.pb`/`*.db`; CLI baseline string unchanged and anchor stays green)
