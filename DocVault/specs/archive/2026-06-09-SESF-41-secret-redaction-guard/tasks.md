---
sketch: SESF-41-secret-redaction-guard
phase: tasks
created: 2026-06-08
approved: 2026-06-08
---

# SESF-41 — Tasks

_Concrete checklist grouped into Sprint Cohorts. `[P]` marks tasks that can run in parallel within a cohort. Tasks reference file paths from approach.md._

> **Approval gate:** `/sketch run` refuses to run unless the `approved:` frontmatter field above contains a date in `YYYY-MM-DD` format, ≤14 days old. Stamp it only after reviewing all four sketch files.

> **Skill-name discipline:** Closing tasks below name specific skills (`/release patch`, `/vault-update`, `codacy-analysis-cli`, `/sketch archive`). When generating or executing tasks.md, **invoke skills by name verbatim** — paraphrasing the steps inline is not equivalent because the skill enforces project-specific rules the prose can't carry. If a closing task is genuinely irrelevant to the project, mark it `N/A — <one-line reason>` rather than dropping the task. Audible skips beat silent skips.

## Sprint Cohort 0 — Setup (sequential)

_Cohort 0 ensures the worktree exists before implementation begins. If the worktree is missing, the executing agent creates it — this is setup work, not a stop-the-world gate._

- [x] **0.1** — Ensure sketch worktree exists
  - **File(s):** _no file changes — verification only_
  - **Acceptance:** `git worktree list` shows a worktree on branch `sketch/SESF-41-secret-redaction-guard`. Working directory is that worktree, not the main `/Volumes/DATA/GitHub/SessionFlow` checkout. SessionFlow has **no** project-local worktree skill and no `.context/sketch-conventions.md`, so the generic convention applies.
  - **If the worktree does not exist yet:** Create it via the `using-git-worktrees` skill, branch `sketch/SESF-41-secret-redaction-guard` off `main`. This is expected on the first run — satisfy the task by doing the setup, not by stopping.
  - **Leverage:** `using-git-worktrees` skill. SessionFlow git rules (AGENTS.md): `main` default, signed commits + PR required, worktree branches for all changes.

## Sprint Cohort A — Foundation (parallel-safe)

_Independent files, no behavioral logic — dependency wiring and the pure-module skeleton. Dispatchable concurrently._

- [x] **A.1 [P]** — Add `detect-secrets` as a runtime dependency and verify the import
  - **File(s):** `requirements.txt`
  - **Acceptance:** `detect-secrets` is listed in `requirements.txt` (NOT `requirements-dev.txt` — it is a runtime dep). After `./venv/bin/pip install -r requirements.txt`, `./venv/bin/python -c "import detect_secrets"` exits 0. Pin or floor the version at `>=1.5.0` (the version the D-1 plugin mapping was verified against).
  - **Leverage:** approach.md File Map (Modified → `requirements.txt`); approach Risk Note ("`detect-secrets` not installed in venv → add as runtime dep"); venv at `./venv/` (AGENTS.md Tech Stack).
  - **Maps to:** AC-5 (Tier-1 plugin engine dependency), AC-16 (no-network library — keep verification policies off)

- [x] **A.2 [P]** — Scaffold the pure `secret_redaction.py` module (skeleton only, no detection logic)
  - **File(s):** `secret_redaction.py` (new)
  - **Acceptance:** Module exists with a module docstring (ruff `D1`); a `Hit` type (e.g. `NamedTuple`/dataclass) carrying `rule_name: str` and `tier: int` and **no field that exposes a raw secret value** to reporting consumers (AC-18); the placeholder-format helper/constant rendering `[REDACTED:<RULE_NAME>]` with fallback `[REDACTED:SECRET]` (AC-4); and the public signature `def redact(text, *, mode, allowlist=None) -> tuple[str, list[Hit]]` with a Google-style docstring and a stub body that returns `(text, [])` unchanged. No `detect_secrets` import yet, no I/O (AC-16). The stub makes Cohort B tests import-able while still failing (RED).
  - **Leverage:** `tests/test_issue_id_extraction.py` (pure-function analog); `ruff.toml` (`D1`, Google convention — module + public-`redact` docstrings, private helpers exempt); approach D-2 (placeholder shape), AC-4 (fallback name).
  - **Maps to:** AC-4, AC-16

## Sprint Cohort B — Tests · RED (sequential)

_TDD red phase. Encode each EARS AC as a failing test BEFORE implementation. All new tests live in `tests/test_secret_redaction.py` (the only test file in approach.md's File Map). Synthetic, non-functional tokens only — no raw secret values anywhere (AC-18). Tests MUST fail at this point: the `redact` stub (A.2) returns text unchanged and no hook is wired yet._

- [x] **B.1** — Failing tests for the pure `redact` engine (detector + masker)
  - **File(s):** `tests/test_secret_redaction.py` (new)
  - **Acceptance:** At least one failing assertion per pure-function AC:
    - **AC-5** — one case per Tier-1 class, asserting the correct rule name: OpenAI, Anthropic, GitHub, AWS, Google, Slack, Stripe, GitLab, HuggingFace, npm, PEM private-key block, JWT. (Map each class to its owning detector per the approach **D-1 plugin mapping** — built-in plugins vs. custom Tier-1 regexes for Anthropic/Google/HuggingFace; JWT is a built-in.)
    - **AC-6** — Tier-2 contextual `key: value` / `key=value` assignments incl. the MCP-config-dump leak class.
    - **AC-7** — placeholder-shape values (`***`, `<your-key>`, `changeme`, `xxxx`) and sub-minimum-length values are NOT flagged.
    - **AC-8** — high-entropy token adjacent to a Tier-2 keyword (same line, ≤40 chars) or inside a ```` ``` ````-fenced block is redacted; an isolated entropy hit is reported (in `hits`) but NOT masked.
    - **AC-9** — allowlist exemptions: UUID, 40-hex git SHA, 64-hex SHA-256, a content-hash `doc_id`, an issue ID, and an operator-supplied pattern (passed via the `allowlist` arg) are NOT flagged.
    - **AC-14** — determinism: `redact(x)` over identical bytes yields byte-identical output (no random salt); two inputs differing only in secret values yield identical redacted text.
    - **AC-15** — idempotency: `redact(redact(x)) == redact(x)`; an existing `[REDACTED:…]` placeholder is left unchanged even when an inner rule could match the placeholder text.
    - **AC-16** — `secret_redaction.py` performs no I/O / no network (e.g. assert the module imports no filesystem/socket calls in `redact`; the `allowlist` is injected, never loaded inside `redact`).
    - **AC-18** — fixtures and any asserted output contain only synthetic tokens and rule/class names; no raw secret value appears in the test file.
  - **Acceptance (RED):** Every AC above has **at least one assertion that fails** against the A.2 stub (not "every assertion fails"). Pair each negative/static guardrail with a positive control that fails on the stub: AC-7 — a real synthetic secret that IS flagged alongside each placeholder-shape no-hit; AC-9 — a flaggable token alongside each allowlisted no-hit; AC-15 — assert `redact(x) != x` (first-call) before `redact(redact(x)) == redact(x)`; AC-16 — pair the no-I/O check with a positive assertion that `redact` returns correctly redacted output (fails on the stub); AC-18 — synthetic-only fixtures whose positive controls drive the AC-5/AC-6 hits.
  - **Depends on:** A.2 (imports `redact`/`Hit`), A.1 (detect-secrets importable for the Tier-1 fixtures)
  - **Leverage:** `tests/conftest.py` (fixture set); `tests/test_issue_id_extraction.py` (pure-fn test shape); approach D-1 mapping, D-2/D-3/D-6; discovery External References (`detect-secrets` programmatic surface).
  - **Maps to:** AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-14, AC-15, AC-16, AC-18

- [x] **B.2** — Failing tests for the `add_turns` hook, config, report mode, and AC-17 scrub
  - **File(s):** `tests/test_secret_redaction.py`
  - **Acceptance:** At least one failing assertion per integration AC, exercising `rag_engine.add_turns` with `embed_texts` / Milvus client / FTS insert monkeypatched (no live Milvus/MLX). **Monkeypatch detail:** `milvus_client` is a `@contextmanager` (rag_engine.py:668 uses `yield`), so the mock must support `with milvus_client(...) as client:` — use `unittest.mock.patch` with a context-manager-aware mock or `contextlib.contextmanager` wrapper.
    - **AC-1** — in `enforce` mode each detected secret in a Turn is replaced with `[REDACTED:<RULE>]` before the text reaches the embed input, the Milvus `document` field, and the FTS `content` column.
    - **AC-2** — the embedding is computed over the **redacted** text: two Turns differing only in secret values feed identical strings to `embed_texts` (assert the captured embed input).
    - **AC-3** — a single hook covers both entry points: redaction is applied when ingesting via `add_turns` AND via `add_turns_async`, with no per-Provider change; redaction also runs **before** `_extract_issue_ids` (issue IDs survive).
    - **AC-10** — `report` mode stores Turn text UNREDACTED yet emits per-rule detection counts (rule names only, no values) — assert the histogram log line AND that the per-rule counters appear in `get_stats` output under the `redaction` key (the concrete storage resolved in C.2 — assert the surface, not an accidental in-memory dict); embed/`document`/`content` are unaltered.
    - **AC-11** — `SESSIONFLOW_REDACT` unset ⇒ enabled in `report` mode (the default).
    - **AC-12** — `SESSIONFLOW_REDACT` off ⇒ no detection, no mutation; Turn text stored exactly as before.
    - **AC-13** — mode is read from `SESSIONFLOW_REDACT_MODE` (`enforce`|`report`) and the allowlist path from the configured allowlist-path env var.
    - **AC-17** — when an FTS insert fails or an embedding error is raised, the logged/re-raised exception text is scrubbed through `redact` so no secret fragment echoes (assert a planted secret in the failing payload does not appear in the captured log / re-raised message).
  - **Acceptance (RED):** Every AC above has **at least one assertion that fails** against the no-hook state (not "every assertion fails"). Pair the static/off-state guardrails with positive controls that fail today: AC-12 ("off" ⇒ old behavior exactly) pairs with a positive control proving enforce/report DOES mutate; report-mode "stores unredacted text" pairs with the new `get_stats` counter-emission assertion (AC-10); cover default/report/enforce behavior plus counter emission so the gate cannot pass against the current no-hook code.
  - **Depends on:** B.1 (same file; sequential append), A.2
  - **Leverage:** `rag_engine.py:689` (`add_turns`), `:725`/`:742` (embed input/call), `:764` (Milvus `document`), `:791` (FTS `content`), `:809` (FTS error log), `:744–745` (embed error boundary), `:779`/`:804` (`_extract_issue_ids` ordering), `:595` (boolean-env idiom), `:2033–2051` (`add_turns_async` wrapper); `tests/conftest.py` fixtures; `monkeypatch` for `embed_texts`/FTS/Milvus.
  - **Maps to:** AC-1, AC-2, AC-3, AC-10, AC-11, AC-12, AC-13, AC-17

## Sprint Cohort C — Implementation · GREEN (sequential)

_TDD green phase. Minimum code to make Cohort B pass. Tasks depend on Cohort B (tests exist and are red)._

- [x] **C.1** — Implement the pure tiered detector + value-replacement masker in `secret_redaction.py`
  - **File(s):** `secret_redaction.py`
  - **Acceptance:** All B.1 tests pass (green). Implements: Tier-1 via `detect-secrets` built-in plugins through `transient_settings` + custom Tier-1 regexes for Anthropic / Google API key / HuggingFace (D-1 mapping); Tier-2 in-memory contextual `:`/`=` assignment scanner (D-1 — `scan_line()`'s adhoc-filename behavior cannot do this); Tier-3 entropy plugins gated advisory-only, auto-masked only on keyword-adjacency (same line ≤40 chars) or inside a ```` ``` ````-fenced block (D-3, AC-8); PEM multi-line pre-pass masking the whole block as `[REDACTED:PRIVATE_KEY]` (D-6); value-replacement masker (D-2 — `PotentialSecret.secret_value`, longest-value-first, skips any candidate whose span lies inside an existing `[REDACTED:…]` placeholder → idempotency, AC-15); allowlist filtering using `detect-secrets` structural filters (`is_potential_uuid`, hex-shape) plus the injected operator patterns (D-4, AC-9, AC-7). Stays pure / no I/O (AC-16). Google docstring on `redact` (ruff `D1`).
  - **Depends on:** B.1, A.1, A.2
  - **Leverage:** `detect-secrets 1.5.0` — `transient_settings`, `detect_secrets.core.scan.scan_line`, `from_plugin_classname`, `PotentialSecret.secret_value`, `detect_secrets.filters.heuristic.is_potential_uuid`; approach D-1/D-2/D-3/D-4/D-6 + the D-1 plugin mapping note; discovery Constraints (hashed-by-default → use `secret_value`; line-by-line → PEM pre-pass).
  - **Maps to:** AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-14, AC-15, AC-16, AC-18

- [x] **C.2** — Wire the redaction hook, inline config, allowlist loader, and AC-10 reporting into `add_turns`
  - **File(s):** `rag_engine.py`
  - **Acceptance:** B.2 hook/config/report tests (AC-1, AC-2, AC-3, AC-10, AC-11, AC-12, AC-13) pass (green). Adds: the redaction hook in `add_turns` between the dedup early-return (`:717–718`) and embed assembly (`:720`/`:725`), rewriting each surviving `turn["text"]` in place so all three durable sinks and `add_turns_async` are covered with no per-Provider change, and running before `_extract_issue_ids` (ordering invariant); inline `os.getenv` reads for `SESSIONFLOW_REDACT` / `SESSIONFLOW_REDACT_MODE` / the allowlist-path var using the boolean idiom at `:595` (default on + `report` when `SESSIONFLOW_REDACT` unset; off ⇒ skip entirely); the impure `load_allowlist(path) -> list[re.Pattern]` helper in `rag_engine.py` (NOT in `secret_redaction.py` — keeps the module pure, D-4/AC-16), loaded once and passed into `redact`; AC-10 reporting — aggregate the returned `hits` by `rule_name`, log a single INFO per-rule histogram line via the existing logger, and update durable per-rule counters on the status surface (rule names only, no values). **Counter storage (AC-10):** the approach says "durable per-rule counters on the status surface" but does not specify the storage location or the surface. **Resolve before implementation:** add a module-level `_redaction_counters: dict[str, int] = {}` in `rag_engine.py`, increment it in the hook, and expose it via `get_stats` under a `redaction` key (e.g., `stats["redaction"] = {"enabled": True, "mode": "report", "counts": _redaction_counters}`). B.2 tests assert the counter appears in `get_stats` output; `/health` does not need it (operator-facing stats are sufficient).
  - **Depends on:** B.2, C.1
  - **Leverage:** `rag_engine.py:716–718` (dedup filter + early-return — hook lands just after), `:595` (boolean-env idiom), `:205` (module-level env-read precedent), `:725`/`:742`/`:764`/`:791` (the three sinks), `:779`/`:804` (`_extract_issue_ids`), `:2033–2051` (`add_turns_async`); approach §High-Level Architecture + D-4; the existing status/counter surface for AC-10.
  - **Maps to:** AC-1, AC-2, AC-3, AC-10, AC-11, AC-12, AC-13

- [x] **C.3** — Harden the two AC-17 error boundaries with `redact`-scrubbed exception text
  - **File(s):** `rag_engine.py`
  - **Acceptance:** B.2 AC-17 tests pass (green). FTS-insert failure path (`:809`) runs the logged exception text through `redact` before `logger.warning`; the embedding-error path sanitizes the exception message through `redact` **before the bare re-raise at `:745`** (verified: `:744`'s `budget.after_batch(..., error=e)` only updates counters and does not log — `embedding_control.py:180–188` — so scrubbing before the re-raise means every upstream site that later stringifies the exception gets an already-clean message). Scope is exactly these two boundaries (D-7); the broader infra-error surface is an explicit follow-up.
  - **Depends on:** B.2, C.2 (same file — sequential after the hook lands)
  - **Leverage:** `rag_engine.py:809` (FTS warning), `:744–745` (embed error + bare re-raise), `embedding_control.py:180–188` (`after_batch` is counters-only); approach D-7; discovery AC-17 leak-surface table (the deferred upstream sites — do NOT touch them here).
  - **Maps to:** AC-17

---

## Standard Closing Tasks

> **Numbering:** Continues after the last sprint task (C.3).

- [x] **CLOSE-1. Run full test suite** — zero regressions
  - **File:** _no file changes — verification only_
  - Run `./venv/bin/python -m pytest` (the project's complete test command). All existing tests pass; all new Cohort B tests pass (green after Cohort C). Hard 10-minute suite timeout (CLAUDE.md) — skip pre-existing flakes; do not fix unrelated failures.
  - If anything fails: fix the implementation, not the test. Tests are the spec.

- [x] **CLOSE-2. Codacy CLI scan + ruff gate** — security + quality
  - **File:** _no file changes — scan only_
  - Invoke the `codacy-analysis-cli` skill against changed files (`codacy-analysis analyze --diff`). Triage: Critical/High must fix, Medium fix-or-document, Low/Info advisory. Compare any file-level finding on `rag_engine.py` against an `origin/main` baseline scan before treating it as introduced (`--diff` reports pre-existing issues in touched files).
  - **Plus the project ruff gate (CLAUDE.md Code Style):** run `./venv/bin/ruff check secret_redaction.py rag_engine.py` on the changed `.py` files. New `secret_redaction.py` must carry a module docstring + Google docstring on `redact` (`D1`); `tests/**` are `D`-exempt. (A bare `ruff check .` still surfaces the SESF-31 backlog — scope to changed files.)

- [x] **CLOSE-3. Generate verification stamp**
  - **File:** Append to the bottom of this `tasks.md` under heading `## Verification Stamp`.
  - For EACH acceptance criterion in `requirements.md` (AC-1 … AC-18), write exactly one line:
    - `- [x] AC-N — verified at <relative/file/path>:<line>`, OR
    - `- [x] AC-N — verified by <test name>`, OR
    - `- [ ] AC-N — gap: <one-line reason>`.
  - The stamp block MUST list every AC. Status-only notes are not equivalent — per-AC traceability is the gate.
  - **UI verification:** `approach.md`'s `## UI Contract` is **N/A — no UI surface** (backend ingestion-guard + config only; no mockup/playground/screenshot cited). The UI-visual-verification requirement therefore does not apply — test/implementation citations are sufficient for every AC.
  - Refuse to proceed to CLOSE-4 if any `[ ]` remains in the stamp block.

- [x] **CLOSE-4. Version bump**
  - **File:** _no file changes_
  - **N/A — project opts out of version management.** Verified 2026-06-08: SessionFlow has no `devops/version.lock`, no `VERSION` file, and no version field in `pyproject.toml`/`setup.py`/`setup.cfg`; it ships via PR-to-`main` with no semver tags. `/release patch` does not apply. If a version constant is later introduced, restore this task and invoke `/release patch` as a skill.

- [x] **CLOSE-5. Vault update**
  - **MUST invoke `/vault-update`** as a skill — even if you believe no foundation docs are affected, the skill performs the audit. Candidate doc updates this sketch introduces: a new Operational Gotcha for the `SESSIONFLOW_REDACT` / `SESSIONFLOW_REDACT_MODE` / allowlist-path env vars and the report-vs-enforce default (AGENTS.md + CLAUDE.md operational gotchas). Let the skill confirm scope.
  - _Plane Done transition moved to CLOSE-8 (post-merge) — see Review Archive. SessionFlow requires signed commits + PR review before merge; marking Done here would falsely close the issue if CI/review fails._

- [x] **CLOSE-6. Open PR**
  - Use the `sketch/SESF-41-secret-redaction-guard` worktree branch. Title: `feat(SESF-41): ingestion-time secret redaction guard` (operator-facing — new env vars + redaction behavior; use `chore(SESF-41):` only if scoped down).
  - Body must include: link to [SESF-41](https://plane.lbruton.cc/lbruton/browse/SESF-41/); link to the sketch folder (`DocVault/Projects/SessionFlow/sketches/SESF-41-secret-redaction-guard/`); a test-plan checklist. Signed commits + PR required (AGENTS.md git rules).

- [x] **CLOSE-7. Resolve PR review threads**
  - **File:** _GitHub PR threads only — code fixes land via the worktree as needed_
  - **MUST invoke `/pr-resolve`** as a skill. Scan **both** inline diff threads AND review-body findings (Codacy/Copilot often post critical findings as summary-style prose outside the diff). All Critical/High: fix or mark false-positive with explicit reasoning; Medium: fix or document waiver; Low/Info: advisory.
  - Scanners re-post on each new commit — after running `/pr-resolve`, check for newly-appeared auto-scanner threads and address them before merge.

- [x] **CLOSE-8. Archive sketch + close issue** (after PR merges)
  - Mark the source issue Done in Plane: `mcp__plane__update_issue` on SESF-41 → state "Done". (Moved here from CLOSE-5 — the transition runs only after the PR has merged, so the issue state reflects actual delivery, not code-complete.)
  - **MUST invoke `/sketch archive SESF-41`** as a skill — moves the folder to `archive/YYYY-MM-DD-SESF-41-secret-redaction-guard/` and saves a mem0 summary.

---

> **Follow-up issues to file** (from approach.md "Out of Scope" — file via `/issue` after merge, do not fold into this sketch): (1) broader error-surface hardening beyond the two AC-17 boundaries (`http_server.py`, `file_watcher.py:411`, `tools.py:563`, `rag_engine.py:283`, provider parse paths); (2) redact-or-drop the `content` mirror key on turn dicts if/when a log/report consumer begins reading `turn["content"]`. SESF-42 (retroactive sanitizer) already exists and depends on the `secret_redaction.py` this sketch ships.

> **Multi-model dispatch hint:** Cohort A `[P]` tasks split cleanly — A.1 (`requirements.txt`) and A.2 (`secret_redaction.py`) touch disjoint files with no shared symbols. Reconverge before Cohort B. The B (RED) → C (GREEN) boundary is the natural model-routing seam; C.1 (pure module) and C.2/C.3 (`rag_engine.py` integration) are sequential within C because C.2/C.3 share `rag_engine.py` and depend on C.1's `redact`.

## Review Archive — tasks (2026-06-08)

### Resolution Summary

- **Accepted: 3** — (1) B.1/B.2 RED-gate reformulation to "every AC has ≥1 assertion that fails" + positive controls (CODEX+QWEN consensus); (2) AC-10 counter assertion targeted at `get_stats` `redaction` key in B.2 (CODEX+QWEN consensus; C.2 already carried the storage resolution); (3) CLOSE-5 → CLOSE-8 Plane Done transition move (CODEX+QWEN consensus).
- **Rejected: 0.**
- **Resolved with your input: 0** — all findings were consensus with a single specified fix; no judgment calls.
- _Note: QWEN's `milvus_client` `@contextmanager` monkeypatch detail (concern #4) was already folded into the B.2 body prior to this reconcile — no further action._

The original reviewer sections are preserved verbatim below (heading levels demoted to nest under this archive).

### CODEX Review (2026-06-09)

#### Verified

- Read `DocVault/INDEX.md`, `DocVault/sketch/conventions.md`, `DocVault/sketch/templates/tasks-template.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, and all four SESF-41 phase files; prior discovery/approach comments are under `## Review Archive`, and `tasks.md` had no prior reviewer block.
- Verified live repo state at `5e150f1`: `rag_engine.py:689-811`, `rag_engine.py:1541-1583`, `rag_engine.py:2033-2051`, `embedding_control.py:180-188`, provider `content` mirror sites, `/health` status payload in `http_server.py:599-611`, CLI/MCP stats formatting, `requirements.txt`, `requirements-dev.txt`, `ruff.toml`, and the existing ingestion test helpers.
- Confirmed SessionFlow has no `.context/sketch-conventions.md`; generic branch/worktree convention applies. Also confirmed no `devops/version.lock`, `VERSION`, `pyproject.toml`, `setup.py`, or `setup.cfg` version surface exists, so CLOSE-4's version N/A is grounded.
- Queried SessionFlow and mem0 for SESF-41 and sketch-review workflow context; SessionFlow found the SESF-41 scaffold/discovery trail and mem0 mainly returned cross-harness/closing-task workflow reminders.

#### Top concerns

- B.1 and B.2 require all RED tests to fail, but several negative/default/static guardrails can pass against the current stub/no-hook state; tasks need positive controls or a narrower "at least one failing assertion per AC" rule.
- AC-10's durable counter requirement is underspecified: the live status surfaces do not currently expose redaction counters, and the File Map does not name a persistent state file or exact operator-visible endpoint.
- CLOSE-5 moves SESF-41 to Done before PR creation, review resolution, and merge, which is premature for SessionFlow's PR-required workflow.

#### Unverified assumptions

- Did not run the test suite or install `detect-secrets`; this was a review-only DocVault edit.
- Did not inspect the live Plane state for SESF-41; relied on the phase docs and prior SessionFlow recall for issue context.

### QWEN Review (2026-06-09)

#### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, all four SESF-41 phase files (requirements, discovery, approach, tasks); prior discovery/approach comments are under `## Review Archive`, and CODEX's tasks review is present.
- Verified live repo state at `5e150f1` (matches the cited commit):
  - `rag_engine.py:689-811` — `add_turns` signature, dedup loop `:699-714`, filter comprehension `:716`, early-return `:717-718`, embed assembly `:725`, embed call `:742`, error boundary `:744-745`, Milvus `document` sink `:764`, `_extract_issue_ids` `:779`/`:804`, FTS `content` sink `:791`, FTS error log `:809`. All match the tasks' claims.
  - `rag_engine.py:668-684` — `milvus_client` is a `@contextmanager` (uses `yield`), so B.2's monkeypatch must support the `with` protocol.
  - `rag_engine.py:1541-1583` — `get_stats` returns `total_turns`, `sessions`, `branches`, `by_type`, `providers`, `fts_*` fields; no redaction counters currently exposed.
  - `rag_engine.py:2033-2051` — `add_turns_async` wraps `add_turns` via `run_in_executor`, confirming a single hook covers both entry points.
  - `embedding_control.py:180-188` — `after_batch` increments `self.errors` counter but does not log or stringify the exception text, so C.3's "sanitize before re-raise at `:745`" is correct (the leak is the re-raised exception that upstream sites stringify).
  - `http_server.py:599-611` — `/health` returns `status`, `server`, `port`, `model_name`, `model_loaded`, `milvus`, `milvus_backend`, `watchers`, `providers`, `backfill`, `embedding`, `fts`; no redaction surface.
  - `rag_engine.py:1142` — DB-row reconstruction path reads `row.get("content")`, a DB-column fallback, not the adapter-emitted `turn["content"]` (D-5 is correct).
  - `requirements.txt` — confirmed `detect-secrets` is absent (A.1 must add it).
- Confirmed SessionFlow has no `.context/sketch-conventions.md`; generic branch/worktree convention applies. No `devops/version.lock`, `VERSION`, or version-bearing config files exist, so CLOSE-4's version N/A is grounded.
- Confirmed no unreconciled review marks in `requirements.md`, `discovery.md`, or `approach.md` (all have `## Review Archive` blocks — already reconciled).

#### Top concerns

1. **B.1 and B.2 RED gate satisfiability (echoing CODEX).** Independently confirmed: AC-7, AC-9, AC-15, and parts of AC-16/AC-18 can pass against the A.2 stub or the no-hook state. The "all tests must fail" requirement is too strict. **Fix:** reformulate the gate as "every AC has at least one assertion that fails against the stub/no-hook state" and pair each negative/static guardrail with a positive-control assertion (e.g., AC-7 placeholder-shape no-hit paired with a real-secret hit; AC-15 idempotency paired with a first-call redaction assertion). Added inline guidance for B.1 and B.2.

2. **AC-10 counter storage is underspecified.** The approach says "durable per-rule counters on the status surface" but the File Map does not name a persistent state file or the exact operator surface. Live `get_stats` (`rag_engine.py:1541-1583`) and `/health` (`http_server.py:599-611`) expose no redaction counters. **Resolve before implementation:** add a module-level `_redaction_counters: dict[str, int] = {}` in `rag_engine.py`, increment it in the hook, and expose it via `get_stats` under a `redaction` key (e.g., `stats["redaction"] = {"enabled": True, "mode": "report", "counts": _redaction_counters}`). B.2 tests assert the counter appears in `get_stats` output. Added inline to C.2.

3. **CLOSE-5 marks SESF-41 Done before PR creation, review, and merge.** Independently confirmed: SessionFlow's `AGENTS.md` requires signed commits + PR review before merge. Marking Done before the PR exists creates a window where CI could fail, review could request changes, or the branch could be abandoned, leaving the issue falsely Done. **Fix:** move the `mcp__plane__update_issue` Done transition to CLOSE-8 (post-merge archive) so the issue state reflects actual delivery. Keep `/vault-update` in CLOSE-5. Added inline to CLOSE-5.

4. **B.2 monkeypatch detail for `milvus_client`.** `milvus_client` is a `@contextmanager` (`rag_engine.py:668` uses `yield`), so the mock must support `with milvus_client(...) as client:`. The task says "monkeypatched" but doesn't specify this. Added inline to B.2.

#### Unverified assumptions

- Did not run the test suite or install `detect-secrets`; this was a review-only DocVault edit.
- Did not independently verify `detect-secrets` internal API (`transient_settings`, `PotentialSecret.secret_value`, plugin class names) — relied on discovery's Context7 research and CODEX's verification (both flagged unverified there).
- Did not check whether `transient_settings` from `detect-secrets` is thread-safe in server mode (it modifies a global `Settings` singleton); `_write_lock` on `add_turns` protects it for now, but if search or other paths ever use `detect-secrets`, this could become a concern. Not a blocker for this sketch; noted as a follow-up constraint.

---

## Verification Stamp

_Generated at CLOSE-3 (2026-06-08). Every acceptance criterion in `requirements.md` (AC-1 … AC-18) is traced to a passing test in `tests/test_secret_redaction.py`. Full suite: 287 passed, 0 failures._

- [x] AC-1 — verified by `test_ac1_enforce_redacts_all_three_sinks`
- [x] AC-2 — verified by `test_ac2_embedding_over_redacted_text_identical`
- [x] AC-3 — verified by `test_ac3_async_entry_point_also_redacted` + `test_ac3_redaction_runs_before_issue_extraction`
- [x] AC-4 — verified by `test_ac4_placeholder_format_and_fallback`
- [x] AC-5 — verified by `test_ac5_tier1_provider_token_masked_with_rule_name` (9 classes) + `test_ac5_npm_token_in_npmrc_context` + `test_ac5_jwt_masked` + `test_ac5_pem_private_key_block_masked_whole`
- [x] AC-6 — verified by `test_ac6_tier2_assignment_masked` + `test_ac6_mcp_config_dump_leak_class`
- [x] AC-7 — verified by `test_ac7_placeholder_shape_or_short_value_not_flagged` (5 cases)
- [x] AC-8 — verified by `test_ac8_isolated_entropy_reported_not_masked` + `test_ac8_entropy_in_fenced_block_masked`
- [x] AC-9 — verified by `test_ac9_structural_allowlist_not_flagged` (5 shapes) + `test_ac9_operator_supplied_allowlist_pattern`
- [x] AC-10 — verified by `test_ac10_report_stores_raw_but_emits_counts`
- [x] AC-11 — verified by `test_ac11_unset_defaults_to_report_enabled`
- [x] AC-12 — verified by `test_ac12_off_no_detection_no_mutation`
- [x] AC-13 — verified by `test_ac13_mode_and_allowlist_path_read_from_env`
- [x] AC-14 — verified by `test_ac14_determinism_identical_bytes` + `test_ac14_determinism_differ_only_in_secret`
- [x] AC-15 — verified by `test_ac15_idempotent_redact_of_redacted_text` + `test_ac15_existing_placeholder_not_rematched`
- [x] AC-16 — verified by `test_ac16_no_io_during_redact`
- [x] AC-17 — verified by `test_ac17_fts_insert_error_is_scrubbed` + `test_ac17_embedding_error_is_scrubbed`
- [x] AC-18 — verified by `test_ac18_hits_carry_no_raw_value`

**UI verification:** N/A — `approach.md` `## UI Contract` is "no UI surface" (backend ingestion-guard + config only). Test/implementation citations are sufficient for every AC.

**Implementation note (engine deviation):** detect-secrets' Base64/Hex entropy plugins are not used — the low-level `core.scan.scan_line()` does not honour their entropy `limit` and flags every word as high-entropy. Tier-3 entropy is instead a length-gated Shannon scanner in `secret_redaction.py`, which matches D-3's advisory-only, adjacency-gated intent. GitHub/GitLab also moved to custom Tier-1 regexes because detect-secrets reports a truncated prefix as `secret_value`. Both deviations preserve the AC behaviour; no test was weakened.
