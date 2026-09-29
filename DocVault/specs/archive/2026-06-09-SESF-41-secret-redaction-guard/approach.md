---
sketch: "SESF-41-secret-redaction-guard"
phase: approach
created: 2026-06-08
---

# SESF-41 — Approach

_How we'll build it. **Don't write code or tests** — the tasks phase produces the work plan._

## High-Level Architecture

A new pure module `secret_redaction.py` exposes `redact(text, *, mode, allowlist=None) -> (redacted_text, hits)` and is invoked from **one hook** inside `add_turns` (`rag_engine.py`). The hook fires after the dedup-by-`doc_id` filter comprehension (`:716`) and early-return (`:717–718`) — landing between `:718` and the embed budget setup (`:720`), before embed-input assembly — rewriting each surviving Turn's `turn["text"]` in place. Because all three durable sinks read from `turn["text"]` (the embedding vector, the Milvus `document` field `:764`, and the FTS `content` column `:791`) and `add_turns_async` simply wraps `add_turns` (`:2033–2051`), this single point covers every Provider and both entry points with no per-adapter changes (AC-3). It also sits **before** `_extract_issue_ids` (`:779`/`:804`), so issue-ID extraction operates on redacted text per the discovery ordering invariant.

Detection is a **hybrid tiered engine**, not pure `detect-secrets`. `detect-secrets` (adopted per `requirements.md`) drives Tier-1 structured-provider detection through a curated plugin set via `transient_settings`, and supplies the opt-in entropy plugins for Tier-3. But discovery established that `detect_secrets.core.scan.scan_line()` hardcodes an adhoc filename, so its `KeywordDetector` sees `FileType.OTHER` and applies quote-required rules — it cannot reliably catch the unquoted `api_key=value` / `api_key: value` assignments AC-6 requires, nor the MCP-config-dump leak class. So a thin SessionFlow layer supplies Tier-2 contextual-assignment scanning, Tier-3 entropy-adjacency gating, and PEM multi-line block expansion. Both engines emit `(raw_value, rule_name)` pairs into one deterministic value-replacement masker.

Config is decentralized (no `config.py`): the hook reads `os.getenv` flags inline using the canonical boolean idiom (`rag_engine.py:595`), loads the operator allowlist outside the pure `redact` function (AC-16, via `load_allowlist` in `rag_engine.py` — see File Map), and passes it in. **Reporting (AC-10):** after each `add_turns` call the hook aggregates the returned `hits` by `rule_name`, logs a single INFO histogram line via the existing logger, **and** updates durable per-rule counters on the status surface — so report mode (the default, AC-11) emits operator-facing per-rule detection counts even while storing raw text. The MVP also hardens the two AC-17-named error boundaries: it scrubs the FTS-insert path (`:809`), and for the embedding-error path it sanitizes the exception **before** the bare re-raise at `:745` (verified: `:744`'s `budget.after_batch(..., error=e)` only updates counters and does not log the error — `embedding_control.py:180–188` — so the actual leak is the re-raised exception text that upstream catch sites stringify; running it through `redact` at the raise site means every upstream copy is already clean) so a failure can't echo a Turn-text fragment.

## Key Decisions

| # | Decision | Rationale | Tradeoff |
|---|----------|-----------|----------|
| D-1 | **Hybrid detection: `detect-secrets` for Tier-1 + entropy plugins; a thin custom layer for Tier-2 assignments, Tier-3 adjacency, and PEM blocks.** | Honors the requirements' `detect-secrets` decision and reuses its maintained provider-token regexes where they're strongest; supplies the in-memory assignment scanning `scan_line()`'s adhoc-filename behavior can't do (AC-6). See the **D-1 plugin mapping** note below the table for the AC-5 plugin/custom-regex split. | Two detection paths to keep coherent, plus a new runtime dependency that earns its keep mainly for Tier-1 + entropy. |
| D-2 | **Mask by value-replacement, not character offset.** Each path yields `(raw_value, rule_name)`; the masker does `str.replace(raw_value, "[REDACTED:{rule_name}]")`, longest-value-first. Raw values come from `PotentialSecret.secret_value` (storage enabled). | `detect-secrets` provides no char offsets; value-replace is the only mechanism uniform across both engines, and is naturally idempotent — a `[REDACTED:…]` placeholder contains no secret value left to re-match; the masker **additionally skips any candidate match whose span falls inside an existing `[REDACTED:…]` placeholder**, so a re-run (`redact(redact(x))`) is a no-op even when an inner rule could match the placeholder text (AC-15; Cohort B asserts this). | A secret value that also appears as a benign substring elsewhere in the same Turn gets masked too (over-redaction). Rare; accepted. |
| D-3 | **Tier-3 entropy is advisory-only; auto-redacted only on adjacency.** Entropy hits always go to `hits`; they are masked in `enforce` mode only when a Tier-2 keyword is **on the same line, within 40 characters, or the hit sits inside a ` ``` `-fenced block** (AC-8). "Fenced block" = any region delimited by a pair of triple-backtick fences. | Entropy alone is the false-positive engine; keyword/fence adjacency is the actual signal. | A high-entropy secret with no nearby keyword and outside a fence is reported but not masked — a deliberate precision-over-recall choice. |
| D-4 | **Two-part allowlist: `detect-secrets` structural filters + an operator regex file.** `is_potential_uuid` and hex-shape checks cover UUID / 40-hex SHA / 64-hex SHA-256 / `doc_id` / issue IDs (AC-9); operator patterns load from the configured allowlist-path env var, passed into `redact` as an argument. | Structural exemptions are universal and belong in code; operator patterns are deployment-specific and must stay out of the pure function (AC-16). | The caller (hook) must load and pass the allowlist — a small wiring step to preserve `redact` purity. |
| D-5 | **Constrain the guard to the durable `text` sink; do NOT redact the `content` mirror key.** The hook rewrites `turn["text"]` only. | `add_turns` never reads `turn["content"]` for any durable sink (verified); redacting it would touch turn-dict assumptions across **four adapters (five content-setting sites)** — `provider_claude.py:83`, `provider_codex.py:187`, `provider_antigravity.py:346`, `provider_opencode.py:295` + legacy `:370` — for zero current benefit. (`rag_engine.py:1142` reads `row.get("content")`, a DB-column fallback in a row-reconstruction path, **not** the adapter-emitted `turn["content"]` — so it isn't a counterexample.) | A *future* log/report path that reads raw `turn["content"]` would see cleartext. Mitigated by the AC-17 scrub layer and a noted follow-up (see Out of Scope). |
| D-6 | **PEM / multi-line blocks handled by a pre-pass above the line scanner.** A regex matches `-----BEGIN … PRIVATE KEY-----` … `-----END … -----` and masks the whole block as `[REDACTED:PRIVATE_KEY]`. | `detect-secrets`' `PrivateKeyDetector` flags only the marker line; the key body would otherwise survive into all three sinks. | A custom block regex to maintain — acceptable, PEM framing is stable. |
| D-7 | **AC-17 hardening at the two named boundaries** (FTS insert `:809`; embedding error sanitized **before the bare re-raise** at `:745`), reusing `redact` on the exception text. | FTS-insert holds Turn `content` directly. For the embedding path, `:744`'s `after_batch(..., error=e)` only updates counters (verified `embedding_control.py:180–188`) and `:745` bare-`raise`s — so the leak is the re-raised exception text that upstream sites stringify (`http_server.py:301/303`, `file_watcher.py:322/411`, `provider_ingestion.py:92–97`, etc.). Running the exception message through `redact` before the re-raise means every upstream copy is already clean, satisfying AC-17 at a single in-`add_turns` touch point. AC-17 names exactly these two events. | The broader infra-error surface discovery enumerated (those upstream catch sites for *infra* exceptions, not Turn text) is left for a follow-up. |

**D-1 plugin mapping (AC-5).** Verified against `detect-secrets 1.5.0`. Tier-1 **built-in** plugins (loaded via `transient_settings`): `AWSKeyDetector` (AWS), `GitHubTokenDetector` (GitHub), `GitLabTokenDetector` (GitLab), `SlackDetector` (Slack), `StripeDetector` (Stripe), `OpenAIDetector` (OpenAI `sk-…`), `NpmDetector` (npm token), `PrivateKeyDetector` (PEM marker line — extended to the full block by the D-6 pre-pass), and `JwtTokenDetector` (JWT — a **built-in**, not custom). Tier-3 entropy: `Base64HighEntropyString`, `HexHighEntropyString`. **No built-in covers Anthropic (`sk-ant-…`), Google API keys (`AIza…`), or HuggingFace (`hf_…` / `api_…`)** — these route to **custom Tier-1 regexes** in `secret_redaction.py`. Cohort B maps each AC-5 class to its owning detector.

## File Map

_Every file this sketch will create, modify, or delete. Tasks.md will reference these paths._

### New
- `secret_redaction.py` — **pure** detector + masker, no I/O (AC-16). Public `redact(text, *, mode, allowlist=None) -> tuple[str, list[Hit]]`; tiered engine (D-1, D-3, D-6), value-replacement masker (D-2), idempotency guard (AC-15). Module docstring + Google docstring on `redact` (ruff D1). The impure allowlist loader (`load_allowlist(path) -> list[re.Pattern]`) lives in `rag_engine.py`, **not** here, so this module stays I/O-free (D-4, AC-16).
- `tests/test_secret_redaction.py` — Cohort-B AC assertions; synthetic, non-functional tokens only (AC-18). Slots into the `tests/conftest.py` fixture set; `tests/**` are ruff-`D`-exempt.

### Modified
- `rag_engine.py` — (a) the redaction hook in `add_turns` between the dedup early-return (`:718`) and embed assembly (AC-1/2/3, ordering invariant); (b) inline `os.getenv` config reads for `SESSIONFLOW_REDACT` / `SESSIONFLOW_REDACT_MODE` / the allowlist-path var (AC-11/12/13), boolean idiom per `:595`; (c) the `load_allowlist(path)` helper + load-and-pass-through at the hook (D-4, keeping `secret_redaction.py` pure); (d) AC-10 reporting — per-rule histogram log + durable counters built from the hook's returned `hits`; (e) FTS-error scrub (`:809`) and embedding-error sanitize-before-re-raise (`:745`) per AC-17 (D-7).
- `requirements.txt` — add `detect-secrets` as a **runtime** dependency (not `requirements-dev.txt`).

### Deleted
- None.

## Data / Schema Changes

None — no Milvus schema field is added (explicit non-goal). `doc_id` is computed upstream from original bytes, so redacting inside `add_turns` leaves dedup semantics unchanged (AC-14) — no migration, no backfill.

## Tradeoffs Surfaced for Review

- **Hybrid engine vs. either pure path (D-1).** More moving parts than pure-custom or pure-`detect-secrets`. Chosen to satisfy the requirements' `detect-secrets` decision while covering the AC-6/AC-8 cases `detect-secrets` can't do in-memory. If the custom Tier-2/Tier-3 layer ends up carrying most of the load, revisit whether `detect-secrets` still earns the dependency.
- **D-5 leaves the `content` mirror raw.** The conservative-scope choice. If a reviewer wants belt-and-suspenders, the alternative is redacting both `text` and `content` keys at the hook — a small code cost but it touches turn-dict shape assumptions across four adapters (five content-setting sites, incl. the opencode legacy parse path). Flagging explicitly before tasks commits to text-only.
- **AC-17 scoped to two boundaries (D-7).** The broader error-surface hardening discovery verified is deferred. Confirm that matches the intended MVP scope rather than shipping the full scrub now.

## UI Contract

N/A — no UI surface. This is a backend ingestion-guard + config change; `requirements.md` and `discovery.md` reference no mockup, playground, screenshot, or prototype.

## Out of Scope (follow-up issues)

- **Broader error-surface hardening beyond the two AC-17 boundaries** — `http_server.py:220,301,303,522,545,582`, `file_watcher.py:411`, `tools.py:563`, `rag_engine.py:283`, `provider_claude.py:75`, `provider_antigravity.py:299`. Defense-in-depth; these stringify infra exceptions, not Turn text. File as a follow-up under SessionFlow.
- **Redact-or-drop the `content` mirror key** on turn dicts (D-5) — only if/when a log or report consumer begins reading `turn["content"]`. File as a follow-up if such a consumer is introduced.
- **SESF-42 retroactive sanitizer** — cleaning already-indexed Turns. Already a separate `/spec`-sized issue; this sketch builds the shared `secret_redaction.py` it depends on.

## Risk Notes

- Risk: relying on `detect_secrets.scan_line()` for assignments would silently miss AC-6 → mitigation: the custom Tier-2 assignment scanner (D-1) owns contextual `:`/`=` detection; `scan_line` is used only for Tier-1 + entropy.
- Risk: value-replacement over-redacts a benign substring (D-2) → mitigation: longest-value-first replacement to avoid partial-overlap corruption; rare benign masking accepted.
- Risk: `detect-secrets` is not installed in the venv and absent from `requirements.txt` → mitigation: add it as a runtime dep (file map) and `pip install -r requirements.txt` before tests; tasks Cohort 0 must verify the import.
- Risk: redaction adds per-Turn latency before the 200ms MLX embed throttle → low; regex/`detect-secrets` over ≤~8000-char Turns is negligible vs. embedding, and runs on already-deduped turns only.

---

> **Phase complete?** Architecture clear, decisions logged with rationale, file map complete. Then advance: `/sketch-tasks SESF-41`.

## Review Archive — approach (2026-06-08)

### Resolution Summary

Reconciled via `/sketch-reconcile SESF-41 approach` on 2026-06-08 — **5 accepted, 0 rejected, 3 resolved with operator input.** All reviewer code claims were verified against live `main` before applying.

**Accepted (5):**
1. Hook line-refs corrected — dedup filter comprehension `:716`, early-return `:717–718`, hook lands between `:718` and `:720` (was the imprecise `:699–714`).
2. D-1 plugin→AC-5 mapping note added — `detect-secrets 1.5.0` built-ins enumerated; **JWT confirmed a built-in** (`JwtTokenDetector`); Anthropic / Google API key / HuggingFace routed to custom Tier-1 regexes.
3. D-2 idempotency strengthened — masker skips any candidate whose span is inside an existing `[REDACTED:…]` placeholder; `redact(redact(x))` Cohort-B assertion noted.
4. D-5 factual fixes — "five adapters" → "four adapters (five content-setting sites)" (D-5 row + Tradeoffs); `:1142` citation corrected to a DB-row reconstruction fallback, not the adapter-emitted `turn["content"]`.
5. `load_allowlist` moved out of `secret_redaction.py` into `rag_engine.py` — keeps the module pure / I/O-free per AC-16 (File Map + D-4 + Modified bullet).

**Resolved with operator input (3):**
- **AC-17 embedding half** → sanitize the exception **before the bare re-raise at `:745`** (verified `:744`'s `after_batch` only updates counters, `embedding_control.py:180–188`), so upstream stringified copies are already clean — single in-`add_turns` touch point.
- **AC-10 report surface** → **both** an INFO per-rule histogram log **and** durable per-rule counters, built from the hook's returned `hits`.
- **AC-8 adjacency** → entropy auto-redacts when a Tier-2 keyword is **same-line, within 40 chars, or the hit is inside a ` ``` `-fenced block**.

**Rejected (0).**

The original reviewer sections are preserved verbatim below.

## QWEN Review (2026-06-08)

### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, `requirements.md`, `discovery.md` (including its Review Archive), and this `approach.md`; no `.context/sketch-conventions.md` exists for SessionFlow.
- Verified hook placement against live `main` (`rag_engine.py:689–811`): dedup loop `:699–714`, filter comprehension `:716`, early return `:717–718`, embed assembly `:725`, Milvus `document` sink `:764`, FTS `content` sink `:791` (from `t["text"]`), `_extract_issue_ids` at `:779`/`:804`, FTS error log `:809`, embed error boundary `:744–745`, `add_turns_async` wrapper `:2033–2051`. All match the approach's claims.
- Verified `:1142` is in a DB-row reconstruction function (reads `row` dict, not `turn` dict) — the `content` there is a DB column fallback, not the adapter-emitted `turn["content"]`.
- Verified provider adapter `"content"` sites: 4 adapters, 5 content-setting lines (`provider_claude.py:83`, `provider_codex.py:187`, `provider_antigravity.py:346`, `provider_opencode.py:295` + legacy `:370`).
- Confirmed `detect-secrets` still absent from `requirements.txt`.
- Confirmed no unreconciled review marks in `requirements.md` or `discovery.md` (discovery has a `## Review Archive` block — already reconciled).

### Top concerns

1. **D-7 / AC-17 embedding error boundary is underspecified.** `rag_engine.py:744` calls `budget.after_batch(..., error=e)` then `:745` does a bare `raise`. The approach says to "scrub at the embed-error boundary" but doesn't clarify what exactly leaks. If `budget.after_batch` doesn't log the error text internally, then `:744` isn't itself a leak surface — the leak would be wherever the re-raised exception is ultimately caught (likely `http_server.py:301`, already deferred). The tasks phase needs this resolved before writing scrub code. Recommend clarifying whether the scrub target is (a) `budget.after_batch` internal logging, (b) the upstream catch-and-log, or (c) wrapping the `raise` itself to sanitize the exception message.

2. **D-1 doesn't enumerate the Tier-1 plugin-to-AC-5 mapping.** AC-5 lists 10 structured provider classes plus PEM and JWT. The approach says "curated plugin set via `transient_settings`" but doesn't specify which detect-secrets plugins cover which classes, or which need custom Tier-1 regexes. JWT is not a standard detect-secrets plugin — confirm it lands in the custom layer. The tasks phase needs this mapping to write concrete Cohort-B test assertions for AC-5.

### Unverified assumptions

- Did not independently verify detect-secrets plugin names or `transient_settings` API — relied on discovery's Context7 research (flagged unverified there).
- Did not check `budget.after_batch` source to confirm whether it logs error text internally (the D-7 concern above).

## CODEX Review (2026-06-08)

### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, `requirements.md`, `discovery.md` including its Review Archive, and this `approach.md`; no project-local `.context/sketch-conventions.md` exists.
- Verified live ingestion flow in `rag_engine.py:689-811` and `rag_engine.py:2033-2051`: `turn["text"]` feeds embeddings, Milvus `document`, FTS Sidecar `content`, and `_extract_issue_ids`; `add_turns_async` wraps `add_turns`.
- Verified Provider Adapter `content` mirror sites: `provider_claude.py:83`, `provider_codex.py:187`, `provider_antigravity.py:346`, `provider_opencode.py:295` and `provider_opencode.py:370`.
- Verified exception stringification surfaces around `add_turns_async`: `http_server.py:301/303`, `file_watcher.py:322/411`, `provider_ingestion.py:92-97`, `verify_provider_ingestion.py:62-63`, and MCP `tools.py:562-563`; `embedding_control.py:180-188` does not stringify or log `error`.
- Checked `detect-secrets` via Context7 `/yelp/detect-secrets` and the current PyPI wheel (`detect-secrets 1.5.0`) for `transient_settings`, `scan_line`, `PotentialSecret.secret_value`, `KeywordDetector` filetype behavior, and built-in plugin class names.

### Top concerns

- Report mode lacks an output design for AC-10 per-rule detection counts, even though report mode is the default per AC-11.
- `secret_redaction.py` is called a pure no-I/O module, but the file map puts impure allowlist file loading in that module, conflicting with the source issue and AC-16.
- D-7 still does not satisfy the embedding-error half of AC-17 unless it sanitizes the re-raised exception before the upstream catch sites stringify it, or includes those catch sites in scope.
- D-1/D-3 leave test-defining detector details open: AC-5 needs a plugin/custom-regex mapping for Anthropic, Google API keys, and HuggingFace; AC-8 needs a concrete entropy adjacency window.

### Unverified assumptions

- Did not run SessionFlow tests or install `detect-secrets` into the project venv; this was a review-only document edit.
- Did not verify whether provider ingestion/backfill currently routes through MCP `tools.py` for the exact embedding-error path; `tools.py` remains a plausible generic MCP catch surface.
