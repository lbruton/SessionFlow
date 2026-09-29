---
sketch: "SESF-41-secret-redaction-guard"
phase: discovery
created: 2026-06-08
---

# SESF-41 — Discovery

_Research the existing system and prior art. **No solutions** — that's the approach phase._

## Existing Code

All line numbers verified against current `main` (`5e150f1`) — they match the SESF-35 research note exactly.

| Path | Role | Notes |
|------|------|-------|
| `rag_engine.py:689` | `add_turns(turns, db_path)` — the single ingestion chokepoint | Dedup-by-`doc_id` loop (`:699–714`) reads only `turn["doc_id"]`, **not** text — so redaction order vs. dedup is independent. |
| `rag_engine.py:725` | `texts = [t["text"] for t in batch]` — embed input assembled | Feeds `embed_texts` at `:742`. The vector is a function of `turn["text"]`. |
| `rag_engine.py:764` | Milvus `document` sink — `_truncate_utf8(turn["text"], 65535)` | Sink #2. Schema `document` VARCHAR(65535). |
| `rag_engine.py:791` | FTS Sidecar `content` sink — `"content": t["text"]` | Sink #3. The `content` sink is derived from `t["text"]`, never from `turn["content"]`, so rewriting `turn["text"]` alone covers all three **durable** sinks. Note: provider adapters DO emit a raw `content` mirror on the turn dict (`provider_antigravity.py:346`, `provider_codex.py:187`, `provider_claude.py:83`, `provider_opencode.py:295` + legacy path `:370`); `add_turns` never reads it for durable sinks, but a future log/report path could (see Open Questions). |
| `rag_engine.py:779`, `:804` | `_extract_issue_ids(turn.get("text", ""))` | Reads raw text but emits only `PREFIX-123` tokens — not a leak vector; in enforce mode it should run on the **redacted** text (issue IDs survive redaction). |

| `rag_engine.py:809` | `logger.warning("FTS insert failed (non-fatal): %s", e)` | Error-log leak surface (AC-17) — a SQLite error could echo a value fragment. |
| `rag_engine.py:744` | `budget.after_batch(..., error=e); raise` | Embedding error boundary — re-raises; scrub per AC-17. |
| `rag_engine.py:2033` → `:2051` | `add_turns_async` wraps `add_turns` via `run_in_executor(... lambda: add_turns(...))` | Confirms a single hook at the top of `add_turns` covers **both** entry points. |
| `rag_engine.py:205` | `_MODEL_NAME = os.getenv("SESSIONFLOW_MODEL", "embeddinggemma").lower()` | Precedent: module-level env read at import. |
| `rag_engine.py:595` | `os.getenv("SESSIONFLOW_AUTO_MIGRATE_SCHEMA", "").lower() in {"1","true","yes","on"}` | Canonical boolean-env idiom to reuse for `SESSIONFLOW_REDACT`. |
| `tests/test_issue_id_extraction.py` | Unit tests for a pure text→tokens function | Closest analog for testing pure `redact(text)`. |
| `tests/conftest.py` | pytest fixtures | New `tests/test_secret_redaction.py` slots in here. |
| `requirements.txt` | Runtime deps | `detect-secrets` added here (NOT requirements-dev.txt). `detect-secrets` is **not currently installed**. |
| `ruff.toml` | `select=["D1"]` docstring coverage, Google convention | New `secret_redaction.py` needs a module docstring + Google docstring on public `redact`. `tests/**` are `D`-exempt. |

### AC-17 leak surface — outward error-formatting paths (verified)

Beyond the two `add_turns`-local boundaries above, the same exception-stringification pattern recurs across the codebase. All verified against `main` @ `5e150f1`; all **low-risk** (they fire on infrastructure errors — Milvus/HTTP/parse — not on Turn text), but the AC-17 scrubber plan should enumerate them rather than discover them mid-implementation:

| Path | Surface |
|------|---------|
| `http_server.py:301` | `logger.error(... traceback.format_exc())` (ASGI handler) |
| `http_server.py:303` | `detail: str(exc)` in the 500 response body |
| `http_server.py:220` | `message = str(exc)` (HTTPException formatter) |
| `http_server.py:522, :545, :582` | `str(exc)` in health / identity / status payloads |
| `file_watcher.py:411` | `_log(f"Backfill error … : {e}")` |
| `tools.py:563` | `f"Error executing {name}: {str(e)}"` (MCP error response) |
| `rag_engine.py:283` | `"error": str(exc)` (embedding-identity error) |
| `provider_claude.py:75`, `provider_antigravity.py:299` | `errors=[str(exc)]` (parse-error paths) |

## Prior Decisions

- **2026-06-07 — SESF-35 research note** (`DocVault/Research/SessionFlow/Secret Redaction & Retroactive Sanitization (SESF-35).md`): the design backbone. Established the single-chokepoint hook (§7), typed-placeholder + re-embed masking (§5), the three-tier detector with entropy-advisory-only (§4), `detect-secrets`-over-trufflehog (no live-verification / no exfiltration, §4), deterministic masking for `doc_id` dedup (§7), and the prevention/cleanup split (§13). _mem0 query: `"SessionFlow secret redaction add_turns detect-secrets ingestion guard"`._
- **2026-06-07 — SESF-28 `/discover` memory** (mem0 id `f4abfd36…`): precedent that this project runs `/discover`→issue-split workflows; report-first rollout pattern. _mem0 query: `"SessionFlow SESF-41 secret redaction handoff next steps"`._
- **This session (2026-06-08)** — user decisions: adopt `detect-secrets`; `report` = detect+count only (store raw); `enforce` = redact+placeholder; default **on + report** when `SESSIONFLOW_REDACT` unset.

## External References

- [`detect-secrets` (Yelp)](https://github.com/yelp/detect-secrets) — context7 `/yelp/detect-secrets`. Programmatic surface:
  - `from detect_secrets import SecretsCollection` + `from detect_secrets.settings import transient_settings` / `default_settings` — the documented library entry point, but `SecretsCollection.scan_file(path)` reads **from disk**.
  - Lower-level in-memory scan: `detect_secrets.core.scan.scan_line(line)` and `detect_secrets.core.plugins.initialize.from_plugin_classname('AWSKeyDetector')` for per-line, per-plugin analysis.
    - **Caveat (unverified — `detect-secrets` not installed in venv):** upstream hardcodes `filename='adhoc-string-scan'` in `scan_line()`, so `KeywordDetector` sees `FileType.OTHER` and applies quote-required rules rather than the CONFIG/YAML/TOML rules that allow unquoted `api_key: value` / `api_key=value` patterns. Since AC-6 requires contextual `:`/`=` assignments, plain `scan_line()` is likely insufficient — a filename-aware/custom wrapper or a separate assignment layer is needed. Surfaced for the approach phase to resolve.
  - **Built-in filters that satisfy our allowlist/guard ACs:** `detect_secrets.filters.heuristic.is_potential_uuid` (AC-9 UUID), `is_likely_id_string`, `is_prefixed_with_dollar_sign`; plus `--word-list` / `exclude-secrets` regex for operator allowlist + placeholder guard (AC-7).
  - **Entropy is opt-in plugins** (`Base64HighEntropyString` limit 4.5, `HexHighEntropyString` limit 3.0) — selecting/deselecting them controls Tier-3 (AC-8 advisory-only).
  - **No network** by default; verification only runs under `--only-verified`/verify policies — keep off to satisfy AC-16/no-exfiltration.
- EARS / sketch templates — already in repo, no external ref needed.

## Constraints

- **detect-secrets reports type + (by default) a *hashed* secret + line number — not a character offset or the raw substring.** In-place masking (`[REDACTED:<TYPE>]`) needs the raw matched value; it is reachable as `PotentialSecret.secret_value` when scanning with storage enabled, then replaced by value in the line. Offset-precise masking is not provided out of the box.
- **detect-secrets scans line-by-line.** Multi-line secrets (PEM `-----BEGIN … PRIVATE KEY-----` blocks) are matched on the marker line by `PrivateKeyDetector`; whole-block masking needs handling above the line scanner.
- **Determinism (AC-14):** detect-secrets matching is deterministic; the placeholder must carry no random salt. `doc_id` is computed **upstream** in the parser/adapter from original bytes, so redacting inside `add_turns` leaves `doc_id` semantics unchanged — the simplest correct choice.
- **Redaction-order invariant:** the guard must rewrite `turn["text"]` **before** `_extract_issue_ids` runs at `:779`/`:804`, so issue-ID extraction operates on redacted text (issue IDs survive redaction). A hook at the top of `add_turns` (`:689`) satisfies this by construction. This is a load-bearing constraint of the current code, not just an observation — a future refactor that relocates either the hook or the issue-ID extraction must preserve the ordering.
- **Embedding Budget / MLX:** redaction runs before the existing 200ms-floor embed throttle; regex over ≤~8000-char Turns is negligible vs. embedding. No change to the budget path.
- **Config is decentralized** — no `config.py`; env vars are read inline (`os.getenv`) at point of use. New `SESSIONFLOW_REDACT*` vars follow that idiom.
- **Docstring gate (ruff D1):** module + public-`redact` Google docstrings required; private helpers exempt.
- **No raw values anywhere (AC-18):** fixtures, logs, and the report-mode counts must use synthetic tokens / rule names only.

## Open Questions

_None block approach — the integration surface is now known. The following are **approach-phase design choices**, not unknowns:_

- Masking mechanism: drive detect-secrets via `transient_settings` + `secret_value`, vs. a thin custom regex layer for offset-precise Tier-1/Tier-2 spans (research note leans detect-secrets; revisit if replace-by-value proves lossy).
- Tier-3 adjacency (AC-8): detect-secrets has no native "entropy near keyword" rule — implement as a thin wrapper, or disable entropy plugins for `enforce` and surface them only in `report`.
- Allowlist composition: reuse detect-secrets' `is_potential_uuid` / `--word-list` filters vs. a SessionFlow allowlist file at the configured path (likely both).
- Content-key handling: redact the raw `content` mirror that provider adapters emit (antigravity:346, codex:187, claude:83, opencode:295 + legacy :370) as well as `text`, or explicitly constrain the guard to the durable `text` sinks so a later log/report path cannot re-use stale raw `content`? `add_turns` ignores `content` today, but the field is present on the turn dict.

## Discovery Summary

The work lands at one hook at the top of `add_turns` (`rag_engine.py:689`) plus a new pure `secret_redaction.py`; rewriting `turn["text"]` there propagates to all three sinks (vector `:742`, `document` `:764`, FTS `content` `:791`) with no per-Provider changes and no schema change. The tricky part is the masking mechanism — `detect-secrets` is file/line-oriented and hashes by default, so the approach must extract raw matched values for in-place replacement and add a thin layer for entropy-adjacency and PEM multi-line blocks. Everything else (config idiom, test analog, dedup invariance, docstring gate) is already-paved.

---

> **Phase complete?** Existing code mapped. Prior decisions surfaced. Open questions resolved. Then advance: `/sketch-approach SESF-41`.

## Review Archive — discovery (2026-06-08)

### Resolution Summary

- **Accepted (4):**
  - A1 (consensus CODEX+QWEN) — corrected the `content`-mirror wording at the FTS sink row; adapters do emit a raw `content` mirror, but `add_turns` never reads it for durable sinks. Added a content-key handling Open Question.
  - A2 (consensus CODEX+QWEN) — added a verified AC-17 leak-surface table for outward error-formatting paths (`http_server.py`, `file_watcher.py`, `tools.py`, `rag_engine.py:283`, provider parse paths).
  - A3 (CODEX) — appended the `scan_line()` `FileType.OTHER` caveat to External References (flagged unverified — detect-secrets not installed).
  - A4 (QWEN) — added a redaction-order invariant to Constraints (guard rewrites `turn["text"]` before `_extract_issue_ids`).
- **Rejected (0):** none — all findings verified true against `main` @ `5e150f1`.
- **Resolved with your input (0):** none.

Original reviewer artifacts preserved verbatim below.

### CODEX Review (2026-06-08)

#### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, `requirements.md`, and this `discovery.md`; no project-local `.context/sketch-conventions.md` exists.
- Verified current repo state on `main` at `5e150f1`; checked `rag_engine.py:689-811`, `rag_engine.py:2033-2051`, Provider Adapter Turn construction, `requirements.txt`, `requirements-dev.txt`, `ruff.toml`, `tests/test_issue_id_extraction.py`, and `tests/conftest.py`.
- Checked `detect-secrets` via Context7 `/yelp/detect-secrets` plus upstream source for `scan_line`, `PotentialSecret.secret_value`, plugin initialization, filters, and `KeywordDetector` filetype behavior.
- Queried SessionFlow and mem0 for SESF-41/secret-redaction prior context; SessionFlow found the SESF-41 scaffold/discovery trail, while mem0 did not return the exact cited SESF-28 memory id.

#### Top concerns

- The `turn["content"]` mirror claim is false against live Provider Adapters. `add_turns` currently ignores that field for durable sinks, but approach should not inherit "field absent" as an assumption.
- Plain `detect_secrets.core.scan.scan_line(line)` is likely insufficient for AC-6 contextual assignments because it uses an adhoc filename and bypasses CONFIG/YAML/TOML unquoted-assignment handling.
- The embedding-error leak surface extends beyond `rag_engine.py`; HTTP, watcher, and MCP/tool error formatting paths stringify exceptions and need to be included in the AC-17 scrub plan.

#### Unverified assumptions

- Did not run `detect-secrets` inside the project venv because it is not installed; API findings are from Context7 and upstream source.
- Could not confirm mem0 id `f4abfd36...`; only the broader SessionFlow SESF-41 decision trail was verified.

### QWEN Review (2026-06-08)

#### Verified

- Read `DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, `requirements.md`, and this `discovery.md`; no `.context/sketch-conventions.md` exists for SessionFlow.
- Confirmed HEAD at `5e150f1` matches the cited commit.
- Verified every line number in the Existing Code table against live `main`:
  - `rag_engine.py:689` (`add_turns` signature), `:699-714` (dedup loop reads only `doc_id`), `:725` (embed input assembly), `:742-745` (embed call + error boundary), `:764` (Milvus `document` sink), `:779` / `:804` (`_extract_issue_ids`), `:791` (FTS `content` sink from `t["text"]`), `:809` (FTS error log), `:205` (`_MODEL_NAME` env precedent), `:595` (boolean env idiom), `:2033-2051` (`add_turns_async` wraps `add_turns`).
  - All four provider adapters emit `content` mirrors: `provider_antigravity.py:345-346`, `provider_codex.py:185-187`, `provider_claude.py:78-83`, `provider_opencode.py:293-295` (db path) **and** `provider_opencode.py:368-370` (legacy path — not cited in discovery or CODEX review).
  - Error-log leak surfaces beyond `rag_engine.py`: `http_server.py:301` (`traceback.format_exc()`), `http_server.py:303` (`str(exc)` in HTTP response), `http_server.py:220` (`str(exc)` in HTTPException), `http_server.py:522,545,582` (health/identity/status `str(exc)`), `file_watcher.py:411` (`{e}` in log), `tools.py:563` (`str(e)` in MCP error response), `rag_engine.py:283` (embedding identity `str(exc)`), `provider_claude.py:75` / `provider_antigravity.py:299` (parse error `str(exc)`).
- Confirmed `detect-secrets` is not installed in the project venv and not in `requirements.txt`.
- Confirmed `add_turns` reads only `turn["text"]` for all three durable sinks (vector `:742`, `document` `:764`, FTS `content` `:791`); the `content` key set by adapters is never read by `add_turns`.

#### Top concerns

- No new top concerns beyond CODEX's three. The discovery is accurate and thorough.
- Minor completeness gap: `provider_opencode.py` has a second parse path (`_parse_legacy_source` at line 324, `content` mirror at line 370) not cited in the discovery table or the CODEX review. If the approach chooses to redact the `content` key on turn dicts, both opencode parse paths must be covered.
- The `_extract_issue_ids` ordering (redaction must fire before lines 779/804) is stated as an observation, not a constraint. The approach should lock this as an invariant.
- The AC-17 scrub surface extends further than the CODEX review enumerated: `rag_engine.py:283` (embedding identity error), `http_server.py:220,522,545,582` (multiple `str(exc)` paths beyond the ASGI handler), and the provider adapter parse error paths (`provider_claude.py:75`, `provider_antigravity.py:299`) also stringify exceptions. None of these are high-risk (they fire on infrastructure errors, not Turn text), but the approach should enumerate them explicitly rather than discover them during implementation.

#### Unverified assumptions

- Did not independently verify `detect-secrets` internal API (`scan_line`, `PotentialSecret.secret_value`, `KeywordDetector` filetype behavior) — relied on CODEX's Context7 research and upstream source.
- Did not check whether `_parse_legacy_source` in `provider_opencode.py` is still reachable in production or is a deprecated fallback.
