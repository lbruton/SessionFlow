---
sketch: "SESF-17-desktop-project-detection"
phase: discovery
created: 2026-05-28
updated: 2026-05-30
---

# SESF-17 — Discovery

_Research the existing system and prior art. **Don't propose solutions** — that's the next phase._

## Existing Code

_Files and modules already in the project that this work will touch or build on._

| Path | Role | Notes |
|------|------|-------|
| `provider_antigravity.py:22-35` | `AntigravityAdapter.__init__` | Desktop variant sets `self.root = ~/.gemini/antigravity`; CLI sets `~/.gemini/antigravity-cli`. The summaries `.pb` lives under the **desktop** root only — so the desktop/CLI split already isolates the new source (helps AC-6). |
| `provider_antigravity.py:37-59` | `_load_history()` | Reads `self.root / "history.jsonl"`. Desktop has no such file → returns `{}` today. Already catches `OSError` + `JSONDecodeError` and returns a partial/empty mapping — the existing graceful-degradation pattern to mirror for the summaries loader (AC-5). |
| `provider_antigravity.py:61-81` | `discover_sources()` | **The injection point.** Globs `brain/*/.system_generated/logs/transcript.jsonl`; `conversation_id = path.parents[2].name` (the `brain/<uuid>/` dir name); sets `project_root=history.get(conversation_id, "unknown")` (line 77). Loads history once at line 63 — the summaries map should load once here too (carried-forward perf note). |
| `provider_antigravity.py:172-173` | `_has_opaque_binary_artifacts()` | Globs `brain/**/*.pb` / `brain/**/*.db` — i.e. **per-conversation** artifacts, NOT the root-level `agyhub_summaries_proto.pb`. AC-8 wording must separate "summaries metadata now parsed" from these still-opaque brain artifacts. |
| `provider_antigravity.py:178-190` | `health()` | Emits `limitations = ["Protobuf/database artifacts are not parsed in SESF-6."]` (line 180). This is the literal AC-8 target string. |
| `provider_adapters.py:122-146` | `ProviderSource` dataclass | `project_root: str` (line 130). Confirms AC-7 — store the path string; no new field needed. |
| `provider_adapters.py:186-200` | `ProviderHealth` dataclass | `limitations: List[str]` — where AC-8 text surfaces. |
| `provider_adapters.py:88-110` | `normalize_timestamp()` | Precedent for a small stdlib coercion helper living in `provider_adapters.py` and reused by the adapter — a parallel home for a `file://`-normalizer or summaries-loader helper if shared. |
| `http_server.py:302-303` | Provider registry | `AntigravityAdapter(source_kind="desktop")` is already wired into the live provider list — no new registration needed. |
| `tests/test_provider_antigravity.py:23-34` | `test_..._does_not_claim_protobuf_support` | Asserts `"protobuf" in " ".join(health.limitations).lower()`. **AC-8 must keep this test green** — new wording still has to name protobuf artifacts (the opaque ones), just not claim the summaries file is unparsed. Update consciously, don't weaken. |
| `tests/conftest.py:190-207` | `synthetic_antigravity_home` | CLI-only fixture (writes `antigravity-cli/.../history.jsonl`). No desktop/`.pb` fixture exists yet — a synthetic desktop fixture (brain dirs + a hand-built `.pb`) is needed for AC-1/AC-2/AC-4/AC-5 tests. |

## Prior Decisions

_Search mem0 + recent sessions. Query strings recorded for reproducibility._

- mem0 query: `"SessionFlow antigravity desktop provider project_root resolution summaries protobuf SESF-6"`
- **2026-05-22 — SESF-19** (mem0 `9f20b491`): the timestamp fix in *this same file* added priority-ordered field resolution (`timestamp`/`created_at`/`createdAt`/`time`, in that order per `provider_antigravity.py:125-130`) routed through `normalize_timestamp()`, and content from first non-empty of `text`/`content`/`message`. **Precedent**: graceful, precedence-based resolution is the established pattern here — AC-3's `history → summaries → unknown` precedence fits it directly.
- **2026-05-22** (mem0 `cee918b3`): confirmed the desktop (`~/.gemini/antigravity/...`) vs CLI (`~/.gemini/antigravity-cli/...`) root split is intentional and these are distinct products sharing the `~/.gemini` tree.
- **2026-05-22** (mem0 `ac23aad4`): Antigravity backfill jobs only drain on server restart (no `/watch` trigger) — operational note: a desktop re-tag won't take effect until the next backfill/restart cycle (relevant to verifying the fix end-to-end, not to the code change itself).
- **No prior decision exists on parsing `agyhub_summaries_proto.pb`** — first time touching this surface. The SESF-6 limitation explicitly left protobuf artifacts unparsed.

## External References

_Libraries, formats, and stdlib facilities worth borrowing from._

- [Protocol Buffers — Encoding](https://protobuf.dev/programming-guides/encoding/) — wire format. Each field is a varint **tag** = `(field_number << 3) | wire_type`; workspace strings are **wire type 2 (length-delimited)**: a varint length followed by raw bytes. A schema-free decoder only needs varint + length-delimited handling to walk the conversation→workspace association. **Validated below** against the live file.
- Python stdlib `urllib.parse` — `urlparse` + `unquote` for AC-2 (`file://` → decoded filesystem path). No third-party dep.
- Python stdlib `int.from_bytes` / manual varint loop — sufficient for the decoder; **no `protobuf`/`google` package** (none is in `requirements.txt`).

## Constraints

_Things the implementation must respect._

- **Stdlib-only parser** — `requirements.txt` has no `protobuf`/`google` dependency, and adding the full schema package would contradict the "not parsing the full protobuf schema" non-goal. The decoder must be a schema-free varint walker (consistent with GEMINI's validated prototype).
- **Length-delimited framing is mandatory, not optional** — a greedy/`strings`-style scrape corrupts path boundaries (see validation: `SessionFlowz`). Must read the varint length prefix and slice exactly. Nested submessages (Field 2 → 9 → 1) are themselves wire type 2, so the walker must decode their lengths **recursively** (or iterate them sequentially) rather than flat-scanning — sibling fields can carry similar wire-type-2 tags and throw off a naive scan. String decode during traversal must tolerate `UnicodeDecodeError` (catch it or decode with `errors="replace"`) so a corrupt/locked file degrades to `{}` instead of crashing.
- **Shared adapter** — `cli` and `desktop` share one class. Summaries resolution must be gated to the desktop variant (the CLI root lacks the `.pb` anyway, but gate explicitly to guarantee AC-6 = CLI unchanged).
- **`.pb` is rewritten in place by the live daemon** — observed size drift (41,824 → 39,723 → 41,719 → 42,713 bytes across reads, mode `0600`) is illustrative, not stable. No live write race was observed *in progress*; the in-place rewrite is **inferred** from the snapshot drift plus the absence of temp/lock siblings at rest. Reads can nonetheless race a concurrent rewrite / hit a truncated file → must tolerate `OSError`/`PermissionError` and binary-boundary errors (`IndexError`/`ValueError`) per AC-5, falling back to `{}`.
- **Parse once per `discover_sources()`** — carried forward from the requirements reconcile (rejected as a requirements AC, owned here/approach): load + decode the `.pb` a single time per discovery pass, not per-source.
- **`project_root` is a path string** (`ProviderSource.project_root: str`) — AC-7. No project-name field.
- **Keep `test_..._does_not_claim_protobuf_support` green** — AC-8 rewording must still reference protobuf/opaque artifacts; don't delete the assertion.

## Validation of forward-guidance (from requirements Review Archive)

Both reviewers' *Unverified assumptions* were actioned against the live local artifact on 2026-05-30:

1. **CODEX — "validate field boundaries / extraction rule against real samples."**
   - `~/.gemini/antigravity/agyhub_summaries_proto.pb`: mode `0600`. **File size and distinct-UUID count are illustrative snapshots only** — they drift between reads (size seen at 39,723 / 41,719 / 42,713 bytes; distinct-UUID count at 127 / 138 across reads). Approach/tasks must **not** bake any byte size or UUID count into tests or parser expectations.
   - **Distinct UUIDs in the file ≫ 41 conversations** → the file holds far more than conversation IDs; extraction must follow the **structured field path** (GEMINI's Field 1 = conversation UUID, Field 2→9→1 = workspace URI), not scrape all UUIDs.
   - **Join validated (stable across reads):** of **41** `brain/` transcript conversations, **39 are present** in the `.pb` (AC-1 hits) and **2 are absent** (AC-4 → stay `unknown`). Confirms both ACs are exercised by real data.
   - **Field-boundary proof:** `strings` extraction yielded both `file:///Volumes/DATA/GitHub/SessionFlow` and a corrupted `file:///Volumes/DATA/GitHub/SessionFlowz`. The trailing `z` = `0x7A` = protobuf tag for **field 15, wire type 2** — the next field's framing byte. This concretely proves a naive string match corrupts the path and length-delimited parsing is required (CODEX's AC-2 concern, demonstrated).
   - **AC-2 %-decode caveat:** all local workspace paths are clean ASCII (no spaces/`%20`). The percent-decoding branch of AC-2 **cannot be exercised by the live file** — it must be covered by a synthetic fixture with an encoded path.
2. **GEMINI — "does the daemon rotate the `.pb` under a different filename / use temp files?"**
   - No `.tmp`/`.lock`/`.bak`/`~`/`.swp`/`.partial` siblings at rest in `~/.gemini/antigravity/`. The file shrank between snapshots → the daemon **rewrites in place** (or atomic-renames a temp that isn't observable at rest), not append-only rotation. The ingestion read must therefore tolerate a mid-rewrite truncated/short file (already mandated by AC-5).

## Open Questions

_None block `approach.md`._ The join, the structured-extraction requirement, the rotation/rewrite behavior, the dependency posture, and the test surface are all characterized. Residual items below are **implementation-time** decisions for the approach/tasks phases, not blockers:

- Exact defensive shape of the varint walker (how strictly to validate the Field 2→9→1 path vs. a more tolerant "first `file://` length-delimited string under each conversation record") — an approach-phase design choice; both reviewers' prototypes agree the association is recoverable.
- Whether the `file://`-normalizer and `.pb`-loader live as private methods on `AntigravityAdapter` or as shared helpers in `provider_adapters.py` (mirrors `normalize_timestamp`).
- The `file://`-normalizer should **fall back to the raw absolute path** if a bare (scheme-less) path is ever encountered — workspace URIs are *assumed* to carry the `file://` prefix, but that assumption is unverified for every record (GEMINI).
- Tests must explicitly exercise corrupt/truncated/locked `.pb` inputs and assert graceful `unknown` fallback (not just the happy-path join) — the synthetic desktop fixture should include a malformed variant alongside the valid one.

## Discovery Summary

The work lands almost entirely in `provider_antigravity.py:discover_sources()` (the `project_root=history.get(...)` line) plus a new stdlib varint reader for `~/.gemini/antigravity/agyhub_summaries_proto.pb` and an AC-8 health-text tweak — `ProviderSource`/registry/CLI variant all stay as-is. Live validation de-risked the core: 39/41 desktop conversations join cleanly, the `.pb` demands true length-delimited parsing (the `SessionFlowz` boundary bug proves a scrape won't do), and the file is rewritten in place by the daemon so AC-5's truncation handling is real, not hypothetical. The only thing real data can't cover is AC-2's percent-decoding (no encoded paths locally) — that needs a synthetic fixture, and a new desktop test fixture is required regardless.

---

> **Phase complete?** Existing code mapped. Prior decisions surfaced. Open questions resolved. Then advance: `/sketch-approach SESF-17`.

---

## Review Archive — discovery (2026-05-30)

### Resolution Summary

Reconciled 2026-05-30 via `/sketch-reconcile SESF-17 discovery`. **7 accepted / 0 rejected / 0 resolved with your input.** CODEX and GEMINI were in full agreement — no divergence.

- **Accepted:** (1) corrected the SESF-19 timestamp precedence in Prior Decisions to the live `timestamp → created_at → createdAt → time` order (verified at `provider_antigravity.py:127`); (2) reframed `.pb` byte sizes and UUID counts as illustrative snapshots, not parser/test input; (3) folded recursive nested-submessage (Field 2→9→1) decoding into the length-delimited Constraint; (4) added `UnicodeDecodeError` tolerance to the parser Constraint; (5) added an explicit corrupt/truncated `.pb` test-coverage note to Open Questions; (6) added a `file://`-scheme fallback-to-raw-path note for the normalizer; (7) reframed the in-place-rewrite constraint to note the write race is inferred from snapshot drift, not observed live.
- **Rejected:** none.
- **Your input:** none — all findings were factual corrections or non-divergent forward guidance.

The verbatim reviewer sections below are preserved as the audit trail.

## CODEX Review (2026-05-30)

### Verified

- Read `/Volumes/DATA/GitHub/DocVault/sketch/conventions.md`, SessionFlow `AGENTS.md`, `.context/GLOSSARY.md`, and the reconciled SESF-17 `requirements.md`.
- Checked `provider_antigravity.py:22-81` for root selection, history loading, `discover_sources()` injection point, and current `project_root="unknown"` behavior; checked `provider_antigravity.py:172-190` for opaque artifact and health limitation behavior.
- Checked `provider_adapters.py:88-110`, `provider_adapters.py:122-146`, `provider_adapters.py:186-200`, `http_server.py:297-304`, `tests/test_provider_antigravity.py:23-34`, `tests/conftest.py:190-207`, and `requirements.txt`.
- Re-ran live desktop discovery: `AntigravityAdapter(source_kind="desktop").discover_sources()` currently returns 41 sources, all with `project_root == "unknown"`.
- Re-checked the local summaries artifact: `agyhub_summaries_proto.pb` currently has mode `0600`, size 41,719 bytes, 138 distinct UUID strings, 41 brain transcripts, 39 transcript IDs mapped through Field 1 / Field 2→9→1, and 2 transcript IDs unmapped.
- Confirmed the Protocol Buffers encoding reference still supports the discovery's wire-format framing claims: tags are `(field_number << 3) | wire_type`, and wire type 2 is length-delimited for strings/bytes/submessages.

### Top concerns

- The Prior Decisions section overstates the exact SESF-19 timestamp precedence; live code uses `timestamp → created_at → createdAt → time`, not `created_at → createdAt → time → timestamp`.
- The binary artifact's exact size and UUID-count evidence is already stale. The mapping conclusion still holds, but the numeric snapshot should be framed as illustrative, not stable parser/test input.

### Unverified assumptions

- I did not observe an active daemon write race in progress; the in-place rewrite/truncation concern remains inferred from snapshot drift and should still be covered by corrupt/truncated synthetic fixtures.

## GEMINI Review (2026-05-30)

### Verified

- Checked [/Volumes/DATA/GitHub/DocVault/sketch/conventions.md](file:///Volumes/DATA/GitHub/DocVault/sketch/conventions.md), SessionFlow [AGENTS.md](file:///Volumes/DATA/GitHub/SessionFlow/AGENTS.md), and the reconciled SESF-17 [requirements.md](file:///Volumes/DATA/GitHub/DocVault/Projects/SessionFlow/sketches/SESF-17-desktop-project-detection/requirements.md).
- Verified current codebase files and paths listed in the Existing Code section:
  - [provider_antigravity.py:22-35](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py#L22-L35) (`AntigravityAdapter.__init__`)
  - [provider_antigravity.py:37-59](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py#L37-L59) (`_load_history()`)
  - [provider_antigravity.py:61-81](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py#L61-L81) (`discover_sources()`)
  - [provider_antigravity.py:172-173](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py#L172-L173) (`_has_opaque_binary_artifacts()`)
  - [provider_antigravity.py:178-190](file:///Volumes/DATA/GitHub/SessionFlow/provider_antigravity.py#L178-L190) (`health()`)
  - [provider_adapters.py:88-110](file:///Volumes/DATA/GitHub/SessionFlow/provider_adapters.py#L88-L110) (`normalize_timestamp()`)
  - [provider_adapters.py:122-146](file:///Volumes/DATA/GitHub/SessionFlow/provider_adapters.py#L122-L146) (`ProviderSource` dataclass)
  - [provider_adapters.py:186-200](file:///Volumes/DATA/GitHub/SessionFlow/provider_adapters.py#L186-L195) (`ProviderHealth` dataclass)
  - [http_server.py:302-303](file:///Volumes/DATA/GitHub/SessionFlow/http_server.py#L302-L303) (Adapter instantiation inside health cache)
  - [tests/test_provider_antigravity.py:23-34](file:///Volumes/DATA/GitHub/SessionFlow/tests/test_provider_antigravity.py#L23-L34) (`test_antigravity_adapter_does_not_claim_protobuf_support`)
  - [tests/conftest.py:190-207](file:///Volumes/DATA/GitHub/SessionFlow/tests/conftest.py#L190-L207) (`synthetic_antigravity_home` fixture)
- Checked provider adapter registry list inside [provider_ingestion.py:24-25](file:///Volumes/DATA/GitHub/SessionFlow/provider_ingestion.py#L24-L25).
- Re-ran the whole unit test suite (`./venv/bin/pytest`) and confirmed 124/124 tests pass successfully.
- Checked the local summaries file path `/Users/lbruton/.gemini/antigravity/agyhub_summaries_proto.pb` and confirmed it exists and size has drifted to 42,713 bytes, validating that binary counts and sizes are highly dynamic snapshots.

### Top concerns

- **UTF-8 Decoding Resilience**: String decoding during schema-free varint traversal must handle decoding failures gracefully (e.g. catch `UnicodeDecodeError` or decode with `errors="replace"`), ensuring a corrupt/locked file does not crash discovery.
- **Nested Submessage Boundaries**: The schema-free walker must decode nested submessage sizes recursively (Field 2 -> Field 9 -> Field 1) rather than assuming a flat tag scan, preventing collisions with sibling fields that might contain similar wire-type 2 tags.
- **Test Scenarios for Malformation**: Unit tests should explicitly verify parser robustness (returning `"unknown"` gracefully) under corrupted or truncated synthetic protobuf file inputs.

### Unverified assumptions

- We assume workspace URIs encoded in `agyhub_summaries_proto.pb` always carry a `file://` scheme prefix; the URL normalizer should fall back gracefully to return raw absolute paths if a bare path is ever encountered.
