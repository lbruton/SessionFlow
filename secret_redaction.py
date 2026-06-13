"""Pure, side-effect-free secret detection and redaction for ingested Turn text.

SessionFlow writes raw Turn text into three durable sinks (the embedding vector,
the Milvus ``document`` field, and the FTS Sidecar ``content`` column). This module
is the prevention guard (SESF-41): a single pure function, :func:`redact`, that
detects secrets in a string and replaces each with a typed placeholder of the form
``[REDACTED:<RULE_NAME>]``.

Design invariants (see the SESF-41 sketch ``approach.md``):

* **Pure** — no filesystem or network I/O. The operator allowlist is loaded by the
  caller and injected via the ``allowlist`` argument (AC-16). This module imports no
  I/O machinery.
* **Deterministic** — identical input bytes yield byte-identical output; there is no
  random salt, so re-ingesting the same source still dedups by ``doc_id`` (AC-14).
* **Idempotent** — ``redact(redact(x)) == redact(x)``; an existing ``[REDACTED:…]``
  placeholder is never re-matched or re-wrapped (AC-15).
* **No raw secret values** ever appear in this module, its callers' logs, or any
  reported metadata — only rule/class names (AC-18).

The detection engine is hybrid (D-1): ``detect-secrets`` supplies Tier-1 structured
provider tokens and the Tier-3 entropy plugins, while a thin local layer supplies
the custom Tier-1 regexes (Anthropic / Google / HuggingFace), Tier-2 contextual
``key: value`` / ``key=value`` assignment scanning, Tier-3 entropy-adjacency gating,
and PEM multi-line block expansion. Every detection path yields ``(raw_value, rule)``
pairs into one deterministic value-replacement masker (D-2).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterator, NamedTuple, Optional, Pattern

from detect_secrets.core.scan import scan_line
from detect_secrets.filters.heuristic import is_potential_uuid
from detect_secrets.settings import transient_settings

# Fallback rule name when a matched detector has no specific class name (AC-4).
_FALLBACK_RULE = "SECRET"

# --- detect-secrets configuration (D-1) -------------------------------------

# Tier-1 built-in structured-provider plugins, loaded via transient_settings.
# NOTE: detect-secrets' Base64/Hex entropy plugins are intentionally NOT used here.
# The low-level core.scan.scan_line() does not apply their entropy *limit*, so it
# flags every word in a line as "high entropy" (e.g. "is", "ok"). Tier-3 entropy is
# instead handled by the length-gated Shannon scanner below (_entropy_tokens), which
# matches D-3's advisory-only, adjacency-gated intent precisely.
_PLUGINS = [
    {"name": "AWSKeyDetector"},
    {"name": "SlackDetector"},
    {"name": "StripeDetector"},
    {"name": "OpenAIDetector"},
    {"name": "NpmDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "JwtTokenDetector"},
]

# detect-secrets ``secret_type`` → canonical SESF-41 rule name (Tier 1).
# GitHub / GitLab are handled by custom regexes below: detect-secrets reports a
# truncated prefix (e.g. "ghp") as the secret_value, which the value-replacement
# masker cannot use. The custom regexes capture the full token.
_SECRET_TYPE_MAP = {
    "AWS Access Key": "AWS",
    "Slack Token": "SLACK",
    "Stripe Access Key": "STRIPE",
    "OpenAI Token": "OPENAI",
    "NPM tokens": "NPM",
    "JSON Web Token": "JWT",
    "Private Key": "PRIVATE_KEY",
}

# Tier-3 entropy (D-3, AC-8): a length-gated Shannon scanner. A token must be at
# least _ENTROPY_MIN_LEN chars and reach _ENTROPY_THRESHOLD bits/char to be an
# advisory hit. Entropy hits are masked only when keyword-adjacent or fenced.
_ENTROPY_RULE = "HIGH_ENTROPY"
_ENTROPY_MIN_LEN = 20
_ENTROPY_THRESHOLD = 4.0
_ENTROPY_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{%d,}" % _ENTROPY_MIN_LEN)

# Custom Tier-1 regexes: classes with no built-in plugin (Anthropic / Google /
# HuggingFace) plus GitHub / GitLab, whose built-in plugins report only a truncated
# prefix as secret_value (D-1 mapping). Each captures the full token for masking.
_CUSTOM_TIER1 = [
    ("ANTHROPIC", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("GOOGLE_API_KEY", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("HUGGINGFACE", re.compile(r"hf_[A-Za-z0-9]{34,}")),
    ("GITHUB", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("GITLAB", re.compile(r"glpat-[A-Za-z0-9_\-]{20,}")),
]

# PEM / multi-line private-key blocks — masked whole (D-6); detect-secrets'
# PrivateKeyDetector only flags the marker line.
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

# Tier-2 contextual assignment: a secret-bearing key, then ':'/'=', then a value
# (optionally quoted). Covers the MCP-config-dump leak class (AC-6).
_SECRET_KEYWORD = (  # nosec B105 — detection regex of keyword NAMES, not a credential
    r"api[_-]?key|secret|token|password|passwd|pwd|auth|credential|access[_-]?key"
)
_ASSIGN_RE = re.compile(
    r"(?i)(?P<key>[A-Za-z0-9_\-]*(?:" + _SECRET_KEYWORD + r")[A-Za-z0-9_\-]*)"
    r"[\"']?\s*[:=]\s*"  # optional closing quote on the key (JSON "KEY": "value")
    r"[\"']?(?P<value>[^\"'\s,}{]+)[\"']?"
)
_KEYWORD_RE = re.compile(r"(?i)(?:" + _SECRET_KEYWORD + r")")

# Value-shape guard (AC-7): reject placeholder shapes / sub-minimum-length values.
_MIN_VALUE_LEN = 8
_PLACEHOLDER_WORDS = {
    "changeme", "changethis", "password", "secret", "example", "test",
    "your-key", "yourkey", "redacted", "placeholder", "none", "null", "todo",
}
_PLACEHOLDER_SHAPE_RE = re.compile(r"^[\*x\-_<>.\s]+$", re.IGNORECASE)

# Existing redaction placeholders — protected spans for the idempotent masker.
_PLACEHOLDER_SPAN_RE = re.compile(r"\[REDACTED:[^\]]*\]")

# Hex shapes treated as structurally non-secret (AC-9): 40-hex git SHA, 64-hex
# SHA-256 / content-hash doc_id.
_HEX_ALLOW_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
# Issue-ID shape (e.g. SESF-41) — never a secret (AC-9).
_ISSUE_ID_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


class Hit(NamedTuple):
    """A single detected secret, carrying only non-sensitive metadata.

    Deliberately holds no raw secret value so a reporting consumer (the AC-10
    per-rule histogram / counters) can never echo cleartext (AC-18).

    Attributes:
        rule_name: The detector class name used to build the placeholder, e.g.
            ``"OPENAI"`` → ``[REDACTED:OPENAI]``.
        tier: The detection tier that produced the hit (1 = structured provider
            tokens / PEM / JWT, 2 = contextual assignment, 3 = entropy).
    """

    rule_name: str
    tier: int


class Span(NamedTuple):
    """A single detected secret occurrence with value-free audit metadata (SESF-42).

    Unlike :class:`Hit` (which dedups by value), one ``Span`` is emitted per
    *occurrence*: the same secret appearing twice yields two spans. The offsets
    index into the original scanned text and are integers (not sensitive); the
    snippet is pre-masked so no raw value ever leaves the module (AC-18).

    Attributes:
        rule_name: The detector class name used to build the placeholder, e.g.
            ``"OPENAI"`` → ``[REDACTED:OPENAI]``.
        tier: The detection tier that produced the hit (1 = structured provider
            tokens / PEM / JWT, 2 = contextual assignment, 3 = entropy).
        start: Offset of the occurrence in the original text (inclusive).
        end: Offset of the occurrence in the original text (exclusive), so
            ``text[start:end]`` is the matched value.
        masked_snippet: A fixed-width window of the original text around the match
            (±:data:`_SNIPPET_PAD` chars) in which every detected occurrence — the
            target and any reported neighbour intersecting the window — is replaced
            by its typed ``[REDACTED:…]`` placeholder, masked deterministically by
            offset (not by re-running detection over the slice). It never contains a
            raw secret value.
    """

    rule_name: str
    tier: int
    start: int
    end: int
    masked_snippet: str


# Half-width of the value-free context window stored on each Span (SESF-42). The
# window is ``text[start - _SNIPPET_PAD : end + _SNIPPET_PAD]``, then masked.
_SNIPPET_PAD = 24


def _placeholder(rule_name: str) -> str:
    """Render the typed redaction placeholder for ``rule_name`` (AC-4 fallback)."""
    return f"[REDACTED:{rule_name or _FALLBACK_RULE}]"


def _is_placeholder_value(value: str) -> bool:
    """Return True if a Tier-2 candidate value is a placeholder or too short (AC-7)."""
    stripped = value.strip().strip("\"'")
    inner = stripped.strip("<>")
    if len(inner) < _MIN_VALUE_LEN:
        return True
    if inner.lower() in _PLACEHOLDER_WORDS:
        return True
    if _PLACEHOLDER_SHAPE_RE.match(stripped):
        return True
    return False


def _is_structurally_allowlisted(value: str) -> bool:
    """Return True if ``value`` is a structurally non-secret shape (AC-9).

    UUID / 40-hex git SHA / 64-hex SHA-256 / content-hash doc_id / issue ID. This is
    applied ONLY to Tier-3 entropy candidates — a Tier-1/Tier-2 detector hit that
    happens to be all-hex (e.g. a hex secret in `api_key=<64hex>`) must still be
    redacted, so the hex shape is not a blanket exemption.
    """
    return bool(
        is_potential_uuid(value)
        or _HEX_ALLOW_RE.match(value)
        or _ISSUE_ID_RE.match(value)
    )


def _is_operator_allowlisted(value: str, allowlist: list[Pattern[str]]) -> bool:
    """Return True if ``value`` matches an operator-supplied allowlist pattern (AC-9)."""
    return any(pattern.search(value) for pattern in allowlist)


def _shannon_entropy(value: str) -> float:
    """Return the Shannon entropy of ``value`` in bits per character."""
    if not value:
        return 0.0
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(value).values()
    )


def _entropy_tokens(line: str) -> Iterator[str]:
    """Yield length-gated, high-entropy tokens on ``line`` (Tier-3, AC-8)."""
    for match in _ENTROPY_TOKEN_RE.finditer(line):
        token = match.group(0)
        if _shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            yield token


def _keyword_adjacent(line: str, value: str) -> bool:
    """Return True if a keyword sits within 40 chars of ANY occurrence of ``value`` (AC-8).

    Checks every occurrence of ``value`` on the line, not just the first — a token can
    repeat and only a later occurrence be keyword-adjacent.
    """
    keyword_positions = [match.start() for match in _KEYWORD_RE.finditer(line)]
    if not keyword_positions:
        return False
    cursor = 0
    while True:
        idx = line.find(value, cursor)
        if idx == -1:
            return False
        if any(abs(pos - idx) <= 40 for pos in keyword_positions):
            return True
        cursor = idx + 1


def _replace_outside_placeholders(text: str, value: str, replacement: str) -> str:
    """Replace every ``value`` not already inside a ``[REDACTED:…]`` span (AC-15)."""
    if not value:
        return text
    protected = [
        (m.start(), m.end()) for m in _PLACEHOLDER_SPAN_RE.finditer(text)
    ]
    out: list[str] = []
    cursor = 0
    while True:
        idx = text.find(value, cursor)
        if idx == -1:
            out.append(text[cursor:])
            break
        inside = any(start <= idx < end for start, end in protected)
        if inside:
            out.append(text[cursor:idx + len(value)])
        else:
            out.append(text[cursor:idx])
            out.append(replacement)
        cursor = idx + len(value)
    return "".join(out)


def _collect_candidates(text: str) -> list[tuple[str, str, int, bool]]:
    """Collect ``(value, rule_name, tier, force_mask)`` detection candidates.

    ``force_mask`` is True for Tier-1/Tier-2 (always masked in enforce mode) and for
    Tier-3 entropy hits that are keyword-adjacent or inside a fenced block (AC-8).
    Allowlisted values are dropped later by ``_is_allowlisted`` in :func:`redact`.
    """
    candidates: list[tuple[str, str, int, bool]] = []

    # D-6: PEM blocks (whole-block) before any line scanning, off the original text.
    for match in _PEM_RE.finditer(text):
        candidates.append((match.group(0), "PRIVATE_KEY", 1, True))

    # Strip PEM blocks so key bodies are never re-scanned as entropy.
    work = _PEM_RE.sub("\n", text)

    # Custom Tier-1 regexes (Anthropic / Google / HuggingFace).
    for rule_name, pattern in _CUSTOM_TIER1:
        for match in pattern.finditer(work):
            candidates.append((match.group(0), rule_name, 1, True))

    # Line scan: detect-secrets Tier-1 structured plugins, the Shannon entropy
    # scanner (Tier-3), and the Tier-2 assignment scanner.
    in_fence = False
    with transient_settings({"plugins_used": _PLUGINS}):
        for line in work.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            # Tier-1 structured providers (ignore any non-mapped types).
            for secret in scan_line(line):
                value = secret.secret_value
                rule_name = _SECRET_TYPE_MAP.get(secret.type)
                if value and rule_name is not None:
                    candidates.append((value, rule_name, 1, True))
            # Tier-3 entropy (advisory; force-masked on adjacency or in a fence).
            for value in _entropy_tokens(line):
                force = in_fence or _keyword_adjacent(line, value)
                candidates.append((value, _ENTROPY_RULE, 3, force))
            # Tier-2 contextual assignments.
            for match in _ASSIGN_RE.finditer(line):
                value = match.group("value")
                if _is_placeholder_value(value):
                    continue
                candidates.append((value, "ASSIGNMENT", 2, True))

    return candidates


def redact(
    text: str,
    *,
    mode: str,
    allowlist: Optional[list[Pattern[str]]] = None,
) -> tuple[str, list[Hit]]:
    """Detect secrets in ``text`` and return the redacted text plus the hit list.

    Pure and side-effect-free: performs no I/O and no network calls (AC-16). The
    output is deterministic (AC-14) and idempotent (AC-15).

    Args:
        text: The Turn text to scan.
        mode: ``"enforce"`` to replace each detected secret with a
            ``[REDACTED:<RULE_NAME>]`` placeholder, or ``"report"`` to detect and
            return hits without mutating ``text``.
        allowlist: Operator-supplied compiled regex patterns whose matches are
            exempt from detection (AC-9). Loaded by the caller to keep this module
            pure (D-4); ``None`` means no operator patterns.

    Returns:
        A ``(redacted_text, hits)`` tuple. In ``report`` mode ``redacted_text`` is
        ``text`` unchanged; in both modes ``hits`` lists every detected secret as a
        :class:`Hit` (rule name + tier, never a raw value). Allowlisted values are
        excluded from both the hit list and any masking.
    """
    allowlist = allowlist or []

    # De-duplicate by value so one secret yields one classification. When the same
    # value is matched by several tiers (e.g. a structured token the entropy scanner
    # also flags), the lowest tier wins and ``force`` is OR-ed across occurrences.
    # Allowlist (AC-9): operator patterns apply to ALL tiers; the structural shape
    # exemption (UUID/hex/issue) applies to Tier-3 entropy ONLY, so a high-confidence
    # Tier-1/Tier-2 hit on an all-hex secret is still redacted.
    agg: dict[str, dict] = {}
    for value, rule_name, tier, force in _collect_candidates(text):
        if _is_operator_allowlisted(value, allowlist):
            continue
        if tier == 3 and _is_structurally_allowlisted(value):
            continue
        entry = agg.get(value)
        if entry is None:
            agg[value] = {"tier": tier, "rule": rule_name, "force": force}
        else:
            if tier < entry["tier"]:
                entry["tier"], entry["rule"] = tier, rule_name
            entry["force"] = entry["force"] or force

    hits = [Hit(rule_name=e["rule"], tier=e["tier"]) for e in agg.values()]

    if mode != "enforce":
        return text, hits

    # Mask by value-replacement in tier order (Tier-1, then Tier-2, then forced
    # Tier-3), longest value first within each tier (D-2). Tier order ensures a
    # structured/assignment value wins over an overlapping entropy superset; the
    # placeholder-aware replace then leaves the already-masked span untouched (AC-15).
    out = text
    for target_tier in (1, 2, 3):
        tier_items = [
            (value, entry["rule"])
            for value, entry in agg.items()
            if entry["tier"] == target_tier and (target_tier in (1, 2) or entry["force"])
        ]
        tier_items.sort(key=lambda pair: len(pair[0]), reverse=True)
        for value, rule_name in tier_items:
            out = _replace_outside_placeholders(out, value, _placeholder(rule_name))
    return out, hits


def _maskable_values(
    text: str,
    allowlist: list[Pattern[str]],
) -> list[tuple[str, str, int]]:
    """Return the ``(value, rule_name, tier)`` triples that enforce mode would mask.

    Mirrors :func:`redact`'s aggregation exactly — operator allowlist on all tiers,
    structural shape exemption on Tier-3 only, dedup-by-value with the lowest tier
    winning and ``force`` OR-ed — but returns one triple per distinct value rather
    than a :class:`Hit`. A Tier-3 value is included only when it is forced (keyword
    adjacent or fenced), matching the values enforce actually replaces. Ordered by
    tier then descending value length so longer/structured values are placed first,
    the same precedence the masker uses.
    """
    agg: dict[str, dict] = {}
    for value, rule_name, tier, force in _collect_candidates(text):
        if _is_operator_allowlisted(value, allowlist):
            continue
        if tier == 3 and _is_structurally_allowlisted(value):
            continue
        entry = agg.get(value)
        if entry is None:
            agg[value] = {"tier": tier, "rule": rule_name, "force": force}
        else:
            if tier < entry["tier"]:
                entry["tier"], entry["rule"] = tier, rule_name
            entry["force"] = entry["force"] or force

    maskable = [
        (value, entry["rule"], entry["tier"])
        for value, entry in agg.items()
        if entry["tier"] in (1, 2) or entry["force"]
    ]
    maskable.sort(key=lambda triple: (triple[2], -len(triple[0])))
    return maskable


def _collect_spans(text: str, allowlist: list[Pattern[str]]) -> list[Span]:
    """Collect per-occurrence :class:`Span` metadata for ``text`` (SESF-42).

    Locates every occurrence of each maskable value in the *original* ``text``,
    skipping occurrences inside an existing ``[REDACTED:…]`` placeholder (AC-15)
    and occurrences overlapping an already-claimed higher-precedence span, so an
    entropy superset never double-counts a structured token it contains. Each
    surviving occurrence becomes its own span (no dedup-by-value). The snippet is
    produced last, once the full span set is known, so neighbours can be masked.
    """
    protected = [(m.start(), m.end()) for m in _PLACEHOLDER_SPAN_RE.finditer(text)]
    claimed: list[tuple[int, int]] = []
    raw: list[tuple[str, int, int, int]] = []  # (rule_name, tier, start, end)

    def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
        return any(start < c_end and c_start < end for c_start, c_end in spans)

    for value, rule_name, tier in _maskable_values(text, allowlist):
        if not value:
            continue
        cursor = 0
        while True:
            idx = text.find(value, cursor)
            if idx == -1:
                break
            end = idx + len(value)
            cursor = end
            if _overlaps(protected, idx, end):
                continue
            if _overlaps(claimed, idx, end):
                continue
            claimed.append((idx, end))
            raw.append((rule_name, tier, idx, end))

    raw.sort(key=lambda item: item[2])

    # Build each snippet deterministically by masking EVERY detected occurrence in
    # ``raw`` that intersects the window — the target plus any reported neighbour —
    # rather than re-running ``redact()`` on a truncated window (SESF-42). The old
    # window-redact path re-detected against the slice, so a forced Tier-3 entropy
    # token whose forcing keyword fell outside the ±_SNIPPET_PAD window was
    # re-evaluated as un-forced and left RAW. Masking by offset guarantees every
    # reported span's snippet is value-free regardless of tier or keyword distance.
    # Occurrences overlapping an existing ``[REDACTED:…]`` span are absent from
    # ``raw`` (skipped above), so emitting the window text verbatim between
    # occurrences preserves those placeholders without double-wrapping (AC-15).
    spans: list[Span] = []
    for rule_name, tier, start, end in raw:
        w_start = max(0, start - _SNIPPET_PAD)
        w_end = end + _SNIPPET_PAD
        parts: list[str] = []
        cursor = w_start
        for occ_rule, _occ_tier, occ_start, occ_end in raw:
            if occ_end <= w_start or occ_start >= w_end:
                continue  # occurrence does not intersect the window
            clamped_start = max(occ_start, w_start)
            clamped_end = min(occ_end, w_end)
            if clamped_start > cursor:
                parts.append(text[cursor:clamped_start])
            parts.append(_placeholder(occ_rule))
            cursor = max(cursor, clamped_end)
        parts.append(text[cursor:w_end])
        masked_snippet = "".join(parts)
        spans.append(
            Span(
                rule_name=rule_name,
                tier=tier,
                start=start,
                end=end,
                masked_snippet=masked_snippet,
            )
        )
    return spans


def scan_spans(
    text: str,
    *,
    mode: str,
    allowlist: Optional[list[Pattern[str]]] = None,
) -> tuple[str, list[Span]]:
    """Detect secrets in ``text`` and return per-occurrence, value-free spans (SESF-42).

    Companion to :func:`redact` for retroactive auditing: where ``redact`` returns
    one :class:`Hit` per distinct value, this returns one :class:`Span` per
    occurrence, with offsets into the original ``text`` and a pre-masked context
    snippet. Pure, deterministic, and idempotent — it shares ``redact``'s detection
    tiers, allowlist handling, and placeholder-aware skipping, and leaves
    ``redact``'s own public contract unchanged.

    Args:
        text: The text to scan.
        mode: ``"enforce"`` to also return the fully masked text (identical to
            ``redact(text, mode="enforce")``), or ``"report"`` to return ``text``
            unchanged. The span list is identical in both modes.
        allowlist: Operator-supplied compiled regex patterns whose matches are
            exempt from detection — applied exactly as in :func:`redact` (all tiers
            for operator patterns; the structural shape exemption is Tier-3 only).
            ``None`` means no operator patterns.

    Returns:
        A ``(scanned_text, spans)`` tuple. ``scanned_text`` is the masked text in
        ``enforce`` mode and the unchanged ``text`` in ``report`` mode; ``spans``
        lists every detected occurrence as a :class:`Span` (rule/tier/offsets plus
        a value-free snippet), the same in both modes.
    """
    allowlist = allowlist or []
    spans = _collect_spans(text, allowlist)
    if mode == "enforce":
        scanned_text, _ = redact(text, mode="enforce", allowlist=allowlist)
    else:
        scanned_text = text
    return scanned_text, spans
