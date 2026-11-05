"""Issue-ID patterns: which tokens in a turn count as ticket references.

The dashed pattern (Plane, Linear, Jira) is always on. A config file adds
presets, ``regex:`` patterns and ``deny:`` prefixes; see
``issue_ids.conf.example``. Matching ignores case, and tokens are stored in
upper case.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger("sessionflow.issue_patterns")

ENV_VAR = "SESSIONFLOW_ISSUE_ID_PATTERNS"
DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".sessionflow", "issue_ids.conf")

PRESETS: Dict[str, str] = {
    "dashed": r"\b[A-Z][A-Z0-9]+-\d+\b",
    "servicenow": r"\b(?:INC|CHG|REQ|RITM|CTASK|PRB|SCTASK|TASK)\d{7}\b",
}

# Technical-standard prefixes that match the dashed pattern but never name a ticket.
BUILTIN_DENYLIST = frozenset(
    {"UTF", "SHA", "HTTP", "HTTPS", "ISO", "RFC", "IPV", "MD", "BASE"}
)

# A stored token sits in a comma-wrapped field and in Milvus LIKE and FTS MATCH
# filters, so it holds only letters, digits and inner dashes.
_SAFE_TOKEN = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_MAX_TOKEN_CHARS = 64
_DENY_PREFIX = re.compile(r"^[A-Z0-9]+$")
_LEADING_LETTERS = re.compile(r"^[A-Z]+")


@dataclass(frozen=True)
class IssuePatterns:
    """The compiled issue-ID patterns and prefix denylist in force.

    Attributes:
        labels: One label per pattern, such as ``preset:dashed`` or ``regex:<pattern>``.
        regexes: The compiled patterns, in ``labels`` order.
        deny: Prefixes added by the config file, on top of ``BUILTIN_DENYLIST``.
        source: The config file read, or None for the built-in default.
        skipped: Config lines that were not understood.
    """

    labels: Tuple[str, ...]
    regexes: Tuple[Pattern[str], ...]
    deny: frozenset = field(default_factory=frozenset)
    source: Optional[str] = None
    skipped: int = 0

    def _denied(self, token: str) -> bool:
        if "-" in token:
            prefix = token.split("-", 1)[0]
        else:
            match = _LEADING_LETTERS.match(token)
            prefix = match.group(0) if match else ""
        return prefix in BUILTIN_DENYLIST or prefix in self.deny

    def _keep(self, token: str) -> bool:
        return (len(token) <= _MAX_TOKEN_CHARS and _SAFE_TOKEN.match(token) is not None
                and not self._denied(token))

    def extract(self, text: str) -> List[str]:
        """Return the issue tokens in ``text``, upper case, once each, in text order.

        Args:
            text: Turn or memory text to scan.

        Returns:
            Tokens matched by any pattern, minus denied prefixes and tokens with
            characters other than letters, digits and inner dashes.
        """
        if not text or not isinstance(text, str):
            return []
        hits = []
        for regex in self.regexes:
            for match in regex.finditer(text):
                hits.append((match.start(), match.group(0).upper()))
        hits.sort(key=lambda hit: hit[0])
        seen: List[str] = []
        seen_set: set = set()
        for _, token in hits:
            if token in seen_set or not self._keep(token):
                continue
            seen_set.add(token)
            seen.append(token)
        return seen

    def matches(self, raw: object) -> bool:
        """Whether ``raw`` is one whole issue token that extraction would store.

        Args:
            raw: A caller-supplied filter value; whitespace and case are ignored.

        Returns:
            False for non-strings, unsafe characters, denied prefixes and tokens
            no pattern matches in full.
        """
        if not isinstance(raw, str):
            return False
        token = raw.strip().upper()
        if not self._keep(token):
            return False
        return any(regex.fullmatch(token) for regex in self.regexes)

    def describe(self) -> Dict[str, object]:
        """Return the source file, pattern labels, extra denied prefixes and skipped-line count."""
        return {
            "source": self.source,
            "patterns": list(self.labels),
            "deny": sorted(self.deny),
            "skipped": self.skipped,
        }


def format_lines(info: Dict[str, object]) -> List[str]:
    """Return bullet lines for an ``IssuePatterns.describe`` dict.

    Args:
        info: The dict from ``describe``, as ``get_stats`` reports it.

    Returns:
        Lines naming the patterns, any extra denied prefixes and the config file.
    """
    lines = [f"- Patterns: {', '.join(info.get('patterns') or [])}"]
    if info.get("deny"):
        lines.append(f"- Denied prefixes: {', '.join(info['deny'])}")
    source = str(info.get("source") or "none (built-in default)")
    skipped = int(info.get("skipped") or 0)
    if skipped:
        source += f" ({skipped} line{'' if skipped == 1 else 's'} skipped)"
    lines.append(f"- Config: {source}")
    return lines


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _default() -> IssuePatterns:
    return IssuePatterns(labels=("preset:dashed",), regexes=(_compile(PRESETS["dashed"]),))


def _strip_comment(value: str) -> str:
    """Drop a `` # comment`` from a preset or deny value; regex values keep every character."""
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def parse(lines: List[str], source: Optional[str] = None) -> IssuePatterns:
    """Build issue patterns from config lines.

    Args:
        lines: Config file lines. Blank lines and ``#`` comments are ignored.
        source: The file the lines came from, for warnings and ``describe``.

    Returns:
        The dashed preset plus every valid line; a bad line logs a WARN with
        its line number and counts in ``skipped``.
    """
    labels: List[str] = ["preset:dashed"]
    regexes: List[Pattern[str]] = [_compile(PRESETS["dashed"])]
    deny: set = set()
    skipped = 0
    where = source or "issue ID config"

    def bad(line_no: int, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        logger.warning("Skipping %s line %d: %s", where, line_no, reason)

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kind, sep, value = stripped.partition(":")
        kind = kind.strip().lower()
        if not sep:
            bad(line_no, "expected preset:, regex: or deny:")
        elif kind == "preset":
            name = _strip_comment(value).lower()
            if name not in PRESETS:
                bad(line_no, f"unknown preset {name!r}")
            elif f"preset:{name}" not in labels:
                labels.append(f"preset:{name}")
                regexes.append(_compile(PRESETS[name]))
        elif kind == "regex":
            pattern = value.strip()
            try:
                compiled = _compile(pattern)
            except re.error as exc:
                bad(line_no, f"invalid regex ({exc})")
                continue
            if not pattern:
                bad(line_no, "empty regex")
            elif f"regex:{pattern}" not in labels:
                labels.append(f"regex:{pattern}")
                regexes.append(compiled)
        elif kind == "deny":
            prefix = _strip_comment(value).upper()
            if not _DENY_PREFIX.match(prefix):
                bad(line_no, "a deny prefix holds only letters and digits")
            else:
                deny.add(prefix)
        else:
            bad(line_no, f"unknown entry {kind!r}")

    return IssuePatterns(labels=tuple(labels), regexes=tuple(regexes),
                         deny=frozenset(deny), source=source, skipped=skipped)


def load(path: Optional[str]) -> IssuePatterns:
    """Read issue patterns from ``path``.

    Args:
        path: Config file path; None returns the built-in default.

    Returns:
        The parsed patterns, or the default with a WARN when the file cannot be read.
    """
    if not path:
        return _default()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        logger.warning("Could not read issue ID config %s: %s; using the dashed preset",
                       path, exc)
        return _default()
    return parse(lines, source=path)


_cache: Dict[str, object] = {}
_cache_lock = threading.Lock()


def _resolve_path() -> Tuple[Optional[str], Optional[float]]:
    """Return the config path in force and its mtime; a missing default file means none."""
    configured = os.getenv(ENV_VAR)
    path = configured or DEFAULT_PATH
    try:
        return path, os.path.getmtime(path)
    except OSError:
        return (path, None) if configured else (None, None)


def active() -> IssuePatterns:
    """Return the patterns in force, reloading the config file when its mtime changes.

    The file is ``SESSIONFLOW_ISSUE_ID_PATTERNS``, else ``~/.sessionflow/issue_ids.conf``
    when it exists. A configured path that is missing warns once and uses the default.
    """
    path, mtime = _resolve_path()
    key = (path, mtime)
    with _cache_lock:
        if _cache.get("key") == key:
            return _cache["patterns"]  # type: ignore[return-value]
    patterns = load(path)
    with _cache_lock:
        _cache["key"] = key
        _cache["patterns"] = patterns
    return patterns


def reset() -> None:
    """Forget the cached patterns, so the next ``active`` call reads the config again."""
    with _cache_lock:
        _cache.clear()
