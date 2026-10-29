"""Strip harness boilerplate from turn text before it is embedded and stored.

Pure module: ``strip(text) -> (clean_text, removed)`` where ``removed`` maps a rule
name to the number of blocks it removed. A block counts only when its opening tag
starts a line, so a tag mentioned inside a sentence stays in the text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_USER = "User: "
_ASSISTANT = "Assistant: "
_SEPARATOR = "\n\n" + _ASSISTANT

# Opening tag at the start of a section or a line.
_LINE_START = r"(?:\A|(?<=\n))"


def _block(tag: str) -> re.Pattern:
    # A block missing its closing tag (a truncated turn) runs to the section end.
    return re.compile(
        _LINE_START + rf"<{re.escape(tag)}>.*?(?:</{re.escape(tag)}>\n?|\Z)", re.DOTALL,
    )


# Applied in order: an outer block removes anything nested inside it.
_REMOVE_RULES: List[Tuple[str, re.Pattern]] = [
    ("compaction_summary", re.compile(
        r"\AThis session is being continued from a previous conversation.*\Z", re.DOTALL,
    )),
    ("task_notification", _block("task-notification")),
    ("side_sessions_event", _block("side-sessions-event")),
    ("ci_monitor_event", _block("ci-monitor-event")),
    ("system_reminder", _block("system-reminder")),
    ("wake", _block("wake")),
    ("project_claude_message", _block("project_claude_message")),
    ("local_command", _block("local-command-caveat")),
    ("local_command", _block("local-command-stdout")),
    ("local_command", _block("local-command-stderr")),
    ("bash_output", _block("bash-stdout")),
    ("bash_output", _block("bash-stderr")),
]

# Claude Code indents the second and later tags of some echoes.
_COMMAND_RUN = re.compile(
    _LINE_START
    + r"(?:[ \t]*<command-(?:message|name|args)>.*?</command-(?:message|name|args)>[ \t]*\n?)+",
    re.DOTALL,
)
_COMMAND_PART = re.compile(r"<command-(message|name|args)>(.*?)</command-\1>", re.DOTALL)
_BASH_INPUT = re.compile(_LINE_START + r"<bash-input>(.*?)</bash-input>[ \t]*\n?", re.DOTALL)


def _command_marker(match: re.Match) -> str:
    parts = {kind: value.strip() for kind, value in _COMMAND_PART.findall(match.group(0))}
    name = parts.get("name") or "/" + parts.get("message", "")
    args = parts.get("args", "")
    return f"[{name} {args}]\n" if args else f"[{name}]\n"


def _strip_section(section: str, removed: Dict[str, int]) -> str:
    for rule, pattern in _REMOVE_RULES:
        section, count = pattern.subn("", section)
        if count:
            removed[rule] = removed.get(rule, 0) + count
    section, count = _COMMAND_RUN.subn(_command_marker, section)
    if count:
        removed["command_echo"] = removed.get("command_echo", 0) + count
    section = _BASH_INPUT.sub(lambda m: f"[! {m.group(1).strip()}]\n", section)
    return section.strip()


def strip(text: str) -> Tuple[str, Dict[str, int]]:
    """Remove harness boilerplate from a turn.

    The user and assistant parts of a ``User: …\\n\\nAssistant: …`` turn are stripped
    separately, and a part left empty loses its label. Slash-command echoes become a
    ``[/name args]`` marker and shell input a ``[! command]`` marker. Text with
    nothing to remove is returned unchanged.

    Args:
        text: The turn text as a provider built it.

    Returns:
        ``(clean_text, removed)``, where ``removed`` counts the blocks per rule.
    """
    removed: Dict[str, int] = {}
    if text.startswith(_USER):
        user, sep, assistant = text[len(_USER):].partition(_SEPARATOR)
        sections = [(_USER, user)] + ([(_ASSISTANT, assistant)] if sep else [])
    elif text.startswith(_ASSISTANT):
        sections = [(_ASSISTANT, text[len(_ASSISTANT):])]
    else:
        sections = [("", text)]

    cleaned = [(label, _strip_section(body, removed)) for label, body in sections]
    if not removed:
        return text, removed
    return "\n\n".join(label + body for label, body in cleaned if body), removed
