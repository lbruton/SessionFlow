"""Token-bounded passages of long turns.

A turn longer than ``PASSAGE_TOKENS`` keeps its full parent row and also gets
passage rows that cover it in overlapping windows. Each window fits inside the
embedding model's 512-token sliding window, so a passage embeds the way a short
turn does. The module never imports ``rag_engine``, which imports it; callers
pass the token offsets.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from provider_adapters import REPLACE_SAME_TURN

PASSAGE_CHUNK_TYPE = "turn_passage"
PASSAGE_TOKENS = 450
PASSAGE_OVERLAP = 64
TITLE_CHARS = 120

_SUFFIX = re.compile(r"::p(\d+)$")
_NO_PROJECT = {"", "/", "unknown"}


def passage_doc_id(parent_doc_id: str, index: int) -> str:
    """Return the doc_id of passage ``index`` of ``parent_doc_id``."""
    return f"{parent_doc_id}::p{index}"


def passage_parent(doc_id: str) -> Optional[str]:
    """Return the parent doc_id of a passage doc_id, or None for any other doc_id."""
    m = _SUFFIX.search(doc_id)
    return doc_id[:m.start()] if m else None


def passage_index(doc_id: str) -> Optional[int]:
    """Return the passage ordinal of a passage doc_id, or None for any other doc_id."""
    m = _SUFFIX.search(doc_id)
    return int(m.group(1)) if m else None


def passage_spans(offsets: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return the character spans of the passages for a text's token offsets.

    Empty spans (special tokens) are ignored. A text of ``PASSAGE_TOKENS`` or
    fewer tokens has no passages. Windows step by ``PASSAGE_TOKENS -
    PASSAGE_OVERLAP``, and the last window ends at the last token.
    """
    tokens = [(s, e) for s, e in offsets if e > s]
    if len(tokens) <= PASSAGE_TOKENS:
        return []
    step = PASSAGE_TOKENS - PASSAGE_OVERLAP
    spans = []
    start = 0
    while True:
        end = start + PASSAGE_TOKENS
        if end >= len(tokens):
            start = len(tokens) - PASSAGE_TOKENS
            spans.append((tokens[start][0], tokens[-1][1]))
            return spans
        spans.append((tokens[start][0], tokens[end - 1][1]))
        start += step


def _user_message(text: str) -> str:
    if not text.startswith("User: "):
        return ""
    return text[len("User: "):].split("\n\nAssistant: ", 1)[0]


def passage_title(project_root: str, text: str) -> str:
    """Return the embedding title of a turn's passages: project and user question.

    The ``|`` separator of EmbeddingGemma's prompt becomes ``/``, so the title
    cannot end the title slot early. Returns ``"none"`` when both parts are empty.
    """
    parts = []
    name = os.path.basename((project_root or "").rstrip("/"))
    if (project_root or "") not in _NO_PROJECT and name:
        parts.append(name)
    question = " ".join(_user_message(text).split()).replace("|", "/")[:TITLE_CHARS]
    if question:
        parts.append(question)
    return " · ".join(parts) or "none"


def build_passages(turn: Dict, offsets: Sequence[Tuple[int, int]]) -> List[Dict]:
    """Return the passage rows of ``turn`` from the token offsets of its text.

    Each passage copies the parent's metadata, takes ``PASSAGE_CHUNK_TYPE`` and
    the doc_id from ``passage_doc_id``, and carries the embedding ``title``.
    """
    text = turn["text"]
    spans = passage_spans(offsets)
    if not spans:
        return []
    title = passage_title(turn.get("project_root", ""), text)
    base = {k: v for k, v in turn.items() if k != REPLACE_SAME_TURN}
    return [
        {**base, "doc_id": passage_doc_id(turn["doc_id"], i), "text": text[start:end],
         "chunk_type": PASSAGE_CHUNK_TYPE, "title": title}
        for i, (start, end) in enumerate(spans)
    ]
