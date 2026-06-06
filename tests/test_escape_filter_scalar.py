"""SESF-33 — ``_escape_filter_scalar`` must harden Milvus filter literals.

Milvus boolean-expression string literals are C-style double-quoted strings
(per Plan.g4): an embedded double-quote is written as backslash-quote and a
literal backslash as a doubled backslash. Milvus does NOT honor ""-doubling, so
the helper that the engine relied on (doubling quotes) was both injectable via a
trailing backslash AND wrong for quote-bearing values.

The shared escaper must therefore double every backslash and escape every
double-quote as backslash-quote (not by doubling it), with backslashes escaped
FIRST so the backslash introduced when escaping a quote is not itself doubled. A
trailing backslash in any scalar (session_id, git_branch, project_root, date_*,
issue_id) must never escape the literal's closing quote and consume following
clauses.

These tests drive the private helper directly and assert the composed clauses
from ``_build_milvus_filter`` across every escaped scalar.
"""

from __future__ import annotations

import importlib

import pytest

rag_engine = importlib.import_module("rag_engine")

# Single literal characters, named so the escaped expectations stay readable.
BACKSLASH = "\\"  # one backslash character
QUOTE = '"'       # one double-quote character

esc = rag_engine._escape_filter_scalar


def _build(**overrides):
    """Call ``_build_milvus_filter`` with all-None defaults, overriding by keyword."""
    params = dict(
        session_id=None, git_branch=None, project_root=None,
        provider=None, source_kind=None, date_from=None, date_to=None,
        issue_id=None,
    )
    params.update(overrides)
    return rag_engine._build_milvus_filter(**params)


def _trailing_backslashes_before_closing_quote(expr: str) -> int:
    """Count backslashes immediately before the literal's closing quote.

    Every composed clause ends with the closing ``"``. An even count means those
    backslashes are all escaped (literal) and the quote truly terminates the
    string; an odd count means the final backslash escapes the quote — exactly
    the break-out condition this fix prevents.
    """
    assert expr.endswith(QUOTE)
    body = expr[:-1]
    return len(body) - len(body.rstrip(BACKSLASH))


# ---------------------------------------------------------------------------
# _escape_filter_scalar — character-level escaping
# ---------------------------------------------------------------------------

def test_backslash_is_doubled():
    # \ -> \\  so Milvus reads one literal backslash, not an escape lead-in.
    assert esc(BACKSLASH) == BACKSLASH * 2


def test_double_quote_is_c_style_escaped_not_doubled():
    # " -> \"  (C-style). The old "" doubling is a parse error in Milvus.
    assert esc("a" + QUOTE + "b") == "a" + BACKSLASH + QUOTE + "b"
    assert QUOTE + QUOTE not in esc("a" + QUOTE + "b")


def test_backslash_before_quote_is_fully_escaped():
    # The issue's vector: \" -> \\\"  (backslash doubled, then quote escaped).
    assert esc(BACKSLASH + QUOTE) == BACKSLASH * 3 + QUOTE


def test_backslash_escaped_before_quote_order():
    # Order guard: backslashes must be escaped BEFORE quotes. Quote-first would
    # double the backslash that " -> \" introduces, mangling the value.
    # a " \  ->  a \" \\
    assert esc("a" + QUOTE + BACKSLASH) == "a" + BACKSLASH + QUOTE + BACKSLASH * 2


def test_nul_byte_is_rejected():
    # Pre-existing guard must survive the rewrite.
    with pytest.raises(ValueError):
        esc("before\x00after")


def test_clean_values_are_unchanged():
    assert esc("SESF-25") == "SESF-25"
    assert esc("") == ""


def test_whitespace_is_preserved():
    # Unlike _issue_id_containment_token, the scalar escaper must not strip.
    assert esc("  spaced value  ") == "  spaced value  "


def test_newline_is_escaped():
    # A literal newline is not a valid Milvus string char (grammar); -> \n.
    assert esc("a\nb") == "a" + BACKSLASH + "nb"


def test_carriage_return_is_escaped():
    # A literal CR is likewise grammar-forbidden inside the literal; -> \r.
    assert esc("a\rb") == "a" + BACKSLASH + "rb"


def test_tab_is_escaped():
    # Tabs are legal literals but escaped for consistency; -> \t.
    assert esc("a\tb") == "a" + BACKSLASH + "tb"


def test_crlf_is_escaped_to_both_sequences():
    # CRLF -> \r\n; the two control chars escape independently.
    assert esc("a\r\nb") == "a" + BACKSLASH + "r" + BACKSLASH + "nb"


def test_literal_backslash_n_is_not_treated_as_newline():
    # Input is backslash + 'n' (two chars), not a newline: only the backslash is
    # doubled, the 'n' is untouched — proves control-char escaping runs after,
    # and only on, real control characters.
    assert esc("a" + BACKSLASH + "nb") == "a" + BACKSLASH * 2 + "nb"


# ---------------------------------------------------------------------------
# _build_milvus_filter — backslash-bearing values across every escaped scalar
# ---------------------------------------------------------------------------

def test_session_id_trailing_backslash_cannot_escape_closing_quote():
    expr = _build(session_id="x" + BACKSLASH)
    assert expr == 'session_id == "x' + BACKSLASH * 2 + '"'
    assert _trailing_backslashes_before_closing_quote(expr) % 2 == 0


def test_git_branch_trailing_backslash_is_doubled():
    expr = _build(git_branch="feature" + BACKSLASH)
    assert expr == 'git_branch == "feature' + BACKSLASH * 2 + '"'
    assert _trailing_backslashes_before_closing_quote(expr) % 2 == 0


def test_project_root_quote_and_backslash_escaped():
    # A path carrying both a quote and a trailing backslash.
    expr = _build(project_root="/a" + QUOTE + "b" + BACKSLASH)
    assert expr == 'project_root == "/a' + BACKSLASH + QUOTE + "b" + BACKSLASH * 2 + '"'
    assert _trailing_backslashes_before_closing_quote(expr) % 2 == 0


def test_project_root_newline_does_not_reach_filter_raw():
    # Reviewer example (PR #31): a newline in a path field must be escaped, not
    # interpolated raw, or Milvus rejects the literal (DoubleSChar excludes LF).
    expr = _build(project_root="/a\nb")
    assert expr == 'project_root == "/a' + BACKSLASH + 'nb"'
    assert "\n" not in expr


def test_date_from_trailing_backslash_is_doubled():
    expr = _build(date_from="2026-05-01" + BACKSLASH)
    assert expr == 'timestamp >= "2026-05-01' + BACKSLASH * 2 + '"'


def test_issue_id_token_backslash_is_doubled():
    # issue_id flows through _issue_id_containment_token (upper + wildcard-strip)
    # then _escape_filter_scalar; a backslash survives normalization and must be
    # doubled inside the LIKE literal.
    expr = _build(issue_id="sesf-25" + BACKSLASH)
    assert expr == 'issue_ids like "%,SESF-25' + BACKSLASH * 2 + ',%"'


def test_session_id_embedded_quote_uses_c_style_not_doubling():
    # Latent-bug guard: the composed clause must use \" not "" for an embedded
    # quote (the only form Milvus's grammar accepts).
    expr = _build(session_id="a" + QUOTE + "b")
    assert expr == 'session_id == "a' + BACKSLASH + QUOTE + 'b"'
    assert QUOTE + QUOTE not in expr
