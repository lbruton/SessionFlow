"""SESF-25 red-phase tests for the issue-ID extractor (`_extract_issue_ids`).

The extractor is a pure function over a turn's text. Per design.md Component 1:

    _extract_issue_ids(text: str) -> str

It applies the regex ``\\b[A-Z][A-Z0-9]+-\\d+\\b`` over UPPERCASED text, drops a
denylist of technical-standard prefixes (UTF, SHA, HTTP, HTTPS, ISO, RFC, IPV,
MD, BASE), deduplicates in first-seen order, and returns a DELIMITER-WRAPPED,
comma-joined, uppercased string (e.g. ``",SESF-25,SESF-26,"``) or ``""`` when no
token is found. Output is capped to the field length (4096 chars).

Covers Requirements 1.2 (dedup), 1.3 (empty set), 1.4 (case-canonicalization),
6.3 (denylist + token boundary), and the length cap.
"""

from __future__ import annotations

import importlib

rag_engine = importlib.import_module("rag_engine")


def test_single_id_is_delimiter_wrapped():
    # Baseline contract: one id → leading + trailing comma wrap.
    assert rag_engine._extract_issue_ids("Fixing SESF-25 today") == ",SESF-25,"


def test_dedup_same_id_recorded_once_per_turn():
    # Req 1.2 — repeated id collapses to a single entry.
    text = "SESF-25 then more on SESF-25 and again SESF-25"
    assert rag_engine._extract_issue_ids(text) == ",SESF-25,"


def test_multiple_ids_dedup_first_seen_order():
    # Req 1.2 — distinct ids preserved in first-seen order, each once.
    text = "SESF-26 references SESF-25 which supersedes SESF-26"
    assert rag_engine._extract_issue_ids(text) == ",SESF-26,SESF-25,"


def test_empty_set_when_no_token():
    # Req 1.3 — no issue token → empty string (and ingestion proceeds).
    assert rag_engine._extract_issue_ids("just some prose with no ids") == ""


def test_empty_string_input():
    # Req 1.3 — empty input → empty output.
    assert rag_engine._extract_issue_ids("") == ""


def test_case_canonicalization_lowercase_in_prose():
    # Req 1.4 — a lowercased id in prose still matches and is stored uppercased.
    assert rag_engine._extract_issue_ids("see sesf-25 for context") == ",SESF-25,"


def test_case_canonicalization_mixed_case_dedups_with_upper():
    # Req 1.4 — mixed-case + uppercase forms canonicalize to ONE stored id.
    assert rag_engine._extract_issue_ids("Sesf-25 and SESF-25") == ",SESF-25,"


def test_denylist_rejects_utf8():
    # Req 6.3 — UTF-8 must not be treated as an issue ref.
    assert rag_engine._extract_issue_ids("encoded as UTF-8 text") == ""


def test_denylist_rejects_http2():
    # Req 6.3 — HTTP-2 must not be treated as an issue ref.
    assert rag_engine._extract_issue_ids("served over HTTP-2 here") == ""


def test_denylist_rejects_sha256():
    # Req 6.3 — SHA-256 must not be treated as an issue ref.
    assert rag_engine._extract_issue_ids("hashed with SHA-256 digest") == ""


def test_denylist_does_not_swallow_real_id_alongside():
    # Req 6.3 — denylisted tokens are dropped but real ids survive.
    text = "UTF-8 payload referenced in SESF-25"
    assert rag_engine._extract_issue_ids(text) == ",SESF-25,"


def test_token_boundary_sesf42_not_sesf420():
    # Req 6.3 / boundary — SESF-42 must not match the longer SESF-420 token.
    assert rag_engine._extract_issue_ids("work on SESF-420 now") == ",SESF-420,"
    assert rag_engine._extract_issue_ids("work on SESF-42 now") == ",SESF-42,"


def test_token_boundary_distinguishes_42_from_420():
    # Both appear; each is a distinct, separately-bounded token.
    out = rag_engine._extract_issue_ids("SESF-42 and SESF-420")
    assert out == ",SESF-42,SESF-420,"


def test_length_cap_does_not_exceed_field_max():
    # Output is capped to the Milvus field length (4096) so an insert can't fail.
    # Build far more ids than fit; result must stay within the cap and remain
    # well-formed (delimiter-wrapped).
    text = " ".join(f"SESF-{i}" for i in range(2000))
    out = rag_engine._extract_issue_ids(text)
    assert len(out) <= 4096
    if out:
        assert out.startswith(",")
        assert out.endswith(",")
