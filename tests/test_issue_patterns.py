"""Tests for configurable issue-ID patterns (``issue_patterns``).

Covers the config file format, the ServiceNow preset, ``regex:`` and ``deny:``
lines, the unchanged default, and how extraction and token validation in
``rag_engine`` and ``provider_adapters`` follow the active patterns.
"""

from __future__ import annotations

import importlib
import logging
import os

import pytest

issue_patterns = importlib.import_module("issue_patterns")
rag_engine = importlib.import_module("rag_engine")
provider_adapters = importlib.import_module("provider_adapters")


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "issue_ids.conf"
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def use_config(tmp_path, monkeypatch):
    """Point the active patterns at a config file holding ``text``."""

    def _use(text: str) -> str:
        path = _write(tmp_path, text)
        monkeypatch.setenv(issue_patterns.ENV_VAR, path)
        issue_patterns.reset()
        return path

    return _use


# --- default ---------------------------------------------------------------------


def test_no_config_file_keeps_dashed_extraction():
    patterns = issue_patterns.active()
    assert patterns.source is None
    assert patterns.extract("Fixing SESF-25 and ops-50, not UTF-8") == ["SESF-25", "OPS-50"]
    assert rag_engine._extract_issue_ids("INC0012345 and SESF-25") == ",SESF-25,"


def test_default_does_not_read_a_real_home_file():
    # The autouse fixture points the default path at a missing file.
    assert not os.path.exists(issue_patterns.DEFAULT_PATH)


def test_default_validation_is_unchanged():
    assert provider_adapters.is_valid_issue_token(" sesf-25 ")
    assert not provider_adapters.is_valid_issue_token("INC0012345")
    assert not provider_adapters.is_valid_issue_token("SESF-25%")
    assert not provider_adapters.is_valid_issue_token(None)


def test_missing_configured_file_warns_and_uses_default(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(issue_patterns.ENV_VAR, str(tmp_path / "absent.conf"))
    issue_patterns.reset()
    with caplog.at_level(logging.WARNING):
        patterns = issue_patterns.active()
    assert patterns.extract("SESF-25 INC0012345") == ["SESF-25"]
    assert "absent.conf" in caplog.text


# --- presets ---------------------------------------------------------------------


def test_servicenow_preset_extracts_tokens(use_config):
    use_config("preset:servicenow\n")
    text = "See inc0012345, RITM0012345 and SCTASK0000001 for SESF-25"
    assert rag_engine._extract_issue_ids(text) == (
        ",INC0012345,RITM0012345,SCTASK0000001,SESF-25,")


@pytest.mark.parametrize("text", [
    "INC001234",       # 6 digits
    "INC00123456",     # 8 digits
    "XINC0012345",     # prefix inside a longer word
    "INC0012345X",     # trailing letter
    "TASKS0012345",    # unknown prefix
])
def test_servicenow_near_misses_are_ignored(use_config, text):
    use_config("preset:servicenow\n")
    assert rag_engine._extract_issue_ids(text) == ""


def test_extraction_keeps_text_order_across_patterns(use_config):
    use_config("preset:servicenow\n")
    text = "SESF-1 then INC0000001 then SESF-2 then INC0000001 again"
    assert issue_patterns.active().extract(text) == ["SESF-1", "INC0000001", "SESF-2"]


def test_dashed_stays_on_when_the_file_names_other_patterns(use_config):
    use_config("preset:servicenow\n")
    assert issue_patterns.active().extract("OPS-50 DRG-12 PATCH-3") == [
        "OPS-50", "DRG-12", "PATCH-3"]


def test_servicenow_tokens_validate(use_config):
    use_config("preset:servicenow\n")
    assert provider_adapters.is_valid_issue_token("inc0012345")
    assert provider_adapters.is_valid_issue_token("SESF-25")
    assert not provider_adapters.is_valid_issue_token("INC001234")


# --- regex: and deny: --------------------------------------------------------------


def test_regex_line_applies(use_config):
    use_config("regex:\\bCASE\\d{5}\\b\n")
    assert issue_patterns.active().extract("case12345 and CASE1234") == ["CASE12345"]
    assert provider_adapters.is_valid_issue_token("CASE12345")


def test_deny_line_drops_a_prefix(use_config):
    use_config("deny:ops\n")
    assert issue_patterns.active().extract("OPS-50 SESF-25 UTF-8") == ["SESF-25"]
    assert not provider_adapters.is_valid_issue_token("OPS-50")


def test_deny_applies_to_undashed_tokens(use_config):
    use_config("preset:servicenow\ndeny:TASK\n")
    assert issue_patterns.active().extract("TASK0000001 INC0000001") == ["INC0000001"]


def test_comments_and_blank_lines_are_ignored(use_config):
    use_config("# ticket systems\n\n   \npreset:servicenow   # ServiceNow\n")
    patterns = issue_patterns.active()
    assert patterns.skipped == 0
    assert patterns.extract("INC0000001") == ["INC0000001"]


def test_bad_lines_warn_with_line_number_and_are_skipped(use_config, caplog):
    use_config("regex:(unclosed\npreset:nope\nbogus line\ndeny:bad prefix\npreset:servicenow\n")
    with caplog.at_level(logging.WARNING):
        patterns = issue_patterns.active()
    assert patterns.skipped == 4
    for line_no in (1, 2, 3, 4):
        assert f"line {line_no}" in caplog.text
    assert patterns.extract("INC0000001 SESF-1") == ["INC0000001", "SESF-1"]


def test_tokens_with_unsafe_characters_are_dropped(use_config):
    # A comma would break the comma-wrapped field; % and _ are LIKE wildcards.
    use_config("regex:\\bT[,%_]\\d+\\b\nregex:\\bOK\\d+\\b\n")
    patterns = issue_patterns.active()
    assert patterns.extract("T,1 T%2 T_3 OK4") == ["OK4"]
    assert not patterns.matches("T_3")


def test_preset_dashed_line_is_accepted(use_config):
    use_config("preset:dashed\n")
    patterns = issue_patterns.active()
    assert patterns.skipped == 0
    assert patterns.extract("SESF-1") == ["SESF-1"]


# --- reload and reporting --------------------------------------------------------


def test_edited_file_is_reloaded(use_config):
    path = use_config("preset:servicenow\n")
    assert issue_patterns.active().extract("INC0000001") == ["INC0000001"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("deny:SESF\n")
    stat = os.stat(path)
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))
    patterns = issue_patterns.active()
    assert patterns.extract("INC0000001 SESF-1 OPS-2") == ["OPS-2"]


def test_describe_names_source_patterns_and_denylist(use_config):
    path = use_config("preset:servicenow\nregex:\\bCASE\\d{5}\\b\ndeny:OPS\nregex:(bad\n")
    info = issue_patterns.active().describe()
    assert info["source"] == path
    assert info["patterns"] == ["preset:dashed", "preset:servicenow", "regex:\\bCASE\\d{5}\\b"]
    assert info["deny"] == ["OPS"]
    assert info["skipped"] == 1


def test_describe_default():
    info = issue_patterns.active().describe()
    assert info == {"source": None, "patterns": ["preset:dashed"], "deny": [], "skipped": 0,
                    "error": None}


def test_unreadable_configured_file_keeps_its_path_in_describe(tmp_path, monkeypatch):
    missing = str(tmp_path / "absent.conf")
    monkeypatch.setenv(issue_patterns.ENV_VAR, missing)
    issue_patterns.reset()
    info = issue_patterns.active().describe()
    assert info["source"] == missing
    assert info["patterns"] == ["preset:dashed"]
    assert info["error"]
    text = "\n".join(issue_patterns.format_lines(info))
    assert missing in text
    assert "not read" in text
    assert "built-in default" not in text


def test_config_that_is_not_utf8_falls_back_to_dashed(tmp_path, monkeypatch, caplog):
    path = tmp_path / "issue_ids.conf"
    path.write_bytes(b"preset:servicenow\n\xff\xfe bad bytes\n")
    monkeypatch.setenv(issue_patterns.ENV_VAR, str(path))
    issue_patterns.reset()
    with caplog.at_level(logging.WARNING):
        patterns = issue_patterns.active()
    assert patterns.extract("SESF-1 INC0000001") == ["SESF-1"]
    info = patterns.describe()
    assert info["source"] == str(path)
    assert info["error"]
    assert str(path) in caplog.text


@pytest.mark.parametrize("pattern", [
    "(?<=#)\\d+\\b", "(?<!X)CASE\\d+", "CASE\\d+(?=\\s)", "CASE\\d+(?!X)",
])
def test_lookaround_patterns_are_skipped_with_a_warning(use_config, caplog, pattern):
    use_config(f"regex:{pattern}\n")
    with caplog.at_level(logging.WARNING):
        patterns = issue_patterns.active()
    assert patterns.skipped == 1
    assert "line 1" in caplog.text
    assert patterns.describe()["patterns"] == ["preset:dashed"]


def test_every_extracted_token_passes_validation(use_config):
    # \B depends on the character before the match; a token that only matched
    # in context would be stored but refused by every issue_id filter.
    use_config("regex:\\BCASE\\d+\\b\nregex:\\bOK\\d+\\b\n")
    patterns = issue_patterns.active()
    tokens = patterns.extract("XCASE12 OK3")
    assert tokens == ["OK3"]
    assert all(patterns.matches(token) for token in tokens)


def test_get_stats_reports_issue_patterns_without_a_collection(monkeypatch):
    import contextlib

    class Client:
        def has_collection(self, name):
            return False

    @contextlib.contextmanager
    def fake_client(db_path=None):
        yield Client()

    monkeypatch.setattr(rag_engine, "milvus_client", fake_client)
    stats = rag_engine.get_stats()
    assert stats["issue_patterns"]["patterns"] == ["preset:dashed"]


def test_format_stats_prints_issue_patterns():
    tools = importlib.import_module("tools")
    text = tools.format_stats({
        "total_turns": 1, "sessions": 1,
        "issue_patterns": {"source": "/x/issue_ids.conf",
                           "patterns": ["preset:dashed", "preset:servicenow"],
                           "deny": ["OPS"], "skipped": 1},
    }, "/db")
    assert "preset:dashed, preset:servicenow" in text
    assert "/x/issue_ids.conf" in text
    assert "OPS" in text
    assert "1 line skipped" in text


def test_example_file_parses_cleanly():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = issue_patterns.load(os.path.join(root, "issue_ids.conf.example"))
    assert patterns.skipped == 0
    assert "preset:servicenow" in patterns.describe()["patterns"]
