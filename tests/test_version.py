"""The package version lives in version.py and matches CHANGELOG.md, /health and serverInfo."""

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
HEADING = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)


def _headings():
    return HEADING.findall((REPO / "CHANGELOG.md").read_text(encoding="utf-8"))


def test_version_is_semver():
    import version

    assert SEMVER.match(version.__version__)


def test_changelog_opens_with_unreleased_then_the_current_version():
    import version

    headings = _headings()
    assert headings[0] == "Unreleased"
    assert headings[1] == version.__version__


def test_changelog_versions_are_semver_and_descending():
    versions = _headings()[1:]
    assert versions, "CHANGELOG.md has no released version"
    for v in versions:
        assert SEMVER.match(v), v
    keys = [tuple(int(p) for p in v.split(".")) for v in versions]
    assert keys == sorted(keys, reverse=True)
    assert len(set(keys)) == len(keys)


def test_released_headings_carry_an_iso_date():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## \[(\d[^\]]*)\](.*)$", text, re.MULTILINE)
    for v, rest in released:
        assert re.fullmatch(r" - \d{4}-\d{2}-\d{2}", rest), f"{v}: {rest!r}"


def test_mcp_server_info_reports_the_version(stub_rag_engine, monkeypatch):
    import version

    monkeypatch.delitem(sys.modules, "http_server", raising=False)
    http_server = importlib.import_module("http_server")

    options = http_server.mcp_server.create_initialization_options()
    assert options.server_version == version.__version__


@pytest.mark.anyio
async def test_health_reports_the_version(stub_rag_engine, monkeypatch):
    import version

    monkeypatch.delitem(sys.modules, "http_server", raising=False)
    http_server = importlib.import_module("http_server")

    response = await http_server.health(None)

    assert json.loads(response.body)["version"] == version.__version__
