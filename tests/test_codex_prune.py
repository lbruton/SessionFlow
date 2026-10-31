"""prune-codex runs dedupe-codex, so it deletes only rows a new row covers.

* Dry run by default; ``--apply`` refuses without ``--yes`` before any read.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def no_live_index(monkeypatch):
    """Fail any write that would reach the real index."""
    import rag_engine

    def _refuse(*args, **kwargs):
        raise AssertionError("test reached the live index")

    for name in ("delete_by_doc_id", "add_turns", "stored_metadata_by_provider", "_query_batches"):
        monkeypatch.setattr(rag_engine, name, _refuse)


@pytest.fixture
def cleanup_mod():
    return importlib.import_module("cleanup")


def _run(cleanup_mod, argv):
    return cleanup_mod.cmd_prune_codex(cleanup_mod.build_parser().parse_args(argv))


def test_prune_codex_runs_dedupe_codex(cleanup_mod, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cleanup_mod, "cmd_dedupe_codex", lambda args: seen.append(args) or 7)

    assert _run(cleanup_mod, ["prune-codex", "--apply", "--yes"]) == 7
    assert seen and seen[0].apply and seen[0].yes
    assert "dedupe-codex" in capsys.readouterr().err


def test_prune_codex_apply_without_yes_refuses(cleanup_mod, monkeypatch, capsys):
    import dedupe_codex

    monkeypatch.setattr(dedupe_codex, "plan", lambda **k: pytest.fail("read before the gate"))

    assert _run(cleanup_mod, ["prune-codex", "--apply"]) == 1
    assert "--yes" in capsys.readouterr().err
