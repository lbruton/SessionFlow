"""SESF-64 red-phase tests: project scope resolution.

Project filters matched ``project_root`` exactly, so a short name such as
``CoWork``, a subfolder root or a worktree path matched nothing. The scope
argument now resolves against the indexed roots: a path keeps its subfolders,
a worktree resolves to its repository, an unknown path falls back to the
nearest indexed parent, and a bare name matches root basenames. A miss widens
to every project, and every result set opens with a ``Scope:`` line.
"""

from __future__ import annotations

import asyncio
import importlib
from contextlib import contextmanager

import pytest

from fts_hybrid import FTSIndex
from project_scope import ProjectScope, resolve_project_scope

rag_engine = importlib.import_module("rag_engine")
tools = importlib.import_module("tools")

KNOWN = [
    "/",
    "/Users/lbruton/Claude/Projects/CoWork",
    "/Users/lbruton/Claude/Projects/CoWork/DocVault-CoWork",
    "/Users/lbruton/Claude/Projects/Finances",
    "/Volumes/DATA/GitHub/SessionFlow",
    "/Volumes/DATA/GitHub/SessionFlow/.worktrees/fix-sesf-64",
    "/Volumes/DATA/GitHub/StakTrakr",
    "/Volumes/DATA/GitHub/Archive/StakTrakr",
    "unknown",
]


def _resolve(arg, header=None):
    return resolve_project_scope(arg, header, KNOWN)


# ---------------------------------------------------------------------------
# resolve_project_scope
# ---------------------------------------------------------------------------

def test_star_is_all_projects():
    scope = _resolve("*", header="/Volumes/DATA/GitHub/SessionFlow")
    assert scope.roots is None
    assert scope.label() == "all projects"


def test_missing_arg_uses_header_root():
    scope = _resolve(None, header="/Users/lbruton/Claude/Projects/Finances")
    assert scope.roots == ("/Users/lbruton/Claude/Projects/Finances",)


def test_no_arg_and_no_header_is_all_projects_with_note():
    scope = _resolve(None, header=None)
    assert scope.roots is None
    assert scope.note


def test_known_path_includes_subfolders():
    scope = _resolve("/Users/lbruton/Claude/Projects/CoWork")
    assert scope.roots == (
        "/Users/lbruton/Claude/Projects/CoWork",
        "/Users/lbruton/Claude/Projects/CoWork/DocVault-CoWork",
    )


def test_trailing_slash_is_ignored():
    scope = _resolve("/Users/lbruton/Claude/Projects/Finances/")
    assert scope.roots == ("/Users/lbruton/Claude/Projects/Finances",)


def test_parent_segment_is_collapsed():
    scope = _resolve("/Users/lbruton/Claude/Projects/CoWork/../Finances")
    assert scope.roots == ("/Users/lbruton/Claude/Projects/Finances",)


def test_dot_and_doubled_slashes_resolve_like_the_clean_path():
    clean = _resolve("/Users/lbruton/Claude/Projects/Finances")
    assert _resolve("/Users/lbruton/Claude/./Projects//Finances/.").roots == clean.roots


def test_leading_double_slash_resolves_like_a_single_slash():
    scope = _resolve("//Users/lbruton/Claude/Projects/Finances")
    assert scope.roots == ("/Users/lbruton/Claude/Projects/Finances",)


def test_parent_segment_out_of_a_worktree_keeps_no_marker():
    scope = _resolve("/Volumes/DATA/GitHub/SessionFlow/.worktrees/x/../../..")
    assert "/Volumes/DATA/GitHub/StakTrakr" in scope.roots
    assert "/" not in scope.roots


def test_worktree_path_resolves_to_repository():
    scope = _resolve(None, header="/Volumes/DATA/GitHub/SessionFlow/.worktrees/fix-sesf-64")
    assert scope.roots == (
        "/Volumes/DATA/GitHub/SessionFlow",
        "/Volumes/DATA/GitHub/SessionFlow/.worktrees/fix-sesf-64",
    )


def test_desktop_worktree_path_resolves_to_repository():
    scope = _resolve(None, header="/Volumes/DATA/GitHub/SessionFlow/.claude/worktrees/nice-curie")
    assert scope.roots[0] == "/Volumes/DATA/GitHub/SessionFlow"


def test_unknown_path_with_indexed_subfolders_matches_them():
    scope = _resolve("/Users/lbruton/Claude/Projects")
    assert scope.roots == (
        "/Users/lbruton/Claude/Projects/CoWork",
        "/Users/lbruton/Claude/Projects/CoWork/DocVault-CoWork",
        "/Users/lbruton/Claude/Projects/Finances",
    )


def test_unknown_path_falls_back_to_nearest_indexed_parent():
    scope = _resolve("/Users/lbruton/Claude/Projects/CoWork/notes/2026")
    assert scope.roots == ("/Users/lbruton/Claude/Projects/CoWork",)
    assert "parent" in scope.note


def test_slash_root_is_not_a_parent_of_everything():
    scope = _resolve("/Volumes/Other/Project")
    assert scope.roots is None
    assert "/Volumes/Other/Project" in scope.note


def test_slash_root_matches_only_itself():
    assert _resolve("/").roots == ("/",)


def test_bare_name_matches_basename_case_insensitively():
    scope = _resolve("cowork")
    assert scope.roots == (
        "/Users/lbruton/Claude/Projects/CoWork",
        "/Users/lbruton/Claude/Projects/CoWork/DocVault-CoWork",
    )


def test_ambiguous_name_matches_every_root_with_that_basename():
    scope = _resolve("StakTrakr")
    assert scope.roots == (
        "/Volumes/DATA/GitHub/Archive/StakTrakr",
        "/Volumes/DATA/GitHub/StakTrakr",
    )


def test_unknown_name_falls_back_to_all_projects_with_note():
    scope = _resolve("NoSuchProject")
    assert scope.roots is None
    assert "NoSuchProject" in scope.note


def test_name_unknown_matches_the_unknown_bucket():
    assert _resolve("unknown").roots == ("unknown",)


def test_label_lists_up_to_three_roots():
    scope = ProjectScope(roots=("/a", "/a/b"))
    assert scope.label() == "/a, /a/b"


def test_label_truncates_long_root_lists():
    scope = ProjectScope(roots=("/a", "/a/b", "/a/c", "/a/d", "/a/e"))
    assert scope.label() == "/a, /a/b, /a/c (+2 more)"


# ---------------------------------------------------------------------------
# Milvus filter
# ---------------------------------------------------------------------------

def test_milvus_filter_single_root_is_equality():
    expr = rag_engine._build_milvus_filter(None, None, ["/a"], None, None, None, None)
    assert expr == 'project_root == "/a"'


def test_milvus_filter_several_roots_uses_in_and_escapes():
    expr = rag_engine._build_milvus_filter(
        None, None, ["/a", '/b"q'], None, None, None, None)
    assert expr == 'project_root in ["/a", "/b\\"q"]'


def test_milvus_filter_string_root_is_unchanged():
    expr = rag_engine._build_milvus_filter(None, None, "/a", None, None, None, None)
    assert expr == 'project_root == "/a"'


# ---------------------------------------------------------------------------
# FTS filter and distinct roots
# ---------------------------------------------------------------------------

def _fts_with_rows(tmp_path):
    fts = FTSIndex("turns_fts", ["session_id", "project_root"])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    try:
        fts.insert(conn, [
            {"doc_id": "d1", "content": "deploy notes", "project_root": "/a"},
            {"doc_id": "d2", "content": "deploy notes", "project_root": "/a/b"},
            {"doc_id": "d3", "content": "deploy notes", "project_root": '/c"q'},
        ])
    finally:
        fts.close_ephemeral(conn)
    return fts, db_path


def test_fts_in_filter_matches_any_listed_root(tmp_path):
    fts, db_path = _fts_with_rows(tmp_path)
    rows = fts.search("deploy", n=10, filters={"project_root": ("in", ["/a/b", '/c"q'])},
                      db_path=db_path)
    assert sorted(r["doc_id"] for r in rows) == ["d2", "d3"]


def test_fts_count_rows_accepts_several_roots(tmp_path):
    fts, db_path = _fts_with_rows(tmp_path)
    conn = fts.connection(db_path)
    try:
        assert fts.count_rows(conn, project_root=["/a", "/a/b"]) == 2
        assert fts.count_rows(conn, project_root="/a") == 1
    finally:
        fts.close_ephemeral(conn)


def test_fts_distinct_values(tmp_path):
    fts, db_path = _fts_with_rows(tmp_path)
    conn = fts.connection(db_path)
    try:
        assert sorted(fts.distinct_values(conn, "project_root")) == ["/a", "/a/b", '/c"q']
    finally:
        fts.close_ephemeral(conn)


def test_known_project_roots_reads_fts_and_caches(tmp_path, monkeypatch):
    fts, db_path = _fts_with_rows(tmp_path)
    monkeypatch.setattr(rag_engine, "_fts", fts)
    rag_engine._known_roots_cache.clear()
    assert sorted(rag_engine.known_project_roots(db_path)) == ["/a", "/a/b", '/c"q']

    def _boom(*a, **kw):
        raise AssertionError("cache miss")

    monkeypatch.setattr(fts, "distinct_values", _boom)
    assert sorted(rag_engine.known_project_roots(db_path)) == ["/a", "/a/b", '/c"q']
    rag_engine._known_roots_cache.clear()


# ---------------------------------------------------------------------------
# search sends the root list to both engines
# ---------------------------------------------------------------------------

def _patch_search(monkeypatch) -> dict:
    seen: dict = {}

    class _Client:
        def search(self, *a, **kw):
            seen["milvus_filter"] = kw.get("filter")
            return [[]]

    @contextmanager
    def _fake_client(db_path=None):
        yield _Client()

    def _fts_search(query, n=15, filters=None, db_path=None):
        seen["fts_filters"] = filters
        return []

    monkeypatch.setattr(rag_engine, "embed_texts", lambda *a, **kw: [[0.0] * 768])
    monkeypatch.setattr(rag_engine, "milvus_client", _fake_client)
    monkeypatch.setattr(rag_engine._fts, "search", _fts_search)
    return seen


def test_search_passes_root_list_to_both_engines(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("deploy notes", n=5, project_root=["/a", "/a/b"])
    assert seen["milvus_filter"] == 'project_root in ["/a", "/a/b"]'
    assert seen["fts_filters"] == {"project_root": ("in", ["/a", "/a/b"])}


def test_search_single_string_root_keeps_equality(monkeypatch):
    seen = _patch_search(monkeypatch)
    rag_engine.search("deploy notes", n=5, project_root="/a")
    assert seen["fts_filters"] == {"project_root": "/a"}


# ---------------------------------------------------------------------------
# tools: scope resolution and the Scope line
# ---------------------------------------------------------------------------

def test_resolve_scope_uses_known_roots(monkeypatch):
    monkeypatch.setattr(rag_engine, "known_project_roots", lambda db_path=None: KNOWN)
    scope = asyncio.run(tools.resolve_scope("cowork", "/Volumes/DATA/GitHub/SessionFlow", "db"))
    assert scope.roots[0] == "/Users/lbruton/Claude/Projects/CoWork"


def test_resolve_scope_star_skips_the_root_lookup(monkeypatch):
    def _boom(db_path=None):
        raise AssertionError("lookup not needed")

    monkeypatch.setattr(rag_engine, "known_project_roots", _boom)
    scope = asyncio.run(tools.resolve_scope("*", "/x", "db"))
    assert scope.roots is None


def test_format_results_opens_with_scope_line():
    row = {"content": "body", "session_id": "s", "turn_index": 1, "chunk_type": "turn"}
    scope = ProjectScope(roots=None, note='no project named "Nope"')
    rendered = tools.format_results([row], scope=scope)
    assert rendered.startswith('Scope: all projects (no project named "Nope")\n')


def test_format_results_empty_still_shows_scope():
    rendered = tools.format_results([], scope=ProjectScope(roots=("/a",)))
    assert rendered == "Scope: /a\n\nNo results found."


def test_format_results_without_scope_is_unchanged():
    assert tools.format_results([]) == "No results found."


@pytest.mark.parametrize("roots, expected", [
    (None, None),
    (("/a",), ["/a"]),
    (("/a", "/a/b"), ["/a", "/a/b"]),
])
def test_scope_filter_value(roots, expected):
    assert ProjectScope(roots=roots).filter_value() == expected
