"""Resolve a search's project argument to the indexed ``project_root`` values it covers."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Worktree folders inside a repository: ours, and the Claude desktop app's.
_WORKTREE_MARKERS = ("/.worktrees/", "/.claude/worktrees/")
_LABEL_ROOTS = 3


@dataclass(frozen=True)
class ProjectScope:
    """The project roots a search covers.

    Attributes:
        roots: Sorted ``project_root`` values to match, or None for every project.
        note: Why the scope differs from the argument, or an empty string.
    """

    roots: Optional[Tuple[str, ...]]
    note: str = ""

    def filter_value(self) -> Optional[List[str]]:
        """Return the roots as a list for the engine filters, or None for every project."""
        return list(self.roots) if self.roots else None

    def label(self) -> str:
        """Return the roots for display, truncated after three."""
        if not self.roots:
            return "all projects"
        shown = ", ".join(self.roots[:_LABEL_ROOTS])
        extra = len(self.roots) - _LABEL_ROOTS
        return f"{shown} (+{extra} more)" if extra > 0 else shown


def _with_subfolders(base: str, known: List[str]) -> List[str]:
    if base == "/":
        return [r for r in known if r == "/"]
    return [r for r in known if r == base or r.startswith(base + "/")]


def _nearest_parent(path: str, known: List[str]) -> Optional[str]:
    parents = [r for r in known if r != "/" and path.startswith(r + "/")]
    return max(parents, key=len) if parents else None


def _basename(root: str) -> str:
    return root.rstrip("/").rsplit("/", 1)[-1]


def resolve_project_scope(arg: Optional[str], header_root: Optional[str],
                          known_roots: Iterable[str]) -> ProjectScope:
    """Resolve a project argument against the indexed roots.

    ``'*'`` covers every project. An empty argument falls back to ``header_root``.
    A path covers its indexed root and subfolders; a path inside a worktree folder
    resolves to its repository first, and an unindexed path to its nearest indexed
    parent. A bare name matches root basenames case-insensitively. A miss covers
    every project and says why in ``note``.

    Args:
        arg: The caller's ``project_root`` argument.
        header_root: The current project from the request header.
        known_roots: Every ``project_root`` value in the index.

    Returns:
        The resolved scope.
    """
    if arg == "*":
        return ProjectScope(roots=None)
    target = (arg or "").strip() or (header_root or "").strip()
    if not target:
        return ProjectScope(roots=None, note="no current project")
    known = sorted(set(known_roots))

    if target.startswith("/"):
        # normpath keeps exactly two leading slashes, so collapse them.
        path = "/" + posixpath.normpath(target).lstrip("/")
        for marker in _WORKTREE_MARKERS:
            path = path.split(marker, 1)[0]
        matched = _with_subfolders(path, known)
        if matched:
            return ProjectScope(roots=tuple(matched))
        parent = _nearest_parent(path, known)
        if parent:
            return ProjectScope(roots=(parent,), note=f"nearest indexed parent of {target}")
        return ProjectScope(roots=None, note=f"no indexed turns under {target}")

    name = target.lower()
    matched = sorted({sub for r in known if _basename(r).lower() == name
                      for sub in _with_subfolders(r, known)})
    if matched:
        return ProjectScope(roots=tuple(matched))
    return ProjectScope(roots=None, note=f'no project named "{target}"')
