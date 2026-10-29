"""Tests for the Claude Code ``headersHelper`` script (``mcp-headers-helper.sh``).

Claude Code runs the helper with its working directory set to ``~/.claude``,
so the helper takes the project from ``CLAUDE_PROJECT_DIR`` or from the
working directory of the process that spawned it.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "mcp-headers-helper.sh"


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path.resolve()


def _run(parent_cwd: Path, helper_cwd: Path, **extra_env) -> dict:
    """Run the helper from ``helper_cwd`` as the child of a shell in ``parent_cwd``.

    The subshell changes directory and execs the helper, so the helper's parent
    is the outer shell, whose working directory stays ``parent_cwd``.
    """
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": os.environ["HOME"], **extra_env}
    script = f'(cd "{helper_cwd}" && exec /bin/bash "{HELPER}"); :'
    out = subprocess.run(
        ["/bin/bash", "-c", script],
        cwd=parent_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(out.stdout)


@pytest.fixture
def claude_home(tmp_path):
    """A stand-in for ``~/.claude``: not a git repository."""
    path = tmp_path / "dot-claude"
    path.mkdir()
    return path.resolve()


def test_project_dir_env_wins(tmp_path, claude_home):
    repo = _git_repo(tmp_path / "repo")
    other = _git_repo(tmp_path / "other")
    headers = _run(other, claude_home, CLAUDE_PROJECT_DIR=str(repo))
    assert headers == {"X-Project-Root": str(repo)}


def test_project_dir_env_subfolder_resolves_to_repo_root(tmp_path, claude_home):
    repo = _git_repo(tmp_path / "repo")
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    headers = _run(claude_home, claude_home, CLAUDE_PROJECT_DIR=str(sub))
    assert headers == {"X-Project-Root": str(repo)}


def test_parent_cwd_used_when_helper_runs_from_claude_home(tmp_path, claude_home):
    repo = _git_repo(tmp_path / "repo")
    headers = _run(repo, claude_home)
    assert headers == {"X-Project-Root": str(repo)}


def test_parent_cwd_outside_git_is_sent_as_is(tmp_path, claude_home):
    folder = tmp_path / "plain"
    folder.mkdir()
    headers = _run(folder.resolve(), claude_home)
    assert headers == {"X-Project-Root": str(folder.resolve())}


def test_root_folder_sends_no_header(claude_home):
    assert _run(Path("/"), Path("/")) == {}


def test_claude_home_is_never_sent(claude_home):
    headers = _run(claude_home, claude_home, SESSIONFLOW_CLAUDE_HOME=str(claude_home))
    assert headers == {}


def test_path_is_json_escaped(tmp_path, claude_home):
    folder = tmp_path / 'quo"te\\dir'
    folder.mkdir()
    headers = _run(folder.resolve(), claude_home)
    assert headers == {"X-Project-Root": str(folder.resolve())}


def test_setup_installs_the_repo_helper():
    setup = (REPO / "setup.sh").read_text()
    assert "mcp-headers-helper.sh" in setup
    assert "git rev-parse --show-toplevel 2>/dev/null || pwd" not in setup
