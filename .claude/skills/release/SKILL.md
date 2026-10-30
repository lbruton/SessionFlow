---
name: release
description: >
  SessionFlow version bump for a feat or fix PR: pick patch, minor or major from the bump rules, claim the
  version in devops/version.lock, edit version.py, move the CHANGELOG.md Unreleased entries under a dated
  heading, verify and commit on the PR's own branch. No git tags, no GitHub Releases.
  Triggers: "/release", "bump the version", "release patch", "release minor".
---

# Release (SessionFlow)

Project-owned skill. `.claude/project.json` excludes `release` from skills-deploy, so this file is the
master. Keep `.agents/skills/release/SKILL.md` identical; `skills-deploy --check` reports drift between them.

The version lives in `version.py` (`__version__`) and in the newest released heading of `CHANGELOG.md`.
`/health` and the MCP `serverInfo` read `version.py`. `tests/test_version.py` checks that the two agree.

## When to run

- **Every `feat` or `fix` PR** runs `/release` once, on its own branch, after the code is done and before
  the PR is marked ready. The `Version` workflow fails a `feat`/`fix` PR that does not raise `version.py`
  and change `CHANGELOG.md`.
- **No bump** for `chore`, `docs`, `ci`, `test` and `refactor` PRs, instruction files and skills deploys.
- There is no separate release branch. The bump commit rides the feature PR.

## Arguments

`$ARGUMENTS` is one of `patch`, `minor`, `major` or `dry-run`. With no argument, propose a level from the
rules below and ask the user.

## Bump rules (before 1.0)

| Bump | When |
|---|---|
| `patch` 0.8.x | A bug fix, a performance fix or an internal refactor. No MCP tool, parameter, CLI command, env var or stored schema changes. |
| `minor` 0.x.0 | A new or changed MCP tool or parameter, a new provider, a new CLI command or env var, a Milvus, FTS or Postgres schema change, a one-time migration or rewrite of stored data, or a change that refuses calls that worked before. Breaking changes go here before 1.0. |
| `major` 1.0.0 | Once, when a new Apple Silicon user can install from the README with no edits (SESF-88). After 1.0, `major` is for breaking MCP or config changes. |

- When one PR meets several rows, the highest wins.
- Each PR bumps on its own, even when several merge on one day.
- Put the level and a one-line reason in the PR body. The user confirms or changes it at review.

## Changelog entries

- Add bullets under `## [Unreleased]` while you work, in the Keep a Changelog groups: Added, Changed,
  Deprecated, Removed, Fixed, Security.
- One bullet per user-visible change. Name the tool, command or env var, and end with the issue ID in
  parentheses, for example `(SESF-67)`.
- The changelog holds history, so the state-not-change prose rule does not apply to it.

## Step 1: Sync with main

```bash
git fetch origin main
git rev-list --count HEAD..origin/main
```

When the branch is behind, rebase it on `origin/main` first. Another PR may have taken the next version.

## Step 2: Claim the version

`devops/version.lock` lives in the main checkout (`$(git rev-parse --path-format=absolute --git-common-dir)/..`)
and is gitignored. Claude and Codex sessions share it.

1. Read it with the Read tool. A missing file means no claims.
2. Drop claims whose `expires_at` has passed.
3. Take the highest version among `origin/main:version.py` and the remaining claims. Bump it by the
   chosen level: `patch` raises the last part, `minor` raises the middle part and zeroes the last, `major`
   raises the first part and zeroes the rest.
4. Append the claim and write the file:

```json
{
  "claims": [
    {
      "version": "0.9.0",
      "issue": "SESF-67",
      "branch": "feat/SESF-67-issue-patterns",
      "claimed_by": "claude-code",
      "claimed_at": "2026-10-30T22:00:00Z",
      "expires_at": "2026-11-06T22:00:00Z"
    }
  ]
}
```

A claim expires after 7 days, because a PR can wait that long for review.

## Step 3: Edit the files

1. `version.py`: set `__version__ = "NEW_VERSION"`.
2. `CHANGELOG.md`: insert `## [NEW_VERSION] - YYYY-MM-DD` (the local date, America/Chicago) under
   `## [Unreleased]`, and move every Unreleased bullet below it. Leave `## [Unreleased]` empty.
3. When Unreleased was empty, write the bullets now from the branch's commits
   (`git log --oneline origin/main..HEAD`).

`dry-run` shows these edits and the claim, and writes nothing.

## Step 4: Verify

```bash
./venv/bin/python -m pytest -q tests/test_version.py
git diff --stat origin/main...HEAD -- version.py CHANGELOG.md
```

## Step 5: Commit

```bash
git add version.py CHANGELOG.md
git commit -m "chore(SESF-NN): version NEW_VERSION"
```

Add a `Version` line to the PR body: `Version: 0.9.0 (minor): adds the issue_ids.conf env var and CLI command.`

## Step 6: After the merge

Remove this PR's claim from `devops/version.lock` and leave the other claims alone. Create no git tag and
no GitHub Release.

## Conflicts

- Two open PRs that claim the same version conflict in `version.py` and `CHANGELOG.md`. The second to merge
  rebases and runs `/release` again for the next number.
- The `Version` workflow also fails when `version.py` does not rise above the base branch.
