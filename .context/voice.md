# SessionFlow voice

This file sets the dials for `.claude/rules/prose-core.md`. Terms come from `.context/GLOSSARY.md`.

## Audiences

- Agents read the MCP tool descriptions, the parameter descriptions and the server `instructions`. They act on
  the text at once, so write commands.
- The maintainer reads the code, the README and `.context/`. Write for one engineer who knows Python and MCP.

## MCP tool descriptions

- Sentence 1 says what the tool returns, or what it deletes or changes.
- Name the required arguments. Say what the tool does when an optional argument is left out, if that matters.
- Give one usage rule, as a command: "Call get_turns to expand a hit."
- Keep a tool description to 3 or 4 sentences. A parameter description is one sentence plus an example.
- Write commands to the agent ("Pass", "Omit", "Call"). Do not write "you can" or "optionally".
- Use glossary terms: turn, provider, source kind, project root. Do not write "message" or "chunk".
- Do not describe internals the agent cannot act on (Milvus fields, FTS columns, header names).

## Code

- Public functions, classes and MCP tool handlers get Google-style docstrings: a summary line, then `Args:` and
  `Returns:` where they add information.
- Private helpers get one line or none.
- A docstring states a constraint once. Other docstrings name the function that enforces it.
- Comments state the constraint or the reason for a value, never the issue that changed it.

## README

- Keep the prose under 1,000 words, code blocks and tables excluded. Cover what SessionFlow does, setup, running, configuration and maintenance commands.
- Tie every number to a config value or a code constant. Do not quote sizes, counts or timings from a machine.
- Long operational detail goes to `CLAUDE.md` gotchas or the project DocVault.

## Exemptions

- `CHANGELOG.md` and `UPSTREAM-README.md` keep their history and wording.
- The `CLAUDE.md` and `AGENTS.md` gotcha lists keep issue IDs as the index into Plane.
- The README Reranking section may quote retrieval-eval results, the reranker model size and the added search time. Name the version and date they were measured on. Refresh them after a ranking change.
