---
tags: [index, sessionflow]
updated: "2026-05-21"
---

# SessionFlow

Semantic search over AI coding session transcripts (Claude Code CLI, Codex, OpenCode, Antigravity CLI + desktop) via Milvus + EmbeddingGemma; HTTP MCP server with hybrid vector + FTS5 retrieval. Embeddings stay local on MLX.

**Repo:** [lbruton/SessionFlow](https://github.com/lbruton/SessionFlow)

## Pages

| Page | Summary |
|------|---------|
| [Overview](/Volumes/DATA/GitHub/Devops/DocVault/Projects/SessionFlow/Overview.md) | Architecture, provider coverage, MCP tools, env vars, operational notes, gotchas |

## Issues

Tracked in Plane: <https://plane.lbruton.cc/lbruton/projects/3835ead1-4cc4-4f89-8145-4923068f7403/>. Pre-migration markdown archived at [[../../Archive/Issues-Pre-Plane/SessionFlow/SessionFlow|Archive/Issues-Pre-Plane/SessionFlow]].

## Specs

- [[../../specflow/SessionFlow/specs/SESF-6-multi-harness-session-ingestion-codex-opencode-antigravity/requirements|SESF-6 Multi-Harness Session Ingestion]] — Codex/OpenCode/Antigravity ingestion, provider-aware search, local-only embeddings (45/45 tests)
