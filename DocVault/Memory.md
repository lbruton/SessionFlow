---
tags: [sessionflow, memory, design]
doc_type: design
project: SessionFlow
source: manual
created: "2026-10-27"
updated: "2026-10-27"
---

# SessionFlow memories

> **Public copy.** Hosts, addresses and the deployment checklist are in the private companion
> (`vault-path private`) and the DEVS-145 cutover runbook.

A memory is a note an agent writes on purpose: a session summary, a decision, a handoff or a fact. SessionFlow
stores each one verbatim with fixed metadata, keeps every version, and makes it searchable next to the
transcript turns. No LLM rewrites, merges or classifies a memory.

## Components

| Part | Role |
| ---- | ---- |
| Postgres (`memory_store.py`) | Canonical store: current rows, `memory_versions`, `memory_index_acks`, `memory_index_meta`, `maintenance_runs`. |
| Milvus `memories` collection (`memory_index.py`) | One vector and the filter fields per memory. It holds no text. |
| `~/.sessionflow/memories_fts.db` (`memory_index.py`) | Keyword index on each machine, with a per-record version watermark. |
| `memory_service.py` | Validation, the write path, search, the audit and related conversations. |
| `tools.py` | The seven MCP tools, rendered as Markdown or JSON. |
| `memory_cli.py` | `rebuild`, `reindex`, `audit`, `import`, `restore-check`. It never embeds. |
| `hooks/memory-session-start.py` | Session-start context for Claude Code and Codex, standard library only. |

## Write path

1. `MemoryService` checks the text (at most 65,536 UTF-8 bytes, no secret-scanner hit), the agent, the type and
   the project, and extracts issue IDs with the transcript extractor.
2. `MemoryStore` runs one READ COMMITTED transaction. It locks the target, any topic-key or content-hash match
   and the supersede targets in `memory_id` order. It writes the current row and a version row, takes a new
   `change_seq` and deletes the record's acknowledgements. A unique-index race retries at most three times.
3. After the commit, the service applies the record to both indexes and lists similar memories. An index
   failure here still returns success, with `index: "pending"`.

A topic-key match replaces that record in place. A content-hash match returns the existing record. Both
matching different records is a conflict that names both IDs. A delete keeps the row as a tombstone and writes
a final version, so no replay or rebuild brings it back.

## Indexes and acknowledgements

Each machine has two consumers: the shared `milvus` collection and its own `fts:<host>` file. A record is
pending for a consumer while its acknowledgement is missing or older than the canonical version, or, for
Milvus, older than the current embedding generation. The sync loop applies pending records every few seconds.
A full reconcile compares the indexes with Postgres at start-up and every 6 hours.

- A Milvus worker embeds outside any lock, then re-reads the row under `pg_advisory_xact_lock`. It writes only
  if the content hash still matches; otherwise it embeds again. A tombstone becomes a delete.
- The FTS worker commits the version check, the change and the new watermark in one SQLite transaction.
- An acknowledgement never moves backwards: an update with an older version or generation changes nothing.
- `change_seq` orders rebuild work. It does not follow commit order, so it never decides what is pending.

## Search

Both channels overfetch three times the requested count. The candidates are read from Postgres in one query.
A hit whose version, or vector generation, differs from the canonical row is dropped, and so is a hit that
fails the filters. The survivors are fused with Reciprocal Rank Fusion and ranked on `created_at` with a
`memory_id` tie-break. A short result is fetched once more with a larger window. If it is still short while
the indexes lag, the result says `incomplete`. A Milvus failure leaves keyword search alone, reported as
`channels: ["fts"]`.

## Embedding generations

`memory_index_meta` names the model and dimension of the memory vectors and a generation number. A
`cleanup.py reembed` run starts a new generation, which makes every Milvus acknowledgement pending, and the
server re-embeds from the Postgres text. A server whose local model differs from the stored one refuses to
write memory vectors and reports why in `/health`.

## Configuration

SessionFlow reads the Postgres connection string from `SESSIONFLOW_MEMORY_DSN` only. How the value reaches the
server environment, from a secrets manager or a launcher, is the operator's choice. Without it, memory reports
"not configured" and transcript search runs as before. The README lists the other settings.

## Operations

- Back up Postgres. The indexes are rebuilt from it with `memory_cli.py rebuild --yes`.
- `memory_cli.py restore-check <dump> --target-dsn <empty database>` restores a dump, asks the server to index
  it into temporary targets and compares the result with the dump.
- `memory_cli.py audit` lists likely duplicates, aged open items and blockers, counts and unknown projects. It
  changes nothing.
- Transcript cleanup commands never touch memory data.
