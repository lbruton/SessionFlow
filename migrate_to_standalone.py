#!/usr/bin/env python3
"""
One-time migration: Milvus Lite → Milvus Standalone.

Exports all vectors from the local ~/.sessionflow/milvus.db and bulk-inserts
them into a remote Milvus Standalone instance. No re-embedding needed — vectors
are copied as-is.

The collection is created with an HNSW index (vs FLAT on Lite) for O(log n) search.

Usage:
    python migrate_to_standalone.py --target http://192.168.1.81:19530 [--dry-run]

Requirements:
    - SessionFlow server must be STOPPED (Milvus Lite is single-process)
    - Target Milvus Standalone must be running and reachable
"""

import argparse
import sys
import time
from pathlib import Path

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

COLLECTION_NAME = "sessions"
EMBED_DIM = 768
BATCH_SIZE = 1000
LITE_DB = str(Path.home() / ".sessionflow" / "milvus.db")

ALL_FIELDS = [
    "id", "vector", "document", "doc_id", "session_id",
    "transcript_file", "turn_index", "timestamp",
    "git_branch", "chunk_type", "project_root",
]


def create_collection(client: MilvusClient):
    """Create sessions collection with HNSW index on Standalone."""
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists on target.")
        return False

    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="transcript_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="turn_index", dtype=DataType.INT64),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="git_branch", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="project_root", dtype=DataType.VARCHAR, max_length=512),
    ])

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}' with HNSW index.")
    return True


def export_all(source: MilvusClient) -> list:
    """Export all rows from Lite using id-based pagination (avoids 16384 offset cap)."""
    all_rows = []
    last_id = -1

    while True:
        batch = source.query(
            collection_name=COLLECTION_NAME,
            filter=f"id > {last_id}",
            limit=BATCH_SIZE,
            output_fields=ALL_FIELDS,
        )
        if not batch:
            break
        all_rows.extend(batch)
        last_id = max(row["id"] for row in batch)
        print(f"  Exported {len(all_rows)} rows (last_id={last_id})...", end="\r")

    print(f"  Exported {len(all_rows)} rows total.          ")
    return all_rows


def insert_batches(target: MilvusClient, rows: list):
    """Insert rows into Standalone in batches."""
    total = len(rows)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        target.insert(collection_name=COLLECTION_NAME, data=batch)
        inserted += len(batch)
        print(f"  Inserted {inserted}/{total} rows...", end="\r")

    print(f"  Inserted {inserted}/{total} rows total.          ")
    return inserted


def verify(source: MilvusClient, target: MilvusClient):
    """Compare row counts between source and target."""
    source_stats = source.query(
        collection_name=COLLECTION_NAME, filter="", limit=1, output_fields=["id"])
    target_stats = target.query(
        collection_name=COLLECTION_NAME, filter="", limit=1, output_fields=["id"])

    # Count all rows via pagination
    source_count = 0
    last_id = -1
    while True:
        batch = source.query(
            collection_name=COLLECTION_NAME,
            filter=f"id > {last_id}", limit=BATCH_SIZE, output_fields=["id"])
        if not batch:
            break
        source_count += len(batch)
        last_id = max(row["id"] for row in batch)

    target_count = 0
    last_id = -1
    while True:
        batch = target.query(
            collection_name=COLLECTION_NAME,
            filter=f"id > {last_id}", limit=BATCH_SIZE, output_fields=["id"])
        if not batch:
            break
        target_count += len(batch)
        last_id = max(row["id"] for row in batch)

    return source_count, target_count


def main():
    parser = argparse.ArgumentParser(description="Migrate SessionFlow from Milvus Lite to Standalone")
    parser.add_argument("--target", required=True, help="Standalone URI (e.g. http://192.168.1.81:19530)")
    parser.add_argument("--source", default=LITE_DB, help=f"Lite DB path (default: {LITE_DB})")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Error: Source DB not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print()

    # Connect to both
    print("Connecting to Milvus Lite...")
    source = MilvusClient(args.source)
    if not source.has_collection(COLLECTION_NAME):
        print(f"Error: No '{COLLECTION_NAME}' collection in source.", file=sys.stderr)
        sys.exit(1)

    print("Connecting to Milvus Standalone...")
    target = MilvusClient(args.target)

    # Export from Lite
    print("\nExporting from Lite...")
    start = time.monotonic()
    rows = export_all(source)
    export_time = time.monotonic() - start
    print(f"Export took {export_time:.1f}s")

    if args.dry_run:
        print(f"\n[DRY RUN] Would migrate {len(rows)} rows. Exiting.")
        source.close()
        target.close()
        return

    if not rows:
        print("No rows to migrate.")
        source.close()
        target.close()
        return

    # Create collection on Standalone
    print("\nCreating collection on Standalone...")
    create_collection(target)

    # Insert into Standalone
    print("\nInserting into Standalone...")
    start = time.monotonic()
    inserted = insert_batches(target, rows)
    insert_time = time.monotonic() - start
    print(f"Insert took {insert_time:.1f}s")

    # Verify
    print("\nVerifying row counts...")
    source_count, target_count = verify(source, target)
    print(f"  Source: {source_count}")
    print(f"  Target: {target_count}")

    if source_count == target_count:
        print("\nMigration successful! Row counts match.")
    else:
        print(f"\nWARNING: Row count mismatch! Source={source_count}, Target={target_count}",
              file=sys.stderr)

    source.close()
    target.close()

    print(f"\nNext steps:")
    print(f"  1. export SESSIONFLOW_MILVUS_URI={args.target}")
    print(f"  2. ./sessionflow-server.sh restart")
    print(f"  3. curl http://127.0.0.1:7102/health  # verify milvus_backend=standalone")


if __name__ == "__main__":
    main()
