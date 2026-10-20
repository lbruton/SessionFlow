#!/usr/bin/env python3
"""Verify newest local provider artifacts can be parsed, and optionally indexed."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import rag_engine
from provider_antigravity import AntigravityAdapter
from provider_claude import ClaudeCodeCliAdapter
from provider_codex import CodexAdapter
from provider_opencode import OpenCodeAdapter


def _newest_source(adapter):
    sources = [source for source in adapter.discover_sources() if source.status == "eligible"]
    if not sources:
        return None
    def _mtime(source):
        try:
            return Path(source.path).stat().st_mtime
        except (FileNotFoundError, OSError):
            return 0

    return max(sources, key=_mtime)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true", help="Index parsed turns into the configured SessionFlow DB")
    parser.add_argument("--db-path", default=str(Path.home() / ".sessionflow" / "milvus.db"))
    args = parser.parse_args()

    adapters = [
        ClaudeCodeCliAdapter(),
        CodexAdapter(),
        OpenCodeAdapter(),
        AntigravityAdapter(source_kind="cli"),
        AntigravityAdapter(source_kind="desktop"),
    ]

    exit_code = 0
    for adapter in adapters:
        source = _newest_source(adapter)
        if source is None:
            print(f"{adapter.provider}: no eligible local source found")
            continue
        try:
            result = adapter.parse_source(source, cursor=None)
        except Exception as exc:
            print(f"{adapter.provider}: parse_source failed — {exc}")
            exit_code = 1
            continue
        if result.errors:
            exit_code = 1
        indexed = 0
        if args.index and result.turns:
            try:
                indexed = await rag_engine.add_turns_async(result.turns, db_path=args.db_path)
            except Exception as exc:
                print(f"{adapter.provider}: add_turns_async failed — {exc}")
                exit_code = 1
        print(
            f"{adapter.provider}: source={Path(source.path).name} "
            f"turns={len(result.turns)} indexed={indexed} errors={len(result.errors)}"
        )
        if not result.turns:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

