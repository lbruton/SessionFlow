"""Memory indexes: the ``memories`` Milvus collection and the local FTS file.

Postgres is the canonical store. Everything here is derived from it and can be
rebuilt, so an index write that fails after a commit leaves the record pending
for the sync loop instead of failing the write.
"""

from __future__ import annotations

import asyncio
from typing import List

import embedding_control
import rag_engine


async def embed_for_index(texts: List[str]) -> List[List[float]]:
    """Embed memory texts as documents on the ``mlx-embed`` thread.

    Waits on the shared throttle instead of the transcript budget, which drops
    batches. ``rag_engine.embed_texts`` routes the MLX work to ``mlx-embed``.
    """
    throttle = embedding_control.get_embed_throttle()
    await throttle.wait()
    try:
        loop = asyncio.get_running_loop()
        executor = rag_engine._embed_executor
        if executor is not None:
            return await loop.run_in_executor(executor, rag_engine.embed_texts, texts)
        return await asyncio.to_thread(rag_engine.embed_texts, texts)
    finally:
        throttle.done()
