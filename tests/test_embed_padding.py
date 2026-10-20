"""SESF-57: batched document embeddings must not depend on padding.

mlx-embeddings 0.0.5 hands the 0/1 tokenizer mask to SDPA as an *additive*
float mask, so padded keys are barely down-weighted and leak into every
shorter text in a bidirectional EmbeddingGemma batch. These tests pin the
boolean key-padding mask and the batched == single invariant.
"""

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

mx = pytest.importorskip("mlx.core")

import rag_engine  # noqa: E402


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_padding_mask_is_boolean_key_mask():
    attention_mask = mx.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    mask = rag_engine._key_padding_mask(attention_mask)

    assert mask.dtype == mx.bool_
    assert mask.shape == (2, 1, 5, 5)
    # Every query row sees the same key pattern: real keys only.
    for q in range(5):
        assert mask[0, 0, q].tolist() == [True, True, True, False, False]
        assert mask[1, 0, q].tolist() == [True] * 5


_model_cached = rag_engine._MODEL_CACHE.exists() and "gemma" in rag_engine._MODEL_NAME
needs_model = pytest.mark.skipif(not _model_cached, reason="EmbeddingGemma not cached locally")

_MIXED_TEXTS = [
    "fix the padding leak",
    "Milvus standalone runs on the homelab box; FTS5 sidecar handles keyword search.",
    "ok",
    "The backfill throttle is 200ms between inserts because MLX Metal crashes "
    "the GPU driver after about fifty minutes of continuous compute. " * 6,
    "SESF-57",
]


@needs_model
def test_batched_matches_single_for_mixed_lengths():
    batched = rag_engine.embed_texts(_MIXED_TEXTS)
    for text, vec in zip(_MIXED_TEXTS, batched):
        single = rag_engine.embed_texts([text])[0]
        assert _cos(single, vec) >= 0.999, text[:40]


@needs_model
def test_single_text_vector_unchanged_from_library_path():
    """A batch of one has no padding, so the fix must not move it.

    Existing query vectors and unbatched documents stay comparable.
    """
    model, tokenizer = rag_engine.get_model()
    for text in _MIXED_TEXTS[:2]:
        prefixed = rag_engine._DOCUMENT_PREFIX + text
        encoded = tokenizer.batch_encode_plus(
            [prefixed], return_tensors="mlx", padding=True,
            truncation=True, max_length=rag_engine._MODEL_CFG["max_tokens"],
        )
        legacy = model(encoded["input_ids"], attention_mask=encoded["attention_mask"])
        legacy_vec = legacy.text_embeds.tolist()[0]
        assert _cos(legacy_vec, rag_engine.embed_texts([text])[0]) >= 0.9999
