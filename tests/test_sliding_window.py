"""EmbeddingGemma's sliding layers must see only keys within the configured window.

The model config sets ``sliding_window`` and puts a global layer every
``sliding_window_pattern`` layers. Hugging Face's bidirectional mask keeps a key
when ``|q - k| <= sliding_window``; global layers see every real key.
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
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_local_mask_keeps_keys_within_the_window():
    attention_mask = mx.array([[1, 1, 1, 1, 1, 1]])
    global_mask, local_mask = rag_engine._attention_masks(attention_mask, window=2)

    assert global_mask.shape == local_mask.shape == (1, 1, 6, 6)
    assert local_mask.dtype == mx.bool_
    for q in range(6):
        assert global_mask[0, 0, q].tolist() == [True] * 6
        assert local_mask[0, 0, q].tolist() == [abs(q - k) <= 2 for k in range(6)]


def test_local_mask_still_excludes_padding():
    attention_mask = mx.array([[1, 1, 1, 1, 0, 0]])
    _, local_mask = rag_engine._attention_masks(attention_mask, window=1)

    for q in range(6):
        assert local_mask[0, 0, q].tolist() == [abs(q - k) <= 1 and k < 4 for k in range(6)]


def test_text_inside_the_window_uses_the_padding_mask_for_every_layer():
    attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])
    global_mask, local_mask = rag_engine._attention_masks(attention_mask, window=3)

    assert local_mask is global_mask


_model_cached = rag_engine._MODEL_CACHE.exists() and "gemma" in rag_engine._MODEL_NAME
needs_model = pytest.mark.skipif(not _model_cached, reason="EmbeddingGemma not cached locally")


def _long_text():
    topics = ["Milvus upserts", "FTS5 healing", "MLX Metal crashes", "Plane states",
              "Codex rollouts", "launchctl kickstart", "boilerplate rules", "RRF weights"]
    return " ".join(
        f"Step {i}: the {topics[i % len(topics)]} check passed after change {i * 7} "
        f"and the next run used batch size {i % 5 + 1}." for i in range(160)
    )


def _full_attention_vector(model, tokenizer, text):
    """The forward without a window: every layer sees every real key."""
    from mlx_embeddings.models.base import mean_pooling, normalize_embeddings

    encoded = tokenizer.batch_encode_plus(
        [rag_engine._DOCUMENT_PREFIX + text], return_tensors="mlx", padding=True,
        truncation=True, max_length=rag_engine._MODEL_CFG["max_tokens"],
    )
    mask = encoded["attention_mask"]
    out = model.model(encoded["input_ids"], rag_engine._key_padding_mask(mask))
    for dense in model.dense:
        out = dense(out)
    return normalize_embeddings(mean_pooling(out, mask)).tolist()[0], encoded["input_ids"].shape[-1]


@needs_model
def test_short_text_vector_is_unchanged_by_the_window():
    model, tokenizer = rag_engine.get_model()
    text = "Milvus standalone runs on the homelab box; FTS5 sidecar handles keyword search."
    full, length = _full_attention_vector(model, tokenizer, text)

    assert length <= rag_engine._sliding_window(model)
    assert _cos(full, rag_engine.embed_texts([text])[0]) >= 0.99999


@needs_model
def test_long_text_vector_differs_from_full_attention():
    model, tokenizer = rag_engine.get_model()
    text = _long_text()
    full, length = _full_attention_vector(model, tokenizer, text)

    assert length > 2 * rag_engine._sliding_window(model)
    assert _cos(full, rag_engine.embed_texts([text])[0]) < 0.9999


@needs_model
def test_long_text_batched_with_short_texts_matches_single():
    texts = ["ok", _long_text(), "fix the padding leak in a mixed batch"]
    batched = rag_engine.embed_texts(texts)
    for text, vec in zip(texts, batched):
        assert _cos(rag_engine.embed_texts([text])[0], vec) >= 0.999, text[:40]
