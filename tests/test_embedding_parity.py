"""MLX EmbeddingGemma vectors must match the sentence-transformers reference.

The other real-model tests compare MLX variants with each other. This one compares
``rag_engine.embed_texts`` with ``google/embeddinggemma-300m`` run by
sentence-transformers on the CPU in float32, for a short text and for a text long
enough to reach the sliding window. It skips unless both models are cached and
sentence-transformers is installed.
"""

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("mlx.core")

import rag_engine  # noqa: E402

REFERENCE_ID = "google/embeddinggemma-300m"
REFERENCE_CACHE = Path.home() / ".cache/huggingface/hub" / "models--google--embeddinggemma-300m"
PARITY_MIN_COS = 0.999

_mlx_cached = rag_engine._MODEL_CACHE.exists() and "gemma" in rag_engine._MODEL_NAME
pytestmark = pytest.mark.skipif(
    not (_mlx_cached and REFERENCE_CACHE.exists()),
    reason=f"EmbeddingGemma MLX or {REFERENCE_ID} not cached locally",
)


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture(scope="module")
def reference():
    st = pytest.importorskip("sentence_transformers")
    model = st.SentenceTransformer(REFERENCE_ID, device="cpu", model_kwargs={"torch_dtype": "float32"})
    # The text already carries the rag_engine prefix; a default prompt would add a second one.
    model.default_prompt_name = None
    return model


def _reference_vector(model, text):
    return model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()


def _long_text(tokens=1500):
    _, tokenizer = rag_engine.get_model()
    words = ("the backfill queue drains each provider in turn and checkpoints its cursor "
             "so a restart resumes where the last run stopped ").split()
    text, i = "", 0
    while len(tokenizer.encode(text)) < tokens:
        text += words[i % len(words)] + " "
        i += 1
    return text


@pytest.mark.parametrize("is_query", [True, False])
def test_short_text_matches_the_reference(reference, is_query):
    text = "How does SessionFlow keep MLX work on one thread?"
    prefix = rag_engine._SEARCH_PREFIX if is_query else rag_engine._DOCUMENT_PREFIX

    mlx_vec = rag_engine.embed_texts([text], is_query=is_query)[0]
    ref_vec = _reference_vector(reference, prefix + text)

    assert _cos(mlx_vec, ref_vec) >= PARITY_MIN_COS


def test_long_text_matches_the_reference(reference):
    text = _long_text()

    mlx_vec = rag_engine.embed_texts([text])[0]
    ref_vec = _reference_vector(reference, rag_engine._DOCUMENT_PREFIX + text)

    assert _cos(mlx_vec, ref_vec) >= PARITY_MIN_COS
