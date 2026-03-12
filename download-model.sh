#!/bin/bash
# Download embedding model for MLX.
#
# Usage:
#   ./download-model.sh [model_name]
#
# model_name can be "modernbert" (default) or "embeddinggemma".
# Also respects SESSION_RAG_MODEL env var.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"

MODEL_NAME="${1:-${SESSION_RAG_MODEL:-embeddinggemma}}"
MODEL_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

case "$MODEL_NAME" in
    modernbert)
        MODEL_ID="nomic-ai/modernbert-embed-base"
        TEST_PREFIX="search_document: "
        IS_GEMMA="False"
        ;;
    embeddinggemma)
        MODEL_ID="mlx-community/embeddinggemma-300m-bf16"
        TEST_PREFIX="title: none | text: "
        IS_GEMMA="True"
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Valid options: modernbert, embeddinggemma"
        exit 1
        ;;
esac

echo "Downloading $MODEL_ID..."
echo "  Model will be cached in ~/.cache/huggingface/hub/"

"$PYTHON" -c "
from mlx_embeddings.utils import load, generate
model_id = '$MODEL_ID'
test_prefix = '$TEST_PREFIX'
is_gemma = $IS_GEMMA
print(f'  Loading {model_id} from HuggingFace...')
model, tokenizer = load(model_id)

if is_gemma:
    # gemma3_text __call__ expects 'inputs' not 'input_ids'
    encoded = tokenizer.batch_encode_plus(
        [test_prefix + 'test'], return_tensors='mlx', padding=True,
        truncation=True, max_length=2048,
    )
    output = model(encoded['input_ids'], attention_mask=encoded.get('attention_mask'))
else:
    output = generate(model, tokenizer, texts=[test_prefix + 'test'])

dims = output.text_embeds.shape[1]
print(f'  Model ready: {dims} dimensions')
"

echo "  Download complete."
