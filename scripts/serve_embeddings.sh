#!/usr/bin/env bash
# Run in a separate vLLM environment compatible with Qwen3-Embedding-8B.
set -euo pipefail
exec "${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
  --model "${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to the embedding model}" \
  --served-model-name Qwen3-Embedding-8B \
  --task embed \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8001}" "$@"
