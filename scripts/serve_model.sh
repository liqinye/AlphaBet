#!/usr/bin/env bash
# Evaluation only: Slime manages its own policy servers during training.
set -euo pipefail
exec "${PYTHON:-python3}" -m sglang.launch_server \
  --model-path "${MODEL_PATH:?Set MODEL_PATH to the policy checkpoint}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30000}" \
  --tp "${TENSOR_PARALLEL_SIZE:-4}" "$@"
