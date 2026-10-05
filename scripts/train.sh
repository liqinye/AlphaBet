#!/usr/bin/env bash
# Submit to an existing Ray cluster with AlphaBet and its training stack installed.
# Training framework compatibility requirements are documented in the README.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/train.sh [memory-free|memory-on|single_date] [--dry-run] [-- SLIME_ARGS...]

Required for training: SLIME_ROOT MODEL_PATH REF_LOAD TRAIN_DATA INDEX_ROOT EMBEDDING_ENDPOINT
Optional: EVAL_DATA MODEL_ARGS_SH TRAIN_PYTHON RAY_DASHBOARD_ADDRESS RUN_DIR MEGATRON_ROOT
The default model is GLM-4.5-Air on 16 x 8 GPUs; hardware values are configurable.
Set FORECAST_GRPO_STD_NORM=1 for the standardization ablation.
--dry-run prints the configuration and command without loading dependencies or data.
USAGE
}

mode=memory-free
dry_run=0
extra_args=()
while (($#)); do
  case "$1" in
    memory-free|memory-on|single_date) mode="$1" ;;
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/configs/train/common.sh"
source "$repo_root/configs/train/$mode.sh"

if ((dry_run)); then
  SLIME_ROOT="${SLIME_ROOT:-/path/to/slime}"
  MODEL_PATH="${MODEL_PATH:-/path/to/GLM-4.5-Air}"
  REF_LOAD="${REF_LOAD:-/path/to/GLM-4.5-Air_torch_dist}"
  TRAIN_DATA="${TRAIN_DATA:-/path/to/${mode}_train.jsonl}"
  INDEX_ROOT="${INDEX_ROOT:-/path/to/faiss_index_monthly}"
  EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-http://localhost:8001/v1}"
else
  for required in SLIME_ROOT MODEL_PATH REF_LOAD TRAIN_DATA INDEX_ROOT EMBEDDING_ENDPOINT; do
    if [[ -z "${!required:-}" ]]; then
      printf 'Required environment variable is unset: %s\n' "$required" >&2
      exit 2
    fi
  done
fi

TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
MODEL_ARGS_SH="${MODEL_ARGS_SH:-$SLIME_ROOT/scripts/models/glm4.5-106B-A12B.sh}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
RUN_DIR="${RUN_DIR:-$repo_root/outputs/$mode}"
EVAL_DATA="${EVAL_DATA:-}"
export FORECAST_INDEX_ROOT="$INDEX_ROOT"
export FORECAST_EMBEDDING_ENDPOINT="$EMBEDDING_ENDPOINT"
export PYTHONPATH="$SLIME_ROOT:$repo_root/src:$repo_root${MEGATRON_ROOT:+:$MEGATRON_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RAY_USE_UVLOOP="${RAY_USE_UVLOOP:-0}"

args=(
  --actor-num-nodes "$ACTOR_NUM_NODES" --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" --colocate
  --hf-checkpoint "$MODEL_PATH" --ref-load "$REF_LOAD"
  --load "$RUN_DIR/checkpoints" --save "$RUN_DIR/checkpoints" --save-interval "$SAVE_INTERVAL"
  --prompt-data "$TRAIN_DATA" --input-key prompt --label-key label --metadata-key metadata
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-temperature 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --balance-data
  --advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-6 --min-lr 1e-6 --lr-warmup-iters 0 --lr-decay-style constant
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  --tensor-model-parallel-size "$TP_SIZE" --pipeline-model-parallel-size "$PP_SIZE"
  --context-parallel-size "$CP_SIZE" --expert-model-parallel-size "$EP_SIZE"
  --expert-tensor-parallel-size "$ETP_SIZE" --sequence-parallel
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --log-probs-chunk-size 1024
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$SGLANG_MEM_FRAC" --sglang-context-length 131072
  --sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY"
  --sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL"
  --sglang-enable-dp-attention --sglang-dp-size "$SGLANG_DP_SIZE"
  --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-disable-custom-all-reduce
  --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --moe-token-dispatcher-type alltoall
  --make-vocab-size-divisible-by 1 --update-weight-buffer-size 2147483648
  --use-routing-replay --use-rollout-routing-replay
  --custom-generate-function-path training.generate_with_forecast.generate_and_rm
  --custom-rollout-log-function-path training.generate_with_forecast.log_rollout_metrics
  --custom-reward-post-process-path training.grpo_group_normalizer.normalize_by_group_index
)
if [[ "$ROLLOUT_SHUFFLE" == 1 ]]; then args+=(--rollout-shuffle); fi
if [[ -n "$EVAL_DATA" ]]; then
  args+=(
    --eval-interval "$EVAL_INTERVAL" --eval-prompt-data forecast "$EVAL_DATA"
    --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT" --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
    --eval-top-p 1 --eval-temperature 0.6
    --custom-eval-rollout-log-function-path training.generate_with_forecast.log_eval_rollout_metrics
    --eval-function-path training.generate_with_forecast.eval_generate_rollout
  )
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  args+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "${WANDB_GROUP:-$mode}")
fi
args+=("${extra_args[@]}")

if ((dry_run)); then
  printf '# mode=%s reward=%s std_normalization=%s compute=%s groups=%s optimizer_batch=%s\n' \
    "$FORECAST_MODE" "$FORECAST_REWARD_METRIC" "$FORECAST_GRPO_STD_NORM" "$FORECAST_ENABLE_CODE" \
    "$ROLLOUT_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
  printf '# Ray runtime environment forwards FORECAST_*, WANDB_*, PYTHONPATH, and runtime settings.\n'
  printf '# MODEL_ARGS and RUNTIME_ENV_JSON below are populated at launch; no dependencies have been loaded.\n'
  printf 'source %q\n' "$MODEL_ARGS_SH"
  printf '%q ' "$TRAIN_PYTHON" -m ray.scripts.scripts job submit "--address=$RAY_DASHBOARD_ADDRESS"
  printf '%s ' '--runtime-env-json="$RUNTIME_ENV_JSON"' --
  printf '%q ' "$TRAIN_PYTHON" "$SLIME_ROOT/train.py"
  printf '%s ' '"${MODEL_ARGS[@]}"'
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

for required_file in "$SLIME_ROOT/train.py" "$MODEL_ARGS_SH" "$TRAIN_DATA"; do
  [[ -f "$required_file" ]] || { printf 'File does not exist: %s\n' "$required_file" >&2; exit 2; }
done
if [[ -n "$EVAL_DATA" && ! -f "$EVAL_DATA" ]]; then
  printf 'Evaluation data does not exist: %s\n' "$EVAL_DATA" >&2; exit 2
fi
for required_dir in "$MODEL_PATH" "$REF_LOAD" "$INDEX_ROOT"; do
  [[ -d "$required_dir" ]] || { printf 'Directory does not exist: %s\n' "$required_dir" >&2; exit 2; }
done
MODEL_ARGS=()
source "$MODEL_ARGS_SH"
[[ ${#MODEL_ARGS[@]} -gt 0 ]] || { printf 'MODEL_ARGS_SH must define a nonempty MODEL_ARGS array.\n' >&2; exit 2; }

# Ray workers need the same task configuration as this driver. Construct JSON
# with Python rather than shell interpolation so spaces and quotes stay valid.
runtime_env_json="$("$TRAIN_PYTHON" - <<'PY'
import json
import os

names = {
    "PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "RAY_USE_UVLOOP",
    "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "MASTER_ADDR", "NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY",
}
env = {k: v for k, v in os.environ.items()
       if k.startswith(("FORECAST_", "WANDB_")) or k in names}
print(json.dumps({"env_vars": env}))
PY
)"
exec "$TRAIN_PYTHON" -m ray.scripts.scripts job submit --address="$RAY_DASHBOARD_ADDRESS" \
  --runtime-env-json="$runtime_env_json" -- "$TRAIN_PYTHON" "$SLIME_ROOT/train.py" \
  "${MODEL_ARGS[@]}" "${args[@]}"
