# AlphaBet

**Reinforcement Learning for LLM Event Forecasting Agents**

AlphaBet trains a shared policy to **search, read, compute, and forecast** across
multiple pre-resolution dates. Each forecast receives outcome-based feedback,
using GRPO with per-date mean centering and no reward standard-deviation
normalization by default.

| Setting | Mode | Context across dates |
| --- | --- | --- |
| Memory-free | `memory-free` | Each forecast starts afresh |
| Memory-on | `memory-on` | A belief notebook carries forward |
| Single-date | `single_date` | One fixed date per event; memory-free interface |

## Data and corpus

**Training data.** We use forecasting tasks from
[Forecast-Dojo on Hugging Face](https://huggingface.co/datasets/foye107/Forecast-Dojo).
Prepare inputs for the selected memory mode with the utilities in `src/data/`;
see `bash scripts/prepare_data.sh --help` for the available commands.

**News corpus.** We construct our corpus following the same data collection,
preprocessing, and indexing pipeline as Forecast-Dojo. Refer to the
[Forecast-Dojo paper](https://arxiv.org/abs/2609.28876) and
[repository](https://github.com/liqinye/Forecast-Dojo) for the corpus methodology.

Datasets, model weights, and indexes are stored outside this repository. The
examples use `/datasets`, `/models`, and `/indexes`. Retrieval expects dated
shards containing `faiss.index`, `texts.parquet`, and `metadata.json`, with
`day_offsets` in monthly shard metadata.

## Environment

| Purpose | Docker image |
| --- | --- |
| Training | `liqinye27/alpha-forecast-efa:20260718` |
| Embedding and GPU search dependencies | `liqinye27/alpha-embedder-faiss-gpu:20260708` |

```bash
docker pull liqinye27/alpha-forecast-efa:20260718
docker pull liqinye27/alpha-embedder-faiss-gpu:20260708
```

Use Linux x86-64 with NVIDIA drivers and NVIDIA Container Toolkit. Mount the
repository and external assets into the containers. Multi-node EFA training
requires compatible host devices and networking; Compute requires Linux
namespace isolation to be available inside the worker containers.

Install this checkout inside the training environment while preserving its
existing dependencies:

```bash
python -m pip install --no-deps -e ./strands-env -e .
python -m pip check
```

For a custom environment, use Python 3.10+ and install with
`python -m pip install -e ./strands-env -e .`. Training also requires compatible
Slime, Megatron-LM, Ray, PyTorch, and SGLang installations.

<details>
<summary>Image digests for reproducible deployments</summary>

Use these references in place of the tags to pin the exact images:

```text
liqinye27/alpha-forecast-efa@sha256:7b72bfdba690fcc996553c497259c0b314de77c1a1d39c82e1697ecb0f7ca950
liqinye27/alpha-embedder-faiss-gpu@sha256:7137db0bd69be46d07eb217c29f61fc0955559ea4ad7fb891d03cec15ed86314
```

</details>

## Training

Start the embedding service in its container:

```bash
HOST=0.0.0.0 EMBEDDING_MODEL_PATH=/models/Qwen3-Embedding-8B \
  bash scripts/serve_embeddings.sh --tensor-parallel-size 1 --max-model-len 4096
```

With a Ray cluster running, adjust the paths and embedding host, then launch:

```bash
SLIME_ROOT=/root/slime \
MEGATRON_ROOT=/root/Megatron-LM \
TRAIN_PYTHON=python \
MODEL_PATH=/models/GLM-4.5-Air \
REF_LOAD=/models/megatron_checkpoint \
TRAIN_DATA=/datasets/memory-free/forecast_train.jsonl \
EVAL_DATA=/datasets/memory-free/forecast_validation.jsonl \
INDEX_ROOT=/indexes/faiss_index_monthly \
EMBEDDING_ENDPOINT=http://embedding-host:8001/v1 \
bash scripts/train.sh memory-free
```

Use `memory-on` or `single_date` with the corresponding prepared inputs. Inspect
commands with `bash scripts/train.sh memory-free --dry-run`. Defaults target
GLM-4.5-Air on 16 × 8 GPUs; hardware and batch settings live in `configs/train/`.
Set `FORECAST_GRPO_STD_NORM=1` for the standardization ablation or
`FORECAST_REWARD_METRIC=log_prob` for log-score rewards.

## Evaluation

Keep the embedding service running and start a policy server in a separate
terminal:

```bash
HOST=0.0.0.0 MODEL_PATH=/models/policy_checkpoint bash scripts/serve_model.sh
```

Set the model, data, index paths, and service addresses in the evaluation YAML,
then run from another terminal:

```bash
bash scripts/evaluate.sh --config configs/eval/memory-free.yaml
# Memory-on: configs/eval/memory-on.yaml
```

The configurations use four rollouts per task and Search/Read/Compute tools.
Search defaults to local CPU FAISS; the optional GPU search server is deployed
separately and is not included here.

## Repository layout

```text
configs/          Training and evaluation settings
scripts/          Data, training, evaluation, and service launchers
prompts/          Shared system prompts
src/
  data/           Task preparation
  environment/    Agent loop, tools, notebook, and rewards
  training/       Rollout callbacks and per-date GRPO grouping
  evaluation/     Forecast execution and scoring
strands-env/      Shared environment runtime
```

Third-party copyright and license notices are retained in the source files;
see [strands-env/LICENSE](strands-env/LICENSE).
