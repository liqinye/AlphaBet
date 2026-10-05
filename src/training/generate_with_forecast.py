# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forecasting rollout, reward, and logging callbacks for slime.

Each forecast step becomes a training sample with its own tokens and reward.
Memory-on episodes share a rollout ID and group rewards by (question, step).
Use `training.grpo_group_normalizer.normalize_by_group_index` with this callback.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState  # type: ignore
from slime.utils.types import Sample  # type: ignore
from strands_sglang import get_client_from_slime_args
from strands_sglang.tool_parsers import get_tool_parser

from strands_env.core.models import sglang_model_factory
from environment import ForecastEnv, ForecastMode
from environment.reward import ForecastReward
from strands_env.utils.slime_logger import RolloutLogger

logger = logging.getLogger(__name__)

INDEX_ROOT = os.environ["FORECAST_INDEX_ROOT"]
EMBEDDING_ENDPOINT = os.environ.get("FORECAST_EMBEDDING_ENDPOINT", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.environ.get("FORECAST_EMBEDDING_MODEL", "Qwen3-Embedding-8B")

ENABLE_CODE = os.environ.get("FORECAST_ENABLE_CODE", "0") == "1"

# Subprocess limits apply per call and per rollout-worker process.
CODE_TIMEOUT = int(os.environ.get("FORECAST_CODE_TIMEOUT", "10"))
CODE_CONCURRENCY = int(os.environ.get("FORECAST_CODE_CONCURRENCY", "24"))
CODE_MEMORY_MB = int(os.environ.get("FORECAST_CODE_MEMORY_MB", "4096"))

RETRIEVER_CONCURRENCY = int(os.environ.get("FORECAST_RETRIEVER_CONCURRENCY", "64"))

# Recency reranking is disabled when alpha=1 and oversample=1.
DECAY_ALPHA = float(os.environ.get("FORECAST_DECAY_ALPHA", "1.0"))
DECAY_HORIZON_DAYS = int(os.environ.get("FORECAST_DECAY_HORIZON_DAYS", "180"))
RERANK_OVERSAMPLE = int(os.environ.get("FORECAST_RERANK_OVERSAMPLE", "1"))

# Each forecast step receives the full tool budget.
MAX_TOOL_ITERS = 80
MAX_TOOL_CALLS = 200

# Must cover the dataset's maximum episode length to avoid GRPO group collisions.
MAX_STEPS = int(os.environ.get("FORECAST_MAX_STEPS", "100"))

# Mode determines metadata shape, notebook carry-forward, and reward formatting.
MODE = ForecastMode(os.environ.get("FORECAST_MODE", "memory-on"))

# Both proper scores are logged; this selects the scalar optimized by GRPO.
REWARD_METRIC = os.environ.get("FORECAST_REWARD_METRIC", "log_prob")
FORMAT_COEF = float(os.environ.get("FORECAST_FORMAT_COEF", "1.0"))
_REWARD = ForecastReward(metric=REWARD_METRIC, format_coef=FORMAT_COEF, score_notebook=MODE.uses_notebook)

# Optional penalty: coefficient * (fraction of known tool calls - 1).
TOOL_FORMAT_COEF = float(os.environ.get("FORECAST_TOOL_FORMAT_COEF", "0.0"))

# Match the policy's tool syntax: glm for GLM, hermes for Qwen/Hermes.
TOOL_PARSER = os.environ.get("FORECAST_TOOL_PARSER", "hermes")

# Load overrides at startup; otherwise the environment selects from prompts/.
_SYSTEM_PROMPT_PATH = os.environ.get("FORECAST_SYSTEM_PROMPT_PATH", "")
SYSTEM_PROMPT = (
    open(_SYSTEM_PROMPT_PATH, encoding="utf-8").read().strip() if _SYSTEM_PROMPT_PATH else None
)

# Belief kind is nested because slime forwards only the metadata column.
FORECAST_METRIC_SLICES = ("domain", "question.belief_kind")

# Require both sample count and distinct GRPO groups before logging a slice.
FORECAST_MIN_SLICE_SAMPLES = int(os.environ.get("FORECAST_MIN_SLICE_SAMPLES", "8"))
FORECAST_MIN_SLICE_GROUPS = int(os.environ.get("FORECAST_MIN_SLICE_GROUPS", "8"))

_ROLLOUT_LOGGER = RolloutLogger(
    backend="wandb",
    n_rollouts_per_step=3,
    log_per_tool_metrics=True,
    slice_by=FORECAST_METRIC_SLICES,
    min_slice_samples=FORECAST_MIN_SLICE_SAMPLES,
    min_slice_groups=FORECAST_MIN_SLICE_GROUPS,
)


# The eval generator and logger run sequentially in the rollout manager.
_EVAL_ROLLOUT_TIME: float | None = None


def eval_generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """Time slime's eval rollout for `log_eval_rollout_metrics`.

    The training path delegates without recording an extra timer.
    """
    import time

    from slime.rollout.sglang_rollout import generate_rollout  # type: ignore

    global _EVAL_ROLLOUT_TIME

    if not evaluation:
        return generate_rollout(args, rollout_id, data_source, evaluation=evaluation)

    start_time = time.time()
    try:
        return generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        _EVAL_ROLLOUT_TIME = time.time() - start_time


def log_eval_rollout_metrics(rollout_id, args, data, extra_metrics):
    """Log environment metrics and consume the eval timer once.

    Each dataset receives the elapsed time of the entire concurrent eval batch.
    The delegate's return value preserves slime's built-in logging.
    """
    global _EVAL_ROLLOUT_TIME

    result = _ROLLOUT_LOGGER.log_eval_rollouts(rollout_id, args, data, extra_metrics)

    # Consume once so a later untimed evaluation cannot reuse this duration.
    rollout_time, _EVAL_ROLLOUT_TIME = _EVAL_ROLLOUT_TIME, None
    if rollout_time is None:
        return result

    log_dict = {f"eval/{key}/rollout_time": rollout_time for key in (data or {})}
    if not log_dict:
        return result

    try:
        from slime.utils import logging_utils  # type: ignore
        from slime.utils.metric_utils import compute_rollout_step  # type: ignore

        log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, log_dict, step_key="eval/step")
    except Exception:  # noqa: BLE001 - logging must never break eval
        logger.exception("Failed to log eval rollout_time (rollout %d)", rollout_id)

    return result


from training import _diagnostics  # noqa: E402

_diagnostics.install()


def log_rollout_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Log rollout metrics and flush timing diagnostics."""
    result = _ROLLOUT_LOGGER.log_rollouts(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    _diagnostics.log_to_wandb(args, rollout_id)
    return result


_env: ForecastEnv | None = None
_env_lock = asyncio.Lock()


def _capture_routing(args, evaluation: bool) -> bool:
    """Request routed experts only when training uses MoE routing replay."""
    return bool(getattr(args, "use_rollout_routing_replay", False)) and not evaluation


def _build_model_factory(tokenizer, args, sampling_params, evaluation: bool = False):
    """Connect to slime's policy server with the configured tool parser.

    The timeout allows long responses to finish before a retry restarts generation.
    """
    return sglang_model_factory(
        tokenizer=tokenizer,
        client=get_client_from_slime_args(args, timeout=1200.0),
        sampling_params=sampling_params,
        tool_parser=get_tool_parser(TOOL_PARSER),
        return_routed_experts=_capture_routing(args, evaluation),
    )


async def _get_shared_env(args, sampling_params, tokenizer) -> ForecastEnv:
    """Build and prewarm one environment per worker.

    The retriever cache is shared; `run_episode` isolates date windows with ContextVar.
    """
    global _env
    if _env is not None:
        return _env
    async with _env_lock:
        if _env is not None:
            return _env
        env = ForecastEnv(
            model_factory=_build_model_factory(tokenizer, args, sampling_params),
            reward_fn=None,
            mode=MODE.value,
            system_prompt=SYSTEM_PROMPT,
            index_root=INDEX_ROOT,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_model=EMBEDDING_MODEL,
            enable_code=ENABLE_CODE,
            code_execution_timeout=CODE_TIMEOUT,
            code_concurrency=CODE_CONCURRENCY,
            code_memory_limit_mb=CODE_MEMORY_MB,
            retriever_concurrency=RETRIEVER_CONCURRENCY,
            decay_alpha=DECAY_ALPHA,
            decay_horizon_days=DECAY_HORIZON_DAYS,
            rerank_oversample=RERANK_OVERSAMPLE,
            max_tool_iters=MAX_TOOL_ITERS,
            max_tool_calls=MAX_TOOL_CALLS,
            verbose=False,
        )
        logger.info(
            "retrieval recency rerank: %s (decay_alpha=%.2f horizon=%dd oversample=%dx)",
            "ON" if DECAY_ALPHA < 1.0 else "OFF",
            DECAY_ALPHA,
            DECAY_HORIZON_DAYS,
            RERANK_OVERSAMPLE,
        )
        logger.info("prewarming FAISS shards under %s ...", INDEX_ROOT)
        n_loaded = await env.retriever_toolkit.prewarm()
        logger.info("prewarmed %d shard(s)", n_loaded)
        _env = env
        return _env


def _step_group_index(base_group_index: int | None, step_idx: int) -> int | None:
    """Group the same forecast step across rollouts of one question.

    `step_idx` must be below `MAX_STEPS` to keep question groups disjoint.
    """
    if base_group_index is None:
        return None
    return base_group_index * MAX_STEPS + step_idx


def _eval_group_index(base_sample: Sample) -> int | None:
    """Recover missing eval group IDs from (question, start_date, forecast_date).

    Use a stable digest so sibling rollouts agree across Ray worker processes.
    Python's built-in hash is randomized per process.
    """
    meta = getattr(base_sample, "metadata", None) or {}
    if "question" not in meta:
        return None
    identity = json.dumps(
        [meta.get("question"), meta.get("start_date"), meta.get("forecast_date")],
        sort_keys=True,
        default=str,
    )
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**40)


def _tool_quality(step) -> tuple[float, dict[str, float]]:
    """Return the fraction of calls to known tools and its logging components.

    Steps without tool calls receive 1.0, so valid non-use is not penalized.
    """
    metrics = getattr(step, "metrics", None) or {}
    per_tool = metrics.get("per_tool_metrics") or {}
    total_calls = 0
    known_calls = 0
    for tm in per_tool.values():
        c = tm.get("calls", 0) or 0
        total_calls += c
        if tm.get("is_known"):
            known_calls += c
    if total_calls == 0:
        return 1.0, {"tool_quality": 1.0, "tool_calls": 0.0, "tool_known_call_rate": 1.0}
    q = known_calls / total_calls
    return q, {
        "tool_quality": q,
        "tool_calls": float(total_calls),
        "tool_known_call_rate": q,
    }


def _build_step_sample(
    base_sample: Sample,
    step,
    step_idx: int,
    reward: float,
    reward_info: dict[str, Any],
    rollout_id: int,
    tokenizer,
    args=None,
    evaluation: bool = False,
) -> Sample | None:
    """Build one slime sample, or return None if the step has no token rollout."""
    rollout = step.rollout
    if not rollout:
        return None

    s = copy.copy(base_sample)
    prompt_len = rollout.initial_prompt_length
    s.tokens = rollout.token_ids
    # slime expects response-only masks and log probabilities.
    s.loss_mask = rollout.loss_mask[prompt_len:]
    s.rollout_log_probs = rollout.logprobs[prompt_len:]
    s.response_length = len(rollout.token_ids) - prompt_len
    s.response = tokenizer.decode(rollout.token_ids[prompt_len:], skip_special_tokens=False)

    # Routing replay expects int32[num_tokens - 1, num_layers, top_k].
    if args is not None and _capture_routing(args, evaluation) and rollout.routed_experts:
        s.rollout_routed_experts = rollout.decode_routed_experts(
            num_layers=args.num_layers, top_k=args.moe_router_topk
        )

    # Memory-free rows already identify (question, date); memory-on rows need step keys.
    # Eval rows may omit group_index, so recover a stable identity before remapping.
    base_group_index = base_sample.group_index
    if base_group_index is None:
        base_group_index = _eval_group_index(base_sample)
    s.group_index = (
        base_group_index
        if MODE is ForecastMode.MEMORY_FREE
        else _step_group_index(base_group_index, step_idx)
    )
    # Sharing the episode ID makes slime's loss reducer count the episode once.
    s.rollout_id = rollout_id
    s.reward = float(reward)
    s.status = (
        Sample.Status.COMPLETED
        if step.termination_reason.value == "task_complete"
        else Sample.Status.TRUNCATED
    )
    s.metrics = step.metrics
    s.metadata = {
        **(base_sample.metadata or {}),
        "step_idx": step_idx,
        "forecast_date": step.forecast_date,
        "reward_info": reward_info,
    }
    # RolloutLogger consumes the observation, termination reason, and reward interface.
    s.step_result = SimpleNamespace(
        observation=SimpleNamespace(metrics=step.metrics, rollout=rollout),
        termination_reason=step.termination_reason,
        reward=SimpleNamespace(reward=float(reward), info=reward_info),
    )
    return s


# Zero uses the local event loop; positive values dispatch to rollout workers.
ROLLOUT_SHARDS = int(os.environ.get("FORECAST_ROLLOUT_SHARDS", "0"))


async def generate_and_rm(args, sample: Sample, sampling_params, evaluation: bool = False) -> list[Sample]:
    """Dispatch one episode locally or to a rollout worker.

    slime passes `evaluation=True` for eval, which disables routed-expert capture.
    """
    if ROLLOUT_SHARDS > 0:
        if os.environ.get("FORECAST_USE_ACTOR_POOL", "0") == "1":
            from training import actor_pool_rollout

            return await actor_pool_rollout.dispatch(
                args, sample, sampling_params, n_actors=ROLLOUT_SHARDS, evaluation=evaluation
            )
        from training import rollout_shards

        return await rollout_shards.dispatch(
            args, sample, sampling_params, n_shards=ROLLOUT_SHARDS, evaluation=evaluation
        )
    return await _generate_and_rm_local(args, sample, sampling_params, evaluation=evaluation)


async def _generate_and_rm_local(
    args, sample: Sample, sampling_params, evaluation: bool = False
) -> list[Sample]:
    """Run one episode and return a sample for each forecast step.

    Memory-on samples share a rollout ID and use per-step GRPO groups; the
    group-index reward normalizer is required for uneven episode lengths.
    """
    assert not args.partial_rollout, "Partial rollout not supported."

    state = GenerateState(args)
    env = await _get_shared_env(args, sampling_params, state.tokenizer)
    # run_episode snapshots the factory; refresh it for this call's sampling parameters.
    env.model_factory = _build_model_factory(state.tokenizer, args, sampling_params, evaluation=evaluation)

    meta = getattr(sample, "metadata", None) or {}
    start_date = meta["start_date"]
    question_prompt = ForecastEnv.render_question(meta["question"])
    forecast_dates, crowd_probs = MODE.parse_metadata(meta)

    episode = await env.run_episode(
        question_prompt=question_prompt,
        forecast_dates=forecast_dates,
        start_date=start_date,
    )
    steps = episode.steps

    rollout_id = sample.index if sample.index is not None else sample.group_index

    out: list[Sample] = []
    for step_idx, step in enumerate(steps):
        crowd_prob = crowd_probs[step_idx] if step_idx < len(crowd_probs) else None
        reward_result = await _REWARD.score_step(
            notebook=step.final_response, label=sample.label, crowd_prob=crowd_prob
        )
        # Tool-format penalties require trajectory metrics unavailable to score_step.
        tool_reward = reward_result.reward
        tool_info = dict(reward_result.info)
        if TOOL_FORMAT_COEF:
            tq, tq_components = _tool_quality(step)
            tool_penalty = TOOL_FORMAT_COEF * (tq - 1.0)
            tool_reward = reward_result.reward + tool_penalty
            tool_info.update(tq_components)
            tool_info["tool_format_penalty"] = tool_penalty
            tool_info["reward"] = tool_reward
        step_sample = _build_step_sample(
            base_sample=sample,
            step=step,
            step_idx=step_idx,
            reward=tool_reward,
            reward_info=tool_info,
            rollout_id=rollout_id,
            tokenizer=state.tokenizer,
            args=args,
            evaluation=evaluation,
        )
        if step_sample is not None:
            out.append(step_sample)

    return out
