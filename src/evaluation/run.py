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

"""Run forecast evaluation with shared prompts, environments, and training rewards."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prompts import PROMPTS_DIR, get_prompt_path

if TYPE_CHECKING:
    from environment import ForecastEnv, ForecastMode
    from environment.reward import ForecastReward
    from strands_env.core.models import ModelFactory
    from strands_env.core.types import RewardResult

Backend = Literal["sglang", "bedrock", "kimi"]
Mode = Literal["memory-on", "memory-free"]
Metric = Literal["log_prob", "brier"]


@dataclass
class ModelSpec:
    """Model backend and sampling settings.

    When model_id is unset, name selects a preset from MODEL_PRESETS.
    """

    name: str
    backend: Backend = "sglang"
    model_id: str | None = None
    # Variants use <variant>.<canonical filename> in prompts/; explicit paths take precedence.
    prompt_variant: str | None = None
    system_prompt_path: str | None = None
    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None
    tool_parser: str | None = None  # Inferred from model_id when omitted.
    max_connections: int = 1000
    # Bedrock
    region_name: str | None = None
    profile_name: str | None = None
    role_arn: str | None = None
    # Backends translate max_new_tokens to their native token limit parameter.
    sampling_params: dict[str, Any] = field(
        default_factory=lambda: {"max_new_tokens": 10240, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    )


@dataclass
class EnvSpec:
    """Environment tools, retrieval settings, and memory-on or memory-free mode."""

    index_root: str
    search: bool = True
    mode: Mode = "memory-on"
    embedding_endpoint: str = "http://localhost:8001/v1"
    embedding_model: str = "Qwen3-Embedding-8B"
    enable_code: bool = False
    code_role_arn: str | None = None
    retriever_concurrency: int | None = None  # None -> max(32, 2 * RunConfig.concurrency)
    max_tool_iters: int = 80
    max_tool_calls: int = 200
    # Retrieval-window lower bound; None -> earliest indexed shard date.
    history_start_date: str | None = None


@dataclass
class DataSpec:
    """Prepared dataset and row slice; row shape must match EnvSpec.mode."""

    questions_path: str
    split: str = "eval"  # label only, stamped into every output row
    question_start: int = 0
    question_end: int | None = None


@dataclass
class RunConfig:
    """A benchmark run over one model, environment, and data split."""

    model: ModelSpec
    env: EnvSpec
    data: DataSpec
    output_path: str
    concurrency: int = 8
    n_rollouts: int = 1  # independent episodes per question (sampling stochasticity)
    reward_metric: Metric = "log_prob"  # selected scalar; both rules always logged
    format_coef: float = 1.0  # matches the trainer's format-quality penalty weight
    keep_trace: bool = True  # dump the full agent transcript into each record
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            arm = "search" if self.env.search else "nosearch"
            self.run_id = f"{self.model.name}.{self.data.split}.{self.env.mode}.{arm}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        """Build a run configuration from a nested YAML/JSON dictionary."""
        return cls(
            model=ModelSpec(**d["model"]),
            env=EnvSpec(**d["env"]),
            data=DataSpec(**d["data"]),
            **{k: v for k, v in d.items() if k not in ("model", "env", "data")},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run configuration as a nested dictionary."""
        return {f.name: _as_plain(getattr(self, f.name)) for f in fields(self)}


def _as_plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


@dataclass
class EpisodeSpec:
    """Prepared question and resolved label, with aligned forecast dates and crowd probabilities."""

    event_id: str
    slug: str | None
    belief_kind: str
    label: str
    start_date: str
    question: dict[str, Any]
    forecast_dates: list[str]
    crowd_probs: list[float | None]

    @property
    def title(self) -> str:
        return self.question.get("title", "")


def load_episodes(data: DataSpec, mode: str) -> list[EpisodeSpec]:
    """Decode the prepared dataset and apply [question_start, question_end).

    The first row must match the requested memory-on or memory-free mode.
    """
    from environment import ForecastMode

    fmode = ForecastMode(mode)
    rows = _read_rows(Path(data.questions_path))
    if rows:
        _check_mode(rows[0], fmode, data.questions_path)

    end = data.question_end if data.question_end is not None else len(rows)
    episodes: list[EpisodeSpec] = []
    for row in rows[data.question_start : end]:
        meta = row["metadata"]
        forecast_dates, crowd_probs = fmode.parse_metadata(meta)
        episodes.append(
            EpisodeSpec(
                event_id=str(row.get("event_id")),
                slug=row.get("slug"),
                belief_kind=row["belief_kind"],
                label=row["label"],
                start_date=meta["start_date"],
                question=meta["question"],
                forecast_dates=list(forecast_dates),
                crowd_probs=list(crowd_probs),
            )
        )
    return episodes


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _check_mode(row: dict[str, Any], mode: ForecastMode, path: str) -> None:
    meta = row.get("metadata") or {}
    has_memory_on = "forecast_dates" in meta
    has_memory_free = "forecast_date" in meta
    file_mode = "memory-on" if has_memory_on else "memory-free" if has_memory_free else None
    if file_mode is None:
        raise ValueError(f"{path}: rows carry neither `forecast_dates` nor `forecast_date` — not a build_dataset file")
    if file_mode != mode.value:
        raise ValueError(
            f"mode mismatch: EnvSpec.mode={mode.value!r} but {path} contains {file_mode!r} rows. "
            f"Point at the matching split (training/{mode.value}/...) or set env.mode={file_mode!r}."
        )


MODEL_PRESETS: dict[str, ModelSpec] = {
    "glm-4.5-air": ModelSpec(
        name="glm-4.5-air",
        backend="sglang",
        model_id="zai-org/GLM-4.5-Air",
        tool_parser="glm",
    ),
    "qwen3-30b": ModelSpec(
        name="qwen3-30b",
        backend="sglang",
        model_id="Qwen/Qwen3-30B-A3B",
        tool_parser="hermes",
    ),
    "claude-opus": ModelSpec(
        name="claude-opus",
        backend="bedrock",
        model_id="us.anthropic.claude-opus-4-8",
        sampling_params={"max_new_tokens": 4096},
    ),
}


def resolve_spec(spec: ModelSpec) -> ModelSpec:
    """Apply preset values, then infer missing SGLang parser and tokenizer.

    Fields differing from ModelSpec defaults override the preset. The parser
    uses glm for GLM models and hermes otherwise; tokenizer_path uses model_id.
    """
    if spec.model_id is None and spec.name in MODEL_PRESETS:
        spec = _merge(MODEL_PRESETS[spec.name], spec)

    if spec.backend == "sglang":
        if spec.tool_parser is None:
            spec = replace(
                spec, tool_parser="glm" if "glm" in (spec.model_id or "").lower() else "hermes"
            )
        if spec.tokenizer_path is None:
            spec = replace(spec, tokenizer_path=spec.model_id)
    return spec


def build_factory(spec: ModelSpec) -> ModelFactory:
    from strands_env.core.models import ModelConfig, build_model_factory

    spec = resolve_spec(spec)
    return build_model_factory(
        ModelConfig(
            backend=spec.backend,
            base_url=spec.base_url,
            model_id=spec.model_id,
            tokenizer_path=spec.tokenizer_path,
            tool_parser=spec.tool_parser,
            max_connections=spec.max_connections,
            region_name=spec.region_name,
            profile_name=spec.profile_name,
            role_arn=spec.role_arn,
            sampling_params=spec.sampling_params,
        )
    )


def _merge(base: ModelSpec, override: ModelSpec) -> ModelSpec:
    """Overlay fields differing from ModelSpec defaults; always retain override.name."""
    defaults = ModelSpec(name=override.name)
    merged = asdict(base)
    for f in fields(override):
        value = getattr(override, f.name)
        if value != getattr(defaults, f.name):
            merged[f.name] = value
    merged["name"] = override.name
    return ModelSpec(**merged)


def get_prompt(model: ModelSpec, env: EnvSpec) -> tuple[str, str]:
    """Return prompt text and source for the configured mode and tools.

    An explicit path takes precedence over <variant>.<canonical filename> in
    prompts/. Missing variants fall back to the shared prompt.
    """
    if model.system_prompt_path:
        path = Path(model.system_prompt_path)
        return path.read_text(encoding="utf-8").strip(), str(path)

    path = get_prompt_path(env.mode, enable_code=env.enable_code, search=env.search)
    if model.prompt_variant:
        variant_path = PROMPTS_DIR / f"{model.prompt_variant}.{path.name}"
        if variant_path.is_file():
            path = variant_path

    return path.read_text(encoding="utf-8").strip(), f"prompts/{path.name}"


# Explicit keys keep the output schema stable as reward diagnostics evolve.
_REWARD_KEYS = (
    "reward",
    "metric",
    "proper_score",
    "format_penalty",
    "brier",
    "log_prob",
    "accuracy",
    "p_truth",
    "parse_ok",
    "prob_sum_raw",
    "truth_in_forecast",
    "info_alpha",
    "crowd_prob",
    "format_quality",
    "format_answer_ok",
    "format_notebook_ok",
)


def build_record(
    *,
    config: RunConfig,
    episode: EpisodeSpec,
    rollout_idx: int,
    step_idx: int,
    forecast_date: str,
    agent_forecast: dict[str, float] | None,
    final_response: str | None,
    agent_trace: list[dict[str, Any]] | None,
    prompt_source: str,
    reward: RewardResult,
    termination: str,
    step_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Combine run identity, forecast output, reward diagnostics, and optional trace."""
    info = reward.info
    row: dict[str, Any] = {
        "run_id": config.run_id,
        "model_name": config.model.name,
        "split": config.data.split,
        "mode": config.env.mode,
        "search": config.env.search,
        "prompt_source": prompt_source,
        "event_id": episode.event_id,
        "slug": episode.slug,
        "title": episode.title,
        "belief_kind": episode.belief_kind,
        "rollout_idx": rollout_idx,
        "step_idx": step_idx,
        "forecast_date": forecast_date,
        "resolved_label": episode.label,
        "agent_forecast": agent_forecast,
        "final_response": final_response,
        **{k: info.get(k) for k in _REWARD_KEYS},
        "termination": termination,
        "metrics": step_metrics,
    }
    if agent_trace is not None:
        row["agent_trace"] = agent_trace
    return row


def resume_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return (event_id, rollout_idx), treating missing rollout_idx as zero."""
    return str(row.get("event_id")), int(row.get("rollout_idx", 0))


def extract_full_trace(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return message text, tool calls, and tool results in conversation order.

    Assistant turns are numbered from one. Tool results share the current
    assistant turn; user messages without tool results use zero. Text,
    including any reasoning blocks, is preserved verbatim.
    """
    out: list[dict[str, Any]] = []
    assistant_turn = 0
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content") or []

        if role == "assistant":
            assistant_turn += 1
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"] or {}
                    tool_uses.append({
                        "id": tu.get("toolUseId") or "",
                        "name": tu.get("name"),
                        "input": tu.get("input") or {},
                    })
            out.append({
                "role": "assistant",
                "turn": assistant_turn,
                "text": "\n".join(text_parts),
                "tool_uses": tool_uses,
            })

        elif role == "user":
            text_parts = []
            tool_results: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolResult" in block:
                    tr = block["toolResult"] or {}
                    pieces = [
                        c["text"] for c in (tr.get("content") or [])
                        if isinstance(c, dict) and isinstance(c.get("text"), str)
                    ]
                    tool_results.append({
                        "id": tr.get("toolUseId") or "",
                        "status": tr.get("status", "success"),
                        "text": "\n".join(pieces),
                    })
            out.append({
                "role": "user",
                "turn": assistant_turn if tool_results else 0,
                "text": "\n".join(text_parts),
                "tool_results": tool_results,
            })

    return out


logger = logging.getLogger("eval.runner")


async def run(config: RunConfig) -> Path:
    """Run the benchmark, skipping recorded rollouts, and return the output path."""
    if not config.env.search:
        raise NotImplementedError(
            "no-search arm is not wired yet: ForecastNoSearchEnv has no run_episode "
            "(it extends Environment, not ForecastEnv). Add a single-step path first."
        )

    from environment import ForecastMode
    from environment.reward import ForecastReward

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_completed(output_path)
    logger.info("run_id=%s | resuming past %d completed rollout(s)", config.run_id, len(done))

    episodes = load_episodes(config.data, config.env.mode)
    logger.info("loaded %d episode(s) from %s", len(episodes), config.data.questions_path)

    env, prompt_source = _build_env(config)
    _write_manifest(output_path, config, prompt_source, env.system_prompt)
    reward = ForecastReward(
        metric=config.reward_metric,
        format_coef=config.format_coef,
        score_notebook=ForecastMode(config.env.mode).uses_notebook,
    )
    await _prewarm(env, episodes)

    units = [
        (ep, r)
        for ep in episodes
        for r in range(config.n_rollouts)
        if (ep.event_id, r) not in done
    ]
    logger.info("dispatching %d rollout(s) at concurrency %d", len(units), config.concurrency)

    sem = asyncio.Semaphore(max(1, config.concurrency))
    write_lock = asyncio.Lock()
    with output_path.open("a", encoding="utf-8") as out:

        async def worker(episode: EpisodeSpec, rollout_idx: int) -> None:
            async with sem:
                rows = await _run_rollout(config, env, reward, episode, rollout_idx, prompt_source)
            async with write_lock:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

        await asyncio.gather(*(worker(ep, r) for ep, r in units))

    await env.cleanup()
    logger.info("done -> %s", output_path)
    return output_path


async def _run_rollout(
    config: RunConfig,
    env: ForecastEnv,
    reward: ForecastReward,
    episode: EpisodeSpec,
    rollout_idx: int,
    prompt_source: str,
) -> list[dict[str, Any]]:
    """Run one episode and return one benchmark record per step (or an error row)."""
    from environment.notebook import extract_answer

    try:
        result = await env.run_episode(
            question_prompt=env.render_question(episode.question),
            forecast_dates=episode.forecast_dates,
            start_date=episode.start_date,
        )
    except Exception as e:  # Error records also count as completed on resume.
        logger.exception("episode failed: event_id=%s rollout=%d", episode.event_id, rollout_idx)
        return [{"event_id": episode.event_id, "rollout_idx": rollout_idx, "error": f"{type(e).__name__}: {e}"}]

    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(result.steps):
        crowd_prob = episode.crowd_probs[step_idx] if step_idx < len(episode.crowd_probs) else None
        reward_result = await reward.score_step(
            notebook=step.final_response, label=episode.label, crowd_prob=crowd_prob
        )
        rows.append(
            build_record(
                config=config,
                episode=episode,
                rollout_idx=rollout_idx,
                step_idx=step_idx,
                forecast_date=step.forecast_date,
                agent_forecast=extract_answer(step.final_response),
                final_response=step.final_response,
                agent_trace=extract_full_trace(step.messages) if config.keep_trace else None,
                prompt_source=prompt_source,
                reward=reward_result,
                termination=step.termination_reason.value,
                step_metrics=step.metrics,
            )
        )
    return rows


def _build_env(config: RunConfig) -> tuple[ForecastEnv, str]:
    """Build the shared environment with the selected prompt and external scoring.

    run_episode replaces the initial retrieval window for each step.
    """
    from environment import ForecastEnv

    env_spec = config.env
    window = "1970-01-01"
    system_prompt, prompt_source = get_prompt(config.model, env_spec)
    logger.info("system prompt: %s", prompt_source)
    env = ForecastEnv(
        model_factory=build_factory(config.model),
        reward_fn=None,
        mode=env_spec.mode,
        system_prompt=system_prompt,
        index_root=env_spec.index_root,
        embedding_endpoint=env_spec.embedding_endpoint,
        embedding_model=env_spec.embedding_model,
        enable_code=env_spec.enable_code,
        code_role_arn=env_spec.code_role_arn,
        retriever_concurrency=env_spec.retriever_concurrency or max(32, 2 * config.concurrency),
        max_tool_iters=env_spec.max_tool_iters,
        max_tool_calls=env_spec.max_tool_calls,
        start_date=window,
        cutoff_date=window,
        verbose=False,
    )
    return env, prompt_source


def _write_manifest(output_path: Path, config: RunConfig, prompt_source: str, system_prompt: str | None) -> None:
    """Write configuration and exact prompt text to an output manifest sidecar."""
    manifest = {
        "run_id": config.run_id,
        "config": config.to_dict(),
        "prompt_source": prompt_source,
        "system_prompt": system_prompt,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote run manifest -> %s", manifest_path)


async def _prewarm(env: ForecastEnv, episodes: list[EpisodeSpec]) -> None:
    """Prewarm the union of episode retrieval windows.

    run_episode narrows retrieval to each step's window.
    """
    earliest = min((ep.start_date for ep in episodes), default=None)
    latest = max((d for ep in episodes for d in ep.forecast_dates), default=None)
    if not (earliest and latest):
        return
    env.retriever_toolkit.start_date = earliest
    env.retriever_toolkit.cutoff_date = latest
    logger.info("prewarming shards in [%s, %s] ...", earliest, latest)
    n = await env.retriever_toolkit.prewarm()
    logger.info("prewarmed %d shard(s)", n)


def _load_completed(path: Path) -> set[tuple[str, int]]:
    """`(event_id, rollout_idx)` pairs already on disk — the resume set."""
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(resume_key(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return done


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _build_config(args)
    from .metrics import format_scorecard, score_run

    output_path = asyncio.run(run(config))
    print(format_scorecard(score_run(output_path)))


def _build_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig, applying supported top-level CLI overrides."""
    if args.config:
        base = json.loads(Path(args.config).read_text()) if args.config.endswith(".json") else _load_yaml(args.config)
        overrides = {k: v for k, v in _top_level_overrides(args).items() if v is not None}
        return RunConfig.from_dict({**base, **overrides})

    if not (args.model and args.index_root and args.questions_path and args.output_path):
        raise SystemExit("without --config, these are required: --model --index-root --questions-path --output-path")

    return RunConfig(
        model=ModelSpec(name=args.model, model_id=args.model_id, backend=args.backend or "sglang"),
        env=EnvSpec(index_root=args.index_root, search=not args.no_search, mode=args.mode or "memory-on"),
        data=DataSpec(questions_path=args.questions_path, split=args.split or "eval"),
        output_path=args.output_path,
        concurrency=args.concurrency or 8,
        n_rollouts=args.n_rollouts or 1,
    )


def _top_level_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {"output_path": args.output_path, "concurrency": args.concurrency, "n_rollouts": args.n_rollouts}


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evaluation", description="Forecast environment benchmark harness.")
    p.add_argument("--config", help="Path to a YAML/JSON RunConfig.")
    p.add_argument("--model", help="Model preset name (see evaluation.run.MODEL_PRESETS) or an ad-hoc label.")
    p.add_argument("--model-id", help="Override the preset's model_id / path.")
    p.add_argument("--backend", choices=["sglang", "bedrock", "kimi"])
    p.add_argument("--index-root", help="FAISS index root.")
    p.add_argument("--questions-path", help="Filtered questions JSONL (the input split).")
    p.add_argument("--split", help="Split label stamped into output (e.g. train/eval).")
    p.add_argument("--mode", choices=["memory-on", "memory-free"])
    p.add_argument("--no-search", action="store_true", help="Use the tool-less baseline env.")
    p.add_argument("--output-path", help="Output JSONL path.")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--n-rollouts", type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
