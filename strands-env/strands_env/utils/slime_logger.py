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

"""Rollout logger for `slime` training, with a pluggable logging backend.

`RolloutLogger` aggregates per-rollout environment metrics into slime's
`rollout_extra_metrics` and publishes a sample of decoded rollouts to the
configured `backend`:

- `"wandb"` — metrics to `wandb`, samples to a W&B Weave dataset.
- `"mlflow"` — metrics and samples to MLflow.

Instantiate one and pass its bound `log_rollouts` as slime's
`--custom-rollout-log-function-path` callback. Backend libraries are imported
lazily, so only the selected backend needs to be installed.
"""

from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Any, Literal

from slime.rollout.sglang_rollout import GenerateState  # type: ignore
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix  # type: ignore
from slime.utils.types import Sample  # type: ignore

from strands_env.core.types import StepResult

if TYPE_CHECKING:
    from weave.trace.refs import ObjectRef
    from weave.trace.weave_client import WeaveClient

logger = logging.getLogger(__name__)

# NOTE: response_len in default logging refers to loss_mask=1 tokens

# Characters not safe in a metric key. Backend keys are `/`-delimited, so a slice
# VALUE containing `/` would silently open a nested namespace and split one
# slice's curves across two charts; anything else exotic is normalized too.
_SLICE_KEY_RE = re.compile(r"[^0-9A-Za-z_.@-]+")

# Default `slice_fields` whitelist: the scores worth reading per slice. Slicing
# EVERY `reward.info` field multiplies series count by (#fields x #stats x
# #slices) — with the forecast env's 25 info fields that is ~1100 new series per
# namespace, which the backend accepts but no one can read. Fields absent from a
# given env's `info` are simply never emitted, so this is safe as a default for
# any environment.
DEFAULT_SLICE_FIELDS: tuple[str, ...] = (
    "reward",
    "brier",
    "log_prob",
    "accuracy",
    "p_truth",
    "info_alpha",
)


def resolve_metadata_path(metadata: Any, path: str) -> str | None:
    """Follow a dotted `path` through nested dicts and return a metric-safe label.

    `"domain"` reads `metadata["domain"]`; `"question.belief_kind"` reads
    `metadata["question"]["belief_kind"]` (the forecast dataset keeps
    `belief_kind` nested there, because slime's reader copies only the
    `--metadata-key` column and drops sibling top-level columns).

    Returns None when any hop is missing / not a dict, or when the leaf is not a
    scalar — a sample that lacks the field then joins no slice and contributes to
    the un-sliced metrics exactly as before.
    """
    node: Any = metadata
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool):  # before int: bool is an int subclass
        node = str(node).lower()
    elif isinstance(node, (int, float)):
        node = str(node)
    elif not isinstance(node, str):
        return None  # dict / list / None leaf is not a slice label
    return _SLICE_KEY_RE.sub("_", node).strip("_") or None


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one group: `1 - C(n-c, k) / C(n, k)`.

    `n` = rollouts in the group, `c` = correct ones, `k` = budget. Same estimator
    as `strands_env.eval.metrics.compute_pass_at_k` (kept local to avoid importing
    the eval/EvalSample types into the slime logging path).
    """
    import math

    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)


def _auto_k_ladder(max_n: int) -> tuple[int, ...]:
    """Powers of 2 up to `max_n`, plus `max_n` itself — the default pass@k ladder.

    Derived from the group size (= number of rollouts per (question, step)), so a
    16-rollout training group yields (1, 2, 4, 8, 16) and a 1-sample eval group
    yields (1,). `max_n` not a power of 2 is included as the top rung, e.g. n=12
    -> (1, 2, 4, 8, 12).
    """
    if max_n < 1:
        return ()
    ladder = [1]
    while ladder[-1] * 2 < max_n:
        ladder.append(ladder[-1] * 2)
    if ladder[-1] != max_n:
        ladder.append(max_n)
    return tuple(ladder)


def compute_pass_at_k_by_group(
    groups: dict[Any, int], group_sizes: dict[Any, int], k_values: tuple[int, ...] | None
) -> dict[str, float]:
    """Average unbiased pass@k across groups, for each k with enough rollouts.

    Args:
        groups: `group_key -> #correct` (rollouts whose argmax hit the truth).
        group_sizes: `group_key -> #rollouts` in that group.
        k_values: explicit k's to report, or None to AUTO-derive the ladder from
            the largest group size (powers of 2 up to and including it). Auto is
            the default so the ladder always reaches the actual rollout count
            (e.g. n_samples=16 -> pass@1,2,4,8,16) with no config to keep in sync.

    Returns:
        `{"pass@k": mean over groups with size >= k}`; a k with no eligible group
        is omitted (so pass@16 disappears when every group has < 16).
    """
    if k_values is None:
        k_values = _auto_k_ladder(max(group_sizes.values())) if group_sizes else ()
    out: dict[str, float] = {}
    for k in k_values:
        scores = [
            _pass_at_k_unbiased(group_sizes[g], groups[g], k)
            for g in group_sizes
            if group_sizes[g] >= k
        ]
        if scores:
            out[f"pass@{k}"] = sum(scores) / len(scores)
    return out


class RolloutLogger:
    """Custom `slime` rollout logger with a pluggable logging backend.

    Aggregates per-rollout env metrics into slime's `rollout_extra_metrics` and
    publishes a sample of decoded rollouts to `backend` (`"wandb"` → a W&B Weave
    dataset, `"mlflow"` → MLflow JSON artifacts).

    Instantiate once and use `log_rollouts` as the
    `--custom-rollout-log-function-path` callback. Sample logging is controlled by
    `n_rollouts_per_step` (default 3); set to 0 to disable. `max_rollouts` caps the
    accumulated Weave dataset and is ignored by the MLflow backend.
    """

    def __init__(
        self,
        backend: Literal["wandb", "mlflow"] = "wandb",
        n_rollouts_per_step: int = 3,
        max_rollouts: int = 3000,
        log_per_tool_metrics: bool = False,
        pass_at_k: tuple[int, ...] | None = None,
        slice_by: tuple[str, ...] = (),
        slice_fields: tuple[str, ...] = DEFAULT_SLICE_FIELDS,
        min_slice_samples: int = 1,
        min_slice_groups: int = 1,
    ) -> None:
        """Initialize a `RolloutLogger` instance.

        `pass_at_k` is the k-ladder for pass@k (a question-step "passes" if any of
        its k rollouts got the argmax right, i.e. accuracy==1). Default `None`
        AUTO-derives the ladder from the actual group size (= rollouts per
        (question, step)): powers of 2 up to and including it. So training with
        `n_samples_per_prompt=16` reports pass@1,2,4,8,16 and eval with
        `n_samples_per_eval_prompt=1` reports pass@1 — always matching the real
        rollout count, no config to keep in sync. Pass an explicit tuple to fix
        the ladder; `()` disables pass@k.

        `slice_by` turns on per-slice reward breakdowns: each entry is a dotted
        path into `sample.metadata` whose VALUE names a slice, e.g.
        `("domain", "question.belief_kind")` yields
        `reward_info/sports/brier_min` and `reward_info/binary/accuracy_mean`.
        Slices from all paths share one flat namespace (the caller is responsible
        for keeping values distinct across paths). Empty (the default) is a
        complete no-op: the un-sliced metrics are byte-identical to before, so
        environments that don't set this — and datasets built before the field
        existed — are unaffected.

        `slice_fields` whitelists which `reward.info` fields get sliced (all
        stats of each).

        Two independent gates keep a thin cell from publishing a spike shaped
        like a trend; a slice must clear BOTH to be emitted:

        - `min_slice_samples` — fewer than N rollouts in this call.
        - `min_slice_groups` — fewer than N distinct `group_index` values. This
          is the gate that bites when one prompt fans out to many samples
          (`n_samples_per_prompt` x steps per episode): 16 rollouts of ONE
          question clear any sample threshold while carrying a single question's
          worth of information. Samples with `group_index=None` can't be
          attributed, so a slice made up entirely of them counts 0 groups and is
          gated out unless this is left at 1 (the default, i.e. off).

        Each surviving slice also reports `n_samples` and `n_groups` so the
        denominator is visible on the chart rather than implied.
        """
        self.backend = backend
        self.n_rollouts_per_step = n_rollouts_per_step
        self.log_per_tool_metrics = log_per_tool_metrics
        self.max_rollouts = max_rollouts
        # None = auto-derive the k-ladder from group size at log time; a tuple
        # fixes it; () disables. Preserve None (don't tuple() it).
        self.pass_at_k = pass_at_k if pass_at_k is None else tuple(pass_at_k)
        self.slice_by = tuple(slice_by)
        self.slice_fields = frozenset(slice_fields)
        self.min_slice_samples = max(1, int(min_slice_samples))
        self.min_slice_groups = max(1, int(min_slice_groups))
        # Weave (wandb backend) state, lazily initialized on first publish.
        self._weave_init = False
        self._weave_import_failed = False
        self._weave_client: WeaveClient | None = None
        self._prev_ref: ObjectRef | None = None
        self._rows: list[dict] = []
        self.run_name: str | None = None

    def log_rollouts(
        self,
        rollout_id: int,
        args: Any,
        samples: list[Sample],
        rollout_extra_metrics: dict | None,
        _rollout_time: float,
    ) -> bool:
        """Log env metrics and optionally publish sampled rollouts.

        Returns `False` so slime's default logging still runs.
        """
        # Check if step results are attached to samples
        for sample in samples:
            if not getattr(sample, "step_result", None):
                logger.warning("Skip custom rollout logging for rollout %d: missing `step_result`", rollout_id)
                return False

        self.log_rollout_metrics(samples=samples, rollout_extra_metrics=rollout_extra_metrics, args=args, rollout_id=rollout_id)
        self.log_rollout_samples(rollout_id=rollout_id, args=args, samples=samples)

        return False

    def log_eval_rollouts(
        self,
        rollout_id: int,
        args: Any,
        data: dict[str, dict[str, Any]],
        extra_metrics: dict | None,
    ) -> bool:
        """Eval-side custom log hook (slime `--custom-eval-rollout-log-function-path`).

        slime calls this with `data = {eval_key: {"rewards", "samples", ...}}`
        (one entry per `--eval-prompt-data` set). For each key we aggregate the
        same env + `reward.info` metrics as training, under `eval/{key}/...`, so
        held-out brier / accuracy / log_prob / info_alpha / pass@k sit alongside
        the train `rollout/reward_info/*` (and slime's built-in `eval/{key}/*`).

        Logging path (the subtle part): unlike the training hook — where slime
        passes a real `rollout_extra_metrics` dict that the built-in then logs —
        the eval call passes `result.metrics`, which is **None** for a custom
        generate that returns a bare `list[Sample]` (slime wraps it as
        `RolloutFnEvalOutput(metrics=None)`). Mutating that None can't work, and
        the built-in reads `extra_metrics or {}`, discarding any object we hand
        back. So we LOG OUR METRICS OURSELVES via `logging_utils.log`, adding the
        `eval/step` key as the shared x-axis (exactly as slime's built-in does),
        then return `False` so slime's built-in eval logging (`eval/{key}` reward
        mean, `eval/{key}/response_len/*`, truncated ratio) still runs as its own
        separate log call. If a real dict IS passed (future-proof / non-None), we
        merge into it instead and let the built-in carry our keys.

        Returns `False` so slime's built-in eval logging always runs too.
        """
        from slime.utils import logging_utils  # type: ignore

        log_dict: dict[str, float] = {}
        for key, payload in (data or {}).items():
            samples = payload.get("samples")
            if not samples:
                continue
            if any(not getattr(s, "step_result", None) for s in samples):
                logger.warning("Skip eval metrics for %r (rollout %d): a sample lacks `step_result`", key, rollout_id)
                continue
            log_dict |= dict_add_prefix(self._aggregate_sample_metrics(samples), f"eval/{key}/")

        if not log_dict:
            return False

        if extra_metrics is not None:
            # Real dict (e.g. a future slime that populates eval metrics): let the
            # built-in carry our keys in its single log call.
            extra_metrics.update(log_dict)
        else:
            # Actual eval path: extra_metrics is None, so self-log. `eval/step` is
            # the x-axis coordinate (NOT a metric bucket) so these points overlay
            # slime's built-in eval/{key}/* curve at the same step.
            log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
            # Also print the dict (slime's own `eval N: {...}` line carries only its generic keys), so held-out
            # brier / log_prob / accuracy survive in train_head.log for offline tools (scripts/eval_ckpt_series.sh).
            logger.info("eval_env %d: %s", rollout_id, log_dict)
            logging_utils.log(args, log_dict, step_key="eval/step")

        return False

    def log_rollout_metrics(
        self,
        samples: list[Sample],
        rollout_extra_metrics: dict | None,
        args: Any = None,
        rollout_id: int | None = None,
    ) -> None:
        """Aggregate env + reward metrics and log them on the `rollout/step` x-axis.

        X-axis fix: previously this only mutated `rollout_extra_metrics` and relied
        on slime's separate `_log_rollout_data` call to carry `rollout/step`. But
        the Weave sample-publish (and slime's own multi-call logging) advance
        wandb's internal auto-step, so the metrics charted against that auto-step
        (the "47" axis) instead of `rollout/step`. We now SELF-LOG with the
        `rollout/step` key explicitly included (same approach as `log_eval_rollouts`
        with `eval/step`), so `rollout/reward_info/*` aligns with `rollout/step`
        and `eval/step`. Falls back to the mutate path if `args` isn't provided.
        """
        log_dict = dict_add_prefix(self._aggregate_sample_metrics(samples), "rollout/")
        if args is not None and rollout_id is not None:
            from slime.utils import logging_utils  # type: ignore
            from slime.utils.metric_utils import compute_rollout_step  # type: ignore

            log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
            logging_utils.log(args, log_dict, step_key="rollout/step")
        elif rollout_extra_metrics is not None:
            rollout_extra_metrics.update(log_dict)
        else:
            logger.warning("rollout_extra_metrics is None and no args; env metrics will not be logged")

    def _slice_labels(self, sample: Sample) -> tuple[str, ...]:
        """Slice labels for one sample: one per `slice_by` path that resolves.

        Deduplicated while preserving `slice_by` order, so two paths that happen
        to resolve to the same value contribute one slice (and the sample is not
        double-counted in that slice's stats).
        """
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            return ()
        labels: list[str] = []
        for path in self.slice_by:
            label = resolve_metadata_path(metadata, path)
            if label is not None and label not in labels:
                labels.append(label)
        return tuple(labels)

    def _aggregate_sample_metrics(self, samples: list[Sample]) -> dict[str, float]:
        """Aggregate env metrics + reward diagnostics across samples (UN-prefixed).

        Shared by the training (`log_rollout_metrics` → `rollout/`) and eval
        (`log_eval_rollouts` → `eval/{key}/`) paths so both report the same
        `reward_info/*` (brier / log_prob / accuracy / p_truth / info_alpha) and
        tool/model metrics. The caller adds the leading prefix.

        When `slice_by` is set, the whitelisted `reward.info` fields are ALSO
        emitted per slice as `reward_info/<slice>/<field>_<stat>`. Because both
        paths funnel through here, one config gives the train and eval namespaces
        the same breakdown (`rollout/reward_info/sports/brier_min` and
        `eval/forecast/reward_info/sports/brier_min`) with no per-path wiring.
        """
        per_sample: dict[str, list[float]] = {
            "message_count": [],
            "model_calls": [],
            "model_latency_s": [],
            "cache_hit_rate": [],
            "tool_iters": [],
            "tool_calls": [],
            "executed_tool_calls": [],
            "cancelled_tool_calls": [],
            "tool_latency_s": [],
        }

        aggregated: dict[str, float] = {
            "tool_name_error_rate": 0.0,
            "tool_success_rate": 0.0,
            "tool_parse_error_rate": 0.0,
        }

        # Per-sample scalar reward diagnostics from `step_result.reward.info`
        # (e.g. forecast brier / log_prob / accuracy / p_truth). Collected
        # generically: only scalar int/float/bool fields, varying keys allowed
        # (error paths emit fewer keys), missing `info` skipped — so this stays
        # safe for any environment that shares this logger.
        reward_info_values: dict[str, list[float]] = {}

        # Per-slice copies of the whitelisted `reward.info` fields, keyed
        # `(slice_label, field)`; `slice_groups` tracks the DISTINCT group_index
        # values behind each slice, which both feeds the `min_slice_groups` gate
        # and is emitted as `n_groups` so a cell's real breadth is visible.
        # Both stay empty when `slice_by` is unset -> zero added keys.
        slice_values: dict[tuple[str, str], list[float]] = {}
        slice_groups: dict[str, set[Any]] = {}

        # pass@k accumulators, keyed by `group_index` (= one (question, step) group
        # = k rollouts of the same question at the same forecast date). A rollout
        # "passes" if its argmax hit the truth (info["accuracy"] >= 1). Groups with
        # no group_index are skipped (defends the eval path if slime omits it).
        # Enabled unless explicitly disabled with `()`; None = auto-ladder.
        pass_at_k_enabled = self.pass_at_k is None or len(self.pass_at_k) > 0
        pass_correct: dict[Any, int] = {}
        pass_total: dict[Any, int] = {}

        # Per-tool call-quality accumulators (self-contained; independent of the
        # `per_sample` / `aggregated` machinery above). Counters are summed over
        # ALL samples first — a tool absent from a sample contributes 0 — so the
        # emitted averages/rates have no per-sample-denominator bias. Emitted at
        # the end as `tool_<name>_avg_calls|success_rate|parse_error_rate`, keys
        # distinct from the global `tool_success_rate`/`tool_parse_error_rate`.
        tool_totals: dict[str, dict[str, int]] = {}
        n_valid_samples = 0

        total_executed_tool_calls = 0
        for sample in samples:
            reward = getattr(sample.step_result, "reward", None)
            info = getattr(reward, "info", None)
            if isinstance(info, dict):
                # Slice labels for THIS sample, resolved once and reused for
                # every whitelisted field below.
                slice_labels = self._slice_labels(sample) if self.slice_by else ()
                for k, v in info.items():
                    if isinstance(v, bool):
                        reward_info_values.setdefault(k, []).append(float(v))
                    elif isinstance(v, (int, float)):
                        reward_info_values.setdefault(k, []).append(float(v))
                    else:
                        continue  # non-scalar (forecast dict, truth str) — not a metric
                    if k in self.slice_fields:
                        for label in slice_labels:
                            slice_values.setdefault((label, k), []).append(float(v))
                gi = getattr(sample, "group_index", None)
                for label in slice_labels:
                    slice_groups.setdefault(label, set()).add(gi)
                if pass_at_k_enabled and gi is not None and "accuracy" in info:
                    pass_total[gi] = pass_total.get(gi, 0) + 1
                    pass_correct[gi] = pass_correct.get(gi, 0) + (1 if info["accuracy"] >= 1.0 else 0)

            metrics: dict[str, Any] = sample.step_result.observation.metrics
            if not metrics:
                continue
            n_valid_samples += 1

            per_sample["message_count"].append(metrics.get("message_count", 0))
            per_sample["tool_iters"].append(metrics.get("tool_iters", 0))
            per_sample["tool_calls"].append(metrics.get("tool_calls", 0))
            per_sample["cancelled_tool_calls"].append(metrics.get("cancelled_tool_calls", 0))
            per_sample["model_calls"].append(metrics.get("model_calls", 0))
            latency = metrics.get("model_latency_s")
            per_sample["model_latency_s"].append(latency["total"] if latency else 0)
            per_sample["cache_hit_rate"].append(metrics.get("cache_hit_rate") or 0)

            executed_tool_calls = 0
            tool_latency_s = 0.0
            for tool_name, tm in (metrics.get("per_tool_metrics") or {}).items():
                key = f"{tool_name}_tool"
                calls = tm["calls"]
                if tm["is_known"]:
                    if self.log_per_tool_metrics:
                        per_sample.setdefault(f"{key}_calls", []).append(calls)
                        per_sample.setdefault(f"{key}_latency_s", []).append(tm["latency_s"])
                        per_sample.setdefault(f"{key}_success_rate", []).append(tm["successes"] / calls)
                        per_sample.setdefault(f"{key}_parse_error_rate", []).append(tm.get("parse_errors", 0) / calls)
                else:
                    aggregated["tool_name_error_rate"] += calls
                executed_tool_calls += calls
                aggregated["tool_success_rate"] += tm["successes"]
                aggregated["tool_parse_error_rate"] += tm.get("parse_errors", 0)
                tool_latency_s += tm["latency_s"]
                # Accumulate per-tool totals for the dedicated per-tool block below.
                acc = tool_totals.setdefault(tool_name, {"calls": 0, "successes": 0, "parse_errors": 0})
                acc["calls"] += calls
                acc["successes"] += tm["successes"]
                acc["parse_errors"] += tm.get("parse_errors", 0)
            total_executed_tool_calls += executed_tool_calls
            per_sample["executed_tool_calls"].append(executed_tool_calls)
            per_sample["tool_latency_s"].append(tool_latency_s / executed_tool_calls if executed_tool_calls else 0.0)

        log_dict: dict[str, float] = {}
        for name, values in per_sample.items():
            if values:  # may be empty on the eval path if no sample carried metrics
                log_dict |= {f"{name}_{k}": v for k, v in compute_statistics(values).items()}
        for name, value in aggregated.items():
            log_dict[f"{name}"] = value / total_executed_tool_calls if total_executed_tool_calls else 0
        # Per-tool call quality (one set of keys per tool). Denominators:
        #   tool_<name>_avg_calls        = total calls / n_valid_samples
        #   tool_<name>_success_rate     = total successes / total calls
        #   tool_<name>_parse_error_rate = total parse_errors / (calls + parse_errors)
        for tool_name, acc in tool_totals.items():
            calls, successes, parse_errors = acc["calls"], acc["successes"], acc["parse_errors"]
            attempts = calls + parse_errors
            log_dict[f"tool_{tool_name}_avg_calls"] = calls / n_valid_samples if n_valid_samples else 0.0
            log_dict[f"tool_{tool_name}_success_rate"] = successes / calls if calls else 0.0
            log_dict[f"tool_{tool_name}_parse_error_rate"] = parse_errors / attempts if attempts else 0.0
        # Reward diagnostics under `reward_info/<field>_<stat>` (e.g.
        # reward_info/brier_mean, .../accuracy_mean, .../info_alpha_mean).
        reward_info_dict: dict[str, float] = {}
        for name, values in reward_info_values.items():
            if values:
                reward_info_dict |= {f"{name}_{k}": v for k, v in compute_statistics(values).items()}
        # Per-slice breakdown of the whitelisted fields, in the SAME namespace:
        # `reward_info/<slice>/<field>_<stat>` (e.g. reward_info/sports/brier_min).
        # A slice must clear BOTH gates: enough samples AND enough distinct
        # groups. The group gate is the load-bearing one when a prompt fans out
        # (n_samples_per_prompt x episode steps), since 16 rollouts of one
        # question trivially pass any sample threshold.
        emitted_slices: dict[str, int] = {}
        for (label, field), values in slice_values.items():
            if len(values) < self.min_slice_samples:
                continue
            # `> 1` guard, not `>= 1`: an env whose samples carry no group_index
            # counts 0 groups, and the DEFAULT must not silently suppress those
            # slices. The gate applies only once a caller opts into it.
            if self.min_slice_groups > 1:
                n_groups = len(slice_groups.get(label, frozenset()) - {None})
                if n_groups < self.min_slice_groups:
                    continue
            reward_info_dict |= {
                f"{label}/{field}_{stat}": v for stat, v in compute_statistics(values).items()
            }
            # Widest surviving field = how many samples this slice really had
            # (error paths emit fewer keys, so fields differ in length).
            emitted_slices[label] = max(emitted_slices.get(label, 0), len(values))
        # Denominators, so a chart reader can tell a broad estimate from one
        # question's k rollouts instead of guessing. `n_groups` counts distinct
        # `group_index` values — the caller's grouping unit, so for a stepwise
        # episode env this is (question, step) pairs, not questions. A None
        # group_index means slime never set one, so it can't be attributed.
        for label, n_samples in emitted_slices.items():
            reward_info_dict[f"{label}/n_samples"] = float(n_samples)
            reward_info_dict[f"{label}/n_groups"] = float(len(slice_groups.get(label, frozenset()) - {None}))
        log_dict |= dict_add_prefix(reward_info_dict, "reward_info/")
        # pass@k over (question, step) groups -> reward_info/pass@1, pass@2, ...
        if pass_total:
            log_dict |= dict_add_prefix(
                compute_pass_at_k_by_group(pass_correct, pass_total, self.pass_at_k), "reward_info/"
            )
        return log_dict

    def _build_sample_rows(self, rollout_id: int, step: int, args: Any, samples: list[Sample]) -> list[dict]:
        """Decode a random subset of rollouts into serializable rows for backend logging."""
        tokenizer = GenerateState(args).tokenizer
        n_saved = min(len(samples), self.n_rollouts_per_step)
        rows = []
        for s in random.sample(samples, k=n_saved):
            step_result: StepResult = s.step_result
            obs = step_result.observation
            rollout = obs.rollout
            if not rollout:
                logger.warning("rollout %d missing token rollout", rollout_id)
                continue

            prompt_len = rollout.initial_prompt_length
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "step": step,
                    "group_index": s.group_index,
                    "index": s.index,
                    "prompt": tokenizer.decode(rollout.token_ids[:prompt_len], skip_special_tokens=False),
                    "response": tokenizer.decode(rollout.token_ids[prompt_len:], skip_special_tokens=False),
                    "termination_reason": step_result.termination_reason.value,
                    "reward": step_result.reward.reward if step_result.reward else None,
                    "reward_info": step_result.reward.info if step_result.reward else None,
                    "metrics": obs.metrics,
                }
            )
        return rows

    def log_rollout_samples(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish a sample of decoded rollouts to the configured backend."""
        match self.backend:
            case "wandb":
                self._log_samples_wandb(rollout_id, args, samples)
            case "mlflow":
                self._log_samples_mlflow(rollout_id, args, samples)
            case _:
                raise ValueError(f"Unknown logging backend {self.backend!r} (expected 'wandb' or 'mlflow')")

    def _log_samples_wandb(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish sampled rollout step_results to a single W&B Weave dataset per run."""
        project = getattr(args, "wandb_project", None)
        if (
            not getattr(args, "use_wandb", False)
            or not project
            or not samples
            or self.n_rollouts_per_step <= 0
            or self._weave_import_failed
        ):
            return
        try:
            import wandb
            import weave
        except ImportError as exc:
            self._weave_import_failed = True
            logger.warning("Skipping Weave rollout samples: %s. Scalar metrics remain enabled.", exc)
            return

        # Lazy Weave init from args.wandb_project
        if not self._weave_init:
            self._weave_client = weave.init(project)
            self.run_name = wandb.run.name if wandb.run else "unknown"
            self._weave_init = True

        step = compute_rollout_step(args, rollout_id)
        rows = self._build_sample_rows(rollout_id, step, args, samples)
        if not rows:
            return

        # Accumulate rows locally, cap at max_rollouts, publish fresh each time.
        self._rows.extend(rows)
        if len(self._rows) > self.max_rollouts:
            self._rows = self._rows[-self.max_rollouts :]

        dataset_name = f"{self.run_name}_rollouts"
        dataset = weave.Dataset(name=dataset_name, rows=weave.Table(rows=self._rows))
        new_ref = weave.publish(dataset)

        # Delete previous version (each version is a superset, so old ones are redundant).
        if self._prev_ref is not None and self._weave_client is not None:
            try:
                self._weave_client.delete_object_version(self._prev_ref)
            except Exception:
                logger.debug("Failed to delete previous Weave dataset version", exc_info=True)
        self._prev_ref = new_ref

        logger.info("Published %d new samples to Weave (rollout %d, step %d)", len(rows), rollout_id, step)

    def _log_samples_mlflow(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish sampled rollout step_results to MLflow as a per-step JSON artifact."""
        import mlflow

        step = compute_rollout_step(args, rollout_id)
        rows = self._build_sample_rows(rollout_id, step, args, samples)
        if not rows:
            return

        mlflow.log_dict(rows, f"rollout_samples/step_{step:05d}.json")  # type: ignore[arg-type]

        logger.info("Logged %d samples to MLflow (rollout %d, step %d)", len(rows), rollout_id, step)
