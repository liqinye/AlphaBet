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

"""Normalize GRPO rewards by `Sample.group_index`, preserving sample order.

Forecast episodes may emit different numbers of step samples, so slime's
fixed-size reward grouping cannot identify the intended (question, step) groups.
Register this callback with `--custom-reward-post-process-path`:

    training.grpo_group_normalizer.normalize_by_group_index
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

# The environment variable can disable scaling independently of slime CLI defaults.
_STD_NORM_ENV = os.environ.get("FORECAST_GRPO_STD_NORM", "1") != "0"


def normalize_by_group_index(args: Any, samples: list) -> tuple[list[float], list[float]]:
    """Center rewards within each group, optionally dividing by its standard deviation.

    Returns `(raw_rewards, normalized_rewards)` aligned with the flat `samples` list.
    Both `args.grpo_std_normalization` and `FORECAST_GRPO_STD_NORM` control scaling.
    """
    import torch

    raw_rewards = [s.get_reward_value(args) for s in samples]
    use_std = getattr(args, "grpo_std_normalization", True) and _STD_NORM_ENV

    groups: dict[Any, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.group_index].append(i)

    normalized = [0.0] * len(samples)
    for positions in groups.values():
        rewards = torch.tensor([raw_rewards[p] for p in positions], dtype=torch.float)
        # A singleton has no relative advantage and centers to zero.
        rewards = rewards - rewards.mean()
        # Unbiased std is NaN for singletons; epsilon handles equal multi-sample rewards.
        if use_std and len(positions) > 1:
            rewards = rewards / (rewards.std() + 1e-6)
        for pos, r in zip(positions, rewards.tolist(), strict=True):
            normalized[pos] = r

    return raw_rewards, normalized
