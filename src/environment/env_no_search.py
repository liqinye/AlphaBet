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

"""Tool-free forecasting baseline."""

from prompts import get_prompt_path
from typing_extensions import Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction


class ForecastNoSearchEnv(Environment):
    """Forecast from prior knowledge with the memory-free, no-tools prompt."""

    default_system_prompt_path = get_prompt_path("memory-free", search=False)

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[EnvironmentConfig],
    ):
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)  # type: ignore[misc]

    @override
    def get_tools(self) -> list:
        return []

    async def cleanup(self) -> None:
        pass
