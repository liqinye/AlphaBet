"""Dispatch forecasting episodes through strands_env's EnvironmentActorPool.

Enable with `FORECAST_USE_ACTOR_POOL=1` and `FORECAST_ROLLOUT_SHARDS > 0`.
Uses the same episode callback and diagnostics as `training.rollout_shards`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

IMPL = "training.generate_with_forecast._generate_and_rm_local"

_pool: Any = None
_pool_lock = asyncio.Lock()


async def _get_pool(n_actors: int) -> Any:
    """Create the pool once in the rollout manager's event loop.

    `FORECAST_POOL_ACTORS_PER_NODE > 0` scales actors per live Ray node; zero uses
    the fixed `n_actors` total with SPREAD placement.
    """
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                from strands_env.core.distributed import EnvironmentActorPool

                per_node = int(os.environ.get("FORECAST_POOL_ACTORS_PER_NODE", "1"))
                sizing: dict[str, int] = (
                    {"n_actors_per_node": per_node} if per_node > 0 else {"n_actors_total": n_actors}
                )
                pool = EnvironmentActorPool(
                    impl_path=IMPL,
                    stats_hook_path="training._diagnostics.drain",
                    forward_env_prefixes=("FORECAST_", "PYTHONPATH"),
                    actor_namespace="forecast_rollout_pool",
                    num_cpus=float(os.environ.get("FORECAST_SHARD_NUM_CPUS", "2")),
                    **sizing,
                )
                # Surface worker initialization failures before dispatch.
                await pool.ready()
                logger.info("actor-pool rollout up: %d actors", len(pool.actors))
                _pool = pool
    return _pool


async def dispatch(
    args: Any, sample: Any, sampling_params: dict, *, n_actors: int, evaluation: bool = False
) -> list:
    """Dispatch one episode to a pool actor and await its samples."""
    pool = await _get_pool(n_actors)
    return await pool.run(args, sample, sampling_params, evaluation=evaluation)


def drain_all_diagnostics() -> list[dict]:
    """Drain pool diagnostics, or return an empty list if the pool is inactive."""
    return _pool.drain_all_stats() if _pool is not None else []
