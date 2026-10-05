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

"""Distribute forecast episodes across Ray actors with separate event loops.

Workers run the same episode callback as the local path and return its samples
unchanged. Each owns its environment cache and model client. Diagnostics are
drained into the rollout manager before logging.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import ray

logger = logging.getLogger(__name__)

DEFAULT_IMPL = "training.generate_with_forecast._generate_and_rm_local"

SHARD_NUM_CPUS = float(os.environ.get("FORECAST_SHARD_NUM_CPUS", "2"))


@ray.remote(max_restarts=1)
class RolloutShardWorker:
    """A rollout worker with its own environment cache and clients.

    Ray restarts a failed worker once; exceptions from in-flight calls propagate.
    """

    def __init__(self, shard_id: int, impl_path: str = DEFAULT_IMPL) -> None:
        """Configure worker logging and defer loading the episode callback."""
        logging.basicConfig(level=logging.INFO)
        # Enable per-request HTTP logs only for recognized tool-trace modes.
        _trace = {
            t.strip()
            for t in os.environ.get("FORECAST_TOOL_TRACE", "").strip().lower().split(",")
            if t.strip()
        }
        if not _trace & {"1", "all", "search", "scrape"}:
            logging.getLogger("httpx").setLevel(logging.WARNING)
        self.shard_id = shard_id
        self._impl_path = impl_path
        self._impl = None
        logger.info("rollout shard %d up (pid %d, impl %s)", shard_id, os.getpid(), impl_path)

    def _resolve_impl(self):
        """Import the episode callback on first use, after environment setup."""
        if self._impl is None:
            module_path, func_name = self._impl_path.rsplit(".", 1)
            import importlib

            self._impl = getattr(importlib.import_module(module_path), func_name)
        return self._impl

    async def generate(self, args: Any, sample: Any, sampling_params: dict, evaluation: bool = False) -> list:
        """Run one episode and return its samples unchanged.

        Forward `evaluation` only when the callback accepts it.
        """
        import inspect

        impl = self._resolve_impl()
        if "evaluation" in inspect.signature(impl).parameters:
            return await impl(args, sample, sampling_params, evaluation=evaluation)
        return await impl(args, sample, sampling_params)

    def drain_diagnostics(self) -> dict:
        """Return and clear this worker's diagnostics for the manager to merge."""
        try:
            from training import _diagnostics

            return _diagnostics.drain()
        except Exception:
            logger.exception("shard %d: drain_diagnostics failed", self.shard_id)
            return {}

    def ping(self) -> int:
        """Return the shard ID as a liveness probe."""
        return self.shard_id


_workers: list[Any] = []
_rr_counter = 0
_pool_lock = asyncio.Lock()


def _forwarded_env() -> dict[str, str]:
    """Forward forecast configuration and Python imports to Ray workers.

    The episode module reads configuration, including the required index root,
    at import time.
    """
    fwd = {k: v for k, v in os.environ.items() if k.startswith("FORECAST_")}
    if "PYTHONPATH" in os.environ:
        fwd["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return fwd


async def _get_pool(n_shards: int, impl_path: str = DEFAULT_IMPL) -> list:
    """Create the shard pool once with SPREAD placement across nodes."""
    global _workers
    if _workers:
        return _workers
    async with _pool_lock:
        if _workers:
            return _workers
        workers = [
            RolloutShardWorker.options(
                num_cpus=SHARD_NUM_CPUS,
                scheduling_strategy="SPREAD",
                runtime_env={"env_vars": _forwarded_env()},
                name=f"forecast_rollout_shard_{i}",
                namespace="forecast_rollout_shards",
                # Reattach existing actors after a rollout-manager restart.
                get_if_exists=True,
            ).remote(i, impl_path)
            for i in range(n_shards)
        ]
        # Check worker startup before dispatching any episodes.
        ids = await asyncio.gather(*(asyncio.wrap_future(w.ping.remote().future()) for w in workers))
        logger.info("rollout shard pool up: %d shards %s", len(workers), sorted(ids))
        _workers = workers
        return _workers


async def dispatch(
    args: Any, sample: Any, sampling_params: dict, *, n_shards: int, evaluation: bool = False
) -> list:
    """Assign an episode round-robin and await its samples.

    Coroutine cancellation also cancels the remote task to avoid orphaned episodes.
    """
    global _rr_counter
    workers = await _get_pool(n_shards)
    worker = workers[_rr_counter % len(workers)]
    _rr_counter += 1
    ref = worker.generate.remote(args, sample, sampling_params, evaluation)
    try:
        return await asyncio.wrap_future(ref.future())
    except asyncio.CancelledError:
        try:
            ray.cancel(ref, force=False)
        except Exception:  # Cleanup must preserve the original cancellation.
            pass
        raise


def drain_all_diagnostics(timeout_s: float = 10.0) -> list[dict]:
    """Drain worker diagnostics for the synchronous rollout logging hook.

    The timeout prevents an unresponsive worker from stalling logging indefinitely.
    """
    if not _workers:
        return []
    try:
        return ray.get([w.drain_diagnostics.remote() for w in _workers], timeout=timeout_s)
    except Exception:
        logger.exception("drain_all_diagnostics failed (non-fatal)")
        return []
