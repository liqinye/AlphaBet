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

"""Ray actor pool for distributing environment episodes across processes.

Two contracts, selectable per pool:
  - typed:   `step(action) → StepResult` / `compute_reward(...)` via `env_hook_path`
             (any environment; JSON or pickle wire).
  - generic: `run(*args, **kwargs) → Any` via `impl_path` — the actor executes an
             arbitrary async callable and ships args/results by Ray serialization.
             This is the contract RL trainers need when one episode fans out to
             many samples (e.g. slime's `list[Sample]`).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import subprocess
from collections.abc import Sequence
from typing import Any, Literal

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from strands_env.utils.loader import load_function

from .types import Action, RewardResult, StepResult

logger = logging.getLogger(__name__)


@ray.remote(max_restarts=1)
class EnvironmentActor:
    """Remote worker that runs environment episodes in a dedicated process.

    Fully generic — every hook is a dotted path loaded inside the actor process,
    so nothing unpicklable ever crosses the wire. All domain logic lives in hooks.

    Args:
        env_hook_path: Dotted path to a callable returning an `AsyncEnvFactory`.
            Enables `step` / `compute_reward`. Optional if `impl_path` is given.
        env_hook_config: Configuration passed to the env hook callable.
        reuse_env: Keep ONE env alive for this actor's lifetime instead of building
            (and cleaning up) a fresh one per call — keeps expensive in-process
            caches warm across episodes. Only enable when BOTH hold: the factory
            ignores its `action` argument, and the env isolates concurrent episodes
            itself (e.g. per-task `ContextVar` state). The shared env is `reset()`
            exactly once at build — a per-episode reset would clobber episodes
            running concurrently.
        impl_path: Dotted path to an async callable; enables `run`.
        stats_hook_path: Dotted path to a zero-arg callable returning this
            process's stats snapshot (return-and-clear); enables `drain_stats`.
    """

    def __init__(
        self,
        env_hook_path: str | None = None,
        env_hook_config: dict[str, Any] | None = None,
        reuse_env: bool = False,
        impl_path: str | None = None,
        stats_hook_path: str | None = None,
    ) -> None:
        """Initialize an `EnvironmentActor` instance."""
        if env_hook_path is None and impl_path is None:
            raise ValueError("Provide env_hook_path (step/compute_reward) and/or impl_path (run).")
        self.env_factory = load_function(env_hook_path)(**(env_hook_config or {})) if env_hook_path else None
        self._impl = load_function(impl_path) if impl_path else None
        self._stats_hook = load_function(stats_hook_path) if stats_hook_path else None
        self.reuse_env = reuse_env
        self._env: Any = None
        self._env_lock = asyncio.Lock()

    def ping(self) -> bool:
        """Liveness probe; forces `__init__` (hook import) errors to surface early."""
        return True

    def drain_stats(self) -> Any:
        """Return-and-clear this process's stats via `stats_hook_path` (no hook → None)."""
        return self._stats_hook() if self._stats_hook else None

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the `impl_path` callable; args/results travel by Ray serialization."""
        if self._impl is None:
            raise ValueError("EnvironmentActor was created without impl_path")
        return await self._impl(*args, **kwargs)

    async def _acquire_env(self, action: Action, *, reset: bool) -> tuple[Any, bool]:
        """Return `(env, dispose)`; `dispose` tells the caller to `cleanup()` after use.

        Default mode reproduces the original behavior exactly: fresh env per call,
        `reset()` only where the caller asks for it, always disposed. Reuse mode
        builds once under a lock (two racing first-calls must not build twice).
        """
        if self.env_factory is None:
            raise ValueError("EnvironmentActor was created without env_hook_path")
        if not self.reuse_env:
            env = await self.env_factory(action)
            if reset:
                await env.reset()
            return env, True
        async with self._env_lock:
            if self._env is None:
                env = await self.env_factory(action)
                await env.reset()  # once per actor lifetime, never per episode
                self._env = env
        return self._env, False

    async def step(self, action: str | Action) -> str | StepResult:
        """Run one environment step; the result mirrors the input encoding.

        Args:
            action: JSON string from `Action.model_dump_json()`, or an `Action`
                object when the pool runs with `wire="pickle"`.

        Returns:
            `StepResult` as a JSON string for a JSON input, else as an object.
        """
        as_json = isinstance(action, str)
        act = Action.model_validate_json(action) if as_json else action
        env, dispose = await self._acquire_env(act, reset=True)
        try:
            step_result = await env.step(act)
            return step_result.model_dump_json() if as_json else step_result
        finally:
            if dispose:
                await env.cleanup()

    async def compute_reward(self, action: str | Action, step_result: str | StepResult) -> str | RewardResult:
        """Recompute reward for an existing rollout without re-running the agent.

        Args:
            action: JSON string from `Action.model_dump_json()`, or an `Action`.
            step_result: JSON string from `StepResult.model_dump_json()`, or a
                `StepResult`.

        Returns:
            `RewardResult` as a JSON string for a JSON input, else as an object.
        """
        as_json = isinstance(action, str)
        act = Action.model_validate_json(action) if as_json else action
        result = StepResult.model_validate_json(step_result) if isinstance(step_result, str) else step_result
        env, dispose = await self._acquire_env(act, reset=False)  # reward-only: no episode init
        try:
            if env.reward_fn is None:
                raise ValueError("Environment has no reward function configured")
            reward_result = await env.reward_fn.compute(action=act, step_result=result)
            return reward_result.model_dump_json() if as_json else reward_result
        finally:
            if dispose:
                await env.cleanup()


class EnvironmentActorPool:
    """Pool of `EnvironmentActor` instances distributed across Ray nodes.

    Each actor runs in its own process with a separate GIL and event loop,
    enabling true CPU parallelism for agent episodes.

    Args:
        env_hook_path: Dotted path to a callable returning an `AsyncEnvFactory`.
        env_hook_config: Configuration passed to the hook callable in each actor.
        n_actors_per_node: Actors per alive Ray node (hard node affinity). Exactly
            one of this and `n_actors_total` must be given.
        n_actors_total: Total actor count, `SPREAD` across the cluster.
        reuse_env: Forwarded to every actor — see `EnvironmentActor`.
        wire: `"json"` round-trips pydantic JSON (works for any env; the default).
            `"pickle"` ships the objects themselves via Ray — no encode/parse cost,
            requires the action/result contents to be picklable.
        impl_path: Forwarded to every actor; enables `pool.run(...)`.
        stats_hook_path: Forwarded to every actor; enables `drain_all_stats()`.
        forward_env_prefixes: Copy the driver's env vars whose names start with any
            of these prefixes into every actor's runtime env (an exact name is its
            own prefix, so `"PYTHONPATH"` forwards just that variable).
        actor_namespace: Opt-in named actors (`<namespace>_<i>` + `get_if_exists`):
            a restarted driver re-attaches to live actors instead of duplicating
            them. Leave None for anonymous actors — two concurrent pools sharing a
            namespace would otherwise attach to each other's actors.
        num_cpus: Logical-CPU reservation per actor (advisory for Ray's scheduler).
    """

    def __init__(
        self,
        env_hook_path: str | None = None,
        env_hook_config: dict[str, Any] | None = None,
        n_actors_per_node: int | None = None,
        *,
        n_actors_total: int | None = None,
        reuse_env: bool = False,
        wire: Literal["json", "pickle"] = "json",
        impl_path: str | None = None,
        stats_hook_path: str | None = None,
        forward_env_prefixes: Sequence[str] = (),
        actor_namespace: str | None = None,
        num_cpus: float = 0.001,
    ) -> None:
        """Initialize an `EnvironmentActorPool` instance."""
        if (n_actors_per_node is None) == (n_actors_total is None):
            raise ValueError("Provide exactly one of n_actors_per_node or n_actors_total.")
        self.wire_json = wire == "json"
        env_vars = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_env_prefixes)}

        def make(i: int, strategy: Any) -> ActorHandle:
            opts: dict[str, Any] = {"num_cpus": num_cpus, "scheduling_strategy": strategy}
            if env_vars:
                opts["runtime_env"] = {"env_vars": env_vars}
            if actor_namespace:
                opts.update(name=f"{actor_namespace}_{i}", namespace=actor_namespace, get_if_exists=True)
            return EnvironmentActor.options(**opts).remote(  # type: ignore[attr-defined]
                env_hook_path=env_hook_path,
                env_hook_config=env_hook_config,
                reuse_env=reuse_env,
                impl_path=impl_path,
                stats_hook_path=stats_hook_path,
            )

        self.actors: list[ActorHandle] = []
        if n_actors_total is not None:
            self.actors = [make(i, "SPREAD") for i in range(n_actors_total)]
        else:
            nodes = [n for n in ray.nodes() if n.get("Alive")]
            if not nodes:
                raise RuntimeError("No alive Ray nodes for EnvironmentActor placement.")
            for node in nodes:
                scheduling = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                for _ in range(n_actors_per_node):  # type: ignore[arg-type]
                    self.actors.append(make(len(self.actors), scheduling))

        self.cycle = itertools.cycle(self.actors)
        logger.info("Created %d EnvironmentActor(s).", len(self.actors))

    async def ready(self) -> None:
        """Block until every actor is constructed — hook import errors surface HERE,
        before any episode is dispatched, instead of on the first call."""
        await asyncio.gather(*(asyncio.wrap_future(a.ping.remote().future()) for a in self.actors))

    async def _await_ref(self, obj_ref: Any) -> Any:
        """Await a Ray ref asyncio-natively; propagate cancellation to the actor task.

        Replaces `asyncio.to_thread(ray.get, ...)`, which funneled every result
        through the default (~32-thread) executor — head-of-line blocking at high
        concurrency — and leaked the still-running episode when the caller was
        cancelled.
        """
        try:
            return await asyncio.wrap_future(obj_ref.future())
        except asyncio.CancelledError:
            try:
                ray.cancel(obj_ref, force=False)
            except Exception:  # cancellation cleanup must not mask the CancelledError
                pass
            raise

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Round-robin one `impl_path` call to the next actor."""
        actor = next(self.cycle)
        return await self._await_ref(actor.run.remote(*args, **kwargs))

    async def step(self, action: Action) -> StepResult:
        """Run one environment step on the next available actor."""
        actor = next(self.cycle)
        payload: str | Action = action.model_dump_json() if self.wire_json else action
        result = await self._await_ref(actor.step.remote(payload))
        return StepResult.model_validate_json(result) if self.wire_json else result

    async def compute_reward(self, action: Action, step_result: StepResult) -> RewardResult:
        """Recompute reward for an existing rollout on the next available actor."""
        actor = next(self.cycle)
        if self.wire_json:
            obj_ref = actor.compute_reward.remote(action.model_dump_json(), step_result.model_dump_json())
            return RewardResult.model_validate_json(await self._await_ref(obj_ref))
        return await self._await_ref(actor.compute_reward.remote(action, step_result))

    def drain_all_stats(self, timeout_s: float = 10.0) -> list[Any]:
        """Collect (and clear) every actor's stats snapshot; [] on any failure —
        stats collection must never break the caller."""
        try:
            snaps = ray.get([a.drain_stats.remote() for a in self.actors], timeout=timeout_s)
            return [s for s in snaps if s]
        except Exception:
            logger.exception("drain_all_stats failed (non-fatal)")
            return []

    def shutdown(self) -> None:
        """Tear down the Ray cluster on every node."""
        self.actors.clear()
        self.cycle = itertools.cycle([])

        @ray.remote(num_cpus=0)  # type: ignore[untyped-decorator]
        def ray_stop() -> None:
            subprocess.Popen(
                "(setsid ray stop --force </dev/null >/dev/null 2>&1 &)",
                shell=True,
            )

        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ray_stop.options(scheduling_strategy=strategy).remote()

        ray.shutdown()
        logger.info("Shut down Ray cluster across all nodes.")
