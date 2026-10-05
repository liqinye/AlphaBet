"""Collect model, tool, and event-loop timings for rollout diagnostics.

Runtime wrappers instrument the imported clients without editing dependencies.
Workers accumulate statistics locally and merge them before each training log
flush. Eval statistics are included in the following training flush.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Statistics are written from both the event loop and FAISS worker threads.
_LOCK = threading.Lock()
_SAMPLES: dict[str, list[float]] = {}
_COUNTERS: dict[str, float] = {}


def _rec(name: str, value: float) -> None:
    with _LOCK:
        _SAMPLES.setdefault(name, []).append(value)


def _inc(name: str, value: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0.0) + value


_installed = False


def install() -> None:
    """Install timing probes once, logging and skipping installation failures."""
    global _installed
    if _installed:
        return
    _installed = True
    for name, fn in (
        ("sglang client timing", _patch_sglang_client),
        ("retry counter", _attach_retry_counter),
        ("retriever timing", _patch_retriever),
        ("code tool timing", _patch_code_tool),
    ):
        try:
            fn()
            logger.info("diag: installed %s", name)
        except Exception:
            logger.exception("diag: FAILED to install %s (continuing)", name)


def _patch_sglang_client() -> None:
    from strands_sglang.client import SGLangClient

    orig_generate = SGLangClient.generate

    async def generate(self: Any, input_ids: list[int], **kwargs: Any) -> dict[str, Any]:
        try:
            _ensure_probes()
        except Exception:
            pass
        t0 = time.perf_counter()
        out = await orig_generate(self, input_ids, **kwargs)
        # Unexpected response metadata must not interrupt a successful model call.
        try:
            client_s = time.perf_counter() - t0
            _rec("model_call/client_s", client_s)
            meta = out.get("meta_info") if isinstance(out, dict) else None
            if isinstance(meta, dict):
                e2e = meta.get("e2e_latency")
                if isinstance(e2e, (int, float)):
                    _rec("model_call/server_s", float(e2e))
                    _rec("model_call/gap_s", client_s - float(e2e))
                    ct = meta.get("completion_tokens")
                    if isinstance(ct, (int, float)) and e2e > 0:
                        _rec("model_call/decode_tok_per_s", float(ct) / float(e2e))
        except Exception:
            pass
        return out

    SGLangClient.generate = generate


# Match the client's warning/error messages without modifying its retry loop.
_RETRY_RE = re.compile(r"SGLang request failed.*?: (\w+):")


class _RetryCountingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            msg = record.getMessage()
            if "SGLang request failed" not in msg:
                return
            if "after" in msg and "attempts" in msg:
                _inc("retries/n_exhausted")
            else:
                _inc("retries/n_failed_attempts")
            m = _RETRY_RE.search(msg)
            if m:
                _inc(f"retries/by_type/{m.group(1)}")
        except Exception:
            pass


def _attach_retry_counter() -> None:
    logging.getLogger("strands_sglang.client").addHandler(_RetryCountingHandler())


_probe_loop: asyncio.AbstractEventLoop | None = None
# Keep strong references so background tasks are not garbage-collected.
_probe_tasks: list[asyncio.Task] = []


def _ensure_probes() -> None:
    """Start probes on the current event loop, or restart them if it changed."""
    global _probe_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _probe_loop is loop:
        return
    _probe_loop = loop
    _probe_tasks[:] = [
        loop.create_task(_loop_lag_probe()),
        loop.create_task(_thread_pool_probe()),
    ]


async def _loop_lag_probe() -> None:
    try:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(1.0)
            _rec("loop/lag_s", max(0.0, time.perf_counter() - t0 - 1.0))
    except (asyncio.CancelledError, Exception):
        pass


async def _thread_pool_probe() -> None:
    try:
        while True:
            t0 = time.perf_counter()
            await asyncio.to_thread(lambda: None)
            _rec("loop/to_thread_rtt_s", time.perf_counter() - t0)
            await asyncio.sleep(5.0)
    except (asyncio.CancelledError, Exception):
        pass


def _patch_retriever() -> None:
    from environment import tool as _tool

    rt = _tool.RetrieverToolkit

    orig_embed = rt._embed_query

    async def _embed_query(self: Any, query: str) -> Any:
        t0 = time.perf_counter()
        out = await orig_embed(self, query)
        _rec("search/embed_s", time.perf_counter() - t0)
        return out

    rt._embed_query = _embed_query

    orig_all = rt._search_all

    async def _search_all(self: Any, query: str, top_k: int) -> Any:
        t0 = time.perf_counter()
        out = await orig_all(self, query, top_k)
        _rec("search/total_s", time.perf_counter() - t0)
        return out

    rt._search_all = _search_all

    # Class access unwraps staticmethod; restore it when installing the wrapper.
    orig_shard = rt._search_shard_sync

    def _search_shard_sync(shard: Any, qvec: Any, top_k: int, selector: Any = None) -> Any:
        t0 = time.perf_counter()
        out = orig_shard(shard, qvec, top_k, selector)
        _rec("search/faiss_shard_s", time.perf_counter() - t0)
        return out

    rt._search_shard_sync = staticmethod(_search_shard_sync)

    orig_load = rt._load_shard

    def _load_shard(self: Any, date: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_load(self, date)
        _rec("search/load_shard_s", time.perf_counter() - t0)
        return out

    rt._load_shard = _load_shard

    orig_gpu = getattr(rt, "_search_all_gpu", None)
    if orig_gpu is not None:

        async def _search_all_gpu(self: Any, qvec: Any, shards: Any, pool_top_k: int) -> Any:
            t0 = time.perf_counter()
            out = await orig_gpu(self, qvec, shards, pool_top_k)
            _rec("search/gpu_post_s", time.perf_counter() - t0)
            return out

        rt._search_all_gpu = _search_all_gpu

    # Dedicated I/O-loop timings exclude delays resuming the caller's event loop.
    orig_embed_io = getattr(rt, "_embed_query_io", None)
    if orig_embed_io is not None:

        async def _embed_query_io(self: Any, prompt: str) -> Any:
            t0 = time.perf_counter()
            out = await orig_embed_io(self, prompt)
            _rec("search/embed_io_s", time.perf_counter() - t0)
            return out

        rt._embed_query_io = _embed_query_io

    orig_gpu_io = getattr(rt, "_search_gpu_post_io", None)
    if orig_gpu_io is not None:

        async def _search_gpu_post_io(self: Any, payload: Any) -> Any:
            t0 = time.perf_counter()
            out = await orig_gpu_io(self, payload)
            _rec("search/gpu_post_io_s", time.perf_counter() - t0)
            return out

        rt._search_gpu_post_io = _search_gpu_post_io


def _patch_code_tool() -> None:
    """Measure subprocess spawn, execution, and cleanup time from worker threads."""
    from environment.subprocess_interpreter import SubprocessInterpreter

    orig_run = SubprocessInterpreter.run

    def run(self: Any, code: str, code_type: str) -> str:
        t0 = time.perf_counter()
        try:
            return orig_run(self, code, code_type)
        finally:
            _rec("code/exec_s", time.perf_counter() - t0)

    SubprocessInterpreter.run = run


def drain() -> dict[str, Any]:
    """Atomically return and clear this process's statistics for cross-worker merging."""
    with _LOCK:
        out = {"samples": dict(_SAMPLES), "counters": dict(_COUNTERS)}
        _SAMPLES.clear()
        _COUNTERS.clear()
    return out


def absorb(drained: dict[str, Any]) -> None:
    """Merge a worker's drained statistics into this process."""
    if not drained:
        return
    with _LOCK:
        for name, values in (drained.get("samples") or {}).items():
            _SAMPLES.setdefault(name, []).extend(values)
        for name, value in (drained.get("counters") or {}).items():
            _COUNTERS[name] = _COUNTERS.get(name, 0.0) + value


def log_to_wandb(args: Any, rollout_id: int) -> None:
    """Merge worker statistics and log `diag/*` once per rollout step."""
    try:
        from slime.utils import logging_utils  # type: ignore
        from slime.utils.metric_utils import compute_rollout_step  # type: ignore

        try:
            from training import actor_pool_rollout, rollout_shards

            # Only one dispatcher is active; the inactive pool returns no payloads.
            for payload in (
                *rollout_shards.drain_all_diagnostics(),
                *actor_pool_rollout.drain_all_diagnostics(),
            ):
                absorb(payload)
        except Exception:
            logger.exception("diag: shard drain failed (non-fatal)")

        with _LOCK:
            samples = dict(_SAMPLES)
            _SAMPLES.clear()
            counters = dict(_COUNTERS)
            _COUNTERS.clear()

        log_dict: dict[str, float] = {}
        for name, values in samples.items():
            if not values:
                continue
            values.sort()
            n = len(values)
            log_dict[f"diag/{name}/mean"] = sum(values) / n
            log_dict[f"diag/{name}/p50"] = values[n // 2]
            log_dict[f"diag/{name}/p95"] = values[min(n - 1, int(n * 0.95))]
            log_dict[f"diag/{name}/max"] = values[-1]
            log_dict[f"diag/{name}/n"] = float(n)
        for name, value in counters.items():
            log_dict[f"diag/{name}"] = value

        if not log_dict:
            return
        _define_diag_axis(args)
        log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, log_dict, step_key="rollout/step")
    except Exception:
        logger.exception("diag: wandb flush failed (non-fatal)")


_axis_defined = False


def _define_diag_axis(args: Any) -> None:
    """Chart `diag/*` against `rollout/step`."""
    global _axis_defined
    if _axis_defined or not getattr(args, "use_wandb", False):
        return
    _axis_defined = True
    try:
        import wandb

        if wandb.run is not None:
            wandb.define_metric("diag/*", step_metric="rollout/step")
    except Exception:
        pass
