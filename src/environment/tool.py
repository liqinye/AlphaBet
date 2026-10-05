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

"""RAG retriever toolkit backed by FAISS + a vLLM-served Qwen3-Embedding-8B endpoint."""

from __future__ import annotations

import asyncio
import contextvars
import heapq
import itertools
import json
import logging
import os
import re
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import faiss
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from openai import AsyncOpenAI
from strands import tool

# Date bounds and visible article IDs belong to each forecast task.
_window_start: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_window_start", default=None,
)
_window_cutoff: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_window_cutoff", default=None,
)

_seen_articles: contextvars.ContextVar[
    dict[str, tuple[str, int]] | None
] = contextvars.ContextVar("rag_seen_articles", default=None)

# Normalize equivalent UUID renderings without admitting unseen article IDs.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def _canonical_article_id(raw: str) -> str:
    """Reduce any rendering of an article id to one stable catalog key."""
    s = (raw or "").strip().strip('"').strip("'").strip()
    s = s.removeprefix("id=").strip()
    s = s.strip("<>").strip().rstrip(",.;")
    s = s.removeprefix("urn:uuid:")
    match = _UUID_RE.search(s.lower())
    return match.group(0) if match else s.lower()


def _display_article_id(canonical: str) -> str:
    """Render a catalog key back in the form `search` shows to the agent."""
    return f"<urn:uuid:{canonical}>" if _UUID_RE.fullmatch(canonical) else canonical


def set_window(start_date: str | None, cutoff_date: str | None) -> contextvars.Context:
    """Set task-local date bounds and reset the article catalog for a new forecast step.

    Instance date bounds remain the fallback for unset task bounds.
    """
    _window_start.set(start_date)
    _window_cutoff.set(cutoff_date)

    _seen_articles.set({})
    return contextvars.copy_context()


logger = logging.getLogger(__name__)

# FORECAST_TOOL_TRACE: search, scrape, both comma-separated, or 1/all; default DEBUG.
_TOOL_TRACE = os.environ.get("FORECAST_TOOL_TRACE", "").strip().lower()
_TOOL_TRACE_ON = {t.strip() for t in _TOOL_TRACE.split(",") if t.strip()}


def _trace_level(tool_name: str) -> int:
    """INFO if this tool's per-call tracing is enabled, else DEBUG (suppressed)."""
    if _TOOL_TRACE_ON & {"1", "all"}:
        return logging.INFO
    return logging.INFO if tool_name in _TOOL_TRACE_ON else logging.DEBUG


_SEARCH_TRACE_LEVEL = _trace_level("search")
_SCRAPE_TRACE_LEVEL = _trace_level("scrape")

DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-8B"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_TOP_K = 5
DEFAULT_TEXT_TOKEN_BUDGET = 8000

# CPU searches local FAISS; GPU uses an external service with the same row ranges.
DEFAULT_SEARCH_BACKEND = os.environ.get("FORECAST_SEARCH_BACKEND", "cpu")
DEFAULT_SEARCH_ENDPOINT = os.environ.get("FORECAST_SEARCH_ENDPOINT", "http://localhost:8100")

# A weight of 1 keeps pure cosine ranking.
DEFAULT_DECAY_ALPHA = 1.0
DEFAULT_DECAY_HORIZON_DAYS = 180
DEFAULT_RERANK_OVERSAMPLE = 1

# Apply the asymmetric embedding instruction to queries only, not indexed documents.
QWEN3_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

TOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4")

# Keep loop-bound HTTP clients on a dedicated loop; resolve task state before dispatch.
_IO_LOOP_LOCK = threading.Lock()
_IO_LOOP: asyncio.AbstractEventLoop | None = None
_IO_LOOP_PID: int | None = None

_T = TypeVar("_T")


def _get_io_loop() -> asyncio.AbstractEventLoop:
    """Return the shared I/O loop, creating a new daemon thread after initialization or fork."""
    global _IO_LOOP, _IO_LOOP_PID
    pid = os.getpid()
    loop = _IO_LOOP
    if loop is None or loop.is_closed() or _IO_LOOP_PID != pid:
        with _IO_LOOP_LOCK:
            loop = _IO_LOOP
            if loop is None or loop.is_closed() or _IO_LOOP_PID != pid:
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, name="retriever-io", daemon=True
                ).start()
                _IO_LOOP = loop
                _IO_LOOP_PID = pid
    return loop


async def _io_call(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine on the I/O loop, preserving exception and cancellation propagation."""
    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coro, _get_io_loop())
    )

SEARCH_RESULT_TEMPLATE = """{rank}. {title} ({url})
[id={article_id}, published={published_date}, score={score:.4f}]
{snippet}"""

ARTICLE_TEMPLATE = """[id={article_id}, title={title!r}, url={url}, published={published_date}]

{text}"""

SEARCH_SNIPPET_MAX_CHARS = 280

_ISO_PREFIX_RE = re.compile(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def _parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO-like date prefix, returning None for missing or invalid dates."""
    if not value or not isinstance(value, str):
        return None
    m = _ISO_PREFIX_RE.match(value)
    if m is None:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))

        return date(y, mo, d)
    except ValueError:
        return None


@dataclass
class _DateShard:
    """Row-aligned FAISS and article metadata, with article bodies loaded on demand."""

    date: str

    # Remote GPU search does not load a local FAISS index.
    index: faiss.Index | None

    ntotal: int
    titles: list[str]
    article_ids: list[str]
    urls: list[str]
    published_dates: list[str]

    summaries: list[str]

    # Materialize article bodies only for selected rows.
    _texts: pa.ChunkedArray

    _texts_truncated: pa.ChunkedArray | None = None

    # Monthly shards map ISO dates to contiguous [lo, hi) ranges.
    day_offsets: dict[str, list[int]] | None = None

    def text_at(self, row: int) -> str:
        """Materialize the article body for a single row."""
        v = self._texts[row].as_py()
        return v if isinstance(v, str) else ""

    def truncated_at(self, row: int) -> str | None:
        """Precomputed scrape body for a row, or None if this shard lacks it."""
        if self._texts_truncated is None:
            return None
        v = self._texts_truncated[row].as_py()
        return v if isinstance(v, str) else ""

    def summary_at(self, row: int) -> str:
        """Return the publisher summary for a single row (`""` if absent)."""
        return self.summaries[row] or ""

    def __len__(self) -> int:
        return len(self._texts)


class RetrieverToolkit:
    """Search dated FAISS shards and serve their stored article text.

    Task-local date windows and article catalogs isolate concurrent forecasts.
    Shard loads are cached, and HTTP clients run on a shared I/O loop.
    """

    def __init__(
        self,
        embedding_endpoint: str,
        index_root: str | Path,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        text_token_budget: int = DEFAULT_TEXT_TOKEN_BUDGET,
        date_filter: list[str] | None = None,
        start_date: str | None = None,
        cutoff_date: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        concurrency: asyncio.Semaphore | int = DEFAULT_MAX_CONCURRENCY,
        decay_alpha: float = DEFAULT_DECAY_ALPHA,
        decay_horizon_days: int = DEFAULT_DECAY_HORIZON_DAYS,
        rerank_oversample: int = DEFAULT_RERANK_OVERSAMPLE,
        search_backend: str = DEFAULT_SEARCH_BACKEND,
        search_endpoint: str = DEFAULT_SEARCH_ENDPOINT,
    ):
        """Configure dated retrieval and embedding endpoints.

        embedding_endpoint accepts comma-separated replicas; concurrency limits their
        combined requests and its semaphore must only be used on the I/O loop.
        start_date and cutoff_date are inclusive ISO date bounds.
        decay_alpha=1 disables recency reranking; rerank_oversample scales its candidate pool.
        """
        self.embedding_endpoints = [e.strip() for e in embedding_endpoint.split(",") if e.strip()]
        if not self.embedding_endpoints:
            raise ValueError(f"embedding_endpoint has no usable URL: {embedding_endpoint!r}")

        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.index_root = Path(index_root)
        self.text_token_budget = text_token_budget
        self.date_filter = date_filter
        self.start_date = start_date
        self.cutoff_date = cutoff_date
        self.timeout = timeout
        self.semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)

        if not 0.0 <= decay_alpha <= 1.0:
            raise ValueError(f"decay_alpha must be in [0, 1]; got {decay_alpha}")
        if decay_horizon_days <= 0:
            raise ValueError(f"decay_horizon_days must be > 0; got {decay_horizon_days}")
        if rerank_oversample < 1:
            raise ValueError(f"rerank_oversample must be >= 1; got {rerank_oversample}")
        self.decay_alpha = float(decay_alpha)
        self.decay_horizon_days = int(decay_horizon_days)
        self.rerank_oversample = int(rerank_oversample)

        # Replicas share one concurrency limit and are selected round-robin.
        self._clients: list[AsyncOpenAI | None] = [None] * len(self.embedding_endpoints)
        self._rr = itertools.cycle(range(len(self.embedding_endpoints)))
        self._shards: dict[str, _DateShard] = {}

        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_outer = threading.Lock()

        self.search_backend = search_backend
        self.search_endpoint = search_endpoint.rstrip("/")
        if self.search_backend not in ("cpu", "gpu"):
            raise ValueError(f"search_backend must be 'cpu' or 'gpu'; got {search_backend!r}")
        self._search_client: httpx.AsyncClient | None = None

    def _get_client(self, idx: int) -> AsyncOpenAI:
        """Get or lazily create the embedding HTTP client for endpoint `idx`."""
        client = self._clients[idx]
        if client is None:
            client = AsyncOpenAI(
                base_url=self.embedding_endpoints[idx],
                api_key="EMPTY",
                timeout=self.timeout,
            )
            self._clients[idx] = client
        return client

    def _next_client(self) -> AsyncOpenAI:
        """Pick the next embedding client, round-robin across all replicas."""
        return self._get_client(next(self._rr))

    def _active_window(self) -> tuple[str | None, str | None]:
        """Resolve task-local date bounds, falling back to the instance defaults."""
        s = _window_start.get()
        c = _window_cutoff.get()
        if s is None:
            s = self.start_date
        if c is None:
            c = self.cutoff_date
        return s, c

    def _list_dates(self) -> list[str]:
        """Resolve the set of date subdirs, dropping shards fully outside `[start_date, cutoff_date]`."""
        if self.date_filter is not None:
            dates = list(self.date_filter)
        else:
            dates = sorted(
                d.name for d in self.index_root.iterdir() if d.is_dir() and not d.name.startswith("_")
            )
        start, cutoff = self._active_window()

        # Prefix comparison supports both daily and monthly shard names.
        if cutoff is not None:
            dates = [d for d in dates if d <= cutoff[: len(d)]]
        if start is not None:
            dates = [d for d in dates if d >= start[: len(d)]]
        return dates

    def _get_shard_lock(self, date: str) -> threading.Lock:
        """Get a per-shard lock; different shards can load concurrently."""
        with self._load_locks_outer:
            lock = self._load_locks.get(date)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[date] = lock
            return lock

    def _scrape_precompute_ok(self, date_dir: Path) -> bool:
        """Check that cached article truncation matches the active tokenizer and token budget."""
        meta_path = date_dir / "metadata.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        prov = meta.get("scrape_truncation")
        if not isinstance(prov, dict):
            return False
        ok = (prov.get("tokenizer") == TOKEN_ENCODING.name
              and prov.get("token_budget") == self.text_token_budget)
        if not ok:
            logger.warning(
                "[rag_retriever] %s: text_truncated provenance %s != expected "
                "(tokenizer=%s budget=%d); using live truncation",
                date_dir.name, prov, TOKEN_ENCODING.name, self.text_token_budget,
            )
        return ok

    @staticmethod
    def _read_day_offsets(date_dir: Path) -> dict[str, list[int]] | None:
        """Read the monthly date-to-[lo, hi) row map, or None when absent or invalid."""
        meta_path = date_dir / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        offs = meta.get("day_offsets")
        return offs if isinstance(offs, dict) and offs else None

    def _load_shard(self, date: str) -> _DateShard | None:
        """Load one date's FAISS index + parquet sidecar; cached per process."""
        if date in self._shards:
            return self._shards[date]
        with self._get_shard_lock(date):
            if date in self._shards:
                return self._shards[date]
            date_dir = self.index_root / date
            index_path = date_dir / "faiss.index"
            parquet_path = date_dir / "texts.parquet"

            gpu_backend = (self.search_backend == "gpu")
            if not parquet_path.exists() or (not gpu_backend and not index_path.exists()):
                logger.warning("[rag_retriever] missing index or parquet for date %s", date)
                return None

            index = (
                None if gpu_backend
                else faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            )

            # Keep large text columns as Arrow arrays; eagerly load compact metadata.
            available = set(pq.read_schema(parquet_path).names)
            cols = ["title", "text", "article_id", "url"]
            if "published_date" in available:
                cols.append("published_date")

            if "summary" in available:
                cols.append("summary")

            # Cached truncation is valid only for the same tokenizer and budget.
            use_precomputed = "text_truncated" in available and self._scrape_precompute_ok(date_dir)
            if use_precomputed:
                cols.append("text_truncated")
            table = pq.read_table(parquet_path, columns=cols)
            n_rows = table.num_rows
            published = (
                table.column("published_date").to_pylist()
                if "published_date" in available
                else [""] * n_rows
            )
            summaries = (
                table.column("summary").to_pylist()
                if "summary" in available
                else [""] * n_rows
            )

            day_offsets = self._read_day_offsets(date_dir)

            ntotal = n_rows if gpu_backend else int(index.ntotal)
            shard = _DateShard(
                date=date,
                index=index,
                ntotal=ntotal,
                titles=table.column("title").to_pylist(),
                article_ids=table.column("article_id").to_pylist(),
                urls=table.column("url").to_pylist(),
                published_dates=published,
                summaries=summaries,
                _texts=table.column("text"),
                _texts_truncated=table.column("text_truncated") if use_precomputed else None,
                day_offsets=day_offsets,
            )

            # FAISS IDs must stay aligned with parquet rows.
            if index is not None and index.ntotal != len(shard):
                logger.error(
                    "[rag_retriever] %s: faiss ntotal=%d != parquet rows=%d (alignment broken)",
                    date,
                    index.ntotal,
                    len(shard),
                )
                return None
            self._shards[date] = shard
            logger.info("[rag_retriever] loaded shard %s (%d docs, index=%s)",
                        date, ntotal, "cpu" if index is not None else "gpu-remote")
            return shard

    async def prewarm(self, dates: list[str] | None = None) -> int:
        """Load the requested shards concurrently, defaulting to the active date window.

        Return the number successfully loaded, including already-cached shards.
        """
        target = dates if dates is not None else self._list_dates()
        if not target:
            return 0
        results = await asyncio.gather(
            *(asyncio.to_thread(self._load_shard, d) for d in target)
        )
        return sum(1 for s in results if s is not None)

    async def cleanup(self) -> None:
        """Close HTTP clients on their owning I/O loop; shard caches remain until process exit."""
        await _io_call(self._cleanup_io())

    async def _cleanup_io(self) -> None:
        """Close loop-bound embedding and search clients."""
        for i, client in enumerate(self._clients):
            if client is not None:
                await client.close()
                self._clients[i] = None
        if self._search_client is not None:
            await self._search_client.aclose()
            self._search_client = None

    async def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query as an L2-normalized float32 vector of shape (1, dim)."""
        prompt = QWEN3_QUERY_PREFIX + query
        return await _io_call(self._embed_query_io(prompt))

    async def _embed_query_io(self, prompt: str) -> np.ndarray:
        """Embed and normalize on the I/O loop, using the next replica and shared semaphore."""
        async with self.semaphore:
            resp = await self._next_client().embeddings.create(
                model=self.embedding_model,
                input=prompt,
            )
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.reshape(1, -1)

    @staticmethod
    def _search_shard_sync(
        shard: _DateShard,
        qvec: np.ndarray,
        top_k: int,
        selector: faiss.IDSelector | None = None,
    ) -> list[tuple[float, str, int]]:
        """Return descending (score, shard date, row) hits from the selected FAISS rows."""
        if selector is not None:
            params = faiss.SearchParameters(sel=selector)
            scores, ids = shard.index.search(qvec, top_k, params=params)
        else:
            scores, ids = shard.index.search(qvec, top_k)
        out: list[tuple[float, str, int]] = []
        for score, row in zip(scores[0].tolist(), ids[0].tolist()):
            if row < 0:
                continue
            out.append((float(score), shard.date, int(row)))
        return out

    def _in_window_ids(self, shard: _DateShard) -> np.ndarray:
        """Row IDs in this shard whose `published_date` falls in the active window."""
        lo, hi = self._active_window()
        return np.asarray(
            [
                i for i, pd in enumerate(shard.published_dates)
                if pd and (lo is None or pd >= lo) and (hi is None or pd <= hi)
            ],
            dtype=np.int64,
        )

    def _rerank_by_recency(
        self,
        hits: list[tuple[float, str, int]],
        cutoff_date: str | None,
    ) -> list[tuple[float, str, int]]:
        """Apply alpha * similarity + (1 - alpha) * recency to each hit.

        Recency decreases linearly to zero over decay_horizon_days; unknown dates get zero.
        Without a valid cutoff, or with alpha=1, return hits unchanged. Results are unsorted.
        """
        if self.decay_alpha >= 1.0 or cutoff_date is None or not hits:
            return hits
        cutoff = _parse_iso_date(cutoff_date)
        if cutoff is None:
            logger.debug("[rag_retriever] rerank skipped: cutoff_date %r unparseable",
                         cutoff_date)
            return hits

        alpha = self.decay_alpha
        horizon = self.decay_horizon_days
        out: list[tuple[float, str, int]] = []
        for sim, shard_date, row in hits:
            shard = self._shards.get(shard_date)
            if shard is None:
                recency = 0.0
            else:
                pub = _parse_iso_date(shard.published_dates[row])
                if pub is None:
                    recency = 0.0
                else:
                    # Clamp future dates so recency cannot exceed one.
                    age = max(0, (cutoff - pub).days)
                    recency = 1.0 - min(age / horizon, 1.0)
            final = alpha * sim + (1.0 - alpha) * recency
            out.append((final, shard_date, row))
        return out

    def _window_row_range(self, shard: _DateShard) -> tuple[int, int]:
        """Return the shared CPU/GPU [lo, hi) date filter from sorted monthly offsets.

        Shards without offsets are unrestricted. Return (0, 0) if no day is in range.
        """
        n = shard.ntotal
        offs = shard.day_offsets
        if not offs:
            return (0, n)
        start, cutoff = self._active_window()
        if start is None and cutoff is None:
            return (0, n)
        lo = hi = None
        for day, (d_lo, d_hi) in offs.items():
            if start is not None and day < start[:10]:
                continue
            if cutoff is not None and day > cutoff[:10]:
                continue
            lo = d_lo if lo is None or d_lo < lo else lo
            hi = d_hi if hi is None or d_hi > hi else hi
        if lo is None:
            return (0, 0)
        return (lo, hi)

    def _shard_selector(self, shard: _DateShard) -> faiss.IDSelector | None:
        """Build an IDSelectorRange, or None when the entire shard is in range."""
        lo, hi = self._window_row_range(shard)
        if lo == 0 and hi == shard.ntotal:
            return None
        return faiss.IDSelectorRange(lo, hi)

    async def _search_all(self, query: str, top_k: int) -> list[tuple[float, str, int]]:
        """Search in-window shard rows and return the global top-k hits.

        When recency reranking is enabled, rerank the global cosine candidate pool
        before selecting the final top-k.
        """
        qvec = await self._embed_query(query)

        dates = self._list_dates()
        shards = [s for s in (self._load_shard(d) for d in dates) if s is not None]
        if not shards:
            return []

        _, win_cutoff = self._active_window()

        rerank_active = self.decay_alpha < 1.0
        pool_top_k = top_k * (self.rerank_oversample if rerank_active else 1)

        if self.search_backend == "gpu":
            flat = await self._search_all_gpu(qvec, shards, pool_top_k)
        else:
            def _per_shard(shard: _DateShard) -> list[tuple[float, str, int]]:
                return self._search_shard_sync(shard, qvec, pool_top_k, self._shard_selector(shard))

            per_shard = await asyncio.gather(
                *(asyncio.to_thread(_per_shard, s) for s in shards)
            )
            flat = [hit for hits in per_shard for hit in hits]
        if not flat:
            return []

        # Select the global cosine candidate pool before applying recency scores.
        pool = heapq.nlargest(pool_top_k, flat, key=lambda x: x[0])
        if rerank_active:
            pool = self._rerank_by_recency(pool, win_cutoff)
            pool.sort(key=lambda x: x[0], reverse=True)
        return pool[:top_k]

    async def _search_all_gpu(
        self, qvec: np.ndarray, shards: list[_DateShard], pool_top_k: int
    ) -> list[tuple[float, str, int]]:
        """Send the query and shared CPU/GPU date ranges to the search service."""
        # Resolve task-local date ranges before handing the payload to the I/O loop.
        specs = []
        for s in shards:
            lo, hi = self._window_row_range(s)
            if hi <= lo:
                continue
            specs.append({"month": s.date, "lo": int(lo), "hi": int(hi)})
        if not specs:
            return []
        payload = {"query": qvec[0].tolist(), "shards": specs, "top_k": pool_top_k}
        return await _io_call(self._search_gpu_post_io(payload))

    async def _search_gpu_post_io(self, payload: dict) -> list[tuple[float, str, int]]:
        """Post a prepared search payload and decode hits on the I/O loop."""
        if self._search_client is None:
            self._search_client = httpx.AsyncClient(timeout=self.timeout)
        resp = await self._search_client.post(
            f"{self.search_endpoint}/search", json=payload,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])

        return [(float(score), month, int(row)) for score, month, row in hits]

    def _truncate(self, text: str) -> str:
        """Truncate `text` to `text_token_budget` tokens via tiktoken."""
        if not text:
            return ""
        tokens = TOKEN_ENCODING.encode(text, allowed_special="all")
        if len(tokens) > self.text_token_budget:
            return TOKEN_ENCODING.decode(tokens[: self.text_token_budget]) + "..."
        return text

    def _snippet(self, shard: _DateShard, row: int) -> str:
        """Use the publisher summary when available, otherwise a shortened article lede."""
        desc = shard.summary_at(row).strip()
        if desc:
            desc = " ".join(desc.split())
            if len(desc) > SEARCH_SNIPPET_MAX_CHARS:
                return desc[:SEARCH_SNIPPET_MAX_CHARS].rstrip() + "…"
            return desc

        body = (shard.text_at(row) or "").strip()
        if not body:
            return ""
        body = " ".join(body.split())
        if len(body) <= SEARCH_SNIPPET_MAX_CHARS:
            return body

        # Prefer a sentence boundary in the latter half of the snippet.
        snippet = body[:SEARCH_SNIPPET_MAX_CHARS]
        for sep in (". ", "! ", "? "):
            idx = snippet.rfind(sep)
            if idx >= SEARCH_SNIPPET_MAX_CHARS // 2:
                return snippet[: idx + 1]
        return snippet.rstrip() + "…"

    def _format_search(self, hits: list[tuple[float, str, int]]) -> str:
        """Render compact search hits and register their IDs in the task-local catalog."""
        if not hits:
            return "No results found."

        seen = _seen_articles.get()
        if seen is None:
            seen = {}
            _seen_articles.set(seen)

        lines: list[str] = []
        for rank, (score, date, row) in enumerate(hits, 1):
            shard = self._shards[date]
            article_id = shard.article_ids[row] or ""

            if article_id:
                # Scrape may resolve only IDs surfaced in this task.
                seen[_canonical_article_id(article_id)] = (date, row)
            lines.append(
                SEARCH_RESULT_TEMPLATE.format(
                    rank=rank,
                    title=shard.titles[row] or "(no title)",
                    url=shard.urls[row] or "(no url)",
                    score=score,
                    published_date=shard.published_dates[row] or "(no date)",
                    article_id=article_id or "(no id)",
                    snippet=self._snippet(shard, row),
                )
            )
        return "\n\n".join(lines)

    def _format_article(self, shard: _DateShard, row: int) -> str:
        """Render the article body, using compatible cached truncation when available."""
        body = shard.truncated_at(row)
        if body is None:
            body = self._truncate(shard.text_at(row))
        return ARTICLE_TEMPLATE.format(
            article_id=shard.article_ids[row] or "(no id)",
            title=shard.titles[row] or "(no title)",
            url=shard.urls[row] or "(no url)",
            published_date=shard.published_dates[row] or "(no date)",
            text=body,
        )

    def _published_in_window(self, shard: _DateShard, row: int) -> bool:
        """Recheck the active date bounds before serving a previously retrieved article."""
        start, cutoff = self._active_window()
        if start is None and cutoff is None:
            return True
        pub = (shard.published_dates[row] or "")
        if not pub:
            return False
        if start is not None and pub < start:
            return False
        if cutoff is not None and pub > cutoff:
            return False
        return True

    @tool
    async def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Search the local news corpus for articles relevant to a query.

        Returns a numbered list of search hits. Each hit shows:
          - id          article identifier (use this with `scrape`)
          - title, url, published (date), score
          - snippet     a short triage preview (~280 chars). To read the
                        full article body, call `scrape(article_id)`.

        Use this for fast triage across many candidates; only call
        `scrape` on the ones whose snippets actually look relevant.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return (across all date shards).
        """
        logger.log(_SEARCH_TRACE_LEVEL, "[rag_retriever.search] query=%s, top_k=%s", query, top_k)
        try:
            hits = await self._search_all(query, top_k)
            return self._format_search(hits)
        except Exception as e:
            logger.error("[rag_retriever.search] error: %s", e)
            return f"Search failed: {e}."

    @tool
    async def scrape(self, article_id: str) -> str:
        """Fetch the full body text of an article previously surfaced by search.

        Pass an `article_id` value from a prior `search` result. Returns
        the article's title, URL, publication date, and the
        (token-truncated) full body text. Use this when a search snippet
        suggests an article is relevant and you need the full content.

        Constraints:
          - You can only read article_ids that appeared in your own prior
            `search` results in this session. Unknown ids return an error.
          - Articles outside the search window (post-resolution) are
            blocked even if known.

        Args:
            article_id: The `id` value from a prior search hit, e.g.
                "<urn:uuid:...>".
        """
        logger.log(_SCRAPE_TRACE_LEVEL, "[rag_retriever.scrape] id=%s", article_id)
        aid = _canonical_article_id(article_id)
        if not aid:
            return "scrape failed: empty article_id."

        seen = _seen_articles.get()
        if not seen:
            return ("scrape failed: no articles have been surfaced "
                    "yet in this session. Call `search` first, then pass "
                    "an `id` value from one of its results.")
        # Unknown IDs and out-of-window articles produce distinct errors.
        loc = seen.get(aid)
        if loc is None:
            sample = ", ".join(_display_article_id(k) for k in list(seen)[:3])
            return (f"scrape failed: no article with id {article_id!r} has been "
                    "returned to you. You may only read ids that appeared in "
                    "your own `search` results — do not construct or guess ids. "
                    f"Ids available from your searches so far: {sample}.")

        shard_name, row = loc
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = self._load_shard(shard_name)
            if shard is None:
                return (f"scrape failed: source shard {shard_name!r} "
                        "is no longer loadable.")
        if not self._published_in_window(shard, row):
            return (f"scrape failed: article_id={article_id!r} is outside "
                    "the active retrieval window.")
        try:
            body = self._format_article(shard, row)
        except Exception as e:
            logger.error("[rag_retriever.scrape] error: %s", e)
            return f"scrape failed: {e}."

        # Resolve formatting differences while showing the agent the canonical ID.
        canonical = shard.article_ids[row] or ""
        if canonical and (article_id or "").strip() != canonical:
            body = (f"[note: id resolved, but you passed {(article_id or '').strip()!r}. "
                    f"The canonical form is {canonical!r} — copy it verbatim from "
                    "the search result next time.]\n\n" + body)
        return body
