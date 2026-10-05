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

"""Aggregate per-step benchmark records into Brier, log-loss, accuracy, and Info-Alpha.

Exclude error rows and unparsed forecasts from metric means; count errors
separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def score_run(path: str | Path) -> dict[str, Any]:
    """Compute a scorecard, reporting log_loss as -log_prob.

    Info-Alpha averages only steps with a crowd baseline.
    """
    rows = _read_rows(Path(path))
    scored = [r for r in rows if "error" not in r and r.get("parse_ok")]

    brier = [r["brier"] for r in scored if _num(r.get("brier"))]
    log_prob = [r["log_prob"] for r in scored if _num(r.get("log_prob"))]
    accuracy = [r["accuracy"] for r in scored if _num(r.get("accuracy"))]
    info_alpha = [r["info_alpha"] for r in scored if _num(r.get("info_alpha"))]

    return {
        "run_id": rows[0].get("run_id") if rows else None,
        "model_name": rows[0].get("model_name") if rows else None,
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_errors": sum(1 for r in rows if "error" in r),
        "brier": _mean(brier),
        "log_loss": (-_mean(log_prob)) if log_prob else None,
        "accuracy": _mean(accuracy),
        "info_alpha": _mean(info_alpha),
        "n_info_alpha": len(info_alpha),
    }


def format_scorecard(card: dict[str, Any]) -> str:
    """Render a scorecard dict as an aligned, human-readable block."""
    keys = ["n_rows", "n_scored", "n_errors", "brier", "log_loss", "accuracy", "info_alpha"]
    lines = [f"── {card.get('run_id') or 'run'} ──"]
    for k in keys:
        v = card.get(k)
        lines.append(f"  {k:<12} {v:.4f}" if isinstance(v, float) else f"  {k:<12} {v}")
    return "\n".join(lines)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, skipping blank and malformed lines."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean(xs: list[float]) -> float | None:
    return fmean(xs) if xs else None
