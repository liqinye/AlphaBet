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

"""Filter reserved placeholder legs from Polymarket multi-leg questions.

A candidate must match a reserved name pattern and have non-positive lifetime
notional volume. Questions whose resolved leg matches are rejected. Other
multi-leg questions retain at least min_real_legs after filtering.
"""

from __future__ import annotations

import re

# Reserved codes include A, AA, 12, and A1.
_SLOT_CODE = r"(?:[A-Z]{1,2}|\d{1,2}|[A-Z]\d{1,2})"

# Whole-label slots, including compound names such as "Song E by Artist E".
PLACEHOLDER_LABEL_RE = re.compile(
    rf"^\s*[A-Z][a-z]+\s+{_SLOT_CODE}"
    rf"(?:\s+by\s+[A-Z][a-z]+\s+{_SLOT_CODE})?"
    r"\s*[\?.!]*\s*$",
)
PLACEHOLDER_LITERAL_RE = re.compile(r"^\s*placeholder\b.*$", re.IGNORECASE)
# Embedded slots use a fixed noun list to avoid matching real names in questions.
PLACEHOLDER_EMBEDDED_RE = re.compile(
    rf"\b(?:Player|Company|Candidate|Team|Option|App|Person|Party|Song|Movie|"
    rf"Show|Album|Artist|Country|Driver|Constructor|Game|Nominee|Contestant|"
    rf"Entrant|Placeholder)\s+{_SLOT_CODE}\b",
)
OTHER_LABEL_RE = re.compile(r"^\s*(?:Other|None(?:\s+of\s+the\s+above)?)\s*$", re.IGNORECASE)


def looks_like_placeholder_label(label: str | None) -> bool:
    """Match reserved slot names, literal placeholders, and Other/None labels.

    This name check requires the volume gate in is_placeholder_market before
    a leg can be removed.
    """
    if not label:
        return False
    return bool(
        PLACEHOLDER_LABEL_RE.match(label)
        or PLACEHOLDER_LITERAL_RE.match(label)
        or PLACEHOLDER_EMBEDDED_RE.search(label)
        or OTHER_LABEL_RE.match(label)
    )


def market_lifetime_notional(market: dict, q: dict) -> float:
    """Total notional traded on this leg, summed over its full history."""
    market_id = market.get("market_id")
    if not market_id:
        return 0.0
    days = (q.get("daily_volume_by_market") or {}).get(market_id) or []
    total = 0.0
    for d in days:
        v = d.get("notional") if d else None
        if isinstance(v, (int, float)):
            total += float(v)
    return total


def is_placeholder_market(market: dict, q: dict) -> bool:
    """Check for a placeholder name with non-positive lifetime notional volume."""
    if not looks_like_placeholder_label(market.get("label")):
        return False
    return market_lifetime_notional(market, q) <= 0.0


def filter_question(q: dict, *, min_real_legs: int = 2) -> dict | None:
    """Remove placeholder legs, or return None for an invalid question.

    Reject a placeholder resolved leg or fewer than min_real_legs remaining.
    Only markets and num_markets change; histories stay intact. Unchanged
    questions are returned without copying.
    """
    if q.get("belief_kind") != "multi_neg_risk":
        return q

    truth = (q.get("resolved_label") or "").strip()
    markets = q.get("markets") or []

    truth_market = next(
        (m for m in markets if (m.get("label") or "").strip() == truth),
        None,
    )
    if truth_market is not None and is_placeholder_market(truth_market, q):
        return None

    keep = [m for m in markets if not is_placeholder_market(m, q)]
    if len(keep) < min_real_legs:
        return None
    if len(keep) == len(markets):
        return q

    cleaned = dict(q)
    cleaned["markets"] = keep
    cleaned["num_markets"] = len(keep)
    return cleaned
