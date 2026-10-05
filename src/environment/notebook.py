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

"""Parse forecast answers and belief notebooks.

Extractors use the last complete tag. Detailed results retain format
flags so rewards can distinguish valid output from lenient repairs.
Notebook carry-forward preserves the raw text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_ANSWER_TAG = "answer"
_NOTEBOOK_TAG = "belief_notebook"


def _extract_last_tag_inner(text: str, tag: str) -> str | None:
    """Return the last complete, case-sensitive tag block's inner text.

    Excluding nested openers prevents an earlier unclosed tag in reasoning
    from swallowing the final block.
    """
    if not text:
        return None
    matches = re.findall(rf"<{tag}>((?:(?!<{tag}>).)*?)</{tag}>", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _loads_lenient(blob: str) -> tuple[dict | None, bool]:
    """Return `(object, strict_json)` after parsing a JSON object.

    Fall back to quoting bare keys and then Python literal parsing. Repaired
    objects have `strict_json=False`; non-objects are rejected.
    """
    blob = blob.strip()
    if not blob:
        return None, False
    try:
        result = json.loads(blob)
        return (result, True) if isinstance(result, dict) else (None, False)
    except json.JSONDecodeError:
        pass
    # Repaired objects remain non-strict for format scoring.
    coerced = re.sub(r"([{,]\s*)([A-Za-z_][\w\- ]*?)(\s*:)", r'\1"\2"\3', blob)
    try:
        result = json.loads(coerced)
        if isinstance(result, dict):
            return result, False
    except json.JSONDecodeError:
        pass
    import ast

    for candidate in (blob, coerced):
        try:
            result = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(result, dict):
            return {str(k): v for k, v in result.items()}, False
    return None, False


@dataclass
class AnswerParse:
    """Forecast probabilities and format flags before normalization.

    `strict_json` excludes repaired parses; `all_numeric` requires a
    non-empty object with no values dropped.
    """

    forecast: dict[str, float] | None
    tag_present: bool
    strict_json: bool
    all_numeric: bool


@dataclass
class NotebookParse:
    """Notebook text and structural flags used by the reward.

    `parseable` accepts repaired objects. `has_assessment_p` requires a
    non-empty numeric distribution, retained in `p` for answer comparison.
    """

    raw_inner: str | None
    tag_present: bool
    parseable: bool
    has_assessment_p: bool
    p: dict[str, float] | None = field(default=None)


def _numeric_dict(d: dict) -> dict[str, float]:
    """Keep only numeric (int/float, non-bool) values, stringifying keys."""
    return {
        str(k): float(v)
        for k, v in d.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def extract_answer_detailed(text: str | None) -> AnswerParse:
    """Extract answer probabilities and format flags before normalization."""
    inner = _extract_last_tag_inner(text or "", _ANSWER_TAG)
    if inner is None:
        return AnswerParse(forecast=None, tag_present=False, strict_json=False, all_numeric=False)
    parsed, strict_ok = _loads_lenient(inner)
    if parsed is None:
        return AnswerParse(forecast=None, tag_present=True, strict_json=strict_ok, all_numeric=False)
    numeric = _numeric_dict(parsed)
    all_numeric = len(numeric) == len(parsed) and len(parsed) > 0
    return AnswerParse(
        forecast=numeric or None,
        tag_present=True,
        strict_json=strict_ok,
        all_numeric=all_numeric,
    )


def extract_notebook_detailed(text: str | None) -> NotebookParse:
    """Inspect notebook structure and `assessment.p`, preserving the raw text."""
    inner = _extract_last_tag_inner(text or "", _NOTEBOOK_TAG)
    if inner is None:
        return NotebookParse(raw_inner=None, tag_present=False, parseable=False, has_assessment_p=False)
    parsed, _strict = _loads_lenient(inner)
    if parsed is None:
        return NotebookParse(raw_inner=inner, tag_present=True, parseable=False, has_assessment_p=False)
    assessment = parsed.get("assessment")
    p_raw = assessment.get("p") if isinstance(assessment, dict) else None
    if isinstance(p_raw, dict):
        p_numeric = _numeric_dict(p_raw)
        has_p = len(p_numeric) == len(p_raw) and len(p_numeric) > 0
        return NotebookParse(
            raw_inner=inner,
            tag_present=True,
            parseable=True,
            has_assessment_p=has_p,
            p=p_numeric or None,
        )
    return NotebookParse(raw_inner=inner, tag_present=True, parseable=True, has_assessment_p=False)


def extract_answer(text: str | None) -> dict[str, float] | None:
    """Return numeric probabilities from the last answer tag, or None.

    Label order is preserved. Values are not normalized, and non-numeric
    entries are dropped.
    """
    return extract_answer_detailed(text).forecast


def extract_belief_notebook(text: str | None) -> str | None:
    """Return raw inner text from the last notebook tag, or None.

    Carry malformed text forward unchanged so the model can repair it;
    validation and penalties belong to the reward.
    """
    return _extract_last_tag_inner(text or "", _NOTEBOOK_TAG)
