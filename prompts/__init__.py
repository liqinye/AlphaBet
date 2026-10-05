"""Shared system-prompt paths for forecasting training and evaluation."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def prompt_filename(
    mode: str, *, enable_code: bool = True, search: bool = True
) -> str:
    """Select a prompt matching the forecast mode and available tools."""
    names = {"memory-on": "memory-on", "memory-free": "memory-free"}
    if mode not in names:
        raise ValueError(f"Unknown forecast mode: {mode!r}")
    suffix = "-no-tools" if not search else "-no-code" if not enable_code else ""
    return f"{names[mode]}{suffix}.md"


def get_prompt_path(
    mode: str, *, enable_code: bool = True, search: bool = True
) -> Path:
    """Return the packaged prompt path for one forecast configuration."""
    return PROMPTS_DIR / prompt_filename(mode, enable_code=enable_code, search=search)
