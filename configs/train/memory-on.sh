#!/usr/bin/env bash
# Memory-on training: one input per event, with chronological forecast dates.
export FORECAST_MODE=memory-on
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"

