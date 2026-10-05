#!/usr/bin/env bash
# Memory-free multi-date training: one input per (question, forecast date).
export FORECAST_MODE=memory-free
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"

