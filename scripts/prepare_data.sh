#!/usr/bin/env bash
# Prepare datasets from externally supplied Forecast-Dojo questions.
# Install the AlphaBet package before running this launcher.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/prepare_data.sh memory-free --input QUESTIONS --output DATASET --auto-k [options]
  bash scripts/prepare_data.sh memory-on --input QUESTIONS --output DATASET --auto-k [options]
  bash scripts/prepare_data.sh single-date --input MEMORY_FREE_DATASET --out DATASET [--seed SEED]

memory-free: one question-date pair per row.
memory-on: one dated event episode per row.
single-date: select one fixed forecast date per event.

Pass --help after a mode for the preparation module's full options.
Data, corpus indexes, and generated outputs must be supplied separately.
USAGE
}

mode="${1:-}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
    usage
    exit 0
fi
if [[ -z "$mode" ]]; then
    usage >&2
    exit 2
fi
shift

case "$mode" in
    memory-free|memory-on)
        exec "${PYTHON:-python3}" -m data.build_dataset --mode "$mode" "$@"
        ;;
    single-date)
        exec "${PYTHON:-python3}" -m data.build_single_date "$@"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
