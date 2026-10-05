#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" -m evaluation.run "$@"
