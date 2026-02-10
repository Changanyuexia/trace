#!/usr/bin/env bash
set -euo pipefail

# Run TRACE variant on a single SWE-bench Verified instance.
# Usage: ./scripts/run_one_swe.sh [instance_id]
# Example: ./scripts/run_one_swe.sh sympy__sympy-13647
# Requires: TRACE_WORK_ROOT in .env (workdir/index/meta/logs from config).
# For SWE-bench, bid is always 0 (pid = instance_id).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PID="${1:-astropy__astropy-12907}"
BID="${2:-0}"

WORK_ROOT="${TRACE_WORK_ROOT:-/tmp/trace_work}"
echo "[INFO] Running TRACE on SWE-bench instance ${PID}"
echo "[INFO] TRACE_WORK_ROOT: ${WORK_ROOT} (workdir/index from config)"

python run_trace.py \
  --dataset swebench_verified \
  --pid "${PID}" \
  --bid "${BID}" \
  --variant TRACE \
  --model example
