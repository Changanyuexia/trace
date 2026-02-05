#!/usr/bin/env bash
set -euo pipefail

# Run TRACE variant on a single Defects4J bug.
# Usage: ./scripts/run_one_defects4j.sh Chart 1
# Requires: TRACE_WORK_ROOT in .env (workdir/index/meta/logs come from config).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PID="${1:-Chart}"
BID="${2:-1}"

WORK_ROOT="${TRACE_WORK_ROOT:-/tmp/trace_work}"
echo "[INFO] Running TRACE on ${PID}-${BID}b"
echo "[INFO] TRACE_WORK_ROOT: ${WORK_ROOT} (workdir/index from config)"

python run_trace.py \
  --dataset defects4j \
  --pid "${PID}" \
  --bid "${BID}" \
  --variant TRACE \
  --model example

