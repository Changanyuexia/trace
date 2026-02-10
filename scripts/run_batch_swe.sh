#!/usr/bin/env bash
set -euo pipefail

# Run TRACE variant on a list of SWE-bench Verified instances.
# Usage example:
#   ./scripts/run_batch_swe.sh dataset/swebench_verified_bugs.txt
#
# List file contains one instance_id per line, e.g.:
#   astropy__astropy-12907
#   sympy__sympy-13647

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LIST_FILE="${1:-dataset/swebench_verified_bugs.txt}"

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "[ERROR] Instance list file not found: ${LIST_FILE}" >&2
  exit 1
fi

echo "[INFO] TRACE_WORK_ROOT: ${TRACE_WORK_ROOT:-/tmp/trace_work} (workdir/index from config)"

while read -r line; do
  PID=$(echo "${line}" | sed 's/[[:space:]].*//')
  [[ -z "${PID}" ]] && continue
  echo "=== Running ${PID} ==="
  python run_trace.py \
    --dataset swebench_verified \
    --pid "${PID}" \
    --bid 0 \
    --variant TRACE \
    --model example
done < "${LIST_FILE}"
