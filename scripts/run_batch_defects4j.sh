#!/usr/bin/env bash
set -euo pipefail

# Run TRACE variant on a list of Defects4J bugs.
# Usage example:
#   ./scripts/run_batch_defects4j.sh bugs.txt
#
# where bugs.txt contains lines like:
#   Chart 1
#   Lang 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LIST_FILE="${1:-bugs.txt}"

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "[ERROR] Bug list file not found: ${LIST_FILE}" >&2
  exit 1
fi

echo "[INFO] TRACE_WORK_ROOT: ${TRACE_WORK_ROOT:-/tmp/trace_work} (workdir/index from config)"

while read -r PID BID; do
  [[ -z "${PID}" || -z "${BID}" ]] && continue
  echo "=== Running ${PID}-${BID}b ==="
  python run_trace.py \
    --dataset defects4j \
    --pid "${PID}" \
    --bid "${BID}" \
    --variant TRACE \
    --model example
done < "${LIST_FILE}"

