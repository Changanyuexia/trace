#!/usr/bin/env bash
set -euo pipefail

# Build TRACE retrieval index for a single instance.
# Supports:
#   - Defects4J bug:   --dataset d4j  --pid PID --bid BID [--workdir DIR]
#   - SWE-bench bug:   --dataset swe  --instance-id ID --workdir DIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET=""
PID=""
BID=""
VF="b"
INSTANCE_ID=""
WORKDIR=""

usage() {
  cat <<EOF
Usage:
  bash bin/build_index.sh --dataset d4j --pid PID --bid BID [--vf b|f] [--workdir DIR]
  bash bin/build_index.sh --dataset swe --instance-id ID --workdir DIR

Examples:
  # Defects4J (Chart-1b); workdir/index under TRACE_WORK_ROOT:
  TRACE_WORK_ROOT=/tmp/trace_work bash bin/build_index.sh --dataset d4j --pid Chart --bid 1

  # SWE-bench, existing workdir:
  bash bin/build_index.sh --dataset swe --instance-id sympy__sympy-20590 --workdir /path/to/workdir
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2;;
    --pid) PID="${2:-}"; shift 2;;
    --bid) BID="${2:-}"; shift 2;;
    --vf) VF="${2:-b}"; shift 2;;
    --instance-id) INSTANCE_ID="${2:-}"; shift 2;;
    --workdir) WORKDIR="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ "$DATASET" != "d4j" ] && [ "$DATASET" != "swe" ]; then
  echo "[ERROR] --dataset must be d4j or swe" >&2
  usage
  exit 2
fi

if [ "$DATASET" = "d4j" ]; then
  if [ -z "$PID" ] || [ -z "$BID" ]; then
    echo "[ERROR] --pid and --bid are required for dataset d4j" >&2
    usage
    exit 2
  fi

  WORK_ROOT="${TRACE_WORK_ROOT:-/tmp/trace_work}"
  if [ -z "$WORKDIR" ]; then
    BUG_ID="${PID}-${BID}b"
    WORKDIR="${WORK_ROOT}/workdirs/defects4j/${BUG_ID}"
  fi

  INDEX_DIR="${WORK_ROOT}/defects4j_index"
  mkdir -p "${INDEX_DIR}"

  echo "[INFO] dataset=d4j pid=${PID} bid=${BID} vf=${VF}"
  echo "[INFO] workdir=${WORKDIR}"
  echo "[INFO] index_dir=${INDEX_DIR}"

  python - <<PYEOF
from agent.tools_build_index import build_retrieval_index
from pathlib import Path

workdir = "${WORKDIR}"
index_dir = Path("${INDEX_DIR}")
pid = "${PID}"
bid = "${BID}"
bug_id = f"{pid}-{bid}b"
out = index_dir / f"{bug_id}_index.json"

res = build_retrieval_index(
    workdir=workdir,
    out_path=str(out),
    benchmark="defects4j",
    project=pid,
    revision=f"{bid}b",
    language="java",
    force=False,
)
print(res)
PYEOF

  exit 0
fi

if [ "$DATASET" = "swe" ]; then
  if [ -z "$INSTANCE_ID" ] || [ -z "$WORKDIR" ]; then
    echo "[ERROR] --instance-id and --workdir are required for dataset swe" >&2
    usage
    exit 2
  fi

  WORK_ROOT="${TRACE_WORK_ROOT:-/tmp/trace_work}"
  INDEX_DIR="${WORK_ROOT}/swebench_index"
  mkdir -p "${INDEX_DIR}"

  echo "[INFO] dataset=swe instance_id=${INSTANCE_ID}"
  echo "[INFO] workdir=${WORKDIR}"
  echo "[INFO] index_dir=${INDEX_DIR}"

  python - <<PYEOF
from agent.tools_build_index import build_retrieval_index
from pathlib import Path

workdir = "${WORKDIR}"
index_dir = Path("${INDEX_DIR}")
inst_id = "${INSTANCE_ID}"
out = index_dir / f"{inst_id}_index.json"

res = build_retrieval_index(
    workdir=workdir,
    out_path=str(out),
    benchmark="swebench_verified",
    project=inst_id.split("__")[0],
    revision=inst_id,
    language="java",  # SWE-bench projects are mixed; TRACE index currently supports Java only
    force=False,
)
print(res)
PYEOF

  exit 0
fi

