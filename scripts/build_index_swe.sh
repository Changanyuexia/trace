#!/usr/bin/env bash
set -euo pipefail

# Build retrieval index for a single SWE-bench instance (Python; same as D4J, only TRACE_WORK_ROOT needed).
# Usage: ./scripts/build_index_swe.sh <instance_id> [workdir]
# Example: ./scripts/build_index_swe.sh sympy__sympy-13647

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_ID="${1:?instance_id required}"
WORKDIR="${2:-}"
if [[ -n "$WORKDIR" ]]; then
  exec bash "${ROOT_DIR}/bin/build_index.sh" --dataset swe --instance-id "$INSTANCE_ID" --workdir "$WORKDIR"
else
  exec bash "${ROOT_DIR}/bin/build_index.sh" --dataset swe --instance-id "$INSTANCE_ID"
fi
