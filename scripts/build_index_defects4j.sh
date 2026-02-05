#!/usr/bin/env bash
set -euo pipefail

# Build retrieval index for a single Defects4J bug (wrapper around bin/build_index.sh).
# Usage: ./scripts/build_index_defects4j.sh Chart 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/bin/build_index.sh" --dataset d4j --pid "${1:-Chart}" --bid "${2:-1}"
