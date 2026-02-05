#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$1"
LOGFILE="$2"
mkdir -p "$(dirname "$LOGFILE")"

cd "$WORKDIR"

# defects4j test returns non-zero if tests fail, so we must not exit the script
set +e
defects4j test > "$LOGFILE" 2>&1
RC=$?
set -e

# print exit code so python can parse
echo "$RC"













