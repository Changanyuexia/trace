#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$1"
TRIGGER_FILE="$2"
LOGFILE="$3"

mkdir -p "$(dirname "$LOGFILE")"
cd "$WORKDIR"
: > "$LOGFILE"

FAIL=0
while IFS= read -r t; do
  [[ -z "$t" ]] && continue
  echo "=== RUN $t ===" | tee -a "$LOGFILE"
  set +e
  defects4j test -t "$t" >> "$LOGFILE" 2>&1
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    FAIL=1
  fi
done < "$TRIGGER_FILE"

exit $FAIL













