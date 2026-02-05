#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"

cd "$WORKDIR"

# common directories
defects4j export -p dir.src.classes   > "$OUTDIR/dir.src.classes.txt"
defects4j export -p dir.src.tests     > "$OUTDIR/dir.src.tests.txt"

# useful narrowing signals
defects4j export -p classes.modified  > "$OUTDIR/classes.modified.txt"
defects4j export -p tests.trigger     > "$OUTDIR/tests.trigger.txt"
defects4j export -p tests.relevant    > "$OUTDIR/tests.relevant.txt"

# optional: classpaths
defects4j export -p cp.compile        > "$OUTDIR/cp.compile.txt"
defects4j export -p cp.test           > "$OUTDIR/cp.test.txt"

echo "$OUTDIR"













