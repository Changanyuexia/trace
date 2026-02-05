#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$1"
TEST_NAME="$2"   # e.g., org.apache.commons.lang3.math.NumberUtilsTest::TestLang747
LOGFILE="$3"

mkdir -p "$(dirname "$LOGFILE")"
cd "$WORKDIR"

set +e
defects4j test -t "$TEST_NAME" > "$LOGFILE" 2>&1
RC=$?
set -e

# Defects4J often writes detailed failure info to WORKDIR/failing_tests instead of stdout/stderr.
# To make red.log self-contained for localization, append failing_tests content if present.
# Cap the appended content to avoid huge logs.
if [ -f "failing_tests" ]; then
    if [ -s "failing_tests" ]; then
        {
            echo ""
            echo "===== BEGIN failing_tests (from $WORKDIR/failing_tests) ====="
            head -n 400 "failing_tests"
            echo "===== END failing_tests ====="
        } >> "$LOGFILE" 2>&1 || true
    fi
fi

# Check if test actually failed by looking at the log
# defects4j test returns 0 even if tests fail, so we need to check the log
TEST_FAILED=0
if grep -q "Failing tests:" "$LOGFILE" && grep -q "$TEST_NAME" "$LOGFILE"; then
    # Test is listed in failing tests
    TEST_FAILED=1
elif grep -q "FAIL\|FAILURE\|Exception\|Error" "$LOGFILE" && ! grep -q "All tests passed" "$LOGFILE"; then
    # Test failed (has FAIL/FAILURE/Exception/Error and no "All tests passed")
    TEST_FAILED=1
fi

# Return test failure status (1 if failed, 0 if passed)
echo "$TEST_FAILED"
exit "$TEST_FAILED"



