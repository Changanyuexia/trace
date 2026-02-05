#!/usr/bin/env bash
set -euo pipefail

PID="$1"      # e.g., Lang
BID="$2"      # e.g., 1
WORKDIR="$3"  # e.g., /tmp/Lang-1b

# CRITICAL: Set Java 11 environment for defects4j
# defects4j requires Java 11, and the script checks Java version at startup
JAVA11_PATH="/usr/lib/jvm/java-11-openjdk-11.0.25.0.9-7.el9.x86_64"
if [ -d "$JAVA11_PATH" ]; then
    export JAVA_HOME="$JAVA11_PATH"
    export PATH="$JAVA11_PATH/bin:$PATH"
else
    # Try to find Java 11 automatically
    JAVA11_AUTO=$(ls -d /usr/lib/jvm/java-11-openjdk* 2>/dev/null | head -1)
    if [ -n "$JAVA11_AUTO" ] && [ -d "$JAVA11_AUTO" ]; then
        export JAVA_HOME="$JAVA11_AUTO"
        export PATH="$JAVA11_AUTO/bin:$PATH"
    fi
fi

# Ensure DEFECTS4J_HOME is set (if not already set)
# Use environment variable or default relative path
if [ -z "${DEFECTS4J_HOME:-}" ]; then
    # Default: relative to trace/ directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    DEFECTS4J_DEFAULT="${SCRIPT_DIR}/../defects4j"
    if [ -d "$DEFECTS4J_DEFAULT" ]; then
        export DEFECTS4J_HOME="$DEFECTS4J_DEFAULT"
    else
        echo "ERROR: DEFECTS4J_HOME not set and default path not found: $DEFECTS4J_DEFAULT" >&2
        exit 1
    fi
fi

# Ensure defects4j is in PATH
if [ -d "$DEFECTS4J_HOME/framework/bin" ]; then
    export PATH="$DEFECTS4J_HOME/framework/bin:$PATH"
fi

# Ensure TZ is set (defects4j requirement)
if [ -z "${TZ:-}" ]; then
    export TZ="America/Los_Angeles"
fi

rm -rf "$WORKDIR"
defects4j checkout -p "$PID" -v "${BID}b" -w "$WORKDIR"
echo "$WORKDIR"












