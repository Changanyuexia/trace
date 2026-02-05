"""
Convenience entrypoint for running the TRACE APR framework.

This is a thin wrapper around `ablation.main_ablation.main`, so that
the README command:

    python run_trace.py --dataset defects4j --workdir ... --pid ... --bid ... --model example

matches the actual file layout in `trace/`.
"""

from ablation.main_ablation import main as _main


if __name__ == "__main__":
    import sys

    result = _main()
    # `main()` already prints a JSON result; return 0/1 based on "ok"
    sys.exit(0 if result.get("ok") else 1)

