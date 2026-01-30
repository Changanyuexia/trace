"""
Ablation study module for APR framework.

This module provides configurable versions of the agent with different feature combinations:
- G0: Baseline (grep/read_file localization, unified diff, full test validation)
- G1: G0 + TDD Gate (RED/GREEN verification)
- G2: G0 + Retrieval Index (symbol_lookup/find_references/read_span)
- G3: G0 + Patch/Compile Gate (git apply check, canonical diff, compile gate)
- G5: Full system (all features enabled)
"""











