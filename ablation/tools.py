"""
Tool registration for ablation runs.

Centralizes:
- tools_schema (OpenAI tool schema list)
- func_map (ToolRuntime mapping)

Keeps behavior consistent with previous main_ablation.py wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.tool_runtime import ToolRuntime
from agent.tools_common import read_file, search_in_files

from agent.tools_localize import localize_tool_schemas, register_localize_tools
from agent.tools_patch import register_patch_tools
from agent.tools_verify import register_verify_tools


def setup_tools(
    *,
    workdir: str,
    enable_index_retrieval: bool,
    index_exists: bool,
    enable_patch_compile_gate: bool,
    enable_tdd_gate: bool,
    red_test_name: Optional[str],
    red_log: Optional[str],
    green_log: Optional[str],
    adapter=None,
    meta_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], ToolRuntime]:
    """
    Returns:
      (tools_schema, tool_runtime)
    """
    # Phase mapping (schemas):
    # - Localize tools: read_file/search_in_files (+ retrieval tools for G2/TRACE)
    # - Patch tools: apply_patch/apply_edits/get_git_diff (+ check_compile when enabled)
    # - Verify tools: verify_red/verify_green when enabled
    tools_schema = localize_tool_schemas(enable_index_retrieval=enable_index_retrieval)

    func_map: Dict[str, Any] = {}

    # Localize-stage registration
    register_localize_tools(func_map, workdir=workdir, enable_index_retrieval=enable_index_retrieval)

    # Patch-stage registration
    register_patch_tools(func_map, workdir=workdir)

    if enable_patch_compile_gate and adapter is not None:
        func_map["check_compile"] = lambda: adapter.check_compile(workdir)

    # Verify-stage registration (TDD Gate)
    if enable_tdd_gate:
        register_verify_tools(
            func_map,
            adapter=adapter,
            workdir=workdir,
            red_test_name=red_test_name,
            red_log=red_log,
            green_log=green_log,
            meta_dir=meta_dir,
        )

    return tools_schema, ToolRuntime(func_map)


