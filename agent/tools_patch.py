"""
Patch phase tools (LLM-callable).

These tools are intended for the PATCH stage:
- apply_patch (unified diff)
- apply_edits (structured edits)
- get_git_diff
- check_compile (via adapter; registered in ablation/tools.py)
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.tools_common import apply_patch, apply_edits, get_git_diff


def patch_tool_schemas() -> List[Dict[str, Any]]:
    # NOTE: apply_patch/apply_edits/get_git_diff are exposed via ablation/tools.py func_map,
    # and are not declared here as schemas because schemas are centralized by phase in ablation/tools.py.
    # This module primarily provides registration helpers.
    return []


def register_patch_tools(func_map: Dict[str, Any], *, workdir: str) -> None:
    func_map["apply_patch"] = lambda patch_text: apply_patch(workdir, patch_text)
    # Support both 'edits' and 'edits_json' parameter names for compatibility
    # Qwen3CoderToolParser may use 'edits' while other models use 'edits_json'
    func_map["apply_edits"] = lambda edits=None, edits_json=None: apply_edits(workdir, edits_json or edits or "[]")
    func_map["get_git_diff"] = lambda: get_git_diff(workdir)




