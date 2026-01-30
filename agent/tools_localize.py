"""
Localize phase tools (LLM-callable).

These tools are intended for the LOCALIZE stage:
- read_file
- search_in_files
- (G2/G5) symbol_lookup / find_references / read_span
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agent.tools_common import read_file, search_in_files
from agent.tools_build_index import symbol_lookup, find_references, read_span


def localize_tool_schemas(*, enable_index_retrieval: bool) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read file lines with line numbers",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            }, "required": ["path"]},
        }},
        {"type": "function", "function": {
            "name": "search_in_files",
            "description": "Search query under root",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "root": {"type": "string"},
                "glob": {"type": "string"},
                "max_hits": {"type": "integer"},
            }, "required": ["query", "root"]},
        }},
    ]

    if enable_index_retrieval:
        schemas.extend([
            {"type": "function", "function": {
                "name": "symbol_lookup",
                "description": "Look up a symbol definition in the retrieval index",
                "parameters": {"type": "object", "properties": {
                    "index_path": {"type": "string"},
                    "symbol": {"type": "string"},
                }, "required": ["index_path", "symbol"]},
            }},
            {"type": "function", "function": {
                "name": "find_references",
                "description": "Find references to a symbol in the retrieval index",
                "parameters": {"type": "object", "properties": {
                    "index_path": {"type": "string"},
                    "symbol": {"type": "string"},
                }, "required": ["index_path", "symbol"]},
            }},
            {"type": "function", "function": {
                "name": "read_span",
                "description": "Read a span of code from a file (workdir-relative supported)",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                }, "required": ["path", "start_line", "end_line"]},
            }},
        ])

    return schemas


def register_localize_tools(func_map: Dict[str, Any], *, workdir: str, enable_index_retrieval: bool) -> None:
    """Populate func_map with localize-stage tools."""

    def read_file_wrapper(path: str, start_line=None, end_line=None):
        p = Path(path)
        if not p.is_absolute():
            p = Path(workdir) / path
        if p.is_dir():
            return {"ok": False, "error": f"path is a directory, not a file: {path}"}
        start = start_line if start_line is not None else 1
        end = end_line if end_line is not None else 200
        return read_file(str(p), start, end)

    func_map["read_file"] = read_file_wrapper
    # search_in_files doesn't support start_line/end_line, ignore them if provided
    def search_in_files_wrapper(query: str, root: str, glob: str = "**/*", max_hits: int = 50, **kwargs):
        # Ignore unsupported parameters like start_line, end_line that LLM might pass
        return search_in_files(query, root, glob, max_hits)
    func_map["search_in_files"] = search_in_files_wrapper

    if enable_index_retrieval:
        func_map["symbol_lookup"] = lambda index_path, symbol, max_candidates=10: symbol_lookup(index_path, symbol, max_candidates)
        func_map["find_references"] = lambda index_path, symbol: find_references(index_path, symbol)
        func_map["read_span"] = lambda path, start_line, end_line: read_span(path, start_line, end_line, workdir)


