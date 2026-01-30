import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

def _run(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    # Be robust to non-UTF8 bytes from tools/logs (avoid crashing the whole run).
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def read_file(path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    snippet = "\n".join(f"{i+1}: {lines[i]}" for i in range(start-1, end))
    return {"ok": True, "path": str(p), "start_line": start, "end_line": end, "snippet": snippet}

def search_in_files(query: str, root: str, glob: str = "*", max_hits: int = 50) -> Dict[str, Any]:
    rootp = Path(root)
    if not rootp.exists():
        return {"ok": False, "error": f"root not found: {root}"}

    rg = shutil.which("rg")
    hits = []
    if rg:
        cmd = [rg, "-n", "--no-heading", "--glob", glob, query, str(rootp)]
        r = _run(cmd)
        if r["rc"] not in (0, 1):  # 1 means no matches
            return {"ok": False, "error": "rg failed", **r}
        for line in (r["stdout"] or "").splitlines()[:max_hits]:
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    hits.append({"path": parts[0], "line": int(parts[1]), "text": parts[2]})
                except Exception:
                    continue
        return {"ok": True, "engine": "rg", "hits": hits}

    # fallback (slower)
    for fp in rootp.rglob(glob):
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, s in enumerate(txt, start=1):
            if query in s:
                hits.append({"path": str(fp), "line": i, "text": s})
                if len(hits) >= max_hits:
                    return {"ok": True, "engine": "py", "hits": hits}
    return {"ok": True, "engine": "py", "hits": hits}

def apply_patch(workdir: str, unified_diff: str) -> Dict[str, Any]:
    wd = Path(workdir)
    if not wd.exists():
        return {"ok": False, "error": f"workdir not found: {workdir}"}
    
    # Check if it's a git repo
    if not (wd / ".git").exists():
        return {"ok": False, "error": f"not a git repository: {workdir}"}
    
    # Normalize: ensure patch ends with newline (some git apply errors manifest at EOF)
    if unified_diff and not unified_diff.endswith("\n"):
        unified_diff = unified_diff + "\n"

    patch_path = wd / ".agent_patch.diff"
    patch_path.write_text(unified_diff, encoding="utf-8")
    
    # First check if patch can be applied (dry-run)
    check_r = _run(["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)], cwd=str(wd))
    if check_r["rc"] != 0:
        error_detail = (check_r.get("stderr", "") or check_r.get("stdout", ""))[:800]
        return {"ok": False, "error": "patch check failed (patch may be corrupt or incompatible)", "stderr": error_detail, "check_failed": True, **check_r}
    
    # Try to apply with more lenient options
    r = _run(["git", "apply", "--whitespace=nowarn", "--ignore-space-change", "--ignore-whitespace", str(patch_path)], cwd=str(wd))
    if r["rc"] != 0:
        # Get more detailed error information
        error_detail = (r.get("stderr", "") or r.get("stdout", ""))[:800]
        _run(["git", "reset", "--hard"], cwd=str(wd))
        return {"ok": False, "error": f"git apply failed; repo reset", "stderr": error_detail, **r}
    return {"ok": True, "applied": True}

def apply_edits(workdir: str, edits_json: str) -> Dict[str, Any]:
    """
    Apply structured edits to files in workdir.
    
    edits_json: JSON string with format:
    [
      {
        "path": "relative/path/to/file.java",
        "ops": [
          {"type": "replace", "start_line": 10, "end_line": 12, "text": "new code\n"},
          {"type": "insert", "start_line": 15, "text": "inserted code\n"},
          {"type": "delete", "start_line": 20, "end_line": 22}
        ]
      }
    ]
    """
    import json
    wd = Path(workdir)
    if not wd.exists():
        return {"ok": False, "error": f"workdir not found: {workdir}"}
    
    try:
        edits = json.loads(edits_json)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON: {e}"}
    
    if not isinstance(edits, list):
        return {"ok": False, "error": "edits must be a list"}
    
    applied_files = []
    errors = []
    
    for file_edit in edits:
        if not isinstance(file_edit, dict) or "path" not in file_edit or "ops" not in file_edit:
            errors.append(f"Invalid file_edit structure: {file_edit}")
            continue
        
        file_path = wd / file_edit["path"]
        if not file_path.exists():
            errors.append(f"File not found: {file_edit['path']}")
            continue
        
        try:
            # Read file - preserve original content for comparison
            original_content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = original_content.splitlines(keepends=True)
            if not lines:
                lines = [""]
            
            # Apply operations in reverse order to maintain line numbers
            ops = sorted(file_edit["ops"], key=lambda op: op.get("start_line", 0), reverse=True)
            
            for op in ops:
                op_type = op.get("type")
                start_line = op.get("start_line", 1) - 1  # Convert to 0-indexed
                
                if op_type == "replace":
                    end_line = op.get("end_line", start_line + 1) - 1
                    new_text = op.get("text", "")
                    # Ensure new_text ends with newline if it's not empty
                    if new_text and not new_text.endswith("\n"):
                        new_text += "\n"
                    # Replace lines
                    if new_text:
                        # Split new_text into lines, preserving newlines
                        new_lines = new_text.splitlines(keepends=True)
                        if new_lines and not new_lines[-1].endswith("\n"):
                            new_lines[-1] += "\n"
                        lines[start_line:end_line+1] = new_lines
                    else:
                        lines[start_line:end_line+1] = []
                elif op_type == "insert":
                    new_text = op.get("text", "")
                    # Ensure new_text ends with newline
                    if new_text and not new_text.endswith("\n"):
                        new_text += "\n"
                    # Insert at start_line
                    lines.insert(start_line, new_text)
                elif op_type == "delete":
                    end_line = op.get("end_line", start_line + 1) - 1
                    # Delete lines
                    del lines[start_line:end_line+1]
                else:
                    errors.append(f"Unknown operation type: {op_type}")
                    continue
            
            # Write file back - ensure it ends with newline
            new_content = "".join(lines)
            if new_content and not new_content.endswith("\n"):
                new_content += "\n"
            
            # Check if content actually changed (normalize whitespace for comparison)
            # This helps detect if edits are effectively no-ops
            original_normalized = original_content.rstrip()
            new_normalized = new_content.rstrip()
            
            # Debug: Log comparison for investigation
            import sys
            if original_normalized == new_normalized:
                # Content is effectively unchanged, skip writing this file
                # But don't add to applied_files since no actual change was made
                print(f"[DEBUG] File {file_edit['path']}: Content unchanged after edits, skipping write", file=sys.stderr, flush=True)
                continue
            
            # Content changed, write it
            file_path.write_text(new_content, encoding="utf-8")
            applied_files.append(file_edit["path"])
            
        except Exception as e:
            errors.append(f"Error applying edits to {file_edit['path']}: {e}")
            continue
    
    if errors:
        return {"ok": False, "error": "; ".join(errors), "applied_files": applied_files}
    
    # If no files were actually modified, return a warning
    if not applied_files:
        return {
            "ok": True,
            "applied_files": [],
            "warning": "No files were modified (edits resulted in no actual changes)"
        }
    
    return {"ok": True, "applied_files": applied_files}

def get_git_diff(workdir: str) -> Dict[str, Any]:
    """
    Get git diff of current changes in workdir.
    Returns the unified diff format that can be used as a patch.
    """
    wd = Path(workdir)
    if not wd.exists():
        return {"ok": False, "error": f"workdir not found: {workdir}"}
    
    if not (wd / ".git").exists():
        return {"ok": False, "error": f"not a git repository: {workdir}"}
    
    # Get diff
    r = _run(["git", "diff", "--no-color"], cwd=str(wd))
    if r["rc"] != 0:
        return {"ok": False, "error": "git diff failed", **r}
    
    diff_text = r["stdout"] or ""
    
    # Check if there are any changes
    if not diff_text.strip():
        return {"ok": True, "diff": "", "has_changes": False}
    
    return {"ok": True, "diff": diff_text, "has_changes": True}


