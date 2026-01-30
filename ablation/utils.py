"""
Utility functions for ablation experiments.
"""
import re
from typing import List, Set


def is_code_file(file_path: str) -> bool:
    """
    Check if a file path is a code file (not log, temp, or dependency files).
    
    Args:
        file_path: File path to check
        
    Returns:
        True if it's a code file, False otherwise
    """
    if not file_path:
        return False
    
    # Normalize path
    path = file_path.strip()
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path[1:]
    
    # Exclude log files
    if path.endswith((".log", ".out", ".err")):
        return False
    
    # Exclude temporary/diff files
    if path.startswith(".") and any(path.startswith(prefix) for prefix in [".swebench", ".agent", ".apr_"]):
        return False
    if "/.swebench" in path or "/.agent" in path:
        return False
    
    # Exclude dependency directories (pip install locations)
    if path.startswith(".apr_site/") or "/.apr_site/" in path:
        return False
    
    # Exclude build/artifact directories
    if any(path.startswith(prefix) for prefix in ["__pycache__/", ".pytest_cache/", ".git/", "node_modules/", "build/", "dist/", ".eggs/"]):
        return False
    
    # Exclude absolute paths outside repo (likely log files)
    if file_path.startswith("/") and ("/logs/" in file_path or "/log/" in file_path):
        return False
    
    return True


def extract_files_from_patch(patch_text: str) -> List[str]:
    """
    Extract file paths from a unified diff patch.
    
    Args:
        patch_text: Unified diff patch text
        
    Returns:
        List of file paths (relative to repo root), filtered to only code files
    """
    if not patch_text:
        return []
    
    files = []
    lines = patch_text.splitlines()
    
    for line in lines:
        # Match "diff --git a/path/to/file b/path/to/file"
        m = re.match(r'^diff --git\s+a/([^\s]+)\s+b/([^\s]+)', line)
        if m:
            file_path = m.group(2)  # Use 'b' side (new file)
            if file_path not in files and is_code_file(file_path):
                files.append(file_path)
            continue
        
        # Match "--- a/path/to/file" or "+++ b/path/to/file"
        m = re.match(r'^[+-]{3}\s+[ab]/([^\s]+)', line)
        if m:
            file_path = m.group(1)
            if file_path != "/dev/null" and file_path not in files and is_code_file(file_path):
                files.append(file_path)
    
    return files


def calculate_file_hit_at_k(predicted_files: List[str], actual_files: List[str], k: int) -> bool:
    """
    Calculate File Hit@k metric.
    
    Only considers code files (filters out logs, temp files, dependencies).
    
    Args:
        predicted_files: List of predicted file paths (in order of confidence)
        actual_files: List of actual modified file paths
        k: Top k files to consider
        
    Returns:
        True if any of the top k predicted code files is in actual_files, False otherwise
    """
    if not predicted_files or not actual_files:
        return False
    
    # Filter to only code files
    predicted_code_files = [f for f in predicted_files if is_code_file(f)]
    actual_code_files = [f for f in actual_files if is_code_file(f)]
    
    if not predicted_code_files or not actual_code_files:
        return False
    
    # Normalize file paths (remove leading/trailing slashes, handle relative paths)
    def normalize_path(p: str) -> str:
        p = p.strip()
        if p.startswith("./"):
            p = p[2:]
        if p.startswith("/"):
            p = p[1:]
        return p
    
    # Take top k code files from predicted list
    predicted_normalized = [normalize_path(f) for f in predicted_code_files[:k]]
    actual_normalized = {normalize_path(f) for f in actual_code_files}
    
    # Check if any predicted file matches any actual file
    for pred_file in predicted_normalized:
        # Exact match
        if pred_file in actual_normalized:
            return True
        # Check if predicted file is a substring of actual file or vice versa
        # (handles cases like "src/main/java/Foo.java" vs "Foo.java")
        for actual_file in actual_normalized:
            if pred_file in actual_file or actual_file in pred_file:
                return True
            # Check basename match (for cases where path differs)
            if pred_file.split("/")[-1] == actual_file.split("/")[-1]:
                return True
    
    return False
