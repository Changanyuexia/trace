"""
Utility functions for the APR framework.
"""
import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# Global tool call cache
_tool_call_cache = {}


def get_cache_key(func_name: str, args: dict) -> str:
    """Generate a cache key for a tool call."""
    # Sort args to ensure consistent keys
    sorted_args = json.dumps(args, sort_keys=True, ensure_ascii=False)
    key_str = f"{func_name}:{sorted_args}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_result(func_name: str, args: dict) -> Optional[Dict[str, Any]]:
    """Get cached result for a tool call if available."""
    cache_key = get_cache_key(func_name, args)
    if cache_key in _tool_call_cache:
        cached = _tool_call_cache[cache_key]
        # Mark as cached in the result
        result = cached.copy()
        result["_cached"] = True
        return result
    return None


def cache_tool_result(func_name: str, args: dict, result: Dict[str, Any]):
    """Cache a tool call result."""
    # Only cache successful results for certain functions
    cacheable_functions = ["read_file", "read_span", "symbol_lookup", "find_references"]
    if func_name in cacheable_functions and result.get("ok"):
        cache_key = get_cache_key(func_name, args)
        _tool_call_cache[cache_key] = result.copy()


def clear_cache():
    """Clear the tool call cache."""
    global _tool_call_cache
    _tool_call_cache = {}


def extract_test_failure_info(logfile: str, workdir: str) -> Dict[str, Any]:
    """
    Extract test failure information from log file and failing_tests file.
    Returns filtered information without exposing file paths and line numbers.
    """
    # Extract failure information
    failure_info = {
        "ok": True,
        "test_name": None,
        "exception_type": None,
        "exception_message": None,
        "assertion_failure": None,
        "stack_trace_summary": [],  # Filtered stack trace without file paths/line numbers
        "key_error_lines": []
    }
    
    # First, try to read the log file
    log_path = Path(logfile)
    log_content = ""
    if log_path.exists():
        try:
            log_content = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[WARN] Failed to read log file {logfile}: {e}", file=sys.stderr, flush=True)
    
    # Also try to read failing_tests file (usually has more detailed stack trace)
    failing_tests_path = Path(workdir) / "failing_tests"
    failing_tests_content = ""
    if failing_tests_path.exists():
        try:
            failing_tests_content = failing_tests_path.read_text(encoding="utf-8", errors="replace")
            print(f"[INFO] Reading failing_tests file for detailed stack trace", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] Failed to read failing_tests file: {e}", file=sys.stderr, flush=True)
    
    # Prefer failing_tests content if available (usually has more details)
    content = failing_tests_content if failing_tests_content else log_content
    
    if not content:
        return {"ok": False, "error": "No log content available"}
    
    lines = content.splitlines()
    
    # Find test name (usually in first few lines or after "---")
    for i, line in enumerate(lines[:50]):
        if "---" in line and "::" in line:
            # Extract test name: "--- org.example.Test::testMethod"
            parts = line.split("---")
            if len(parts) > 1:
                test_name = parts[-1].strip()
                failure_info["test_name"] = test_name
                break
    
    # Find exception/assertion failure
    in_stack_trace = False
    stack_depth = 0
    max_stack_depth = 10  # Limit stack trace depth
    
    for i, line in enumerate(lines):
        # Look for exception types
        exception_patterns = [
            r"java\.lang\.(\w+Exception):",
            r"junit\.framework\.AssertionFailedError",
            r"org\.junit\.AssertionError",
            r"AssertionError",
        ]
        
        for pattern in exception_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                failure_info["exception_type"] = match.group(1) if match.groups() else match.group(0)
                # Extract exception message (next line or same line)
                if ":" in line:
                    msg_part = line.split(":", 1)[1].strip()
                    if msg_part:
                        failure_info["exception_message"] = filter_file_paths(msg_part)
                in_stack_trace = True
                stack_depth = 0
                break
        
        # Look for assertion failures
        if "assert" in line.lower() and ("failed" in line.lower() or "error" in line.lower()):
            # Extract assertion message
            assertion_msg = filter_file_paths(line)
            failure_info["assertion_failure"] = assertion_msg
            failure_info["key_error_lines"].append(assertion_msg)
        
        # Collect stack trace (filtered)
        if in_stack_trace and stack_depth < max_stack_depth:
            # Stack trace lines typically start with "at" or contain method names
            if re.match(r"^\s*at\s+", line) or ("(" in line and ")" in line):
                # Extract method name and class, but remove file path and line number
                filtered_line = filter_stack_trace_line(line)
                if filtered_line:
                    failure_info["stack_trace_summary"].append(filtered_line)
                    stack_depth += 1
            elif line.strip() == "" or not (line.strip().startswith("at") or "(" in line):
                # End of stack trace
                if stack_depth > 0:
                    in_stack_trace = False
    
    # Extract key error messages (lines containing "error", "failed", "exception")
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ["error", "failed", "exception", "assertion"]):
            filtered = filter_file_paths(line)
            if filtered and filtered not in failure_info["key_error_lines"]:
                failure_info["key_error_lines"].append(filtered)
                if len(failure_info["key_error_lines"]) >= 5:  # Limit to 5 key lines
                    break
    
    # If we didn't find enough information in the log, try failing_tests file
    if not failure_info.get("stack_trace_summary") and not failure_info.get("exception_type"):
        failing_tests_path = Path(workdir) / "failing_tests"
        if failing_tests_path.exists():
            try:
                failing_content = failing_tests_path.read_text(encoding="utf-8", errors="replace")
                failing_lines = failing_content.splitlines()
                
                # Look for stack trace in failing_tests (usually more detailed)
                in_stack = False
                stack_depth = 0
                for line in failing_lines:
                    # Find exception
                    if not failure_info.get("exception_type"):
                        for pattern in exception_patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                failure_info["exception_type"] = match.group(1) if match.groups() else match.group(0)
                                if ":" in line:
                                    msg_part = line.split(":", 1)[1].strip()
                                    if msg_part:
                                        failure_info["exception_message"] = filter_file_paths(msg_part)
                                in_stack = True
                                stack_depth = 0
                                break
                    
                    # Collect stack trace
                    if in_stack and stack_depth < max_stack_depth:
                        if re.match(r"^\s*at\s+", line) or ("(" in line and ")" in line and ".java" in line):
                            filtered_line = filter_stack_trace_line(line)
                            if filtered_line and filtered_line not in failure_info["stack_trace_summary"]:
                                failure_info["stack_trace_summary"].append(filtered_line)
                                stack_depth += 1
                        elif line.strip() == "":
                            if stack_depth > 0:
                                in_stack = False
            except Exception as e:
                print(f"[WARN] Failed to read failing_tests file: {e}", file=sys.stderr, flush=True)
    
    return failure_info


def filter_file_paths(text: str) -> str:
    """Remove file paths and line numbers from text to prevent information leakage."""
    # Remove absolute paths
    text = re.sub(r'/[^\s:]+\.java:\d+', '[FILE:LINE]', text)
    text = re.sub(r'[A-Z]:\\[^\s:]+\.java:\d+', '[FILE:LINE]', text)  # Windows paths
    
    # Remove relative paths with line numbers
    text = re.sub(r'[^\s/]+/[^\s/]+\.java:\d+', '[FILE:LINE]', text)
    
    # Remove standalone line numbers after colons (but keep method signatures)
    text = re.sub(r':(\d+)(?=\s|$)', ':[LINE]', text)
    
    return text


def filter_stack_trace_line(line: str) -> str:
    """
    Filter a stack trace line to remove file paths and line numbers.
    Example: "at org.example.Class.method(Class.java:123)" -> "at org.example.Class.method([FILE:LINE])"
    """
    # Pattern: "at package.Class.method(File.java:123)"
    pattern = r'(at\s+[^(]+)\([^)]+\.java:\d+\)'
    match = re.search(pattern, line)
    if match:
        method_part = match.group(1)
        return f"{method_part}([FILE:LINE])"
    
    # Pattern: "package.Class.method(File.java:123)"
    pattern2 = r'([^(]+)\([^)]+\.java:\d+\)'
    match2 = re.search(pattern2, line)
    if match2:
        method_part = match2.group(1)
        return f"{method_part}([FILE:LINE])"
    
    # If no pattern matches, just filter file paths
    return filter_file_paths(line)


def format_failure_summary(failure_info: Dict[str, Any]) -> str:
    """Format failure information as a summary string."""
    if not failure_info.get("ok"):
        return f"Failed to extract failure info: {failure_info.get('error', 'unknown')}"
    
    parts = []
    
    if failure_info.get("test_name"):
        parts.append(f"Test: {failure_info['test_name']}")
    
    if failure_info.get("exception_type"):
        parts.append(f"Exception: {failure_info['exception_type']}")
    
    if failure_info.get("exception_message"):
        parts.append(f"Message: {failure_info['exception_message']}")
    
    if failure_info.get("assertion_failure"):
        parts.append(f"Assertion: {failure_info['assertion_failure']}")
    
    if failure_info.get("stack_trace_summary"):
        parts.append("Stack trace (filtered):")
        for i, frame in enumerate(failure_info["stack_trace_summary"][:5], 1):
            parts.append(f"  {i}. {frame}")
    
    if failure_info.get("key_error_lines"):
        parts.append("Key error lines:")
        for line in failure_info["key_error_lines"][:3]:
            parts.append(f"  - {line}")
    
    return "\n".join(parts) if parts else "No failure information extracted"


def validate_localization_result(result: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate localization result format and content.
    Returns (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, "Localization result must be a dictionary"
    
    # Try to parse if it's a JSON string
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return False, "Localization result is not valid JSON"
    
    # Required fields
    required_fields = ["red_test", "suspects"]
    for field in required_fields:
        if field not in result:
            return False, f"Missing required field: {field}"
    
    # Validate suspects
    suspects = result.get("suspects", [])
    if not isinstance(suspects, list):
        return False, "suspects must be a list"
    
    if len(suspects) == 0:
        return False, "suspects list cannot be empty"
    
    # Validate each suspect
    for i, suspect in enumerate(suspects):
        if not isinstance(suspect, dict):
            return False, f"suspect[{i}] must be a dictionary"
        
        if "file" not in suspect:
            return False, f"suspect[{i}] missing 'file' field"
        
        if "start_line" in suspect and "end_line" in suspect:
            start = suspect.get("start_line")
            end = suspect.get("end_line")
            if not isinstance(start, int) or not isinstance(end, int):
                return False, f"suspect[{i}] start_line and end_line must be integers"
            if start > end:
                return False, f"suspect[{i}] start_line ({start}) > end_line ({end})"
            if start < 1:
                return False, f"suspect[{i}] start_line must be >= 1"
    
    return True, None

