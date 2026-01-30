"""
Ablation study version of agent core.

This module provides a configurable agent loop that can enable/disable different features
for ablation studies. Based on the original core.py but with feature flags.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from ablation.config import AblationConfig

# Try to import get_model_id for vLLM model support
try:
    from ablation.model_loader import get_model_id
except ImportError:
    # Fallback if model_loader is not available
    def get_model_id(model_name: str) -> str:
        return model_name


# Global metrics tracker (will be set by run_agent_loop_ablation)
_metrics_tracker = None

def _log_usage(resp, phase: str, metrics_tracker: Optional[Dict[str, Any]] = None) -> None:
    """
    Best-effort token usage logging for OpenAI-compatible SDK responses.
    DeepSeek/OpenAI-compat typically returns resp.usage with prompt/completion/total tokens.
    Also updates metrics tracker if provided.
    """
    try:
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if prompt is None and completion is None and total is None:
            return
        # If provider doesn't return total_tokens, compute it.
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        print(f"[USAGE] phase={phase} prompt_tokens={prompt} completion_tokens={completion} total_tokens={total}", file=sys.stderr, flush=True)
        
        # Update metrics tracker
        if metrics_tracker is not None:
            # Overall totals (we only store total_tokens per user request)
            if total is not None:
                metrics_tracker["total_tokens"] = metrics_tracker.get("total_tokens", 0) + total

            # Phase totals (do NOT use max limits; record actual usage)
            phase_key = None
            if phase == "localize":
                phase_key = "localization"
            elif phase == "patch":
                phase_key = "patch"
            if phase_key:
                if total is not None:
                    metrics_tracker[f"{phase_key}_total_tokens"] = metrics_tracker.get(f"{phase_key}_total_tokens", 0) + total
    except Exception:
        return

def is_unified_diff(text: str) -> bool:
    """Check if text is a unified diff format."""
    return ("diff --git" in text) and (("\n--- " in text) or ("\n+++" in text))

def validate_unified_diff(text: str) -> Dict[str, Any]:
    """
    Validate basic unified-diff integrity beyond `is_unified_diff`.
    - Requires at least one hunk reminder: '@@ -a,b +c,d @@'
    - Verifies, per hunk, that old_count matches (# of ' ' + '-' lines) and new_count matches (# of ' ' + '+' lines)
    - Rejects placeholder lines like '...'
    """
    import re

    if not text or not text.strip():
        return {"ok": False, "error": "empty patch"}

    if "..." in text:
        # Common LLM failure mode: placeholder instead of real code
        return {"ok": False, "error": "patch contains placeholder '...'"}

    if not is_unified_diff(text):
        return {"ok": False, "error": "not unified diff (missing diff --git/---/+++)"}  # quick fail

    lines = text.splitlines()
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    hunks_found = 0
    i = 0
    while i < len(lines):
        m = hunk_re.match(lines[i])
        if not m:
            i += 1
            continue

        hunks_found += 1
        old_count = int(m.group(2) or "1")
        new_count = int(m.group(4) or "1")

        # Count hunk body lines until next hunk or next file header or EOF
        old_seen = 0
        new_seen = 0
        j = i + 1
        while j < len(lines):
            ln = lines[j]
            if ln.startswith("@@ "):
                break
            if ln.startswith("diff --git") or ln.startswith("--- ") or ln.startswith("+++ "):
                break
            if ln.startswith("\\ No newline at end of file"):
                j += 1
                continue

            if ln.startswith(" "):
                old_seen += 1
                new_seen += 1
            elif ln.startswith("-"):
                # Exclude file header lines; those are handled above
                if not ln.startswith("--- "):
                    old_seen += 1
            elif ln.startswith("+"):
                if not ln.startswith("+++ "):
                    new_seen += 1
            else:
                # Invalid line inside a hunk (must start with ' ', '+', '-', or '\')
                return {"ok": False, "error": f"invalid hunk line prefix at line {j+1}: {ln[:80]}"}
            j += 1

        if old_seen != old_count or new_seen != new_count:
            return {
                "ok": False,
                "error": "hunk line counts do not match @@ header",
                "hunk_header": lines[i],
                "expected_old": old_count,
                "expected_new": new_count,
                "seen_old": old_seen,
                "seen_new": new_seen,
            }

        i = j

    if hunks_found == 0:
        return {"ok": False, "error": "no @@ hunks found in patch"}

    return {"ok": True}

def _is_insufficient_balance_error(error_type: str, error_msg: str) -> bool:
    """
    Detect "out of credits / insufficient balance" style errors for DeepSeek/OpenAI-compatible APIs.
    Treat these as fatal: stop immediately to avoid burning retries/loops when the account has no money.
    """
    t = (error_type or "").lower()
    m = (error_msg or "")

    # Keep both raw and lower() because some providers return localized messages.
    ml = m.lower()

    # DeepSeek official error:
    # "402 - Insufficient Balance"
    # Use OR to be tolerant to message formatting differences.
    if ("402" in ml) or ("insufficient balance" in ml):
        return True

    keywords = [
        # Common provider error codes / phrases
        "insufficient_quota",
        "insufficient quota",
        "insufficient_balance",
        "insufficient balance",
        "exceeded your current quota",
        "quota exceeded",
        "payment required",
        "billing",
        "recharge",
        "top up",
        "out of credit",
        "out of credits",
        "no credit",
        # Chinese messages (seen in some proxies)
        "余额不足",
        "余额",
        "欠费",
    ]
    if any(k in ml for k in keywords) or any(k in m for k in ["余额不足", "欠费"]):
        return True

    # Conservative fallback
    if "authentication" in t and ("quota" in ml or "balance" in ml or "insufficient" in ml):
        return True

    return False


def _is_fatal_stop_immediately(error_type: str, error_msg: str) -> bool:
    """
    仅对 credits/quota exhausted（含 429+quota 文案）立即跳过、不重试。
    纯 429（rate limit）不在此处处理，交给下层 is_rate_limit 重试，避免已消耗大量 API 后因一次 429 直接放弃。
    """
    return _is_insufficient_balance_error(error_type, error_msg)


def _retry_wait_seconds(retry_count: int, retry_delay: float, error_msg: str, is_rate_limit: bool) -> float:
    """429 TPM/RPM 按分钟重置：若错误含 per min / TPM / RPM，至少等 60s 以跨过分钟边界。"""
    wait = retry_delay * (2 ** (retry_count - 1))
    if is_rate_limit and error_msg and any(x in (error_msg or "").lower() for x in ("per min", "tpm", "rpm")):
        wait = max(wait, 60.0)
    return wait

def clean_patch_text(text: str) -> str:
    """
    Minimal, non-heuristic cleanup for patch text.
    - Strip leading/trailing whitespace
    - Remove markdown fence lines (``` / ```diff / etc.)
    - If "diff --git" exists, return substring from the first "diff --git" to the end
    We intentionally avoid any guessy "end-of-patch" detection to prevent truncating valid diffs.
    """
    if not text:
        return ""
    text = text.strip()

    # Remove markdown fence lines, but keep everything else (avoid truncation)
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    text = "\n".join(lines).strip()

    if "diff --git" in text:
        text = text[text.find("diff --git") :].strip()

    # Remove trailing stray backticks if any
    text = text.rstrip("`").strip()
    return text

def _extract_file_line_from_localize(localize_text: str) -> Dict[str, Any]:
    """Best-effort extraction of (file, line) from localization output."""
    import re
    if not localize_text:
        return {"ok": False}

    # JSON first
    try:
        if localize_text.strip().startswith("{"):
            obj = json.loads(localize_text)
            if isinstance(obj, dict):
                f = obj.get("file") or obj.get("path")
                line = obj.get("line") or obj.get("start_line")
                if f and line:
                    try:
                        return {"ok": True, "file": str(f), "line": int(line)}
                    except Exception:
                        return {"ok": True, "file": str(f), "line": line}
    except Exception:
        pass

    # Patterns like: src/main/.../Foo.java:123
    m = re.search(r"(src/(?:main|test)/java/[^\s:]+\.java)\s*[:#]\s*(\d+)", localize_text)
    if m:
        return {"ok": True, "file": m.group(1), "line": int(m.group(2))}

    # Pattern: any path ending .java with line
    m2 = re.search(r"([^\s:]+\.java)\s*[:#]\s*(\d+)", localize_text)
    if m2:
        return {"ok": True, "file": m2.group(1), "line": int(m2.group(2))}

    # If only file path appears, return without line
    m3 = re.search(r"(src/(?:main|test)/java/[^\s]+\.java)", localize_text)
    if m3:
        return {"ok": True, "file": m3.group(1), "line": None}

    return {"ok": False}

def _extract_file_line_from_red_log(red_log_path: Path, workdir: Path) -> Dict[str, Any]:
    """Fallback: extract first relevant (file,line) from a Java stack trace in red.log."""
    import re
    if not red_log_path.exists():
        return {"ok": False}
    txt = red_log_path.read_text(encoding="utf-8", errors="ignore")
    # at ... (Foo.java:123)
    matches = re.findall(r"\(([^():]+\.java):(\d+)\)", txt)
    if not matches:
        return {"ok": False}

    # Prefer src/main/java paths if we can locate them
    for filename, ln in matches:
        # Find candidates in workdir
        cands = list(workdir.glob(f"**/{filename}"))
        if not cands:
            continue
        # Prefer main/java
        cands.sort(key=lambda p: ("src/main/java" not in str(p), len(str(p))))
        rel = str(cands[0].relative_to(workdir))
        return {"ok": True, "file": rel, "line": int(ln)}
    return {"ok": False}

def _extract_file_line_from_failing_tests(workdir: Path) -> Dict[str, Any]:
    """
    Fallback: parse Defects4J-generated `failing_tests` file in workdir.
    It usually contains a full stack trace with (Foo.java:123).
    """
    import re
    ft = workdir / "failing_tests"
    if not ft.exists():
        return {"ok": False}
    txt = ft.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\(([^():]+\.java):(\d+)\)", txt)
    if not matches:
        return {"ok": False}
    # First pass: prefer frames that resolve into src/main/java (production code)
    for filename, ln in matches:
        cands = list(workdir.glob(f"**/{filename}"))
        if not cands:
            continue
        main = [p for p in cands if "src/main/java" in str(p)]
        if not main:
            continue
        main.sort(key=lambda p: len(str(p)))
        rel = str(main[0].relative_to(workdir))
        return {"ok": True, "file": rel, "line": int(ln)}

    # Second pass: accept anything (including tests) if no production frame exists
    for filename, ln in matches:
        cands = list(workdir.glob(f"**/{filename}"))
        if not cands:
            continue
        cands.sort(key=lambda p: ("src/test/java" in str(p), len(str(p))))
        rel = str(cands[0].relative_to(workdir))
        return {"ok": True, "file": rel, "line": int(ln)}
    return {"ok": False}

def _read_context_snippet(workdir: Path, rel_file: str, line: Optional[int], radius: int = 80) -> Optional[str]:
    """Read a numbered code snippet around line (1-based)."""
    try:
        fp = (workdir / rel_file).resolve()
        if not fp.exists():
            return None
        content = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line is None:
            start = 1
            end = min(len(content), 200)
        else:
            start = max(1, int(line) - radius)
            end = min(len(content), int(line) + radius)
        out_lines = []
        for i in range(start, end + 1):
            out_lines.append(f"{i:4d}: {content[i-1]}")
        return "\n".join(out_lines)
    except Exception:
        return None

def run_agent_loop_ablation(
    client,
    model: str,
    prompts: Dict[str, str],
    tools_schema: List[Dict[str, Any]],
    tool_runtime,
    harness_fn: Callable[[], Dict[str, Any]],
    validate_fn: Callable[[str], Dict[str, Any]],
    apply_patch_fn: Callable[[str], Dict[str, Any]],
    read_log_hint: str,
    max_iters: int,
    config: AblationConfig,
    adapter=None,
    checkout_fn=None,
) -> Dict[str, Any]:
    """
    Ablation study version of agent loop with configurable features.
    
    Args:
        config: AblationConfig object controlling which features are enabled
    """
    print(f"[INFO] Starting ablation agent loop (variant: {config})...", file=sys.stderr, flush=True)
    print(f"[INFO] Feature flags: TDD={config.enable_tdd_gate}, Index={config.enable_index_retrieval}, Patch/Compile={config.enable_patch_compile_gate}", file=sys.stderr, flush=True)
    
    # Get actual model_id for API calls (for vLLM models, use model_id instead of config file name)
    api_model_name = get_model_id(model)
    if api_model_name != model:
        print(f"[INFO] Using model_id for API calls: {api_model_name} (config: {model})", file=sys.stderr, flush=True)
    
    # Initialize metrics collection
    import time
    start_time = time.time()
    # Store compile_result for later inclusion in final result
    initial_compile_result = None
    metrics = {
        "total_api_calls": 0,
        "localization_api_calls": 0,
        "patch_api_calls": 0,
        # Aliases to emphasize "actual total calls by phase" (requested)
        "localization_total_api_calls": 0,
        "patch_total_api_calls": 0,
        "total_tool_calls": 0,
        "localization_tool_calls": 0,
        "patch_tool_calls": 0,
        "localization_tool_calls_by_type": {},  # 定位阶段每个tool的调用次数
        "patch_tool_calls_by_type": {},  # patch阶段每个tool的调用次数
        "total_tool_calls_by_type": {},  # 全阶段每个tool的调用次数
        "total_tokens": 0,
        # Phase token usage (actual)
        "localization_total_tokens": 0,
        "patch_total_tokens": 0,
        "compile_failures": 0,
        "git_apply_failures": 0,
        "validation_failures": 0,
        "patch_attempts": 0,
        # Patch gate counters (store numerator/denominator; do NOT compute rates here)
        "apply_attempt_count": 0,
        "apply_success_count": 0,
        "compile_attempt_count": 0,
        "compile_success_count": 0,
        "tdd_gate_red_verified": False,
        "tdd_gate_green_verified": False,
        "runtime_seconds": 0.0,
        "localization_predicted_files": [],  # 定位阶段预测的文件列表
        "actual_modified_files": [],  # 实际修改的文件列表（从patch提取）
        "file_hit_at_1": False,  # File Hit@1
        "file_hit_at_3": False,  # File Hit@3
    }
    
    def _make_error_result(error_msg: str, **kwargs) -> Dict[str, Any]:
        """Helper to create error result with metrics."""
        end_time = time.time()
        metrics["runtime_seconds"] = end_time - start_time
        result = {"ok": False, "error": error_msg, "metrics": metrics}
        result.update(kwargs)
        # 如果harness_info存在，记录环境状态
        if "harness_info" in kwargs:
            harness_info = kwargs["harness_info"]
            result["harness_ok"] = harness_info.get("ok", False)
            if not result["harness_ok"]:
                result["harness_error"] = harness_info.get("error", "Harness failed")
        # 添加编译结果（如果存在且未在kwargs中提供）
        if initial_compile_result is not None and "compile_result" not in result:
            result["compile_result"] = initial_compile_result
        return result
    
    # Clear cache if available
    try:
        from agent.utils import clear_cache
        clear_cache()
    except ImportError:
        pass
    
    messages = [{"role": "system", "content": prompts["system"]}]
    
    # Initial harness
    print("[INFO] Running harness (checkout, export, test)...", file=sys.stderr, flush=True)
    harness_info = harness_fn()
    actual_workdir = harness_info.get("workdir", "")
    
    # Check if harness succeeded
    if not harness_info.get("ok", True):
        error_msg = harness_info.get("error", "Harness failed")
        checkout_info = harness_info.get("checkout", {})
        print(f"[ERROR] Harness failed: {error_msg}", file=sys.stderr, flush=True)
        if checkout_info:
            print(f"[ERROR] Checkout details: {checkout_info}", file=sys.stderr, flush=True)
        return _make_error_result(f"Harness failed: {error_msg}", harness_info=harness_info)
    
    print(f"[INFO] Harness completed. Workdir: {actual_workdir}", file=sys.stderr, flush=True)
    
    # IMPORTANT: Restore Java environment variables after index building
    # Index building may have changed JAVA_HOME to Java 17+ (for JDT Language Server)
    # but Defects4J requires Java 11, so we need to restore the original Java environment
    import os
    original_java_home = os.environ.get("JAVA_HOME", "")
    original_path = os.environ.get("PATH", "")
    
    # Find and restore Java 11 (required by Defects4J)
    # Check if we need to restore Java 11
    java11_paths = [
        "/usr/lib/jvm/java-11-openjdk",
        "/usr/lib/jvm/java-11",
        "/usr/lib/jvm/java-1.11.0-openjdk",
    ]
    java11_found = None
    # Trust the Java environment already configured by load_dataset_env.sh (from defects4j.json).
    # The config prefers Java 8 for older projects, Java 11 for most projects.
    # Only restore if it was changed by index building (Java 17+), otherwise keep the configured version.
    current_java_home = os.environ.get("JAVA_HOME", "")
    if current_java_home and ("java-17" in current_java_home or "java-21" in current_java_home or "java-1.17" in current_java_home or "java-1.21" in current_java_home):
        # Index building changed to Java 17+, restore to original (likely Java 8 or 11 from config)
        if original_java_home and Path(original_java_home).exists():
            os.environ["JAVA_HOME"] = original_java_home
            os.environ["PATH"] = original_path
            print(f"[INFO] Restored Java environment from index build (JAVA_HOME={original_java_home})", file=sys.stderr, flush=True)
        else:
            # Fallback: try to find Java 11 if original is missing
            java11_found = None
            for java_path in java11_paths:
                import glob
                matches = glob.glob(f"{java_path}*")
                if matches:
                    java11_found = matches[0]
                    break
            if java11_found and Path(java11_found).exists():
                os.environ["JAVA_HOME"] = str(java11_found)
                java11_bin = str(Path(java11_found) / "bin")
                path_parts = original_path.split(":")
                path_parts = [p for p in path_parts if "java-17" not in p and "java-21" not in p and "java-1.17" not in p and "java-1.21" not in p]
                os.environ["PATH"] = f"{java11_bin}:{':'.join(path_parts)}"
                print(f"[INFO] Restored Java 11 environment (fallback, JAVA_HOME={java11_found})", file=sys.stderr, flush=True)
    # If current Java is already 8 or 11 (from config), keep it - no need to restore
    
    # Update workdir in check_compile if it was set before harness
    if actual_workdir and "check_compile" in tool_runtime.func_map:
        # Re-register check_compile with the actual workdir from harness
        if adapter is not None:
            tool_runtime.func_map["check_compile"] = lambda: adapter.check_compile(actual_workdir)
        else:
            from agent.adapters import defects4j as d4j
            tool_runtime.func_map["check_compile"] = lambda: d4j.check_compile(actual_workdir)
    
    # G1: TDD Gate - Verify RED (if enabled)
    red_test_name = None
    if config.enable_tdd_gate and config.verify_red_test:
        print("[INFO] [G1] TDD Gate: Verifying RED test...", file=sys.stderr, flush=True)
        
        # Ensure compilation succeeds before running tests
        if "check_compile" in tool_runtime.func_map:
            print("[INFO] Ensuring compilation succeeds before verifying RED test...", file=sys.stderr, flush=True)
            compile_result = tool_runtime.func_map["check_compile"]()
            # Store compile_result for inclusion in final result
            initial_compile_result = compile_result
            # Make it explicit when compile gate is effectively skipped (e.g. swebench_verified)
            try:
                cr_ok = bool(compile_result.get("ok"))
                cr_skipped = bool(compile_result.get("skipped"))
                cr_reason = compile_result.get("reason", "")
                print(
                    f"[INFO] Compile gate result: ok={cr_ok} skipped={cr_skipped} reason={cr_reason}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            if not compile_result.get("ok"):
                error_summary = compile_result.get("error_summary", "") or compile_result.get("stderr", "") or compile_result.get("stdout", "")
                print(f"[ERROR] Compilation failed before RED test verification: {error_summary[:500]}", file=sys.stderr, flush=True)
                return _make_error_result("Compilation failed before RED test verification; cannot proceed", compile_result=compile_result)
            
            # Get verify_red_fn from tool_runtime if available
            red_rc = None
            red_test_name = "unknown"
            red_logfile = ""
            red_result = {}
            
            if "verify_red" in tool_runtime.func_map:
                verify_red_fn = tool_runtime.func_map["verify_red"]
                print("[INFO] [G1] Starting RED test execution...", file=sys.stderr, flush=True)
                print("[INFO] [G1] RED test will run in Apptainer container with SWE-bench testbed environment", file=sys.stderr, flush=True)
                red_result = verify_red_fn()
                red_rc = red_result.get("rc")
                red_test_name = red_result.get("test_name", "unknown")
                red_logfile = red_result.get("logfile", "")
                # Treat infrastructure errors as fatal (do NOT accept as "RED failed")
                if not red_result.get("ran", True) or red_rc in (None, -1, 255):
                    err = (red_result.get("error") or "") + "\n" + (red_result.get("stderr") or "")
                    print(f"[ERROR] RED test execution failed (infrastructure error, rc={red_rc}): {err[:800]}", file=sys.stderr, flush=True)
                    return _make_error_result("RED test execution failed; cannot proceed", red_result=red_result)
                
                if red_rc == 0:
                    print("[ERROR] RED test did not fail (rc=0); cannot proceed", file=sys.stderr, flush=True)
                    return _make_error_result("RED test did not fail; cannot proceed", red_result=red_result)
                # rc=2: pytest config/collection error; rc=4: no tests collected. Do not proceed to localize/patch.
                # 兼容 red_rc 为 str（如 "4"）的情形，确保一定会直接停止
                try:
                    _rc_int = red_rc if isinstance(red_rc, int) else (int(red_rc) if red_rc is not None else None)
                except (TypeError, ValueError):
                    _rc_int = None
                if _rc_int in (2, 4):
                    print(f"[ERROR] RED test rc={red_rc} (pytest config/collection or no tests collected); skipping localize/patch", file=sys.stderr, flush=True)
                    return _make_error_result(f"RED test rc={red_rc} (pytest config or no tests collected); cannot proceed", red_result=red_result)
            else:
                # verify_red not available, skip RED test verification
                print("[WARN] [G1] verify_red function not available, skipping RED test verification", file=sys.stderr, flush=True)
                return _make_error_result("TDD gate enabled but verify_red function not available")
            
            print(f"[INFO] [G1] RED test verified: {red_test_name} failed (rc={red_rc})", file=sys.stderr, flush=True)
            metrics["tdd_gate_red_verified"] = True
            
            # DEBUG: Log RED test result details
            print(f"[DEBUG] RED test result: rc={red_rc}, test_name={red_test_name}, logfile={red_logfile}", file=sys.stderr, flush=True)
            if red_result.get("stderr"):
                print(f"[DEBUG] RED test stderr: {red_result.get('stderr')[:500]}", file=sys.stderr, flush=True)
            if red_result.get("stdout"):
                print(f"[DEBUG] RED test stdout: {red_result.get('stdout')[:500]}", file=sys.stderr, flush=True)
            
            # Add RED test info to messages
            messages.append({
                "role": "user",
                "content": f"TDD_GATE_RED:\nTest: {red_test_name}\nRC: {red_rc}\nLogfile: {red_logfile}\n\nIMPORTANT: Read the red.log file (path above) FIRST to get failure details. DO NOT read test.full.log - it is too large and inefficient. The red.log contains the focused failure information you need."
            })
    
    # Add harness result to messages
    harness_json = json.dumps(harness_info, ensure_ascii=False)
    # Truncate harness_json if too large (keep first 4000 and last 2000 chars)
    if len(harness_json) > 8000:
        harness_json = harness_json[:4000] + "\n\n[... truncated ...]\n\n" + harness_json[-2000:]
        print(f"[WARN] Truncated harness_json from {len(json.dumps(harness_info, ensure_ascii=False))} to {len(harness_json)} chars", file=sys.stderr, flush=True)
    messages.append({
        "role": "user",
        "content": "HARNESS_RESULT:\n" + harness_json + "\n\n" + read_log_hint
    })
    
    # G2: Add index hint if index retrieval is enabled and index exists
    if config.enable_index_retrieval:
        index_path = harness_info.get("index_path", "")
        if index_path:
            # Verify index file actually exists
            index_file = Path(index_path)
            if index_file.exists():
                messages.append({
                    "role": "user",
                    "content": f"RETRIEVAL_INDEX: {index_path}\n\nYou can use symbol_lookup, find_references, and read_span tools with this index."
                })
                print(f"[INFO] [G2/G5] Index available at {index_path}, retrieval tools can be used", file=sys.stderr, flush=True)
            else:
                print(f"[WARN] [G2/G5] Index path provided but file does not exist: {index_path}", file=sys.stderr, flush=True)
                messages.append({
                    "role": "user",
                    "content": "RETRIEVAL_INDEX_UNAVAILABLE: The retrieval index was not successfully built. Please use grep/read_file tools for localization instead."
                })
        else:
            print(f"[WARN] [G2/G5] Index retrieval enabled but index_path is None/empty, using grep/read_file instead", file=sys.stderr, flush=True)
            messages.append({
                "role": "user",
                "content": "RETRIEVAL_INDEX_UNAVAILABLE: The retrieval index was not successfully built. Please use grep/read_file tools for localization instead."
            })
    
    last_patch = None
    
    # Timeout configuration: 20 minutes (1200 seconds)
    MAX_RUNTIME_SECONDS = 1200
    
    for it in range(1, max_iters + 1):
        # Check timeout before starting each iteration
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= MAX_RUNTIME_SECONDS:
            print(f"[WARN] Timeout reached ({elapsed_time:.1f}s >= {MAX_RUNTIME_SECONDS}s), stopping repair process", file=sys.stderr, flush=True)
            end_time = time.time()
            metrics["runtime_seconds"] = end_time - start_time
            result = {
                "ok": False,
                "iterations": it - 1,
                "patch": last_patch,
                "error": f"Timeout: exceeded {MAX_RUNTIME_SECONDS}s runtime limit",
                "metrics": metrics
            }
            result["harness_ok"] = harness_info.get("ok", True)
            if not result["harness_ok"]:
                result["harness_error"] = harness_info.get("error", "Harness failed")
            if "test_suite_verification" in harness_info:
                result["test_suite_verification"] = harness_info["test_suite_verification"]
            if initial_compile_result is not None:
                result["compile_result"] = initial_compile_result
            return result
        
        print(f"[INFO] === Iteration {it}/{max_iters} ===", file=sys.stderr, flush=True)
        print(f"[INFO] Elapsed time: {elapsed_time:.1f}s / {MAX_RUNTIME_SECONDS}s", file=sys.stderr, flush=True)
        
        # Localize phase
        print("[INFO] Localize phase: asking LLM to localize bug...", file=sys.stderr, flush=True)
        messages.append({"role": "user", "content": prompts["localize"]})
        
        tool_call_count = 0
        max_tool_calls = 15
        symbol_blocks_read = 0  # G2: Working set limit
        localization_api_count = 0  # Track API calls in localization phase
        max_localization_api_calls = config.max_localization_api_calls
        predicted_files = []  # Collect files accessed during localization (for File Hit@k)
        # Note: consecutive_direct_patches is only used in patch phase, not in localization phase
        
        while True:
            # Check timeout in localization loop
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= MAX_RUNTIME_SECONDS:
                print(f"[WARN] Timeout reached during localization ({elapsed_time:.1f}s >= {MAX_RUNTIME_SECONDS}s), stopping repair process", file=sys.stderr, flush=True)
                end_time = time.time()
                metrics["runtime_seconds"] = end_time - start_time
                result = {
                    "ok": False,
                    "iterations": it - 1,
                    "patch": last_patch,
                    "error": f"Timeout: exceeded {MAX_RUNTIME_SECONDS}s runtime limit during localization",
                    "metrics": metrics
                }
                result["harness_ok"] = harness_info.get("ok", True)
                if not result["harness_ok"]:
                    result["harness_error"] = harness_info.get("error", "Harness failed")
                if "test_suite_verification" in harness_info:
                    result["test_suite_verification"] = harness_info["test_suite_verification"]
                if initial_compile_result is not None:
                    result["compile_result"] = initial_compile_result
                return result
            
            if tool_call_count >= max_tool_calls:
                print(f"[WARN] Reached max tool calls ({max_tool_calls}), forcing LLM to return localization result...", file=sys.stderr, flush=True)
                messages.append({
                    "role": "user",
                    "content": "You have reached the maximum number of tool calls. Please return your localization result now (as JSON)."
                })
                
                # Retry mechanism for forced localization result
                max_retries = 5
                retry_delay = 2
                retry_count = 0
                msg = None
                
                while retry_count < max_retries:
                    try:
                        resp = client.chat.completions.create(
                            model=api_model_name,
                            messages=messages,
                            tools=tools_schema,
                            tool_choice="none",
                            timeout=180,
                        )
                        msg = resp.choices[0].message
                        break
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if _is_fatal_stop_immediately(error_type, error_msg):
                            print(f"[WARN] LLM API credits/quota exhausted, skip this bug: {error_type}: {error_msg[:300]}", file=sys.stderr, flush=True)
                            return _make_error_result(f"LLM credits/quota exhausted: {error_type}: {error_msg[:200]}")
                        is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in error_type
                        is_temporary = "timeout" in error_msg.lower() or "503" in error_msg or "502" in error_msg or "500" in error_msg
                        
                        retry_count += 1
                        if (is_rate_limit or is_temporary) and retry_count < max_retries:
                            wait_time = _retry_wait_seconds(retry_count, retry_delay, error_msg, is_rate_limit)
                            print(f"[WARN] Retrying forced localization result in {wait_time} seconds... (attempt {retry_count}/{max_retries})", file=sys.stderr, flush=True)
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[ERROR] Failed to get forced localization result: {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                            if is_rate_limit:
                                return _make_error_result(f"Failed to get forced localization result (rate limit 429): {error_type}: {error_msg[:200]}")
                            if is_temporary:
                                return _make_error_result(f"Failed to get forced localization result (timeout/server error): {error_type}: {error_msg[:200]}")
                            return _make_error_result(f"Failed to get forced localization result: {error_type}: {error_msg[:200]}")
                
                if msg is None:
                    return _make_error_result("Failed to get forced localization result after retries")
                
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
            
            # G2: Working set limit check
            if config.enable_index_retrieval and symbol_blocks_read >= config.max_symbol_blocks_per_round:
                print(f"[WARN] [G2] Reached working set limit ({config.max_symbol_blocks_per_round} symbol blocks), forcing localization result...", file=sys.stderr, flush=True)
                messages.append({
                    "role": "user",
                    "content": f"You have read {symbol_blocks_read} symbol blocks (limit: {config.max_symbol_blocks_per_round}). Please return your localization result now."
                })
                
                # Retry mechanism for forced localization result
                max_retries = 5
                retry_delay = 2
                retry_count = 0
                msg = None
                
                while retry_count < max_retries:
                    try:
                        resp = client.chat.completions.create(
                            model=api_model_name,
                            messages=messages,
                            tools=tools_schema,
                            tool_choice="none",
                            timeout=180,
                        )
                        msg = resp.choices[0].message
                        break
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if _is_fatal_stop_immediately(error_type, error_msg):
                            print(f"[WARN] LLM API credits/quota exhausted, skip this bug: {error_type}: {error_msg[:300]}", file=sys.stderr, flush=True)
                            return _make_error_result(f"LLM credits/quota exhausted: {error_type}: {error_msg[:200]}")
                        is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in error_type
                        is_temporary = "timeout" in error_msg.lower() or "503" in error_msg or "502" in error_msg or "500" in error_msg
                        
                        retry_count += 1
                        if (is_rate_limit or is_temporary) and retry_count < max_retries:
                            wait_time = _retry_wait_seconds(retry_count, retry_delay, error_msg, is_rate_limit)
                            print(f"[WARN] Retrying forced localization result in {wait_time} seconds... (attempt {retry_count}/{max_retries})", file=sys.stderr, flush=True)
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[ERROR] Failed to get forced localization result: {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                            if is_rate_limit:
                                return _make_error_result(f"Failed to get forced localization result (rate limit 429): {error_type}: {error_msg[:200]}")
                            if is_temporary:
                                return _make_error_result(f"Failed to get forced localization result (timeout/server error): {error_type}: {error_msg[:200]}")
                            return _make_error_result(f"Failed to get forced localization result: {error_type}: {error_msg[:200]}")
                
                if msg is None:
                    return _make_error_result("Failed to get forced localization result after retries")
                
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
            
            # Check localization API call limit
            if localization_api_count >= max_localization_api_calls:
                print(f"[WARN] Reached max localization API calls ({max_localization_api_calls}), forcing LLM to return localization result...", file=sys.stderr, flush=True)
                messages.append({
                    "role": "user",
                    "content": "You have reached the maximum number of API calls in localization phase. Please return your localization result now (as JSON)."
                })
                # Force return (similar to max_tool_calls logic)
                max_retries = 5
                retry_delay = 2
                retry_count = 0
                msg = None
                while retry_count < max_retries:
                    try:
                        resp = client.chat.completions.create(
                            model=api_model_name,
                            messages=messages,
                            tools=tools_schema,
                            tool_choice="none",
                            timeout=180,
                        )
                        msg = resp.choices[0].message
                        break
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if _is_fatal_stop_immediately(error_type, error_msg):
                            print(f"[WARN] LLM API credits/quota exhausted, skip this bug: {error_type}: {error_msg[:300]}", file=sys.stderr, flush=True)
                            return _make_error_result(f"LLM credits/quota exhausted: {error_type}: {error_msg[:200]}")
                        is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in error_type
                        retry_count += 1
                        if retry_count < max_retries:
                            import time
                            wait_time = _retry_wait_seconds(retry_count, retry_delay, error_msg, is_rate_limit)
                            time.sleep(wait_time)
                        else:
                            if is_rate_limit:
                                return _make_error_result(f"Failed to get forced localization result (rate limit 429): {error_type}: {error_msg[:200]}")
                            return _make_error_result(f"Failed to get forced localization result after reaching API limit: {error_type}: {error_msg[:200]}")
                if msg:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    break
                else:
                    return _make_error_result("Failed to get forced localization result")
            
            print(f"[INFO] Calling LLM API (localize, tool calls: {tool_call_count}, API calls: {localization_api_count}/{max_localization_api_calls})...", file=sys.stderr, flush=True)
            localization_api_count += 1  # Increment API call counter (will be added to metrics in _log_usage)
            
            # Retry mechanism for LLM API calls
            max_retries = 5
            retry_delay = 2  # seconds
            retry_count = 0
            msg = None
            
            while retry_count < max_retries:
                try:
                    resp = client.chat.completions.create(
                        model=api_model_name,
                        messages=messages,
                        tools=tools_schema,
                        tool_choice="auto",
                        timeout=180,
                    )
                    msg = resp.choices[0].message
                    # Only count successful API calls (not retries)
                    if retry_count == 0:
                        metrics["localization_api_calls"] += 1
                        metrics["localization_total_api_calls"] += 1
                        metrics["total_api_calls"] += 1
                    _log_usage(resp, phase="localize", metrics_tracker=metrics)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # credits/quota exhausted：不重试本条，跳过跑下一条。纯 429 走下面 is_rate_limit 重试。
                    if _is_fatal_stop_immediately(error_type, error_msg):
                        print(f"[WARN] LLM API credits/quota exhausted, skip this bug: {error_type}: {error_msg[:300]}", file=sys.stderr, flush=True)
                        return _make_error_result(f"LLM credits/quota exhausted: {error_type}: {error_msg[:200]}")
                    
                    # Check if it's a rate limit error (429) or temporary error
                    is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in error_type
                    is_temporary = "timeout" in error_msg.lower() or "503" in error_msg or "502" in error_msg or "500" in error_msg
                    
                    retry_count += 1
                    
                    if is_rate_limit or is_temporary:
                        if retry_count < max_retries:
                            wait_time = _retry_wait_seconds(retry_count, retry_delay, error_msg, is_rate_limit)
                            print(f"[WARN] LLM API call failed ({error_type}): {error_msg[:200]}", file=sys.stderr, flush=True)
                            print(f"[WARN] Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})", file=sys.stderr, flush=True)
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            import traceback
                            print(f"[ERROR] LLM API call failed after {max_retries} retries: {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                            print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                            if is_rate_limit:
                                return _make_error_result(f"LLM API rate limit (429) after {max_retries} retries in localization: {error_type}: {error_msg[:200]}")
                            return _make_error_result(f"LLM API call failed after {max_retries} retries (timeout/server error) in localization: {error_type}: {error_msg[:200]}")
                    else:
                        # Non-retryable error, fail immediately
                        print(f"[ERROR] LLM API call failed (non-retryable): {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                        import traceback
                        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                        return _make_error_result(f"LLM API call failed in localization phase: {error_type}: {error_msg[:200]}")
            
            if msg is None:
                return _make_error_result("Failed to get LLM response after retries")
            
            # DEBUG: Log LLM response details
            content_length = len(msg.content or "")
            tool_calls_count = len(getattr(msg, "tool_calls", []) or [])
            print(f"[DEBUG] LLM response - content length: {content_length}, tool_calls: {tool_calls_count}", file=sys.stderr, flush=True)
            if content_length > 0:
                preview = (msg.content or "")[:200] + ("..." if content_length > 200 else "")
                print(f"[DEBUG] LLM content preview: {preview}", file=sys.stderr, flush=True)
            if tool_calls_count > 0:
                for i, tc in enumerate(getattr(msg, "tool_calls", []) or [], 1):
                    func_name = getattr(tc.function, "name", "unknown")
                    args_preview = (getattr(tc.function, "arguments", "") or "")[:150]
                    print(f"[DEBUG] Tool call {i}: {func_name}({args_preview}...)", file=sys.stderr, flush=True)
            
            if getattr(msg, "tool_calls", None):
                tool_call_count += len(msg.tool_calls)
                metrics["localization_tool_calls"] += len(msg.tool_calls)
                metrics["total_tool_calls"] += len(msg.tool_calls)
                print(f"[INFO] LLM requested {len(msg.tool_calls)} tool call(s) (total: {tool_call_count})", file=sys.stderr, flush=True)
                
                # DEBUG: Log tool call details and extract file paths for File Hit@k
                for i, tc in enumerate(msg.tool_calls, 1):
                    func_name = getattr(tc.function, "name", "unknown")
                    args_raw = getattr(tc.function, "arguments", "") or "{}"
                    # Support both JSON string (OpenAI) and dict (e.g. Gemini OpenAI-compat may return dict)
                    if isinstance(args_raw, dict):
                        args = args_raw
                        args_str = json.dumps(args_raw, ensure_ascii=False)
                    else:
                        args_str = args_raw if isinstance(args_raw, str) else str(args_raw)
                        args = {}
                    
                    # Count tool calls by type for localization phase
                    if func_name not in metrics["localization_tool_calls_by_type"]:
                        metrics["localization_tool_calls_by_type"][func_name] = 0
                    metrics["localization_tool_calls_by_type"][func_name] += 1
                    if func_name not in metrics["total_tool_calls_by_type"]:
                        metrics["total_tool_calls_by_type"][func_name] = 0
                    metrics["total_tool_calls_by_type"][func_name] += 1
                    
                    try:
                        if not isinstance(args_raw, dict):
                            args = json.loads(args_str)
                        print(f"[DEBUG] Tool call {i} details: function={func_name}, args={json.dumps(args, ensure_ascii=False)[:200]}", file=sys.stderr, flush=True)
                        
                        # Extract file paths from tool calls for File Hit@k metric
                        if func_name in ["read_file", "read_span", "grep", "search_in_files"]:
                            file_path = args.get("path") or args.get("file")
                            if file_path and isinstance(file_path, str):
                                # Normalize path (remove workdir prefix if present)
                                workdir_str = str(harness_info.get("workdir", ""))
                                if workdir_str and file_path.startswith(workdir_str):
                                    file_path = file_path[len(workdir_str):].lstrip("/")
                                # Also handle absolute paths that might be relative to workdir
                                if file_path.startswith("/"):
                                    try:
                                        workdir_path = Path(workdir_str)
                                        file_path_obj = Path(file_path)
                                        if workdir_path.exists() and file_path_obj.is_relative_to(workdir_path):
                                            file_path = str(file_path_obj.relative_to(workdir_path))
                                    except Exception:
                                        pass
                                # Only add code files (filter out logs, temp files, etc.)
                                from ablation.utils import is_code_file
                                if file_path and is_code_file(file_path) and file_path not in predicted_files:
                                    predicted_files.append(file_path)
                    except:
                        print(f"[DEBUG] Tool call {i} details: function={func_name}, args={args_str[:200]}", file=sys.stderr, flush=True)
                
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
                
                tool_results = tool_runtime.handle_tool_calls(msg.tool_calls)
                
                # Truncate tool result content to prevent context overflow
                for tr in tool_results:
                    if tr.get("role") == "tool":
                        content = tr.get("content", "")
                        if isinstance(content, str) and len(content) > 10000:
                            truncated = content[:5000] + "\n\n[... truncated ...]\n\n" + content[-500:]
                            tr["content"] = truncated
                            print(f"[WARN] Truncated tool result from {len(content)} to {len(truncated)} chars", file=sys.stderr, flush=True)
                        
                        # G2: Track symbol blocks read
                        tool_name = tr.get("name", "")
                        if tool_name in ["read_span", "symbol_lookup"]:
                            symbol_blocks_read += 1
                        
                        # Extract file paths from tool calls for File Hit@k metric
                        # (This is done in the tool call processing loop above, not here)
                
                messages.extend(tool_results)
                
                # Check message count and truncate if needed (keep last 20 messages)
                if len(messages) > 30:
                    print(f"[WARN] Message count ({len(messages)}) exceeds limit, truncating old messages...", file=sys.stderr, flush=True)
                    # Keep system messages and last 20 messages
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    other_msgs = [m for m in messages if m.get("role") != "system"]
                    # Keep first few (harness result, localize prompt) and last 15
                    keep_first = 3  # Keep harness result and initial prompts
                    keep_last = 15
                    
                    if len(other_msgs) > keep_first + keep_last:
                        # Find the cut point (where to start keeping messages)
                        cut_point = len(other_msgs) - keep_last
                        
                        # Ensure we don't cut in the middle of an assistant-tool sequence
                        # Scan backwards from cut_point to find a safe place to cut
                        safe_cut_point = cut_point
                        for i in range(cut_point - 1, keep_first - 1, -1):
                            # If we find an assistant message with tool_calls before cut_point
                            if other_msgs[i].get("role") == "assistant" and other_msgs[i].get("tool_calls"):
                                tool_calls = other_msgs[i].get("tool_calls", [])
                                tool_call_count = len(tool_calls) if tool_calls else 0
                                
                                # Count tool responses after this assistant
                                tool_responses_after = 0
                                for j in range(i + 1, len(other_msgs)):
                                    if other_msgs[j].get("role") == "tool":
                                        tool_responses_after += 1
                                    else:
                                        break
                                
                                # If this assistant's tool responses extend beyond cut_point, we need to include them
                                if tool_responses_after > 0 and i + tool_responses_after >= cut_point:
                                    # This assistant's tool responses are in the range we want to keep
                                    # So we should cut before this assistant message
                                    safe_cut_point = i
                                    break
                        
                        kept_msgs = other_msgs[:keep_first] + other_msgs[safe_cut_point:]
                    else:
                        kept_msgs = other_msgs  # Keep all if not enough to truncate
                    
                    # Final safety check: check ALL messages in kept_msgs for incomplete sequences
                    print(f"[DEBUG] Final safety check: checking {len(kept_msgs)} kept messages", file=sys.stderr, flush=True)
                    messages_to_remove = []
                    for i in range(len(kept_msgs)):
                        msg = kept_msgs[i]
                        if msg.get("role") == "assistant":
                            tool_calls = msg.get("tool_calls")
                            if tool_calls is not None:
                                if hasattr(tool_calls, '__len__'):
                                    tool_call_count = len(tool_calls)
                                else:
                                    tool_call_count = 0
                                
                                if tool_call_count > 0:
                                    # Count tool responses after this assistant
                                    tool_responses_after = 0
                                    for j in range(i + 1, len(kept_msgs)):
                                        if kept_msgs[j].get("role") == "tool":
                                            tool_responses_after += 1
                                        else:
                                            break
                                    
                                    if tool_responses_after < tool_call_count:
                                        # Incomplete sequence - mark for removal
                                        messages_to_remove.append(i)
                                        print(f"[WARN] Found incomplete sequence at kept index {i}: {tool_call_count} tool_calls but only {tool_responses_after} responses", file=sys.stderr, flush=True)
                    
                    # Remove incomplete sequences (from end to start to maintain indices)
                    if messages_to_remove:
                        for idx in reversed(messages_to_remove):
                            # Find the end of this sequence (after all tool responses)
                            seq_end = idx + 1
                            while seq_end < len(kept_msgs) and kept_msgs[seq_end].get("role") == "tool":
                                seq_end += 1
                            # Remove the entire sequence
                            kept_msgs = kept_msgs[:idx] + kept_msgs[seq_end:]
                            print(f"[WARN] Removed incomplete sequence starting at kept index {idx}", file=sys.stderr, flush=True)
                    
                    print(f"[DEBUG] Final safety check complete: {len(kept_msgs)} messages remaining", file=sys.stderr, flush=True)
                    
                    messages = system_msgs + kept_msgs
                    print(f"[INFO] Truncated to {len(messages)} messages (kept {len(system_msgs)} system, {keep_first} initial, {len(kept_msgs) - keep_first} recent)", file=sys.stderr, flush=True)
                
                continue
            else:
                # No tool calls, LLM returned localization result
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
        
        # Parse localization result (simplified for G0)
        # For G0, we expect simple JSON with file and line
        localize_result = None
        localize_result_raw = ""
        try:
            localize_result_raw = messages[-1].get("content", "")
            if localize_result_raw.strip().startswith("{"):
                localize_result = json.loads(localize_result_raw)
                print(f"[INFO] Localization result: {localize_result.get('file', 'N/A')}:{localize_result.get('line', 'N/A')}", file=sys.stderr, flush=True)
                
                # 提取预测的文件（可能单个文件或多个文件）
                from ablation.utils import is_code_file
                pred_file = localize_result.get("file") or localize_result.get("path")
                if pred_file and is_code_file(str(pred_file)):
                    if str(pred_file) not in predicted_files:
                        predicted_files.append(str(pred_file))
                # 支持多文件格式
                if "files" in localize_result and isinstance(localize_result["files"], list):
                    for f in localize_result["files"]:
                        if f and is_code_file(str(f)) and str(f) not in predicted_files:
                            predicted_files.append(str(f))
        except Exception as e:
            print(f"[WARN] Failed to parse localization result: {e}", file=sys.stderr, flush=True)
        
        # 去重并保存（predicted_files已在tool call处理时更新）
        predicted_files = list(dict.fromkeys(predicted_files))  # 保持顺序的去重
        metrics["localization_predicted_files"] = predicted_files
        if predicted_files:
            print(f"[INFO] Localization predicted files ({len(predicted_files)}): {predicted_files[:5]}{'...' if len(predicted_files) > 5 else ''}", file=sys.stderr, flush=True)
        
        # Patch phase
        print("[INFO] Patch phase: generating patch...", file=sys.stderr, flush=True)
        # Q3 TDD Gates: False Start Rate indicator (only meaningful when TDD+RED is enabled).
        # True if we enter patch phase without having confirmed RED failure.
        if config.enable_tdd_gate and config.verify_red_test:
            metrics.setdefault("false_start", False)
            if not metrics.get("tdd_gate_red_verified", False):
                metrics["false_start"] = True
        patch_messages = messages.copy()
        
        # [B] Inject patch context based on localization (file+line). Fallback to RED stack trace if needed.
        try:
            workdir = Path(harness_info.get("workdir", "")).resolve()
            pid = harness_info.get("pid", "")
            bid = harness_info.get("bid", "")
            apr_dir = Path(__file__).resolve().parents[1]
            red_log = apr_dir / "logs" / f"{pid}-{bid}b" / "red.log" if pid and bid else None

            target = _extract_file_line_from_localize(localize_result_raw)
            if not target.get("ok") or not target.get("file"):
                # Prefer failing_tests (usually contains stack trace)
                target = _extract_file_line_from_failing_tests(workdir)
            if (not target.get("ok") or not target.get("file")) and red_log is not None:
                target = _extract_file_line_from_red_log(red_log, workdir)

            if target.get("ok") and target.get("file"):
                rel_file = target.get("file")
                line_no = target.get("line")
                snippet = _read_context_snippet(workdir, rel_file, line_no, radius=80)
                if snippet:
                    ctx_msg = (
                        "PATCH_CONTEXT (use this exact code as ground truth; include proper context lines in your diff hunks):\n"
                        f"TARGET_FILE: {rel_file}\n"
                        f"TARGET_LINE: {line_no}\n\n"
                        f"{snippet}\n"
                    )
                    patch_messages.append({"role": "user", "content": ctx_msg})
                    print(f"[INFO] Injected patch context: {rel_file}:{line_no}", file=sys.stderr, flush=True)
                else:
                    print(f"[WARN] Failed to read context snippet for {rel_file}", file=sys.stderr, flush=True)
            else:
                print("[WARN] No patch context could be extracted from localization or red log", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] Patch context injection failed: {e}", file=sys.stderr, flush=True)

        patch_messages.append({"role": "user", "content": prompts["patch"]})
        
        # Patch-phase limits
        # - max_patch_phase_api_calls: hard cap on total LLM calls in PATCH stage
        # - max_tool_calls_per_patch: soft cap on total tool calls in PATCH stage (to avoid endless "read more" loops)
        tool_call_count = 0  # counts ONLY tool calls in patch phase
        patch_attempt_count = 0  # counts patch outputs (assistant content without tool_calls)
        max_tool_calls_per_patch = getattr(config, 'max_tool_calls_per_patch', 5)
        compile_fail_count = 0  # G3: Track compilation failures separately
        max_compile_failures = config.max_compile_failures
        patch_phase_api_count = 0  # Track total API calls in patch phase
        max_patch_phase_api_calls = config.max_patch_phase_api_calls
        git_apply_fail_count = 0  # Track consecutive git apply failures
        max_git_apply_failures = config.max_git_apply_failures
        consecutive_direct_patches = 0  # Track consecutive direct patch returns
        max_consecutive_direct = config.max_consecutive_direct_patches
        
        # Simple + robust patch feedback mechanism:
        # - Keep only ONE failure summary message (overwrite each time)
        # - Stop early if the same failure repeats consecutively (avoid wasting API)
        last_fail_type = None
        last_fail_sig = None
        repeated_fail_count = 0
        
        def _set_patch_fail_summary(summary: str, fail_type: str, sig: str):
            """Overwrite the single patch failure summary message in patch_messages."""
            nonlocal last_fail_type, last_fail_sig, repeated_fail_count, consecutive_direct_patches
            
            sig_short = (sig or "")[:200]
            if fail_type == last_fail_type and sig_short == last_fail_sig:
                repeated_fail_count += 1
            else:
                repeated_fail_count = 1
                last_fail_type = fail_type
                last_fail_sig = sig_short
            
            # Remove existing summaries (keep context clean)
            cleaned = []
            for m in patch_messages:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, str) and c.startswith("PATCH_FAIL_SUMMARY:"):
                        continue
                cleaned.append(m)
            patch_messages[:] = cleaned
            
            patch_messages.append({
                "role": "user",
                "content": "PATCH_FAIL_SUMMARY:\n"
                           f"type={fail_type}\n"
                           f"repeat={repeated_fail_count}\n\n"
                           f"{summary}".strip()
            })

            # Reset direct-patch loop counter whenever we inject feedback, so we don't prematurely
            # stop due to "infinite loop pattern" while still making progress via feedback.
            consecutive_direct_patches = 0
        
        def _should_stop_due_to_repeat() -> bool:
            # Format errors are often recoverable; allow more retries before stopping.
            if last_fail_type == "format_error":
                return repeated_fail_count >= 4
            return repeated_fail_count >= 2

        force_structured_edits = False
        
        while True:
            # Check timeout in patch loop
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= MAX_RUNTIME_SECONDS:
                print(f"[WARN] Timeout reached during patch generation ({elapsed_time:.1f}s >= {MAX_RUNTIME_SECONDS}s), stopping repair process", file=sys.stderr, flush=True)
                end_time = time.time()
                metrics["runtime_seconds"] = end_time - start_time
                result = {
                    "ok": False,
                    "iterations": it - 1,
                    "patch": last_patch,
                    "error": f"Timeout: exceeded {MAX_RUNTIME_SECONDS}s runtime limit during patch generation",
                    "metrics": metrics
                }
                result["harness_ok"] = harness_info.get("ok", True)
                if not result["harness_ok"]:
                    result["harness_error"] = harness_info.get("error", "Harness failed")
                if "test_suite_verification" in harness_info:
                    result["test_suite_verification"] = harness_info["test_suite_verification"]
                if initial_compile_result is not None:
                    result["compile_result"] = initial_compile_result
                return result
            
            # Check patch phase API call limit
            if patch_phase_api_count >= max_patch_phase_api_calls:
                print(f"[WARN] Reached max patch phase API calls ({max_patch_phase_api_calls}), stopping patch generation...", file=sys.stderr, flush=True)
                break
            
            # Check for infinite loop pattern (consecutive direct patches)
            if consecutive_direct_patches >= max_consecutive_direct:
                print(f"[WARN] Detected infinite loop pattern ({consecutive_direct_patches} consecutive direct patches), stopping patch generation...", file=sys.stderr, flush=True)
                break
            
            print(f"[INFO] Calling LLM API (patch, tool calls: {tool_call_count}, API calls: {patch_phase_api_count}/{max_patch_phase_api_calls})...", file=sys.stderr, flush=True)
            patch_phase_api_count += 1  # Increment API call counter
            
            # Retry mechanism for LLM API calls
            max_retries = 5
            retry_delay = 2  # seconds
            retry_count = 0
            patch_msg = None
            
            while retry_count < max_retries:
                try:
                    resp = client.chat.completions.create(
                        model=api_model_name,
                        messages=patch_messages,
                        tools=tools_schema,
                        tool_choice="auto",
                        timeout=180,
                    )
                    patch_msg = resp.choices[0].message
                    # Only count successful API calls (not retries)
                    if retry_count == 0:
                        metrics["patch_api_calls"] += 1
                        metrics["patch_total_api_calls"] += 1
                        metrics["total_api_calls"] += 1
                    _log_usage(resp, phase="patch", metrics_tracker=metrics)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # credits/quota exhausted：不重试本条，跳过跑下一条。纯 429 走下面 is_rate_limit 重试。
                    if _is_fatal_stop_immediately(error_type, error_msg):
                        print(f"[WARN] LLM API credits/quota exhausted, skip this bug: {error_type}: {error_msg[:300]}", file=sys.stderr, flush=True)
                        return _make_error_result(f"LLM credits/quota exhausted: {error_type}: {error_msg[:200]}", iterations=it, patch=None)
                    
                    # Check if it's a rate limit error (429) or temporary error
                    is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in error_type
                    is_temporary = "timeout" in error_msg.lower() or "503" in error_msg or "502" in error_msg or "500" in error_msg
                    
                    retry_count += 1
                    
                    if is_rate_limit or is_temporary:
                        if retry_count < max_retries:
                            wait_time = _retry_wait_seconds(retry_count, retry_delay, error_msg, is_rate_limit)
                            print(f"[WARN] LLM API call failed ({error_type}): {error_msg[:200]}", file=sys.stderr, flush=True)
                            print(f"[WARN] Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})", file=sys.stderr, flush=True)
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            if is_rate_limit:
                                print(f"[ERROR] LLM API rate limit (429) after {max_retries} retries: {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                            else:
                                print(f"[ERROR] LLM API call failed after {max_retries} retries (timeout/server error): {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                            import traceback
                            print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                            # For patch phase, continue to next tool call attempt instead of failing completely
                            break
                    else:
                        # Non-retryable error
                        print(f"[ERROR] LLM API call failed (non-retryable): {error_type}: {error_msg[:200]}", file=sys.stderr, flush=True)
                        import traceback
                        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                        break
            
            if patch_msg is None:
                print(f"[WARN] Failed to get LLM response after retries, trying next tool call attempt...", file=sys.stderr, flush=True)
                if tool_call_count < max_tool_calls_per_patch - 1:
                    continue
                else:
                    break
            
            # DEBUG: Log LLM response details
            content_length = len(patch_msg.content or "")
            tool_calls_count = len(getattr(patch_msg, "tool_calls", []) or [])
            print(f"[DEBUG] LLM response (patch) - content length: {content_length}, tool_calls: {tool_calls_count}", file=sys.stderr, flush=True)
            if content_length > 0:
                preview = (patch_msg.content or "")[:300] + ("..." if content_length > 300 else "")
                print(f"[DEBUG] LLM content preview (patch): {preview}", file=sys.stderr, flush=True)
            
            if getattr(patch_msg, "tool_calls", None):
                tool_call_count += len(patch_msg.tool_calls)
                metrics["patch_tool_calls"] += len(patch_msg.tool_calls)
                metrics["total_tool_calls"] += len(patch_msg.tool_calls)
                consecutive_direct_patches = 0  # Reset counter when tool calls are made
                
                # Count tool calls by type for patch phase
                for tc in patch_msg.tool_calls:
                    func_name = getattr(tc.function, "name", "unknown")
                    if func_name not in metrics["patch_tool_calls_by_type"]:
                        metrics["patch_tool_calls_by_type"][func_name] = 0
                    metrics["patch_tool_calls_by_type"][func_name] += 1
                    if func_name not in metrics["total_tool_calls_by_type"]:
                        metrics["total_tool_calls_by_type"][func_name] = 0
                    metrics["total_tool_calls_by_type"][func_name] += 1
            else:
                # LLM returned patch directly (no tool_calls)
                patch_attempt_count += 1
                metrics["patch_attempts"] += 1
                consecutive_direct_patches += 1  # Track consecutive direct patches
                print(f"[INFO] LLM returned patch directly (no tool_calls), patch_attempts: {patch_attempt_count}, consecutive direct: {consecutive_direct_patches}", file=sys.stderr, flush=True)
            
            if getattr(patch_msg, "tool_calls", None):
                print(f"[INFO] LLM requested {len(patch_msg.tool_calls)} tool call(s) (total: {tool_call_count})", file=sys.stderr, flush=True)
                
                # DEBUG: Log tool call details
                for i, tc in enumerate(patch_msg.tool_calls, 1):
                    func_name = getattr(tc.function, "name", "unknown")
                    args_str = getattr(tc.function, "arguments", "") or "{}"
                    try:
                        args = json.loads(args_str)
                        print(f"[DEBUG] Tool call {i} details (patch): function={func_name}, args={json.dumps(args, ensure_ascii=False)[:200]}", file=sys.stderr, flush=True)
                    except:
                        print(f"[DEBUG] Tool call {i} details (patch): function={func_name}, args={args_str[:200]}", file=sys.stderr, flush=True)
                
                patch_messages.append({
                    "role": "assistant",
                    "content": patch_msg.content or "",
                    "tool_calls": patch_msg.tool_calls,
                })
                tool_results = tool_runtime.handle_tool_calls(patch_msg.tool_calls)
                
                # Truncate tool result content to prevent context overflow
                for tr in tool_results:
                    if tr.get("role") == "tool":
                        content = tr.get("content", "")
                        if isinstance(content, str) and len(content) > 10000:
                            truncated = content[:5000] + "\n\n[... truncated ...]\n\n" + content[-500:]
                            tr["content"] = truncated
                            print(f"[WARN] Truncated patch tool result from {len(content)} to {len(truncated)} chars", file=sys.stderr, flush=True)
                
                # DEBUG: Log tool call results
                for i, tr in enumerate(tool_results, 1):
                    if tr.get("role") == "tool":
                        tool_name = tr.get("name", "unknown")
                        content = tr.get("content", "")
                        try:
                            result = json.loads(content) if content else {}
                            result_preview = json.dumps(result, ensure_ascii=False)[:300]
                            print(f"[DEBUG] Tool result {i} (patch): {tool_name} -> {result_preview}...", file=sys.stderr, flush=True)
                        except:
                            content_preview = content[:300] if content else ""
                            print(f"[DEBUG] Tool result {i} (patch): {tool_name} -> {content_preview}...", file=sys.stderr, flush=True)
                
                patch_messages.extend(tool_results)
                
                # Keep patch context small and stable (avoid truncating tool-call sequences)
                if len(patch_messages) > 30:
                    print(f"[WARN] Patch message count ({len(patch_messages)}) exceeds limit, truncating old messages...", file=sys.stderr, flush=True)
                    # Keep first few (localization result, patch prompt) and last 20
                    keep_first = 2
                    keep_last = 20
                                        
                    if len(patch_messages) > keep_first + keep_last:
                        # Find safe cut point to avoid cutting in the middle of assistant-tool sequence
                        safe_cut_point = len(patch_messages) - keep_last
                        
                        # Scan backwards to find a safe place to cut
                        for i in range(safe_cut_point - 1, keep_first - 1, -1):
                            if patch_messages[i].get("role") == "assistant" and patch_messages[i].get("tool_calls"):
                                # Found an assistant with tool_calls before cut point
                                tool_calls = patch_messages[i].get("tool_calls", [])
                                tool_call_count = len(tool_calls) if tool_calls else 0
                                
                                # Count tool responses after this assistant
                                tool_responses_after = 0
                                for j in range(i + 1, len(patch_messages)):
                                    if patch_messages[j].get("role") == "tool":
                                        tool_responses_after += 1
                                    else:
                                        break
                                
                                # If tool responses extend beyond cut point, cut before this assistant
                                if tool_responses_after > 0 and i + tool_responses_after >= safe_cut_point:
                                    safe_cut_point = i
                                    break
                                                
                        kept_msgs = patch_messages[:keep_first] + patch_messages[safe_cut_point:]
                    else:
                        kept_msgs = patch_messages  # Keep all if not enough to truncate
                    
                    # Final safety check: check ALL messages in kept_msgs for incomplete sequences
                    print(f"[DEBUG] Final safety check (patch): checking {len(kept_msgs)} kept messages", file=sys.stderr, flush=True)
                    messages_to_remove = []
                    for i in range(len(kept_msgs)):
                        msg = kept_msgs[i]
                        if msg.get("role") == "assistant":
                            tool_calls = msg.get("tool_calls")
                            if tool_calls is not None:
                                if hasattr(tool_calls, '__len__'):
                                    tool_call_count = len(tool_calls)
                                else:
                                    tool_call_count = 0
                                
                                if tool_call_count > 0:
                                    # Count tool responses after this assistant
                                    tool_responses_after = 0
                                    for j in range(i + 1, len(kept_msgs)):
                                        if kept_msgs[j].get("role") == "tool":
                                            tool_responses_after += 1
                                        else:
                                            break
                                    
                                    if tool_responses_after < tool_call_count:
                                        # Incomplete sequence - mark for removal
                                        messages_to_remove.append(i)
                                        print(f"[WARN] Found incomplete sequence at kept index {i}: {tool_call_count} tool_calls but only {tool_responses_after} responses", file=sys.stderr, flush=True)
                    
                    # Remove incomplete sequences (from end to start to maintain indices)
                    if messages_to_remove:
                        for idx in reversed(messages_to_remove):
                            # Find the end of this sequence (after all tool responses)
                            seq_end = idx + 1
                            while seq_end < len(kept_msgs) and kept_msgs[seq_end].get("role") == "tool":
                                seq_end += 1
                            # Remove the entire sequence
                            kept_msgs = kept_msgs[:idx] + kept_msgs[seq_end:]
                            print(f"[WARN] Removed incomplete sequence starting at kept index {idx}", file=sys.stderr, flush=True)
                    
                    print(f"[DEBUG] Final safety check (patch) complete: {len(kept_msgs)} messages remaining", file=sys.stderr, flush=True)
                    
                    patch_messages = kept_msgs
                    print(f"[INFO] Truncated patch messages to {len(patch_messages)} (kept {keep_first} initial, {len(patch_messages) - keep_first} recent)", file=sys.stderr, flush=True)
                
                continue
            
            # LLM returned patch
            patch_text_raw = patch_msg.content or ""
            patch_text = clean_patch_text(patch_text_raw)
            
            # DEBUG: Log patch content preview
            if patch_text:
                print(f"[DEBUG] Patch text length: {len(patch_text)} chars", file=sys.stderr, flush=True)
                patch_preview = patch_text[:500] + ("..." if len(patch_text) > 500 else "")
                print(f"[DEBUG] Patch preview:\n{patch_preview}", file=sys.stderr, flush=True)
            
            if not patch_text:
                print("[WARN] LLM returned empty patch", file=sys.stderr, flush=True)
                _set_patch_fail_summary(
                    "EMPTY_PATCH: Your last response contained no patch. Output either a valid unified diff (starting with 'diff --git') or structured edits JSON.",
                    fail_type="format_error",
                    sig="empty_patch",
                )
                if _should_stop_due_to_repeat():
                    print("[WARN] Repeated empty patch, stopping patch generation early.", file=sys.stderr, flush=True)
                    break
                continue
            
            # Check if patch_text is JSON format (structured edits)
            is_json_format = False
            structured_edits = None
            patch_already_applied = False  # Track if edits were already applied via apply_edits
            try:
                parsed = json.loads(patch_text)
                # Check if it's the structured edits format with "patches" field
                if isinstance(parsed, dict) and "patches" in parsed:
                    is_json_format = True
                    structured_edits = parsed["patches"]
                    print("[INFO] Detected JSON format structured edits, converting to unified diff...", file=sys.stderr, flush=True)
                elif isinstance(parsed, list):
                    # Also support direct list format
                    is_json_format = True
                    structured_edits = parsed
                    print("[INFO] Detected JSON format structured edits (list), converting to unified diff...", file=sys.stderr, flush=True)
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON format, treat as unified diff

            # If we've decided to force structured edits, reject any non-JSON patch output.
            if force_structured_edits and not is_json_format:
                _set_patch_fail_summary(
                    "PATCH_FORMAT_ERROR:\n"
                    "You MUST output STRUCTURED EDITS JSON now. Do NOT output unified diff.\n\n"
                    "Output ONLY JSON (no markdown):\n"
                    "[\n"
                    "  {\n"
                    "    \"path\": \"relative/path/to/file.java\",\n"
                    "    \"ops\": [\n"
                    "      {\"type\": \"replace\", \"start_line\": 10, \"end_line\": 12, \"text\": \"...\\n\"}\n"
                    "    ]\n"
                    "  }\n"
                    "]\n\n"
                    "Rules:\n"
                    "- Output ONLY JSON\n"
                    "- No markdown fences, no explanations\n"
                    "- No '...' placeholders\n",
                    fail_type="format_error",
                    sig="expected_structured_edits_json",
                )
                if _should_stop_due_to_repeat():
                    print("[WARN] Repeated refusal to output structured edits JSON, stopping patch generation early.", file=sys.stderr, flush=True)
                    break
                continue
            
            # If JSON format, handle multi-patch candidates (like apr version)
            if is_json_format and structured_edits:
                workdir = harness_info.get("workdir", "")
                workdir_path = Path(workdir) if workdir else None
                
                if workdir and "apply_edits" in tool_runtime.func_map:
                    # Parse patch candidates (same logic as apr version)
                    try:
                        # structured_edits is already parsed from patch_text
                        # Check if it's the new multi-patch format
                        patch_candidates = []
                        if isinstance(structured_edits, list):
                            # Check if it's multi-patch format: [{"id": 1, "strategy": "...", "edits": [...]}, ...]
                            if len(structured_edits) > 0 and isinstance(structured_edits[0], dict) and "edits" in structured_edits[0]:
                                # Multi-patch format
                                print(f"[INFO] Detected multi-patch format with {len(structured_edits)} candidates", file=sys.stderr, flush=True)
                                for patch_candidate in structured_edits:
                                    if "edits" in patch_candidate:
                                        strategy = patch_candidate.get("strategy", "unknown")
                                        reasoning = patch_candidate.get("reasoning", "")
                                        patch_id = patch_candidate.get("id", len(patch_candidates) + 1)
                                        edits_list = patch_candidate["edits"]
                                        if isinstance(edits_list, list) and len(edits_list) > 0:
                                            patch_candidates.append({
                                                "id": patch_id,
                                                "strategy": strategy,
                                                "reasoning": reasoning,
                                                "edits": edits_list
                                            })
                                            print(f"[INFO] Patch candidate {patch_id}: {strategy} - {reasoning[:50]}...", file=sys.stderr, flush=True)
                            elif len(structured_edits) > 0 and isinstance(structured_edits[0], dict) and "path" in structured_edits[0] and "ops" in structured_edits[0]:
                                # Old format: [{"path": "...", "ops": [...]}]
                                print("[INFO] Detected single-patch format (legacy), converting to candidate list...", file=sys.stderr, flush=True)
                                patch_candidates.append({
                                    "id": 1,
                                    "strategy": "single patch",
                                    "reasoning": "Legacy single patch format",
                                    "edits": structured_edits
                                })
                        
                        if not patch_candidates:
                            raise ValueError("No valid patch candidates found")
                        
                        # Try each patch candidate in order (same as apr version)
                        patch_applied = False
                        patch_already_applied = False
                        compilation_errors_collected = []  # Collect compilation errors for feedback
                        
                        for candidate_idx, candidate in enumerate(patch_candidates, 1):
                            print(f"[INFO] Trying patch candidate {candidate['id']}/{len(patch_candidates)}: {candidate['strategy']}", file=sys.stderr, flush=True)
                            candidate_edits_json = json.dumps(candidate["edits"], ensure_ascii=False)
                            
                            # Apply edits
                            print(f"[INFO] Applying patch candidate {candidate['id']}...", file=sys.stderr, flush=True)
                            apply_result = tool_runtime.func_map["apply_edits"](candidate_edits_json)
                            if not apply_result.get("ok"):
                                error_msg = apply_result.get("error", "unknown error")
                                print(f"[WARN] Patch candidate {candidate['id']} failed to apply: {error_msg}", file=sys.stderr, flush=True)
                                # Continue to next candidate
                                continue
                            
                            # Check if any files were actually modified
                            applied_files = apply_result.get("applied_files", [])
                            applied_files_list = list(applied_files) if applied_files else []
                            if not applied_files_list or apply_result.get("warning"):
                                warning_msg = apply_result.get("warning", "No files were modified")
                                print(f"[WARN] Patch candidate {candidate['id']}: {warning_msg}", file=sys.stderr, flush=True)
                                # Reset and continue to next candidate
                                if workdir_path and workdir_path.exists() and (workdir_path / ".git").exists():
                                    import subprocess
                                    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workdir_path), capture_output=True)
                                continue
                            
                            # Mark that edits have been applied
                            patch_already_applied = True
                            
                            # Get git diff
                            if "get_git_diff" in tool_runtime.func_map:
                                diff_result = tool_runtime.func_map["get_git_diff"]()
                                if not diff_result.get("ok") or not diff_result.get("has_changes"):
                                    print(f"[WARN] Patch candidate {candidate['id']}: No changes detected after applying edits", file=sys.stderr, flush=True)
                                    # Reset and continue to next candidate
                                    if workdir_path and workdir_path.exists() and (workdir_path / ".git").exists():
                                        import subprocess
                                        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workdir_path), capture_output=True)
                                    patch_already_applied = False
                                    continue
                                
                                patch_text = diff_result["diff"]
                                print(f"[INFO] Patch candidate {candidate['id']} applied successfully (diff length: {len(patch_text)} chars)", file=sys.stderr, flush=True)
                                patch_applied = True

                                # Sanity compile check for STRUCTURED EDITS.
                                # Limit to G3/G5 only (i.e., when compile gate is enabled in the variant),
                                # so G0/G1/G2 behavior remains unchanged.
                                if config.enable_patch_compile_gate and config.use_compile_gate:
                                    can_compile_check = False
                                    if adapter is not None and hasattr(adapter, "check_compile"):
                                        can_compile_check = True
                                    elif "check_compile" in tool_runtime.func_map:
                                        can_compile_check = True

                                    if can_compile_check:
                                        print(f"[INFO] Sanity checking compilation for structured-edits patch candidate {candidate['id']}...", file=sys.stderr, flush=True)
                                        try:
                                            if adapter is not None and hasattr(adapter, "check_compile"):
                                                compile_result = adapter.check_compile(workdir)
                                            else:
                                                compile_result = tool_runtime.func_map["check_compile"]()
                                        except Exception as e:
                                            compile_result = {"ok": False, "error_summary": str(e), "rc": -1}

                                        if not compile_result.get("ok"):
                                            error_summary = compile_result.get("error_summary", "") or compile_result.get("stderr", "") or compile_result.get("stdout", "")
                                            print(f"[WARN] Structured-edits patch candidate {candidate['id']} compilation failed (rc={compile_result.get('rc')}): {error_summary[:300]}", file=sys.stderr, flush=True)
                                            compilation_errors_collected.append({
                                                "candidate_id": candidate['id'],
                                                "strategy": candidate.get('strategy', 'unknown'),
                                                "error": error_summary[:1000] if error_summary else "Compilation failed"
                                            })
                                            # Reset and try next candidate
                                            if workdir_path and workdir_path.exists() and (workdir_path / ".git").exists():
                                                import subprocess
                                                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workdir_path), capture_output=True)
                                            patch_text = None
                                            patch_already_applied = False
                                            patch_applied = False
                                            continue
                                        else:
                                            print(f"[INFO] Structured-edits patch candidate {candidate['id']} compilation succeeded!", file=sys.stderr, flush=True)

                                # Compile-gate check (kept for backward compatibility)
                                if config.enable_patch_compile_gate and config.use_compile_gate and "check_compile" in tool_runtime.func_map:
                                    print(f"[INFO] Checking compilation for patch candidate {candidate['id']}...", file=sys.stderr, flush=True)
                                    metrics["compile_attempt_count"] = metrics.get("compile_attempt_count", 0) + 1
                                    compile_result = tool_runtime.func_map["check_compile"]()
                                    if compile_result.get("ok"):
                                        metrics["compile_success_count"] = metrics.get("compile_success_count", 0) + 1
                                    if not compile_result.get("ok"):
                                        error_summary = compile_result.get("error_summary", "") or compile_result.get("stderr", "") or compile_result.get("stdout", "")
                                        print(f"[WARN] Patch candidate {candidate['id']} compilation failed (rc={compile_result.get('rc')}): {error_summary[:300]}", file=sys.stderr, flush=True)
                                        
                                        # Collect compilation error for feedback
                                        compilation_errors_collected.append({
                                            "candidate_id": candidate['id'],
                                            "strategy": candidate.get('strategy', 'unknown'),
                                            "error": error_summary[:1000] if error_summary else "Compilation failed"
                                        })
                                        
                                        # Reset and try next candidate
                                        if workdir_path and workdir_path.exists() and (workdir_path / ".git").exists():
                                            import subprocess
                                            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workdir_path), capture_output=True)
                                        patch_text = None
                                        patch_already_applied = False
                                        patch_applied = False
                                        # Continue to next candidate
                                        continue
                                    else:
                                        print(f"[INFO] Patch candidate {candidate['id']} compilation succeeded!", file=sys.stderr, flush=True)
                                        # This patch compiles, use it
                                        break
                                else:
                                    # No compile check or compile gate disabled, use this patch
                                    break
                            else:
                                print("[WARN] get_git_diff not available, cannot convert structured edits", file=sys.stderr, flush=True)
                                # Reset and continue to next candidate
                                if workdir_path and workdir_path.exists() and (workdir_path / ".git").exists():
                                    import subprocess
                                    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workdir_path), capture_output=True)
                                patch_already_applied = False
                                continue
                        
                        if not patch_applied:
                            # All candidates failed
                            print(f"[WARN] All {len(patch_candidates)} patch candidates failed", file=sys.stderr, flush=True)
                            
                            # Compilation error feedback: if we have compilation errors and still have tool calls, provide feedback
                            if compilation_errors_collected:
                                compile_fail_count += 1
                                metrics["compile_failures"] += 1
                                print(f"[INFO] [G3] Multi-candidate compilation failure count: {compile_fail_count}/{max_compile_failures}", file=sys.stderr, flush=True)
                                
                                # Limit compilation failure retries to prevent infinite loops
                                if compile_fail_count >= max_compile_failures:
                                    print(f"[WARN] [G3] Reached max compilation failures ({max_compile_failures}), breaking patch loop", file=sys.stderr, flush=True)
                                    break
                                
                                if tool_call_count < max_tool_calls_per_patch:
                                    # Keep feedback short; store in single summary message
                                    top_errs = compilation_errors_collected[:2]
                                    err_lines = []
                                    for err_info in top_errs:
                                        err_lines.append(
                                            f"Candidate {err_info['candidate_id']} ({err_info.get('strategy','unknown')}):\n"
                                            f"{(err_info.get('error') or '')[:500]}"
                                        )
                                    error_feedback = (
                                        "COMPILATION_ERROR_FEEDBACK:\n"
                                        f"All {len(patch_candidates)} patch candidates failed to compile.\n\n"
                                        + "\n\n".join(err_lines)
                                        + "\n\nPlease regenerate the patch focusing on fixing the compilation error(s)."
                                    )
                                    _set_patch_fail_summary(
                                        error_feedback,
                                        fail_type="compile_error",
                                        sig=json.dumps(top_errs, ensure_ascii=False),
                                    )
                                    print(f"[INFO] Providing compilation error feedback to LLM for regeneration (tool calls: {tool_call_count}/{max_tool_calls_per_patch}, compile fails: {compile_fail_count}/{max_compile_failures})", file=sys.stderr, flush=True)
                                    if _should_stop_due_to_repeat():
                                        print("[WARN] Repeated same multi-candidate compile error, stopping patch generation early.", file=sys.stderr, flush=True)
                                        break
                                    # Continue tool call loop to let LLM regenerate with error feedback
                                    continue
                            else:
                                # No compilation errors collected or out of tool calls
                                _set_patch_fail_summary(
                                    "ALL_PATCH_CANDIDATES_FAILED:\n"
                                    f"Tried {len(patch_candidates)} strategies, but none worked.\n\n"
                                    "Please regenerate the patch with a different approach and ensure it actually changes the target code.",
                                    fail_type="candidate_error",
                                    sig=f"candidates_failed:{len(patch_candidates)}",
                                )
                                if tool_call_count < max_tool_calls_per_patch:
                                    if _should_stop_due_to_repeat():
                                        print("[WARN] Repeated same candidate failure, stopping patch generation early.", file=sys.stderr, flush=True)
                                        break
                                    continue
                                else:
                                    break
                        else:
                            # Patch applied successfully, continue with git apply --check if needed
                            pass
                            
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"[WARN] Failed to parse patch candidates: {e}", file=sys.stderr, flush=True)
                        _set_patch_fail_summary(
                            f"STRUCTURED_EDITS_PARSE_FAILED:\n{str(e)}\n\nPlease output a valid unified diff (starting with 'diff --git').",
                            fail_type="format_error",
                            sig=str(e),
                        )
                        if tool_call_count < max_tool_calls_per_patch:
                            if _should_stop_due_to_repeat():
                                print("[WARN] Repeated structured-edits parse failure, stopping patch generation early.", file=sys.stderr, flush=True)
                                break
                            continue
                        else:
                            break
                else:
                    print("[WARN] apply_edits or workdir not available, cannot apply structured edits", file=sys.stderr, flush=True)
                    _set_patch_fail_summary(
                        "STRUCTURED_EDITS_TOOL_UNAVAILABLE: apply_edits tool or workdir not available.\n\n"
                        "Please output unified diff format directly.",
                        fail_type="format_error",
                        sig="structured_edits_tool_unavailable",
                    )
                    if tool_call_count < max_tool_calls_per_patch:
                        if _should_stop_due_to_repeat():
                            print("[WARN] Repeated structured-edits tool unavailable, stopping patch generation early.", file=sys.stderr, flush=True)
                            break
                        continue
                    else:
                        break
            
            # G0: Expect unified diff format
            if config.use_unified_diff and not is_json_format:
                # Validate unified diff format (includes is_unified_diff check internally)
                # This replaces the redundant is_unified_diff check that was here before
                v = validate_unified_diff(patch_text)
                if not v.get("ok"):
                    detail = json.dumps(v, ensure_ascii=False)
                    print(f"[WARN] Patch failed unified-diff validation: {detail}", file=sys.stderr, flush=True)
                    # After repeated format failures, switch to structured edits to ensure we can still apply a patch.
                    if not force_structured_edits and repeated_fail_count >= 2:
                        force_structured_edits = True
                        feedback = (
                            "PATCH_FORMAT_ERROR:\n"
                            f"{detail}\n\n"
                            "You have repeatedly produced an invalid unified diff.\n"
                            "Switch output format NOW to STRUCTURED EDITS JSON (no markdown):\n"
                            "[\n"
                            "  {\n"
                            "    \"path\": \"relative/path/to/file.java\",\n"
                            "    \"ops\": [\n"
                            "      {\"type\": \"replace\", \"start_line\": 10, \"end_line\": 12, \"text\": \"...\\n\"}\n"
                            "    ]\n"
                            "  }\n"
                            "]\n\n"
                            "Rules:\n"
                            "- Output ONLY JSON (no markdown, no explanations)\n"
                            "- Use exact line numbers; do not use '...'\n"
                        )
                    else:
                        feedback = (
                            "PATCH_FORMAT_ERROR:\n"
                            f"{detail}\n\n"
                            "Rules:\n"
                            "- Output ONLY unified diff (no markdown, no explanations)\n"
                            "- Do NOT use '...' placeholders\n"
                            "- Ensure @@ hunk header counts match the hunk body\n"
                            "- Ensure the patch ends with a newline\n"
                        )
                    _set_patch_fail_summary(feedback, fail_type="format_error", sig=detail)
                    if _should_stop_due_to_repeat():
                        print("[WARN] Repeated same patch format error, stopping patch generation early.", file=sys.stderr, flush=True)
                        break
                    continue
            
            # Apply patch (only if not already applied via apply_edits)
            # Note: apply_patch_fn internally performs git apply --check, so we don't need to do it separately
            # This keeps the code consistent with apr version
            # Patch apply counters (attempt-level; do NOT compute rates here)
            metrics["apply_attempt_count"] = metrics.get("apply_attempt_count", 0) + 1
            if patch_already_applied:
                print("[INFO] Patch already applied via apply_edits, skipping git apply...", file=sys.stderr, flush=True)
                ap = {"ok": True, "message": "Patch already applied via structured edits"}
            else:
                print("[INFO] Applying patch (apply_patch will perform git apply --check internally)...", file=sys.stderr, flush=True)
                
                # Check if patch_text is None (can happen if patch was reset)
                if patch_text is None:
                    print("[ERROR] patch_text is None, cannot apply patch", file=sys.stderr, flush=True)
                    ap = {"ok": False, "error": "patch_text is None"}
                else:
                    # Extract patch if "diff --git" is not at the start (validate_unified_diff already ensures it exists)
                    # This is a normalization step, not a validation step
                    if not patch_text.strip().startswith("diff --git"):
                        if "diff --git" in patch_text:
                            # Extract from "diff --git" onwards
                            diff_start = patch_text.find("diff --git")
                            patch_text = patch_text[diff_start:]
                            print(f"[INFO] Extracted patch starting from 'diff --git' (length: {len(patch_text)} chars)", file=sys.stderr, flush=True)
                        else:
                            # This should not happen if validate_unified_diff passed, but handle gracefully
                            print("[ERROR] No 'diff --git' found in patch text (validation should have caught this)", file=sys.stderr, flush=True)
                            break

                    # Normalize: git apply is picky about final newline in some cases
                    if patch_text and not patch_text.endswith("\n"):
                        patch_text = patch_text + "\n"

                    # Check workdir exists before applying patch
                    import os
                    workdir = harness_info.get("workdir", "")
                    if workdir and not os.path.exists(workdir):
                        error_msg = f"workdir not found: {workdir} (may have been deleted during execution)"
                        print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
                        # Try to recover by re-checking out the workdir
                        print(f"[INFO] Attempting to recover workdir by re-checking out...", file=sys.stderr, flush=True)
                        try:
                            pid = harness_info.get("pid", "")
                            bid = harness_info.get("bid", "")
                            if pid and bid:
                                if checkout_fn is None and adapter is not None:
                                    checkout_fn = adapter.checkout
                                # Re-checkout the workdir
                                if checkout_fn is not None:
                                    checkout_result = checkout_fn(pid, int(bid), workdir)
                                else:
                                    from agent.adapters import defects4j as d4j
                                    checkout_result = d4j.d4j_checkout(pid, int(bid), workdir)
                                if checkout_result.get("ok"):
                                    print(f"[INFO] Successfully re-checked out workdir: {workdir}", file=sys.stderr, flush=True)
                                    # Update harness_info with new workdir
                                    harness_info["workdir"] = workdir
                                    # Re-register check_compile with the recovered workdir
                                    if "check_compile" in tool_runtime.func_map:
                                        if adapter is not None:
                                            tool_runtime.func_map["check_compile"] = lambda: adapter.check_compile(workdir)
                                        else:
                                            from agent.adapters import defects4j as d4j
                                            tool_runtime.func_map["check_compile"] = lambda: d4j.check_compile(workdir)
                                    ap = apply_patch_fn(patch_text)
                                else:
                                    print(f"[ERROR] Failed to re-checkout workdir: {checkout_result.get('stderr', 'unknown error')}", file=sys.stderr, flush=True)
                                    ap = {"ok": False, "error": error_msg}
                            else:
                                print(f"[ERROR] Cannot recover workdir: missing pid or bid in harness_info", file=sys.stderr, flush=True)
                                ap = {"ok": False, "error": error_msg}
                        except Exception as e:
                            print(f"[ERROR] Exception during workdir recovery: {e}", file=sys.stderr, flush=True)
                            ap = {"ok": False, "error": error_msg}
                    else:
                        ap = apply_patch_fn(patch_text)
            
            # DEBUG: Log patch apply result details
            print(f"[DEBUG] Patch apply result: ok={ap.get('ok')}, error={ap.get('error', 'N/A')}", file=sys.stderr, flush=True)
            if ap.get("stderr"):
                print(f"[DEBUG] Patch apply stderr: {ap.get('stderr')[:500]}", file=sys.stderr, flush=True)
            if ap.get("stdout"):
                print(f"[DEBUG] Patch apply stdout: {ap.get('stdout')[:500]}", file=sys.stderr, flush=True)
            
            if not ap.get("ok"):
                git_apply_fail_count += 1
                metrics["git_apply_failures"] += 1
                print(f"[WARN] Patch apply failed: {ap.get('error', 'unknown error')} (failures: {git_apply_fail_count}/{max_git_apply_failures})", file=sys.stderr, flush=True)
                
                # Check if too many consecutive git apply failures
                if git_apply_fail_count >= max_git_apply_failures:
                    print(f"[WARN] Too many consecutive git apply failures ({git_apply_fail_count}), stopping patch generation...", file=sys.stderr, flush=True)
                    break
                
                # Check if it's a workdir not found error - this is unrecoverable, stop immediately
                error_msg = ap.get('error', 'unknown')
                if 'workdir not found' in error_msg.lower() or 'may have been deleted' in error_msg.lower():
                    print(f"[ERROR] Workdir not found - unrecoverable error, stopping iteration", file=sys.stderr, flush=True)
                    return {
                        "ok": False,
                        "iterations": it,
                        "patch": None,
                        "error": f"Workdir not found: {error_msg}"
                    }
                
                # Check if it's a patch check failure (from git apply --check inside apply_patch)
                if ap.get('check_failed') or 'patch check failed' in error_msg.lower():
                    # Classify the error: format_error vs apply_error
                    stderr_msg = ap.get('stderr', '')
                    error_combined = (error_msg + " " + stderr_msg).lower()
                    
                    # Keep classification strict (avoid guessy heuristics):
                    # - Only explicit parse/format errors => format_error
                    # - Otherwise treat as apply/context mismatch => apply_error
                    format_error_keywords = [
                        'corrupt patch',
                        'invalid patch',
                        'patch fragment without header',
                        'unrecognized input',
                        'malformed patch',
                    ]
                    is_format_error = any(k in error_combined for k in format_error_keywords)
                    is_apply_error = not is_format_error
                    
                    # Generate appropriate feedback
                    if is_format_error:
                        feedback = f"PATCH_FORMAT_ERROR: The patch format is invalid or corrupt.\n\nError: {error_msg}"
                        if stderr_msg:
                            feedback += f"\n\nGit apply stderr:\n{stderr_msg[:500]}"
                        feedback += "\n\nPlease output ONLY a valid unified diff patch starting with 'diff --git', with:\n"
                        feedback += "- No explanation text before or after the patch\n"
                        feedback += "- No markdown code blocks\n"
                        feedback += "- Correct line numbers in @@ lines\n"
                        feedback += "- Complete patch (all lines declared in @@ must be present)\n\n"
                        feedback += "Ensure the patch format matches exactly the example in the prompt."
                    else:
                        # Apply/context mismatch
                        feedback = f"PATCH_APPLY_FAILED: The patch cannot be applied (context mismatch / wrong file / wrong hunk location).\n\nError: {error_msg}"
                        if stderr_msg:
                            feedback += f"\n\nGit apply stderr:\n{stderr_msg[:500]}"
                        feedback += "\n\nPlease:\n"
                        feedback += "1. Re-read the TARGET_FILE snippet in PATCH_CONTEXT (above) and use it as ground truth\n"
                        feedback += "2. Make sure your diff modifies lines that actually exist in that snippet\n"
                        feedback += "3. Include enough context lines (leading space) so git can locate the hunk\n"
                        feedback += "4. Regenerate the patch with correct file path and matching context"
                    _set_patch_fail_summary(feedback, fail_type=("format_error" if is_format_error else "apply_error"), sig=error_msg + "\n" + (stderr_msg or ""))
                    print(f"[INFO] Providing patch error feedback to LLM (format_error={is_format_error}, apply_error={is_apply_error})", file=sys.stderr, flush=True)
                else:
                    # Other apply errors
                    _set_patch_fail_summary(
                        f"PATCH_APPLY_FAILED:\n{error_msg}\n\nPlease regenerate the patch with correct file path and matching context.",
                        fail_type="apply_error",
                        sig=error_msg,
                    )
                if tool_call_count < max_tool_calls_per_patch:
                    if _should_stop_due_to_repeat():
                        print("[WARN] Repeated same patch apply failure, stopping patch generation early.", file=sys.stderr, flush=True)
                        break
                    continue
                else:
                    break
            else:
                # Reset git apply fail count on success
                git_apply_fail_count = 0
                metrics["apply_success_count"] = metrics.get("apply_success_count", 0) + 1
            
            # G3: Compile Gate
            if config.enable_patch_compile_gate and config.use_compile_gate:
                print("[INFO] [G3] Compile Gate: Checking compilation...", file=sys.stderr, flush=True)
                if "check_compile" in tool_runtime.func_map:
                    # Check workdir exists before compiling
                    workdir = harness_info.get("workdir", "")
                    if workdir and not os.path.exists(workdir):
                        error_msg = f"workdir not found: {workdir} (may have been deleted during execution)"
                        print(f"[ERROR] {error_msg} - attempting recovery...", file=sys.stderr, flush=True)
                        # Try to recover by re-checking out the workdir
                        try:
                            pid = harness_info.get("pid", "")
                            bid = harness_info.get("bid", "")
                            if pid and bid:
                                if checkout_fn is None and adapter is not None:
                                    checkout_fn = adapter.checkout
                                if checkout_fn is not None:
                                    checkout_result = checkout_fn(pid, int(bid), workdir)
                                else:
                                    from agent.adapters import defects4j as d4j
                                    checkout_result = d4j.d4j_checkout(pid, int(bid), workdir)
                                if checkout_result.get("ok"):
                                    print(f"[INFO] Successfully recovered workdir: {workdir}", file=sys.stderr, flush=True)
                                    harness_info["workdir"] = workdir
                                    # Re-register check_compile with the recovered workdir
                                    if adapter is not None:
                                        tool_runtime.func_map["check_compile"] = lambda: adapter.check_compile(workdir)
                                    else:
                                        from agent.adapters import defects4j as d4j
                                        tool_runtime.func_map["check_compile"] = lambda: d4j.check_compile(workdir)
                                else:
                                    print(f"[ERROR] Failed to recover workdir: {checkout_result.get('stderr', 'unknown error')}", file=sys.stderr, flush=True)
                                    return {
                                        "ok": False,
                                        "iterations": it,
                                        "patch": None,
                                        "error": f"Workdir not found during compilation: {error_msg}"
                                    }
                            else:
                                return {
                                    "ok": False,
                                    "iterations": it,
                                    "patch": None,
                                    "error": f"Workdir not found during compilation: {error_msg}"
                                }
                        except Exception as e:
                            print(f"[ERROR] Exception during workdir recovery: {e}", file=sys.stderr, flush=True)
                            return {
                                "ok": False,
                                "iterations": it,
                                "patch": None,
                                "error": f"Workdir not found during compilation: {error_msg}"
                            }
                    
                    metrics["compile_attempt_count"] = metrics.get("compile_attempt_count", 0) + 1
                    compile_result = tool_runtime.func_map["check_compile"]()
                    if compile_result.get("ok"):
                        metrics["compile_success_count"] = metrics.get("compile_success_count", 0) + 1
                    # DEBUG: Log compilation result details
                    print(f"[DEBUG] Compilation result: ok={compile_result.get('ok')}, rc={compile_result.get('rc', 'N/A')}", file=sys.stderr, flush=True)
                    if not compile_result.get("ok"):
                        error_summary = compile_result.get("error_summary", "") or compile_result.get("stderr", "") or compile_result.get("stdout", "")
                        print(f"[WARN] [G3] Compilation failed: {error_summary[:300]}", file=sys.stderr, flush=True)
                        # DEBUG: Log full compilation error
                        if error_summary:
                            print(f"[DEBUG] Full compilation error:\n{error_summary[:1000]}", file=sys.stderr, flush=True)
                        # Reset workdir
                        workdir = harness_info.get("workdir", "")
                        if workdir and Path(workdir).exists() and (Path(workdir) / ".git").exists():
                            import subprocess
                            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=workdir, capture_output=True)
                        
                        # Compilation error feedback: provide detailed feedback to LLM for regeneration
                        compile_fail_count += 1
                        print(f"[INFO] [G3] Compilation failure count: {compile_fail_count}/{max_compile_failures}", file=sys.stderr, flush=True)
                        
                        # Limit compilation failure retries to prevent infinite loops
                        if compile_fail_count >= max_compile_failures:
                            print(f"[WARN] [G3] Reached max compilation failures ({max_compile_failures}), breaking patch loop", file=sys.stderr, flush=True)
                            break
                        
                        if tool_call_count < max_tool_calls_per_patch:
                            error_feedback = (
                                "COMPILATION_ERROR_FEEDBACK:\n"
                                "The patch failed to compile. Key error (truncated):\n\n"
                                f"{(error_summary or 'Compilation failed')[:800]}\n\n"
                                "Please regenerate the patch. Focus on fixing the compile error (imports, signatures, syntax)."
                            )
                            _set_patch_fail_summary(error_feedback, fail_type="compile_error", sig=error_summary or "compile_failed")
                            print(f"[INFO] Providing compilation error feedback to LLM for regeneration (tool calls: {tool_call_count}/{max_tool_calls_per_patch}, compile fails: {compile_fail_count}/{max_compile_failures})", file=sys.stderr, flush=True)
                            if _should_stop_due_to_repeat():
                                print("[WARN] Repeated same compile error, stopping patch generation early.", file=sys.stderr, flush=True)
                                break
                            # Continue tool call loop to let LLM regenerate with error feedback
                            continue
                        else:
                            # Out of tool calls, provide basic error message
                            print(f"[WARN] [G3] Out of tool calls, breaking patch loop", file=sys.stderr, flush=True)
                            break
            
            # G3: Generate canonical diff (git diff)
            if config.enable_patch_compile_gate and config.use_canonical_diff:
                print("[INFO] [G3] Generating canonical diff (git diff)...", file=sys.stderr, flush=True)
                if "get_git_diff" in tool_runtime.func_map:
                    diff_result = tool_runtime.func_map["get_git_diff"]()
                    if diff_result.get("ok") and diff_result.get("has_changes"):
                        patch_text = diff_result["diff"]
                        print(f"[INFO] [G3] Canonical diff generated (length: {len(patch_text)} chars)", file=sys.stderr, flush=True)
            
            # We now have a patch applied in workdir. Keep the latest patch for reporting, and then
            # run GREEN + validation INSIDE the patch loop so we can try multiple patches within
            # a single iteration (without increasing max-iters).
            last_patch = patch_text

            def _reset_workdir_to_head():
                workdir = harness_info.get("workdir", "")
                try:
                    if workdir and Path(workdir).exists() and (Path(workdir) / ".git").exists():
                        import subprocess
                        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=workdir, capture_output=True)
                except Exception as e:
                    print(f"[WARN] Failed to reset workdir after patch failure: {e}", file=sys.stderr, flush=True)

            # G1: TDD Gate - Verify GREEN (if enabled)
            if config.enable_tdd_gate and config.verify_green_test:
                # Early rejection indicator is only meaningful when GREEN verification is enabled.
                metrics.setdefault("early_rejection", False)
                print("[INFO] [G1] TDD Gate: Verifying GREEN test...", file=sys.stderr, flush=True)
                if "verify_green" in tool_runtime.func_map:
                    verify_green_fn = tool_runtime.func_map["verify_green"]
                    green_result = verify_green_fn()
                    green_rc = green_result.get("rc")

                    # Check timeout after GREEN test execution (GREEN test may take a long time)
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= MAX_RUNTIME_SECONDS:
                        print(f"[WARN] Timeout reached after GREEN test execution ({elapsed_time:.1f}s >= {MAX_RUNTIME_SECONDS}s), stopping repair process", file=sys.stderr, flush=True)
                        end_time = time.time()
                        metrics["runtime_seconds"] = end_time - start_time
                        result = {
                            "ok": False,
                            "iterations": it - 1,
                            "patch": last_patch,
                            "error": f"Timeout: exceeded {MAX_RUNTIME_SECONDS}s runtime limit after GREEN test execution",
                            "metrics": metrics
                        }
                        result["harness_ok"] = harness_info.get("ok", True)
                        if not result["harness_ok"]:
                            result["harness_error"] = harness_info.get("error", "Harness failed")
                        if "test_suite_verification" in harness_info:
                            result["test_suite_verification"] = harness_info["test_suite_verification"]
                        if initial_compile_result is not None:
                            result["compile_result"] = initial_compile_result
                        return result

                    # DEBUG: Log GREEN test result details
                    print(f"[DEBUG] GREEN test result: rc={green_rc}, test_name={green_result.get('test_name', 'N/A')}, logfile={green_result.get('logfile', 'N/A')}", file=sys.stderr, flush=True)
                    if green_result.get("stderr"):
                        print(f"[DEBUG] GREEN test stderr: {green_result.get('stderr')[:500]}", file=sys.stderr, flush=True)
                    if green_result.get("stdout"):
                        print(f"[DEBUG] GREEN test stdout: {green_result.get('stdout')[:500]}", file=sys.stderr, flush=True)

                    if green_rc != 0:
                        print(f"[FAIL] [G1] GREEN test failed (rc={green_rc}), patch failed TDD Gate validation", file=sys.stderr, flush=True)
                        # Early rejection happened: GREEN failed and we skip full validation for this patch attempt
                        metrics["early_rejection"] = True
                        _set_patch_fail_summary(
                            "GREEN_TEST_FAILED:\n"
                            f"rc={green_rc}\n"
                            f"test_name={green_result.get('test_name', 'N/A')}\n"
                            f"logfile={green_result.get('logfile', 'N/A')}\n\n"
                            "The patch avoided RED failure but did not make the test pass.\n"
                            "Regenerate a new patch (different approach) to make GREEN pass.\n",
                            fail_type="green_failed",
                            sig=f"rc={green_rc};test={green_result.get('test_name','')}",
                        )
                        _reset_workdir_to_head()
                        print(f"[WARN] Patch failed TDD Gate (GREEN test), trying next patch attempt in same iteration...", file=sys.stderr, flush=True)
                        continue
                    else:
                        print(f"[INFO] [G1] GREEN test passed (rc=0)", file=sys.stderr, flush=True)
                        metrics["tdd_gate_green_verified"] = True

            # Full test validation (on current workdir state)
            print("[INFO] Validating patch (running full test suite)...", file=sys.stderr, flush=True)
            try:
                validation_result = validate_fn(last_patch)
            except Exception as validation_err:
                import traceback
                error_msg = str(validation_err)
                error_trace = traceback.format_exc()
                print(f"[ERROR] Validation function raised exception: {error_msg}", file=sys.stderr, flush=True)
                print(f"[ERROR] Validation traceback:\n{error_trace}", file=sys.stderr, flush=True)
                # Return error result instead of crashing
                validation_result = {
                    "passed": False,
                    "error": f"Validation exception: {error_msg}",
                    "exception": error_trace[:1000]  # Truncate long tracebacks
                }
                metrics["validation_failures"] = metrics.get("validation_failures", 0) + 1

            # Check timeout after full test validation (validation may take a long time)
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= MAX_RUNTIME_SECONDS:
                print(f"[WARN] Timeout reached after full test validation ({elapsed_time:.1f}s >= {MAX_RUNTIME_SECONDS}s), stopping repair process", file=sys.stderr, flush=True)
                end_time = time.time()
                metrics["runtime_seconds"] = end_time - start_time
                result = {
                    "ok": False,
                    "iterations": it - 1,
                    "patch": last_patch,
                    "error": f"Timeout: exceeded {MAX_RUNTIME_SECONDS}s runtime limit after full test validation",
                    "metrics": metrics
                }
                result["harness_ok"] = harness_info.get("ok", True)
                if not result["harness_ok"]:
                    result["harness_error"] = harness_info.get("error", "Harness failed")
                if "test_suite_verification" in harness_info:
                    result["test_suite_verification"] = harness_info["test_suite_verification"]
                if initial_compile_result is not None:
                    result["compile_result"] = initial_compile_result
                return result

            # DEBUG: Log validation result details
            print(f"[DEBUG] Validation result: passed={validation_result.get('passed')}", file=sys.stderr, flush=True)
            
            # SWE-bench adapter returns: rc, stdout, stderr
            if "rc" in validation_result:
                rc = validation_result.get("rc", "N/A")
                stdout = validation_result.get("stdout", "")
                stderr = validation_result.get("stderr", "")
                print(f"[DEBUG] SWE-bench validation: rc={rc}", file=sys.stderr, flush=True)
                if stdout:
                    stdout_preview = stdout[-500:] if len(stdout) > 500 else stdout
                    print(f"[DEBUG] SWE-bench stdout (last 500 chars):\n{stdout_preview}", file=sys.stderr, flush=True)
                if stderr:
                    stderr_preview = stderr[-500:] if len(stderr) > 500 else stderr
                    print(f"[DEBUG] SWE-bench stderr (last 500 chars):\n{stderr_preview}", file=sys.stderr, flush=True)
                if "instance_id" in validation_result:
                    print(f"[DEBUG] SWE-bench instance_id: {validation_result.get('instance_id')}", file=sys.stderr, flush=True)
            
            # Defects4J adapter returns: test_full, test_trigger
            if "test_full" in validation_result:
                test_full = validation_result["test_full"]
                test_rc = test_full.get('test_rc', 'N/A')
                rc = test_full.get('rc', 'N/A')
                logfile = test_full.get('logfile', 'N/A')
                print(f"[DEBUG] Test full: test_rc={test_rc}, rc={rc}, logfile={logfile}", file=sys.stderr, flush=True)
                stdout = test_full.get('stdout', '')
                stderr = test_full.get('stderr', '')
                if stdout:
                    stdout_preview = stdout[-500:] if len(stdout) > 500 else stdout
                    print(f"[DEBUG] Test full stdout (last 500 chars):\n{stdout_preview}", file=sys.stderr, flush=True)
                if stderr:
                    stderr_preview = stderr[-500:] if len(stderr) > 500 else stderr
                    print(f"[DEBUG] Test full stderr (last 500 chars):\n{stderr_preview}", file=sys.stderr, flush=True)
            if "test_trigger" in validation_result:
                test_trigger = validation_result["test_trigger"]
                passed = test_trigger.get('passed', 'N/A')
                rc = test_trigger.get('rc', 'N/A')
                logfile = test_trigger.get('logfile', 'N/A')
                print(f"[DEBUG] Test trigger: passed={passed}, rc={rc}, logfile={logfile}", file=sys.stderr, flush=True)
                stdout = test_trigger.get('stdout', '')
                stderr = test_trigger.get('stderr', '')
                if stdout:
                    stdout_preview = stdout[-500:] if len(stdout) > 500 else stdout
                    print(f"[DEBUG] Test trigger stdout (last 500 chars):\n{stdout_preview}", file=sys.stderr, flush=True)
                if stderr:
                    stderr_preview = stderr[-500:] if len(stderr) > 500 else stderr
                    print(f"[DEBUG] Test trigger stderr (last 500 chars):\n{stderr_preview}", file=sys.stderr, flush=True)

            if validation_result.get("passed"):
                print("[SUCCESS] Patch passed validation!", file=sys.stderr, flush=True)
                
                # Extract actual modified files from patch
                from ablation.utils import extract_files_from_patch, calculate_file_hit_at_k
                actual_files = extract_files_from_patch(last_patch) if last_patch else []
                metrics["actual_modified_files"] = actual_files
                
                # Calculate File Hit@k
                predicted_files = metrics.get("localization_predicted_files", [])
                metrics["file_hit_at_1"] = calculate_file_hit_at_k(predicted_files, actual_files, k=1)
                metrics["file_hit_at_3"] = calculate_file_hit_at_k(predicted_files, actual_files, k=3)
                
                if actual_files:
                    print(f"[INFO] Actual modified files: {actual_files}", file=sys.stderr, flush=True)
                    print(f"[INFO] File Hit@1: {metrics['file_hit_at_1']}, File Hit@3: {metrics['file_hit_at_3']}", file=sys.stderr, flush=True)
                
                end_time = time.time()
                metrics["runtime_seconds"] = end_time - start_time
                result = {
                    "ok": True,
                    "iterations": it,
                    "patch": last_patch,
                    "validation": validation_result,
                    "metrics": metrics
                }
                # 记录环境状态（harness是否成功）
                result["harness_ok"] = harness_info.get("ok", True)
                if not result["harness_ok"]:
                    result["harness_error"] = harness_info.get("error", "Harness failed")
                # 添加测试套件验证结果（如果存在）
                if "test_suite_verification" in harness_info:
                    result["test_suite_verification"] = harness_info["test_suite_verification"]
                # 添加编译结果（如果存在）
                if initial_compile_result is not None:
                    result["compile_result"] = initial_compile_result
                return result
            else:
                metrics["validation_failures"] += 1
                print(f"[WARN] Patch failed validation, trying next patch attempt in same iteration...", file=sys.stderr, flush=True)
                # Build detailed feedback message for the LLM
                feedback_parts = ["VALIDATION_FAILED:\nPatch did not pass all tests.\n"]
                
                # For SWE-bench adapter: extract rc, stdout, stderr
                if "rc" in validation_result:
                    rc = validation_result.get("rc", "N/A")
                    stdout = validation_result.get("stdout", "")
                    stderr = validation_result.get("stderr", "")
                    feedback_parts.append(f"Return code: {rc}\n")
                    if stderr:
                        # stderr usually contains the actual error message
                        stderr_clean = stderr.strip()[-1500:]  # Last 1500 chars
                        feedback_parts.append(f"Error output:\n{stderr_clean}\n")
                    if stdout and not stderr:
                        # If no stderr, check stdout for errors
                        stdout_clean = stdout.strip()[-1500:]
                        feedback_parts.append(f"Output:\n{stdout_clean}\n")
                    if "instance_id" in validation_result:
                        feedback_parts.append(f"Instance: {validation_result.get('instance_id')}\n")
                
                # For Defects4J adapter: extract detailed info from test_full and test_trigger
                elif "test_full" in validation_result or "test_trigger" in validation_result:
                    test_full = validation_result.get("test_full", {})
                    test_trigger = validation_result.get("test_trigger", {})
                    
                    if test_full:
                        test_rc = test_full.get("test_rc", "N/A")
                        rc = test_full.get("rc", "N/A")
                        feedback_parts.append(f"Full test suite: test_rc={test_rc}, rc={rc}\n")
                        stderr = test_full.get("stderr", "")
                        if stderr:
                            stderr_clean = stderr.strip()[-1500:]
                            feedback_parts.append(f"Full test stderr:\n{stderr_clean}\n")
                        elif test_full.get("stdout"):
                            stdout_clean = test_full.get("stdout", "").strip()[-1500:]
                            feedback_parts.append(f"Full test output:\n{stdout_clean}\n")
                        logfile = test_full.get("logfile", "")
                        if logfile:
                            feedback_parts.append(f"Full test log: {logfile}\n")
                    
                    if test_trigger:
                        passed = test_trigger.get("passed", False)
                        rc = test_trigger.get("rc", "N/A")
                        feedback_parts.append(f"Trigger tests: passed={passed}, rc={rc}\n")
                        stderr = test_trigger.get("stderr", "")
                        if stderr:
                            stderr_clean = stderr.strip()[-1500:]
                            feedback_parts.append(f"Trigger test stderr:\n{stderr_clean}\n")
                        elif test_trigger.get("stdout"):
                            stdout_clean = test_trigger.get("stdout", "").strip()[-1500:]
                            feedback_parts.append(f"Trigger test output:\n{stdout_clean}\n")
                        logfile = test_trigger.get("logfile", "")
                        if logfile:
                            feedback_parts.append(f"Trigger test log: {logfile}\n")
                else:
                    # Fallback: use JSON dump
                    val_preview = json.dumps(validation_result, ensure_ascii=False)[:800]
                    feedback_parts.append(f"Validation result:\n{val_preview}\n")
                
                feedback_parts.append("\nPlease regenerate a different patch that passes all tests.")
                feedback_msg = "".join(feedback_parts)
                
                _set_patch_fail_summary(
                    feedback_msg,
                    fail_type="validation_failed",
                    sig=json.dumps(validation_result, ensure_ascii=False)[:500],
                )
                _reset_workdir_to_head()
                continue
    
    print(f"[WARN] Reached max iterations ({max_iters}) without success", file=sys.stderr, flush=True)
    
    # Even if failed, try to extract files from last patch if available
    if last_patch:
        try:
            from ablation.utils import extract_files_from_patch, calculate_file_hit_at_k
            actual_files = extract_files_from_patch(last_patch)
            metrics["actual_modified_files"] = actual_files
            predicted_files = metrics.get("localization_predicted_files", [])
            metrics["file_hit_at_1"] = calculate_file_hit_at_k(predicted_files, actual_files, k=1)
            metrics["file_hit_at_3"] = calculate_file_hit_at_k(predicted_files, actual_files, k=3)
        except Exception as e:
            print(f"[WARN] Failed to calculate File Hit@k: {e}", file=sys.stderr, flush=True)
    
    end_time = time.time()
    metrics["runtime_seconds"] = end_time - start_time
    result = {
        "ok": False,
        "iterations": max_iters,
        "patch": last_patch,
        "error": "Reached max iterations without successful patch",
        "metrics": metrics
    }
    # 记录环境状态（harness是否成功）
    result["harness_ok"] = harness_info.get("ok", True)
    if not result["harness_ok"]:
        result["harness_error"] = harness_info.get("error", "Harness failed")
    # 添加测试套件验证结果（如果存在）
    if "test_suite_verification" in harness_info:
        result["test_suite_verification"] = harness_info["test_suite_verification"]
    # 添加编译结果（如果存在）
    if initial_compile_result is not None:
        result["compile_result"] = initial_compile_result
    return result

