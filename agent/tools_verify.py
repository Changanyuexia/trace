"""
Verify phase tools (LLM-callable).

These tools are intended for VERIFY / TDD-Gate:
- verify_red
- verify_green
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def register_verify_tools(
    func_map: Dict[str, Any],
    *,
    adapter,
    workdir: str,
    red_test_name: Optional[str],
    red_log: Optional[str],
    green_log: Optional[str],
    meta_dir: Optional[str] = None,
) -> None:
    if adapter is None:
        return
    # red_log 和 green_log 是必需的，但 red_test_name 可以为空（某些 SWE-bench 实例可能没有）
    if not red_log or not green_log:
        return
    
    # 如果 red_test_name 为空，尝试从 adapter 或 red_log 中提取
    effective_test_name = red_test_name
    if not effective_test_name:
        # 尝试从 adapter 获取（对于 SWE-bench，可能可以从 instance metadata 获取）
        try:
            if hasattr(adapter, "_get_instance"):
                from pathlib import Path
                parts = Path(workdir).parts
                if "swebench_verified" in parts:
                    idx = parts.index("swebench_verified")
                    # Handle different workdir path structures:
                    # 1. Normal: .../swebench_verified/{instance_id}/workdir
                    # 2. Archive extracted: .../apr_extracted/swebench_verified/{job_tag}/{instance_id}/{instance_id}
                    if "apr_extracted" in parts:
                        # Archive mode: workdir ends with {instance_id}/{instance_id}, so the last part is instance_id
                        instance_id = parts[-1]
                    elif idx + 1 < len(parts):
                        # Normal mode: next part after "swebench_verified" is instance_id
                        instance_id = parts[idx + 1]
                    else:
                        instance_id = None
                    
                    if instance_id:
                        inst = adapter._get_instance(instance_id)
                        if inst:
                            import json
                            fail_to_pass = inst.get("FAIL_TO_PASS", [])
                            if isinstance(fail_to_pass, str):
                                try:
                                    fail_to_pass = json.loads(fail_to_pass)
                                except:
                                    pass
                            if isinstance(fail_to_pass, list) and len(fail_to_pass) > 0:
                                effective_test_name = fail_to_pass[0]
        except Exception:
            pass
        
        # 对于 Defects4J：如果仍然为空，尝试从 meta_dir/tests.trigger.txt 或直接从 workdir 获取
        if not effective_test_name:
            try:
                from pathlib import Path
                import subprocess
                import sys
                
                # 首先尝试从 meta_dir/tests.trigger.txt 读取
                if meta_dir:
                    trig_file = Path(meta_dir) / "tests.trigger.txt"
                    if trig_file.exists():
                        content = trig_file.read_text().strip()
                        if content:
                            lines = content.splitlines()
                            if lines:
                                # Find first method-level test (contains ::)
                                for line in lines:
                                    if "::" in line:
                                        effective_test_name = line.strip()
                                        break
                                # If no method-level test found, use first line
                                if not effective_test_name:
                                    effective_test_name = lines[0].strip()
                                if effective_test_name:
                                    print(f"[INFO] [TDD] Retrieved trigger test from meta_dir: {effective_test_name}", file=sys.stderr, flush=True)
                
                # 如果仍然为空，尝试直接从 workdir 运行 defects4j export
                # 注意：此时 workdir 可能还没有 checkout，所以这里只尝试，如果失败会在 verify_red 调用时重试
                if not effective_test_name:
                    workdir_path = Path(workdir)
                    if workdir_path.exists() and (workdir_path / ".defects4j.config").exists():
                        try:
                            import os
                            # 运行 defects4j export -p tests.trigger
                            result = subprocess.run(
                                ["defects4j", "export", "-p", "tests.trigger"],
                                cwd=str(workdir_path),
                                capture_output=True,
                                text=True,
                                timeout=10,
                                env=dict(os.environ)
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                content = result.stdout.strip()
                                lines = content.splitlines()
                                if lines:
                                    # Find first method-level test (contains ::)
                                    for line in lines:
                                        if "::" in line:
                                            effective_test_name = line.strip()
                                            break
                                    # If no method-level test found, use first line
                                    if not effective_test_name:
                                        effective_test_name = lines[0].strip()
                                    if effective_test_name:
                                        print(f"[INFO] [TDD] Retrieved trigger test from workdir: {effective_test_name}", file=sys.stderr, flush=True)
                        except Exception as e:
                            # 如果失败，不设置 effective_test_name，让延迟获取逻辑处理
                            pass
            except Exception as e:
                print(f"[WARN] [TDD] Failed to retrieve trigger test name: {e}", file=sys.stderr, flush=True)
    
    # 延迟获取：在 verify_red 调用时（此时 harness() 已经运行，workdir 已经 checkout）
    # 这样可以确保 workdir 存在后再尝试从 workdir 或 meta_dir 获取
    def _get_effective_test_name():
        """延迟获取测试名称，确保在 workdir 已经 checkout 后获取"""
        nonlocal effective_test_name
        
        # 如果已经有有效的测试名称，直接返回
        if effective_test_name and effective_test_name != "unknown_test":
            return effective_test_name
        
        # 延迟获取：此时 harness() 已经运行，workdir 已经 checkout
        # 优先从 meta_dir/tests.trigger.txt 读取（harness() 已经导出）
        if not effective_test_name or effective_test_name == "unknown_test":
            try:
                from pathlib import Path
                import subprocess
                import sys
                import os
                
                # 首先尝试从 meta_dir/tests.trigger.txt 读取（harness() 已经导出）
                if meta_dir:
                    trig_file = Path(meta_dir) / "tests.trigger.txt"
                    if trig_file.exists():
                        try:
                            content = trig_file.read_text().strip()
                            if content:
                                lines = content.splitlines()
                                if lines:
                                    # Find first method-level test (contains ::)
                                    for line in lines:
                                        if "::" in line:
                                            effective_test_name = line.strip()
                                            break
                                    # If no method-level test found, use first line
                                    if not effective_test_name or effective_test_name == "unknown_test":
                                        effective_test_name = lines[0].strip()
                                    if effective_test_name and effective_test_name != "unknown_test":
                                        print(f"[INFO] [TDD] Retrieved trigger test from meta_dir (delayed): {effective_test_name}", file=sys.stderr, flush=True)
                                        return effective_test_name
                        except Exception as e:
                            print(f"[WARN] [TDD] Failed to read meta_dir/tests.trigger.txt: {e}", file=sys.stderr, flush=True)
                
                # 如果 meta_dir 读取失败，尝试从 workdir 直接获取
                workdir_path = Path(workdir)
                # 确保 workdir 已经 checkout（有 .defects4j.config）
                if workdir_path.exists() and (workdir_path / ".defects4j.config").exists():
                    # 运行 defects4j export -p tests.trigger
                    result = subprocess.run(
                        ["defects4j", "export", "-p", "tests.trigger"],
                        cwd=str(workdir_path),
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env=dict(os.environ)
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        content = result.stdout.strip()
                        lines = content.splitlines()
                        if lines:
                            # Find first method-level test (contains ::)
                            for line in lines:
                                if "::" in line:
                                    effective_test_name = line.strip()
                                    break
                            # If no method-level test found, use first line
                            if not effective_test_name or effective_test_name == "unknown_test":
                                effective_test_name = lines[0].strip()
                            if effective_test_name and effective_test_name != "unknown_test":
                                print(f"[INFO] [TDD] Retrieved trigger test from workdir (delayed): {effective_test_name}", file=sys.stderr, flush=True)
                                return effective_test_name
            except Exception as e:
                print(f"[WARN] [TDD] Failed to get trigger test (delayed): {e}", file=sys.stderr, flush=True)
        
        # 如果所有方法都失败，使用占位符
        if not effective_test_name or effective_test_name == "unknown_test":
            print(f"[WARN] [TDD] Could not determine trigger test name, using 'unknown_test' placeholder", file=sys.stderr, flush=True)
            effective_test_name = "unknown_test"
        
        return effective_test_name

    def verify_red(test_name: Optional[str] = None, test_class: Optional[str] = None, test_method: Optional[str] = None):
        """
        Verify RED test (test should fail).
        
        Args:
            test_name: Full test name (e.g., "ClassName::methodName" for Defects4J)
            test_class: Test class name (optional, will be combined with test_method if provided)
            test_method: Test method name (optional, will be combined with test_class if provided)
        
        Returns:
            Dict with test execution result.
        """
        # 如果提供了 test_name，直接使用
        if test_name:
            effective_test_name = test_name
        # 如果提供了 test_class 和 test_method，组合它们
        elif test_class and test_method:
            effective_test_name = f"{test_class}::{test_method}"
        # 否则使用延迟获取的默认测试名称
        else:
            effective_test_name = _get_effective_test_name()
        
        r = adapter.run_one_test(workdir, effective_test_name, red_log)
        r["test_name"] = effective_test_name
        return r

    def verify_green(test_name: Optional[str] = None, test_class: Optional[str] = None, test_method: Optional[str] = None):
        """
        Verify GREEN test (test should pass after patch).
        
        Args:
            test_name: Full test name (e.g., "ClassName::methodName" for Defects4J)
            test_class: Test class name (optional, will be combined with test_method if provided)
            test_method: Test method name (optional, will be combined with test_class if provided)
        
        Returns:
            Dict with test execution result.
        """
        # 如果提供了 test_name，直接使用
        if test_name:
            effective_test_name = test_name
        # 如果提供了 test_class 和 test_method，组合它们
        elif test_class and test_method:
            effective_test_name = f"{test_class}::{test_method}"
        # 否则使用延迟获取的默认测试名称
        else:
            effective_test_name = _get_effective_test_name()
        
        r = adapter.run_one_test(workdir, effective_test_name, green_log)
        r["test_name"] = effective_test_name
        return r

    func_map["verify_red"] = verify_red
    func_map["verify_green"] = verify_green




