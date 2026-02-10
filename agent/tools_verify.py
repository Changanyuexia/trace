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
    # red_log and green_log are required; red_test_name may be empty (some SWE-bench instances)
    if not red_log or not green_log:
        return
    
    # If red_test_name is empty, try to get it from adapter or red_log
    effective_test_name = red_test_name
    if not effective_test_name:
        # Try to get from adapter (e.g. SWE-bench instance metadata)
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
        
        # For Defects4J: if still empty, try meta_dir/tests.trigger.txt or workdir
        if not effective_test_name:
            try:
                from pathlib import Path
                import subprocess
                import sys
                
                # First try to read from meta_dir/tests.trigger.txt
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
                
                # If still empty, try defects4j export from workdir (workdir may not be checkout yet; retry in verify_red)
                if not effective_test_name:
                    workdir_path = Path(workdir)
                    if workdir_path.exists() and (workdir_path / ".defects4j.config").exists():
                        try:
                            import os
                            # Run defects4j export -p tests.trigger
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
                            # On failure, leave effective_test_name unset for deferred resolution
                            pass
            except Exception as e:
                print(f"[WARN] [TDD] Failed to retrieve trigger test name: {e}", file=sys.stderr, flush=True)
    
    # Deferred resolution when verify_red is called (after harness(), workdir checkout)
    def _get_effective_test_name():
        """Resolve test name after workdir is checkout."""
        nonlocal effective_test_name
        
        # If we already have a valid test name, return it
        if effective_test_name and effective_test_name != "unknown_test":
            return effective_test_name
        
        # Deferred: harness() has run, workdir is checkout. Prefer meta_dir/tests.trigger.txt
        if not effective_test_name or effective_test_name == "unknown_test":
            try:
                from pathlib import Path
                import subprocess
                import sys
                import os
                
                # First try to read from meta_dir/tests.trigger.txt (exported by harness())
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
                
                # If meta_dir read failed, try workdir (must have .defects4j.config)
                workdir_path = Path(workdir)
                if workdir_path.exists() and (workdir_path / ".defects4j.config").exists():
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
        
        # If all methods fail, use placeholder
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
        # If test_name provided, use it
        if test_name:
            effective_test_name = test_name
        # If test_class and test_method provided, combine them
        elif test_class and test_method:
            effective_test_name = f"{test_class}::{test_method}"
        # Otherwise use deferred default test name
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
        # If test_name provided, use it
        if test_name:
            effective_test_name = test_name
        # If test_class and test_method provided, combine them
        elif test_class and test_method:
            effective_test_name = f"{test_class}::{test_method}"
        # Otherwise use deferred default test name
        else:
            effective_test_name = _get_effective_test_name()
        
        r = adapter.run_one_test(workdir, effective_test_name, green_log)
        r["test_name"] = effective_test_name
        return r

    func_map["verify_red"] = verify_red
    func_map["verify_green"] = verify_green




