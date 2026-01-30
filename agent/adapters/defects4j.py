from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_defects4j_config(workdir_path: Path, *, pid: Optional[str] = None, bid: Optional[int] = None) -> None:
    """
    Ensure `.defects4j.config` exists in a Defects4J workdir.

    Some extracted workdir archives may miss this file (and may or may not include `.git`).
    Defects4J commands like `defects4j compile` require this file to identify the project/version.
    """
    import re
    config_file = workdir_path / ".defects4j.config"
    if config_file.exists():
        return

    # Determine pid/bid from args or parse from directory name (e.g., Chart-7b -> pid=Chart, bid=7, suffix=b)
    vid_suffix = "b"
    if pid is None or bid is None:
        m = re.match(r"^([A-Za-z]+)-(\d+)([bf]?)$", workdir_path.name)
        if not m:
            return
        pid = m.group(1)
        bid = int(m.group(2))
        vid_suffix = m.group(3) or "b"
    vid = f"{bid}{vid_suffix}"

    try:
        config_file.write_text(f"pid={pid}\nvid={vid}\n", encoding="utf-8")
    except Exception:
        # Best-effort: if we can't write it, leave for caller to handle via re-checkout or error.
        return

# Import environment configuration loader
try:
    from dataset.env_config import apply_defects4j_env, get_dataset_version
    _USE_JSON_CONFIG = True
except ImportError:
    _USE_JSON_CONFIG = False

def _run(cmd, cwd=None, env=None):
    """Run a command with optional environment variables.
    
    Note: Environment variables (DEFECTS4J_HOME, PERL5LIB, TZ, etc.) should be
    set in the shell scripts (e.g., bin/run_ablation.sh) and will be inherited
    by subprocess.run() through os.environ. This function only ensures they
    are passed through if explicitly provided in env parameter.
    """
    import os
    if env is None:
        # Use current environment (set by shell scripts)
        env = os.environ.copy()
    else:
        # Merge with current environment to ensure all variables are available
        merged_env = os.environ.copy()
        merged_env.update(env)
        env = merged_env
    
    # Ensure a Defects4J-compatible Java is used for defects4j-related commands.
    # Priority:
    #   1) DEFECTS4J_JAVA_HOME (explicit override)
    #   2) existing JAVA_HOME
    #   3) auto-detect Java 8, then Java 11
    # We also sanitize PATH to avoid picking up Java 17/21 binaries first.
    is_d4j_cmd = any("d4j" in str(c) or "defects4j" in str(c) for c in cmd)
    if not is_d4j_cmd and len(cmd) > 0:
        # Check if the script name suggests it's a defects4j-related script
        script_name = str(cmd[0])
        if "run_one_test" in script_name or "d4j_" in script_name or "defects4j" in script_name:
            is_d4j_cmd = True
    
    if is_d4j_cmd:
        # Use JSON configuration if available, otherwise fall back to hardcoded logic
        if _USE_JSON_CONFIG:
            d4j_env = apply_defects4j_env(overrides=env)
            env.update(d4j_env)
        else:
            # Fallback: original hardcoded logic (for backward compatibility)
            java_home = env.get("DEFECTS4J_JAVA_HOME") or env.get("JAVA_HOME")
            if not java_home:
                repo_java8 = str(REPO_ROOT / ".jdks" / "java8")
                if os.path.exists(repo_java8):
                    java_home = repo_java8
            if not java_home or not os.path.exists(java_home):
                import glob
                java8_dirs = glob.glob("/usr/lib/jvm/java-1.8.0-openjdk*") + glob.glob("/usr/lib/jvm/java-8-openjdk*")
                java11_dirs = glob.glob("/usr/lib/jvm/java-11-openjdk*")
                for cand in (java8_dirs + java11_dirs):
                    if cand and os.path.exists(cand):
                        java_home = cand
                        break

            if java_home and os.path.exists(java_home):
                env["JAVA_HOME"] = java_home
                java_bin = f"{java_home}/bin"
                path_parts = env.get("PATH", "").split(":")
                path_parts = [p for p in path_parts if "java-17" not in p and "java-21" not in p and "java-1.17" not in p and "java-1.21" not in p]
                env["PATH"] = f"{java_bin}:{':'.join(path_parts)}"
            
            if "DEFECTS4J_HOME" not in env:
                d4j_home = os.environ.get("DEFECTS4J_HOME", "")
                if d4j_home and os.path.exists(d4j_home):
                    env["DEFECTS4J_HOME"] = d4j_home
                    env["PATH"] = f"{d4j_home}/framework/bin:{env.get('PATH', '')}"

            perl5lib_parts = []
            perl5_dir = os.environ.get("PERL5_DIR", "")
            if perl5_dir and os.path.exists(perl5_dir):
                perl5_lib = os.path.join(perl5_dir, "lib", "perl5")
                if os.path.exists(perl5_lib):
                    try:
                        for arch_dir in os.listdir(perl5_lib):
                            arch_path = os.path.join(perl5_lib, arch_dir)
                            if os.path.isdir(arch_path):
                                perl5lib_parts.append(arch_path)
                    except Exception:
                        pass
                    perl5lib_parts.append(perl5_lib)

            if "DEFECTS4J_HOME" in env:
                d4j_lib = os.path.join(env["DEFECTS4J_HOME"], "framework", "lib")
                d4j_core = os.path.join(env["DEFECTS4J_HOME"], "framework", "core")
                d4j_util = os.path.join(env["DEFECTS4J_HOME"], "framework", "util")
                for p in (d4j_lib, d4j_core, d4j_util):
                    if os.path.exists(p):
                        perl5lib_parts.append(p)

            if perl5lib_parts:
                perl5lib = ":".join(perl5lib_parts)
                if "PERL5LIB" in env:
                    if perl5lib not in env["PERL5LIB"]:
                        env["PERL5LIB"] = f"{perl5lib}:{env['PERL5LIB']}"
                else:
                    env["PERL5LIB"] = perl5lib
            
            if "TZ" not in env:
                env["TZ"] = "America/Los_Angeles"
    
    # For defects4j commands, if cwd is None but workdir is in cmd, use workdir as cwd
    if cwd is None and len(cmd) >= 3 and cmd[0] == "defects4j" and "-w" in cmd:
        try:
            workdir_idx = cmd.index("-w")
            if workdir_idx + 1 < len(cmd):
                workdir = cmd[workdir_idx + 1]
                if os.path.exists(workdir):
                    cwd = workdir
        except (ValueError, IndexError):
            pass
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def d4j_checkout(pid: str, bid: int, workdir: str) -> Dict[str, Any]:
    r = _run([str(SCRIPTS / "d4j_checkout.sh"), pid, str(bid), workdir])
    return {"ok": r["rc"] == 0, "workdir": r["stdout"].strip(), **r}

def d4j_export_meta(workdir: str, outdir: str) -> Dict[str, Any]:
    r = _run([str(SCRIPTS / "d4j_export_meta.sh"), workdir, outdir])
    return {"ok": r["rc"] == 0, "outdir": r["stdout"].strip(), **r}

def d4j_test_full(workdir: str, logfile: str) -> Dict[str, Any]:
    r = _run([str(SCRIPTS / "d4j_test.sh"), workdir, logfile])
    test_rc = None
    try:
        test_rc = int((r["stdout"] or "").strip().splitlines()[-1])
    except Exception:
        pass
    return {"ran": True, "test_rc": test_rc, "logfile": logfile, **r}

def run_trigger_tests(workdir: str, trigger_file: str, logfile: str) -> Dict[str, Any]:
    r = _run([str(SCRIPTS / "run_trigger_tests.sh"), workdir, trigger_file, logfile])
    return {"ran": True, "passed": r["rc"] == 0, "logfile": logfile, **r}

def run_one_test(workdir: str, test_name: str, logfile: str) -> Dict[str, Any]:
    r = _run([str(SCRIPTS / "run_one_test.sh"), workdir, test_name, logfile])
    test_rc = None
    try:
        # Script outputs the exit code to stdout (not stderr)
        stdout_lines = (r.get("stdout") or "").strip().splitlines()
        if stdout_lines:
            test_rc = int(stdout_lines[-1])
    except (ValueError, IndexError, AttributeError):
        # If parsing fails, we can't determine the test result
        pass
    # Return with test_rc (from stdout), not subprocess rc
    result = {"ran": True, "rc": test_rc, "logfile": logfile, "test_name": test_name}
    result.update({k: v for k, v in r.items() if k != "rc"})  # Don't overwrite with subprocess rc
    return result

def harness(pid: str, bid: int, workdir: str, meta_dir: str, full_log: str, trig_log: str, index_dir: str = None) -> Dict[str, Any]:
    import sys
    import os
    print(f"[HARNESS] Checking out {pid}-{bid}b to {workdir}...", file=sys.stderr, flush=True)
    # Optional fast-path: when workdir is restored from an archive, skip checkout.
    # This avoids re-creating thousands of files and reduces inode pressure.
    # Caller should ensure the extracted workdir is already the correct buggy version.
    skip_checkout = os.environ.get("APR_D4J_SKIP_CHECKOUT", "0") == "1"
    if skip_checkout and (Path(workdir) / ".git").exists():
        print(f"[HARNESS] APR_D4J_SKIP_CHECKOUT=1 and .git exists, skipping defects4j checkout", file=sys.stderr, flush=True)
    else:
        d4j_checkout(pid, bid, workdir)
    
    # Fix compilation configuration issues before running tests
    print(f"[HARNESS] Fixing compilation configuration...", file=sys.stderr, flush=True)
    workdir_path = Path(workdir)
    # Ensure Defects4J workdir marker exists even for archive-extracted dirs.
    _ensure_defects4j_config(workdir_path, pid=pid, bid=bid)
    _fix_compilation_config(workdir_path, log_prefix="[HARNESS]")
    
    # Check if metadata already exists (can be shared across variants)
    # Note: We still need to run tests for each workdir to verify the bug state
    # But we can skip metadata export if it already exists
    meta_dir_path = Path(meta_dir)
    trig_file = str(meta_dir_path / "tests.trigger.txt")
    if meta_dir_path.exists() and Path(trig_file).exists():
        print(f"[HARNESS] Metadata already exists at {meta_dir}, skipping export (but will still run tests)...", file=sys.stderr, flush=True)
        # Still need to run tests to verify the bug state in this workdir
        print(f"[HARNESS] Running full test suite...", file=sys.stderr, flush=True)
        t1 = d4j_test_full(workdir, full_log)
        print(f"[HARNESS] Running trigger tests...", file=sys.stderr, flush=True)
        t2 = run_trigger_tests(workdir, trig_file, trig_log)
    else:
        print(f"[HARNESS] Exporting metadata to {meta_dir}...", file=sys.stderr, flush=True)
        d4j_export_meta(workdir, meta_dir)
        print(f"[HARNESS] Running full test suite...", file=sys.stderr, flush=True)
        t1 = d4j_test_full(workdir, full_log)
        trig_file = str(Path(meta_dir) / "tests.trigger.txt")
        print(f"[HARNESS] Running trigger tests...", file=sys.stderr, flush=True)
        t2 = run_trigger_tests(workdir, trig_file, trig_log)
    
    # Build retrieval index. Support ABCoder backend if enabled via USE_ABCODER_INDEX env var.
    # Keep `index_path` field for backward compatibility.
    index_path = None
    if index_dir:
        index_file = str(Path(index_dir) / "index.json")
        index_file_path = Path(index_file)
        if index_file_path.exists():
            print(f"[HARNESS] Index already exists at {index_file}, skipping build...", file=sys.stderr, flush=True)
            index_path = index_file
        else:
            # Default: try to locate ABCoder index from dataset config paths (scratch_base or repo_root).
            # No need for USE_ABCODER_INDEX env var - we always try to use ABCoder index if available.
            try:
                from dataset.env_config import load_dataset_config, resolve_path_template
                dataset_cfg = load_dataset_config("defects4j")
                scratch_base = dataset_cfg.get("paths", {}).get("scratch_base") or os.environ.get("APR_SCRATCH_BASE", "/tmp/apr_scratch")
                
                bug_id = f"{pid}-{bid}b"
                
                # Use index_dir_template with flat format: {bug_id}_index.json
                index_dir_template = dataset_cfg.get("paths", {}).get("index_dir_template")
                if index_dir_template:
                    index_dir = resolve_path_template(index_dir_template, scratch_base=scratch_base)
                    abcoder_index_path = index_dir / f"{bug_id}_index.json"
                    
                    if abcoder_index_path.exists():
                        print(f"[HARNESS] Using ABCoder raw index: {abcoder_index_path}", file=sys.stderr, flush=True)
                        index_path = str(abcoder_index_path)
                    else:
                        print(f"[HARNESS] WARN: ABCoder index not found at {abcoder_index_path} (index retrieval will be disabled)", file=sys.stderr, flush=True)
                else:
                    print(f"[HARNESS] WARN: index_dir_template not configured (index retrieval will be disabled)", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[HARNESS] WARN: Failed to locate ABCoder index: {e} (index retrieval will be disabled)", file=sys.stderr, flush=True)
    
    print(f"[HARNESS] Harness completed", file=sys.stderr, flush=True)
    return {
        "pid": pid, "bid": bid, "workdir": workdir,
        "meta_dir": meta_dir, "full_log": full_log, "trigger_log": trig_log,
        "test_full": t1, "test_trigger": t2,
        "index_path": index_path
    }


def _fix_compilation_config(workdir_path: Path, log_prefix: str = "[D4J]") -> None:
    """
    Fix common Defects4J compilation placeholders and ensure compile.target/source exist.
    Also fix hardcoded source/target in build.xml files.
    """
    import sys
    import re
    prop_files = [
        workdir_path / "default.properties",
        workdir_path / "defects4j.build.properties",
        workdir_path / "build.properties",
    ]
    for prop_file in prop_files:
        if prop_file.exists():
            try:
                content = prop_file.read_text(encoding="utf-8")
                modified = False
                if "${compile.target}" in content:
                    content = content.replace("${compile.target}", "1.6")
                    modified = True
                if "${compile.source}" in content:
                    content = content.replace("${compile.source}", "1.6")
                    modified = True
                if modified:
                    prop_file.write_text(content, encoding="utf-8")
                    print(f"{log_prefix} Fixed {prop_file.name}: replaced ${{compile.target}}/${{compile.source}} with 1.6", file=sys.stderr, flush=True)
            except Exception:
                pass
    default_props = workdir_path / "default.properties"
    if default_props.exists():
        try:
            content = default_props.read_text(encoding="utf-8")
            changed = False
            if "compile.target" not in content:
                content += "\ncompile.target = 1.6\n"
                changed = True
            if "compile.source" not in content:
                content += "\ncompile.source = 1.6\n"
                changed = True
            if changed:
                default_props.write_text(content, encoding="utf-8")
                print(f"{log_prefix} Added compile.target/compile.source to default.properties", file=sys.stderr, flush=True)
        except Exception:
            pass
    
    # Fix hardcoded source/target in build.xml files (Chart project has hardcoded 1.4, Time project has 1.5)
    build_xml_files = [
        workdir_path / "ant" / "build.xml",
        workdir_path / "build.xml",
    ]
    for build_xml in build_xml_files:
        if build_xml.exists():
            try:
                content = build_xml.read_text(encoding="utf-8")
                original = content
                # Replace hardcoded source="1.4" or source='1.4' with source="1.6"
                content = re.sub(r'source=["\']1\.4["\']', 'source="1.6"', content)
                # Replace hardcoded target="1.4" or target='1.4' with target="1.6"
                content = re.sub(r'target=["\']1\.4["\']', 'target="1.6"', content)
                # Replace hardcoded source="1.5" or source='1.5' with source="1.6"
                content = re.sub(r'source=["\']1\.5["\']', 'source="1.6"', content)
                # Replace hardcoded target="1.5" or target='1.5' with target="1.6"
                content = re.sub(r'target=["\']1\.5["\']', 'target="1.6"', content)
                # Also handle numeric values without quotes (less common)
                content = re.sub(r'source=["\']5["\']', 'source="6"', content)
                content = re.sub(r'target=["\']5["\']', 'target="6"', content)
                if content != original:
                    build_xml.write_text(content, encoding="utf-8")
                    changes = []
                    if "1.4" in original and "1.6" in content:
                        changes.append("1.4->1.6")
                    if "1.5" in original and "1.6" in content:
                        changes.append("1.5->1.6")
                    change_desc = "/".join(changes) if changes else "version"
                    print(f"{log_prefix} Fixed {build_xml.name}: replaced hardcoded source/target {change_desc}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"{log_prefix} WARN: Failed to fix {build_xml}: {e}", file=sys.stderr, flush=True)


def build_index_only(
    pid: str,
    bid: int,
    workdir: str,
    meta_dir: str,
    index_dir: str,
    enable_codeql: bool = False,
    use_abcoder: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Fast path for locating ABCoder indexes WITHOUT running tests.
    Only ABCoder format indexes are supported.
    Intended for prebuilding caches for Lang/Math/Time/Chart/Closure.
    
    Args:
        pid: Project ID
        bid: Bug ID
        workdir: Work directory
        meta_dir: Metadata directory
        index_dir: Index output directory
        enable_codeql: (Deprecated - not used, ABCoder indexes don't need CodeQL augmentation)
        use_abcoder: Use ABCoder index if available (None = check USE_ABCODER_INDEX env var)
    
    Returns:
        Dict with ok, pid, bid, workdir, index_path
    """
    import sys
    import os

    workdir_path = Path(workdir)
    print(f"[INDEX] Checking out {pid}-{bid}b to {workdir}...", file=sys.stderr, flush=True)
    co = d4j_checkout(pid, bid, workdir)
    if not co.get("ok") or not Path(workdir).exists():
        return {"ok": False, "pid": pid, "bid": bid, "workdir": workdir, "error": f"checkout failed: {co.get('stderr','')[:200]}"}

    print(f"[INDEX] Fixing compilation configuration...", file=sys.stderr, flush=True)
    _fix_compilation_config(workdir_path, log_prefix="[INDEX]")

    # Ensure meta dir exists (optional for this flow; keep consistent with other outputs)
    meta_dir_path = Path(meta_dir)
    meta_dir_path.mkdir(parents=True, exist_ok=True)

    # Build Layer1 index.json
    index_dir_path = Path(index_dir)
    index_dir_path.mkdir(parents=True, exist_ok=True)
    index_file = str(index_dir_path / "index.json")
    
    # Default: try to locate ABCoder index from dataset config paths.
    # If use_abcoder is explicitly False, skip; otherwise always try to find it.
    if use_abcoder is False:
        return {
            "ok": False,
            "pid": pid,
            "bid": bid,
            "workdir": workdir,
            "error": "ABCoder index retrieval is disabled (use_abcoder=False).",
        }
    
    # Locate ABCoder raw index (default behavior - no env var required)
    try:
        from dataset.env_config import load_dataset_config, resolve_path_template
        dataset_cfg = load_dataset_config("defects4j")
        scratch_base = dataset_cfg.get("paths", {}).get("scratch_base") or os.environ.get("APR_SCRATCH_BASE", "/tmp/apr_scratch")
        
        bug_id = f"{pid}-{bid}b"
        
        # Use index_dir_template with flat format: {bug_id}_index.json
        index_dir_template = dataset_cfg.get("paths", {}).get("index_dir_template")
        if index_dir_template:
            index_dir = resolve_path_template(index_dir_template, scratch_base=scratch_base)
            abcoder_index_path = index_dir / f"{bug_id}_index.json"
            
            if abcoder_index_path.exists():
                print(f"[INDEX] Using ABCoder raw index: {abcoder_index_path}", file=sys.stderr, flush=True)
                # Return ABCoder raw index path - retrieval tools will parse it directly
                return {"ok": True, "pid": pid, "bid": bid, "workdir": workdir, "index_path": str(abcoder_index_path), "engine": "abcoder_raw"}
            else:
                return {
                    "ok": False,
                    "pid": pid,
                    "bid": bid,
                    "workdir": workdir,
                    "error": f"ABCoder index not found at {abcoder_index_path}. Please build ABCoder index first.",
                }
        else:
            return {
                "ok": False,
                "pid": pid,
                "bid": bid,
                "workdir": workdir,
                "error": "index_dir_template not configured.",
            }
    except Exception as e:
        return {
            "ok": False,
            "pid": pid,
            "bid": bid,
            "workdir": workdir,
            "error": f"Failed to locate ABCoder index: {e}",
        }


def _fix_javac_encoding(workdir_path: Path, *, encoding: str, log_prefix: str = "[D4J]") -> bool:
    """
    Best-effort: inject `encoding="..."` into Ant <javac> tasks in common build.xml locations.
    Returns True if any file was modified.
    """
    import re
    import sys

    candidates = [
        workdir_path / "build.xml",
        workdir_path / "ant" / "build.xml",
    ]
    changed = False
    pat = re.compile(r"<javac\b[^>]*>", re.MULTILINE)

    for p in candidates:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")

            def _repl(m: re.Match) -> str:
                s = m.group(0)
                if "encoding=" in s:
                    return s
                return s[:-1] + f' encoding="{encoding}">'

            new = pat.sub(_repl, text)
            if new != text:
                p.write_text(new, encoding="utf-8")
                changed = True
                print(f"{log_prefix} Patched javac encoding in {p} -> {encoding}", file=sys.stderr, flush=True)
        except Exception:
            continue
    return changed


def _set_ant_compile_level(workdir_path: Path, *, source: str, target: str, log_prefix: str = "[D4J]") -> bool:
    """
    Best-effort: ensure Ant build uses desired -source/-target by writing/patching
    `default.properties` / `defects4j.build.properties` if present.

    Returns True if any file was modified.
    """
    import sys
    import re

    prop_files = [
        workdir_path / "default.properties",
        workdir_path / "defects4j.build.properties",
        workdir_path / "build.properties",
    ]

    changed = False
    for p in prop_files:
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
            orig = txt
            # Normalize: ensure keys exist and set to requested values
            if "compile.source" in txt:
                txt = re.sub(r"(?m)^\s*compile\.source\s*=.*$", f"compile.source = {source}", txt)
            else:
                txt = txt.rstrip() + f"\ncompile.source = {source}\n"
            if "compile.target" in txt:
                txt = re.sub(r"(?m)^\s*compile\.target\s*=.*$", f"compile.target = {target}", txt)
            else:
                txt = txt.rstrip() + f"\ncompile.target = {target}\n"
            if txt != orig:
                p.write_text(txt, encoding="utf-8")
                changed = True
                print(f"{log_prefix} Set compile.source/target in {p.name} -> {source}/{target}", file=sys.stderr, flush=True)
        except Exception:
            continue
    return changed


def check_compile(workdir: str) -> Dict[str, Any]:
    """Check if the code compiles. Returns ok=True if compilation succeeds."""
    import os
    import re
    import shutil
    import sys
    from pathlib import Path
    # Clean build directory before compilation to avoid Java version conflicts
    # (archives may contain build artifacts compiled with different Java versions)
    workdir_path = Path(workdir)
    build_dir = workdir_path / "build"
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass  # Ignore errors, continue with compilation
    
    # Fix missing `.defects4j.config` (required by defects4j compile).
    # Do NOT require `.git` to exist: many archives omit it.
    if not (workdir_path / ".defects4j.config").exists():
        _ensure_defects4j_config(workdir_path)
        if (workdir_path / ".defects4j.config").exists():
            print(f"[INFO] Created missing .defects4j.config in {workdir}", file=sys.stderr, flush=True)
    
    # Use defects4j compile command with proper environment
    # Apply Defects4J environment config (includes Java 8/11 auto-detection)
    env = os.environ.copy()
    if _USE_JSON_CONFIG:
        d4j_env = apply_defects4j_env(overrides=env)
        env.update(d4j_env)
    # Note: For Closure-79b and similar bugs with Java keyword issues, we'll override with Java 8 below if needed.
    
    # Ensure DEFECTS4J_HOME is set (should be set by run_ablation.sh or DEFECTS4J_HOME env)
    if "DEFECTS4J_HOME" not in env:
        d4j_home = os.environ.get("DEFECTS4J_HOME", "")
        if d4j_home and os.path.exists(d4j_home):
            env["DEFECTS4J_HOME"] = d4j_home
            env["PATH"] = f"{d4j_home}/framework/bin:{env.get('PATH', '')}"
    
    # Ensure PERL5LIB is set (PERL5_DIR env or run_ablation.sh)
    perl5lib_parts = []
    perl5_dir = os.environ.get("PERL5_DIR", "")
    if perl5_dir and os.path.exists(perl5_dir):
        perl5_lib = os.path.join(perl5_dir, "lib", "perl5")
        if os.path.exists(perl5_lib):
            # Add architecture-specific paths first (e.g., x86_64-linux-thread-multi)
            try:
                for arch_dir in os.listdir(perl5_lib):
                    arch_path = os.path.join(perl5_lib, arch_dir)
                    if os.path.isdir(arch_path) and not arch_dir.endswith((".0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9")):
                        perl5lib_parts.append(arch_path)
            except Exception:
                pass
            # Then add the base lib/perl5
            perl5lib_parts.append(perl5_lib)
    
    # Then add Defects4J framework paths
    if "DEFECTS4J_HOME" in env:
        d4j_lib = os.path.join(env["DEFECTS4J_HOME"], "framework", "lib")
        d4j_core = os.path.join(env["DEFECTS4J_HOME"], "framework", "core")
        d4j_util = os.path.join(env["DEFECTS4J_HOME"], "framework", "util")
        if os.path.exists(d4j_lib):
            perl5lib_parts.append(d4j_lib)
        if os.path.exists(d4j_core):
            perl5lib_parts.append(d4j_core)
        if os.path.exists(d4j_util):
            perl5lib_parts.append(d4j_util)
    
    if perl5lib_parts:
        perl5lib = ":".join(perl5lib_parts)
        if "PERL5LIB" in env:
            if perl5lib not in env["PERL5LIB"]:
                env["PERL5LIB"] = f"{perl5lib}:{env['PERL5LIB']}"
        else:
            env["PERL5LIB"] = perl5lib
    
    # Set timezone (Defects4J requirement)
    if "TZ" not in env:
        env["TZ"] = "America/Los_Angeles"
    
    # Fix build.properties if it has ${compile.target} placeholder
    workdir_path = Path(workdir)
    prop_files = [
        workdir_path / "default.properties",
        workdir_path / "defects4j.build.properties",
        workdir_path / "build.properties"
    ]
    for prop_file in prop_files:
        if prop_file.exists():
            try:
                content = prop_file.read_text(encoding="utf-8")
                original_content = content
                if "${compile.target}" in content:
                    content = content.replace("${compile.target}", "1.6")
                if "${compile.source}" in content:
                    content = content.replace("${compile.source}", "1.6")
                # Also ensure compile.target and compile.source exist
                if "compile.target" not in content:
                    content += "\ncompile.target = 1.6\n"
                if "compile.source" not in content:
                    content += "\ncompile.source = 1.6\n"
                if content != original_content:
                    prop_file.write_text(content, encoding="utf-8")
            except Exception:
                pass  # If we can't fix it, try compiling anyway
    
    def _summarize_error(out: str) -> str:
        # Try to extract key error lines (usually contain "error:" or "BUILD FAILED")
        error_lines = [
            line
            for line in out.splitlines()
            if "error:" in line.lower() or "BUILD FAILED" in line or "Compile failed" in line
        ]
        return "\n".join(error_lines[:10]) if error_lines else out

    def _patch_java14_in_file(p: Path) -> bool:
        """
        Patch a build descriptor to avoid Java 1.4 source/target, which modern javac rejects.
        Only patches files under DEFECTS4J_HOME/framework/projects or within the checked-out workdir.
        """
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        original = txt

        # XML attributes
        txt = re.sub(r'source=(["\'])1\.4\1', r'source=\g<1>1.6\g<1>', txt)
        txt = re.sub(r'target=(["\'])1\.4\1', r'target=\g<1>1.6\g<1>', txt)

        # Maven-era properties used by some Defects4J project descriptors
        txt = re.sub(r'(^\s*maven\.compile\.source\s*=\s*)1\.4(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)
        txt = re.sub(r'(^\s*maven\.compile\.target\s*=\s*)1\.4(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)

        if txt == original:
            return False

        try:
            p.write_text(txt, encoding="utf-8")
            return True
        except Exception:
            return False

    def _patch_java15_in_file(p: Path) -> bool:
        """
        Patch a build descriptor to avoid Java 1.5 source/target, which modern javac rejects.
        Only patches files under DEFECTS4J_HOME/framework/projects or within the checked-out workdir.
        """
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        original = txt

        # XML attributes: source="1.5" or source='1.5' -> source="1.6"
        txt = re.sub(r'source=(["\'])1\.5\1', r'source=\g<1>1.6\g<1>', txt)
        txt = re.sub(r'target=(["\'])1\.5\1', r'target=\g<1>1.6\g<1>', txt)
        
        # Also handle numeric values without quotes (less common but possible)
        txt = re.sub(r'source=(["\'])5\1', r'source=\g<1>6\g<1>', txt)
        txt = re.sub(r'target=(["\'])5\1', r'target=\g<1>6\g<1>', txt)

        # Ant properties: compile.source=1.5 -> compile.source=1.6
        txt = re.sub(r'(compile\.source\s*=\s*)1\.5(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)
        txt = re.sub(r'(compile\.target\s*=\s*)1\.5(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)

        # Maven-era properties
        txt = re.sub(r'(^\s*maven\.compile\.source\s*=\s*)1\.5(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)
        txt = re.sub(r'(^\s*maven\.compile\.target\s*=\s*)1\.5(\s*$)', r"\g<1>1.6\g<2>", txt, flags=re.M)

        if txt == original:
            return False

        try:
            p.write_text(txt, encoding="utf-8")
            return True
        except Exception:
            return False

    # Check for known bugs with Java keyword issues (pre-emptively use Java 8)
    # Some Closure bugs have '_' as identifier in test code, which fails with Java 9+
    workdir_name = workdir_path.name
    # Known bugs with Java keyword issues (bid list)
    known_keyword_bug_ids = {79, 80, 81, 82, 83}
    # Extract pid and bid from workdir name (e.g., "Closure-79b" -> "Closure", 79)
    use_java8_preemptively = False
    match = re.match(r"^([A-Za-z]+)-(\d+)([bf]?)$", workdir_name)
    if match:
        pid = match.group(1)
        bid = int(match.group(2))
        if pid == "Closure" and bid in known_keyword_bug_ids:
            use_java8_preemptively = True
    
    if use_java8_preemptively:
        print(f"[D4J] Known Java keyword issue bug detected, using Java 8 preemptively...", file=sys.stderr, flush=True)
        import glob
        java8_home = None
        
        # Priority 1: Check DEFECTS4J_JAVA_HOME environment variable
        if "DEFECTS4J_JAVA_HOME" in env:
            java8_home = env["DEFECTS4J_JAVA_HOME"]
            print(f"[D4J] Found Java 8 from DEFECTS4J_JAVA_HOME: {java8_home}", file=sys.stderr, flush=True)
        
        # Priority 2: Check system paths
        if not java8_home or not os.path.exists(java8_home):
            java8_dirs = glob.glob("/usr/lib/jvm/java-1.8.0-openjdk*") + glob.glob("/usr/lib/jvm/java-8-openjdk*")
            if java8_dirs:
                java8_home = java8_dirs[0]
                print(f"[D4J] Found Java 8 in system: {java8_home}", file=sys.stderr, flush=True)
        
        # Priority 3: Try repo-local Java 8 (try both apr_new/.jdks and tdel/.jdks)
        if not java8_home or not os.path.exists(java8_home):
            # Try apr_new/.jdks first
            repo_java8 = REPO_ROOT / ".jdks" / "java8"
            if repo_java8.exists():
                java8_home = str(repo_java8.resolve())
                print(f"[D4J] Found Java 8 in apr_new/.jdks: {java8_home}", file=sys.stderr, flush=True)
            else:
                # Try tdel/.jdks (parent directory)
                tdel_java8 = REPO_ROOT.parent / ".jdks" / "java8"
                if tdel_java8.exists():
                    java8_home = str(tdel_java8.resolve())
                    print(f"[D4J] Found Java 8 in tdel/.jdks: {java8_home}", file=sys.stderr, flush=True)
        
        # Priority 4: Try additional common paths
        if not java8_home or not os.path.exists(java8_home):
            additional_paths = [
                "/usr/java/jdk1.8.0",
                "/opt/java/jdk1.8.0",
                "/usr/local/java/jdk1.8.0",
                "/opt/jdk/jdk1.8.0",
                "/usr/local/jdk/jdk1.8.0",
                str(REPO_ROOT / ".jdks" / "java8"),  # apr_new/.jdks
                str(REPO_ROOT.parent / ".jdks" / "java8"),  # tdel/.jdks
            ]
            for path_str in additional_paths:
                path = Path(path_str)
                if path.exists():
                    java_bin = path / "bin" / "java"
                    if java_bin.exists():
                        java8_home = str(path.resolve())
                        print(f"[D4J] Found Java 8 in additional path: {java8_home}", file=sys.stderr, flush=True)
                        break
        
        if java8_home and os.path.exists(java8_home):
            # Verify Java 8 is actually Java 8
            java8_java = os.path.join(java8_home, "bin", "java")
            if os.path.exists(java8_java):
                # Update env to use Java 8 before first compilation
                env["JAVA_HOME"] = java8_home
                java_bin = f"{java8_home}/bin"
                path_parts = env.get("PATH", "").split(":")
                # Remove all Java versions from PATH, then prepend Java 8
                path_parts = [p for p in path_parts if "java-" not in p and "jvm" not in p]
                env["PATH"] = f"{java_bin}:{':'.join(path_parts)}"
                # Also set DEFECTS4J_JAVA_HOME to ensure defects4j uses Java 8
                env["DEFECTS4J_JAVA_HOME"] = java8_home
                print(f"[D4J] Using Java 8: {java8_home}", file=sys.stderr, flush=True)
                print(f"[D4J] JAVA_HOME={java8_home}, PATH starts with {java_bin}", file=sys.stderr, flush=True)
            else:
                print(f"[D4J] WARN: Java 8 directory found but java binary missing: {java8_java}", file=sys.stderr, flush=True)
        else:
            print(f"[D4J] WARN: Java 8 not found. Searched in /usr/lib/jvm/ and {REPO_ROOT / '.jdks' / 'java8'}", file=sys.stderr, flush=True)
            print(f"[D4J] WARN: Compilation may fail due to Java keyword '_' issue (Java 9+).", file=sys.stderr, flush=True)
    
    # First compile attempt
    r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
    compile_ok = r["rc"] == 0
    combined = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
    error_output = _summarize_error(combined) if not compile_ok else ""

    # Some extracted archives are incomplete and are not valid Defects4J working directories.
    # In that case, fall back to a fresh Defects4J checkout and retry once.
    if (not compile_ok) and ("not a Defects4J working directory" in combined):
        print(f"[WARN] Defects4J reports non-workdir at {workdir}; attempting re-checkout and retry compile...", file=sys.stderr, flush=True)
        workdir_name = workdir_path.name
        match = re.match(r"^([A-Za-z]+)-(\d+)([bf]?)$", workdir_name)
        if match:
            pid2 = match.group(1)
            bid2 = int(match.group(2))
            try:
                checkout_result = d4j_checkout(pid2, bid2, str(workdir_path))
                if checkout_result.get("ok"):
                    # Re-apply compilation config fixes after checkout (e.g., source/target placeholders).
                    _fix_compilation_config(workdir_path, log_prefix="[D4J]")
                    # Retry compilation once
                    r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
                    compile_ok = r["rc"] == 0
                    combined = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
                    error_output = _summarize_error(combined) if not compile_ok else ""
                    if compile_ok:
                        print(f"[INFO] Re-checkout fixed non-workdir compile issue for {pid2}-{bid2}b", file=sys.stderr, flush=True)
                else:
                    print(f"[WARN] Re-checkout failed for {pid2}-{bid2}b: {(checkout_result.get('stderr') or '')[:200]}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARN] Re-checkout exception for {pid2}-{bid2}b: {e}", file=sys.stderr, flush=True)

    # Environment-level auto-fix: some Defects4J projects contain non-UTF8 source files, and Ant/javac may
    # default to UTF-8 depending on locale. If compilation fails with "unmappable character for encoding UTF8",
    # retry once with a safer default file encoding for the Ant JVM.
    needs_encoding_fix = (not compile_ok) and (
        "unmappable character for encoding UTF8" in combined
        or "unmappable character for encoding UTF-8" in combined
    )
    if needs_encoding_fix:
        try:
            # Prefer deterministic fix: patch Ant build.xml to specify the correct source encoding.
            # This avoids relying on JVM default charset / ANT_OPTS behavior under Defects4J wrappers.
            _fix_javac_encoding(workdir_path, encoding="ISO-8859-1", log_prefix="[D4J]")
            print("[WARN] Detected source encoding issue; retrying defects4j compile after patching <javac encoding=...>", file=sys.stderr, flush=True)
            r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
            compile_ok = r["rc"] == 0
            combined = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            error_output = _summarize_error(combined) if not compile_ok else ""
            if compile_ok:
                print("[INFO] Compilation succeeded after encoding retry", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] Encoding retry failed unexpectedly: {e}", file=sys.stderr, flush=True)

    # Some very old Java code uses identifiers that became keywords in Java 5 (e.g., package name `...lang.enum`).
    # If compilation fails with "'enum' is a keyword", retry once with -source/-target 1.4.
    needs_enum_fix = (not compile_ok) and (
        "as of release 5, 'enum' is a keyword" in combined
        or "as of release 5, \"enum\" is a keyword" in combined
    )
    if needs_enum_fix:
        try:
            _set_ant_compile_level(workdir_path, source="1.4", target="1.4", log_prefix="[D4J]")
            print("[WARN] Detected Java 5 keyword ('enum') issue; retrying defects4j compile with source/target=1.4", file=sys.stderr, flush=True)
            r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
            compile_ok = r["rc"] == 0
            combined = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            error_output = _summarize_error(combined) if not compile_ok else ""
            if compile_ok:
                print("[INFO] Compilation succeeded after enum/source-level retry", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] enum/source-level retry failed unexpectedly: {e}", file=sys.stderr, flush=True)
    
    # Check for Java keyword errors ('_' is a keyword in Java 9+)
    # If detected and not already using Java 8, retry with Java 8
    has_keyword_error = (not compile_ok) and (
        ("'_' is a keyword" in combined) or ("_ is a keyword" in combined) or ("as of release 9, '_' is a keyword" in combined)
    )
    
    if has_keyword_error and not use_java8_preemptively:
        print(f"[D4J] Detected Java keyword error (Java 9+ issue), retrying with Java 8...", file=sys.stderr, flush=True)
        # Force Java 8 for this compilation
        import glob
        java8_dirs = glob.glob("/usr/lib/jvm/java-1.8.0-openjdk*") + glob.glob("/usr/lib/jvm/java-8-openjdk*")
        java8_home = None
        if java8_dirs:
            java8_home = java8_dirs[0]
            print(f"[D4J] Found Java 8 in system: {java8_home}", file=sys.stderr, flush=True)
        else:
            # Try repo-local Java 8 (try both apr_new/.jdks and tdel/.jdks)
            repo_java8 = REPO_ROOT / ".jdks" / "java8"
            if repo_java8.exists():
                java8_home = str(repo_java8.resolve())
                print(f"[D4J] Found Java 8 in apr_new/.jdks: {java8_home}", file=sys.stderr, flush=True)
            else:
                # Try tdel/.jdks (parent directory)
                tdel_java8 = REPO_ROOT.parent / ".jdks" / "java8"
                if tdel_java8.exists():
                    java8_home = str(tdel_java8.resolve())
                    print(f"[D4J] Found Java 8 in tdel/.jdks: {java8_home}", file=sys.stderr, flush=True)
        
        if java8_home and os.path.exists(java8_home):
            # Update env to use Java 8
            env["JAVA_HOME"] = java8_home
            java_bin = f"{java8_home}/bin"
            path_parts = env.get("PATH", "").split(":")
            # Remove all Java versions from PATH, then prepend Java 8
            path_parts = [p for p in path_parts if "java-" not in p and "jvm" not in p]
            env["PATH"] = f"{java_bin}:{':'.join(path_parts)}"
            # Also set DEFECTS4J_JAVA_HOME to ensure defects4j uses Java 8
            env["DEFECTS4J_JAVA_HOME"] = java8_home
            print(f"[D4J] Using Java 8: {java8_home}", file=sys.stderr, flush=True)
            print(f"[D4J] JAVA_HOME={java8_home}, PATH starts with {java_bin}", file=sys.stderr, flush=True)
            
            # Retry compilation with Java 8
            r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
            compile_ok = r["rc"] == 0
            combined2 = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            error_output = _summarize_error(combined2) if not compile_ok else ""
            if compile_ok:
                print(f"[D4J] Compilation succeeded with Java 8", file=sys.stderr, flush=True)
        else:
            print(f"[D4J] WARN: Java 8 not found, cannot fix keyword error", file=sys.stderr, flush=True)

    # Environment-level auto-fix: javac no longer supports -source/-target 1.4 or 1.5 (even on JDK8+).
    # Defects4J framework may hardcode 1.4 or 1.5 in project build descriptors (e.g., Chart.build.xml, Time.build.xml).
    needs_java14_fix = (not compile_ok) and (
        ("Source option 1.4 is no longer supported" in combined)
        or ("Target option 1.4 is no longer supported" in combined)
    )
    needs_java15_fix = (not compile_ok) and (
        ("Source option 5 is no longer supported" in combined)
        or ("Target option 1.5 is no longer supported" in combined)
        or ("Source option 1.5 is no longer supported" in combined)
    )
    
    if needs_java14_fix or needs_java15_fix:
        d4j_home = Path(env.get("DEFECTS4J_HOME", "")).resolve() if env.get("DEFECTS4J_HOME") else None
        projects_root = (d4j_home / "framework" / "projects") if d4j_home else None
        workdir_root = Path(workdir).resolve()

        # Extract file paths mentioned by ant/javac, e.g. ".../Chart.build.xml:57:" or ".../Time.build.xml:200:"
        mentioned = set(re.findall(r'(/[^:\s]+?\.(?:xml|properties)):\d+', combined))
        patched_any = False
        for fp in sorted(mentioned):
            p = Path(fp)
            if not p.exists():
                continue
            try:
                p_res = p.resolve()
            except Exception:
                p_res = p

            if projects_root and str(p_res).startswith(str(projects_root)):
                if needs_java14_fix:
                    patched_any = _patch_java14_in_file(p_res) or patched_any
                if needs_java15_fix:
                    patched_any = _patch_java15_in_file(p_res) or patched_any
            elif str(p_res).startswith(str(workdir_root)):
                if needs_java14_fix:
                    patched_any = _patch_java14_in_file(p_res) or patched_any
                if needs_java15_fix:
                    patched_any = _patch_java15_in_file(p_res) or patched_any
        
        # Also check workdir's build.xml files directly (in case they weren't mentioned in error output)
        if not patched_any:
            for build_xml in [workdir_root / "build.xml", workdir_root / "ant" / "build.xml"]:
                if build_xml.exists():
                    if needs_java14_fix:
                        patched_any = _patch_java14_in_file(build_xml) or patched_any
                    if needs_java15_fix:
                        patched_any = _patch_java15_in_file(build_xml) or patched_any

        if patched_any:
            print(f"[D4J] Auto-fixed Java version issue in build files, retrying compilation...", file=sys.stderr, flush=True)
            # Retry once after patching
            r = _run(["defects4j", "compile", "-w", workdir], cwd=None, env=env)
            compile_ok = r["rc"] == 0
            combined2 = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            error_output = _summarize_error(combined2) if not compile_ok else ""

    return {
        "ok": compile_ok,
        "rc": r["rc"],
        "stdout": r["stdout"][:2000] if r.get("stdout") else "",
        "stderr": r["stderr"][:2000] if r.get("stderr") else "",
        "error_summary": error_output[:2000] if error_output else "",
    }

def validate(pid: str, bid: int, workdir: str, meta_dir: str, full_log: str, trig_log: str) -> Dict[str, Any]:
    trig_file = str(Path(meta_dir) / "tests.trigger.txt")
    t2 = run_trigger_tests(workdir, trig_file, trig_log)
    t1 = d4j_test_full(workdir, full_log)
    
    # Check both test_rc and test.full.log content for failing tests
    test_rc = t1.get("test_rc")
    has_failing_tests = False
    if Path(full_log).exists():
        try:
            log_content = Path(full_log).read_text(encoding="utf-8")
            # Check for "Failing tests: N" pattern where N > 0
            import re
            failing_match = re.search(r"Failing tests:\s*(\d+)", log_content)
            if failing_match:
                failing_count = int(failing_match.group(1))
                has_failing_tests = (failing_count > 0)
        except Exception:
            # If we can't read the log, fall back to test_rc only
            pass
    
    # Passed only if test_rc == 0 AND no failing tests in log
    passed = (test_rc == 0) and (not has_failing_tests)
    
    return {"passed": passed, "test_full": t1, "test_trigger": t2, "full_log": full_log, "trigger_log": trig_log}


from agent.adapters.base import DatasetAdapter
from typing import Optional


class Defects4JAdapter(DatasetAdapter):
    """
    Thin adapter wrapper around the existing Defects4J module-level functions.

    This keeps behavior identical while allowing ablation code to depend on an adapter interface.
    """

    def harness(
        self,
        pid: str,
        bid: int,
        workdir: str,
        meta_dir: str,
        full_log: str,
        trig_log: str,
        index_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        return harness(pid, bid, workdir, meta_dir, full_log, trig_log, index_dir)

    def validate(
        self,
        pid: str,
        bid: int,
        workdir: str,
        meta_dir: str,
        full_log: str,
        trig_log: str,
    ) -> Dict[str, Any]:
        return validate(pid, bid, workdir, meta_dir, full_log, trig_log)

    def check_compile(self, workdir: str) -> Dict[str, Any]:
        return check_compile(workdir)

    def run_one_test(self, workdir: str, test_name: str, log_file: str) -> Dict[str, Any]:
        return run_one_test(workdir, test_name, log_file)

    def checkout(self, pid: str, bid: int, workdir: str) -> Dict[str, Any]:
        return d4j_checkout(pid, bid, workdir)

