"""
Main entry point for ablation study experiments.

This module provides a configurable entry point that can run different variants
(G0, G1, G2, G3, TRACE) of the APR system for ablation studies.
"""
import os
import sys
from pathlib import Path

# trace/ is self-contained: use trace's dataset config (paths under TRACE_WORK_ROOT).
# apr_new/ is only for agent.* / llm (trace has no agent.llm).
TRACE_ROOT = Path(__file__).resolve().parents[1]
APR_ROOT = TRACE_ROOT.parent / "apr_new"
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))
if str(APR_ROOT) not in sys.path:
    sys.path.insert(0, str(APR_ROOT))

import argparse
from dotenv import load_dotenv

from agent.llm import make_client
from agent.tools_common import apply_patch
from ablation.core_ablation import run_agent_loop_ablation
from ablation.config import AblationConfig
from ablation.variant_loader import load_variant
from ablation.dataset_loader import load_dataset_config, get_adapter, get_paths
from ablation.tools import setup_tools

def main():
    parser = argparse.ArgumentParser(description="APR Ablation Study")
    parser.add_argument("--dataset", type=str, default="defects4j", help="Dataset name")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory (default: from TRACE_WORK_ROOT + dataset config)")
    parser.add_argument("--pid", type=str, required=True, help="Project ID")
    parser.add_argument("--bid", type=int, required=True, help="Bug ID")
    parser.add_argument(
        "--variant",
        type=str,
        default="G0",
        choices=["G0", "G1", "G2", "G3", "TRACE"],
        help="Ablation variant (G0=baseline, G1=+TDD, G2=+Index, G3=+Patch/Compile, TRACE=full)",
    )
    parser.add_argument("--max-iters", type=int, default=0, help="Maximum iterations (0 = harness/verify only, no patch loop)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name")
    
    args = parser.parse_args()
    
    # Load environment (prefer trace/.env, fallback to apr_new/.env)
    env_file = TRACE_ROOT / ".env"
    if not env_file.exists():
        env_file = APR_ROOT / ".env"
    load_dotenv(env_file)
    
    # Load variant (config + prompts)
    variant_cfg, prompts = load_variant(args.variant)
    config = AblationConfig.from_dict(variant_cfg)
    print(f"[INFO] Running ablation variant: {args.variant}", file=sys.stderr, flush=True)
    print(f"[INFO] Config: {config.to_dict()}", file=sys.stderr, flush=True)
    
    # Setup LLM client
    client = make_client(args.model)

    # Dataset adapter + paths
    dataset_cfg = load_dataset_config(args.dataset)
    adapter = get_adapter(args.dataset)
    ds_paths = get_paths(dataset_cfg, pid=args.pid, bid=args.bid)
    
    # Paths: all under TRACE_WORK_ROOT (from dataset config)
    work_root = os.environ.get("TRACE_WORK_ROOT", "/tmp/trace_work")
    if args.workdir:
        workdir = args.workdir
    else:
        workdir = ds_paths.get("workdir") or str(Path(work_root) / "workdirs" / "defects4j" / f"{args.pid}-{args.bid}b")

    index_dir = ds_paths.get("index_dir") if (config.enable_index_retrieval and ds_paths.get("index_dir")) else None
    if index_dir:
        index_dir = str(Path(index_dir))
        # Force index_dir under TRACE_WORK_ROOT (avoid apr_new or other roots)
        if not index_dir.startswith(work_root) and args.dataset.lower() == "defects4j":
            index_dir = str(Path(work_root) / "defects4j_index")

    meta_dir = os.environ.get("APR_META_DIR") or ds_paths.get("meta_dir") or str(Path(os.environ.get("TRACE_WORK_ROOT", "/tmp/trace_work")) / "apr_meta" / "defects4j" / f"{args.pid}-{args.bid}b")
    base_log_dir = ds_paths.get("log_dir") or str(Path(os.environ.get("TRACE_WORK_ROOT", "/tmp/trace_work")) / "logs" / f"{args.pid}-{args.bid}b")
    log_dir = str(Path(base_log_dir) / args.variant)

    # 只在目录不存在时创建，避免不必要的 inode 占用
    meta_path = Path(meta_dir)
    if not meta_path.exists():
        meta_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    full_log = str(Path(log_dir) / "test.full.log")
    trig_log = str(Path(log_dir) / "test.trigger.log")

    index_exists = False
    if index_dir:
        # Flat layout: {index_dir}/{pid}-{bid}b_index.json
        index_exists = (Path(index_dir) / f"{args.pid}-{args.bid}b_index.json").exists()

    # Register TDD Gate functions if enabled (need a RED test name)
    red_test_name = None
    if config.enable_tdd_gate:
        # Get trigger test name from metadata
        meta_dir_path = Path(meta_dir)
        trig_file = meta_dir_path / "tests.trigger.txt"
        if trig_file.exists():
            try:
                content = trig_file.read_text().strip()
                if content:
                    lines = content.splitlines()
                    if lines:
                        # Find first method-level test (contains ::)
                        for line in lines:
                            if "::" in line:
                                red_test_name = line.strip()
                                break
                        # If no method-level test found, use first line
                        if not red_test_name:
                            red_test_name = lines[0].strip()
            except Exception as e:
                print(f"[WARN] Failed to read trigger test file: {e}", file=sys.stderr, flush=True)
        
        # For SWE-bench: if no trigger file, try to get test name from instance.json
        if not red_test_name:
            instance_file = meta_dir_path / "instance.json"
            if instance_file.exists():
                try:
                    import json
                    with open(instance_file) as f:
                        inst_data = json.load(f)
                    # SWE-bench uses FAIL_TO_PASS to indicate the test that should pass after fix
                    fail_to_pass = inst_data.get("FAIL_TO_PASS", [])
                    if fail_to_pass:
                        # Handle both list and string formats
                        if isinstance(fail_to_pass, list) and len(fail_to_pass) > 0:
                            red_test_name = fail_to_pass[0]
                        elif isinstance(fail_to_pass, str):
                            # If it's a JSON string, parse it
                            try:
                                fail_to_pass_list = json.loads(fail_to_pass)
                                if isinstance(fail_to_pass_list, list) and len(fail_to_pass_list) > 0:
                                    red_test_name = fail_to_pass_list[0]
                            except (json.JSONDecodeError, TypeError):
                                # If parsing fails, use as-is
                                red_test_name = fail_to_pass
                        
                        if red_test_name:
                            print(f"[INFO] [G1] Using test name from SWE-bench instance: {red_test_name}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[WARN] Failed to read instance.json for test name: {e}", file=sys.stderr, flush=True)
        
    red_log = str(Path(log_dir) / "red.log")
    green_log = str(Path(log_dir) / "green.log")

    tools_schema, tool_runtime = setup_tools(
        workdir=workdir,
        enable_index_retrieval=config.enable_index_retrieval,
        index_exists=index_exists,
        enable_patch_compile_gate=config.enable_patch_compile_gate,
        enable_tdd_gate=config.enable_tdd_gate,
        red_test_name=red_test_name,
        red_log=red_log,
        green_log=green_log,
        adapter=adapter,
        meta_dir=meta_dir,
    )
    
    def harness_fn():
        return adapter.harness(args.pid, args.bid, workdir, meta_dir, full_log, trig_log, index_dir)
    
    def validate_fn(patch_text: str):
        return adapter.validate(args.pid, args.bid, workdir, meta_dir, full_log, trig_log)
    
    def apply_patch_fn(patch_text: str):
        # Check if workdir exists before applying patch
        import os
        if not os.path.exists(workdir):
            return {"ok": False, "error": f"workdir not found: {workdir} (may have been deleted)"}
        return apply_patch(workdir, patch_text)
    
    read_log_hint = "IMPORTANT: For test failure details, read red.log (NOT test.full.log). The red.log file contains focused failure information (assertion errors, stack traces). The test.full.log file is very large and contains all test outputs - avoid reading it for efficiency."
    
    # Run agent loop
    result = run_agent_loop_ablation(
        client=client,
        model=args.model,
        prompts=prompts,
        tools_schema=tools_schema,
        tool_runtime=tool_runtime,
        harness_fn=harness_fn,
        validate_fn=validate_fn,
        apply_patch_fn=apply_patch_fn,
        read_log_hint=read_log_hint,
        max_iters=args.max_iters,
        config=config,
        adapter=adapter,
        checkout_fn=adapter.checkout,
    )
    
    # Print result
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

if __name__ == "__main__":
    import sys
    sys.exit(0 if main().get("ok") else 1)

