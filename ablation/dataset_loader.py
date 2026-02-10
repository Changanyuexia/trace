"""
Dataset loader.

Loads dataset config JSON and instantiates a DatasetAdapter.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Import path resolution utilities
try:
    from dataset.env_config import resolve_path_template, get_dataset_paths
    _USE_ENV_CONFIG = True
except ImportError:
    _USE_ENV_CONFIG = False

APR_ROOT = Path(__file__).resolve().parents[1]


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    name = dataset_name.lower()
    cfg_path = APR_ROOT / "dataset" / f"{name}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def get_adapter(dataset_name: str):
    """Load adapter from trace/agent so TRACE_WORK_ROOT and trace config are used."""
    cfg = load_dataset_config(dataset_name)
    adapter_path = cfg["adapter_class"]
    module_path, class_name = adapter_path.rsplit(".", 1)
    # Load adapter from trace/agent so we use trace's paths (TRACE_WORK_ROOT), not apr_new's
    trace_agent = APR_ROOT / "agent" / "adapters"
    adapter_file = trace_agent / "defects4j.py" if "defects4j" in adapter_path.lower() else None
    if adapter_file and adapter_file.exists() and "defects4j" in adapter_path.lower():
        spec = importlib.util.spec_from_file_location("agent.adapters.defects4j", adapter_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        return cls()
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_paths(dataset_cfg: Dict[str, Any], *, pid: str, bid: int, scratch_base: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths for a dataset instance. All paths under TRACE_WORK_ROOT.
    Uses dataset_cfg (from trace/dataset/*.json) and scratch_base from env only.
    """
    paths = dataset_cfg.get("paths", {})
    if scratch_base is None or (isinstance(scratch_base, str) and not scratch_base.strip()):
        scratch_base = os.environ.get("TRACE_WORK_ROOT", "/tmp/trace_work")

    def fmt(t: str) -> str:
        if not t or "{" not in t:
            return t or ""
        return (
            t.replace("{scratch_base}", scratch_base)
            .replace("{trace_work_root}", scratch_base)
            .replace("{pid}", pid)
            .replace("{bid}", str(bid))
        )

    return {
        "workdir": fmt(paths.get("workdir_template", "")),
        "index_dir": fmt(paths.get("index_dir_template", "")),
        "log_dir": fmt(paths.get("log_dir_template", "")),
        "meta_dir": fmt(paths.get("meta_dir_template", "")),
    }




