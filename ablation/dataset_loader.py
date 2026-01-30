"""
Dataset loader.

Loads dataset config JSON and instantiates a DatasetAdapter.
"""

from __future__ import annotations

import importlib
import json
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
    cfg = load_dataset_config(dataset_name)
    adapter_path = cfg["adapter_class"]
    module_path, class_name = adapter_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_paths(dataset_cfg: Dict[str, Any], *, pid: str, bid: int, scratch_base: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths for a dataset instance.
    
    Args:
        dataset_cfg: Dataset configuration dictionary
        pid: Project ID
        bid: Bug ID
        scratch_base: Scratch base directory (optional, uses default from config if not provided)
    
    Returns:
        Dictionary of path name -> path string
    """
    paths = dataset_cfg.get("paths", {})
    
    # Get scratch_base from config if not provided
    if scratch_base is None:
        scratch_base = paths.get("scratch_base", os.environ.get("APR_SCRATCH_BASE", "/tmp/apr_scratch"))
    
    if _USE_ENV_CONFIG:
        # Use env_config utilities for proper path resolution
        dataset_name = dataset_cfg.get("name", "")
        resolved_paths = get_dataset_paths(dataset_name, pid=pid, bid=bid, scratch_base=scratch_base)
        return {
            "workdir": str(resolved_paths.get("workdir_template", "")),
            "index_dir": str(resolved_paths.get("index_dir_template", "")),
            "log_dir": str(resolved_paths.get("log_dir_template", "")),
            "meta_dir": str(resolved_paths.get("meta_dir_template", "")),
        }
    else:
        # Fallback: simple string formatting
        def fmt(t: str) -> str:
            return t.format(pid=pid, bid=f"{bid}b", scratch_base=scratch_base)
        return {
            "workdir": fmt(paths.get("workdir_template", "")) if "workdir_template" in paths else "",
            "index_dir": fmt(paths.get("index_dir_template", "")) if "index_dir_template" in paths else "",
            "log_dir": fmt(paths.get("log_dir_template", "")) if "log_dir_template" in paths else "",
            "meta_dir": fmt(paths.get("meta_dir_template", "")) if "meta_dir_template" in paths else "",
        }




