"""
Variant loader.

Loads ablation variant configuration and prompts from on-disk files:
  ablation/variants/<VARIANT>/config.json
  ablation/variants/<VARIANT>/prompts/{system,localize,patch}.txt
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Tuple


APR_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_variant(variant_name: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Returns:
      (config_dict, prompts_dict)
    """
    v = variant_name.upper()
    variant_dir = APR_ROOT / "ablation" / "variants" / v
    cfg_path = variant_dir / "config.json"
    prompts_dir = variant_dir / "prompts"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Variant config not found: {cfg_path}")
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Variant prompts dir not found: {prompts_dir}")

    config_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    prompts = {
        "system": _read_text(prompts_dir / "system.txt"),
        "localize": _read_text(prompts_dir / "localize.txt"),
        "patch": _read_text(prompts_dir / "patch.txt"),
    }
    return config_dict, prompts


def dump_variant(variant_name: str) -> Dict[str, Any]:
    """Convenience: returns a JSON-serializable dict of variant config+prompts."""
    cfg, prompts = load_variant(variant_name)
    return {"variant": variant_name.upper(), "config": cfg, "prompts": prompts}




