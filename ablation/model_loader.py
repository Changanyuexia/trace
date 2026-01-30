"""
Model loader.

Loads model config JSON and creates an OpenAI-compatible client.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


APR_ROOT = Path(__file__).resolve().parents[1]


def load_model_config(model_name: str) -> Dict[str, Any]:
    name = model_name
    cfg_path = APR_ROOT / "models" / f"{name}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def create_client(model_name: str):
    """
    Create a client object for the configured provider.

    Current implementation uses OpenAI Python SDK for both:
      - provider=openai: OpenAI base_url
      - provider=openai_compat: OpenAI-compatible endpoints (e.g. DeepSeek)
    """
    cfg = load_model_config(model_name)
    provider = cfg.get("provider", "openai")
    api_key_env = cfg.get("api_key_env")
    if not api_key_env:
        raise ValueError(f"Model config missing api_key_env: {model_name}")
    api_key = os.environ.get(api_key_env) or cfg.get("api_key_default")
    if not api_key:
        raise ValueError(
            f"Environment variable {api_key_env} not set for model {model_name}. "
            "For local vLLM, add \"api_key_default\": \"dummy\" to the model config."
        )

    base_url = os.environ.get(cfg.get("base_url_env", "")) or cfg.get("base_url")
    from openai import OpenAI
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_default_params(model_name: str) -> Dict[str, Any]:
    cfg = load_model_config(model_name)
    return dict(cfg.get("default_params", {}) or {})


def get_model_id(model_name: str) -> str:
    """
    Get the actual model_id from config file.
    For vLLM models, this returns the model_id (e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct)
    instead of the config file name (e.g., qwen3-coder-30b-a3b-vllm).
    
    If model_id is not in config, returns model_name as-is (for OpenAI models).
    """
    try:
        cfg = load_model_config(model_name)
        model_id = cfg.get("model_id")
        if model_id:
            return model_id
    except (FileNotFoundError, KeyError):
        pass
    # Fallback: return model_name as-is (for OpenAI models like gpt-4o)
    return model_name




