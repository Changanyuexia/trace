"""
Project-local dataset configuration package.

Why this file exists:
- This repo defines `dataset.env_config` under `apr_new/dataset/`.
- Many Python environments also install HuggingFace's `datasets` package.
- Without a local package initializer, imports like `from dataset.env_config import ...`
  can be resolved against the external `datasets` package, causing:
    ModuleNotFoundError: No module named 'dataset.env_config'
"""

