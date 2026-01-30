# TRACE 

APR framework with retrieval-based localization, conditional validation (TDD gates), and adaptive tool reasoning.

## Directory layout

```
trace/
├── ablation/           # G5 variant config and core loop
│   ├── variants/G5/    # Prompts and config (localize + patch + index)
│   ├── core_ablation.py
│   ├── main_ablation.py
│   ├── model_loader.py # Uses api_key_env only (no keys in repo)
│   └── ...
├── agent/              # Tools and dataset adapters
│   ├── adapters/       # Defects4J, SWE-bench (copy swebench_verified.py from full repo if needed)
│   └── tools_*.py
├── dataset/            # Dataset JSON and env_config (paths use placeholders)
├── models/             # Model configs: api_key_env only, no API keys
│   └── example.json
├── run_g5.py           # Entry: --dataset --workdir --pid --bid --model
├── requirements.txt
├── .env.example        # Copy to .env and set OPENAI_API_KEY etc. Do not commit .env.
└── README.md
```

## Setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or the env name in your model config). Do not commit `.env`.
2. Install: `pip install -r requirements.txt`
3. Set `DEFECTS4J_HOME` if using Defects4J.
4. Model configs under `models/` use `api_key_env` (e.g. `OPENAI_API_KEY`); no API keys are stored in the repo.

## Run (G5)

```bash
python run_g5.py --dataset defects4j --workdir /path/to/workdir --pid Chart --bid 1 --model example
```

API keys are read from the environment only (via `api_key_env` in each model JSON). 
