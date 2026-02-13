# TRACE (APR Framework)

TRACE: Agentic Program Repair via Retrieval-Based Localization, Conditional Validation, and Adaptive Tool Use

**Datasets**

- **Defects4J**: [github.com/rjust/defects4j](https://github.com/rjust/defects4j) — benchmark for automated repair of Java bugs.
- **SWE-bench Verified**: [SWE-bench Verified (Hugging Face)](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer/default/test?p=4) — we use the `princeton-nlp/SWE-bench_Verified` dataset (test split) via `agent/adapters/swebench_verified.py`; each `instance_id` (e.g. `django__django-14311`) identifies a single GitHub issue/patch context.

## 1. Setup

**Defects4J** (one venv):

```bash
cd trace
python -m venv .venv_defects4j
source .venv_defects4j/bin/activate
pip install -r requirements_d4j.txt
cp .env.example .env
```

**SWE-bench Verified** (separate venv, extra deps):

```bash
cd trace
python -m venv .venv_swebench
source .venv_swebench/bin/activate
pip install -r requirements_swe.txt
cp .env.example .env
```

Edit `.env`:

- **Common**: **TRACE_WORK_ROOT** (e.g. `/tmp/trace_work`), API key (**OPENAI_API_KEY** or **DEEPSEEK_API_KEY**; must match `api_key_env` in `models/example.json`).
- **Defects4J**: **DEFECTS4J_HOME** (Defects4J install dir), **PERL5_DIR** (Perl 5 lib path). Requires Defects4J, Java 8 or 11, Perl 5 with DBI.
- **SWE-bench**: **APR_SWEBENCH_RUNTIME** (`docker` or `apptainer`). When using **apptainer**, set **APR_SWEBENCH_SIF_PATH** to the path of your SIF (Singularity/Apptainer image file), e.g. a pre-built SWE-bench testbed image; the runner will use this SIF instead of pulling Docker. Data and instance lists come from your experiment repo.

## 2. Code structure

```
trace/
├── ablation/           # Variant config, main loop, prompts
│   ├── config.py       # AblationConfig (G0–G3, TRACE flags)
│   ├── main_ablation.py
│   ├── core_ablation.py
│   ├── dataset_loader.py
│   ├── variant_loader.py
│   └── variants/       # Per-variant config + prompts
│       ├── G0/         # Baseline
│       ├── G1/         # + TDD Gate
│       ├── G2/         # + Index Retrieval
│       ├── G3/         # + Patch/Compile Gate
│       └── TRACE/      # Full system
├── agent/              # Dataset adapters + tools
│   ├── adapters/       # defects4j.py, swebench_verified.py
│   ├── tools_build_index.py
│   ├── tools_localize.py
│   ├── tools_patch.py
│   └── tools_verify.py
├── bin/                # Flow scripts (checkout, export, test, build_index)
├── dataset/            # defects4j.json (paths under TRACE_WORK_ROOT), env_config.py
├── models/             # Model configs (api_key_env)
├── scripts/            # Run helpers (run_one_*, run_batch_*, build_index_defects4j, build_index_swe)
├── test/               # test_d4j.txt, test_swe.txt
├── run_trace.py        # Entry: delegates to ablation.main_ablation
├── requirements_d4j.txt
└── requirements_swe.txt
```

**Variants** (ablation; use `--variant`):

| Variant | Description |
|---------|-------------|
| **G0** | Baseline: grep/read_file localization, unified diff, full test validation. |
| **G1** | G0 + TDD Gate: verify RED before patch, GREEN after patch. |
| **G2** | G0 + Index Retrieval: symbol_lookup, find_references, read_span (requires retrieval index). |
| **G3** | G0 + Patch/Compile Gate: git apply check, canonical diff, compile gate before full tests. |
| **TRACE** | Full: G1 + G2 + G3 (TDD + Index + Compile). |

## 3. Data layout (under TRACE_WORK_ROOT)

All work data lives under **TRACE_WORK_ROOT** (for Defects4J see `dataset/defects4j.json`; for SWE-bench we only use the index directory):

- **workdirs/defects4j/{pid}-{bid}b** – Defects4J checkout for the bug.
- **defects4j_index/** – Retrieval index: one file per bug (e.g. `Chart-1b_index.json`). Built from the checkout; used by TRACE to retrieve relevant code/test context for the LLM. **You must build this once per bug before running TRACE.**
- **swebench_index/** – Retrieval index for SWE-bench Verified: one file per `instance_id` (e.g. `django__django-14311_index.json`). Built from an existing SWE-bench workdir using `bin/build_index.sh`.
- **apr_meta/{pid}-{bid}b**, **logs/{pid}-{bid}b** – Meta and run logs.

## 4. Run

**Step 1 – Build retrieval index (Defects4J)** (once per bug; required for **G2** and **TRACE**; optional for G0/G1/G3):

```bash
cd trace
source .venv_defects4j/bin/activate
bash bin/build_index.sh --dataset d4j --pid Chart --bid 1
```

**Step 2 – Run (Defects4J)** (use `python run_trace.py` so you can set all parameters):

```bash
python run_trace.py \
  --dataset defects4j \
  --pid Chart \
  --bid 1 \
  --variant TRACE \
  --model example \
  --max-iters 1
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | no | `defects4j` | Dataset name (e.g. `defects4j`) |
| `--pid` | **yes** | – | Project ID (e.g. `Chart`, `Lang`) |
| `--bid` | **yes** | – | Bug ID (e.g. `1`) |
| `--variant` | no | `TRACE` | Ablation variant: **G0**, **G1**, **G2**, **G3**, **TRACE** |
| `--model` | no | `gpt-4o` | Model name (must exist under `models/<name>.json`) |
| `--max-iters` | no | `0` | Max patch iterations (0 = harness/verify only, no patch loop) |
| `--workdir` | no | from config | Override workdir (default: from TRACE_WORK_ROOT + dataset config) |

Workdir, index path, meta and logs are taken from `dataset/defects4j.json` (under TRACE_WORK_ROOT). Checkout runs automatically if the workdir does not exist. For batch runs, loop over `pid/bid` or use `scripts/run_batch_defects4j.sh` with a list file (e.g. `test/test_d4j.txt`).

**SWE-bench Verified (environment, index, run)**  
TRACE assumes your SWE-bench experiment repo prepares the workdir (checked-out project + tests) for each `instance_id`, and points `--workdir` here when building the index.

```bash
# 1) Activate SWE-bench virtualenv (install with requirements_swe.txt)
cd trace
python -m venv .venv_swebench
source .venv_swebench/bin/activate
pip install -r requirements_swe.txt

# 2) Build retrieval index once per instance_id (requires an existing workdir; only TRACE_WORK_ROOT)
export TRACE_WORK_ROOT=/tmp/trace_work
bash bin/build_index.sh \
  --dataset swe \
  --instance-id django__django-14311 \
  --workdir /path/to/swe_workdirs/django__django-14311

# 3) Run TRACE on that SWE-bench instance
python run_trace.py \
  --dataset swebench_verified \
  --pid django__django-14311 \
  --bid 0 \
  --variant TRACE \
  --model example \
  --max-iters 1
```

Here:

- `--dataset swebench_verified` tells TRACE to use the SWE-bench Verified adapter (`agent/adapters/swebench_verified.py`).
- `--pid` is exactly the SWE-bench `instance_id` (e.g. entries like `django__django-14311` in `test/test_swe.txt`).
- `--bid` is just a small integer used in path naming (not the SWE-bench ID).
- `--workdir` in `bin/build_index.sh` should point to the instance workdir prepared by your SWE-bench experiment repo or harness.

## 5. Model

`models/example.json`: set `api_key_env`. Copy and edit for other models.

## 6. Results and eval

Result files are stored under `results/`: `results/d4j/*.jsonl` (Defects4J) and `results/swe/*.jsonl` (SWE-bench Verified). Each line is one JSON record (dataset, model, pid, ok, metrics, etc.).

To aggregate fix rate per dataset and model:

```bash
cd trace
python results/eval.py
```

Options:

- **`--results-dir DIR`** — Use a different results root (default: `results/` next to the script).
- **`--json`** — Output a JSON array of `{dataset, model, fixed, total, fix_rate_pct}` instead of a table.
