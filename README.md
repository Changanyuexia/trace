# TRACE (APR Framework)

Minimal APR snapshot for Defects4J and SWE-bench. All work data under **TRACE_WORK_ROOT**.

## 1. Setup

```bash
cd trace
python -m venv .venv_defects4j   # for D4J; use .venv_swebench for SWE-bench
source .venv_defects4j/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

- **Common**: **TRACE_WORK_ROOT** (e.g. `/tmp/trace_work`), API key (**OPENAI_API_KEY** or **DEEPSEEK_API_KEY**; must match `api_key_env` in `models/example.json`).
- **Defects4J**: **DEFECTS4J_HOME** (Defects4J install dir), **PERL5_DIR** (Perl 5 lib path). Requires Defects4J, Java 8 or 11, Perl 5 with DBI.
- **SWE-bench**: **APR_SWEBENCH_RUNTIME** (`docker` or `apptainer`), **APR_SWEBENCH_SIF_PATH** (SIF image path when using Apptainer). Data and images from your experiment repo.

## 2. Run (Defects4J)

Build index once per bug, then run:

```bash
cd trace
source .venv_defects4j/bin/activate
bash bin/build_index.sh --dataset d4j --pid Chart --bid 1
./scripts/run_one_defects4j.sh Chart 1
```

Or direct: `python run_trace.py --dataset defects4j --pid Chart --bid 1 --variant TRACE --model example`

Checkout and paths come from TRACE_WORK_ROOT. Batch: `./scripts/run_batch_defects4j.sh test/test_d4j.txt`.

## 3. Model

`models/example.json`: set `api_key_env`. Copy and edit for other models.
