# TRACE (APR Framework)

TRACE: Agentic Program Repair via Retrieval-Based Localization, Conditional Validation, and Adaptive Tool Use

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

## 2. Data layout (under TRACE_WORK_ROOT)

All work data lives under **TRACE_WORK_ROOT** (see `dataset/defects4j.json`):

- **workdirs/defects4j/{pid}-{bid}b** – Defects4J checkout for the bug.
- **defects4j_index/** – Retrieval index: one file per bug (e.g. `Chart-1b_index.json`). Built from the checkout; used by TRACE to retrieve relevant code/test context for the LLM. **You must build this once per bug before running TRACE.**
- **apr_meta/{pid}-{bid}b**, **logs/{pid}-{bid}b** – Meta and run logs.

## 3. Run an example bug (Defects4J)

**Step 1 – Build retrieval index** (once per bug; creates checkout if needed and writes `{TRACE_WORK_ROOT}/defects4j_index/Chart-1b_index.json`):

```bash
cd trace
source .venv_defects4j/bin/activate
bash bin/build_index.sh --dataset d4j --pid Chart --bid 1
```

**Step 2 – Run TRACE** (reads index and workdir from config; checkout runs automatically if missing):

```bash
./scripts/run_one_defects4j.sh Chart 1
```

Or: `python run_trace.py --dataset defects4j --pid Chart --bid 1 --variant TRACE --model example`

Batch: `./scripts/run_batch_defects4j.sh test/test_d4j.txt`.

## 4. Model

`models/example.json`: set `api_key_env`. Copy and edit for other models.
