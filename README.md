# TRACE (APR Framework)

Minimal APR snapshot for Defects4J and SWE-bench. All work data under **TRACE_WORK_ROOT**.

## 1. Setup

```bash
cd trace
python -m venv .venv_defects4j
source .venv_defects4j/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: set **TRACE_WORK_ROOT** (e.g. `/tmp/trace_work`), **DEFECTS4J_HOME**, **PERL5_DIR**, and an API key (**OPENAI_API_KEY** or **DEEPSEEK_API_KEY** for `models/example.json`). Defects4J + Java 8/11 + Perl 5 with DBI required.

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
