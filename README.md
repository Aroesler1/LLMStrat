# QuantaAlpha US

This repository is the standalone US-equities track for QuantaAlpha. It is focused on S&P 500 daily research, factor mining, paper trading, and eventual live deployment with the US toolchain only.

## What Exists Today

- `quantaalpha_us/data`: CRSP/WRDS + EODHD clients, membership builders, and data-quality checks
- `quantaalpha_us/backtest`: point-in-time universe handling, walk-forward runner, costs, validation gates
- `quantaalpha_us/pipeline`: baseline signal generation
- `quantaalpha_us/llm`: budget guards and mining helpers
- `quantaalpha_us/trading`: Alpaca REST adapter and risk controls
- `scripts/`: CLI entrypoints for membership build, backfill, ingest, research, mining, reporting, orchestration, and trading
- `tests/`: unit coverage for the current US stack

## Current Project Status

Code scaffolding is in place for:

- Phase 0 data ingestion and quality checks
- Phase 1 walk-forward baseline research
- Phase 2 LLM factor-mining runtime guards
- Phase 3 paper/live orchestration and trading controls

What is not done yet is the actual data population and validation run. The main blocker is producing the membership and bar artifacts under `data/us_equities/...`.

Current known limitations:

- CRSP access is implemented via WRDS credentials. Set `CRSP_USERNAME` and `CRSP_API_KEY` for non-interactive runs.
- factor-mining prompt quality is still the main remaining blocker on the LLM side

## Data Source Modes

There are two practical ways to run this repo.

### 1. Research-Grade Mode

Use this when you have:

- CRSP access, or
- EODHD `Fundamentals` + `EOD Historical Data`

This gives you a proper historical S&P 500 membership file and is the mode to use for serious walk-forward research.

The repo now prefers CRSP automatically when `CRSP_USERNAME` and `CRSP_API_KEY` are set, and falls back to EODHD otherwise.

### 2. Approximate Low-Cost Mode

Use this when you only have:

- EODHD `EOD Historical Data - All World` (`$19.99/mo`)

In this mode, the repo builds a constant S&P 500 membership approximation from a current constituent snapshot instead of a true historical point-in-time membership history.

This is good enough for:

- pipeline bring-up
- storage/layout validation
- baseline model debugging
- signal generation and orchestration testing

This is not good enough for final research claims because it introduces survivorship bias.

## Setup

```bash
cd /absolute/path/to/QuantaAlpha_US
./bootstrap.sh
source .venv/bin/activate
cp .env.example .env
set -a && source .env && set +a
```

Required env vars depend on what you are running:

- `CRSP_USERNAME` + `CRSP_API_KEY` for research-grade CRSP/WRDS data
- `EODHD_API_TOKEN` for the EOD data path
- Alpaca paper/live keys for trading scripts
- LLM provider settings for factor mining

## Core Artifacts

The US pipeline expects these files:

- `data/us_equities/reference/sp500_membership_daily.parquet`
- `data/us_equities/reference/gics_sectors.csv`
- `data/us_equities/reference/ticker_mapping.csv`
- `data/us_equities/processed/daily_bars.parquet`

## Phase 0 Commands

### A. Research-Grade Membership Build

This prefers CRSP/WRDS when configured and falls back to EODHD fundamentals otherwise:

```bash
python scripts/sp500_build_membership.py
```

To force one source explicitly:

```bash
python scripts/sp500_build_membership.py --source crsp
python scripts/sp500_build_membership.py --source eodhd
```

### B. Approximate Membership Build

This is the low-cost fallback. It fetches the current S&P 500 list from Wikipedia, normalizes symbols, and expands it into a constant daily membership file:

```bash
python scripts/sp500_build_membership_approx.py
```

If you already have a current constituent CSV/parquet from another free source, use:

```bash
python scripts/sp500_build_membership_approx.py --snapshot-file path/to/current_sp500_snapshot.csv
```

### C. Historical Bar Backfill

```bash
python scripts/sp500_backfill_history.py
python scripts/sp500_data_coverage_report.py
```

The backfill script now prefers CRSP daily stock data when configured and falls back to EODHD when CRSP credentials are absent.

### D. Daily Ingest Update

```bash
python scripts/sp500_ingest_daily.py --date 2026-03-10
```

## Research Flow

Once membership and bars exist:

```bash
python scripts/sp500_generate_signals.py --config configs/backtest_sp500_research.yaml
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

Outputs are written under `data/results/...`.

## LLM Factor Mining

Optional US factor mining entrypoint:

```bash
python scripts/sp500_run_factor_mining.py --config configs/llm_sp500.yaml
```

The current implementation assumes an OpenAI-compatible API base URL for chat completions.

## Paper And Live Operations

Single rebalance:

```bash
python scripts/trade_once.py --config configs/paper_sp500.yaml --dry-run
python scripts/trade_once.py --config configs/paper_sp500.yaml
```

Orchestrated run:

```bash
python scripts/sp500_orchestrator.py --mode research --run-date 2026-03-10
python scripts/sp500_orchestrator.py --mode paper --run-date 2026-03-10 --dry-run
python scripts/sp500_orchestrator.py --mode live --run-date 2026-03-10 --dry-run
```

Daily report:

```bash
python scripts/sp500_daily_report.py
```

Kill switch:

```bash
python scripts/kill_switch.py --config configs/paper_sp500.yaml --level 2 --reason "manual test" --yes
```

## Practical Recommendation

If CRSP access is coming soon:

- use the approximate membership builder only to unblock infrastructure work now
- do not treat resulting research metrics as final
- switch to `CRSP_USERNAME` + `CRSP_API_KEY` and rerun `sp500_build_membership.py` plus `sp500_backfill_history.py`

If CRSP is delayed and you want the cleanest EODHD-only path:

- buy only the `$19.99` EOD plan
- use `sp500_build_membership_approx.py`
- proceed with Phase 0 and Phase 1 as an engineering/debug run

## Tests

```bash
pytest
```
