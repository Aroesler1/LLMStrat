# QuantaAlpha US Project

This folder is an isolated US-equities project track, separate from the legacy CN workflow.

## Structure

- `US/quantaalpha_us/data`: EODHD client, membership builder helpers, quality gate
- `US/quantaalpha_us/backtest`: S&P 500 point-in-time universe and cost model
- `US/quantaalpha_us/factors`: expression sanitizer for LLM factor candidates
- `US/quantaalpha_us/llm`: run budget and fallback helpers
- `US/quantaalpha_us/pipeline`: baseline signal generation logic
- `US/quantaalpha_us/backtest/walk_forward.py`: deterministic walk-forward engine
- `US/quantaalpha_us/backtest/validation.py`: research promotion gate checks
- `US/quantaalpha_us/trading/risk.py`: pre-trade and post-trade risk controls
- `US/scripts`: operational scripts for ingest, backfill, signals, reporting, and portable trading
- `US/configs`: US-specific research/paper/live configs

## Quick Start

Run from anywhere; scripts resolve paths relative to this `US` folder.

0. Environment setup (inside `US`):

```bash
cd /absolute/path/to/US
./bootstrap.sh
source .venv/bin/activate
cp .env.example .env   # then fill in real keys
# optional: export vars from .env
set -a && source .env && set +a
```

1. Build membership artifacts:

```bash
scripts/sp500_build_membership.py
```

2. Backfill bars:

```bash
scripts/sp500_backfill_history.py
scripts/sp500_data_coverage_report.py
```

3. Daily ingest update:

```bash
scripts/sp500_ingest_daily.py --date 2026-02-23
```

4. Generate signals:

```bash
scripts/sp500_generate_signals.py --config configs/backtest_sp500_research.yaml
```

5. Run research walk-forward + validation:

```bash
scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

6. (Optional) run factor mining runtime with budget/sanitizer guards:

```bash
scripts/sp500_run_factor_mining.py --config configs/llm_sp500.yaml
```

7. Write daily report:

```bash
scripts/sp500_daily_report.py
```

8. Trigger kill switch (US project):

```bash
scripts/kill_switch.py --config configs/paper_sp500.yaml --level 2 --reason "manual test" --yes
```

## Portable Trading (No CN Runtime Dependency)

Submit one rebalance from signal file to Alpaca:

```bash
scripts/trade_once.py --config configs/paper_sp500.yaml --dry-run
scripts/trade_once.py --config configs/paper_sp500.yaml
```

## Orchestrated Runs

Run the full sequence (ingest -> signals -> research/trade -> report):

```bash
scripts/sp500_orchestrator.py --mode research --run-date 2026-02-27
scripts/sp500_orchestrator.py --mode paper --run-date 2026-02-27 --dry-run
scripts/sp500_orchestrator.py --mode live --run-date 2026-02-27 --dry-run
```

## Notes

- Keep CN configs and scripts untouched; use `US/configs/*` and `US/scripts/*` only for this track.
- EODHD token is read from `EODHD_API_TOKEN` unless passed via `--api-token`.
- The baseline signal generator is model-free and uses robust price/volume factor ranks.
- Alpaca credentials come from env vars configured in `configs/paper_sp500.yaml` and `configs/live_sp500.yaml`.
