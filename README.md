# LLMStrat

LLM-driven alpha factor mining, backtesting, and retail-oriented paper-trading runtime.

This repository is maintained at `Aroesler1/LLMStrat` and keeps the `quantaalpha` module path for runtime compatibility.

- Paper: https://arxiv.org/abs/2602.07085

## What This Repository Contains

- Factor mining pipeline:
  - `quantaalpha/pipeline/factor_mining.py`
  - `run.sh`
  - `configs/experiment.yaml`
  - `configs/experiment_paper_best.yaml`
- Backtesting engine:
  - `quantaalpha/backtest/run_backtest.py`
  - `quantaalpha/backtest/runner.py`
  - `configs/backtest.yaml`
  - `configs/backtest_paper_best.yaml`
- Trading runtime (mock/paper/live broker adapters):
  - `quantaalpha/trading/*`
  - `configs/trading.yaml`
  - `configs/paper.yaml`
- Turnkey orchestration scripts:
  - `scripts/run_paper_best_repro.sh`
  - `scripts/run_baseline_then_upgraded.sh`
  - `scripts/run_locked_upgrade_suite.py`
  - `scripts/run_oos_sweep.py`

## Installation

### 1. Python environment

`pyproject.toml` declares Python `>=3.10` and classifiers for `3.10`/`3.11`.

Recommended:

```bash
conda create -n quantaalpha python=3.11
conda activate quantaalpha
```

### 2. Install package

```bash
pip install -e .
pip install -r requirements.txt
```

### 3. Environment file

```bash
cp configs/.env.example .env
```

Fill required fields in `.env` (from `configs/.env.example`):

- Paths:
  - `QLIB_DATA_DIR`
  - `DATA_RESULTS_DIR`
- LLM:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `CHAT_MODEL`
  - `REASONING_MODEL`
- Optional GPT-5 effort control:
  - `QA_REASONING_EFFORT` (`none|low|medium|high|xhigh`)
- Optional paper/live broker credentials:
  - `ALPACA_API_KEY`
  - `ALPACA_SECRET_KEY`

## Data Requirements

### Qlib market data

`run.sh` validates that `QLIB_DATA_DIR` exists and contains:

- `calendars/`
- `features/`
- `instruments/`

### Factor execution HDF5 files

Defaults are defined in `quantaalpha/factors/coder/config.py`:

- `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`
- `git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5`

You can override via environment variables:

- `FACTOR_CoSTEER_DATA_FOLDER`
- `FACTOR_CoSTEER_DATA_FOLDER_DEBUG`

Reference dataset metadata is in `hf_data/README.md` (`LLMStrat/qlib_csi300`).

## CLI Overview

Entry point from `pyproject.toml`:

- `quantaalpha = quantaalpha.cli:app`
- `llmstrat = quantaalpha.cli:app`

Main CLI commands (`quantaalpha/cli.py`):

- `quantaalpha mine`
- `quantaalpha backtest`
- `quantaalpha trade ...`
- `quantaalpha paper`
- `quantaalpha status`
- `quantaalpha rebalance`
- `quantaalpha stop`
- `quantaalpha health_check`
- `quantaalpha collect_info`

## How To Run

### A. Factor mining

Preferred wrapper (`run.sh` loads `.env`, validates data, sets runtime guards):

```bash
./run.sh "Price-Volume Factor Mining"
```

With custom factor-library suffix:

```bash
./run.sh "Price-Volume Factor Mining" "my_run_suffix"
```

Direct CLI form:

```bash
quantaalpha mine --direction "Price-Volume Factor Mining" --config_path configs/experiment.yaml
```

Paper-style config:

```bash
quantaalpha mine --direction "Price-Volume Factor Mining" --config_path configs/experiment_paper_best.yaml
```

### B. End-to-end paper-best reproduction

Single script (mining + two OOS backtests):

```bash
bash scripts/run_paper_best_repro.sh "Price-Volume Factor Mining" "paper_best"
```

This script runs:

1. `CONFIG_PATH="configs/experiment_paper_best.yaml" ./run.sh ...`
2. `python -m quantaalpha.backtest.run_backtest -c configs/backtest_paper_best.yaml --factor-source custom ...`
3. `python -m quantaalpha.backtest.run_backtest -c configs/backtest_paper_best.yaml --factor-source combined ...`

### C. Independent backtest

Module entrypoint (`quantaalpha/backtest/run_backtest.py`):

```bash
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source alpha158_20
```

Custom factors:

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library_<suffix>.json
```

Combined official + custom:

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source combined \
  --factor-json data/factorlib/all_factors_library_<suffix>.json
```

Useful flags:

- `--dry-run` (load factors only, skip backtest)
- `--walk-forward` (override config to enable walk-forward)
- `--skip-uncached`
- `-e/--experiment`
- `-v/--verbose`

### D. Locked upgrade protocol and sweeps

Baseline + locked upgraded suite:

```bash
bash scripts/run_baseline_then_upgraded.sh "Price-Volume Factor Mining" "paper_best_gpt52" "combined"
```

Run locked suite directly:

```bash
python scripts/run_locked_upgrade_suite.py \
  --factor-source combined \
  --factor-json data/factorlib/all_factors_library_<suffix>.json
```

Run OOS sweep:

```bash
python scripts/run_oos_sweep.py
```

### E. Paper/sim trading runtime

Status / rebalance with default trading profile:

```bash
quantaalpha status --config_path configs/trading.yaml --paper=True
quantaalpha rebalance --config_path configs/trading.yaml --paper=True --dry_run=True
```

Start runtime once:

```bash
quantaalpha trade start --config_path configs/trading.yaml --paper=True --once=True
```

Alpaca paper profile:

```bash
quantaalpha paper --config_path configs/paper.yaml --once=True
```

Stop runtime:

```bash
quantaalpha stop --config_path configs/trading.yaml --paper=True
```

## Config Reference

### Core run configs

- `configs/experiment.yaml`
  - planning (`planning.*`)
  - execution (`execution.*`)
  - evolution (`evolution.*`)
  - quality gates (`quality_gate.*`)
  - backtest handoff (`backtest.*`)
- `configs/experiment_paper_best.yaml`
  - paper-style mining profile

### Backtest configs

- `configs/backtest.yaml`
  - factor source (`factor_source.*`)
  - data and segments (`data.*`, `dataset.segments`)
  - model (`model.*`)
  - backtest mode:
    - `backtest.mode: qlib` (TopkDropout)
    - `backtest.mode: enhanced` (MVO/risk parity/Kelly/equal)
  - walk-forward (`walk_forward.*`)
  - factor postprocess (`factor_postprocess.*`)
  - costs (`cost_model.*`)
- `configs/backtest_paper_best.yaml`
  - independent OOS validation profile

### Trading runtime configs

- `configs/trading.yaml`
  - default profile is `broker.provider: mock`, `broker.paper: true`, `execution.mode: paper`, `execution.dry_run: true`
- `configs/paper.yaml`
  - Alpaca paper profile (`broker.provider: alpaca`, `paper: true`)

### Environment template

- `configs/.env.example`

## Output Locations

Typical outputs:

- Mining workspaces: `data/results/workspace_exp_*`
- Mining caches: `data/results/pickle_cache_exp_*`
- Mined factor library JSON: `data/factorlib/all_factors_library_<suffix>.json`
- Backtest artifacts: `data/results/*` (depends on each config's `experiment.output_dir`)
- Trading runtime state:
  - `data/results/trading/trading_state.sqlite`
  - `data/results/trading/paper_state.sqlite`
- Detailed branch/evolution logs: `log/<timestamp>/...`

## Web Dashboard (optional)

```bash
cd frontend-v2
bash start.sh
```

`frontend-v2/start.sh` starts backend on `:8000` and frontend dev server on `:3000` (if available).

## Safety Warnings

- This repository is research software. It is not investment advice.
- No live trading is enabled by default:
  - `configs/trading.yaml` uses `mock` broker, paper mode, and `dry_run: true`.
- Live-like execution requires explicit config and credentials:
  - switching broker provider,
  - disabling dry-run,
  - setting API keys.
- Always validate strategies out-of-sample and under conservative cost/risk assumptions before any real capital deployment.

## Documentation

See `docs/` for deeper guides:

- `docs/user_guide.md`
- `docs/experiment_guide.md`
- `docs/experiment_hyperparameters.md`
- `docs/research_protocol.md`
- `docs/paper_to_live_checklist.md`
- `docs/retail_strategy_guide.md`
- `docs/ops_runbook.md`

---

## 📖 Citation

If you use this project in research, please cite the original upstream paper:

```bibtex
@misc{han2026quantaalphaevolutionaryframeworkllmdriven,
      title={LLMStrat: An Evolutionary Framework for LLM-Driven Alpha Mining}, 
      author={Jun Han and Shuo Zhang and Wei Li and Zhi Yang and Yifan Dong and Tu Hu and Jialuo Yuan and Xiaomin Yu and Yumo Zhu and Fangqi Lou and Xin Guo and Zhaowei Liu and Tianyi Jiang and Ruichuan An and Jingping Liu and Biao Wu and Rongze Chen and Kunyi Wang and Yifan Wang and Sen Hu and Xinbing Kong and Liwen Zhang and Ronghao Chen and Huacan Wang},
      year={2026},
      eprint={2602.07085},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2602.07085}, 
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Aroesler1/LLMStrat&type=date&legend=top-left)](https://www.star-history.com/#Aroesler1/LLMStrat&type=date&legend=top-left)

---

<div align="center">

**⭐ If LLMStrat helps you, please give it a star!**

Maintained by Alexander Roesler

</div>
