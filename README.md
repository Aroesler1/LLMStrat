# QuantaAlpha US

`QuantaAlpha_US` is a standalone US equities research and execution stack focused on daily S&P 500 strategies. The project is designed around a practical institutional-style workflow:

- construct a point-in-time investable universe
- maintain daily market data
- generate and evaluate signals
- run walk-forward research with explicit promotion gates
- mine new factors with frontier LLMs under strict runtime controls
- deploy through a retail-accessible broker stack with realistic execution assumptions

The repo is intentionally opinionated. It favors traceable data lineage, explicit validation, and operational safeguards over polished abstractions or optimistic research shortcuts.

## What The System Does

At a high level, the program implements a daily US equities workflow for S&P 500 strategies:

1. source a historical or approximate point-in-time universe
2. ingest and normalize daily OHLCV data
3. validate the integrity and coverage of that dataset
4. compute baseline cross-sectional features
5. construct a constrained long-only portfolio
6. evaluate that portfolio in a walk-forward backtest with costs and validation gates
7. optionally generate new factor candidates with frontier LLMs under strict controls
8. submit paper or live rebalance orders through Alpaca with pre-trade and post-trade checks

In practical terms, this is a research-to-execution system for daily systematic equity strategies rather than a standalone model-training repository or a collection of notebooks.

## End-To-End Process

The full process is intentionally staged so that each output becomes the input to the next step.

### Stage 1: Universe Construction

The pipeline begins by constructing the investable S&P 500 universe.

In research-grade mode, the system pulls constituent history from CRSP or from EODHD fundamentals. That history is converted into a daily point-in-time membership table with one row per symbol per trading date. The membership artifact answers a simple but critical question for every date in the backtest: which names were actually in the index at that moment.

In approximate mode, the system starts from a current constituent snapshot and expands it across trading dates to produce a constant-membership approximation. That path is useful for bring-up and engineering validation, but it is deliberately kept separate from research-grade handling because it does not preserve true historical composition changes.

The output of this stage is:

- a daily membership file
- a sector reference file
- a ticker mapping file for symbol changes and normalization

### Stage 2: Historical Data Ingestion

Once the universe exists, the system retrieves daily OHLCV history for the relevant names.

When CRSP is available, the backfill path uses stable identifiers such as `permno` where possible and translates the result back into symbol-level research artifacts. When CRSP is unavailable, the same workflow can be fed by EODHD.

During ingestion, the system normalizes:

- dates
- symbols
- OHLCV column names
- adjusted close handling
- dollar volume

The purpose of this stage is not only to download data, but to transform vendor-specific outputs into a stable internal format that later research and trading steps can rely on.

The output of this stage is a consolidated daily bars artifact under `data/us_equities/processed/`.

### Stage 3: Data Quality Validation

Before the system generates signals or runs research, it checks whether the market data is credible enough to continue.

The quality layer verifies things like:

- freshness relative to the target date
- completeness against the expected active universe
- OHLC consistency
- duplicate rows
- unexplained return outliers
- unexpected active-symbol gaps
- adjusted-price consistency

This stage is important because it prevents downstream research from quietly proceeding on corrupted or partial datasets. In other words, the system is designed to fail early when market data quality is inadequate.

### Stage 4: Feature Construction

After the bars pass quality checks, the pipeline computes a baseline daily feature set from OHLCV only.

The current baseline includes cross-sectional and time-series features derived from:

- short-horizon momentum
- medium-horizon momentum
- short-term reversal
- realized volatility
- volatility ratio behavior
- volume acceleration
- intraday range behavior

The features are ranked cross-sectionally by date, then combined into a composite score. This keeps the baseline intentionally simple and interpretable while still providing a usable starting point for research and portfolio construction.

### Stage 5: Signal Generation

The signal-generation step takes the daily feature snapshot for a specific date and turns it into a tradable portfolio.

The process is roughly:

1. filter to the active point-in-time universe
2. remove names that fail liquidity thresholds
3. rank names by composite score
4. select the highest-ranked names up to `top_k`
5. apply per-name caps
6. apply sector caps
7. apply turnover control relative to the previous portfolio
8. output target weights

The resulting signal file is not just a list of scores. It is an actual constrained target portfolio that can feed either research or trading.

### Stage 6: Walk-Forward Research

The research engine evaluates the strategy in rolling walk-forward windows rather than a single in-sample fit.

The walk-forward runner:

1. builds train, validation, and test windows
2. steps through each test date in chronological order
3. uses only the information available up to that date
4. generates the target portfolio for that date
5. applies execution alignment and cost assumptions
6. records realized out-of-sample returns

This is a deliberately conservative structure. It is designed to avoid the common failure mode where research results are driven by implicit lookahead, static universes, or unrealistic turnover assumptions.

### Stage 7: Retail-Oriented Execution Simulation

The backtest does not assume frictionless deployment.

Instead, the research path includes a retail-aware execution model that attempts to approximate the operational constraints of a real daily account:

- next-day open execution
- retained cash buffer
- minimum trade-size filtering
- participation limits relative to ADV
- fractional-share support
- spread and slippage costs
- liquidity-sensitive impact estimation

The simulation maintains a portfolio state over time, rather than recomputing each day as if the account could reset frictionlessly. That matters because it allows turnover, residual exposure, and trading frictions to accumulate in a more realistic way.

### Stage 8: Research Validation Gates

Once the walk-forward run is complete, the system evaluates the results against explicit validation gates.

These gates are intended to answer whether the strategy deserves promotion, not just whether it produced a positive backtest.

The current validation layer checks:

- deflated Sharpe
- subperiod stability
- sector concentration
- max drawdown
- turnover
- factor-overlap stability
- baseline comparison
- net Sharpe

This stage is central to the project philosophy. A strategy that produces returns but fails robustness gates is treated as unfinished research, not as a success story.

### Stage 9: Factor Mining

The factor-mining subsystem is separate from the baseline strategy and is designed to generate candidate expressions rather than directly trade model output.

Its workflow is:

1. assemble a prompt batch for a specific factor-generation theme
2. call the configured frontier model
3. enforce exact model policy and fallback order
4. parse the returned JSON
5. sanitize each candidate expression
6. reject invalid or disallowed expressions
7. stop early if invalid output rates exceed threshold
8. store surviving factors as research candidates

This makes the LLM runtime a controlled ideation engine rather than an unbounded text-generation layer.

### Stage 10: Order Formation And Trade Submission

When the system is used in paper or live mode, the generated portfolio weights are translated into broker-facing order intents.

That translation includes:

- loading the latest signal file
- reading current account equity and buying power
- calculating deployable capital after the configured cash buffer
- mapping target weights to target dollar exposure
- comparing target exposure with current positions
- generating buy and sell deltas
- filtering out trivial trades

The system then runs pre-trade checks before any order is submitted.

### Stage 11: Pre-Trade And Post-Trade Risk Controls

The trading path includes operational guardrails before and after submission.

Pre-trade checks include:

- total notional limits
- single-order size limits
- active-universe membership checks
- data freshness checks
- portfolio weight constraints
- buying power checks
- kill-switch status
- trading-session validity

Post-trade checks include:

- order status inspection
- slippage checks relative to expected prices
- reconciliation between target and realized portfolio weights
- cash sanity checks

These checks are there to make the execution layer behave like a trading system rather than a thin API wrapper.

## Project Scope

This codebase covers the full lifecycle of a daily US large-cap systematic strategy:

- historical universe construction
- daily OHLCV ingestion and quality control
- baseline portfolio construction
- walk-forward backtesting
- LLM-assisted factor ideation
- paper and live trading orchestration
- pre-trade and post-trade risk checks

It is not a high-frequency system and does not attempt to model intraday microstructure beyond what is realistic for a daily rebalance process.

## Architecture

### Data

- `quantaalpha_us/data`
  - CRSP/WRDS client
  - EODHD client
  - market-data source selection
  - membership builders
  - data quality checks

### Research

- `quantaalpha_us/pipeline`
  - baseline feature generation
  - signal generation
- `quantaalpha_us/backtest`
  - point-in-time universe handling
  - walk-forward runner
  - transaction cost model
  - validation gates

### LLM Runtime

- `quantaalpha_us/llm`
  - request budgeting
  - fallback handling
  - factor extraction
  - expression sanitization

### Trading

- `quantaalpha_us/trading`
  - Alpaca REST adapter
  - operational risk controls
  - post-trade reconciliation checks

### Entry Points

- `scripts/`
  - membership build
  - backfill and daily ingest
  - research
  - factor mining
  - reporting
  - orchestration
  - trade submission

### Validation

- `tests/`
  - unit coverage over the current stack

## Technical Implementation

The system is implemented as a Python-based research and execution stack with explicit separation between data, research, LLM runtime, and trading concerns.

Key technical characteristics:

- configuration-driven workflows through YAML
- parquet and CSV artifacts for reproducible intermediate datasets
- point-in-time universe handling rather than static ticker lists
- explicit CLI entrypoints for each stage of the pipeline
- deterministic walk-forward evaluation instead of a single in-sample backtest
- automated checks around data quality, turnover, concentration, and research promotion
- broker integration through a REST execution layer rather than notebook-driven manual trading

The code is structured so that research artifacts, signals, and trade actions can be produced from the same underlying pipeline rather than from disconnected scripts.

## Design Priorities

Several design choices define the project:

- data realism before model complexity
- explicit point-in-time universe handling
- research outputs that can fail promotion rather than silently pass
- operational controls around factor mining instead of unconstrained prompting
- retail-aware execution assumptions instead of idealized frictionless backtests

The result is a system that aims to be honest about what is known, what is approximated, and what still needs empirical validation.

## Tools And Technologies

Beyond LLM usage, this project uses and demonstrates familiarity with:

- Python for the full research and execution stack
- `pandas` for feature engineering, panel manipulation, and research outputs
- parquet-based data artifacts for reproducible local research datasets
- CRSP through WRDS for research-grade historical membership and price data
- EODHD as an alternative market-data vendor and fallback path
- Alpaca REST APIs for paper and live execution
- YAML-based configuration for research, mining, and trading profiles
- CLI-oriented orchestration through standalone Python entrypoints
- `pytest` for automated test coverage
- Git and GitHub for versioned development and deployment of research code

From a systems perspective, the project required work across:

- external data integration
- schema normalization
- data quality validation
- portfolio construction logic
- transaction cost modeling
- broker API integration
- runtime fault handling
- test-driven refactoring

## Skills Demonstrated

For a technical reviewer, the project reflects breadth across several domains rather than depth in only one narrow area.

### Data Engineering

- integrating multiple market-data sources behind a common interface
- handling source preference and fallback behavior
- normalizing historical equity data into a usable research format
- managing artifact generation for downstream research and trading stages

### Quantitative Research Engineering

- implementing point-in-time universe logic
- building baseline cross-sectional features from daily OHLCV data
- designing walk-forward evaluation flows
- enforcing promotion gates around Sharpe, drawdown, turnover, concentration, and baseline comparison

### Execution And Trading Systems

- translating model weights into broker-ready orders
- simulating retail-realistic execution assumptions in research
- adding operational risk controls around notional size, buying power, session timing, and reconciliation
- supporting paper and live modes through the same trading path

### Applied LLM Systems

- constraining model usage to a narrow research task
- enforcing exact model selection and fallback policy
- validating generated expressions before they can affect research outputs
- using budgets, structured output expectations, and early-stop criteria to reduce model failure modes

### Software Engineering

- designing a modular codebase with clear subsystem boundaries
- writing regression tests for data, research, and runtime behavior
- improving runtime performance where repeated computations were becoming the bottleneck
- turning a research workflow into a repeatable, scriptable system

## Data Modes

The project supports two operating modes.

### Research-Grade Mode

This mode uses a proper historical membership history sourced from either:

- CRSP through WRDS, or
- EODHD when both historical price data and fundamentals are available

This is the intended mode for serious walk-forward research.

### Approximate Mode

This mode builds a constant-membership S&P 500 approximation from a current constituent snapshot and is retained for:

- bring-up
- pipeline validation
- engineering checks
- low-cost research scaffolding

It is useful operationally, but it is not treated as a substitute for true historical membership data.

## Current Capabilities

At the current stage, the repo supports:

- CRSP-first market data, with EODHD fallback
- historical or approximate S&P 500 membership construction
- daily bar backfill and coverage reporting
- baseline signal generation
- walk-forward research with validation gates
- LLM factor mining with exact model enforcement
- retail-oriented execution simulation
- paper and live trading entrypoints through Alpaca

The main open problem is not infrastructure completeness. The remaining challenge is improving strategy quality enough to satisfy the research gates under realistic assumptions.

## Research Philosophy

The research loop is deliberately conservative.

The backtest includes:

- next-day open execution alignment
- retained cash buffer
- minimum trade-size filtering
- fractional-share handling
- participation limits relative to ADV
- liquidity-aware transaction cost modeling
- sector caps in portfolio construction

The goal is not to create a perfect live execution simulator. The goal is to avoid the far more common failure mode of producing a backtest that is materially cleaner than a retail-accessible implementation could ever achieve.

## LLM Factor Mining

The factor-mining subsystem is treated as a constrained research tool, not a source of unchecked strategy logic.

Current runtime behavior:

- exact primary model: `gpt-5.4`
- exact fallback model: `gemini-3.1-pro`
- OpenAI-compatible endpoint requirement
- strict model-catalog enforcement
- JSON-only output expectation
- sanitizer pass before any expression is accepted
- request and token budgets
- early halt on low-quality output

This is intended to keep model experimentation productive without allowing generated expressions to quietly degrade research quality.

## Setup

```bash
cd /absolute/path/to/QuantaAlpha_US
./bootstrap.sh
source .venv/bin/activate
cp .env.example .env
set -a && source .env && set +a
```

Typical environment configuration includes:

- `CRSP_USERNAME`
- `CRSP_API_KEY`
- `EODHD_API_TOKEN`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_API_SECRET`
- `ALPACA_LIVE_API_KEY`
- `ALPACA_LIVE_API_SECRET`

## Core Artifacts

The central pipeline artifacts are:

- `data/us_equities/reference/sp500_membership_daily.parquet`
- `data/us_equities/reference/gics_sectors.csv`
- `data/us_equities/reference/ticker_mapping.csv`
- `data/us_equities/processed/daily_bars.parquet`

## External Interfaces

The project interacts with several real external systems:

- WRDS / CRSP for research-grade historical data
- EODHD for alternative historical and daily market data
- OpenAI-compatible chat-completions endpoints for factor mining
- Alpaca for execution and account state

That matters because the repository is not only a modeling exercise. It includes API integration, operational failure handling, and the practical engineering needed to move from research outputs to broker-facing actions.

## Representative Workflow

### 1. Build Membership

```bash
python scripts/sp500_build_membership.py
```

Source selection can also be made explicit:

```bash
python scripts/sp500_build_membership.py --source crsp
python scripts/sp500_build_membership.py --source eodhd
python scripts/sp500_build_membership_approx.py
```

### 2. Backfill Daily Bars

```bash
python scripts/sp500_backfill_history.py
python scripts/sp500_data_coverage_report.py
```

### 3. Generate Signals

```bash
python scripts/sp500_generate_signals.py --config configs/backtest_sp500_research.yaml
```

### 4. Run Walk-Forward Research

```bash
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

### 5. Run LLM Factor Mining

```bash
python scripts/sp500_run_factor_mining.py --config configs/llm_sp500.yaml --live-call
```

### 6. Submit a Paper Rebalance

```bash
python scripts/trade_once.py --config configs/paper_sp500.yaml --dry-run
python scripts/trade_once.py --config configs/paper_sp500.yaml
```

### 7. Orchestrate Scheduled Modes

```bash
python scripts/sp500_orchestrator.py --mode research --run-date 2026-03-10
python scripts/sp500_orchestrator.py --mode paper --run-date 2026-03-10 --dry-run
python scripts/sp500_orchestrator.py --mode live --run-date 2026-03-10 --dry-run
```

## Script-Level Flow

The main CLI entrypoints correspond to specific parts of the pipeline:

- `scripts/sp500_build_membership.py`
  - builds historical membership from CRSP or EODHD
- `scripts/sp500_build_membership_approx.py`
  - builds a constant-membership approximation from a current snapshot
- `scripts/sp500_backfill_history.py`
  - backfills historical daily bars
- `scripts/sp500_data_coverage_report.py`
  - measures member-day coverage of the dataset
- `scripts/sp500_ingest_daily.py`
  - performs daily incremental updates
- `scripts/sp500_generate_signals.py`
  - converts market data into target portfolio weights
- `scripts/sp500_run_research.py`
  - runs walk-forward research and validation
- `scripts/sp500_run_factor_mining.py`
  - generates and filters candidate factor expressions
- `scripts/trade_once.py`
  - submits one rebalance cycle
- `scripts/sp500_orchestrator.py`
  - coordinates multi-step scheduled runs

That division is deliberate. Each script has a narrow responsibility, which keeps the project easier to test, inspect, and operate.

## Research Outputs

A research run writes artifacts under `data/results/research/...`, including:

- walk-forward returns
- fold metadata
- research summary
- validation gate results

The validation layer is a first-class part of the project. A run completing successfully is not treated as equivalent to a strategy passing research review.

## Operational Tooling

Additional operational scripts include:

Daily report:

```bash
python scripts/sp500_daily_report.py
```

Kill switch:

```bash
python scripts/kill_switch.py --config configs/paper_sp500.yaml --level 2 --reason "manual test" --yes
```

## Why This Project Exists

The point of this repository is not to present a polished research result. It is to show a full-stack, research-to-execution implementation that takes data integrity, execution realism, and model governance seriously.

In practical terms, that means:

- a real historical universe path exists
- a lower-fidelity approximation path exists when needed
- research and trading share the same operational assumptions where possible
- LLM usage is constrained and reviewable
- promotion depends on validation, not narrative

## Tests

```bash
pytest
```

## Summary

`QuantaAlpha_US` is best understood as a serious systematic trading workbench rather than a toy backtest or a generic LLM wrapper. It combines data engineering, research discipline, execution awareness, and model-safety controls in a single repo.

The remaining work is primarily on strategy quality, not scaffolding. That is the right stage for a project of this kind to be in.
