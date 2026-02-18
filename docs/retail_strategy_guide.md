# Retail Strategy Guide (Non-HFT)

This repo can now be used with a slower rebalancing, retail-friendly workflow:

1. Mine factors with evolution (`configs/experiment.yaml`)
2. Backtest with enhanced portfolio construction (`configs/backtest.yaml`, `backtest.mode=enhanced`)
3. Export daily signals/weights and rebalance weekly (or every 5 trading days)

## 1) Enable Multi-Objective Evolution

In `configs/experiment.yaml`:

```yaml
evolution:
  multi_objective:
    enabled: true
    weights:
      rank_ic: 1.0
      ic: 0.5
      annualized_return: 0.5
      information_ratio: 0.5
      max_drawdown_penalty: 0.75
      complexity_penalty: 0.05
```

This makes parent selection use risk-aware trajectory scores instead of RankIC only.

## 2) Use Enhanced Portfolio Construction

In `configs/backtest.yaml`:

```yaml
backtest:
  mode: "enhanced"
  enhanced:
    optimizer: "mvo"
    topk: 50
    rebalance_frequency: 5
    transaction_cost_bps: 10
    regime:
      enabled: true
```

Supported optimizers: `equal`, `mvo`, `risk_parity`, `kelly`.

## 3) Enable Orthogonalization + Decay Pruning

In `configs/backtest.yaml`:

```yaml
factor_postprocess:
  enabled: true
  correlation_threshold: 0.5
  orthogonalization: "residual"
  rolling_ic_window: 63
  min_abs_rolling_ic: 0.005
```

This reduces redundant factors and removes decayed factors before training.

## 4) Expand Modalities Beyond OHLCV

In `configs/backtest.yaml` under `data`:

```yaml
data:
  extra_fields: ['$turn', '$amount']
  external_feature_files: ['data/features/news_sentiment_daily.csv']
```

External files must include:
- `datetime`
- `instrument`
- one or more feature columns

## 5) Run Backtest

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library.json
```

Outputs:
- `<name>_enhanced_daily.csv` (daily returns, turnover, regime)
- `<name>_cumulative_excess.csv`
- `<name>_backtest_metrics.json`

## 6) Retail Deployment Pattern

- Rebalance frequency: weekly (`rebalance_frequency: 5`) or bi-weekly (`10`)
- Universe: liquid large/mid caps only
- Position limits: keep `max_weight <= 8%`
- Capital scaling: start small, increase only after out-of-sample stability
- Live controls: max daily loss, max turnover cap, hard stop on data/feed errors

## 7) Important Risk Note

No configuration can guarantee profits. Use walk-forward validation, paper trading, and strict risk limits before deploying real capital.
