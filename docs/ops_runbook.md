# Trading Ops Runbook

## Start commands

```bash
# One-off paper rebalance
quantaalpha paper --once=True

# Continuous scheduled paper runtime
quantaalpha paper --once=False

# Manual status
quantaalpha status --config_path configs/paper.yaml --paper=True
```

## Emergency procedures

```bash
# Stop runtime, keep positions
quantaalpha stop --config_path configs/paper.yaml --paper=True

# Stop and flatten all positions
quantaalpha stop --config_path configs/paper.yaml --paper=True --flatten=True
```

## Common issues

1. Missing Alpaca keys
- Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`.

2. Empty target weights
- Check `signals.mode` and `signals.signal_file` in `configs/paper.yaml`.

3. No orders generated
- Verify `min_trade_dollars`, `cash_buffer_pct`, and risk caps.

4. Scheduler running but no trades
- Check `execution.require_market_open`, market session times, and cron timezone.

## State location

- SQLite state: `storage.db_path` in trading YAML.
- Runtime health: `/api/v1/trading/status`.
