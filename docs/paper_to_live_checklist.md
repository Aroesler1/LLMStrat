# Paper to Live Checklist (Retail)

1. Data and latency
- Confirm market data source and timestamps are consistent with your rebalance frequency.
- Validate symbol mapping (universe files, broker symbols, delisted handling).

2. Backtest validation
- Run standard backtest and walk-forward backtest with realistic costs enabled.
- Verify robustness across multiple train/test windows and stressed parameters.

3. Paper trading
- Start with `configs/paper.yaml` and run at least 4-8 weeks in paper mode.
- Track order rejection rate, fill slippage, daily turnover, and drift to target weights.

4. Risk limits
- Set conservative limits in `risk` section (`max_position_pct`, `max_drawdown_pct`, `max_daily_loss_pct`).
- Test kill-switch behavior (`quantaalpha stop --flatten=True` path).

5. Operational readiness
- Ensure SQLite state file is persisted and backed up.
- Confirm monitoring endpoint `/api/v1/trading/status` is reachable.
- Verify scheduler timezone and cron schedule before live cutover.

6. Live cutover
- Use smallest capital tier first.
- Change broker mode to live only after paper metrics are stable.
- Keep `dry_run` toggle available for emergency fallback.
