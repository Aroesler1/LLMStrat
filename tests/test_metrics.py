import pandas as pd

from quantaalpha.backtest.metrics import compute_backtest_metrics


def test_metrics_constant_positive_returns():
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    strat = pd.Series(0.001, index=idx)
    bench = pd.Series(0.0, index=idx)
    turnover = pd.Series(0.1, index=idx)
    costs = pd.Series(0.0, index=idx)

    m = compute_backtest_metrics(
        strategy_returns=strat,
        benchmark_returns=bench,
        turnover=turnover,
        costs=costs,
    )

    assert m.cagr > 0
    assert m.total_return > 0
    assert m.max_drawdown == 0.0
    assert m.win_rate == 1.0
    assert m.information_ratio >= 0


def test_metrics_with_drawdown_and_costs():
    idx = pd.date_range("2023-01-01", periods=8, freq="B")
    strat = pd.Series([0.01, -0.03, 0.02, -0.01, 0.0, 0.01, -0.02, 0.01], index=idx)
    bench = pd.Series([0.002] * len(idx), index=idx)
    turnover = pd.Series([0.2] * len(idx), index=idx)
    costs = pd.Series([0.0005] * len(idx), index=idx)
    npos = pd.Series([10] * len(idx), index=idx)

    m = compute_backtest_metrics(
        strategy_returns=strat,
        benchmark_returns=bench,
        turnover=turnover,
        costs=costs,
        num_positions=npos,
    )

    assert m.max_drawdown < 0
    assert m.total_costs_bps > 0
    assert m.avg_positions == 10
