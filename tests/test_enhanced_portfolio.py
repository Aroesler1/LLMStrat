import pandas as pd

from quantaalpha.backtest.enhanced_portfolio import EnhancedPortfolioBacktester, EnhancedPortfolioConfig


def _build_signal_index(dates, symbols):
    idx = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "instrument"])
    vals = []
    for _d in dates:
        vals.extend([1.0, 0.8, 0.4])
    return pd.Series(vals, index=idx, name="pred")


def test_enhanced_portfolio_respects_max_weight_and_tracks_cash():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    symbols = ["A", "B", "C"]
    signal = _build_signal_index(dates, symbols)

    close = pd.DataFrame(index=dates)
    close["A"] = [100 + i for i in range(len(dates))]
    close["B"] = [120 + i for i in range(len(dates))]
    close["C"] = [80 + i for i in range(len(dates))]

    cfg = EnhancedPortfolioConfig(
        enabled=True,
        optimizer="equal",
        topk=3,
        rebalance_frequency=5,
        max_weight=0.2,
        min_weight=0.0,
        regime_enabled=False,
    )
    bt = EnhancedPortfolioBacktester(cfg)
    daily, metrics = bt.run(signal=signal, close_df=close)

    assert "cash_weight" in daily.columns
    assert "gross_exposure" in daily.columns
    # 3 names * 0.2 cap => 0.6 gross exposure, rest should stay cash.
    assert daily["gross_exposure"].max() <= 0.6000001
    assert daily["cash_weight"].min() >= 0.3999999
    assert metrics.get("avg_cash_weight", 0.0) > 0.0
