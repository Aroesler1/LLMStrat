import pandas as pd

from quantaalpha.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine


def test_walk_forward_generates_windows():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    cfg = WalkForwardConfig(enabled=True, train_months=12, test_months=3, step_months=3)
    engine = WalkForwardEngine(cfg)
    windows = engine.generate_windows(dates)
    assert len(windows) > 0
    assert windows[0].train_start < windows[0].train_end < windows[0].test_start <= windows[0].test_end


def test_walk_forward_run_smoke():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", "2023-12-31", freq="B"), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    x = pd.DataFrame({"f1": range(len(idx))}, index=idx)
    y = pd.Series(0.01, index=idx)

    cfg = WalkForwardConfig(enabled=True, train_months=12, test_months=3, step_months=3)
    engine = WalkForwardEngine(cfg)

    def fit_predict(x_train, y_train, x_test):
        return pd.Series(0.5, index=x_test.index)

    def evaluate(pred, window):
        # Collapse to daily returns for test dates.
        dates = pred.index.get_level_values("datetime").unique()
        oos = pd.Series(0.001, index=dates)
        return {"oos_returns": oos, "daily": pd.DataFrame({"portfolio_return": oos}), "metrics": {}}

    out = engine.run(x, y, fit_predict, evaluate)
    assert "windows" in out
    assert len(out["windows"]) > 0


def test_walk_forward_validation_pct_splits_training_window():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", "2023-12-31", freq="B"), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    x = pd.DataFrame({"f1": range(len(idx))}, index=idx)
    y = pd.Series(0.01, index=idx)

    cfg = WalkForwardConfig(enabled=True, train_months=12, test_months=3, step_months=3, validation_pct=0.2)
    engine = WalkForwardEngine(cfg)
    fit_sizes = []

    def fit_predict(x_train, y_train, x_test):
        fit_sizes.append(len(x_train))
        return pd.Series(0.5, index=x_test.index)

    def evaluate(pred, window):
        dates = pred.index.get_level_values("datetime").unique()
        oos = pd.Series(0.001, index=dates)
        return {"oos_returns": oos, "daily": pd.DataFrame({"portfolio_return": oos}), "metrics": {}}

    out = engine.run(x, y, fit_predict, evaluate)
    assert len(out["windows"]) > 0
    assert len(fit_sizes) == len(out["windows"])
    assert "validation_window" in out["windows"][0]
