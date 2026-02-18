import pandas as pd

from quantaalpha.backtest.universe import UniverseConfig, load_historical_universe_mask


def test_load_historical_universe_mask_from_csv(tmp_path):
    hist_csv = tmp_path / "historical_universe.csv"
    hist_csv.write_text(
        "date,symbol\n2021-01-04,AAPL\n2021-01-04,MSFT\n2021-01-05,AAPL\n",
        encoding="utf-8",
    )
    cfg = UniverseConfig(
        use_historical_universe=True,
        historical_universe_file=str(hist_csv),
    )
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-05"])
    mask = load_historical_universe_mask(cfg=cfg, dates=dates, symbols=["AAPL", "MSFT"])

    assert mask is not None
    assert bool(mask.loc[pd.Timestamp("2021-01-04"), "AAPL"]) is True
    assert bool(mask.loc[pd.Timestamp("2021-01-04"), "MSFT"]) is True
    assert bool(mask.loc[pd.Timestamp("2021-01-05"), "AAPL"]) is True
    assert bool(mask.loc[pd.Timestamp("2021-01-05"), "MSFT"]) is False
