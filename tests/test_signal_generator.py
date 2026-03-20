import pandas as pd

from quantaalpha_us.pipeline.signal_generator import SignalConfig, _apply_turnover_cap, select_signals_from_snapshot


def test_turnover_cap_uses_one_way_turnover() -> None:
    previous = {"AAPL": 0.5, "MSFT": 0.5}
    desired = {"NVDA": 0.5, "META": 0.5}

    capped = _apply_turnover_cap(
        desired_weights=desired,
        previous_weights=previous,
        max_turnover_daily=0.20,
        max_weight=0.5,
    )

    symbols = sorted(set(previous) | set(capped))
    one_way_turnover = 0.5 * sum(abs(capped.get(s, 0.0) - previous.get(s, 0.0)) for s in symbols)
    assert one_way_turnover <= 0.2000001
    assert pd.Series(capped).sum() <= 1.0 + 1e-9


def test_select_signals_from_snapshot_respects_sector_cap() -> None:
    today = pd.DataFrame(
        [
            {"date": "2025-01-02", "symbol": "A", "score": 10.0, "adv20": 1e7},
            {"date": "2025-01-02", "symbol": "B", "score": 9.0, "adv20": 1e7},
            {"date": "2025-01-02", "symbol": "C", "score": 8.0, "adv20": 1e7},
            {"date": "2025-01-02", "symbol": "D", "score": 7.0, "adv20": 1e7},
            {"date": "2025-01-02", "symbol": "E", "score": 6.0, "adv20": 1e7},
            {"date": "2025-01-02", "symbol": "F", "score": 5.0, "adv20": 1e7},
        ]
    )
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health", "E": "Health", "F": "Utilities"}
    signals = select_signals_from_snapshot(
        today,
        config=SignalConfig(top_k=4, max_weight=0.30, max_sector_weight=0.50, min_avg_dollar_volume=1.0),
        sector_map=sector_map,
    )
    assert not signals.empty
    by_sector = signals.assign(sector=signals["symbol"].map(sector_map)).groupby("sector")["weight"].sum()
    assert float(by_sector.max()) <= 0.5000001
    assert float(signals["weight"].sum()) <= 1.0 + 1e-9
