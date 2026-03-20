import pandas as pd

from quantaalpha_us.data.membership import (
    build_membership_daily,
    build_constant_membership_from_snapshot,
    normalize_current_sp500_snapshot,
)


def test_normalize_current_sp500_snapshot_maps_common_columns() -> None:
    snapshot = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "BF.B"],
            "Security": ["Berkshire Hathaway", "Brown-Forman"],
            "GICS Sector": ["Financials", "Consumer Staples"],
            "GICS Sub-Industry": ["Multi-Sector Holdings", "Distillers & Vintners"],
        }
    )

    normalized = normalize_current_sp500_snapshot(snapshot)

    assert normalized["Code"].tolist() == ["BRK-B", "BF-B"]
    assert normalized["Sector"].tolist() == ["Financials", "Consumer Staples"]
    assert normalized["Industry"].tolist() == ["Multi-Sector Holdings", "Distillers & Vintners"]


def test_build_constant_membership_from_snapshot_expands_trading_days() -> None:
    snapshot = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple", "Microsoft"],
            "GICS Sector": ["Information Technology", "Information Technology"],
            "GICS Sub-Industry": ["Technology Hardware", "Systems Software"],
        }
    )

    result = build_constant_membership_from_snapshot(
        snapshot,
        start_date="2025-01-02",
        end_date="2025-01-03",
        trading_days=pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"])),
    )

    assert len(result.membership) == 4
    assert sorted(result.membership["symbol"].unique().tolist()) == ["AAPL", "MSFT"]
    assert sorted(result.sectors["symbol"].tolist()) == ["AAPL", "MSFT"]
    assert list(result.ticker_mapping.columns) == ["old_symbol", "new_symbol", "effective_date", "reason"]


def test_build_membership_daily_carries_permno() -> None:
    constituents = pd.DataFrame(
        {
            "Code": ["AAPL"],
            "StartDate": ["2025-01-02"],
            "EndDate": ["2025-01-03"],
            "permno": [14593],
        }
    )
    membership = build_membership_daily(
        constituents,
        start_date="2025-01-02",
        end_date="2025-01-03",
        trading_days=pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"])),
    )
    assert membership["permno"].dropna().astype(int).tolist() == [14593, 14593]
