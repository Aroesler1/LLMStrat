import pandas as pd

from quantaalpha_us.data.quality import DataQualityGate


def test_quality_passes_basic_case() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-20", "2026-02-19", "2026-02-19"],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "open": [100, 200, 99, 198],
            "high": [101, 202, 100, 201],
            "low": [99, 197, 98, 197],
            "close": [100.5, 201, 99.5, 199],
            "adj_close": [100.5, 201, 99.5, 199],
            "volume": [1000, 1500, 1100, 1600],
            "dollar_volume": [100500, 301500, 109450, 318400],
        }
    )
    membership = pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-20"],
            "symbol": ["AAPL", "MSFT"],
            "active": [True, True],
        }
    )

    gate = DataQualityGate(min_members_warn=2, min_members_halt=2)
    report = gate.run_all_checks(bars, date="2026-02-20", membership_df=membership, mode="strict")
    assert report.passed


def test_quality_fails_on_duplicates() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-20"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "adj_close": [100, 100],
            "volume": [1000, 1000],
            "dollar_volume": [100000, 100000],
        }
    )
    gate = DataQualityGate(min_members_warn=1, min_members_halt=1)
    report = gate.run_all_checks(bars, date="2026-02-20", mode="strict")
    assert not report.passed
    assert report.checks["duplicates"].passed is False


def test_gap_backfill_warning_in_lenient_mode() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-20"],
            "symbol": ["AAPL", "MSFT"],
            "open": [100, 200],
            "high": [101, 202],
            "low": [99, 198],
            "close": [100.5, 201],
            "adj_close": [100.5, 201],
            "volume": [1000, 1000],
            "dollar_volume": [100500, 201000],
        }
    )
    gate = DataQualityGate(min_members_warn=1, min_members_halt=1, max_gap_backfill_days=5)
    result = gate.check_gap_backfill(gate._normalize(bars), pd.Timestamp("2026-02-25"))
    assert result.passed is False
    assert result.severity == "warning"
