import pandas as pd

from quantaalpha_us.backtest.universe import SP500Universe


def test_membership_range_and_validate(tmp_path) -> None:
    path = tmp_path / "membership.csv"
    df = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "active": [True, True, True],
        }
    )
    df.to_csv(path, index=False)

    universe = SP500Universe(str(path))
    assert universe.get_members("2025-01-02") == ["AAPL", "MSFT"]
    assert universe.get_members("2025-01-03") == ["AAPL"]

    report = universe.validate(min_members=1, max_members=3)
    assert report.passed
