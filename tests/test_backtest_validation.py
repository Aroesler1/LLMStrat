import numpy as np
import pandas as pd

from quantaalpha_us.backtest.validation import BacktestValidation


def test_validation_gates_pass_on_good_series() -> None:
    rng = np.random.default_rng(42)
    net = rng.normal(loc=0.0022, scale=0.008, size=600)
    baseline_spy = rng.normal(loc=0.0004, scale=0.01, size=600)
    baseline_eq = rng.normal(loc=0.0005, scale=0.01, size=600)
    baseline_mom = rng.normal(loc=0.0006, scale=0.01, size=600)
    turnover = np.clip(rng.normal(loc=0.15, scale=0.03, size=600), 0, 1)
    df = pd.DataFrame(
        {
            "net_return": net,
            "gross_return": net + 0.0001,
            "turnover": turnover,
            "baseline_spy": baseline_spy,
            "baseline_equal_weight": baseline_eq,
            "baseline_momentum": baseline_mom,
        }
    )

    validator = BacktestValidation()
    report = validator.run_all_gates(
        returns_df=df,
        n_trials=1,
        factor_overlap_score=0.7,
        sector_pnl_share={"Information Technology": 0.33, "Financials": 0.22},
    )
    assert report.passed
