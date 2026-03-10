import pandas as pd

from quantaalpha_us.trading.risk import evaluate_post_trade, evaluate_pre_trade, load_risk_config


def test_pre_trade_checks_pass() -> None:
    cfg = {
        "risk": {
            "max_position_pct": 0.05,
            "max_positions": 30,
            "min_positions": 1,
            "max_single_order_value": 10000,
        }
    }
    risk_cfg = load_risk_config(cfg)
    signals = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.04, 0.03]})
    intents = [
        {"symbol": "AAPL", "side": "buy", "delta_value": 3500.0, "notional": 3500.0},
        {"symbol": "MSFT", "side": "buy", "delta_value": 2500.0, "notional": 2500.0},
    ]
    report = evaluate_pre_trade(
        signals=signals,
        order_intents=intents,
        equity=100000.0,
        buying_power=20000.0,
        risk_config=risk_cfg,
        active_universe={"AAPL", "MSFT", "SPY"},
        latest_data_date=pd.Timestamp("2026-02-27"),
        as_of_date=pd.Timestamp("2026-02-27"),
        kill_switch_engaged=False,
        is_trading_day=True,
    )
    assert report.passed


def test_pre_trade_fails_on_kill_switch() -> None:
    risk_cfg = load_risk_config({"risk": {"min_positions": 1}})
    signals = pd.DataFrame({"symbol": ["AAPL"], "weight": [0.03]})
    report = evaluate_pre_trade(
        signals=signals,
        order_intents=[{"symbol": "AAPL", "side": "buy", "delta_value": 1000.0}],
        equity=100000.0,
        buying_power=100000.0,
        risk_config=risk_cfg,
        active_universe={"AAPL"},
        latest_data_date=pd.Timestamp("2026-02-27"),
        as_of_date=pd.Timestamp("2026-02-27"),
        kill_switch_engaged=True,
        is_trading_day=True,
    )
    assert not report.passed
    assert report.checks["PT-07_kill_switch"].passed is False


def test_post_trade_reconciliation() -> None:
    risk_cfg = load_risk_config({"risk": {"min_cash_pct": 0.01}})
    target = {"AAPL": 0.05, "MSFT": 0.03}
    positions_after = [
        {"symbol": "AAPL", "market_value": 5000},
        {"symbol": "MSFT", "market_value": 3000},
    ]
    responses = [
        {"intent": {"symbol": "AAPL"}, "response": {"status": "filled", "filled_avg_price": "100"}},
        {"intent": {"symbol": "MSFT"}, "response": {"status": "filled", "filled_avg_price": "200"}},
    ]
    report = evaluate_post_trade(
        target_weights=target,
        positions_after=positions_after,
        order_responses=responses,
        equity=100000.0,
        risk_config=risk_cfg,
        expected_open_prices={"AAPL": 100.0, "MSFT": 200.0},
        account_cash=5000.0,
    )
    assert report.passed
