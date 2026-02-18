from quantaalpha.trading.risk import RiskCheckResult, RiskEngine, RiskLimits


def test_risk_blocks_oversized_position():
    engine = RiskEngine(RiskLimits(max_position_pct=0.1, max_positions=10, min_positions=1))
    res = engine.check_portfolio(
        target_weights={"AAPL": 0.25, "MSFT": 0.75},
        current_weights={},
        portfolio_value=100000,
        daily_pnl=0,
        peak_value=100000,
        cash=5000,
    )
    assert not res.approved
    assert any("Position cap exceeded" in v for v in res.violations)


def test_risk_adjust_weights_caps_and_normalizes():
    engine = RiskEngine(RiskLimits(max_position_pct=0.4))
    out = engine.adjust_weights({"A": 0.9, "B": 0.1}, result=RiskCheckResult(approved=False))
    assert abs(sum(out.values()) - 1.0) < 1e-8
    assert max(out.values()) <= 0.4 / (0.4 + 0.1)


def test_risk_kill_switch_triggers_on_drawdown():
    engine = RiskEngine(RiskLimits(max_drawdown_pct=0.1, max_daily_loss_pct=0.2))
    triggered = engine.check_kill_switch(
        current_value=85_000.0,
        peak_value=100_000.0,
        daily_pnl=0.0,
        portfolio_value_start_of_day=100_000.0,
    )
    assert triggered is True


def test_risk_drawdown_is_critical_severity():
    engine = RiskEngine(
        RiskLimits(
            max_position_pct=1.0,
            max_positions=10,
            min_positions=1,
            max_drawdown_pct=0.05,
            max_turnover_daily=2.0,
            min_cash_pct=0.0,
            max_portfolio_value=1_000_000_000.0,
        )
    )
    res = engine.check_portfolio(
        target_weights={"AAPL": 1.0},
        current_weights={"AAPL": 0.9},
        portfolio_value=100_000.0,
        daily_pnl=0.0,
        peak_value=120_000.0,
        cash=0.0,
    )
    assert not res.approved
    assert res.severity == "critical"


def test_risk_blocks_leverage_breach():
    engine = RiskEngine(
        RiskLimits(
            max_position_pct=1.0,
            max_positions=10,
            min_positions=1,
            max_turnover_daily=2.0,
            max_leverage=1.0,
            min_cash_pct=0.0,
            max_portfolio_value=1_000_000_000.0,
        )
    )
    res = engine.check_portfolio(
        target_weights={"AAPL": 0.8, "MSFT": 0.7},
        current_weights={},
        portfolio_value=100_000.0,
        daily_pnl=0.0,
        peak_value=100_000.0,
        cash=10_000.0,
    )
    assert not res.approved
    assert any("Leverage exceeded" in v for v in res.violations)
