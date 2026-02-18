from quantaalpha.backtest.cost_model import CostModel, CostModelConfig


def test_turnover_cost_rate_positive():
    model = CostModel(CostModelConfig(spread_bps=4.0, slippage_bps=6.0))
    c = model.estimate_turnover_cost_rate(turnover=0.5)
    assert c > 0


def test_trade_cost_respects_minimum():
    model = CostModel(CostModelConfig(spread_bps=0.0, slippage_bps=0.0, min_trade_cost=1.0))
    c = model.estimate_trade_cost_dollars(shares=1, price=0.5)
    assert c == 1.0
