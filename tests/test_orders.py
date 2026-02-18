from quantaalpha.trading.broker import MockBroker
import quantaalpha.trading.orders as orders_module
from quantaalpha.trading.orders import OrderManager, OrderState, ProposedOrder


def test_generate_orders_sell_before_buy_and_min_trade_filter():
    current = {"AAA": 10, "BBB": 0}
    target = {"AAA": 5, "BBB": 10}
    prices = {"AAA": 100, "BBB": 20}

    orders = OrderManager.generate_orders(
        current_shares=current,
        target_shares=target,
        prices=prices,
        min_trade_dollars=50,
    )

    assert len(orders) == 2
    assert orders[0].side == "sell"
    assert orders[0].symbol == "AAA"
    assert orders[1].side == "buy"
    assert orders[1].symbol == "BBB"


def test_submit_orders_with_mock_broker_returns_ids():
    broker = MockBroker(equity=10000)
    broker.set_price("AAA", 100)
    mgr = OrderManager(broker=broker)

    current = {}
    target = {"AAA": 2}
    prices = {"AAA": 100}
    orders = mgr.generate_orders(current, target, prices, min_trade_dollars=10)
    ids = mgr.submit_orders(orders=orders, rebalance_id="rb1")

    assert len(ids) == 1
    st = broker.get_order_status(ids[0])
    assert st.status == "filled"


def test_submit_orders_respects_rate_limit(monkeypatch):
    broker = MockBroker(equity=10000)
    broker.set_price("AAA", 100)
    broker.set_price("BBB", 100)
    broker.set_price("CCC", 100)
    mgr = OrderManager(broker=broker, rate_limit_per_second=2.0)

    clock = {"t": 0.0, "sleeps": []}

    def fake_monotonic():
        return clock["t"]

    def fake_sleep(seconds: float):
        clock["sleeps"].append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(orders_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(orders_module.time, "sleep", fake_sleep)

    orders = [
        ProposedOrder(symbol="AAA", side="buy", qty=1.0),
        ProposedOrder(symbol="BBB", side="buy", qty=1.0),
        ProposedOrder(symbol="CCC", side="buy", qty=1.0),
    ]
    ids = mgr.submit_orders(orders=orders, rebalance_id="rb-rate-limit")
    assert len(ids) == 3
    assert len(clock["sleeps"]) == 2
    assert all(s >= 0.5 for s in clock["sleeps"])


def test_order_state_transition_validation_blocks_invalid_backwards_move():
    broker = MockBroker(equity=10000)
    mgr = OrderManager(broker=broker)
    oid = "mock-1"
    mgr._order_states[oid] = OrderState.FILLED

    updated = mgr._update_order_state(oid=oid, next_state=OrderState.PARTIAL)
    assert updated is False
    assert mgr._order_states[oid] == OrderState.FILLED
