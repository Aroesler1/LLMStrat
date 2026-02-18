from quantaalpha.trading.broker import MockBroker


def test_mock_broker_buy_and_sell_flow():
    b = MockBroker(equity=10000)
    b.set_price("AAPL", 100.0)

    oid = b.submit_order(symbol="AAPL", qty=10, side="buy", order_type="market")
    st = b.get_order_status(oid)
    assert st.status == "filled"

    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10

    oid2 = b.submit_order(symbol="AAPL", qty=5, side="sell", order_type="market")
    st2 = b.get_order_status(oid2)
    assert st2.status == "filled"

    positions = b.get_positions()
    assert positions[0].qty == 5
