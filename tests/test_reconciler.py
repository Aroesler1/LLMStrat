from quantaalpha.storage.database import DatabaseManager
from quantaalpha.trading.broker import MockBroker
from quantaalpha.trading.reconciler import Reconciler


def test_reconciler_sync_positions(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.sqlite"))
    broker = MockBroker(equity=10000)
    broker.set_price("SPY", 500)
    broker.submit_order(symbol="SPY", qty=4, side="buy", order_type="market")

    rec = Reconciler(broker=broker, db=db)
    synced = rec.sync_positions()
    assert len(synced) == 1

    saved = db.get_positions()
    assert len(saved) == 1
    assert saved[0]["symbol"] == "SPY"


def test_reconciler_drift():
    drift = Reconciler.calculate_drift({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5})
    assert abs(drift - 0.2) < 1e-8
