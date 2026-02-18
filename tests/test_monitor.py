from quantaalpha.storage.database import DatabaseManager
from quantaalpha.trading.broker import MockBroker
from quantaalpha.trading.monitor import HealthMonitor


def test_health_monitor_reports_broker_connectivity_ok(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.sqlite"))
    broker = MockBroker(equity=10_000.0)
    monitor = HealthMonitor(db=db, stale_seconds=60, broker=broker)
    monitor.heartbeat("engine")
    monitor.heartbeat("scheduler")

    status = monitor.check_health()
    assert status["healthy"] is True
    assert status["broker_connectivity"]["ok"] is True
    assert status["broker_connectivity"]["equity"] > 0


def test_health_monitor_marks_unhealthy_on_broker_failure(tmp_path):
    class FailingBroker:
        def get_account(self):
            raise RuntimeError("broker down")

    db = DatabaseManager(str(tmp_path / "state.sqlite"))
    monitor = HealthMonitor(db=db, stale_seconds=60, broker=FailingBroker())  # type: ignore[arg-type]
    monitor.heartbeat("engine")
    monitor.heartbeat("scheduler")

    status = monitor.check_health()
    assert status["healthy"] is False
    assert status["broker_connectivity"]["ok"] is False
    assert "broker.connectivity" in status["stale_components"]
