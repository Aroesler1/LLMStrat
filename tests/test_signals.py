import pandas as pd

from quantaalpha.trading.market_data import MarketDataFetcher
from quantaalpha.trading.broker import MockBroker
from quantaalpha.trading.signals import SignalConfig, SignalGenerator


def test_signal_file_mode_uses_latest_date(tmp_path):
    p = tmp_path / "signals.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-01", "symbol": "AAA", "weight": 0.6},
            {"date": "2026-01-01", "symbol": "BBB", "weight": 0.4},
            {"date": "2026-01-02", "symbol": "AAA", "weight": 0.2},
            {"date": "2026-01-02", "symbol": "BBB", "weight": 0.8},
        ]
    ).to_csv(p, index=False)

    cfg = SignalConfig(
        mode="file",
        signal_file=str(p),
        max_weight=1.0,
        long_only=True,
    )
    gen = SignalGenerator(cfg, MarketDataFetcher(MockBroker()))
    w = gen.generate_target_weights(as_of=pd.Timestamp("2026-01-03"), universe=["AAA", "BBB"])

    assert set(w.keys()) == {"AAA", "BBB"}
    assert abs(w["AAA"] - 0.2) < 1e-8
    assert abs(w["BBB"] - 0.8) < 1e-8


def test_momentum_mode_generates_weights_with_mock_prices():
    broker = MockBroker()
    broker.set_price("AAA", 110)
    broker.set_price("BBB", 90)
    md = MarketDataFetcher(broker)
    cfg = SignalConfig(mode="momentum", universe=["AAA", "BBB"], topk=2, long_only=False)
    gen = SignalGenerator(cfg, md)

    w = gen.generate_target_weights(universe=["AAA", "BBB"])

    assert len(w) > 0
    assert abs(sum(w.values()) - 1.0) < 1e-8
