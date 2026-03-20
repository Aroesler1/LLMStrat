from pathlib import Path

from quantaalpha_us.data.eodhd_client import EODHDClient


def test_get_eod_history_columns(monkeypatch, tmp_path: Path) -> None:
    client = EODHDClient(api_token="test", cache_dir=str(tmp_path / "cache"))

    def fake_request_json(path, params=None, use_cache=False, cache_ttl_seconds=None):  # noqa: ANN001
        return [
            {
                "date": "2026-02-25",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "adjusted_close": 101,
                "volume": 123456,
            }
        ]

    monkeypatch.setattr(client, "_request_json", fake_request_json)
    df = client.get_eod_history("aapl", exchange="US")
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"]
    assert df.iloc[0]["symbol"] == "AAPL"


def test_get_bulk_eod(monkeypatch, tmp_path: Path) -> None:
    client = EODHDClient(api_token="test", cache_dir=str(tmp_path / "cache"))

    def fake_request_json(path, params=None, use_cache=False, cache_ttl_seconds=None):  # noqa: ANN001
        return [
            {
                "code": "MSFT",
                "date": "2026-02-25",
                "open": 300,
                "high": 305,
                "low": 299,
                "close": 304,
                "adjusted_close": 304,
                "volume": 999,
            }
        ]

    monkeypatch.setattr(client, "_request_json", fake_request_json)
    df = client.get_bulk_eod(exchange="US", date="2026-02-25")
    assert not df.empty
    assert df.iloc[0]["symbol"] == "MSFT"
