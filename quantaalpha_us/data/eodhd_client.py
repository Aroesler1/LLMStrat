from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Simple token-bucket-like limiter based on rolling timestamps."""

    max_calls: int
    period_seconds: float

    def __post_init__(self) -> None:
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.period_seconds:
            self._calls.popleft()

        if len(self._calls) >= self.max_calls:
            wait_for = self.period_seconds - (now - self._calls[0])
            if wait_for > 0:
                time.sleep(wait_for)
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period_seconds:
                self._calls.popleft()

        self._calls.append(time.monotonic())


class EODHDClient:
    """Thin EODHD REST client with retry, rate limiting, and optional file cache."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        api_token_env: str = "EODHD_API_TOKEN",
        base_url: str = "https://eodhd.com/api",
        cache_dir: str = "data/us_equities/raw/cache",
        max_calls_per_minute: int = 500,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        token = api_token or os.environ.get(api_token_env)
        if not token:
            raise ValueError(
                "Missing EODHD API token. Set EODHD_API_TOKEN or pass api_token explicitly."
            )

        self.api_token = token
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.rate_limiter = RateLimiter(max_calls=max_calls_per_minute, period_seconds=60.0)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_symbol(symbol: str, exchange: str = "US") -> str:
        s = str(symbol).strip().upper()
        if "." in s:
            return s
        return f"{s}.{exchange.upper()}"

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        raw = json.dumps({"path": path, "params": params}, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _request_json(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        *,
        use_cache: bool = False,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Any:
        payload = dict(params or {})
        payload["api_token"] = self.api_token
        if "fmt" not in payload:
            payload["fmt"] = "json"

        cache_file = self._cache_path(path=path, params=payload)
        if use_cache and cache_file.exists():
            if cache_ttl_seconds is None:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            age = time.time() - cache_file.stat().st_mtime
            if age <= cache_ttl_seconds:
                return json.loads(cache_file.read_text(encoding="utf-8"))

        query = urlencode({k: v for k, v in payload.items() if v is not None})
        url = f"{self.base_url}{path}?{query}"

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            req = Request(url, headers={"Accept": "application/json"}, method="GET")
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                if use_cache:
                    cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data
            except HTTPError as e:
                last_error = e
                status = int(getattr(e, "code", 0))
                retryable = status == 429 or status >= 500
                if not retryable or attempt >= self.max_retries:
                    raise
                backoff = min(2 ** attempt, 30)
                logger.warning("EODHD request retry: status=%s attempt=%s backoff=%ss", status, attempt + 1, backoff)
                time.sleep(backoff)
            except (URLError, json.JSONDecodeError) as e:
                last_error = e
                if attempt >= self.max_retries:
                    raise
                backoff = min(2 ** attempt, 30)
                logger.warning("EODHD request retry: error=%s attempt=%s backoff=%ss", e, attempt + 1, backoff)
                time.sleep(backoff)

        raise RuntimeError(f"EODHD request failed after retries: {last_error}")

    def get_eod_history(
        self,
        symbol: str,
        exchange: str = "US",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        *,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        symbol_code = self._normalize_symbol(symbol, exchange=exchange)
        data = self._request_json(
            f"/eod/{symbol_code}",
            params={"period": "d", "from": from_date, "to": to_date},
            use_cache=use_cache,
        )
        if not isinstance(data, list):
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"])

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"])

        rename_map = {
            "adjusted_close": "adj_close",
            "adjustedClose": "adj_close",
        }
        df = df.rename(columns=rename_map)
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.NA
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["symbol"] = str(symbol).upper()
        return df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]].dropna(subset=["date"])

    def get_bulk_eod(
        self,
        exchange: str = "US",
        date: Optional[str] = None,
        *,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        data = self._request_json(
            f"/eod-bulk-last-day/{exchange.upper()}",
            params={"date": date},
            use_cache=use_cache,
        )
        if not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        rename_map = {
            "code": "symbol",
            "adjusted_close": "adj_close",
            "adjustedClose": "adj_close",
        }
        df = df.rename(columns=rename_map)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_sp500_constituents_historical(self, *, use_cache: bool = True) -> pd.DataFrame:
        """Return S&P 500 historical constituents from EODHD fundamentals endpoint."""
        data = self._request_json(
            "/fundamentals/GSPC.INDX",
            params={"historical": 1, "filter": "HistoricalTickerComponents"},
            use_cache=use_cache,
            cache_ttl_seconds=3600,
        )

        if isinstance(data, dict):
            # Defensive handling if endpoint returns wrapped payload.
            for key in ("HistoricalTickerComponents", "Components", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            return pd.DataFrame(
                columns=[
                    "Code",
                    "Exchange",
                    "Name",
                    "Sector",
                    "Industry",
                    "StartDate",
                    "EndDate",
                    "IsActiveNow",
                    "IsDelisted",
                ]
            )

        df = pd.DataFrame(data)
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str.upper()
        for col in ("StartDate", "EndDate"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
        for col in ("IsActiveNow", "IsDelisted"):
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df

    def get_splits(self, symbol: str, exchange: str = "US", *, use_cache: bool = False) -> pd.DataFrame:
        symbol_code = self._normalize_symbol(symbol, exchange=exchange)
        data = self._request_json(f"/splits/{symbol_code}", params={}, use_cache=use_cache)
        df = pd.DataFrame(data if isinstance(data, list) else [])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df

    def get_dividends(self, symbol: str, exchange: str = "US", *, use_cache: bool = False) -> pd.DataFrame:
        symbol_code = self._normalize_symbol(symbol, exchange=exchange)
        data = self._request_json(f"/div/{symbol_code}", params={}, use_cache=use_cache)
        df = pd.DataFrame(data if isinstance(data, list) else [])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df

    def get_splits_dividends(self, symbol: str, exchange: str = "US") -> dict[str, pd.DataFrame]:
        return {
            "splits": self.get_splits(symbol=symbol, exchange=exchange),
            "dividends": self.get_dividends(symbol=symbol, exchange=exchange),
        }

    def get_delisted_tickers(self, exchange: str = "US", *, use_cache: bool = True) -> pd.DataFrame:
        data = self._request_json(
            f"/exchange-symbol-list/{exchange.upper()}",
            params={"type": "delisted"},
            use_cache=use_cache,
            cache_ttl_seconds=3600,
        )
        return pd.DataFrame(data if isinstance(data, list) else [])

    def get_exchange_symbols(self, exchange: str = "US", *, use_cache: bool = True) -> pd.DataFrame:
        data = self._request_json(
            f"/exchange-symbol-list/{exchange.upper()}",
            params={},
            use_cache=use_cache,
            cache_ttl_seconds=3600,
        )
        return pd.DataFrame(data if isinstance(data, list) else [])
