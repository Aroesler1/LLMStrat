from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class AlpacaAPIError(RuntimeError):
    pass


@dataclass
class AlpacaRESTClient:
    api_key: str
    api_secret: str
    base_url: str
    timeout_seconds: float = 30.0

    def _request(self, method: str, path: str, *, params: Optional[dict[str, Any]] = None, payload: Optional[dict[str, Any]] = None) -> Any:
        url = self.base_url.rstrip("/") + path
        if params:
            query = urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                url = f"{url}?{query}"

        body = None
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url=url, method=method.upper(), headers=headers, data=body)
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
        except HTTPError as e:
            raw_err = e.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw_err)
            except json.JSONDecodeError:
                parsed = {"raw": raw_err}
            raise AlpacaAPIError(f"Alpaca API error status={e.code} path={path} details={parsed}") from e

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/v2/account")

    def list_positions(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/v2/positions")
        return resp if isinstance(resp, list) else []

    def list_open_orders(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/v2/orders", params={"status": "open", "limit": 500})
        return resp if isinstance(resp, list) else []

    def cancel_all_orders(self) -> Any:
        return self._request("DELETE", "/v2/orders")

    def close_all_positions(self) -> Any:
        return self._request("DELETE", "/v2/positions")

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        if qty is None and notional is None:
            raise ValueError("Either qty or notional must be provided")

        payload: dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "side": side.lower(),
            "type": "market",
            "time_in_force": time_in_force,
            "extended_hours": bool(extended_hours),
        }

        if notional is not None:
            payload["notional"] = round(float(notional), 2)
        if qty is not None:
            payload["qty"] = str(round(float(qty), 6))

        resp = self._request("POST", "/v2/orders", payload=payload)
        return resp if isinstance(resp, dict) else {"response": resp}
