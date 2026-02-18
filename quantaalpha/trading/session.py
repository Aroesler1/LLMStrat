"""
Market session helpers for US retail trading cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo


@dataclass
class SessionConfig:
    timezone: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"
    holidays: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "SessionConfig":
        cfg = cfg or {}
        return cls(
            timezone=str(cfg.get("timezone", "America/New_York")),
            open_time=str(cfg.get("open_time", "09:30")),
            close_time=str(cfg.get("close_time", "16:00")),
            holidays=list(cfg.get("holidays", []) or []),
        )


class MarketSession:
    def __init__(self, config: SessionConfig):
        self.cfg = config
        self.tz = ZoneInfo(config.timezone)
        self._holidays = {str(x) for x in (config.holidays or [])}
        self._open_t = self._parse_hhmm(config.open_time)
        self._close_t = self._parse_hhmm(config.close_time)

    def now(self) -> datetime:
        return datetime.now(tz=self.tz)

    def is_trading_day(self, dt: date | datetime) -> bool:
        d = dt.date() if isinstance(dt, datetime) else dt
        if d.weekday() >= 5:
            return False
        if d.isoformat() in self._holidays:
            return False
        return True

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        now = (dt or self.now()).astimezone(self.tz)
        if not self.is_trading_day(now):
            return False
        cur_t = now.time().replace(tzinfo=None)
        return self._open_t <= cur_t <= self._close_t

    def is_before_close(self, minutes: int, dt: Optional[datetime] = None) -> bool:
        now = (dt or self.now()).astimezone(self.tz)
        if not self.is_trading_day(now):
            return False
        close_dt = datetime.combine(now.date(), self._close_t, tzinfo=self.tz)
        delta_min = (close_dt - now).total_seconds() / 60.0
        return 0.0 <= delta_min <= float(minutes)

    def next_trading_day(self, dt: Optional[datetime] = None) -> date:
        cur = (dt or self.now()).astimezone(self.tz).date()
        while True:
            cur = cur + timedelta(days=1)
            if self.is_trading_day(cur):
                return cur

    def normalize_holiday_list(self, holidays: Iterable[date | str]) -> None:
        out = set()
        for h in holidays:
            if isinstance(h, date):
                out.add(h.isoformat())
            else:
                out.add(str(h))
        self._holidays = out

    @staticmethod
    def _parse_hhmm(s: str) -> time:
        try:
            hh, mm = s.split(":", 1)
            return time(int(hh), int(mm))
        except Exception:
            return time(9, 30)
