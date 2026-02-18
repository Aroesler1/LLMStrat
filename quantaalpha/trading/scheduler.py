"""
APScheduler wrapper for rebalance + health jobs.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional


class TradingScheduler:
    def __init__(
        self,
        rebalance_callback: Callable[[], None],
        heartbeat_callback: Callable[[], None],
        config: Optional[Dict] = None,
    ):
        self.rebalance_callback = rebalance_callback
        self.heartbeat_callback = heartbeat_callback
        self.config = config or {}
        self._scheduler = None

    def start(self):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
        except Exception as e:
            raise ImportError("apscheduler is required for scheduled trading runtime") from e

        tz = self.config.get("timezone", "America/New_York")
        rebalance_cron = self.config.get("rebalance_cron", "45 15 * * 1-5")
        heartbeat_seconds = int(self.config.get("heartbeat_seconds", 60))

        self._scheduler = BackgroundScheduler(timezone=tz)
        self._scheduler.add_job(
            self.rebalance_callback,
            trigger=CronTrigger.from_crontab(rebalance_cron, timezone=tz),
            id="rebalance",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
        )
        self._scheduler.add_job(
            self.heartbeat_callback,
            trigger=IntervalTrigger(seconds=max(10, heartbeat_seconds), timezone=tz),
            id="heartbeat",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        self._scheduler.start()

    def stop(self, wait: bool = False):
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=wait)
            self._scheduler = None

    @property
    def running(self) -> bool:
        return self._scheduler is not None and bool(getattr(self._scheduler, "running", False))
