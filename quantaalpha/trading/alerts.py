"""
Alert helpers for trading runtime.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from quantaalpha.storage.event_log import EventLogger


class AlertManager:
    def __init__(self, event_logger: Optional[EventLogger] = None, verbose: bool = True):
        self.event_logger = event_logger
        self.verbose = verbose

    def notify(self, severity: str, message: str, data: Optional[Any] = None, event_type: str = "alert"):
        if self.verbose:
            level = str(severity or "info").upper()
            if data is not None:
                logger.bind(event_type=event_type, payload=data).log(level, message)
            else:
                logger.bind(event_type=event_type).log(level, message)
        if self.event_logger is not None:
            self.event_logger.log(event_type=event_type, severity=severity, message=message, data=data)

    def info(self, message: str, data: Optional[Any] = None, event_type: str = "alert"):
        self.notify("info", message, data=data, event_type=event_type)

    def warning(self, message: str, data: Optional[Any] = None, event_type: str = "alert"):
        self.notify("warning", message, data=data, event_type=event_type)

    def critical(self, message: str, data: Optional[Any] = None, event_type: str = "alert"):
        self.notify("critical", message, data=data, event_type=event_type)
