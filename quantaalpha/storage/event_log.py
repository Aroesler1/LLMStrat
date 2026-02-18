"""
Structured event logger backed by SQLite.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from quantaalpha.storage.database import DatabaseManager


class EventLogger:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def log(self, event_type: str, severity: str, message: str, data: Optional[Any] = None):
        payload = None
        if data is not None:
            try:
                payload = json.dumps(data, ensure_ascii=False, default=str)
            except Exception:
                payload = str(data)
        self.db.log_event(event_type=event_type, severity=severity, message=message, data_json=payload)
