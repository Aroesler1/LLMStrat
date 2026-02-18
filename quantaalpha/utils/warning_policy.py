from __future__ import annotations

import logging
import warnings

EXPECTED_PYTHONWARNING_RULES = [
    "ignore:Mean of empty slice:RuntimeWarning:qlib\\.utils\\.index_data",
    "ignore:.*Gym has been unmaintained since 2022.*:UserWarning",
]


def build_pythonwarnings_filter(existing: str | None) -> str:
    """Merge project warning policy into PYTHONWARNINGS without duplicates."""
    merged: list[str] = []
    if existing:
        merged.extend([item.strip() for item in existing.split(",") if item.strip()])
    for rule in EXPECTED_PYTHONWARNING_RULES:
        if rule not in merged:
            merged.append(rule)
    return ",".join(merged)


def apply_runtime_warning_filters() -> None:
    """Silence known benign warnings that do not affect backtest correctness."""
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
        module=r"qlib\.utils\.index_data",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Gym has been unmaintained since 2022.*",
        category=UserWarning,
    )
    # Gym may emit startup warnings via its own logger path.
    logging.getLogger("gym").setLevel(logging.ERROR)
    logging.getLogger("gymnasium").setLevel(logging.ERROR)
