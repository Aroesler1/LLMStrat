from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class CheckResult:
    passed: bool
    severity: str
    details: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    passed: bool
    checks: dict[str, CheckResult]
    halt_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "halt_reason": self.halt_reason,
            "checks": {
                name: {
                    "passed": result.passed,
                    "severity": result.severity,
                    "details": result.details,
                    "metrics": result.metrics,
                }
                for name, result in self.checks.items()
            },
        }


class DataQualityGate:
    """Data quality checks for daily SP500 bar pipeline."""

    def __init__(
        self,
        min_members_warn: int = 490,
        min_members_halt: int = 480,
        outlier_abs_return: float = 0.25,
        max_gap_backfill_days: int = 5,
    ) -> None:
        self.min_members_warn = int(min_members_warn)
        self.min_members_halt = int(min_members_halt)
        self.outlier_abs_return = float(outlier_abs_return)
        self.max_gap_backfill_days = int(max_gap_backfill_days)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
        work["symbol"] = work["symbol"].astype(str).str.upper()
        for col in ("open", "high", "low", "close", "adj_close", "volume", "dollar_volume"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        return work.dropna(subset=["date", "symbol"])

    def check_freshness(self, df: pd.DataFrame, target_date: pd.Timestamp) -> CheckResult:
        latest = pd.to_datetime(df["date"], errors="coerce").max()
        passed = pd.notna(latest) and pd.Timestamp(latest).normalize() >= target_date
        details = (
            f"Latest bar date={latest.date()} target={target_date.date()}"
            if pd.notna(latest)
            else "No valid dates in dataframe"
        )
        return CheckResult(
            passed=bool(passed),
            severity="critical" if not passed else "info",
            details=details,
            metrics={"latest_date": latest.isoformat() if pd.notna(latest) else None},
        )

    def check_completeness(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
        membership_df: Optional[pd.DataFrame],
    ) -> CheckResult:
        same_day = df[df["date"] == target_date]
        actual = int(same_day["symbol"].nunique())

        expected = None
        if membership_df is not None and not membership_df.empty:
            m = membership_df.copy()
            m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
            m["symbol"] = m["symbol"].astype(str).str.upper()
            active_col = m["active"] if "active" in m.columns else True
            expected = int(m[(m["date"] == target_date) & (active_col == True)]["symbol"].nunique())

        passed = actual >= self.min_members_halt
        if actual < self.min_members_halt:
            severity = "critical"
        elif actual < self.min_members_warn:
            severity = "warning"
        else:
            severity = "info"

        details = f"Symbols with valid bars={actual}"
        if expected is not None:
            details += f" expected_active={expected}"

        return CheckResult(
            passed=passed,
            severity=severity,
            details=details,
            metrics={"actual": actual, "expected": expected},
        )

    @staticmethod
    def check_ohlc_validity(df: pd.DataFrame, target_date: pd.Timestamp) -> CheckResult:
        same_day = df[df["date"] == target_date].copy()
        if same_day.empty:
            return CheckResult(False, "critical", "No bars for target date", {"invalid_rows": 0})

        invalid = same_day[
            (same_day["low"] > same_day[["open", "close"]].min(axis=1))
            | (same_day["high"] < same_day[["open", "close"]].max(axis=1))
            | (same_day["low"] > same_day["high"])
            | (same_day["close"] <= 0)
            | (same_day["volume"] < 0)
        ]
        invalid_count = int(len(invalid))
        passed = invalid_count == 0
        severity = "critical" if invalid_count > 10 else ("warning" if invalid_count > 0 else "info")
        details = "OHLC validity passed" if passed else f"Invalid OHLC rows={invalid_count}"
        return CheckResult(passed, severity, details, {"invalid_rows": invalid_count})

    def check_return_outliers(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
        explained_symbols: Optional[set[str]] = None,
    ) -> CheckResult:
        explained_symbols = explained_symbols or set()

        work = df.sort_values(["symbol", "date"]).copy()
        px = work["adj_close"] if "adj_close" in work.columns else work["close"]
        work["ret_1d"] = px.groupby(work["symbol"]).pct_change()
        same_day = work[work["date"] == target_date]

        outliers = same_day[same_day["ret_1d"].abs() > self.outlier_abs_return]
        unexplained = outliers[~outliers["symbol"].isin(explained_symbols)]

        cnt_outlier = int(len(outliers))
        cnt_unexplained = int(len(unexplained))
        passed = cnt_unexplained == 0

        if cnt_unexplained > 0:
            severity = "warning"
            details = f"Unexplained outliers={cnt_unexplained} total_outliers={cnt_outlier}"
        else:
            severity = "info"
            details = f"Outlier check passed, total_outliers={cnt_outlier}"

        return CheckResult(
            passed=passed,
            severity=severity,
            details=details,
            metrics={
                "outliers": cnt_outlier,
                "unexplained_outliers": cnt_unexplained,
                "symbols": sorted(unexplained["symbol"].unique().tolist()),
            },
        )

    @staticmethod
    def check_duplicates(df: pd.DataFrame, target_date: pd.Timestamp) -> CheckResult:
        same_day = df[df["date"] == target_date]
        dup_mask = same_day.duplicated(subset=["date", "symbol"], keep=False)
        dup_count = int(dup_mask.sum())
        passed = dup_count == 0
        severity = "critical" if dup_count > 0 else "info"
        details = "No duplicates" if passed else f"Duplicate rows={dup_count}"
        return CheckResult(passed, severity, details, {"duplicates": dup_count})

    def check_gaps(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
        membership_df: Optional[pd.DataFrame],
    ) -> CheckResult:
        if membership_df is None or membership_df.empty:
            return CheckResult(True, "info", "Membership dataframe not provided", {"missing": 0})

        m = membership_df.copy()
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
        m["symbol"] = m["symbol"].astype(str).str.upper()
        if "active" in m.columns:
            expected = set(m[(m["date"] == target_date) & (m["active"] == True)]["symbol"])  # noqa: E712
        else:
            expected = set(m[m["date"] == target_date]["symbol"])

        actual = set(df[df["date"] == target_date]["symbol"])
        missing = sorted(expected - actual)
        passed = len(missing) <= max(0, len(expected) - self.min_members_halt)
        severity = "critical" if not passed else ("warning" if missing else "info")

        details = "No unexpected symbol gaps" if not missing else f"Missing active symbols={len(missing)}"
        return CheckResult(
            passed=passed,
            severity=severity,
            details=details,
            metrics={"missing": len(missing), "sample": missing[:20]},
        )

    @staticmethod
    def check_adjusted_consistency(df: pd.DataFrame, target_date: pd.Timestamp) -> CheckResult:
        if "adj_close" not in df.columns:
            return CheckResult(True, "info", "adj_close not provided; check skipped", {})

        work = df.sort_values(["symbol", "date"]).copy()
        work = work[(work["close"] > 0) & (work["adj_close"] > 0)]
        work["adj_ratio"] = work["adj_close"] / work["close"]
        work["adj_ratio_prev"] = work.groupby("symbol")["adj_ratio"].shift(1)
        same_day = work[work["date"] == target_date].copy()

        if same_day.empty:
            return CheckResult(False, "critical", "No data for adjusted consistency check", {"jumps": 0})

        same_day["ratio_jump"] = (same_day["adj_ratio"] / same_day["adj_ratio_prev"] - 1.0).abs()
        jumps = same_day[same_day["ratio_jump"] > 0.5]
        jump_count = int(len(jumps))

        passed = jump_count == 0
        severity = "warning" if jump_count > 0 else "info"
        details = "Adjusted ratio consistency passed" if passed else f"Large adjusted-ratio jumps={jump_count}"

        return CheckResult(
            passed=passed,
            severity=severity,
            details=details,
            metrics={"jumps": jump_count, "sample_symbols": sorted(jumps["symbol"].head(20).tolist())},
        )

    def check_gap_backfill(self, df: pd.DataFrame, target_date: pd.Timestamp) -> CheckResult:
        latest = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(latest):
            return CheckResult(
                passed=False,
                severity="critical",
                details="Cannot evaluate gap backfill check; latest date missing",
                metrics={"latest_date": None, "target_date": str(target_date.date()), "gap_business_days": None},
            )

        latest_date = pd.Timestamp(latest).normalize()
        target = pd.Timestamp(target_date).normalize()
        if latest_date >= target:
            return CheckResult(
                passed=True,
                severity="info",
                details="No data gap detected before target date",
                metrics={"latest_date": str(latest_date.date()), "target_date": str(target.date()), "gap_business_days": 0},
            )

        gap_days = len(pd.bdate_range(latest_date + pd.Timedelta(days=1), target, freq="B"))
        if gap_days <= 1:
            return CheckResult(
                passed=True,
                severity="info",
                details="No intermediate gap requiring backfill",
                metrics={"latest_date": str(latest_date.date()), "target_date": str(target.date()), "gap_business_days": gap_days},
            )
        if gap_days <= self.max_gap_backfill_days:
            return CheckResult(
                passed=False,
                severity="warning",
                details=f"Detected {gap_days} missing business days; backfill required before bulk ingest",
                metrics={"latest_date": str(latest_date.date()), "target_date": str(target.date()), "gap_business_days": gap_days},
            )
        return CheckResult(
            passed=False,
            severity="critical",
            details=f"Gap too large ({gap_days} business days) for automatic backfill",
            metrics={"latest_date": str(latest_date.date()), "target_date": str(target.date()), "gap_business_days": gap_days},
        )

    def run_all_checks(
        self,
        df: pd.DataFrame,
        *,
        date: str | pd.Timestamp,
        membership_df: Optional[pd.DataFrame] = None,
        explained_outlier_symbols: Optional[set[str]] = None,
        mode: str = "strict",
    ) -> DataQualityReport:
        if df.empty:
            empty_checks = {
                "freshness": CheckResult(False, "critical", "Input dataframe is empty", {}),
            }
            return DataQualityReport(False, empty_checks, halt_reason="Empty input dataframe")

        target_date = pd.Timestamp(date).normalize()
        data = self._normalize(df)

        checks: dict[str, CheckResult] = {
            "freshness": self.check_freshness(data, target_date),
            "completeness": self.check_completeness(data, target_date, membership_df),
            "ohlc_validity": self.check_ohlc_validity(data, target_date),
            "return_outliers": self.check_return_outliers(data, target_date, explained_outlier_symbols),
            "duplicates": self.check_duplicates(data, target_date),
            "gaps": self.check_gaps(data, target_date, membership_df),
            "adjusted_consistency": self.check_adjusted_consistency(data, target_date),
            "gap_backfill": self.check_gap_backfill(data, target_date),
        }

        if mode == "lenient":
            halt_failures = [
                name
                for name, result in checks.items()
                if not result.passed and (
                    name in {"freshness", "completeness"}
                    or (name == "gap_backfill" and result.severity == "critical")
                )
            ]
        else:
            halt_failures = [name for name, result in checks.items() if not result.passed]

        passed = len(halt_failures) == 0
        halt_reason = None if passed else f"Failed checks: {', '.join(halt_failures)}"
        return DataQualityReport(passed=passed, checks=checks, halt_reason=halt_reason)
