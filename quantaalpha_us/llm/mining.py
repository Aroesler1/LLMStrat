from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer
from quantaalpha_us.llm.budget import RunBudget, call_with_fallback


@dataclass
class MiningStats:
    prompts_attempted: int
    requests_used: int
    tokens_used: int
    valid_factors: int
    invalid_factors: int
    parse_error_rate: float
    halted_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompts_attempted": self.prompts_attempted,
            "requests_used": self.requests_used,
            "tokens_used": self.tokens_used,
            "valid_factors": self.valid_factors,
            "invalid_factors": self.invalid_factors,
            "parse_error_rate": self.parse_error_rate,
            "halted_reason": self.halted_reason,
        }


class FactorMiningRuntime:
    """Execute factor mining with sanitizer and run-budget controls."""

    def __init__(
        self,
        *,
        budget: RunBudget,
        sanitizer: ExpressionSanitizer | None = None,
        max_batch_failure_rate: float = 0.50,
    ) -> None:
        self.budget = budget
        self.sanitizer = sanitizer or ExpressionSanitizer()
        self.max_batch_failure_rate = float(max_batch_failure_rate)

    @staticmethod
    def _extract_factor_candidates(response: Any) -> list[str]:
        if response is None:
            return []
        if isinstance(response, dict):
            if isinstance(response.get("factors"), list):
                return [str(x).strip() for x in response["factors"] if str(x).strip()]
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = msg.get("content", "")
                text = str(content).strip()
                if not text:
                    return []
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and isinstance(parsed.get("factors"), list):
                        return [str(x).strip() for x in parsed["factors"] if str(x).strip()]
                except Exception:
                    pass
                return [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(response, str):
            return [line.strip() for line in response.splitlines() if line.strip()]
        return []

    def run(
        self,
        *,
        prompts: Iterable[str],
        models: list[str],
        call_model: Callable[[str, str], Any],
        estimated_tokens_per_request: int = 512,
    ) -> tuple[list[str], MiningStats]:
        valid: list[str] = []
        invalid = 0
        attempted = 0
        halted_reason: str | None = None

        for prompt in prompts:
            attempted += 1
            try:
                response, _model = call_with_fallback(
                    call_model,
                    prompt=prompt,
                    models=models,
                    budget=self.budget,
                    estimated_tokens_per_request=estimated_tokens_per_request,
                )
            except Exception as exc:  # noqa: BLE001
                halted_reason = str(exc)
                break

            candidates = self._extract_factor_candidates(response)
            for expr in candidates:
                sanitized = self.sanitizer.sanitize(expr)
                if sanitized.valid:
                    valid.append(sanitized.cleaned)
                else:
                    invalid += 1

            total_seen = len(valid) + invalid
            if total_seen > 0:
                failure_rate = invalid / float(total_seen)
                if failure_rate > self.max_batch_failure_rate:
                    halted_reason = (
                        f"Invalid expression rate {failure_rate:.2%} "
                        f"exceeded threshold {self.max_batch_failure_rate:.2%}"
                    )
                    break

        total_seen = len(valid) + invalid
        parse_error_rate = (invalid / float(total_seen)) if total_seen > 0 else 0.0
        stats = MiningStats(
            prompts_attempted=attempted,
            requests_used=self.budget.requests_used,
            tokens_used=self.budget.total_tokens_used,
            valid_factors=len(valid),
            invalid_factors=invalid,
            parse_error_rate=parse_error_rate,
            halted_reason=halted_reason,
        )
        return valid, stats
