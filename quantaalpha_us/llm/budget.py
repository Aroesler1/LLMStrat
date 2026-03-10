from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple


@dataclass
class RunBudget:
    """Track request, token, and failure limits for one LLM run."""

    max_requests: int = 250
    max_total_tokens: int = 250000
    max_consecutive_failures: int = 3

    requests_used: int = 0
    total_tokens_used: int = 0
    consecutive_failures: int = 0

    def can_request(self, estimated_tokens: int = 0) -> bool:
        if self.requests_used >= self.max_requests:
            return False
        if self.total_tokens_used + max(estimated_tokens, 0) > self.max_total_tokens:
            return False
        if self.consecutive_failures >= self.max_consecutive_failures:
            return False
        return True

    def record_request(self, tokens_used: int, success: bool) -> None:
        self.requests_used += 1
        self.total_tokens_used += max(int(tokens_used), 0)
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "max_requests": self.max_requests,
            "max_total_tokens": self.max_total_tokens,
            "max_consecutive_failures": self.max_consecutive_failures,
            "requests_used": self.requests_used,
            "total_tokens_used": self.total_tokens_used,
            "consecutive_failures": self.consecutive_failures,
        }


def _extract_total_tokens(response: Any, default: int = 0) -> int:
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, (int, float)):
                return int(total)
    return int(default)


def _is_valid_response(response: Any) -> bool:
    if response is None:
        return False
    if isinstance(response, dict):
        if response.get("error"):
            return False
        if response.get("choices") == []:
            return False
    return True


def call_with_fallback(
    call_model: Callable[[str, str], Any],
    *,
    prompt: str,
    models: Iterable[str],
    budget: RunBudget,
    estimated_tokens_per_request: int = 0,
) -> Tuple[Any, str]:
    """Try models in order and stop when one returns a valid response."""
    errors: dict[str, str] = {}

    for model in models:
        if not budget.can_request(estimated_tokens=estimated_tokens_per_request):
            raise RuntimeError(
                "LLM budget exhausted "
                f"(requests={budget.requests_used}/{budget.max_requests}, "
                f"tokens={budget.total_tokens_used}/{budget.max_total_tokens}, "
                f"consecutive_failures={budget.consecutive_failures}/{budget.max_consecutive_failures})"
            )

        try:
            response = call_model(model, prompt)
            success = _is_valid_response(response)
            used_tokens = _extract_total_tokens(response, default=estimated_tokens_per_request)
            budget.record_request(tokens_used=used_tokens, success=success)

            if success:
                return response, model

            errors[model] = "Invalid response payload"
        except Exception as exc:  # noqa: BLE001
            budget.record_request(tokens_used=estimated_tokens_per_request, success=False)
            errors[model] = str(exc)

        if budget.consecutive_failures >= budget.max_consecutive_failures:
            break

    raise RuntimeError(f"All models failed or budget exceeded: {errors}")
