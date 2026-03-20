from __future__ import annotations

import json
import re
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
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part.strip() for part in parts if str(part).strip()).strip()
        if content is None:
            return ""
        return str(content).strip()

    @staticmethod
    def _extract_from_factors_list(items: Any) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
                continue
            if isinstance(item, dict):
                for key in ("expression", "expr", "factor", "formula"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        out.append(value.strip())
                        break
        return out

    @classmethod
    def _extract_json_payload(cls, text: str) -> list[str]:
        text = str(text).strip()
        if not text:
            return []

        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                direct = cls._extract_from_factors_list(parsed.get("factors"))
                if direct:
                    return direct
        return []

    @staticmethod
    def _extract_line_candidates(text: str) -> list[str]:
        out: list[str] = []
        for raw in str(text).splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\).\s-]+", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            out.append(line.strip())
        return out

    @classmethod
    def _extract_factor_candidates(cls, response: Any) -> list[str]:
        if response is None:
            return []
        if isinstance(response, dict):
            direct = cls._extract_from_factors_list(response.get("factors"))
            if direct:
                return direct
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                text = cls._content_to_text(msg.get("content", ""))
                if not text:
                    return []
                extracted = cls._extract_json_payload(text)
                if extracted:
                    return extracted
                return cls._extract_line_candidates(text)
        if isinstance(response, str):
            extracted = cls._extract_json_payload(response)
            if extracted:
                return extracted
            return cls._extract_line_candidates(response)
        return []

    def run(
        self,
        *,
        prompts: Iterable[str],
        models: list[str],
        call_model: Callable[[str, str], Any],
        estimated_tokens_per_request: int = 512,
        target_valid_factors: int | None = None,
    ) -> tuple[list[str], MiningStats]:
        valid: list[str] = []
        seen_valid: set[str] = set()
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
                    if sanitized.cleaned not in seen_valid:
                        valid.append(sanitized.cleaned)
                        seen_valid.add(sanitized.cleaned)
                else:
                    invalid += 1

            if target_valid_factors is not None and len(valid) >= int(target_valid_factors):
                valid = valid[: int(target_valid_factors)]
                halted_reason = None
                break

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
