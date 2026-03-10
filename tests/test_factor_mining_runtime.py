from quantaalpha_us.llm.budget import RunBudget
from quantaalpha_us.llm.mining import FactorMiningRuntime


def test_factor_mining_runtime_applies_sanitizer_and_budget() -> None:
    budget = RunBudget(max_requests=5, max_total_tokens=5000, max_consecutive_failures=3)
    runtime = FactorMiningRuntime(budget=budget, max_batch_failure_rate=0.8)

    def fake_call(model: str, prompt: str):  # noqa: ANN001
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"factors":["TS_MEAN($close,20)","import os"]}',
                    }
                }
            ],
            "usage": {"total_tokens": 50},
        }

    factors, stats = runtime.run(
        prompts=["p1", "p2"],
        models=["m1"],
        call_model=fake_call,
        estimated_tokens_per_request=32,
    )
    assert len(factors) >= 1
    assert stats.valid_factors >= 1
    assert stats.invalid_factors >= 1
