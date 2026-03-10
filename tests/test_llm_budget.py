from quantaalpha_us.llm.budget import RunBudget, call_with_fallback


def test_budget_limits_requests() -> None:
    budget = RunBudget(max_requests=1, max_total_tokens=100, max_consecutive_failures=3)
    assert budget.can_request()
    budget.record_request(tokens_used=10, success=True)
    assert not budget.can_request()


def test_fallback_uses_second_model() -> None:
    budget = RunBudget(max_requests=5, max_total_tokens=1000, max_consecutive_failures=3)

    def call_model(model: str, prompt: str):
        if model == "bad":
            raise RuntimeError("unavailable")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 12}}

    response, used_model = call_with_fallback(
        call_model,
        prompt="ping",
        models=["bad", "good"],
        budget=budget,
        estimated_tokens_per_request=8,
    )
    assert used_model == "good"
    assert response["choices"][0]["message"]["content"] == "ok"
    assert budget.requests_used == 2
