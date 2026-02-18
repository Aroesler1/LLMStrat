import pytest

from quantaalpha.llm import client as llm_client
from quantaalpha.llm.client import APIBackend


class _DummyEncoder:
    def encode(self, text: str) -> list[int]:
        return [1 for _ in text]


def test_calculate_token_from_messages_lazily_initializes_encoder(monkeypatch) -> None:
    backend = object.__new__(APIBackend)
    backend.use_llama2 = False
    backend.use_gcr_endpoint = False
    backend.chat_model = "gpt-4"
    monkeypatch.setattr(APIBackend, "_get_encoder", lambda self: _DummyEncoder())

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello world"},
    ]

    tokens = APIBackend.calculate_token_from_messages(backend, messages)
    assert tokens > 0
    assert getattr(backend, "encoder", None) is not None


def test_calculate_token_from_messages_handles_unknown_model_name(monkeypatch) -> None:
    backend = object.__new__(APIBackend)
    backend.use_llama2 = False
    backend.use_gcr_endpoint = False
    backend.chat_model = "custom_deployment_name"

    def _raise_key_error(self):
        raise KeyError("unknown model")

    monkeypatch.setattr(APIBackend, "_get_encoder", _raise_key_error)
    monkeypatch.setattr("quantaalpha.llm.client.tiktoken.get_encoding", lambda _: _DummyEncoder())

    messages = [{"role": "user", "content": "Token count fallback path"}]
    tokens = APIBackend.calculate_token_from_messages(backend, messages)

    assert tokens > 0


def test_get_encoder_maps_gpt5_alias_to_o200k_base(monkeypatch) -> None:
    backend = object.__new__(APIBackend)
    backend.chat_model = "gpt-5.2"

    called = {}

    def _fake_get_encoding(name: str):
        called["name"] = name
        return _DummyEncoder()

    monkeypatch.setattr("quantaalpha.llm.client.tiktoken.get_encoding", _fake_get_encoding)

    encoder = APIBackend._get_encoder(backend)

    assert isinstance(encoder, _DummyEncoder)
    assert called["name"] == "o200k_base"


def test_reserve_llm_request_raises_budget_exception(monkeypatch) -> None:
    previous_cap = getattr(llm_client.LLM_SETTINGS, "llm_max_requests_per_run", None)
    previous_requests = llm_client._LLM_BUDGET_STATE["requests"]
    monkeypatch.setattr(llm_client.LLM_SETTINGS, "llm_max_requests_per_run", 1, raising=False)
    llm_client._LLM_BUDGET_STATE["requests"] = 1

    try:
        with pytest.raises(llm_client.LLMBudgetExceededError, match="LLM request budget exceeded"):
            llm_client._reserve_llm_request()
    finally:
        llm_client._LLM_BUDGET_STATE["requests"] = previous_requests
        monkeypatch.setattr(
            llm_client.LLM_SETTINGS,
            "llm_max_requests_per_run",
            previous_cap,
            raising=False,
        )


def test_try_create_chat_completion_stops_retry_on_budget_error(monkeypatch) -> None:
    backend = object.__new__(APIBackend)
    sleep_calls = {"count": 0}

    def _raise_budget(*args, **kwargs):
        raise llm_client.LLMBudgetExceededError("LLM request budget exceeded: 60 >= 60")

    monkeypatch.setattr(APIBackend, "_create_chat_completion_auto_continue", _raise_budget)
    monkeypatch.setattr(
        llm_client.time,
        "sleep",
        lambda *_args, **_kwargs: sleep_calls.__setitem__("count", sleep_calls["count"] + 1),
    )

    with pytest.raises(llm_client.LLMBudgetExceededError):
        APIBackend._try_create_chat_completion_or_embedding(
            backend,
            max_retry=3,
            chat_completion=True,
            messages=[],
        )

    assert sleep_calls["count"] == 0
