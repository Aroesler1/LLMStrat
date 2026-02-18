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
