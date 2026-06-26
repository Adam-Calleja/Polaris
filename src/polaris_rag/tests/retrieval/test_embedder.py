from __future__ import annotations

import sys
import types

import pytest


class _FakeLlamaIndexEmbedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


sys.modules.setdefault(
    "langchain_core.callbacks",
    types.SimpleNamespace(BaseCallbackHandler=object),
)
sys.modules.setdefault(
    "llama_index.embeddings.openai_like",
    types.SimpleNamespace(OpenAILikeEmbedding=_FakeLlamaIndexEmbedder),
)

from polaris_rag.retrieval import embedder as embedder_module


def test_openai_like_embedder_fails_over_to_backup_api_base(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    sleep_calls: list[float] = []
    responses = {
        "http://embed-primary:80/v1": [
            RuntimeError("primary down"),
            RuntimeError("primary still down"),
        ],
        "http://embed-backup:80/v1": [
            types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(index=0, embedding=[1.0, 2.0]),
                    types.SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                ]
            ),
        ],
    }

    class _FakeClient:
        def __init__(self, *, api_key, base_url, timeout, max_retries) -> None:
            self.embeddings = types.SimpleNamespace(
                create=lambda *, model, input, timeout, **kwargs: _create(base_url, input)
            )

    def _create(base_url: str, input_value):
        texts = list(input_value)
        calls.append((base_url, texts))
        response = responses[base_url].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(embedder_module, "OpenAIClient", _FakeClient)
    monkeypatch.setattr(
        embedder_module,
        "_is_retryable_openai_embedding_error",
        lambda exc: isinstance(exc, RuntimeError),
    )
    monkeypatch.setattr(embedder_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    embedder = embedder_module.OpenAILikeEmbedder(
        model_name="fake-model",
        api_base="http://embed-primary:80/v1",
        failover_api_bases=["http://embed-backup:80/v1"],
        request_max_attempts=2,
        request_base_backoff_seconds=0.5,
        embed_batch_size=8,
    )

    vectors = embedder.embed_documents(["alpha", "beta"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [
        ("http://embed-primary:80/v1", ["alpha", "beta"]),
        ("http://embed-primary:80/v1", ["alpha", "beta"]),
        ("http://embed-backup:80/v1", ["alpha", "beta"]),
    ]
    assert sleep_calls == [0.5]
    assert embedder.api_base == "http://embed-backup:80/v1"
    assert embedder.api_bases == (
        "http://embed-primary:80/v1",
        "http://embed-backup:80/v1",
    )


def test_openai_like_embedder_raises_non_retryable_errors_without_failover(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeClient:
        def __init__(self, *, api_key, base_url, timeout, max_retries) -> None:
            self.embeddings = types.SimpleNamespace(
                create=lambda *, model, input, timeout, **kwargs: _create(base_url)
            )

    def _create(base_url: str):
        calls.append(base_url)
        raise ValueError("invalid request")

    monkeypatch.setattr(embedder_module, "OpenAIClient", _FakeClient)

    embedder = embedder_module.OpenAILikeEmbedder(
        model_name="fake-model",
        api_base="http://embed-primary:80/v1",
        failover_api_bases=["http://embed-backup:80/v1"],
        request_max_attempts=2,
    )

    with pytest.raises(ValueError, match="invalid request"):
        embedder.embed_query("hello")

    assert calls == ["http://embed-primary:80/v1"]
    assert embedder.api_base == "http://embed-primary:80/v1"
