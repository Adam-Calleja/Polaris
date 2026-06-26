from __future__ import annotations

from types import SimpleNamespace

import pytest

from polaris_rag.generation import llm_interface


def test_openai_chat_from_config_dict_passes_transport_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(
            self,
            *,
            model: str,
            base_url: str,
            api_key: str | None = None,
            request_timeout: float | None = None,
            max_retries: int | None = None,
            temperature: float | None = None,
            top_p: float | None = None,
        ) -> None:
            captured.update(
                {
                    "model": model,
                    "base_url": base_url,
                    "api_key": api_key,
                    "request_timeout": request_timeout,
                    "max_retries": max_retries,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

    monkeypatch.setattr(llm_interface, "ChatOpenAI", _FakeChatOpenAI)

    llm = llm_interface.OpenAIChatLikeLLM.from_config_dict(
        {
            "type": "openai_chat",
            "model_name": "gemini-2.5-flash",
            "api_base": "https://example.invalid/v1",
            "api_key": "secret",
            "request_timeout": 90,
            "max_retries": 1,
            "model_kwargs": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
    )

    assert captured == {
        "model": "gemini-2.5-flash",
        "base_url": "https://example.invalid/v1",
        "api_key": "secret",
        "request_timeout": 90.0,
        "max_retries": 1,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    assert llm.get_llm() is not None


def test_openai_like_from_config_dict_passes_transport_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(
            self,
            *,
            model_name: str,
            openai_api_base: str,
            openai_api_key: str | None = None,
            timeout: float | None = None,
            max_retries: int | None = None,
            top_p: float | None = None,
            temperature: float | None = None,
        ) -> None:
            captured.update(
                {
                    "model_name": model_name,
                    "openai_api_base": openai_api_base,
                    "openai_api_key": openai_api_key,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "top_p": top_p,
                    "temperature": temperature,
                }
            )

    monkeypatch.setattr(llm_interface, "OpenAI", _FakeOpenAI)

    llm = llm_interface.OpenAILikeLLM.from_config_dict(
        {
            "type": "openai_like",
            "model_name": "qwen",
            "api_base": "http://localhost:8080/v1",
            "api_key": "secret",
            "timeout": 45,
            "max_retries": 2,
            "model_kwargs": {
                "temperature": 0.0,
            },
        }
    )

    assert captured == {
        "model_name": "qwen",
        "openai_api_base": "http://localhost:8080/v1",
        "openai_api_key": "secret",
        "timeout": 45.0,
        "max_retries": 2,
        "top_p": 1,
        "temperature": 0.0,
    }
    assert llm.get_llm() is not None


def test_openai_chat_generate_enforces_per_request_timeout(monkeypatch) -> None:
    class _FakeChatOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(llm_interface, "ChatOpenAI", _FakeChatOpenAI)

    llm = llm_interface.OpenAIChatLikeLLM.from_config_dict(
        {
            "type": "openai_chat",
            "model_name": "timeout-model",
            "api_base": "http://example.invalid/v1",
            "api_key": "secret",
            "request_timeout": 0.05,
            "model_kwargs": {"temperature": 0.0},
        }
    )

    class _FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    raise TimeoutError("timed out")

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(llm, "_build_openai_client", lambda **kwargs: _FakeClient())

    with pytest.raises(llm_interface.GenerationTimeoutError):
        llm.generate("hello", timeout_seconds=0.05)


def test_openai_chat_generate_passes_structured_messages_without_hidden_stop(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(llm_interface, "ChatOpenAI", _FakeChatOpenAI)

    llm = llm_interface.OpenAIChatLikeLLM.from_config_dict(
        {
            "type": "openai_chat",
            "model_name": "chat-model",
            "api_base": "http://example.invalid/v1",
            "api_key": "secret",
            "model_kwargs": {"temperature": 0.0},
        }
    )

    class _FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    captured["request"] = kwargs
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="ok"),
                            )
                        ]
                    )

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(llm, "_build_openai_client", lambda **kwargs: _FakeClient())

    messages = [
        {"role": "system", "content": "Stay grounded."},
        {"role": "user", "content": "Hello"},
    ]

    assert llm.generate(messages) == "ok"

    assert captured["request"] == {
        "model": "chat-model",
        "messages": messages,
        "timeout": None,
        "temperature": 0.0,
    }


def test_openai_like_generate_does_not_inject_default_stop(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(llm_interface, "OpenAI", _FakeOpenAI)

    llm = llm_interface.OpenAILikeLLM.from_config_dict(
        {
            "type": "openai_like",
            "model_name": "completion-model",
            "api_base": "http://example.invalid/v1",
            "api_key": "secret",
            "model_kwargs": {"temperature": 0.0},
        }
    )

    class _FakeClient:
        class _Completions:
            @staticmethod
            def create(**kwargs):  # noqa: ANN003
                captured["request"] = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(text="ok"),
                    ]
                )

        completions = _Completions()

    monkeypatch.setattr(llm, "_build_openai_client", lambda **kwargs: _FakeClient())

    assert llm.generate("Hello") == "ok"

    assert captured["request"] == {
        "model": "completion-model",
        "prompt": "Hello",
        "timeout": None,
        "temperature": 0.0,
    }
