from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

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

    class _SlowHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            time.sleep(0.2)
            payload = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "timeout-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "late"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setattr(llm_interface, "ChatOpenAI", _FakeChatOpenAI)

    llm = llm_interface.OpenAIChatLikeLLM.from_config_dict(
        {
            "type": "openai_chat",
            "model_name": "timeout-model",
            "api_base": f"http://127.0.0.1:{server.server_port}/v1",
            "api_key": "secret",
            "request_timeout": 0.05,
            "model_kwargs": {"temperature": 0.0},
        }
    )

    try:
        with pytest.raises(llm_interface.GenerationTimeoutError):
            llm.generate("hello", timeout_seconds=0.05)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
