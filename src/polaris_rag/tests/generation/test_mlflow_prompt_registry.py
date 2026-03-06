from __future__ import annotations

from dataclasses import dataclass

import pytest

from polaris_rag.generation import mlflow_prompt_registry


@dataclass
class _FakePrompt:
    template: str


class _FakeGenAI:
    def __init__(self) -> None:
        self.load_prompt_calls: list[str] = []
        self.register_prompt_calls: list[dict[str, object]] = []
        self.set_alias_calls: list[dict[str, object]] = []

    def load_prompt(self, name_or_uri: str):
        self.load_prompt_calls.append(name_or_uri)
        return _FakePrompt(template="Answer:\n{{ question }}")

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.register_prompt_calls.append(
            {
                "name": name,
                "template": template,
                "commit_message": commit_message,
                "tags": dict(tags or {}),
            }
        )
        return type("_Version", (), {"version": "7"})

    def set_prompt_alias(self, name: str, alias: str, version: str):
        self.set_alias_calls.append({"name": name, "alias": alias, "version": version})


class _FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri_calls: list[str] = []
        self.genai = _FakeGenAI()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri_calls.append(uri)


def test_resolve_prompt_registry_config_defaults() -> None:
    cfg = {"foo": "bar"}
    resolved = mlflow_prompt_registry.resolve_prompt_registry_config(cfg)

    assert resolved.enabled is False
    assert resolved.name is None
    assert resolved.alias == "prod"


def test_load_prompt_template_from_registry_uses_prompt_alias_uri(monkeypatch) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_prompt_registry, "_import_mlflow", lambda: fake_mlflow)

    loaded = mlflow_prompt_registry.load_prompt_template_from_registry(
        tracking_uri="http://mlflow:5000",
        prompt_name="hpc_prompt",
        alias="prod",
        local_prompt_name="hpc_runtime_prompt",
    )

    assert fake_mlflow.tracking_uri_calls == ["http://mlflow:5000"]
    assert fake_mlflow.genai.load_prompt_calls == ["prompts:/hpc_prompt@prod"]
    assert loaded["name"] == "hpc_runtime_prompt"
    assert "{{ question }}" in loaded["user"]


def test_load_prompt_template_from_registry_falls_back_to_legacy_alias_uri(monkeypatch) -> None:
    class _FallbackGenAI(_FakeGenAI):
        def load_prompt(self, name_or_uri: str):
            self.load_prompt_calls.append(name_or_uri)
            if name_or_uri.endswith("@prod"):
                raise ValueError("Unsupported alias URI format")
            return _FakePrompt(template="Answer:\n{{ question }}")

    fake_mlflow = _FakeMLflow()
    fake_mlflow.genai = _FallbackGenAI()
    monkeypatch.setattr(mlflow_prompt_registry, "_import_mlflow", lambda: fake_mlflow)

    loaded = mlflow_prompt_registry.load_prompt_template_from_registry(
        tracking_uri="http://mlflow:5000",
        prompt_name="hpc_prompt",
        alias="prod",
        local_prompt_name="hpc_runtime_prompt",
    )

    assert fake_mlflow.genai.load_prompt_calls == [
        "prompts:/hpc_prompt@prod",
        "prompts:/hpc_prompt/prod",
    ]
    assert loaded["name"] == "hpc_runtime_prompt"
    assert "{{ question }}" in loaded["user"]


def test_load_prompt_template_from_registry_rejects_empty_prompt(monkeypatch) -> None:
    class _EmptyGenAI(_FakeGenAI):
        def load_prompt(self, name_or_uri: str):
            self.load_prompt_calls.append(name_or_uri)
            return _FakePrompt(template="")

    fake_mlflow = _FakeMLflow()
    fake_mlflow.genai = _EmptyGenAI()
    monkeypatch.setattr(mlflow_prompt_registry, "_import_mlflow", lambda: fake_mlflow)

    with pytest.raises(ValueError, match="empty template"):
        mlflow_prompt_registry.load_prompt_template_from_registry(
            tracking_uri="http://mlflow:5000",
            prompt_name="hpc_prompt",
            alias="prod",
        )


def test_register_prompt_version_sets_alias(monkeypatch) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_prompt_registry, "_import_mlflow", lambda: fake_mlflow)

    result = mlflow_prompt_registry.register_prompt_version(
        tracking_uri="http://mlflow:5000",
        prompt_name="hpc_prompt",
        template_text="Answer: {{ question }}",
        alias="prod",
        commit_message="test registration",
        tags={"source": "unit-test"},
    )

    assert result["name"] == "hpc_prompt"
    assert result["alias"] == "prod"
    assert result["version"] == "7"
    assert fake_mlflow.genai.register_prompt_calls
    assert fake_mlflow.genai.set_alias_calls == [
        {"name": "hpc_prompt", "alias": "prod", "version": "7"}
    ]


def test_load_prompt_template_from_registry_fails_fast_on_missing_alias(monkeypatch) -> None:
    class _MissingAliasGenAI(_FakeGenAI):
        def load_prompt(self, name_or_uri: str):
            self.load_prompt_calls.append(name_or_uri)
            raise RuntimeError("Prompt alias not found")

    fake_mlflow = _FakeMLflow()
    fake_mlflow.genai = _MissingAliasGenAI()
    monkeypatch.setattr(mlflow_prompt_registry, "_import_mlflow", lambda: fake_mlflow)

    with pytest.raises(RuntimeError, match="Failed to load prompt"):
        mlflow_prompt_registry.load_prompt_template_from_registry(
            tracking_uri="http://mlflow:5000",
            prompt_name="hpc_prompt",
            alias="prod",
        )
