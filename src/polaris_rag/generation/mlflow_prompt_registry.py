"""MLflow Prompt Registry integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _import_mlflow() -> Any | None:
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def _filter_supported_kwargs(func: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    allowed = set(sig.parameters.keys())
    return {
        k: v
        for k, v in kwargs.items()
        if k in allowed and v is not None
    }


@dataclass(frozen=True)
class PromptRegistryConfig:
    """Resolved prompt-registry settings from config."""

    enabled: bool = False
    name: str | None = None
    alias: str = "prod"


def resolve_prompt_registry_config(cfg: Any) -> PromptRegistryConfig:
    """Resolve prompt-registry config from GlobalConfig/raw mapping."""

    raw = _as_mapping(getattr(cfg, "raw", cfg))
    mlflow_cfg = _as_mapping(raw.get("mlflow", {}))
    prompt_cfg = _as_mapping(mlflow_cfg.get("prompt_registry", {}))

    name_raw = prompt_cfg.get("name")
    alias_raw = prompt_cfg.get("alias")

    name = str(name_raw).strip() if name_raw else None
    alias = str(alias_raw).strip() if alias_raw else "prod"

    return PromptRegistryConfig(
        enabled=_as_bool(prompt_cfg.get("enabled"), False),
        name=name,
        alias=alias,
    )


def _template_to_text(template: Any) -> str:
    if isinstance(template, str):
        return template

    if isinstance(template, Mapping):
        content = template.get("content")
        return str(content) if content is not None else ""

    if isinstance(template, list):
        parts: list[str] = []
        for item in template:
            if isinstance(item, Mapping):
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role and content:
                    parts.append(f"[{role}] {content}")
                elif content:
                    parts.append(content)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(template)


def load_prompt_template_from_registry(
    *,
    tracking_uri: str | None,
    prompt_name: str,
    alias: str,
    local_prompt_name: str | None = None,
) -> dict[str, Any]:
    """Load prompt template text from MLflow Prompt Registry alias."""

    mlflow = _import_mlflow()
    if mlflow is None:
        raise RuntimeError(
            "Prompt registry is enabled but MLflow is not installed. "
            "Install tracking extras (e.g. '.[tracking]')."
        )

    if tracking_uri:
        set_tracking_uri = getattr(mlflow, "set_tracking_uri", None)
        if set_tracking_uri is not None:
            kwargs = _filter_supported_kwargs(set_tracking_uri, {"uri": tracking_uri})
            set_tracking_uri(**kwargs) if kwargs else set_tracking_uri(tracking_uri)

    genai_api = getattr(mlflow, "genai", None)
    load_prompt = getattr(genai_api, "load_prompt", None)
    if load_prompt is None:
        raise RuntimeError("Installed MLflow build does not expose mlflow.genai.load_prompt().")

    prompt_uri = f"prompts:/{prompt_name}/{alias}"
    prompt_obj = load_prompt(prompt_uri)
    template = getattr(prompt_obj, "template", prompt_obj)
    template_text = _template_to_text(template)

    if not template_text.strip():
        raise ValueError(f"Prompt registry entry '{prompt_uri}' returned an empty template.")

    return {
        "name": local_prompt_name or prompt_name,
        "user": template_text,
    }


def register_prompt_version(
    *,
    tracking_uri: str | None,
    prompt_name: str,
    template_text: str,
    alias: str | None,
    commit_message: str,
    tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Register a prompt version and optionally set/update an alias."""

    if not template_text.strip():
        raise ValueError("Cannot register empty prompt template text.")

    mlflow = _import_mlflow()
    if mlflow is None:
        raise RuntimeError(
            "Prompt registration requires MLflow, but the package is unavailable."
        )

    if tracking_uri:
        set_tracking_uri = getattr(mlflow, "set_tracking_uri", None)
        if set_tracking_uri is not None:
            kwargs = _filter_supported_kwargs(set_tracking_uri, {"uri": tracking_uri})
            set_tracking_uri(**kwargs) if kwargs else set_tracking_uri(tracking_uri)

    genai_api = getattr(mlflow, "genai", None)
    register_prompt = getattr(genai_api, "register_prompt", None)
    if register_prompt is None:
        raise RuntimeError("Installed MLflow build does not expose mlflow.genai.register_prompt().")

    kwargs = _filter_supported_kwargs(
        register_prompt,
        {
            "name": prompt_name,
            "template": template_text,
            "commit_message": commit_message,
            "tags": dict(tags or {}),
        },
    )

    result = register_prompt(**kwargs)
    version = getattr(result, "version", None)

    if alias:
        set_prompt_alias = getattr(genai_api, "set_prompt_alias", None)
        if set_prompt_alias is None:
            raise RuntimeError("Installed MLflow build does not expose mlflow.genai.set_prompt_alias().")

        alias_kwargs = _filter_supported_kwargs(
            set_prompt_alias,
            {
                "name": prompt_name,
                "alias": alias,
                "version": version,
            },
        )
        if alias_kwargs:
            set_prompt_alias(**alias_kwargs)
        else:
            set_prompt_alias(prompt_name, alias, version)

    return {
        "name": prompt_name,
        "alias": alias,
        "version": version,
    }


__all__ = [
    "PromptRegistryConfig",
    "load_prompt_template_from_registry",
    "register_prompt_version",
    "resolve_prompt_registry_config",
]
