"""MLflow Prompt Registry integration utilities.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
PromptRegistryConfig
    Resolved prompt-registry settings from config.

Functions
---------
resolve_prompt_registry_config
    Resolve prompt-registry config from GlobalConfig/raw mapping.
load_prompt_template_from_registry
    Load prompt template text from MLflow Prompt Registry alias.
register_prompt_version
    Register a prompt version and optionally set/update an alias.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    obj : Any
        Value for obj.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    """As Bool.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    default : bool
        Fallback value to use when normalization fails.
    
    Returns
    -------
    bool
        `True` if as Bool; otherwise `False`.
    """
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
    """Import MLflow.
    
    Returns
    -------
    Any or None
        Result of the operation.
    """
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def _filter_supported_kwargs(func: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Filter Supported Kwargs.
    
    Parameters
    ----------
    func : Any
        Value for func.
    kwargs : Mapping[str, Any]
        Value for kwargs.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    """
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
    """Resolved prompt-registry settings from config.
    
    Attributes
    ----------
    enabled : bool
        Value for enabled.
    name : str or None
        Human-readable name for the resource or tracing span.
    alias : str
        Value for alias.
    """

    enabled: bool = False
    name: str | None = None
    alias: str = "prod"


def resolve_prompt_registry_config(cfg: Any) -> PromptRegistryConfig:
    """Resolve prompt-registry config from GlobalConfig/raw mapping.
    
    Parameters
    ----------
    cfg : Any
        Configuration object or mapping used to resolve runtime settings.
    
    Returns
    -------
    PromptRegistryConfig
        Resolved prompt Registry Config.
    """

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
    """Template To Text.
    
    Parameters
    ----------
    template : Any
        Value for template.
    
    Returns
    -------
    str
        Resulting string value.
    """
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


def _load_prompt_by_alias(
    *,
    load_prompt: Any,
    prompt_name: str,
    alias: str,
) -> tuple[Any, str]:
    """Load a prompt by alias with URI-format fallbacks.

    Prefer the MLflow alias form ``prompts:/name@alias`` and fall back to the
    legacy ``prompts:/name/alias`` format for compatibility.
    """
    alias_clean = str(alias).strip()
    prompt_uris = [
        f"prompts:/{prompt_name}@{alias_clean}",
        f"prompts:/{prompt_name}/{alias_clean}",
    ]

    last_error: Exception | None = None
    for prompt_uri in prompt_uris:
        try:
            return load_prompt(prompt_uri), prompt_uri
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to load prompt '{prompt_name}' with alias '{alias_clean}'. "
        f"Tried URIs: {prompt_uris}"
    ) from last_error


def load_prompt_template_from_registry(
    *,
    tracking_uri: str | None,
    prompt_name: str,
    alias: str,
    local_prompt_name: str | None = None,
) -> dict[str, Any]:
    """Load prompt template text from MLflow Prompt Registry alias.
    
    Parameters
    ----------
    tracking_uri : str or None, optional
        Value for tracking Uri.
    prompt_name : str
        Value for prompt Name.
    alias : str
        Value for alias.
    local_prompt_name : str or None, optional
        Value for local Prompt Name.
    
    Returns
    -------
    dict[str, Any]
        Loaded prompt Template From Registry.
    
    Raises
    ------
    RuntimeError
        If `RuntimeError` is raised while executing the operation.
    ValueError
        If the provided value is invalid for the operation.
    """

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

    prompt_obj, resolved_prompt_uri = _load_prompt_by_alias(
        load_prompt=load_prompt,
        prompt_name=prompt_name,
        alias=alias,
    )
    template = getattr(prompt_obj, "template", prompt_obj)
    template_text = _template_to_text(template)

    if not template_text.strip():
        raise ValueError(f"Prompt registry entry '{resolved_prompt_uri}' returned an empty template.")

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
    """Register a prompt version and optionally set/update an alias.
    
    Parameters
    ----------
    tracking_uri : str or None, optional
        Value for tracking Uri.
    prompt_name : str
        Value for prompt Name.
    template_text : str
        Value for template Text.
    alias : str or None, optional
        Value for alias.
    commit_message : str
        Value for commit Message.
    tags : Mapping[str, str] or None, optional
        Value for tags.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    RuntimeError
        If `RuntimeError` is raised while executing the operation.
    """

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
