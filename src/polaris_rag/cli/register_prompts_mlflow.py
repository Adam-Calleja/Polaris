"""CLI entrypoint to register Polaris prompts in MLflow Prompt Registry.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
parse_args
    Parse args.
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polaris_rag.config import GlobalConfig
from polaris_rag.generation.mlflow_prompt_registry import (
    register_prompt_version,
    resolve_prompt_registry_config,
)
from polaris_rag.generation.prompt_builder import PromptBuilder, PromptTemplate
from polaris_rag.observability.mlflow_tracking import load_mlflow_runtime_config


def _load_prompt_builder(cfg: GlobalConfig) -> PromptBuilder:
    """Load prompt Builder.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    
    Returns
    -------
    PromptBuilder
        Result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    TypeError
        If the provided value has an unexpected type.
    """
    prompts = cfg.prompts
    if prompts is None:
        raise ValueError("No prompt sources configured under 'prompts' in config file.")

    builder = PromptBuilder()

    cfg_path = getattr(cfg, "config_path", None)
    base_dir = Path(cfg_path).expanduser().resolve().parent if cfg_path else None

    if isinstance(prompts, str):
        sources = [prompts]
    elif isinstance(prompts, (list, tuple)):
        sources = [str(item) for item in prompts]
    else:
        raise TypeError(f"config.prompts must be a str or list[str], got {type(prompts)!r}")

    for source in sources:
        builder.register_from_source(source, base_dir=base_dir)

    return builder


def _template_to_text(template: PromptTemplate) -> str:
    """Template To Text.
    
    Parameters
    ----------
    template : PromptTemplate
        Value for template.
    
    Returns
    -------
    str
        Resulting string value.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    parts: list[str] = []

    if template.system:
        parts.append(str(template.system))

    for item in template.few_shot:
        content = ""
        if isinstance(item, dict):
            content = str(item.get("content", "") or "")
        else:
            content = str(item or "")
        if content:
            parts.append(content)

    if template.user:
        parts.append(str(template.user))

    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(f"Prompt template '{template.name}' produced an empty body.")
    return text


def _parse_tag_overrides(values: list[str]) -> dict[str, str]:
    """Parse tag Overrides.
    
    Parameters
    ----------
    values : list[str]
        Value for values.
    
    Returns
    -------
    dict[str, str]
        Structured result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    tags: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid tag '{item}'. Expected format key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid tag '{item}'. Key cannot be empty.")
        tags[key] = value
    return tags


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="Register Polaris prompt templates in MLflow Prompt Registry")

    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        help="Path to Polaris config YAML",
    )
    parser.add_argument(
        "--prompt-name",
        default=None,
        help="Local prompt template name to register (defaults to config.prompt_name)",
    )
    parser.add_argument(
        "--registry-name",
        default=None,
        help="MLflow prompt registry name (defaults to mlflow.prompt_registry.name or prompt name)",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Alias to set after registering (defaults to mlflow.prompt_registry.alias or 'prod')",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLflow tracking URI",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Prompt version commit message",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Extra prompt tag in key=value format (repeatable)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line entrypoint.

    Notes
    -----
    Parses CLI arguments, registers the selected prompt templates with MLflow,
    and prints the resulting registry metadata to standard output.
    """
    args = parse_args()

    cfg = GlobalConfig.load(args.config_file)
    builder = _load_prompt_builder(cfg)

    runtime_cfg = load_mlflow_runtime_config(cfg)
    registry_cfg = resolve_prompt_registry_config(cfg)

    source_prompt_name = str(args.prompt_name or cfg.prompt_name)
    local_template = builder.get_template(source_prompt_name)

    registry_name = str(args.registry_name or registry_cfg.name or source_prompt_name)
    alias = str(args.alias or registry_cfg.alias or "prod")

    tracking_uri = str(args.tracking_uri or runtime_cfg.tracking_uri or "").strip() or None

    commit_message = args.commit_message or (
        f"Register '{source_prompt_name}' from Polaris config at "
        f"{datetime.now(tz=timezone.utc).isoformat()}"
    )

    tags = {
        "source_prompt_name": source_prompt_name,
        "source_config": str(Path(args.config_file).expanduser().resolve()),
        "registered_by": "polaris-register-prompts-mlflow",
    }
    tags.update(_parse_tag_overrides(list(args.tag or [])))

    template_text = _template_to_text(local_template)

    result: dict[str, Any] = register_prompt_version(
        tracking_uri=tracking_uri,
        prompt_name=registry_name,
        template_text=template_text,
        alias=alias,
        commit_message=commit_message,
        tags=tags,
    )

    print("Prompt registration complete.")
    print(f"Registry name: {result.get('name')}")
    print(f"Alias: {result.get('alias')}")
    print(f"Version: {result.get('version')}")


if __name__ == "__main__":
    main()
