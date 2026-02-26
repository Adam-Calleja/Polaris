"""polaris_rag.config.global_config

Global configuration loader and accessors.

This module defines a lightweight wrapper around a raw YAML configuration
dictionary, providing validated, cached access to commonly used configuration
sections across the RAG pipeline.

Environment variables of the form ``${VAR}`` are expanded recursively in all
string values at load time.

Classes
-------
GlobalConfig
    Loader and accessor for global project configuration.
"""

import os
import yaml
from pathlib import Path
from functools import cached_property

def _expand_env(obj):
    """Recursively expand environment variables in a nested structure.

    This function walks nested dictionaries and lists and applies
    :func:`os.path.expandvars` to any string values, expanding patterns of the
    form ``${VAR}`` using the current process environment.

    Parameters
    ----------
    obj : Any
        Object to expand. Supported types are dictionaries, lists, and strings.
        Other types are returned unchanged.

    Returns
    -------
    Any
        A structure of the same shape as ``obj`` with environment variables
        expanded in all string values.
    """
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

class GlobalConfig:
    """Loader and accessor for global project configuration.

    This class wraps a raw configuration dictionary (typically loaded from YAML)
    and exposes validated, cached accessors for commonly used configuration
    sections.

    Parameters
    ----------
    raw : dict
        Raw configuration data as loaded from a YAML file.
    """

    def __init__(
            self,
            raw: dict,
            config_path: Path | None = None,
        ):
        """Initialise a GlobalConfig instance.

        Parameters
        ----------
        raw : dict
            Raw configuration data as loaded from a YAML file.
        """
        self.raw = raw
        # Absolute path to the loaded config file (if known). Used for resolving
        # relative paths (e.g., prompts, persist dirs) in a packaging/Docker-safe way.
        self.config_path = config_path

    @classmethod
    def load(
            cls,
            path: str | Path,
        ) -> "GlobalConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        GlobalConfig
            An instance initialised with the loaded and environment-expanded data.

        Notes
        -----
        All string values in the loaded YAML are processed with recursive
        environment-variable expansion (``${VAR}``) via :func:`os.path.expandvars`.
        """
        cfg_path = Path(path).expanduser().resolve()
        with cfg_path.open("r") as f:
            data = yaml.safe_load(f)
        data = _expand_env(data)
        return cls(data, config_path=cfg_path)

    @cached_property
    def generator_llm(self) -> dict:
        """Return the generator LLM configuration section.

        Returns
        -------
        dict
            The ``generator_llm`` section of the configuration.
        """
        return self.raw["generator_llm"]
    
    @cached_property
    def evaluator_llm(self) -> dict:
        """Return the evaluator LLM configuration section.

        Returns
        -------
        dict
            The ``evaluator_llm`` section of the configuration.
        """
        return self.raw["evaluator_llm"]

    @cached_property
    def embedder(self) -> dict:
        """Return the embedder configuration section.

        Returns
        -------
        dict
            The ``embedder`` section of the configuration.
        """
        return self.raw["embedder"]

    @cached_property
    def vector_stores(self) -> dict[str, dict]:
        """Return multi-store vector configuration.

        Returns
        -------
        dict[str, dict]
            Mapping of source name to vector-store config.

        Raises
        ------
        KeyError
            If ``vector_stores`` is missing from configuration.
        TypeError
            If ``vector_stores`` is not a mapping, contains invalid source names,
            or any source config is not a mapping.
        ValueError
            If ``vector_stores`` is empty.
        """
        stores = self.raw.get("vector_stores")
        if stores is None:
            raise KeyError("Missing 'vector_stores' in configuration.")
        if not isinstance(stores, dict):
            raise TypeError("'vector_stores' must be a mapping of source_name -> config.")
        if not stores:
            raise ValueError("'vector_stores' must define at least one source.")

        normalised: dict[str, dict] = {}
        for source_name, source_cfg in stores.items():
            if not isinstance(source_name, str) or not source_name.strip():
                raise TypeError("Each 'vector_stores' key must be a non-empty string.")
            if not isinstance(source_cfg, dict):
                raise TypeError(
                    f"'vector_stores.{source_name}' must be a mapping, got {type(source_cfg)}."
                )
            normalised[source_name] = source_cfg

        return normalised

    @cached_property
    def vector_store(self) -> dict:
        """Return the primary docs vector-store configuration.

        Returns
        -------
        dict
            The ``vector_stores.docs`` configuration.

        Raises
        ------
        KeyError
            If ``vector_stores.docs`` is not present.
        """
        stores = self.vector_stores
        if "docs" not in stores:
            raise KeyError(
                "Missing 'vector_stores.docs' in configuration. "
                "Configure a 'docs' source or migrate callers to use 'vector_stores' directly."
            )
        return stores["docs"]

    @cached_property
    def doc_store(self) -> dict:
        """Return the document store configuration section.

        Returns
        -------
        dict
            The ``doc_store`` section of the configuration, or an empty dict if
            not present.
        """
        return self.raw.get('doc_store', {})

    @cached_property
    def storage_context(self) -> dict:
        """Return the storage context configuration section.

        Returns
        -------
        dict
            The ``storage_context`` section of the configuration, or an empty dict
            if not present.
        """
        return self.raw.get('storage_context', {})

    @cached_property
    def retriever(self) -> dict:
        """Return the retriever configuration section.

        Returns
        -------
        dict
            The ``retriever`` section of the configuration, or an empty dict if
            not present.
        """
        return self.raw.get('retriever', {})

    @cached_property
    def prompts(self):
        """Return the prompts configuration entry.

        Returns
        -------
        str or list[str] or None
            The ``prompts`` entry, which may be a single path, a list of paths,
            or ``None`` if not configured.

        Notes
        -----
        This accessor returns the raw configured value without validation. Callers
        are responsible for handling ``None`` and normalising single vs multiple
        prompt paths.
        """
        return self.raw.get("prompts")
    
    @cached_property
    def prompt_name(self) -> str:
        """Return the configured prompt name.

        Returns
        -------
        str
            The ``prompt_name`` value from the configuration.

        Raises
        ------
        KeyError
            If ``prompt_name`` is missing from the configuration.
        """
        prompt_name = self.raw.get("prompt_name")
        if prompt_name is None:
            raise KeyError("Missing 'prompt_name' in configuration.")
        return prompt_name

    @cached_property 
    def document_preprocess_html_conditions(self) -> list:
        """Return HTML document preprocessing conditions.

        Returns
        -------
        list[dict]
            List of condition dictionaries with keys ``tag`` and ``condition``.

        Raises
        ------
        KeyError
            If the ``document_loader`` section or its ``conditions`` key is missing.
        TypeError
            If ``conditions`` is not a list of dictionaries with the required keys.
        """

        section = self.raw.get("document_loader")
        if section is None:
            raise KeyError("Missing 'document_loader' section in configuration.")
        
        conditions = section.get('conditions')
        if conditions is None:
            raise KeyError("Missing 'conditions' under 'document_loader' in configuration.")
        
        if not isinstance(conditions, list):
            raise TypeError("'document_loader.conditions' must be a list of dictionaries.")

        for i, item in enumerate(conditions):
            if not isinstance(item, dict):
                raise TypeError(f"Condition #{i} must be a dict with keys 'tag' and 'condition'.")
            if "tag" not in item or "condition" not in item:
                raise TypeError(f"Condition #{i} must contain 'tag' and 'condition' keys.")
            if not isinstance(item["tag"], str) or not isinstance(item["condition"], str):
                raise TypeError(f"Condition #{i} 'tag' and 'condition' must be strings.")
            
        return conditions
    
    @cached_property
    def document_preprocess_html_tags(self) -> list:
        """Return HTML tags to remove during preprocessing.

        Returns
        -------
        list[str]
            Tags to remove entirely from HTML documents.

        Raises
        ------
        KeyError
            If the ``document_loader`` section or its ``tags`` key is missing.
        TypeError
            If ``tags`` is not a list of strings.
        """

        section = self.raw.get("document_loader")
        if section is None:
            raise KeyError("Missing 'document_loader' section in configuration.")
        
        tags = section.get('tags')
        if tags is None:
            raise KeyError("Missing 'tags' under 'document_loader' in configuration.")
        
        if not isinstance(tags, list):
            raise TypeError("'document_loader.tags' must be a list of strings.")

        for i, item in enumerate(tags):
            if not isinstance(item, str):
                raise TypeError(f"Condition #{i} must be a string.")
            
        return tags
    
    @cached_property
    def document_preprocess_html_link_classes(self) -> list:
        """Return CSS classes identifying external HTML links.

        Returns
        -------
        list[str]
            CSS class names identifying external links.

        Raises
        ------
        KeyError
            If the ``document_loader`` section or its ``link_classes`` key is missing.
        TypeError
            If ``link_classes`` is not a list of strings.
        """

        section = self.raw.get("document_loader")
        if section is None:
            raise KeyError("Missing 'document_loader' section in configuration.")
        
        link_classes = section.get('link_classes')
        if link_classes is None:
            raise KeyError("Missing 'link_classes' under 'document_loader' in configuration.")
        
        if not isinstance(link_classes, list):
            raise TypeError("'document_loader.link_classes' must be a list of strings.")

        for i, item in enumerate(link_classes):
            if not isinstance(item, str):
                raise TypeError(f"Condition #{i} must be a string.")
            
        return link_classes
    
    @cached_property
    def jira_api_credentials(self) -> dict[str, str]:
        """Return Jira API credentials from configuration.

        Returns
        -------
        dict[str, str]
            Mapping containing ``username`` and ``password`` (or API token).

        Raises
        ------
        KeyError
            If the ``jira_api_credentials`` section or required keys are missing.
        """
        section = self.raw.get("jira_api_credentials")

        if section is None:
            raise KeyError("Missing 'jira_api_credentials' section in configuration.")
        
        username = section.get('username')
        password = section.get('password')

        if username is None:
            raise KeyError("Missing 'username' section in configuration.")
        
        if password is None:
            raise KeyError("Missing 'password' section in configuration.")
        
        return {
            "username": username,
            "password": password, 
        }
