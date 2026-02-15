"""polaris_rag.config

Configuration subsystem for the Polaris RAG pipeline.

This package provides structured access to global and component-level
configuration loaded from YAML files. It exposes validated, documented
interfaces rather than raw configuration dictionaries.

Modules
-------
global_config
    Global configuration loader and cached accessors.
"""
from .global_config import GlobalConfig

__all__ = ["GlobalConfig"]