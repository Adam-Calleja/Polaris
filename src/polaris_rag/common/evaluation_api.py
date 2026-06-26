"""Shared evaluation-specific API constants.

This module provides package-level helpers for the surrounding Polaris subsystem.

Notes
-----
This module currently exposes its behavior through module-level side effects or imported
symbols.
"""

from __future__ import annotations

POLARIS_EVAL_INCLUDE_METADATA_HEADER = "X-Polaris-Eval-Include-Metadata"

__all__ = ["POLARIS_EVAL_INCLUDE_METADATA_HEADER"]
