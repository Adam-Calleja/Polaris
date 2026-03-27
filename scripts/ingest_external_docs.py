"""Thin wrapper for the packaged external-document ingestion entrypoint.

This wrapper preserves direct script execution by delegating to the corresponding
packaged CLI entrypoint under `polaris_rag.cli`.

Notes
-----
This wrapper keeps local script execution aligned with the installed console-script
entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Find Repo Root.
    
    Parameters
    ----------
    start : Path
        Value for start.
    
    Returns
    -------
    Path
        Result of the operation.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.cli.ingest_external_docs import main


if __name__ == "__main__":
    main()
