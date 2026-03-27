"""Thin wrapper for the packaged Jira ingestion entrypoint.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.cli.ingest_jira_tickets import main


if __name__ == "__main__":
    main()
