"""Thin wrapper for `polaris_rag.cli.tune_validity_reranker`.

Usage: python scripts/tune_validity_reranker.py -c config/config.yaml --dataset-path
data/test/dev.jsonl

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

from polaris_rag.cli.tune_validity_reranker import main


if __name__ == "__main__":
    main()
