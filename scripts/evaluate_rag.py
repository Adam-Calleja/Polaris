"""Thin wrapper for `polaris_rag.cli.evaluate_rag`.

Usage:
    python scripts/evaluate_rag.py -c config/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.cli.evaluate_rag import main


if __name__ == "__main__":
    main()
