"""Thin wrapper for `polaris_rag.cli.render_experiment_config`.

Usage: python scripts/experiments/render_config.py --manifest experiments/protocol.template.yaml --stage stage4_source_ablation --condition docs_only
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.cli.render_experiment_config import main


if __name__ == "__main__":
    main()
