"""Render one generated experiment config overlay from a manifest."""

from __future__ import annotations

import argparse

from polaris_rag.evaluation.experiment_automation import render_stage_condition_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config rendering."""

    parser = argparse.ArgumentParser(
        description="Render one generated experiment config overlay from a manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the experiment manifest YAML.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Stage name defined in the experiment manifest.",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Optional condition name for evaluation-grid stages.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path for the rendered config overlay.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entrypoint."""

    args = parse_args()
    output_path = render_stage_condition_config(
        manifest_path=args.manifest,
        stage_name=args.stage,
        condition_name=args.condition,
        output_path=args.output_path,
    )
    print(f"Wrote generated config: {output_path}")


if __name__ == "__main__":
    main()
