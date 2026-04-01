"""Summarize completed experiment runs for one manifest stage."""

from __future__ import annotations

import argparse

from polaris_rag.evaluation.experiment_automation import summarize_experiment_stage


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stage summarization."""

    parser = argparse.ArgumentParser(
        description="Summarize completed experiment runs for one manifest stage."
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
        "--output-dir",
        default=None,
        help="Optional destination directory for summary artifacts.",
    )
    parser.add_argument(
        "--run-comparison",
        action="store_true",
        help="Also generate analyze-eval-runs comparison artifacts for one repeat per condition.",
    )
    parser.add_argument(
        "--comparison-repeat",
        default="latest",
        help="Repeat index to use for run comparison, or 'latest' (default).",
    )
    parser.add_argument(
        "--manual-eval-seed",
        type=int,
        default=42,
        help="Seed forwarded to the run-comparison manual-eval sheet generator.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entrypoint."""

    args = parse_args()
    artifacts = summarize_experiment_stage(
        manifest_path=args.manifest,
        stage_name=args.stage,
        output_dir=args.output_dir,
        run_comparison=bool(args.run_comparison),
        comparison_repeat=str(args.comparison_repeat),
        manual_eval_seed=int(args.manual_eval_seed),
    )
    print(f"Stage summary complete for {args.stage}.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
