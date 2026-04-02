"""Run one experiment stage from a manifest."""

from __future__ import annotations

import argparse

from polaris_rag.evaluation.experiment_automation import (
    BOTH_PHASE,
    SUPPORTED_EXECUTION_PHASES,
    run_experiment_stage,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stage execution."""

    parser = argparse.ArgumentParser(description="Run one experiment stage from a manifest.")
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
        action="append",
        default=[],
        help="Optional condition name to run. Repeat to run multiple conditions.",
    )
    parser.add_argument(
        "--phase",
        choices=sorted(SUPPORTED_EXECUTION_PHASES),
        default=BOTH_PHASE,
        help="Execution phase for evaluation-grid stages: prepare, evaluate, or both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render configs and write the execution record without running subprocesses.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entrypoint."""

    args = parse_args()
    record = run_experiment_stage(
        manifest_path=args.manifest,
        stage_name=args.stage,
        selected_conditions=list(args.condition) or None,
        execution_phase=args.phase,
        dry_run=bool(args.dry_run),
    )
    print(f"Stage complete: {record['stage_name']}")
    print(f"Stage type: {record['stage_type']}")
    if record.get("execution_phase"):
        print(f"Execution phase: {record['execution_phase']}")
    if record.get("dry_run"):
        print("Execution mode: dry-run")


if __name__ == "__main__":
    main()
