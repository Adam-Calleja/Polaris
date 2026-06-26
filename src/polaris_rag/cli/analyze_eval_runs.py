"""CLI entrypoint for Stage 5 run-comparison artifacts.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
parse_args
    Parse args.
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from polaris_rag.evaluation.run_analysis import load_run_input, write_run_comparison_outputs


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Compare saved evaluation runs and emit dissertation-ready analysis artifacts."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run input in the form <condition>=<run_dir>. Repeat for multiple conditions.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for comparison outputs.",
    )
    parser.add_argument(
        "--manual-eval-seed",
        type=int,
        default=42,
        help="Seed controlling per-query label randomization in the manual-eval sheet.",
    )
    return parser.parse_args()


def _parse_run_specs(values: list[str]) -> list[tuple[str, Path]]:
    """Parse run Specs.
    
    Parameters
    ----------
    values : list[str]
        Value for values.
    
    Returns
    -------
    list[tuple[str, Path]]
        Collected results from the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    runs: list[tuple[str, Path]] = []
    seen_conditions: set[str] = set()
    for raw in values:
        condition_name, separator, run_dir = raw.partition("=")
        condition_name = condition_name.strip()
        run_dir = run_dir.strip()
        if not separator or not condition_name or not run_dir:
            raise ValueError(
                "Each --run value must be formatted as <condition>=<run_dir>."
            )
        if condition_name in seen_conditions:
            raise ValueError(f"Duplicate condition name {condition_name!r} passed to --run.")
        seen_conditions.add(condition_name)
        runs.append((condition_name, Path(run_dir).expanduser().resolve()))
    return runs


def main() -> None:
    """Run the command-line entrypoint.

    Notes
    -----
    Parses CLI arguments, loads the requested evaluation runs, and writes the
    generated comparison artifact paths to standard output.
    """
    args = parse_args()
    run_specs = _parse_run_specs(list(args.run))
    runs = [load_run_input(condition_name, run_dir) for condition_name, run_dir in run_specs]
    artifacts = write_run_comparison_outputs(
        runs=runs,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        manual_eval_seed=int(args.manual_eval_seed),
    )
    print(f"Run analysis complete for {len(runs)} conditions.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
