"""Tune validity-aware reranker weights on a development split.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
TrialResult
    Structured result for trial.

Functions
---------
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import yaml

from polaris_rag.app.container import build_container
from polaris_rag.config import GlobalConfig
from polaris_rag.evaluation.evaluation_dataset import (
    PrepProgressEvent,
    PrepRetryPolicy,
    build_prepared_rows,
    load_raw_examples,
    preprocess_rows_for_evaluation,
    to_evaluation_dataset,
)

logger = logging.getLogger(__name__)


OBJECTIVE_METRICS: tuple[str, ...] = (
    "factual_correctness",
    "faithfulness",
    "context_precision_without_reference",
)
DEFAULT_WEIGHT_GRID: dict[str, tuple[float, ...]] = {
    "authority": (0.00, 0.04, 0.08),
    "scope": (0.00, 0.04, 0.08),
    "software": (0.00,),
    "scope_family": (0.00,),
    "version": (0.00,),
    "status": (0.00, 0.04, 0.08),
    "freshness": (0.00, 0.01),
}
DEFAULT_AUTHORITY_VALUES: dict[str, float] = {
    "local_official": 1.0,
    "external_official": 0.5,
    "ticket_memory": 0.0,
    "unknown": 0.0,
}
DEFAULT_STATUS_VALUES: dict[str, float] = {
    "current": 1.0,
    "maintenance": 0.25,
    "legacy": -0.5,
    "eol": -1.0,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class TrialResult:
    """Structured result for trial.
    
    Attributes
    ----------
    weights : dict[str, float]
        Weight values to evaluate or persist.
    objective : float
        Value for objective.
    metric_means : dict[str, float]
        Value for metric Means.
    reranker_fingerprint : str or None
        Value for reranker Fingerprint.
    prepared_rows : int
        Value for prepared Rows.
    """
    weights: dict[str, float]
    objective: float
    metric_means: dict[str, float]
    reranker_fingerprint: str | None
    prepared_rows: int

    def tie_break_key(self) -> tuple[float, float, float, tuple[tuple[str, float], ...]]:
        """Return the deterministic tie-break key for a trial result.
        
        Returns
        -------
        tuple[float, float, float, tuple[tuple[str, float], ...]]
            Result of the operation.
        """
        total_weight = sum(self.weights.values())
        ordered_weights = tuple(sorted((key, float(value)) for key, value in self.weights.items()))
        return (
            float(self.metric_means.get("context_precision_without_reference", float("-inf"))),
            float(self.metric_means.get("faithfulness", float("-inf"))),
            -float(total_weight),
            ordered_weights,
        )


def _compact_error_text(text: str | None, *, limit: int = 96) -> str | None:
    if not text:
        return None
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _format_objective(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number == float("-inf"):
        return "n/a"
    return f"{number:.3f}"


class _TrialProgressRenderer:
    """Render overall stage2 progress with trial counts and percentages."""

    def __init__(
        self,
        *,
        width: int = 24,
        interactive: bool = True,
        log_interval_seconds: float = 30.0,
    ) -> None:
        self.width = max(8, int(width))
        self.interactive = bool(interactive)
        self.log_interval_seconds = max(1.0, float(log_interval_seconds))
        self._started_at = time.perf_counter()
        self._last_logged_elapsed = -1.0
        self._active = False

    def _render_line(
        self,
        *,
        trial_index: int,
        total_trials: int,
        phase: str,
        fraction: float,
        best_objective: float | None,
        detail: str,
        last_error: str | None = None,
    ) -> str:
        safe_total = max(1, int(total_trials))
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        pct = int(round(bounded_fraction * 100.0))
        filled = int(self.width * bounded_fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        total_elapsed = max(0.0, time.perf_counter() - self._started_at)
        line = (
            f"[stage2] [{bar}] {pct:3d}% "
            f"trial={trial_index}/{safe_total} phase={phase} "
            f"{detail} total_elapsed={total_elapsed:6.1f}s best={_format_objective(best_objective)}"
        )
        compact_error = _compact_error_text(last_error)
        if compact_error:
            line += f" last_error={compact_error}"
        return line

    def _emit(self, line: str, *, force: bool = False) -> None:
        elapsed = max(0.0, time.perf_counter() - self._started_at)
        if self.interactive:
            self._active = True
            print(f"\r{line}", end="", file=sys.stderr, flush=True)
            return

        should_log = force or self._last_logged_elapsed < 0 or (elapsed - self._last_logged_elapsed) >= self.log_interval_seconds
        if should_log:
            self._last_logged_elapsed = elapsed
            print(line, file=sys.stderr, flush=True)

    def start_trial(self, *, trial_index: int, total_trials: int, best_objective: float | None) -> None:
        safe_total = max(1, int(total_trials))
        base_fraction = float(max(0, trial_index - 1)) / safe_total
        line = self._render_line(
            trial_index=trial_index,
            total_trials=safe_total,
            phase="starting",
            fraction=base_fraction,
            best_objective=best_objective,
            detail="waiting",
        )
        self._emit(line, force=True)

    def update_prep(
        self,
        *,
        trial_index: int,
        total_trials: int,
        event: PrepProgressEvent,
        best_objective: float | None,
    ) -> None:
        safe_total = max(1, int(total_trials))
        row_total = event.total if event.total > 0 else 1
        row_fraction = event.completed / row_total
        overall_fraction = (max(0, trial_index - 1) + row_fraction * 0.8) / safe_total
        rate = (event.completed / event.elapsed_seconds) if event.elapsed_seconds > 0 else 0.0
        line = self._render_line(
            trial_index=trial_index,
            total_trials=safe_total,
            phase="prep",
            fraction=overall_fraction,
            best_objective=best_objective,
            detail=(
                f"rows={event.completed}/{event.total} errors={event.failures} "
                f"trial_elapsed={event.elapsed_seconds:5.1f}s rate={rate:5.2f}/s"
            ),
            last_error=event.last_error,
        )
        self._emit(line)

    def start_eval(
        self,
        *,
        trial_index: int,
        total_trials: int,
        prepared_rows: int,
        best_objective: float | None,
    ) -> None:
        safe_total = max(1, int(total_trials))
        overall_fraction = (max(0, trial_index - 1) + 0.85) / safe_total
        line = self._render_line(
            trial_index=trial_index,
            total_trials=safe_total,
            phase="eval",
            fraction=overall_fraction,
            best_objective=best_objective,
            detail=f"usable_rows={prepared_rows}",
        )
        self._emit(line, force=True)

    def finish_trial(
        self,
        *,
        trial_index: int,
        total_trials: int,
        result: TrialResult,
        best_objective: float | None,
    ) -> None:
        safe_total = max(1, int(total_trials))
        overall_fraction = float(trial_index) / safe_total
        line = self._render_line(
            trial_index=trial_index,
            total_trials=safe_total,
            phase="done",
            fraction=overall_fraction,
            best_objective=best_objective,
            detail=f"objective={_format_objective(result.objective)} usable_rows={result.prepared_rows}",
        )
        self._emit(line, force=True)

    def finish(self) -> None:
        if self.interactive and self._active:
            print(file=sys.stderr, flush=True)
            self._active = False


def _parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Result of the operation.
    """
    parser = argparse.ArgumentParser(description="Tune validity-aware reranker weights on a dev split.")
    parser.add_argument(
        "-c",
        "--config-file",
        default="config/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the development dataset JSON/JSONL. Defaults to evaluation.dataset.input_path.",
    )
    parser.add_argument(
        "--output-path",
        default="config/weights/validity_reranker.dev_v3.yaml",
        help="Output YAML path for the selected weights.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional JSON path for the trial manifest. Defaults to <output-path>.manifest.json.",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=None,
        help="Override dataset-preparation worker count for tuning.",
    )
    return parser.parse_args()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _resolve_dataset_path(cfg: GlobalConfig, cli_value: str | None) -> Path:
    """Resolve dataset Path.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    cli_value : str or None, optional
        Optional value provided via the command line.
    
    Returns
    -------
    Path
        Result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    eval_cfg = _as_mapping(_as_mapping(getattr(cfg, "raw", {})).get("evaluation", {}))
    dataset_cfg = _as_mapping(eval_cfg.get("dataset", {}))
    configured = dataset_cfg.get("input_path")
    if not configured:
        raise ValueError(
            "No dataset path configured for tuning. Pass --dataset-path or set evaluation.dataset.input_path."
        )
    return Path(str(configured)).expanduser().resolve()


def _weight_trials(grid: Mapping[str, Iterable[float]] | None = None) -> list[dict[str, float]]:
    """Weight Trials.
    
    Parameters
    ----------
    grid : Mapping[str, Iterable[float]] or None, optional
        Value for grid.
    
    Returns
    -------
    list[dict[str, float]]
        Collected results from the operation.
    """
    effective = {key: tuple(values) for key, values in (grid or DEFAULT_WEIGHT_GRID).items()}
    ordered_keys = tuple(sorted(effective.keys()))
    trials: list[dict[str, float]] = []
    for values in product(*(effective[key] for key in ordered_keys)):
        trials.append({key: float(value) for key, value in zip(ordered_keys, values, strict=True)})
    return trials


def _trial_sort_key(trial: TrialResult) -> tuple[float, tuple[float, float, float, tuple[tuple[str, float], ...]]]:
    """Trial Sort Key.
    
    Parameters
    ----------
    trial : TrialResult
        Value for trial.
    
    Returns
    -------
    tuple[float, tuple[float, float, float, tuple[tuple[str, float], ...]]]
        Collected results from the operation.
    """
    return (trial.objective, trial.tie_break_key())


def _select_best_trial(trials: Iterable[TrialResult]) -> TrialResult:
    """Select Best Trial.
    
    Parameters
    ----------
    trials : Iterable[TrialResult]
        Value for trials.
    
    Returns
    -------
    TrialResult
        Result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    ordered = list(trials)
    if not ordered:
        raise ValueError("No tuning trials were produced.")
    return max(ordered, key=_trial_sort_key)


def _coerce_metric_mean(values: Iterable[Any]) -> float:
    """Coerce metric Mean.
    
    Parameters
    ----------
    values : Iterable[Any]
        Value for values.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    numeric: list[float] = []
    for value in values:
        if isinstance(value, bool):
            numeric.append(float(int(value)))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:
            continue
        numeric.append(number)
    if not numeric:
        return float("nan")
    return sum(numeric) / len(numeric)


def _trial_metric_means(result: Any) -> dict[str, float]:
    """Trial Metric Means.
    
    Parameters
    ----------
    result : Any
        Evaluation or backend result object to summarize.
    
    Returns
    -------
    dict[str, float]
        Structured result of the operation.
    """
    scores_df = getattr(result, "scores_df", None)
    if scores_df is None:
        return {}
    metric_means: dict[str, float] = {}
    for metric_name in OBJECTIVE_METRICS:
        if metric_name not in getattr(scores_df, "columns", []):
            continue
        metric_means[metric_name] = _coerce_metric_mean(scores_df[metric_name].tolist())
    return metric_means


def _trial_objective(metric_means: Mapping[str, float]) -> float:
    """Trial Objective.
    
    Parameters
    ----------
    metric_means : Mapping[str, float]
        Value for metric Means.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    usable = [float(metric_means[name]) for name in OBJECTIVE_METRICS if name in metric_means and metric_means[name] == metric_means[name]]
    if not usable:
        return float("-inf")
    return sum(usable) / len(usable)


def _trial_cfg(cfg: GlobalConfig, *, weights: Mapping[str, float]) -> GlobalConfig:
    """Trial Cfg.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    weights : Mapping[str, float]
        Weight values to evaluate or persist.
    
    Returns
    -------
    GlobalConfig
        Result of the operation.
    """
    raw = copy.deepcopy(_as_mapping(getattr(cfg, "raw", {})))
    retriever_cfg = copy.deepcopy(_as_mapping(raw.get("retriever", {})))
    rerank_cfg = copy.deepcopy(_as_mapping(retriever_cfg.get("rerank", {})))
    semantic_base_cfg = copy.deepcopy(_as_mapping(rerank_cfg.get("semantic_base", {})))

    semantic_base_cfg["type"] = "rrf"
    semantic_base_cfg["rrf_k"] = int(semantic_base_cfg.get("rrf_k", rerank_cfg.get("rrf_k", 60)))

    rerank_cfg.update(
        {
            "type": "validity_aware",
            "trace_enabled": True,
            "semantic_base": semantic_base_cfg,
            "weights": {key: float(value) for key, value in weights.items()},
            "authority_values": dict(DEFAULT_AUTHORITY_VALUES),
            "status_values": dict(DEFAULT_STATUS_VALUES),
            "freshness": {"mode": "relative_recency"},
        }
    )
    rerank_cfg.pop("weights_path", None)
    retriever_cfg["rerank"] = rerank_cfg
    raw["retriever"] = retriever_cfg

    eval_cfg = copy.deepcopy(_as_mapping(raw.get("evaluation", {})))
    metric_cfg = copy.deepcopy(_as_mapping(eval_cfg.get("metrics", {})))
    metric_cfg["requested"] = list(OBJECTIVE_METRICS)
    eval_cfg["metrics"] = metric_cfg
    raw["evaluation"] = eval_cfg

    return GlobalConfig(raw=raw, config_path=getattr(cfg, "config_path", None))


def _row_source_error(row: Mapping[str, Any]) -> str | None:
    metadata = _as_mapping(row.get("metadata", {}))
    value = metadata.get("source_error")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _trial_rows_for_evaluation(
    cfg: GlobalConfig,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    usable_rows: list[dict[str, Any]] = []
    dropped_rows = 0

    for row in rows:
        if _row_source_error(row):
            dropped_rows += 1
            continue
        if not str(row.get("response", "") or "").strip():
            dropped_rows += 1
            continue
        usable_rows.append(row)

    eval_cfg = _as_mapping(_as_mapping(getattr(cfg, "raw", {})).get("evaluation", {}))
    preprocessing_cfg = _as_mapping(eval_cfg.get("preprocessing", {}))
    processed_rows, _summary = preprocess_rows_for_evaluation(
        usable_rows,
        preprocessing=preprocessing_cfg,
    )
    return processed_rows, dropped_rows


def _run_trial(
    *,
    cfg: GlobalConfig,
    raw_examples: list[dict[str, Any]],
    weights: Mapping[str, float],
    generation_workers: int | None,
    progress_callback: Any | None = None,
    phase_callback: Any | None = None,
) -> TrialResult:
    """Run Trial.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    raw_examples : list[dict[str, Any]]
        Raw examples value to normalize.
    weights : Mapping[str, float]
        Weight values to evaluate or persist.
    generation_workers : int or None, optional
        Value for generation Workers.
    
    Returns
    -------
    TrialResult
        Result of the operation.
    """
    trial_cfg = _trial_cfg(cfg, weights=weights)
    container = build_container(trial_cfg)
    eval_cfg = _as_mapping(_as_mapping(getattr(trial_cfg, "raw", {})).get("evaluation", {}))
    generation_cfg = _as_mapping(eval_cfg.get("generation", {}))
    workers = int(generation_workers or generation_cfg.get("workers") or 1)
    retry_policy = PrepRetryPolicy.from_value(_as_mapping(generation_cfg.get("retries", {})))

    pipeline = container.pipeline
    reranker_profile = None
    reranker_fingerprint = None
    reranker_profile_getter = getattr(pipeline, "_reranker_profile", None)
    reranker_fingerprint_getter = getattr(pipeline, "_reranker_fingerprint", None)
    if callable(reranker_profile_getter):
        reranker_profile = reranker_profile_getter()
    if callable(reranker_fingerprint_getter):
        reranker_fingerprint = reranker_fingerprint_getter()

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=pipeline,
        generation_workers=max(1, workers),
        llm_generate_overrides=_as_mapping(generation_cfg.get("llm_generate", {})),
        retry_policy=retry_policy,
        progress_callback=progress_callback,
        reranker_profile=reranker_profile,
        reranker_fingerprint=reranker_fingerprint,
    )
    evaluation_rows, dropped_rows = _trial_rows_for_evaluation(trial_cfg, rows)
    if dropped_rows:
        logger.warning(
            "Dropping %s/%s failed prepared rows before stage2 scoring.",
            dropped_rows,
            len(rows),
        )
    if not evaluation_rows:
        return TrialResult(
            weights={key: float(value) for key, value in weights.items()},
            objective=float("-inf"),
            metric_means={},
            reranker_fingerprint=reranker_fingerprint,
            prepared_rows=0,
        )
    if callable(phase_callback):
        phase_callback("eval", len(evaluation_rows))

    dataset = to_evaluation_dataset(evaluation_rows)
    from polaris_rag.evaluation.evaluator import Evaluator

    evaluator = Evaluator.from_global_config(
        trial_cfg,
        requested_metrics=OBJECTIVE_METRICS,
        trace_evaluator_llm=False,
    )
    result = evaluator.evaluate(
        dataset=dataset,
        source_rows=evaluation_rows,
        tune_concurrency=False,
        show_progress=False,
    )

    metric_means = _trial_metric_means(result)
    return TrialResult(
        weights={key: float(value) for key, value in weights.items()},
        objective=_trial_objective(metric_means),
        metric_means=metric_means,
        reranker_fingerprint=reranker_fingerprint,
        prepared_rows=len(evaluation_rows),
    )


def _output_manifest_path(output_path: Path, cli_value: str | None) -> Path:
    """Output Manifest Path.
    
    Parameters
    ----------
    output_path : Path
        Filesystem path used by the operation.
    cli_value : str or None, optional
        Optional value provided via the command line.
    
    Returns
    -------
    Path
        Result of the operation.
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _write_weight_file(
    *,
    path: Path,
    best: TrialResult,
    generated_at: str,
    dataset_path: Path,
) -> None:
    """Write weight File.
    
    Parameters
    ----------
    path : Path
        Filesystem path used by the operation.
    best : TrialResult
        Value for best.
    generated_at : str
        Value for generated At.
    dataset_path : Path
        Filesystem path used by the operation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_name": "validity_reranker.dev_v3",
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "objective_metrics": list(OBJECTIVE_METRICS),
        "weights": dict(best.weights),
        "authority_values": dict(DEFAULT_AUTHORITY_VALUES),
        "status_values": dict(DEFAULT_STATUS_VALUES),
        "freshness": {"mode": "relative_recency"},
        "selection": {
            "objective": float(best.objective),
            "metric_means": dict(best.metric_means),
            "prepared_rows": int(best.prepared_rows),
            "reranker_fingerprint": best.reranker_fingerprint,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_manifest(
    *,
    path: Path,
    cfg: GlobalConfig,
    dataset_path: Path,
    trials: list[TrialResult],
    best: TrialResult,
    generated_at: str,
) -> None:
    """Write manifest.
    
    Parameters
    ----------
    path : Path
        Filesystem path used by the operation.
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    dataset_path : Path
        Filesystem path used by the operation.
    trials : list[TrialResult]
        Value for trials.
    best : TrialResult
        Value for best.
    generated_at : str
        Value for generated At.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at,
        "config_file": str(getattr(cfg, "config_path", "")),
        "dataset_path": str(dataset_path),
        "objective_metrics": list(OBJECTIVE_METRICS),
        "trial_count": len(trials),
        "selected_trial": {
            "weights": dict(best.weights),
            "objective": float(best.objective),
            "metric_means": dict(best.metric_means),
            "prepared_rows": int(best.prepared_rows),
            "reranker_fingerprint": best.reranker_fingerprint,
        },
        "trials": [
            {
                "weights": dict(trial.weights),
                "objective": float(trial.objective),
                "metric_means": dict(trial.metric_means),
                "prepared_rows": int(trial.prepared_rows),
                "reranker_fingerprint": trial.reranker_fingerprint,
            }
            for trial in trials
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run the command-line entrypoint.

    Notes
    -----
    Parses CLI arguments, evaluates the configured reranker weight grid, and
    persists the selected configuration and trial manifest.
    """
    args = _parse_args()
    cfg = GlobalConfig.load(args.config_file)
    dataset_path = _resolve_dataset_path(cfg, args.dataset_path)
    raw_examples = load_raw_examples(dataset_path)
    trials = _weight_trials()
    generated_at = datetime.now(timezone.utc).isoformat()
    progress = _TrialProgressRenderer(interactive=sys.stderr.isatty())

    trial_results: list[TrialResult] = []
    best_so_far: TrialResult | None = None
    total_trials = len(trials)

    try:
        for trial_index, weights in enumerate(trials, start=1):
            progress.start_trial(
                trial_index=trial_index,
                total_trials=total_trials,
                best_objective=best_so_far.objective if best_so_far is not None else None,
            )
            result = _run_trial(
                cfg=cfg,
                raw_examples=raw_examples,
                weights=weights,
                generation_workers=args.generation_workers,
                progress_callback=lambda event, *, _idx=trial_index: progress.update_prep(
                    trial_index=_idx,
                    total_trials=total_trials,
                    event=event,
                    best_objective=best_so_far.objective if best_so_far is not None else None,
                ),
                phase_callback=lambda phase, prepared_rows, *, _idx=trial_index: progress.start_eval(
                    trial_index=_idx,
                    total_trials=total_trials,
                    prepared_rows=prepared_rows,
                    best_objective=best_so_far.objective if best_so_far is not None else None,
                ) if phase == "eval" else None,
            )
            trial_results.append(result)
            best_so_far = _select_best_trial(trial_results)
            progress.finish_trial(
                trial_index=trial_index,
                total_trials=total_trials,
                result=result,
                best_objective=best_so_far.objective if best_so_far is not None else None,
            )
    finally:
        progress.finish()

    best = _select_best_trial(trial_results)

    output_path = Path(args.output_path).expanduser().resolve()
    manifest_path = _output_manifest_path(output_path, args.manifest_path)
    _write_weight_file(
        path=output_path,
        best=best,
        generated_at=generated_at,
        dataset_path=dataset_path,
    )
    _write_manifest(
        path=manifest_path,
        cfg=cfg,
        dataset_path=dataset_path,
        trials=trial_results,
        best=best,
        generated_at=generated_at,
    )

    print(f"Wrote weights: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Selected objective: {best.objective:.6f}")
    print(f"Selected weights: {json.dumps(best.weights, sort_keys=True)}")


if __name__ == "__main__":
    main()
