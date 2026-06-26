"""Subgroup benchmark characterisation helpers.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
build_composition_counts
    Build long-form counts and proportions for each annotation label.
build_combination_counts
    Build joint-distribution counts for the core stage-3 labels.
build_composition_summary
    Build machine-readable summary payload.
build_summary_markdown
    Build a short report-ready markdown summary.
plot_composition_figure
    Render the subgroup-composition chart for experiment 1.
write_analysis_outputs
    Write experiment-1 outputs.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from polaris_rag.evaluation.benchmark_annotations import (
    ANALYSIS_LABEL_COLUMNS,
    ANNOTATION_METADATA_KEY,
    CORE_ANALYSIS_COLUMNS,
    LABEL_VALUE_ORDERS,
)


def _annotation_payload(row: Mapping[str, Any]) -> Mapping[str, str]:
    """Annotation Payload.
    
    Parameters
    ----------
    row : Mapping[str, Any]
        Value for row.
    
    Returns
    -------
    Mapping[str, str]
        Result of the operation.
    """
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    payload = metadata.get(ANNOTATION_METADATA_KEY)
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _split_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Split rows.
    
    Parameters
    ----------
    rows : list[dict[str, Any]]
        Value for rows.
    split_name : str
        Value for split Name.
    
    Returns
    -------
    list[dict[str, Any]]
        Collected results from the operation.
    """
    if split_name == "all":
        return list(rows)
    return [row for row in rows if _annotation_payload(row).get("split") == split_name]


def build_composition_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build long-form counts and proportions for each annotation label.
    
    Parameters
    ----------
    rows : list[dict[str, Any]]
        Value for rows.
    
    Returns
    -------
    list[dict[str, Any]]
        Constructed composition Counts.
    """

    records: list[dict[str, Any]] = []
    for split_name in ("all", "dev", "test"):
        split_rows = _split_rows(rows, split_name)
        total = len(split_rows)
        for label_name in ANALYSIS_LABEL_COLUMNS:
            counter: Counter[str] = Counter()
            for row in split_rows:
                payload = _annotation_payload(row)
                counter[str(payload.get(label_name, ""))] += 1

            for label_value in LABEL_VALUE_ORDERS[label_name]:
                count = int(counter.get(label_value, 0))
                records.append(
                    {
                        "split": split_name,
                        "label_name": label_name,
                        "label_value": label_value,
                        "count": count,
                        "total": total,
                        "proportion": (count / total) if total else 0.0,
                    }
                )
    return records


def build_combination_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build joint-distribution counts for the core stage-3 labels.
    
    Parameters
    ----------
    rows : list[dict[str, Any]]
        Value for rows.
    
    Returns
    -------
    list[dict[str, Any]]
        Constructed combination Counts.
    """

    records: list[dict[str, Any]] = []
    for split_name in ("all", "dev", "test"):
        split_rows = _split_rows(rows, split_name)
        total = len(split_rows)
        counter: Counter[tuple[str, str, str, str]] = Counter()
        for row in split_rows:
            payload = _annotation_payload(row)
            counter.update(
                [
                    tuple(str(payload.get(column, "")) for column in CORE_ANALYSIS_COLUMNS),
                ]
            )

        for key, count in sorted(counter.items()):
            record = {
                "split": split_name,
                "count": int(count),
                "total": total,
                "proportion": (count / total) if total else 0.0,
            }
            for column, value in zip(CORE_ANALYSIS_COLUMNS, key, strict=True):
                record[column] = value
            records.append(record)
    return records


def build_composition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build machine-readable summary payload.
    
    Parameters
    ----------
    rows : list[dict[str, Any]]
        Value for rows.
    
    Returns
    -------
    dict[str, Any]
        Constructed composition Summary.
    """

    totals = {split_name: len(_split_rows(rows, split_name)) for split_name in ("all", "dev", "test")}
    count_records = build_composition_counts(rows)

    label_counts: dict[str, dict[str, dict[str, int]]] = {}
    for record in count_records:
        split_name = str(record["split"])
        label_name = str(record["label_name"])
        label_value = str(record["label_value"])
        label_counts.setdefault(split_name, {}).setdefault(label_name, {})[label_value] = int(record["count"])

    targeted_subsets: dict[str, dict[str, int]] = {}
    for split_name in ("all", "dev", "test"):
        split_rows = _split_rows(rows, split_name)
        targeted_subsets[split_name] = {
            "validity_sensitive": sum(
                1 for row in split_rows if _annotation_payload(row).get("validity_sensitive") == "yes"
            ),
            "attachment_dependent": sum(
                1 for row in split_rows if _annotation_payload(row).get("attachment_dependent") == "yes"
            ),
            "local_official_docs": sum(
                1 for row in split_rows if _annotation_payload(row).get("docs_scope_needed") == "local_official"
            ),
            "external_official_docs": sum(
                1 for row in split_rows if _annotation_payload(row).get("docs_scope_needed") == "external_official"
            ),
            "local_and_external_docs": sum(
                1 for row in split_rows if _annotation_payload(row).get("docs_scope_needed") == "local_and_external"
            ),
            "query_type_software_version": sum(
                1 for row in split_rows if _annotation_payload(row).get("query_type") == "software_version"
            ),
        }

    return {
        "totals": totals,
        "label_counts": label_counts,
        "targeted_subsets": targeted_subsets,
    }


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Build a short report-ready markdown summary.
    
    Parameters
    ----------
    summary : Mapping[str, Any]
        Summary payload to render or persist.
    
    Returns
    -------
    str
        Constructed summary Markdown.
    """

    totals = dict(summary.get("totals", {}))
    targeted = {
        str(key): dict(value)
        for key, value in dict(summary.get("targeted_subsets", {})).items()
    }
    all_targeted = targeted.get("all", {})

    lines = [
        "# Experiment 1: Benchmark Characterisation",
        "",
        (
            f"The benchmark contains {totals.get('all', 0)} total queries, "
            f"with {totals.get('dev', 0)} dev examples and {totals.get('test', 0)} test examples."
        ),
        "",
        (
            f"Validity-sensitive queries account for {all_targeted.get('validity_sensitive', 0)} "
            f"of {totals.get('all', 0)} benchmark items."
        ),
        (
            f"Attachment-dependent queries account for {all_targeted.get('attachment_dependent', 0)} "
            f"of {totals.get('all', 0)} benchmark items."
        ),
        (
            f"Queries needing local official docs account for {all_targeted.get('local_official_docs', 0)} items, "
            f"while external-only official doc needs account for "
            f"{all_targeted.get('external_official_docs', 0)}."
        ),
        (
            f"Software-version queries account for {all_targeted.get('query_type_software_version', 0)} "
            f"benchmark items."
        ),
        "",
        (
            "The dev/test split retains the targeted subsets needed for later source-aware, "
            "validity-aware, and attachment-aware experiments."
        ),
        "",
    ]
    return "\n".join(lines)


def _write_csv_records(path: Path, records: list[Mapping[str, Any]]) -> None:
    """Write csv Records.
    
    Parameters
    ----------
    path : Path
        Filesystem path used by the operation.
    records : list[Mapping[str, Any]]
        Value for records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _import_matplotlib_pyplot():
    """Import Matplotlib Pyplot.
    
    This helper is internal to the surrounding module.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def plot_composition_figure(*, count_records: list[Mapping[str, Any]], output_png: Path, output_svg: Path) -> None:
    """Render the subgroup-composition chart for experiment 1.
    
    Parameters
    ----------
    count_records : list[Mapping[str, Any]]
        Value for count Records.
    output_png : Path
        Value for output Png.
    output_svg : Path
        Value for output Svg.
    """

    plt = _import_matplotlib_pyplot()

    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for record in count_records:
        split_name = str(record["split"])
        label_name = str(record["label_name"])
        label_value = str(record["label_value"])
        grouped.setdefault(label_name, {}).setdefault(split_name, {})[label_value] = int(record["count"])

    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    axes_list = list(axes.flatten())
    colors = {"all": "#1b5e20", "dev": "#1565c0", "test": "#ef6c00"}
    split_order = ("all", "dev", "test")

    for index, label_name in enumerate(ANALYSIS_LABEL_COLUMNS):
        axis = axes_list[index]
        values = list(LABEL_VALUE_ORDERS[label_name])
        x_positions = list(range(len(values)))
        width = 0.24

        for split_offset, split_name in enumerate(split_order):
            counts = [grouped.get(label_name, {}).get(split_name, {}).get(value, 0) for value in values]
            offset = (split_offset - 1) * width
            axis.bar(
                [position + offset for position in x_positions],
                counts,
                width=width,
                label=split_name if index == 0 else None,
                color=colors[split_name],
            )

        axis.set_title(label_name.replace("_", " "))
        axis.set_xticks(x_positions)
        axis.set_xticklabels(values, rotation=25, ha="right")
        axis.set_ylabel("Count")
        axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    for axis in axes_list[len(ANALYSIS_LABEL_COLUMNS) :]:
        axis.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower right", ncol=3, frameon=False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=200)
    figure.savefig(output_svg)
    plt.close(figure)


def write_analysis_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write experiment-1 outputs.
    
    Parameters
    ----------
    rows : list[dict[str, Any]]
        Value for rows.
    output_dir : str or Path
        Value for output Dir.
    
    Returns
    -------
    dict[str, Path]
        Structured result of the operation.
    """

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    count_records = build_composition_counts(rows)
    combination_records = build_combination_counts(rows)
    summary = build_composition_summary(rows)
    summary_markdown = build_summary_markdown(summary)

    counts_path = out_dir / "composition_counts.csv"
    combinations_path = out_dir / "composition_combinations.csv"
    summary_json_path = out_dir / "composition_summary.json"
    summary_md_path = out_dir / "composition_summary.md"
    figure_png_path = out_dir / "composition_figure.png"
    figure_svg_path = out_dir / "composition_figure.svg"

    _write_csv_records(counts_path, count_records)
    _write_csv_records(combinations_path, combination_records)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(summary_markdown, encoding="utf-8")
    plot_composition_figure(
        count_records=count_records,
        output_png=figure_png_path,
        output_svg=figure_svg_path,
    )

    return {
        "composition_counts_csv": counts_path,
        "composition_combinations_csv": combinations_path,
        "composition_summary_json": summary_json_path,
        "composition_summary_md": summary_md_path,
        "composition_figure_png": figure_png_path,
        "composition_figure_svg": figure_svg_path,
    }


__all__ = [
    "build_combination_counts",
    "build_composition_counts",
    "build_composition_summary",
    "build_summary_markdown",
    "plot_composition_figure",
    "write_analysis_outputs",
]
