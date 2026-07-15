from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable
import re

import numpy as np

from .dataclasses import PatientEvaluation


def write_evaluation_plots(
    evaluations: Iterable[PatientEvaluation],
    output_dir: Path,
) -> list[str]:
    # Keep the order produced by run.py. For a ZIP input, inputs.py preserves
    # the H5 order stored in the original archive.
    evaluation_list = list(evaluations)
    if not evaluation_list:
        return []

    case_infos = _build_group_case_infos(evaluation_list)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / "png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    summary_path = plot_dir / "patient_pathology_index.png"
    _plot_patient_summary(
        evaluation_list,
        case_infos=case_infos,
        output_path=summary_path,
        plt=plt,
    )
    created.append(str(summary_path))

    for evaluation, case_info in zip(evaluation_list, case_infos):
        patient_path = (
            plot_dir
            / f"{case_info['file_stem']}_metric_contributions.png"
        )
        _plot_patient_metrics(
            evaluation,
            display_label=case_info["display_label"],
            output_path=patient_path,
            plt=plt,
        )
        created.append(str(patient_path))

    return created


def _plot_patient_summary(
    evaluations: list[PatientEvaluation],
    *,
    case_infos: list[dict[str, object]],
    output_path: Path,
    plt,
) -> None:
    labels = [str(info["display_label"]) for info in case_infos]
    values = [item.pathology_index_percent for item in evaluations]

    fig_width = max(8.0, min(24.0, 0.62 * len(labels) + 4.5))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    positions = np.arange(len(labels))

    bars = ax.bar(
        positions,
        values,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.axhline(33.0, linestyle="--", linewidth=1.0)
    ax.axhline(67.0, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Pathology compatibility index (%)")
    ax.set_xlabel("Case within source subfolder")
    ax.set_title("Fixed-threshold evaluation by patient")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if len(evaluations) <= 40:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 2.0, 98.0),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_patient_metrics(
    evaluation: PatientEvaluation,
    *,
    display_label: str,
    output_path: Path,
    plt,
) -> None:
    available = [
        item for item in evaluation.metric_evaluations if item.available
    ]
    labels = [item.metric_name for item in available]
    values = [item.z_capped for item in available]

    fig_height = max(4.0, 0.42 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    positions = np.arange(len(labels))

    ax.barh(
        positions,
        values,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Capped metric contribution")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(
        f"{display_label}: "
        f"{evaluation.pathology_index_percent:.1f}%"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_group_case_infos(
    evaluations: list[PatientEvaluation],
) -> list[dict[str, object]]:
    counters: dict[str, int] = {}
    infos: list[dict[str, object]] = []

    for evaluation in evaluations:
        group_name = _source_group_name(evaluation)
        group_case_index = counters.get(group_name, 0) + 1
        counters[group_name] = group_case_index

        infos.append(
            {
                "group_name": group_name,
                "group_case_index": group_case_index,
                "display_label": f"{group_name} #{group_case_index}",
                "file_stem": (
                    f"{_safe_name(group_name)}_{group_case_index:03d}"
                ),
            }
        )

    return infos


def _source_group_name(evaluation: PatientEvaluation) -> str:
    """Return the immediate parent folder of the patient H5.

    Examples:
        h5/group1/patient.h5 -> group1
        group1/patient.h5    -> group1

    If the application already extracted the ZIP, `archive_member` may be
    absent. In that case, the parent directory of `source_file` is used.
    """
    if evaluation.archive_member:
        member = PurePosixPath(
            str(evaluation.archive_member).replace("\\", "/")
        )
        clean_parts = [
            part
            for part in member.parts
            if part not in {"", ".", ".."}
        ]
        if len(clean_parts) >= 2:
            return clean_parts[-2]

    source_path = Path(str(evaluation.source_file))
    parent_name = source_path.parent.name.strip()
    if parent_name:
        return parent_name

    return "root"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    safe = safe or "group"
    return safe[:40]