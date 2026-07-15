from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Iterable
import re

from .dataclasses import EvaluationFailure, PatientEvaluation


def write_evaluation_reports(
    evaluations: Iterable[PatientEvaluation],
    failures: Iterable[EvaluationFailure],
    output_dir: Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_dir = output_dir / "patients"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the order received from run.py. For ZIP input, this matches the
    # H5 order stored in the original ZIP.
    evaluation_list = list(evaluations)
    failure_list = list(failures)
    case_infos = _build_group_case_infos(evaluation_list)
    created: list[str] = []

    summary_rows = [
        _patient_summary_row(evaluation, case_info)
        for evaluation, case_info in zip(evaluation_list, case_infos)
    ]
    summary_path = output_dir / "patient_evaluations.csv"
    _write_csv(summary_path, summary_rows, _summary_fields())
    created.append(str(summary_path))

    mapping_rows = [
        _case_mapping_row(evaluation, case_info)
        for evaluation, case_info in zip(evaluation_list, case_infos)
    ]
    mapping_path = output_dir / "case_index_mapping.csv"
    _write_csv(mapping_path, mapping_rows, _mapping_fields())
    created.append(str(mapping_path))

    metric_rows = [
        _metric_row(metric, case_info)
        for evaluation, case_info in zip(evaluation_list, case_infos)
        for metric in evaluation.metric_evaluations
    ]
    metric_path = output_dir / "patient_metric_evaluations.csv"
    _write_csv(metric_path, metric_rows, _metric_fields())
    created.append(str(metric_path))

    # Short per-patient filenames:
    # patients/<subfolder>_001_summary.csv
    for evaluation, case_info in zip(evaluation_list, case_infos):
        case_stem = str(case_info["file_stem"])
        patient_summary = patient_dir / f"{case_stem}_summary.csv"
        patient_metrics = patient_dir / f"{case_stem}_metrics.csv"

        _write_csv(
            patient_summary,
            [_patient_summary_row(evaluation, case_info)],
            _summary_fields(),
        )
        _write_csv(
            patient_metrics,
            [
                _metric_row(metric, case_info)
                for metric in evaluation.metric_evaluations
            ],
            _metric_fields(),
        )
        created.extend([str(patient_summary), str(patient_metrics)])

    if failure_list:
        failure_path = output_dir / "evaluation_failures.csv"
        _write_csv(
            failure_path,
            [asdict(item) for item in failure_list],
            [
                "source_file",
                "archive_member",
                "patient_id",
                "error_type",
                "message",
            ],
        )
        created.append(str(failure_path))

    return created


def _patient_summary_row(
    evaluation: PatientEvaluation,
    case_info: dict[str, object],
) -> dict:
    row = asdict(evaluation)
    row.pop("metric_evaluations", None)
    return {
        "group_name": case_info["group_name"],
        "group_case_index": case_info["group_case_index"],
        "case_label": case_info["display_label"],
        **row,
    }


def _case_mapping_row(
    evaluation: PatientEvaluation,
    case_info: dict[str, object],
) -> dict:
    return {
        "group_name": case_info["group_name"],
        "group_case_index": case_info["group_case_index"],
        "case_label": case_info["display_label"],
        "patient_id": evaluation.patient_id,
        "archive_member": evaluation.archive_member,
        "h5_file_name": evaluation.h5_file_name,
        "source_file": evaluation.source_file,
    }


def _metric_row(
    metric,
    case_info: dict[str, object],
) -> dict:
    row = asdict(metric)
    row = {
        key: _csv_value(value)
        for key, value in row.items()
    }
    return {
        "group_name": case_info["group_name"],
        "group_case_index": case_info["group_case_index"],
        "case_label": case_info["display_label"],
        **row,
    }


def _csv_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _summary_fields() -> list[str]:
    return [
        "group_name",
        "group_case_index",
        "case_label",
        "patient_id",
        "source_file",
        "archive_member",
        "h5_file_name",
        "vessel_type",
        "representation",
        "aggregation",
        "pathology_index",
        "pathology_index_percent",
        "was_c_equivalent",
        "abnormal_fraction",
        "n_metrics_configured",
        "n_metrics_available",
        "n_metrics_abnormal",
        "coverage_fraction",
        "evaluation_label",
    ]


def _mapping_fields() -> list[str]:
    return [
        "group_name",
        "group_case_index",
        "case_label",
        "patient_id",
        "archive_member",
        "h5_file_name",
        "source_file",
    ]


def _metric_fields() -> list[str]:
    return [
        "group_name",
        "group_case_index",
        "case_label",
        "patient_id",
        "source_file",
        "archive_member",
        "vessel_type",
        "representation",
        "metric_key",
        "metric_name",
        "latex_name",
        "available",
        "value",
        "threshold",
        "direction",
        "direction_label",
        "abnormal",
        "control_std",
        "z",
        "z_capped",
        "weight",
        "weighted_contribution",
        "message",
    ]


def _write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


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