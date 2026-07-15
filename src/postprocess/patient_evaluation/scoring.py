from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from input_output.hdf5_io import read_dataset
from input_output.hdf5_schema import find_pipeline_group

from .dataclasses import (
    FixedMetric,
    MetricEvaluation,
    PatientEvaluation,
)
from .metrics import (
    DEFAULT_AGGREGATION,
    DEFAULT_REPRESENTATION,
    DEFAULT_VESSEL_TYPE,
    EVALUATION_BANDS,
    FIXED_METRICS,
    GREATER,
)


def evaluate_h5_patient(
    h5_path: Path,
    *,
    patient_id: str,
    source_file: str,
    archive_member: str | None = None,
    metric_panel: tuple[FixedMetric, ...] = FIXED_METRICS,
    vessel_type: str = DEFAULT_VESSEL_TYPE,
    representation: str = DEFAULT_REPRESENTATION,
    aggregation: str = DEFAULT_AGGREGATION,
) -> PatientEvaluation:
    """Evaluate one patient HDF5 file against a fixed metric panel."""
    h5_path = Path(h5_path)
    metric_evaluations: list[MetricEvaluation] = []

    with h5py.File(h5_path, "r") as h5:
        source_group = find_pipeline_group(h5, "waveform_shape_metrics")
        if source_group is None:
            raise ValueError(
                "Expected 'waveform_shape_metrics' pipeline group not found "
                f"in {h5_path}"
            )

        for metric in metric_panel:
            raw_value = _read_metric_or_ratio(
                source_group,
                metric,
                vessel_type=vessel_type,
                representation=representation,
            )
            if raw_value is None:
                metric_evaluations.append(
                    _missing_metric_evaluation(
                        patient_id=patient_id,
                        source_file=source_file,
                        archive_member=archive_member,
                        metric=metric,
                        vessel_type=vessel_type,
                        representation=representation,
                        message="Metric dataset not found.",
                    )
                )
                continue

            value = _aggregate_metric(raw_value, aggregation)
            if value is None:
                metric_evaluations.append(
                    _missing_metric_evaluation(
                        patient_id=patient_id,
                        source_file=source_file,
                        archive_member=archive_member,
                        metric=metric,
                        vessel_type=vessel_type,
                        representation=representation,
                        message="Metric contains no finite value.",
                    )
                )
                continue

            deviation = float(metric.direction) * (value - float(metric.threshold))
            abnormal = bool(deviation >= 0.0)

            if metric.control_std is None:
                z = 1.0 if abnormal else 0.0
                z_capped = z
            else:
                sigma0 = float(metric.control_std)
                if not np.isfinite(sigma0) or sigma0 <= 0:
                    raise ValueError(
                        f"Invalid control_std={metric.control_std!r} "
                        f"for metric {metric.key!r}."
                    )
                z = max(0.0, deviation / sigma0)
                z_capped = min(1.0, z)

            metric_evaluations.append(
                MetricEvaluation(
                    patient_id=patient_id,
                    source_file=source_file,
                    archive_member=archive_member,
                    vessel_type=vessel_type,
                    representation=representation,
                    metric_key=metric.key,
                    metric_name=metric.name,
                    latex_name=metric.latex_name or metric.name,
                    available=True,
                    value=float(value),
                    threshold=float(metric.threshold),
                    direction=int(metric.direction),
                    direction_label=(
                        "GREATER_OR_EQUAL"
                        if metric.direction == GREATER
                        else "LESS_OR_EQUAL"
                    ),
                    abnormal=abnormal,
                    control_std=metric.control_std,
                    z=float(z),
                    z_capped=float(z_capped),
                    weight=float(metric.weight),
                    weighted_contribution=float(metric.weight) * float(z_capped),
                )
            )

    available = [item for item in metric_evaluations if item.available]
    if not available:
        raise ValueError(
            f"None of the {len(metric_panel)} configured metrics was available "
            f"for {patient_id!r}."
        )

    total_available_weight = float(sum(item.weight for item in available))
    if total_available_weight <= 0:
        raise ValueError("The sum of available metric weights must be positive.")

    pathology_index = float(
        sum(item.weighted_contribution for item in available)
        / total_available_weight
    )
    pathology_index = min(1.0, max(0.0, pathology_index))
    n_abnormal = int(sum(item.abnormal for item in available))
    abnormal_fraction = n_abnormal / len(available)

    return PatientEvaluation(
        patient_id=patient_id,
        source_file=source_file,
        archive_member=archive_member,
        h5_file_name=h5_path.name,
        vessel_type=vessel_type,
        representation=representation,
        aggregation=aggregation,
        pathology_index=pathology_index,
        pathology_index_percent=100.0 * pathology_index,
        was_c_equivalent=10.0 * pathology_index,
        abnormal_fraction=float(abnormal_fraction),
        n_metrics_configured=len(metric_panel),
        n_metrics_available=len(available),
        n_metrics_abnormal=n_abnormal,
        coverage_fraction=len(available) / max(len(metric_panel), 1),
        evaluation_label=_evaluation_label(pathology_index),
        metric_evaluations=tuple(metric_evaluations),
    )


def _read_metric_or_ratio(
    source_group: h5py.Group,
    metric: FixedMetric,
    *,
    vessel_type: str,
    representation: str,
) -> Any | None:
    derived_paths = metric.derived_paths(vessel_type, representation)
    if derived_paths is None:
        return read_dataset(
            source_group,
            metric.path(vessel_type, representation),
            default=None,
        )

    numerator = read_dataset(source_group, derived_paths[0], default=None)
    denominator = read_dataset(source_group, derived_paths[1], default=None)
    if numerator is None or denominator is None:
        # Some pipelines store the ratio directly.
        return read_dataset(
            source_group,
            metric.path(vessel_type, representation),
            default=None,
        )

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.isfinite(denominator) & (denominator != 0),
            numerator / denominator,
            np.nan,
        )


def _aggregate_metric(value: Any, aggregation: str) -> float | None:
    finite = np.asarray(value, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    if aggregation == "median":
        return float(np.nanmedian(finite))
    if aggregation == "mean":
        return float(np.nanmean(finite))
    if aggregation == "max":
        return float(np.nanmax(finite))
    if aggregation == "min":
        return float(np.nanmin(finite))
    raise ValueError(f"Unknown aggregation={aggregation!r}")


def _evaluation_label(pathology_index: float) -> str:
    for upper_bound, label in EVALUATION_BANDS:
        if pathology_index < upper_bound:
            return label
    return EVALUATION_BANDS[-1][1]


def _missing_metric_evaluation(
    *,
    patient_id: str,
    source_file: str,
    archive_member: str | None,
    metric: FixedMetric,
    vessel_type: str,
    representation: str,
    message: str,
) -> MetricEvaluation:
    return MetricEvaluation(
        patient_id=patient_id,
        source_file=source_file,
        archive_member=archive_member,
        vessel_type=vessel_type,
        representation=representation,
        metric_key=metric.key,
        metric_name=metric.name,
        latex_name=metric.latex_name or metric.name,
        available=False,
        value=float("nan"),
        threshold=float(metric.threshold),
        direction=int(metric.direction),
        direction_label=(
            "GREATER_OR_EQUAL"
            if metric.direction == GREATER
            else "LESS_OR_EQUAL"
        ),
        abnormal=False,
        control_std=metric.control_std,
        z=float("nan"),
        z_capped=float("nan"),
        weight=float(metric.weight),
        weighted_contribution=float("nan"),
        message=message,
    )
