from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from input_output.hdf5_io import MetricsTree, append_metrics_trees_to_h5, read_dataset
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT, find_pipeline_group
from math_utils import nanmax

from .dataclasses import Metric, MetricContributionRecord, ScoreRecord
from .metrics import METRICS, PLOT_VESSEL_TYPE, POSTPROCESS_GROUP, REPRESENTATIONS, VESSEL_TYPES


def append_scores_to_file(
    file_path: Path,
    *,
    metric_specs: dict[str, Metric] | None = None,
) -> MetricsTree:
    """Append paper-style WAS and WAS-c scores to one processed HDF5 file.

    WAS   = 10 / Nm * sum(z_m)
    WAS-c = 10 / Nm * sum(min(1, z_m))

    where z_m is the one-sided threshold-excess severity normalized by the
    control-group standard deviation. `metric_specs` must be calibrated by
    optimal_split before calling this function.
    """
    tree = _build_scores_tree(file_path, metric_specs=metric_specs or METRICS)
    append_metrics_trees_to_h5(
        file_path,
        ANGIOEYE_POSTPROCESS_ROOT,
        [tree],
        overwrite=True,
    )
    with h5py.File(file_path, "r+") as h5:
        composite_group = h5[f"{ANGIOEYE_POSTPROCESS_ROOT}/{POSTPROCESS_GROUP}"]
        for vessel_type in VESSEL_TYPES:
            for representation in REPRESENTATIONS:
                composite_group.require_group(
                    f"{vessel_type}/by_segment/{representation}"
                )
    return tree


def score_records_for_tree(
    tree: MetricsTree,
    *,
    cohort: str,
    file_path: Path,
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for representation in REPRESENTATIONS:
        base = f"{PLOT_VESSEL_TYPE}/global/{representation}"
        was = _finite_scalar(tree.metrics.get(f"{base}/WAS"))
        was_c = _finite_scalar(tree.metrics.get(f"{base}/WAS_c"))
        if was is None or was_c is None:
            continue
        records.append(
            ScoreRecord(
                cohort=cohort,
                file_name=file_path.name,
                representation=representation,
                was=was,
                was_c=was_c,
            )
        )
    return records


def contribution_records_for_tree(
    tree: MetricsTree,
    *,
    cohort: str,
    file_path: Path,
    metric_specs: dict[str, Metric],
) -> list[MetricContributionRecord]:
    """Extract per-file, per-metric contributions from an in-memory score tree."""
    records: list[MetricContributionRecord] = []
    nm = max(len(metric_specs), 1)
    scale = 10.0 / nm

    for vessel_type in VESSEL_TYPES:
        for representation in REPRESENTATIONS:
            for metric_key, metric in metric_specs.items():
                base = f"{vessel_type}/global/{representation}/components/{metric_key}"
                z = _finite_scalar(tree.metrics.get(f"{base}/z"))
                z_capped = _finite_scalar(tree.metrics.get(f"{base}/z_capped"))
                threshold = _finite_scalar(tree.metrics.get(f"{base}/threshold"))
                direction = _finite_scalar(tree.metrics.get(f"{base}/direction"))
                control_std = _finite_scalar(tree.metrics.get(f"{base}/control_std"))
                if z is None or z_capped is None:
                    continue
                records.append(
                    MetricContributionRecord(
                        cohort=cohort,
                        file_name=file_path.name,
                        vessel_type=vessel_type,
                        representation=representation,
                        metric_key=metric_key,
                        metric_name=metric.name,
                        z=float(z),
                        z_capped=float(z_capped),
                        was_points=scale * float(z),
                        was_c_points=scale * float(z_capped),
                        threshold=float(threshold) if threshold is not None else float("nan"),
                        direction=int(direction) if direction is not None else 0,
                        control_std=float(control_std) if control_std is not None else float("nan"),
                    )
                )
    return records


def _finite_scalar(value: Any) -> float | None:
    values = _finite_values(value)
    if values.size == 0:
        return None
    return float(values[0])


def _finite_values(value: Any) -> np.ndarray:
    values = np.asarray(value, dtype=float).ravel()
    return values[np.isfinite(values)]


def _metric_z(value: Any, metric: Metric, vessel_type: str) -> float:
    values = _finite_values(value)
    if values.size == 0:
        return 0.0

    sigma0 = float(metric.control_std.get(vessel_type, np.nan))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return 0.0
    if not np.isfinite(metric.threshold) or metric.direction == 0:
        raise ValueError(
            f"Metric {metric.name!r} has not been calibrated: "
            f"threshold={metric.threshold}, direction={metric.direction}."
        )

    deviation = metric.direction * (values - metric.threshold)
    z_values = np.maximum(0.0, deviation / sigma0)
    return float(nanmax(z_values))


def _read_metric_or_ratio(
    source_group: h5py.Group,
    metric: Metric,
    vessel_type: str,
    representation: str,
) -> Any | None:
    paths = metric.derived_paths(vessel_type, representation)
    if paths is None:
        return read_dataset(
            source_group,
            metric.path(vessel_type, representation),
            default=None,
        )

    numerator = read_dataset(source_group, paths[0], default=None)
    denominator = read_dataset(source_group, paths[1], default=None)
    if numerator is None or denominator is None:
        return None

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.isfinite(denominator) & (denominator != 0),
            numerator / denominator,
            np.nan,
        )


def _build_scores_tree(
    file_path: Path,
    *,
    metric_specs: dict[str, Metric],
) -> MetricsTree:
    with h5py.File(file_path, "r") as h5:
        source_group = find_pipeline_group(h5, "waveform_shape_metrics")
        if source_group is None:
            raise ValueError(
                "Expected 'waveform_shape_metrics' pipeline group not found "
                f"in {file_path}"
            )

        metrics: dict[str, Any] = {}
        nm = len(metric_specs)
        if nm == 0:
            raise ValueError("Cannot compute WAS/WAS-c with an empty metric panel.")

        for representation in REPRESENTATIONS:
            for vessel_type in VESSEL_TYPES:
                z_scores: list[float] = []
                missing_input = False

                for metric_key, metric in metric_specs.items():
                    value = _read_metric_or_ratio(
                        source_group,
                        metric,
                        vessel_type,
                        representation,
                    )
                    if value is None:
                        missing_input = True
                        break
                    z = _metric_z(value, metric, vessel_type)
                    z_scores.append(z)

                    base_metric = f"{vessel_type}/global/{representation}/components/{metric_key}"
                    metrics[f"{base_metric}/z"] = np.asarray(z, dtype=float)
                    metrics[f"{base_metric}/z_capped"] = np.asarray(min(1.0, z), dtype=float)
                    metrics[f"{base_metric}/threshold"] = np.asarray(metric.threshold, dtype=float)
                    metrics[f"{base_metric}/direction"] = np.asarray(metric.direction, dtype=int)
                    metrics[f"{base_metric}/control_std"] = np.asarray(
                        metric.control_std.get(vessel_type, np.nan),
                        dtype=float,
                    )

                if missing_input:
                    continue

                z_array = np.asarray(z_scores, dtype=float)
                was = 10.0 / nm * float(np.sum(z_array))
                was_c = 10.0 / nm * float(np.sum(np.minimum(1.0, z_array)))

                base = f"{vessel_type}/global/{representation}"
                metrics[f"{base}/WAS"] = np.asarray(was, dtype=float)
                metrics[f"{base}/WAS_c"] = np.asarray(was_c, dtype=float)

        return MetricsTree(
            name=POSTPROCESS_GROUP,
            metrics=metrics,
            attrs={
                "kind": "postprocess",
                "source_pipeline": str(source_group.name),
                "score_definition": "WAS=10/Nm*sum(z); WAS_c=10/Nm*sum(min(1,z))",
                "metric_panel_size": int(nm),
                "metric_thresholds": ";".join(
                    f"{key}:{metric.threshold}:{metric.direction}"
                    for key, metric in metric_specs.items()
                ),
            },
        )
