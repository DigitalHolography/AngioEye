import h5py
import numpy as np

from pathlib import Path
from typing import Any

from input_output.hdf5_io import MetricsTree, append_metrics_trees_to_h5, read_dataset
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT, find_pipeline_group

from .dataclasses import Metric, ScoreRecord
from .metrics import DOMAINS, METRICS, PLOT_VESSEL_TYPE, POSTPROCESS_GROUP, REPRESENTATIONS, VESSEL_TYPES, GREATER, LESS
 
def append_scores_to_file(file_path: Path) -> MetricsTree:
    tree = _build_scores_tree(file_path)
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
        rwas = _finite_scalar(tree.metrics.get(f"{base}/RWAS"))
        rwas4 = _finite_scalar(tree.metrics.get(f"{base}/RWAS4"))
        if rwas is None or rwas4 is None:
            continue
        records.append(
            ScoreRecord(
                cohort=cohort,
                file_name=file_path.name,
                representation=representation,
                rwas=rwas,
                rwas4=rwas4,
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

def _severity(value: Any, metric: Metric, vessel_type: str) -> float:
    values = _finite_values(value)
    if values.size == 0:
        return 0.0

    deviation = metric.direction * (values - metric.threshold)
    normalized = np.maximum(0.0, deviation / metric.control_std[vessel_type])
    return float(np.nanmax(normalized))

def _has_abnormal_value(value: Any, metric: Metric) -> bool:
    values = _finite_values(value)
    if values.size == 0:
        return False
    if metric.direction == GREATER:
        return bool(np.any(values >= metric.threshold))
    return bool(np.any(values <= metric.threshold))

def _build_scores_tree(file_path: Path) -> MetricsTree:
    with h5py.File(file_path, "r") as h5:
        source_group = find_pipeline_group(h5, "waveform_shape_metrics")
        if source_group is None:
            raise ValueError(
                "Expected 'waveform_shape_metrics' pipeline group not found "
                f"in {file_path}"
            )

        metrics: dict[str, Any] = {}
        for representation in REPRESENTATIONS:
            values: dict[tuple[str, str], Any] = {}
            missing_input = False
            for vessel_type in VESSEL_TYPES:
                for metric_key, metric in METRICS.items():
                    paths = metric.derived_paths(vessel_type, representation)
                    if paths is None:
                        value = read_dataset(
                            source_group,
                            metric.path(vessel_type, representation),
                            default=None,
                        )
                    else:
                        numerator = read_dataset(
                            source_group,
                            paths[0],
                            default=None,
                        )
                        denominator = read_dataset(
                            source_group,
                            paths[1],
                            default=None,
                        )
                        if numerator is None or denominator is None:
                            value = None
                        else:
                            numerator = np.asarray(numerator, dtype=float)
                            denominator = np.asarray(denominator, dtype=float)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                value = np.where(
                                    np.isfinite(denominator) & (denominator != 0),
                                    numerator / denominator,
                                    np.nan,
                                )
                    if value is None:
                        missing_input = True
                        break
                    values[(vessel_type, metric_key)] = value
                if missing_input:
                    break

            if missing_input:
                continue

            weighted_scores = {"artery": 0.0, "vein": 0.0}
            rwas4_score = 0
            for domain in DOMAINS.values():
                domain_has_abnormality = False
                for vessel_type in VESSEL_TYPES:
                    domain_severity = max(
                        _severity(
                            values[(vessel_type, metric_key)],
                            METRICS[metric_key],
                            vessel_type,
                        )
                        for metric_key in domain.metrics
                    )
                    weighted_scores[vessel_type] += domain.weight * domain_severity
                    domain_has_abnormality = domain_has_abnormality or any(
                        _has_abnormal_value(
                            values[(vessel_type, metric_key)],
                            METRICS[metric_key],
                        )
                        for metric_key in domain.metrics
                    )
                rwas4_score += int(domain_has_abnormality)

            rwas_score = max(weighted_scores.values())

            for vessel_type in VESSEL_TYPES:
                base = f"{vessel_type}/global/{representation}"

                metrics[f"{base}/RWAS"] = np.asarray(rwas_score, dtype=float)
                metrics[f"{base}/RWAS4"] = np.asarray(rwas4_score, dtype=int)

        return MetricsTree(
            name=POSTPROCESS_GROUP,
            metrics=metrics,
            attrs={
                "kind": "postprocess",
                "source_pipeline": str(source_group.name),
            },
        )
  
