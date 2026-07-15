from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import mannwhitneyu

from input_output.hdf5_io import read_dataset
from input_output.hdf5_schema import find_pipeline_group
from postprocess.core.grouped_batch import build_group_order, extract_group_name

from .dataclasses import Metric
from .metrics import GREATER, LESS, METRIC_PANEL, VESSEL_TYPES

CONTROL_NAME_HINTS = (
    "control",
    "controle",
    "contrôle",
    "ctrl",
    "healthy",
    "sain",
    "temoin",
    "témoin",
)


@dataclass(frozen=True)
class SplitStats:
    metric_key: str
    metric_name: str
    vessel_type: str
    representation: str
    threshold: float
    direction: int
    direction_label: str
    control_std: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    youden_j: float
    auc_greater: float
    auc_less: float
    separability_auc: float
    p_value_mannwhitney: float
    selected_for_score: bool
    n_control: int
    n_pathology: int
    control_group: str
    pathology_group: str


def calibrate_metrics_from_processed_files(
    processed_files: Iterable[Path],
    output_dir: Path,
    *,
    metric_panel: dict[str, Metric] | None = None,
    control_group: str | None = None,
    pathology_group: str | None = None,
    optimize_vessel_type: str = "artery",
    optimize_representation: str = "bandlimited",
    aggregation: str = "median",
) -> tuple[dict[str, Metric], list[SplitStats]]:
    """Estimate one threshold and one direction per metric using two cohorts.

    The input `metric_panel` only needs path definitions. For each metric, this
    function finds the threshold and direction that best separate control and
    pathology subjects by maximizing Youden's J:

        J = sensitivity + specificity - 1

    It tests both one-sided rules:
        pathology if x >= threshold
        pathology if x <= threshold
    """
    panel = metric_panel or METRIC_PANEL
    paths = [Path(path) for path in processed_files]
    grouped: dict[str, list[Path]] = {}
    for file_path in paths:
        group = extract_group_name(file_path.parent, output_dir)
        grouped.setdefault(group, []).append(file_path)

    if len(grouped) != 2 and (control_group is None or pathology_group is None):
        raise ValueError(
            "Optimal split calibration expects exactly two cohorts, or explicit "
            f"control_group/pathology_group. Found groups: {sorted(grouped)}"
        )

    groups = build_group_order(set(grouped))
    if control_group is None:
        control_group = _infer_control_group(groups)
    if pathology_group is None:
        pathology_candidates = [group for group in groups if group != control_group]
        if len(pathology_candidates) != 1:
            raise ValueError(
                "Could not infer pathology_group. Please provide it explicitly."
            )
        pathology_group = pathology_candidates[0]

    if control_group not in grouped or pathology_group not in grouped:
        raise ValueError(
            f"Requested groups not found. Requested control={control_group!r}, "
            f"pathology={pathology_group!r}; available={sorted(grouped)}"
        )

    control_values = _metric_values_for_panel(
        grouped[control_group],
        panel,
        vessel_types=tuple(dict.fromkeys((optimize_vessel_type, *VESSEL_TYPES))),
        representation=optimize_representation,
        aggregation=aggregation,
    )
    pathology_values = _metric_values_for_panel(
        grouped[pathology_group],
        panel,
        vessel_types=(optimize_vessel_type,),
        representation=optimize_representation,
        aggregation=aggregation,
    )

    calibrated: dict[str, Metric] = {}
    stats: list[SplitStats] = []

    for metric_key, metric in panel.items():
        x0 = control_values.get(
            (metric_key, optimize_vessel_type),
            _EMPTY_VALUES,
        )
        x1 = pathology_values.get(
            (metric_key, optimize_vessel_type),
            _EMPTY_VALUES,
        )
        if x0.size == 0 or x1.size == 0:
            # Metric absent or non-finite in at least one cohort: it cannot be
            # ranked by AUC or scored robustly, so skip it.
            continue

        split = _best_one_dimensional_split(x0, x1)
        auc_greater = _roc_auc_greater(x0, x1)
        auc_less = 1.0 - auc_greater
        separability_auc = max(auc_greater, auc_less)
        p_value_mannwhitney = _mannwhitney_p_value(x0, x1)

        # WAS uses sigma0 by vessel type. The threshold/direction are optimized
        # on one reference vessel/representation, but sigma0 is estimated from
        # the control cohort for every scored vessel type.
        control_std_by_vessel: dict[str, float] = {}
        for vessel_type in VESSEL_TYPES:
            vessel_control = control_values.get(
                (metric_key, vessel_type),
                _EMPTY_VALUES,
            )
            control_std_by_vessel[vessel_type] = _robust_std(vessel_control)

        control_std = control_std_by_vessel.get(optimize_vessel_type, 1.0)
        calibrated[metric_key] = replace(
            metric,
            threshold=float(split["threshold"]),
            direction=int(split["direction"]),
            control_std=control_std_by_vessel,
        )
        stats.append(
            SplitStats(
                metric_key=metric_key,
                metric_name=metric.name,
                vessel_type=optimize_vessel_type,
                representation=optimize_representation,
                threshold=float(split["threshold"]),
                direction=int(split["direction"]),
                direction_label="GREATER"
                if int(split["direction"]) == GREATER
                else "LESS",
                control_std=float(control_std),
                sensitivity=float(split["sensitivity"]),
                specificity=float(split["specificity"]),
                balanced_accuracy=float(split["balanced_accuracy"]),
                youden_j=float(split["youden_j"]),
                auc_greater=float(auc_greater),
                auc_less=float(auc_less),
                separability_auc=float(separability_auc),
                p_value_mannwhitney=float(p_value_mannwhitney),
                selected_for_score=False,
                n_control=int(x0.size),
                n_pathology=int(x1.size),
                control_group=control_group,
                pathology_group=pathology_group,
            )
        )

    return calibrated, stats


def _mannwhitney_p_value(
    control_values: np.ndarray,
    pathology_values: np.ndarray,
) -> float:
    control_values = np.asarray(control_values, dtype=float)
    pathology_values = np.asarray(pathology_values, dtype=float)
    control_values = control_values[np.isfinite(control_values)]
    pathology_values = pathology_values[np.isfinite(pathology_values)]

    if control_values.size == 0 or pathology_values.size == 0:
        return float("nan")

    try:
        result = mannwhitneyu(
            control_values,
            pathology_values,
            alternative="two-sided",
            method="auto",
        )
        return float(result.pvalue)
    except ValueError:
        return float("nan")


def _infer_control_group(groups: list[str]) -> str:
    for group in groups:
        normalized = group.casefold()
        if any(hint in normalized for hint in CONTROL_NAME_HINTS):
            return group
    raise ValueError(
        "Could not infer control group automatically. "
        f"Available groups: {groups}. "
        "Please set control_group explicitly in run.py."
    )


_EMPTY_VALUES = np.asarray([], dtype=float)


def iter_metric_values(
    file_path: Path,
    metric_panel: dict[str, Metric],
    *,
    vessel_types: Sequence[str],
    representation: str,
) -> Iterator[tuple[str, str, Any | None]]:
    """Yield a panel's metric values while opening one HDF5 file only once."""
    with h5py.File(file_path, "r") as h5:
        source_group = find_pipeline_group(h5, "waveform_shape_metrics")
        if source_group is None:
            raise ValueError(
                "Expected 'waveform_shape_metrics' pipeline group not found "
                f"in {file_path}"
            )

        for vessel_type in vessel_types:
            for metric_key, metric in metric_panel.items():
                yield (
                    metric_key,
                    vessel_type,
                    _read_metric_value(
                        source_group,
                        metric,
                        vessel_type,
                        representation,
                    ),
                )


def _metric_values_for_panel(
    file_paths: Iterable[Path],
    metric_panel: dict[str, Metric],
    *,
    vessel_types: Sequence[str],
    representation: str,
    aggregation: str,
) -> dict[tuple[str, str], np.ndarray]:
    """Collect an entire panel with one HDF5 open per processed file."""
    values: dict[tuple[str, str], list[float]] = {
        (metric_key, vessel_type): []
        for metric_key in metric_panel
        for vessel_type in vessel_types
    }
    for file_path in file_paths:
        for metric_key, vessel_type, value in iter_metric_values(
            file_path,
            metric_panel,
            vessel_types=vessel_types,
            representation=representation,
        ):
            aggregated = _aggregate_metric_value(value, aggregation)
            if aggregated is not None:
                values[(metric_key, vessel_type)].append(aggregated)

    return {
        key: np.asarray(metric_values, dtype=float)
        for key, metric_values in values.items()
    }


def _aggregate_metric_value(value: Any | None, aggregation: str) -> float | None:
    if value is None:
        return None
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
    raise ValueError(f"Unknown aggregation={aggregation!r}")


def _read_metric_value(
    source_group: h5py.Group,
    metric: Metric,
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
        return None
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.isfinite(denominator) & (denominator != 0),
            numerator / denominator,
            np.nan,
        )


def _robust_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    std = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else float("nan")
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return std


def _best_one_dimensional_split(
    control_values: np.ndarray,
    pathology_values: np.ndarray,
) -> dict[str, float | int]:
    x0 = control_values[np.isfinite(control_values)]
    x1 = pathology_values[np.isfinite(pathology_values)]
    if x0.size == 0 or x1.size == 0:
        raise ValueError(
            "Cannot optimize split with an empty control or pathology group."
        )

    all_values = np.sort(np.unique(np.concatenate([x0, x1])))
    if all_values.size == 1:
        thresholds = all_values
    else:
        midpoints = (all_values[:-1] + all_values[1:]) / 2.0
        eps = np.finfo(float).eps * max(1.0, float(np.nanmax(np.abs(all_values))))
        thresholds = np.concatenate(
            ([all_values[0] - eps], midpoints, [all_values[-1] + eps])
        )

    best: dict[str, float | int] | None = None
    for direction in (GREATER, LESS):
        for threshold in thresholds:
            if direction == GREATER:
                tp = int(np.sum(x1 >= threshold))
                fn = int(np.sum(x1 < threshold))
                tn = int(np.sum(x0 < threshold))
                fp = int(np.sum(x0 >= threshold))
            else:
                tp = int(np.sum(x1 <= threshold))
                fn = int(np.sum(x1 > threshold))
                tn = int(np.sum(x0 > threshold))
                fp = int(np.sum(x0 <= threshold))

            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            balanced_accuracy = 0.5 * (sensitivity + specificity)
            youden_j = sensitivity + specificity - 1.0

            candidate = {
                "threshold": float(threshold),
                "direction": int(direction),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "balanced_accuracy": float(balanced_accuracy),
                "youden_j": float(youden_j),
            }
            if best is None or _is_better_split(candidate, best):
                best = candidate

    assert best is not None
    return best


def _is_better_split(
    candidate: dict[str, float | int],
    best: dict[str, float | int],
) -> bool:
    c_tuple = (float(candidate["youden_j"]), float(candidate["balanced_accuracy"]))
    b_tuple = (float(best["youden_j"]), float(best["balanced_accuracy"]))
    return c_tuple > b_tuple


def _roc_auc_greater(
    control_values: np.ndarray,
    pathology_values: np.ndarray,
) -> float:
    """ROC AUC for the rule: larger metric values indicate pathology.

    Equivalent to P(x_pathology > x_control) + 0.5 P(tie).
    This implementation has no sklearn dependency.
    """
    x0 = control_values[np.isfinite(control_values)]
    x1 = pathology_values[np.isfinite(pathology_values)]
    if x0.size == 0 or x1.size == 0:
        return float("nan")

    comparisons = x1[:, None] - x0[None, :]
    wins = float(np.sum(comparisons > 0))
    ties = float(np.sum(comparisons == 0))
    return (wins + 0.5 * ties) / float(x0.size * x1.size)
