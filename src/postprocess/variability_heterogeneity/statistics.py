import numpy as np

from math_utils import (
    best_threshold_sensitivity_specificity_cumulative_sweep,
    clean_values,
    mann_whitney_pvalue,
)

from .constants import EPS

best_threshold_sensitivity_specificity = (
    best_threshold_sensitivity_specificity_cumulative_sweep
)


def combine_variability_score(results_for_group, metric_name, higher_metrics, eps=EPS):
    """Build a normalized per-file composite variability score."""
    metric_block = results_for_group.get(metric_name, {})
    median_level = np.asarray(metric_block.get("MED_seg_medbeat", []), dtype=float)
    arrays = []
    for high_name in higher_metrics:
        values = np.asarray(metric_block.get(high_name, []), dtype=float)
        if values.size == 0:
            continue
        if high_name.startswith("CV_"):
            normalized = values
        else:
            min_len = min(len(values), len(median_level))
            if min_len == 0:
                continue
            normalized = values[:min_len] / (
                np.abs(median_level[:min_len]) + eps
            )
        arrays.append(np.asarray(normalized, dtype=float))
    if not arrays:
        return np.asarray([], dtype=float)
    min_len = min(len(values) for values in arrays)
    if min_len == 0:
        return np.asarray([], dtype=float)
    matrix = np.vstack([values[:min_len] for values in arrays]).T
    return clean_values(np.nanmean(matrix, axis=1))


def get_descriptor_values_for_test(
    results_for_group,
    metric_name,
    high_name,
    eps=EPS,
):
    """Return normalized descriptor values used by descriptor-level tests."""
    metric_block = results_for_group.get(metric_name, {})
    values = np.asarray(metric_block.get(high_name, []), dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    if high_name.startswith("CV_"):
        return clean_values(values)
    median_level = np.asarray(metric_block.get("MED_seg_medbeat", []), dtype=float)
    min_len = min(len(values), len(median_level))
    if min_len == 0:
        return np.asarray([], dtype=float)
    normalized = values[:min_len] / (np.abs(median_level[:min_len]) + eps)
    return clean_values(normalized)


__all__ = [
    "best_threshold_sensitivity_specificity",
    "best_threshold_sensitivity_specificity_cumulative_sweep",
    "combine_variability_score",
    "get_descriptor_values_for_test",
    "mann_whitney_pvalue",
]
