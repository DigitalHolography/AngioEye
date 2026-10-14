"""Shared numerical utilities used across AngioEye.

Numerical reductions and cohort statistics are shared by the batch analysis
pipelines and their reports.
"""

from .statistics import (
    DEFAULT_FLOAT_DTYPE,
    auc_from_scores,
    best_threshold_sensitivity_specificity_cumulative_sweep,
    clean_values,
    cohen_d,
    compute_axis_statistics,
    finite_values,
    mann_whitney_pvalue,
    nanargmax,
    nanargmin,
    nanmad,
    nanmax,
    nanmean,
    nanmedian,
    nanmedian_or_nan,
    nanmin,
    nanpercentile,
    nanstd,
    nansum,
    summarize_values,
)

__all__ = [
    "DEFAULT_FLOAT_DTYPE",
    "auc_from_scores",
    "best_threshold_sensitivity_specificity_cumulative_sweep",
    "clean_values",
    "cohen_d",
    "compute_axis_statistics",
    "finite_values",
    "mann_whitney_pvalue",
    "nanargmax",
    "nanargmin",
    "nanmax",
    "nanmad",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanstd",
    "nansum",
    "nanmedian_or_nan",
    "summarize_values",
]
