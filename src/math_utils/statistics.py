"""Reusable NaN-aware reductions and statistical helpers.

NumPy's NaN reductions are the right primitives for this project, but several
of them emit warnings for empty or all-NaN slices.  These wrappers suppress
those expected warnings and return NaN for reductions that have no finite
input.  The module also contains robust descriptive statistics and common
two-sample comparison helpers used by post-processing dashboards.
"""

from __future__ import annotations

import warnings
from math import erf, sqrt

import numpy as np


# AngioEye's numerical data path uses float32.  Keep the precision policy in
# one place so all shared reductions use the same default.
DEFAULT_FLOAT_DTYPE = np.float32


def _as_float_array(
    x: np.ndarray | list | tuple,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Convert input to a numerical array with an explicit precision policy."""
    return np.asarray(x, dtype=dtype)


def _nan_result_for_reduction(
    x: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> np.ndarray:
    """Return an appropriately shaped NaN result for empty reductions."""
    if axis is None:
        return np.asarray(np.nan, dtype=DEFAULT_FLOAT_DTYPE)

    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = x.ndim
    normalized_axes = {item if item >= 0 else ndim + item for item in axes}
    shape = list(x.shape)
    if keepdims:
        for item in normalized_axes:
            shape[item] = 1
    else:
        shape = [size for index, size in enumerate(shape) if index not in normalized_axes]
    return np.full(tuple(shape), np.nan, dtype=DEFAULT_FLOAT_DTYPE)


def nanmean(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nanmean`` without warnings for empty/all-NaN slices."""
    values = _as_float_array(x, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis, keepdims=keepdims)


def nanmedian(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nanmedian`` without warnings for empty/all-NaN slices."""
    values = _as_float_array(x, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis, keepdims=keepdims)


def nanstd(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nanstd`` without warnings for empty/all-NaN slices."""
    values = _as_float_array(x, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis, ddof=ddof, keepdims=keepdims)


def nanvar(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nanvar`` without warnings for empty/all-NaN slices."""
    values = _as_float_array(x, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanvar(values, axis=axis, ddof=ddof, keepdims=keepdims)


def nansum(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nansum`` with the standard zero-for-no-finite-values rule."""
    values = _as_float_array(x, dtype)
    return np.nansum(values, axis=axis, keepdims=keepdims)


def nanmin(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return a warning-free NaN-aware minimum.

    Unlike ``np.nanmin`` on an empty array, this returns NaN instead of
    raising ``ValueError``.  All-NaN slices also return NaN.
    """
    values = _as_float_array(x, dtype)
    if values.size == 0:
        return _nan_result_for_reduction(values, axis, keepdims)

    finite = np.isfinite(values)
    if axis is None and not np.any(finite):
        return np.asarray(np.nan, dtype=DEFAULT_FLOAT_DTYPE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmin(values, axis=axis, keepdims=keepdims)

    if axis is not None:
        finite_count = np.sum(finite, axis=axis, keepdims=keepdims)
        result = np.where(finite_count > 0, result, np.nan)
    return result


def nanmax(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return a warning-free NaN-aware maximum.

    Unlike ``np.nanmax`` on an empty array, this returns NaN instead of
    raising ``ValueError``.  All-NaN slices also return NaN.
    """
    values = _as_float_array(x, dtype)
    if values.size == 0:
        return _nan_result_for_reduction(values, axis, keepdims)

    finite = np.isfinite(values)
    if axis is None and not np.any(finite):
        return np.asarray(np.nan, dtype=DEFAULT_FLOAT_DTYPE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmax(values, axis=axis, keepdims=keepdims)

    if axis is not None:
        finite_count = np.sum(finite, axis=axis, keepdims=keepdims)
        result = np.where(finite_count > 0, result, np.nan)
    return result


def nanpercentile(
    x: np.ndarray | list | tuple,
    q: float | np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return ``np.nanpercentile`` without warnings for empty/all-NaN input."""
    values = _as_float_array(x, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(values, q, axis=axis, keepdims=keepdims)


def _nanarg_reduction(
    x: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    operation,
) -> np.ndarray:
    if x.size == 0:
        result_shape = _nan_result_for_reduction(x, axis, keepdims).shape
        return np.full(result_shape, -1, dtype=int)

    finite = np.isfinite(x)
    if isinstance(axis, tuple):
        axes = tuple(sorted(item if item >= 0 else x.ndim + item for item in axis))
        if any(item < 0 or item >= x.ndim for item in axes):
            raise np.AxisError(axis, x.ndim)
        remaining = tuple(item for item in range(x.ndim) if item not in axes)
        permutation = remaining + axes
        values = np.transpose(x, permutation)
        finite = np.transpose(finite, permutation)
        reduced_size = int(np.prod([x.shape[item] for item in axes], dtype=int))
        leading_shape = tuple(x.shape[item] for item in remaining)
        values = values.reshape((*leading_shape, reduced_size))
        finite = finite.reshape((*leading_shape, reduced_size))
        finite_count = np.sum(finite, axis=-1)
        fill_value = np.inf if operation is np.nanargmin else -np.inf
        values = np.where(finite, values, fill_value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = operation(values, axis=-1)
        result = np.where(finite_count > 0, result, -1)
        if keepdims:
            result_shape = tuple(1 if item in axes else x.shape[item] for item in range(x.ndim))
            return result.reshape(result_shape)
        return result

    if axis is None:
        if not np.any(finite):
            return np.asarray(-1, dtype=int)

        # ``np.nanargmin``/``np.nanargmax`` raise for an all-NaN input.  Use
        # a finite sentinel so the shared API can return its documented -1.
        fill_value = np.inf if operation is np.nanargmin else -np.inf
        values = np.where(finite, x, fill_value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return operation(values, axis=axis, keepdims=keepdims)

    finite_count = np.sum(finite, axis=axis, keepdims=keepdims)
    fill_value = np.inf if operation is np.nanargmin else -np.inf
    values = np.where(finite, x, fill_value)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = operation(values, axis=axis, keepdims=keepdims)

    return np.where(finite_count > 0, result, -1)


def nanargmin(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return a NaN-aware argmin, using ``-1`` for empty/all-NaN slices."""
    return _nanarg_reduction(
        _as_float_array(x, dtype), axis, keepdims, np.nanargmin
    )


def nanargmax(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return a NaN-aware argmax, using ``-1`` for empty/all-NaN slices."""
    return _nanarg_reduction(
        _as_float_array(x, dtype), axis, keepdims, np.nanargmax
    )


def nanmad(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return the median absolute deviation, ignoring non-finite values."""
    values = _as_float_array(x, dtype)
    center = nanmedian(values, axis=axis, keepdims=True, dtype=None)
    return nanmedian(
        np.abs(values - center), axis=axis, keepdims=keepdims, dtype=None
    )


def nancv(
    x: np.ndarray | list | tuple,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    dtype: np.dtype | type | None = DEFAULT_FLOAT_DTYPE,
) -> np.ndarray:
    """Return NaN-aware coefficient of variation ``std / abs(mean)``."""
    values = _as_float_array(x, dtype)
    mean = nanmean(values, axis=axis, keepdims=keepdims, dtype=None)
    std = nanstd(values, axis=axis, ddof=ddof, keepdims=keepdims, dtype=None)
    return std / np.abs(mean)


def finite_values(x: np.ndarray | list | tuple) -> np.ndarray:
    """Return finite values as a one-dimensional float array."""
    values = np.asarray(x, dtype=float)
    return values[np.isfinite(values)]


# Descriptive aliases retained for callers that use these names explicitly.
finite_1d = finite_values
clean_values = finite_values


def iqr_1d(x: np.ndarray | list | tuple) -> float:
    """Return the interquartile range of finite values, or NaN if empty."""
    values = finite_values(x)
    if values.size == 0:
        return np.nan
    q25 = nanpercentile(values, 25, dtype=None)
    q75 = nanpercentile(values, 75, dtype=None)
    return float(q75 - q25)


def mad_1d(x: np.ndarray | list | tuple) -> float:
    """Return the median absolute deviation of finite values, or NaN."""
    values = finite_values(x)
    if values.size == 0:
        return np.nan
    return float(nanmad(values, dtype=None))


def cv_1d(x: np.ndarray | list | tuple, eps: float = 1e-12) -> float:
    """Return sample standard deviation divided by absolute mean."""
    values = finite_values(x)
    if values.size == 0:
        return np.nan
    mean = nanmean(values, dtype=None)
    std = nanstd(values, ddof=1, dtype=None) if values.size > 1 else 0.0
    return float(std / (np.abs(mean) + eps))


def median_1d(x: np.ndarray | list | tuple) -> float:
    """Return the median of finite values, or NaN if empty."""
    values = finite_values(x)
    if values.size == 0:
        return np.nan
    return float(nanmedian(values, dtype=None))


def std_1d(x: np.ndarray | list | tuple) -> float:
    """Return sample standard deviation of finite values, or NaN if empty."""
    values = finite_values(x)
    if values.size == 0:
        return np.nan
    return float(nanstd(values, ddof=1, dtype=None) if values.size > 1 else 0.0)


def nanmedian_or_nan(x: np.ndarray | list | tuple) -> float:
    """Return the finite median, or NaN when no finite value exists."""
    values = np.asarray(x, dtype=float)
    if np.any(np.isfinite(values)):
        return float(nanmedian(values, dtype=None))
    return np.nan


def compute_axis_statistics(
    values: np.ndarray | list | tuple,
    axis: int,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compute robust statistics for every slice along one matrix axis."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Axis statistics require a two-dimensional array.")

    samples = np.moveaxis(values, axis, -1)
    samples = np.where(np.isfinite(samples), samples, np.nan)
    result_size = samples.shape[0]
    result = {
        name: np.full(result_size, np.nan, dtype=float)
        for name in ("median", "std", "iqr", "mad", "cv")
    }

    if result_size == 0 or samples.shape[1] == 0:
        return result

    counts = np.sum(np.isfinite(samples), axis=1)
    valid = counts > 0
    if not np.any(valid):
        return result

    valid_samples = samples[valid]
    valid_counts = counts[valid]
    medians = nanmedian(valid_samples, axis=1, dtype=None)
    quartiles = nanpercentile(valid_samples, (25, 75), axis=1, dtype=None)
    means = nanmean(valid_samples, axis=1, dtype=None)

    stds = np.zeros(len(valid_samples), dtype=float)
    multiple_values = valid_counts > 1
    if np.any(multiple_values):
        stds[multiple_values] = nanstd(
            valid_samples[multiple_values],
            axis=1,
            ddof=1,
            dtype=None,
        )

    result["median"][valid] = medians
    result["std"][valid] = stds
    result["iqr"][valid] = quartiles[1] - quartiles[0]
    result["mad"][valid] = nanmedian(
        np.abs(valid_samples - medians[:, np.newaxis]),
        axis=1,
        dtype=None,
    )
    result["cv"][valid] = stds / (np.abs(means) + eps)
    return result


def summarize_values(values: np.ndarray | list | tuple) -> dict[str, float | int]:
    """Return count, mean, sample standard deviation, median, and IQR."""
    values = finite_values(values)
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }

    return {
        "n": int(values.size),
        "mean": float(nanmean(values, dtype=None)),
        "std": float(nanstd(values, ddof=1, dtype=None) if values.size > 1 else 0.0),
        "median": float(nanmedian(values, dtype=None)),
        "iqr": float(
            nanpercentile(values, 75, dtype=None)
            - nanpercentile(values, 25, dtype=None)
        ),
    }


def cohen_d(
    control_values: np.ndarray | list | tuple,
    group_values: np.ndarray | list | tuple,
) -> float:
    """Return pooled-standard-deviation Cohen's d for group minus control."""
    control = finite_values(control_values)
    group = finite_values(group_values)
    if control.size < 2 or group.size < 2:
        return np.nan

    control_std = nanstd(control, ddof=1)
    group_std = nanstd(group, ddof=1)
    pooled_var = (
        (control.size - 1) * control_std**2
        + (group.size - 1) * group_std**2
    ) / (control.size + group.size - 2)

    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    return float((nanmean(group) - nanmean(control)) / np.sqrt(pooled_var))


def mean_difference_ci95(
    control_values: np.ndarray | list | tuple,
    group_values: np.ndarray | list | tuple,
) -> tuple[float, float, float]:
    """Return approximate 95% CI for the mean difference group minus control."""
    control = finite_values(control_values)
    group = finite_values(group_values)
    if control.size < 2 or group.size < 2:
        return np.nan, np.nan, np.nan

    difference = float(nanmean(group) - nanmean(control))
    standard_error = np.sqrt(
        nanvar(control, ddof=1) / control.size
        + nanvar(group, ddof=1) / group.size
    )

    if not np.isfinite(standard_error):
        return difference, np.nan, np.nan

    return (
        difference,
        float(difference - 1.96 * standard_error),
        float(difference + 1.96 * standard_error),
    )


def mann_whitney_pvalue(
    control_values: np.ndarray | list | tuple,
    group_values: np.ndarray | list | tuple,
) -> float:
    """Return a two-sided Mann-Whitney p-value, or NaN for invalid samples."""
    control = finite_values(control_values)
    group = finite_values(group_values)
    if control.size == 0 or group.size == 0:
        return np.nan

    try:
        from scipy.stats import mannwhitneyu

        result = mannwhitneyu(control, group, alternative="two-sided", method="auto")
        return float(result.pvalue)
    except (ImportError, ValueError):
        return np.nan


def auc_from_scores(
    control_values: np.ndarray | list | tuple,
    group_values: np.ndarray | list | tuple,
) -> float:
    """Return Mann-Whitney ROC AUC oriented toward the compared group."""
    control = finite_values(control_values)
    group = finite_values(group_values)
    if control.size == 0 or group.size == 0:
        return np.nan

    try:
        from scipy.stats import mannwhitneyu

        statistic = mannwhitneyu(
            group,
            control,
            alternative="two-sided",
            method="auto",
        ).statistic
        return float(statistic / (control.size * group.size))
    except (ImportError, ValueError):
        return np.nan


def best_threshold_sensitivity_specificity_cumulative_sweep(
    control_values: np.ndarray | list | tuple,
    group_values: np.ndarray | list | tuple,
    *,
    evaluate_both_directions: bool = False,
) -> tuple[float, float, float, str]:
    """Find the Youden-optimal threshold with one sorted cumulative sweep."""
    control = finite_values(control_values)
    group = finite_values(group_values)

    if control.size == 0 or group.size == 0:
        return np.nan, np.nan, np.nan, "NA"

    scores = np.concatenate([control, group])
    labels = np.concatenate(
        [
            np.zeros(control.size, dtype=np.int64),
            np.ones(group.size, dtype=np.int64),
        ]
    )
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    values, starts, counts = np.unique(
        sorted_scores,
        return_index=True,
        return_counts=True,
    )

    if values.size == 1:
        return float(values[0]), np.nan, np.nan, "NA"

    group_counts = np.add.reduceat(sorted_labels, starts)
    control_counts = counts - group_counts
    cumulative_group = np.cumsum(group_counts)[:-1]
    cumulative_control = np.cumsum(control_counts)[:-1]
    thresholds = (values[:-1] + values[1:]) / 2.0

    def best_for_direction(direction):
        if direction == ">=":
            true_positive = group.size - cumulative_group
            true_negative = cumulative_control
        else:
            true_positive = cumulative_group
            true_negative = control.size - cumulative_control

        sensitivity = true_positive / group.size
        specificity = true_negative / control.size
        youden = sensitivity + specificity - 1.0
        best_index = int(np.argmax(youden))
        return (
            float(youden[best_index]),
            float(thresholds[best_index]),
            float(sensitivity[best_index]),
            float(specificity[best_index]),
            direction,
        )

    preferred_direction = ">=" if nanmedian(group) >= nanmedian(control) else "<="
    best = best_for_direction(preferred_direction)

    if evaluate_both_directions:
        opposite_direction = "<=" if preferred_direction == ">=" else ">="
        opposite = best_for_direction(opposite_direction)
        if opposite[0] > best[0]:
            best = opposite

    _, threshold, sensitivity, specificity, direction = best
    return threshold, sensitivity, specificity, direction


def overlap_from_cohen_d(d: float | None) -> float:
    """Return the equal-variance Gaussian overlap approximation."""
    if d is None or not np.isfinite(d):
        return np.nan
    return float(1.0 + erf(-abs(float(d)) / (2.0 * sqrt(2.0))))


# Explicit aliases make the intent clear at call sites that use the scalar
# helpers from the original pipeline implementations.
safe_nanmean = nanmean
safe_nanmedian = nanmedian
safe_nanstd = nanstd
safe_nanvar = nanvar
