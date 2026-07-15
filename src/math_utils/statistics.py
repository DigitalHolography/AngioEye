"""Warning-free NaN-aware statistical reductions.

NumPy's NaN reductions are the right primitives for this project, but several
of them emit warnings for empty or all-NaN slices.  These wrappers suppress
those expected warnings and return NaN for reductions that have no finite
input.  They accept ``axis`` and ``keepdims`` in the same spirit as NumPy.
"""

from __future__ import annotations

import warnings

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


# Explicit aliases make the intent clear at call sites that use the scalar
# helpers from the original pipeline implementations.
safe_nanmean = nanmean
safe_nanmedian = nanmedian
safe_nanstd = nanstd
safe_nanvar = nanvar
