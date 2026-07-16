"""Reusable Fourier and finite-harmonic reconstruction helpers."""

from __future__ import annotations

import numpy as np


def _axis_slice(ndim: int, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    selection = [slice(None)] * ndim
    selection[axis] = slice(start, stop)
    return tuple(selection)


def rfft_normalized(
    signal: np.ndarray,
    axis: int = -1,
    dtype: np.dtype | type | None = np.float32,
) -> np.ndarray:
    """Return the real FFT normalized by the number of samples.

    The normalization matches the convention used throughout the waveform
    pipelines: ``V = np.fft.rfft(signal) / n``.  ``axis`` may be any signal
    axis, including a batch axis layout such as ``(time, waveform)``.
    """
    values = np.asarray(signal, dtype=dtype)
    n = values.shape[axis]
    if n <= 0:
        raise ValueError("The Fourier transform axis must contain samples")
    return np.fft.rfft(values, axis=axis) / float(n)


def rfft(
    signal: np.ndarray,
    n: int | None = None,
    axis: int = -1,
    dtype: np.dtype | type | None = np.float32,
) -> np.ndarray:
    """Return the unnormalized real FFT while preserving input precision."""
    values = np.asarray(signal, dtype=dtype)
    return np.fft.rfft(values, n=n, axis=axis)


def irfft(
    coefficients: np.ndarray,
    n: int | None = None,
    axis: int = -1,
) -> np.ndarray:
    """Return the inverse real FFT."""
    return np.fft.irfft(np.asarray(coefficients), n=n, axis=axis)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Return real-FFT sample frequencies."""
    return np.fft.rfftfreq(n, d=d)


def fft(
    signal: np.ndarray,
    axis: int = -1,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Return the complex FFT while preserving complex-valued inputs."""
    values = np.asarray(signal, dtype=dtype)
    return np.fft.fft(values, axis=axis)


def ifft(coefficients: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the inverse complex FFT."""
    return np.fft.ifft(np.asarray(coefficients), axis=axis)


def truncate_harmonics(
    coefficients: np.ndarray,
    max_harmonic: int,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep harmonics ``0..H`` and return ``(kept, full_truncated, H)``.

    ``kept`` contains only the retained harmonics.  ``full_truncated`` keeps
    the original FFT shape and is suitable for inverse transformation.
    """
    values = np.asarray(coefficients)
    n_harmonics = values.shape[axis]
    if n_harmonics == 0:
        raise ValueError("The Fourier coefficient axis must not be empty")
    if max_harmonic < 0:
        raise ValueError("max_harmonic must be non-negative")

    H = min(int(max_harmonic), n_harmonics - 1)
    kept = np.array(values[_axis_slice(values.ndim, axis, 0, H + 1)], copy=True)
    truncated = np.zeros_like(values)
    truncated[_axis_slice(values.ndim, axis, 0, H + 1)] = kept
    return kept, truncated, H


def irfft_normalized(
    coefficients: np.ndarray,
    n: int,
    axis: int = -1,
) -> np.ndarray:
    """Reconstruct a signal from coefficients normalized by sample count."""
    if n <= 0:
        raise ValueError("n must be positive")
    values = np.asarray(coefficients)
    return np.fft.irfft(values * float(n), n=n, axis=axis)


def harmonic_pack(
    signal: np.ndarray,
    max_harmonic: int,
    axis: int = -1,
    dtype: np.dtype | type | None = np.float32,
) -> dict[str, np.ndarray | int | None]:
    """Compute full and truncated normalized Fourier representations.

    The returned keys intentionally match the existing waveform metric
    representation: ``V`` (retained coefficients), ``H`` (highest retained
    harmonic), ``vb`` (reconstruction), and ``Vfull`` (full normalized FFT).
    The function supports one waveform or a batch of waveforms.
    """
    values = np.asarray(signal, dtype=dtype)
    n = values.shape[axis]
    if n < 2:
        return {"V": None, "H": 0, "vb": None, "Vfull": None}

    full = rfft_normalized(values, axis=axis)
    kept, truncated, H = truncate_harmonics(full, max_harmonic, axis=axis)
    reconstructed = irfft_normalized(truncated, n=n, axis=axis)
    return {"V": kept, "H": H, "vb": reconstructed, "Vfull": full}
