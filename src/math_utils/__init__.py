"""Shared numerical utilities used across AngioEye.

The package is deliberately independent from the pipeline framework so that
numerical operations can be reused by pipelines, post-processing, and tests.
"""

from .fourier import (
    fft,
    harmonic_pack,
    irfft,
    irfft_normalized,
    ifft,
    rfft_normalized,
    rfft,
    rfftfreq,
    truncate_harmonics,
)
from .statistics import (
    DEFAULT_FLOAT_DTYPE,
    nanargmax,
    nanargmin,
    nanmax,
    nanmad,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanstd,
    nansum,
    nancv,
    nanvar,
    safe_nanmean,
    safe_nanmedian,
    safe_nanstd,
    safe_nanvar,
)

__all__ = [
    "harmonic_pack",
    "fft",
    "DEFAULT_FLOAT_DTYPE",
    "nanargmax",
    "nanargmin",
    "irfft_normalized",
    "irfft",
    "ifft",
    "nanmax",
    "nanmad",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanstd",
    "nansum",
    "nancv",
    "nanvar",
    "rfft_normalized",
    "rfft",
    "rfftfreq",
    "safe_nanmean",
    "safe_nanmedian",
    "safe_nanstd",
    "safe_nanvar",
    "truncate_harmonics",
]
