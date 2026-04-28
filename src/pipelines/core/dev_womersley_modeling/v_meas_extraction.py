import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d


def preprocess_and_interpolate(x_coords, v_profile_segment_onebeat, num_interp_points):
    # v_profile_segment_onebeat is expected to be a 2D array with shape (time, space).
    valid_mask = ~np.isnan(v_profile_segment_onebeat).any(axis=0)

    x_valid = x_coords[valid_mask]
    v_valid = v_profile_segment_onebeat[valid_mask]

    # Perform spatial interpolation
    interpolator = interp1d(
        x_valid,
        v_valid,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )

    x_interp = np.linspace(np.min(x_valid), np.max(x_valid), num_interp_points)
    v_interp = interpolator(x_interp)

    return x_interp, v_interp


def compute_fft(v_profile_segment_onebeat, n_fft, axis=0):
    v_profile_fft = fft(v_profile_segment_onebeat, n=n_fft, axis=axis)
    return v_profile_fft


def extract_harmonic_profile(v_profile_fft, n_harmonic, axis=0):
    slices = [slice(None)] * v_profile_fft.ndim
    slices[axis] = n_harmonic
    v_meas = v_profile_fft[tuple(slices)]

    return v_meas
