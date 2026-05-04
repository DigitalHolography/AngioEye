import numpy as np
from scipy.interpolate import interp1d


def preprocess_and_interpolate(num_interp_points, v_segment_onebeat):
    valid_mask = ~np.isnan(v_segment_onebeat)
    valid_indices = np.where(valid_mask)[0]
    valid_count = np.sum(valid_mask)

    if valid_count <= 8:
        print(f"Warning: Only {valid_count} valid points found. Skipping...")
        return np.zeros(num_interp_points)

    min_idx = valid_indices[0]
    max_idx = valid_indices[-1]

    v_valid = v_segment_onebeat[min_idx : max_idx + 1].copy()
    v_valid[0], v_valid[-1] = 0.0, 0.0
    x_valid = np.arange(len(v_valid))
    x_interp = np.linspace(0, len(v_valid) - 1, num=num_interp_points)

    interpolator = interp1d(
        x_valid,
        v_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )
    v_interp = interpolator(x_interp)

    return np.asanyarray(v_interp)


def extract_v_profile_meas(dataset, num_interp_points, n_harmonic):
    # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
    n_t, n_x, n_branches, n_radii = dataset.shape
    v_profile_fft = np.zeros(
        (n_t, num_interp_points, n_branches, n_radii), dtype=complex
    )
    v_profile_meas = np.zeros(
        (n_t, num_interp_points, n_branches, n_radii), dtype=complex
    )

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for t_idx in range(n_t):
                v_segment_onebeat = np.asarray(dataset[t_idx, :, branch_idx, radii_idx])

                v_interp = preprocess_and_interpolate(
                    num_interp_points=num_interp_points,
                    v_segment_onebeat=v_segment_onebeat,
                )

                v_fft = np.fft.fft(np.asarray(v_interp), n=num_interp_points)
                v_profile_fft[t_idx, :, branch_idx, radii_idx] = v_fft

                v_meas = np.zeros_like(v_fft)
                v_meas[1] = v_fft[n_harmonic]
                v_meas[-1] = v_fft[-n_harmonic]
                v_profile_meas[t_idx, :, branch_idx, radii_idx] = np.fft.ifft(v_meas)

    return v_profile_fft, v_profile_meas
