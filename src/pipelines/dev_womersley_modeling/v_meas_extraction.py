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

    v_valid = v_segment_onebeat[min_idx : max_idx + 1]
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
