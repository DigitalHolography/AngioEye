import h5py
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from math_utils import fft, ifft, irfft, nanargmax, nanmax, nanmean, rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.special import jv

from .core.base import ProcessPipeline, ProcessResult, registerPipeline

num_interp_points_t = 128  # Number of temporal points for interpolation
num_interp_points_x = 16  # Number of spatial points for interpolation
pixel_size = 10e-6  # in m
len_seg = 25  # in pixels
nu = 3.5 * 1e-6  # Viscosity in m^2/s
rho = 1060  # Density in kg/m^3
f0 = 1.2
omega_0 = 2 * np.pi * f0
num_harmonics = 10  # Number of harmonics to consider
min_valid_segments = 4  # Minimum number of valid segments in one branch


# v_profile_meas_extraction


def preprocess_v_profile_meas(num_interp_points_x, v_profile):
    v_profile = np.asarray(v_profile, dtype=float)

    valid_mask = np.isfinite(v_profile)
    valid_indices = np.where(valid_mask)[0]

    if valid_indices.size <= 8:
        return np.full(num_interp_points_x, np.nan), np.nan

    x_valid = valid_indices.astype(float)
    v_valid = v_profile[valid_mask]

    x_interp = np.linspace(
        x_valid[0],
        x_valid[-1],
        num_interp_points_x,
    )

    interpolator = interp1d(
        x_valid,
        v_valid,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    v_interp = interpolator(x_interp)

    ratio = (
        (num_interp_points_x - 1)
        / (x_valid[-1] - x_valid[0])
    )

    return np.asarray(v_interp), ratio


def extract_v_profile_meas(dataset, num_interp_points_x):
    # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
    n_t, n_x, n_branches, n_radii = dataset.shape
    dataset_x = np.full((n_t, num_interp_points_x, n_branches, n_radii), np.nan, dtype=float)
    v_profile_fft = np.zeros((n_t, num_interp_points_x // 2 + 1, n_branches, n_radii), dtype=complex)
    v_profile_meas_n1 = np.zeros((n_t, num_interp_points_x, n_branches, n_radii), dtype=float)
    v_profile_meas_dc = np.zeros((n_t, num_interp_points_x, n_branches, n_radii), dtype=float)
    ratio_map = np.full((n_branches, n_radii), np.nan, dtype=float)

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for t_idx in range(n_t):
                v_profile = np.asarray(dataset[t_idx, :, branch_idx, radii_idx]) * 1e-3

                v_interp, ratio = preprocess_v_profile_meas(
                    num_interp_points_x=num_interp_points_x,
                    v_profile=v_profile,
                )
                dataset_x[t_idx, :, branch_idx, radii_idx] = v_interp

                if not np.isfinite(ratio) or not np.all(np.isfinite(v_interp)):
                    continue

                v_fft = rfft(v_interp, n=num_interp_points_x, axis=0)
                v_profile_fft[t_idx, :, branch_idx, radii_idx] = v_fft

                v_meas = np.zeros_like(v_fft)
                v_meas[1] = v_fft[1]
                v_profile_meas_n1[t_idx, :, branch_idx, radii_idx] = irfft(v_meas)

                v_meas_dc = np.zeros_like(v_fft)
                v_meas_dc[0] = v_fft[0]
                v_profile_meas_dc[t_idx, :, branch_idx, radii_idx] = irfft(v_meas_dc)

                ratio_map[branch_idx, radii_idx] = ratio

    return dataset_x, v_profile_fft, v_profile_meas_n1, v_profile_meas_dc, ratio_map


def decompose_velocity_profile(dataset):

    dataset = np.asarray(dataset, dtype=float)
    dataset_flipped = dataset[:, ::-1, :, :]

    dataset_x_symmetric = 0.5 * (dataset + dataset_flipped)
    dataset_x_antisymmetric = 0.5 * (dataset - dataset_flipped)

    invalid_profiles = np.all(np.isnan(dataset), axis=1,keepdims=True)

    dataset_x_symmetric = np.where(invalid_profiles,np.nan,dataset_x_symmetric)

    dataset_x_antisymmetric = np.where(invalid_profiles,np.nan,dataset_x_antisymmetric)

    return (dataset_x_symmetric, dataset_x_antisymmetric)


# v_profile_meas_extraction


def preprocess_v_pulse_meas(num_interp_points_t, v_pulse):
    v_pulse = np.asarray(v_pulse, dtype=float)

    valid_mask = np.isfinite(v_pulse)
    valid_indices = np.where(valid_mask)[0]

    if valid_indices.size < 2:
        return np.full(num_interp_points_t, np.nan)

    t_valid = valid_indices.astype(float)
    v_valid = v_pulse[valid_mask]

    t_interp = np.linspace(
        t_valid[0],
        t_valid[-1],
        num_interp_points_t,
    )

    interpolator = interp1d(
        t_valid,
        v_valid,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    return np.asarray(interpolator(t_interp))


def extract_v_pulse_meas(dataset, num_interp_points_t):
    # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
    n_t, n_x, n_branches, n_radii = dataset.shape
    v_pulse_fft = np.zeros((num_interp_points_t // 2 + 1, n_x, n_branches, n_radii), dtype=complex)
    v_pulse_meas_n1 = np.zeros((num_interp_points_t, n_x, n_branches, n_radii), dtype=float)
    v_pulse_meas_dc = np.zeros((num_interp_points_t, n_x, n_branches, n_radii), dtype=float)

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for x_idx in range(n_x):
                v_pulse = np.asarray(dataset[:, x_idx, branch_idx, radii_idx])

                v_interp = preprocess_v_pulse_meas(
                    num_interp_points_t=num_interp_points_t,
                    v_pulse=v_pulse,
                )

                if not np.all(np.isfinite(v_interp)):
                    continue
                v_fft = rfft(v_interp, n=num_interp_points_t, axis=0)
                v_pulse_fft[:, x_idx, branch_idx, radii_idx] = v_fft

                v_meas = np.zeros_like(v_fft)
                v_meas[1] = v_fft[1]
                v_pulse_meas_n1[:, x_idx, branch_idx, radii_idx] = irfft(v_meas)

                v_meas_dc = np.zeros_like(v_fft)
                v_meas_dc[0] = v_fft[0]
                v_pulse_meas_dc[:, x_idx, branch_idx, radii_idx] = irfft(v_meas_dc)

    return v_pulse_fft, v_pulse_meas_n1, v_pulse_meas_dc


# forward_modeling


def _abel_cell_integral(x_abs, r_left, r_right):
    if x_abs >= r_right:
        return 0.0
    lower = max(r_left, x_abs)
    upper = r_right
    lower_term = np.sqrt(max(lower**2 - x_abs**2, 0.0))
    upper_term = np.sqrt(max(upper**2 - x_abs**2, 0.0))
    return 1.0 * (upper_term - lower_term)


def apply_abel_projection(L):
    x_grid = np.linspace(1 / ((L - 1) * 2), 1, L // 2)

    r_edges = np.linspace(0, 1.1, L // 2 + 1)

    K_block = np.zeros((L // 2, L // 2))

    for i, x in enumerate(x_grid):
        x_abs = abs(x)

        for j in range(L // 2):
            K_block[i, j] = _abel_cell_integral(
                x_abs,
                r_edges[j],
                r_edges[j + 1],
            )
    A = np.fliplr(np.flipud(K_block))
    B = np.fliplr(A)
    C = np.fliplr(K_block)
    D = K_block
    K = np.block([[A, B], [C, D]])

    return K


def projected_parabola_model(x, A, x0, y0, K):
    parabola = A * (x - x0) ** 2 + y0
    parabola = np.where(parabola > 0, parabola, 0.0)

    return K @ parabola


def projected_parabola_fit(V):
    segment_data = {}

    L = V.shape[1]
    K = apply_abel_projection(L)

    x = np.arange(L)

    for branch_index in range(V.shape[2]):
        for circle_index in range(V.shape[3]):
            profile_complex = V[0, :, branch_index, circle_index]
            profile = np.abs(profile_complex) * 1.04

            if np.all(profile == 0):
                continue

            try:
                A_guess = -0.08
                x0_guess = 7.5
                y0_guess = np.max(profile)

                def fit_func(x_dummy, A, x0, y0):
                    return np.abs(projected_parabola_model(x_dummy, A, x0, y0, K))

                bounds = (
                    [-np.inf, 7.5, 0],  # A<0, x0>0, y0>0
                    [0, 7.51, np.inf],
                )
                popt, pcov = curve_fit(
                    fit_func,
                    x,
                    profile,
                    p0=[A_guess, x0_guess, y0_guess],
                    bounds=bounds,
                )

                A_fit, x0_fit, y0_fit = popt

                r0_fit = np.sqrt(-y0_fit / A_fit)

                segment_data[(branch_index, circle_index)] = {
                    "r0": r0_fit,
                    "y0": y0_fit,
                    "x0": x0_fit,
                    "A": A_fit,
                }

                # print(
                #     f"branch={branch_index}, "
                #     f"circle={circle_index}, "
                #     f"r0={r0_fit:.4f}, "
                #     f"x0={x0_fit:.4f}, "
                #     f"y0={y0_fit:.4f}, "
                #     f"A={A_fit:.4f}"
                # )

            except Exception as e:
                print(f"Fit failed for branch={branch_index}, circle={circle_index}")
                print(e)

    return segment_data


def womersley_Bn(L, R0, nu, omega_n, x0, r0):
    x = np.arange(L)
    x_norm = (x - x0) / r0

    alpha_n = R0 * np.sqrt(omega_n / nu)
    lam = np.exp(1j * 3 * np.pi / 4) * alpha_n

    Bn = 1 - jv(0, lam * np.abs(x_norm)) / jv(0, lam)
    Bn[np.abs(x_norm) > 1] = 0.0

    return Bn.astype(complex)

def compute_Cn(Vn, KBn):
    
    numerator = np.sum(np.conj(KBn) * Vn)
    
    denominator = np.sum(np.abs(KBn ** 2))
        
    return numerator / denominator


def compute_Qn(R0, nu, omega_n, Cn_n):
    alpha_n = R0 * np.sqrt(omega_n / nu)

    lam = np.exp(1j * 3 * np.pi / 4) * alpha_n

    Tn = 1 - 2 * jv(1, lam) / (lam * jv(0, lam))

    Qn = np.pi * (R0**2) * Cn_n * Tn

    return Qn


def compute_tau_n(R0, nu, omega_n, Cn_n, rho):
    alpha_n = R0 * np.sqrt(omega_n / nu)

    lam = np.exp(1j * 3 * np.pi / 4) * alpha_n

    Sn = lam * jv(1, lam) / jv(0, lam)

    tau_n = rho * nu * Cn_n * Sn / R0

    return tau_n


def generate_harmonic_flow_profile(V, segment_data, ratio_map):
    v_model_fft = np.zeros((V.shape[0], V.shape[1], V.shape[2], V.shape[3]), dtype=complex)
    C_n = np.zeros((V.shape[0], V.shape[2], V.shape[3]), dtype=complex)
    Q_n = np.zeros((V.shape[0], V.shape[2], V.shape[3]), dtype=complex)
    Tau_n = np.zeros((V.shape[0], V.shape[2], V.shape[3]), dtype=complex)
    for branch in range(V.shape[2]):
        for circle in range(V.shape[3]):
            if (branch, circle) not in segment_data:
                continue

            r0 = segment_data[(branch, circle)]["r0"]
            y0 = segment_data[(branch, circle)]["y0"]
            x0 = segment_data[(branch, circle)]["x0"]
            A = segment_data[(branch, circle)]["A"]
            dx = ratio_map[branch, circle]

            matrix = V[:, :, branch, circle]
            x = np.arange(matrix.shape[1])

            L = len(x)
            K = apply_abel_projection(L)
            R0 = r0 * pixel_size / dx

            threshold = -1
            model_0 = projected_parabola_model(x, A, x0, y0, K)
            skip_segment = model_0[0] < threshold or model_0[-1] < threshold
            if model_0[0] < threshold or model_0[-1] < threshold:
                print(f"Skip branch={branch}, circle={circle} for Womersley modeling.")
                continue

            if (
                not np.isfinite(R0) or R0 <= 0 or R0 > 1e-4  # 100 μm radius
            ):
                print(
                    f"Reject branch={branch}, circle={circle}, R0={R0} for Womersley modeling."
                )
                continue

            Cn = np.zeros(V.shape[0], dtype=complex)
            Qn = np.zeros(V.shape[0], dtype=complex)
            taun = np.zeros(V.shape[0], dtype=complex)
            Cn[0] = 1.0

            for n in range(num_harmonics):
                Vn = np.array(matrix[n], dtype=complex) / num_interp_points_t

                if n == 0:
                    model = (
                        projected_parabola_model(x, A, x0, y0, K) / num_interp_points_t
                    )

                else:
                    if skip_segment:
                        continue

                    omega_n = n * omega_0
                    Bn = womersley_Bn(L, R0, nu, omega_n, x0, r0)
                    KBn = K @ Bn
                    Cn[n] = compute_Cn(Vn, KBn)
                    Qn[n] = compute_Qn(R0, nu, omega_n, Cn[n])
                    taun[n] = compute_tau_n(R0, nu, omega_n, Cn[n], rho)
                    model = Cn[n] * KBn

                v_model_fft[n, :, branch, circle] = model
                C_n[n, branch, circle] = Cn[n]
                Q_n[n, branch, circle] = Qn[n]
                Tau_n[n, branch, circle] = taun[n]

    v_model = irfft(v_model_fft * num_interp_points_t, axis=0)

    return (
        v_model,
        v_model_fft,
        C_n,
        Q_n,
        Tau_n,
    )


def fit_antisymmetric_curve(dataset_x_antisymmetric, v_model, ratio_map):
    n_t, n_x, n_branches, n_radii = dataset_x_antisymmetric.shape
    disp = np.zeros((n_t, n_branches, n_radii), dtype=float)
    anti_model = np.zeros_like(dataset_x_antisymmetric, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            ratio = ratio_map[branch, circle]

            if not np.isfinite(ratio) or ratio <= 0:
                continue

            dx = pixel_size / ratio
            symmetric_gradient = np.gradient(v_model[:, :, branch, circle], dx, axis=1, edge_order=2)

            for t_idx in range(n_t):
                anti_profile = dataset_x_antisymmetric[t_idx, :, branch, circle]
                gradient_profile = symmetric_gradient[t_idx]
                valid = np.isfinite(anti_profile) & np.isfinite(gradient_profile)

                if np.sum(valid) < 3:
                    continue

                numerator = np.sum(gradient_profile[valid] * anti_profile[valid])
                denominator = np.sum(gradient_profile[valid] ** 2)

                if not np.isfinite(denominator) or denominator <= 0:
                    continue

                delta = -numerator / denominator
                disp[t_idx, branch, circle] = delta
                anti_model[t_idx, :, branch, circle] = -delta * gradient_profile

    disp_scale = np.max(np.abs(disp), axis=0, keepdims=True)
    disp = np.divide(disp, disp_scale, out=np.zeros_like(disp), where=disp_scale > 0)

    return disp, anti_model


def evaluate_anti_model(disp, num_harmonics=3, branch_idx=0, num_examples=3, exclude_last=5, isolated_threshold=0.9, isolation_radius=2, distribution_path="anti_disp_distribution.png", jump_analysis_path="anti_disp_jump_analysis.png"):
    n_t, n_branches, n_radii = disp.shape
    segment_valid = np.any(np.isfinite(disp) & (disp != 0), axis=0)
    disp_valid = np.isfinite(disp) & segment_valid[None, :, :]
    phase = 2 * np.pi * np.arange(n_t) / n_t
    design_matrix = np.column_stack([np.ones(n_t)] + [f(phase * n) for n in range(1, num_harmonics + 1) for f in (np.cos, np.sin)])
    disp_smooth = np.full_like(disp, np.nan, dtype=float)
    disp_residual = np.full_like(disp, np.nan, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            time_valid = disp_valid[:, branch, circle]

            if np.sum(time_valid) < design_matrix.shape[1]:
                continue

            coefficients = np.linalg.lstsq(design_matrix[time_valid], disp[time_valid, branch, circle], rcond=None)[0]
            disp_smooth[:, branch, circle] = design_matrix @ coefficients
            disp_residual[time_valid, branch, circle] = disp[time_valid, branch, circle] - disp_smooth[time_valid, branch, circle]

    temporal_difference = np.full((n_t - 1, n_branches, n_radii), np.nan, dtype=float)
    difference_valid = disp_valid[1:] & disp_valid[:-1] & np.isfinite(disp_residual[1:]) & np.isfinite(disp_residual[:-1])
    raw_difference = np.diff(disp_residual, axis=0)
    temporal_difference[difference_valid] = raw_difference[difference_valid]
    fitting_valid = difference_valid.copy()

    if exclude_last > 0:
        fitting_valid[max(0, n_t - 1 - exclude_last):] = False

    temporal_jump_probability = np.full_like(temporal_difference, np.nan, dtype=float)
    jump_parameters = np.full((n_branches, n_radii, 3), np.nan, dtype=float)

    def gaussian_density(values, std):
        return np.exp(-0.5 * (values / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    def fit_gaussian_mixture(values):
        values = values[np.isfinite(values)]

        if values.size < 10:
            return None

        scale = np.std(values)

        if not np.isfinite(scale) or scale <= 1e-12:
            return None

        normal_std = max(0.5 * scale, 1e-12)
        jump_std = max(2.0 * scale, 2.0 * normal_std)
        jump_probability = 0.1

        for _ in range(200):
            normal_density = gaussian_density(values, normal_std)
            jump_density = gaussian_density(values, jump_std)
            denominator = (1.0 - jump_probability) * normal_density + jump_probability * jump_density
            responsibility = np.divide(jump_probability * jump_density, denominator, out=np.zeros_like(values), where=denominator > 0)
            normal_weight = np.sum(1.0 - responsibility)
            jump_weight = np.sum(responsibility)
            new_jump_probability = np.mean(responsibility)
            new_normal_std = np.sqrt(np.sum((1.0 - responsibility) * values**2) / max(normal_weight, 1e-12))
            new_jump_std = np.sqrt(np.sum(responsibility * values**2) / max(jump_weight, 1e-12))

            if new_normal_std > new_jump_std:
                new_normal_std, new_jump_std = new_jump_std, new_normal_std
                new_jump_probability = 1.0 - new_jump_probability

            new_jump_probability = np.clip(new_jump_probability, 1e-6, 1.0 - 1e-6)
            new_normal_std = max(new_normal_std, 1e-12)
            new_jump_std = max(new_jump_std, new_normal_std)
            old_parameters = np.array([jump_probability, normal_std, jump_std])
            new_parameters = np.array([new_jump_probability, new_normal_std, new_jump_std])
            jump_probability, normal_std, jump_std = new_parameters

            if np.allclose(old_parameters, new_parameters, rtol=1e-6, atol=1e-8):
                break

        return jump_probability, normal_std, jump_std

    for branch in range(n_branches):
        for circle in range(n_radii):
            circle_fitting_valid = fitting_valid[:, branch, circle]
            circle_values = temporal_difference[circle_fitting_valid, branch, circle]
            parameters = fit_gaussian_mixture(circle_values)

            if parameters is None:
                continue

            jump_probability, normal_std, jump_std = parameters
            jump_parameters[branch, circle] = [jump_probability, normal_std, jump_std]
            values = temporal_difference[circle_fitting_valid, branch, circle]
            normal_density = gaussian_density(values, normal_std)
            jump_density = gaussian_density(values, jump_std)
            denominator = (1.0 - jump_probability) * normal_density + jump_probability * jump_density
            temporal_jump_probability[circle_fitting_valid, branch, circle] = np.divide(jump_probability * jump_density, denominator, out=np.full_like(values, np.nan), where=denominator > 0)

    branch_idx = int(np.clip(branch_idx, 0, n_branches - 1))
    residual_values = disp_residual[np.isfinite(disp_residual) & disp_valid]
    difference_values = temporal_difference[fitting_valid & np.isfinite(temporal_difference)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if residual_values.size > 0:
        axes[0].hist(residual_values, bins=60, density=True, alpha=0.7)

    axes[0].set_xlabel("Normalized displacement residual")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("All-point residual distribution")

    if difference_values.size > 0:
        axes[1].hist(difference_values, bins=60, density=True, alpha=0.7)

    axes[1].set_xlabel("Temporal residual difference")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title(f"Temporal differences, last {exclude_last} excluded")

    circle_indices = np.arange(n_radii)
    branch_parameters = jump_parameters[branch_idx]
    axes[2].plot(circle_indices, branch_parameters[:, 1], "o-", label="normal std")
    axes[2].plot(circle_indices, branch_parameters[:, 2], "o-", label="jump std")
    axes[2].set_xlabel("Circle")
    axes[2].set_ylabel("Standard deviation")
    axes[2].set_title(f"Circle-specific mixture parameters, branch {branch_idx}")

    probability_axis = axes[2].twinx()
    probability_axis.plot(circle_indices, branch_parameters[:, 0], "s--", color="tab:red", label="jump probability")
    probability_axis.set_ylabel("Mixture jump probability")
    probability_axis.set_ylim(0, 1)

    lines_1, labels_1 = axes[2].get_legend_handles_labels()
    lines_2, labels_2 = probability_axis.get_legend_handles_labels()
    axes[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    fig.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    branch_probability = temporal_jump_probability[:, branch_idx, :]
    branch_valid = np.isfinite(branch_probability)
    valid_count = np.sum(branch_valid, axis=1)
    probability_sum = np.nansum(branch_probability, axis=1)
    mean_probability = np.divide(probability_sum, valid_count, out=np.full(n_t - 1, np.nan), where=valid_count > 0)
    isolated_mask = np.zeros_like(branch_valid, dtype=bool)

    for circle in range(n_radii):
        for time_idx in range(n_t - 1):
            probability = branch_probability[time_idx, circle]

            if not np.isfinite(probability) or probability < isolated_threshold:
                continue

            start = max(0, time_idx - isolation_radius)
            stop = min(n_t - 1, time_idx + isolation_radius + 1)
            neighbour_probability = branch_probability[start:stop, circle].copy()
            neighbour_probability[time_idx - start] = np.nan
            finite_neighbours = neighbour_probability[np.isfinite(neighbour_probability)]

            if finite_neighbours.size == 0 or np.max(finite_neighbours) < isolated_threshold:
                isolated_mask[time_idx, circle] = True

    candidate_probability = np.where(branch_valid, branch_probability, -np.inf)
    finite_candidates = np.flatnonzero(np.isfinite(candidate_probability.ravel()))
    highest_points = []

    if finite_candidates.size > 0:
        sorted_candidates = finite_candidates[np.argsort(candidate_probability.ravel()[finite_candidates])[::-1]]
        highest_points = [np.unravel_index(index, candidate_probability.shape) for index in sorted_candidates[:num_examples]]

    isolated_candidates = np.flatnonzero(isolated_mask.ravel())
    isolated_points = []

    if isolated_candidates.size > 0:
        sorted_isolated = isolated_candidates[np.argsort(branch_probability.ravel()[isolated_candidates])[::-1]]
        isolated_points = [np.unravel_index(index, branch_probability.shape) for index in sorted_isolated[:num_examples]]

    selected_points = []
    selected_labels = []

    for point in highest_points:
        selected_points.append(point)
        selected_labels.append("highest probability")

    for point in isolated_points:
        if point in selected_points:
            index = selected_points.index(point)
            selected_labels[index] = "highest probability, isolated"
        else:
            selected_points.append(point)
            selected_labels.append("isolated")

    num_selected = len(selected_points)
    fig = plt.figure(figsize=(12, 6 + 3 * max(num_selected, 1)))
    grid = fig.add_gridspec(max(num_selected, 1) + 1, 2, height_ratios=[1.5] + [1] * max(num_selected, 1))

    heatmap_axis = fig.add_subplot(grid[0, 0])
    heatmap_cmap = plt.get_cmap("magma").copy()
    heatmap_cmap.set_bad(color="lightgray")
    heatmap = heatmap_axis.imshow(np.ma.masked_invalid(branch_probability), origin="lower", aspect="auto", vmin=0, vmax=1, cmap=heatmap_cmap)
    heatmap_axis.set_xlabel("Circle")
    heatmap_axis.set_ylabel("Time transition")
    heatmap_axis.set_title(f"Temporal jump probability, branch {branch_idx}")
    fig.colorbar(heatmap, ax=heatmap_axis, label="Jump probability")

    for (time_idx, circle_idx), label in zip(selected_points, selected_labels):
        marker = "o" if "isolated" in label else "x"
        color = "cyan" if "isolated" in label else "lime"
        heatmap_axis.plot(circle_idx, time_idx, marker=marker, color=color, markersize=7, markeredgewidth=1.5)

    mean_axis = fig.add_subplot(grid[0, 1])
    mean_axis.plot(np.arange(n_t - 1), mean_probability, color="tab:blue")
    mean_axis.axhline(isolated_threshold, color="tab:red", linestyle="--", label=f"threshold = {isolated_threshold:.2f}")

    if exclude_last > 0:
        excluded_start = max(0, n_t - 1 - exclude_last)
        mean_axis.axvspan(excluded_start, n_t - 2, color="gray", alpha=0.25, label="excluded")

    mean_axis.set_xlabel("Time transition")
    mean_axis.set_ylabel("Mean jump probability")
    mean_axis.set_ylim(0, 1)
    mean_axis.set_title("Mean probability across valid circles")
    mean_axis.legend()

    if selected_points:
        time_axis = np.arange(n_t)

        for row, ((time_idx, circle_idx), label) in enumerate(zip(selected_points, selected_labels), start=1):
            axis = fig.add_subplot(grid[row, :])
            axis.plot(time_axis, disp[:, branch_idx, circle_idx], label="disp")
            axis.plot(time_axis, disp_smooth[:, branch_idx, circle_idx], label="disp smooth")
            axis.axvline(time_idx, color="red", linestyle="--")
            axis.axvline(time_idx + 1, color="red", linestyle="--")
            probability = branch_probability[time_idx, circle_idx]
            normal_std = jump_parameters[branch_idx, circle_idx, 1]
            jump_std = jump_parameters[branch_idx, circle_idx, 2]
            axis.set_ylabel("Normalized displacement")
            axis.set_title(f"{label}: circle {circle_idx}, transition {time_idx} → {time_idx + 1}, probability = {probability:.4f}, normal std = {normal_std:.4f}, jump std = {jump_std:.4f}")
            axis.legend()

        axis.set_xlabel("Time point")

    else:
        axis = fig.add_subplot(grid[1, :])
        axis.text(0.5, 0.5, "No valid jump candidates", ha="center", va="center")
        axis.set_axis_off()

    fig.tight_layout()
    fig.savefig(jump_analysis_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return disp_smooth, disp_residual, temporal_difference, temporal_jump_probability, jump_parameters
    

@registerPipeline(name="WomersleyModeling")
class WomersleyModeling(ProcessPipeline):
    description = "Womersley Modeling Pipeline"

    v_profile_path = "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"
    b_period_path = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file: h5py.File) -> ProcessResult:
        """
        Executes the Womersley Modeling pipeline.
        """
        obj = h5file[self.v_profile_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.v_profile_path}, but found {type(obj)}"
            )
        dataset = np.asarray(obj[:], dtype=float)

        obj = h5file[self.b_period_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.b_period_path}, but found {type(obj)}"
            )
        # b_period = np.mean(obj[:])
        # print(f"b_period: {b_period}")

        dataset_x, v_profile_fft, v_profile_meas_n1, v_profile_meas_dc, ratio_map = (
            extract_v_profile_meas(
                dataset=dataset,
                num_interp_points_x=num_interp_points_x,
            )
        )

        dataset_x_symmetric, dataset_x_antisymmetric = decompose_velocity_profile(dataset_x) 

        v_pulse_fft, v_pulse_meas_n1, v_pulse_meas_dc = extract_v_pulse_meas(
            dataset=dataset_x_symmetric,
            num_interp_points_t=num_interp_points_t,
        )

        segment_data = projected_parabola_fit(v_pulse_fft)

        v_model, v_model_fft, C_n, Q_n, Tau_n = generate_harmonic_flow_profile(v_pulse_fft, segment_data, ratio_map)

        disp, anti_model = fit_antisymmetric_curve(dataset_x_antisymmetric, v_model, ratio_map)
        disp_smooth, disp_residual, temporal_difference, temporal_jump_probability, jump_parameters = evaluate_anti_model(disp, branch_idx=0)

        metrics: dict = {}
        metrics["dataset_x"] = np.asarray(dataset_x)
        metrics["dataset_x_symmetric"] = np.asarray(dataset_x_symmetric)
        metrics["dataset_x_antisymmetric"] = np.asarray(dataset_x_antisymmetric)
        metrics["v_profile_fft"] = np.asarray(v_profile_fft)
        metrics["v_profile_meas_n1"] = np.asarray(v_profile_meas_n1)
        metrics["v_profile_meas_dc"] = np.asarray(v_profile_meas_dc)
        metrics["v_pulse_fft"] = np.asarray(v_pulse_fft)
        metrics["v_pulse_meas_n1"] = np.asarray(v_pulse_meas_n1)
        metrics["v_pulse_meas_dc"] = np.asarray(v_pulse_meas_dc)
        metrics["v_model"] = np.asarray(v_model)
        metrics["v_model_fft"] = np.asarray(v_model_fft)
        metrics["C_n"] = np.asarray(C_n)
        metrics["Q_n"] = np.asarray(Q_n)
        metrics["Tau_n"] = np.asarray(Tau_n)
        metrics["disp"] = np.asarray(disp)
        metrics["anti_model"] = np.asarray(anti_model)


        return ProcessResult(metrics=metrics)
