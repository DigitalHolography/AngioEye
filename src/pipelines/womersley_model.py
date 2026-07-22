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


def evaluate_anti_model(disp, num_harmonics=3, figure_path="anti_disp_distribution.png"):
    n_t, n_branches, n_radii = disp.shape
    segment_valid = np.any(np.isfinite(disp) & (disp != 0), axis=0)
    disp_valid = np.isfinite(disp) & segment_valid[None, :, :]
    phase = 2 * np.pi * np.arange(n_t) / n_t
    design_matrix = np.column_stack([np.ones(n_t)] + [f(phase * n) for n in range(1, num_harmonics + 1) for f in (np.cos, np.sin)])
    disp_smooth = np.zeros_like(disp)
    disp_residual = np.zeros_like(disp)

    for branch in range(n_branches):
        for circle in range(n_radii):
            time_valid = disp_valid[:, branch, circle]

            if np.sum(time_valid) < design_matrix.shape[1]:
                continue

            coefficients = np.linalg.lstsq(design_matrix[time_valid], disp[time_valid, branch, circle], rcond=None)[0]
            disp_smooth[:, branch, circle] = design_matrix @ coefficients
            disp_residual[time_valid, branch, circle] = disp[time_valid, branch, circle] - disp_smooth[time_valid, branch, circle]

    residual_values = disp_residual[disp_valid]
    temporal_difference = np.zeros((n_t - 1, n_branches, n_radii), dtype=float)
    difference_valid = disp_valid[1:] & disp_valid[:-1]
    raw_difference = np.diff(disp_residual, axis=0)
    temporal_difference[difference_valid] = raw_difference[difference_valid]
    difference_values = temporal_difference[difference_valid]
    temporal_jump_probability = np.zeros_like(temporal_difference)
    jump_probability = 0.0
    normal_std = 0.0
    jump_std = 0.0

    if difference_values.size >= 10 and np.std(difference_values) > 0:
        scale = np.std(difference_values)
        normal_std = max(scale * 0.5, 1e-12)
        jump_std = max(scale * 2.0, normal_std * 2)
        jump_probability = 0.1

        for _ in range(200):
            normal_density = np.exp(-0.5 * (difference_values / normal_std) ** 2) / (np.sqrt(2 * np.pi) * normal_std)
            jump_density = np.exp(-0.5 * (difference_values / jump_std) ** 2) / (np.sqrt(2 * np.pi) * jump_std)
            denominator = (1 - jump_probability) * normal_density + jump_probability * jump_density
            responsibility = np.divide(jump_probability * jump_density, denominator, out=np.zeros_like(difference_values), where=denominator > 0)
            new_jump_probability = np.mean(responsibility)
            new_normal_std = np.sqrt(np.sum((1 - responsibility) * difference_values**2) / max(np.sum(1 - responsibility), 1e-12))
            new_jump_std = np.sqrt(np.sum(responsibility * difference_values**2) / max(np.sum(responsibility), 1e-12))

            if new_normal_std > new_jump_std:
                new_normal_std, new_jump_std = new_jump_std, new_normal_std
                new_jump_probability = 1 - new_jump_probability

            if np.allclose([jump_probability, normal_std, jump_std], [new_jump_probability, new_normal_std, new_jump_std], rtol=1e-6, atol=1e-8):
                jump_probability, normal_std, jump_std = new_jump_probability, new_normal_std, new_jump_std
                break

            jump_probability = np.clip(new_jump_probability, 1e-6, 1 - 1e-6)
            normal_std = max(new_normal_std, 1e-12)
            jump_std = max(new_jump_std, normal_std)

        normal_density = np.exp(-0.5 * (difference_values / normal_std) ** 2) / (np.sqrt(2 * np.pi) * normal_std)
        jump_density = np.exp(-0.5 * (difference_values / jump_std) ** 2) / (np.sqrt(2 * np.pi) * jump_std)
        denominator = (1 - jump_probability) * normal_density + jump_probability * jump_density
        temporal_jump_probability[difference_valid] = np.divide(jump_probability * jump_density, denominator, out=np.zeros_like(difference_values), where=denominator > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if residual_values.size > 0:
        axes[0].hist(residual_values, bins=60, density=True, alpha=0.7)

    axes[0].set_xlabel("Normalized displacement residual")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("All-point residual distribution")

    if difference_values.size > 0:
        axes[1].hist(difference_values, bins=60, density=True, alpha=0.7, label="Data")

        if normal_std > 0 and jump_std > 0:
            x_plot = np.linspace(np.min(difference_values), np.max(difference_values), 500)
            normal_curve = (1 - jump_probability) * np.exp(-0.5 * (x_plot / normal_std) ** 2) / (np.sqrt(2 * np.pi) * normal_std)
            jump_curve = jump_probability * np.exp(-0.5 * (x_plot / jump_std) ** 2) / (np.sqrt(2 * np.pi) * jump_std)
            axes[1].plot(x_plot, normal_curve + jump_curve, label="Gaussian mixture")
            axes[1].plot(x_plot, normal_curve, linestyle="--", label="Normal component")
            axes[1].plot(x_plot, jump_curve, linestyle="--", label="Jump component")
            axes[1].legend()

    axes[1].set_xlabel("Temporal residual difference")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("All-point jump distribution")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    jump_parameters = np.array([jump_probability, normal_std, jump_std])

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
        disp_smooth, disp_residual, temporal_difference, temporal_jump_probability, jump_parameters = evaluate_anti_model(disp)

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
