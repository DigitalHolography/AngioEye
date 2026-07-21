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
    
    numerator = np.sum(np.conj(KBn[valid]) * Vn[valid])
    
    denominator = np.sum(np.abs(KBn[valid]) ** 2)
    
    if not np.isfinite(denominator) or denominator <= 0:
        return 0 + 0j
        
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
    displacement = np.zeros((n_t, n_branches, n_radii), dtype=float)
    antisymmetric_model = np.zeros_like(dataset_x_antisymmetric, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            ratio = ratio_map[branch, circle]

            if not np.isfinite(ratio) or ratio <= 0:
                continue

            dx = pixel_size / ratio
            symmetric_gradient = np.gradient(v_model[:, :, branch, circle], dx, axis=1, edge_order=2)

            for t_idx in range(n_t):
                antisymmetric_profile = dataset_x_antisymmetric[t_idx, :, branch, circle]
                gradient_profile = symmetric_gradient[t_idx]
                valid = np.isfinite(antisymmetric_profile) & np.isfinite(gradient_profile)

                if np.sum(valid) < 3:
                    continue

                numerator = np.sum(gradient_profile[valid] * antisymmetric_profile[valid])
                denominator = np.sum(gradient_profile[valid] ** 2)

                if not np.isfinite(denominator) or denominator <= 0:
                    continue

                delta = -numerator / denominator
                displacement[t_idx, branch, circle] = delta
                antisymmetric_model[t_idx, :, branch, circle] = -delta * gradient_profile

    return displacement, antisymmetric_model


def evaluate_anti_model(dataset_x_antisymmetric, antisymmetric_model):
    valid = np.isfinite(dataset_x_antisymmetric) & np.isfinite(antisymmetric_model)
    count = np.sum(valid, axis=1)
    antisymmetric_energy = np.sum(np.where(valid, dataset_x_antisymmetric**2, 0.0), axis=1)
    residual_energy = np.sum(np.where(valid, (dataset_x_antisymmetric - antisymmetric_model) ** 2, 0.0), axis=1)
    antisymmetric_energy = np.where(count >= 3, antisymmetric_energy, np.nan)
    residual_energy = np.where(count >= 3, residual_energy, np.nan)
    antisymmetric_r2 = np.full_like(antisymmetric_energy, np.nan)
    np.divide(residual_energy, antisymmetric_energy, out=antisymmetric_r2, where=np.isfinite(antisymmetric_energy) & (antisymmetric_energy > 0))
    antisymmetric_r2 = 1.0 - antisymmetric_r2

    def statistics(values, axis):
        finite = np.isfinite(values)
        n = np.sum(finite, axis=axis)
        mean = np.full(np.sum(values, axis=axis).shape, np.nan, dtype=float)
        np.divide(np.sum(np.where(finite, values, 0.0), axis=axis), n, out=mean, where=n > 0)
        variance = np.full_like(mean, np.nan)
        np.divide(np.sum(np.where(finite, (values - np.expand_dims(mean, axis)) ** 2, 0.0), axis=axis), n, out=variance, where=n > 0)
        return mean, np.sqrt(variance)

    def jitter(values, axis):
        differences = np.diff(values, axis=axis)
        finite = np.isfinite(differences)
        n = np.sum(finite, axis=axis)
        result = np.full(np.sum(differences, axis=axis).shape, np.nan, dtype=float)
        np.divide(np.sum(np.where(finite, differences**2, 0.0), axis=axis), n, out=result, where=n > 0)
        return np.sqrt(result)

    temporal_r2_mean, temporal_r2_std = statistics(antisymmetric_r2, axis=0)
    temporal_r2_jitter = jitter(antisymmetric_r2, axis=0)
    spatial_r2_mean, spatial_r2_std = statistics(antisymmetric_r2, axis=2)
    spatial_r2_jitter = jitter(antisymmetric_r2, axis=2)
    return antisymmetric_r2, antisymmetric_energy, residual_energy, temporal_r2_mean, temporal_r2_std, temporal_r2_jitter, spatial_r2_mean, spatial_r2_std, spatial_r2_jitter


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
        dataset[dataset == 0] = np.nan

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

        displacement, antisymmetric_model = fit_antisymmetric_curve(dataset_x_antisymmetric, v_model, ratio_map)
        antisymmetric_r2, antisymmetric_energy, residual_energy, temporal_r2_mean, temporal_r2_std, temporal_r2_jitter, spatial_r2_mean, spatial_r2_std, spatial_r2_jitter = evaluate_anti_model(dataset_x_antisymmetric, antisymmetric_model)

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
        metrics["displacement"] = np.asarray(displacement)
        metrics["antisymmetric_model"] = np.asarray(antisymmetric_model)
        metrics["antisymmetric_r2"] = np.asarray(antisymmetric_r2)
        metrics["antisymmetric_energy"] = np.asarray(antisymmetric_energy)
        metrics["residual_energy"] = np.asarray(residual_energy)
        metrics["temporal_r2_mean"] = np.asarray(temporal_r2_mean)
        metrics["temporal_r2_std"] = np.asarray(temporal_r2_std)
        metrics["temporal_r2_jitter"] = np.asarray(temporal_r2_jitter)
        metrics["spatial_r2_mean"] = np.asarray(spatial_r2_mean)
        metrics["spatial_r2_std"] = np.asarray(spatial_r2_std)
        metrics["spatial_r2_jitter"] = np.asarray(spatial_r2_jitter)


        return ProcessResult(metrics=metrics)
