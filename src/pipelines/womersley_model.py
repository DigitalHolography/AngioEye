import h5py

# import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import ceil
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import convolve, windows
from scipy.special import jv

from .core.base import ProcessPipeline, ProcessResult, registerPipeline

num_interp_points_t = 128  # Number of temporal points for interpolation
num_interp_points_x = 16  # Number of spatial points for interpolation
fwhm = 10 * 1e-6  # Full width at half maximum for Gaussian PSF in m
pixel_size = 4.5e-6  # in m
nu = 3.5 * 1e-6  # Viscosity in m^2/s
f0 = 1.2
omega_0 = 2 * np.pi * f0


# v_profile_meas_extraction


def preprocess_v_profile_meas(num_interp_points_x, v_profile):
    valid_mask = ~np.isnan(v_profile)
    valid_indices = np.where(valid_mask)[0]
    valid_count = np.sum(valid_mask)

    if valid_count <= 8:
        # print(f"Warning: Only {valid_count} valid points found. Skipping...")
        return np.zeros(num_interp_points_x)

    min_idx = valid_indices[0]
    max_idx = valid_indices[-1]

    v_valid = v_profile[min_idx : max_idx + 1].copy()
    # v_valid[0], v_valid[-1] = 0.0, 0.0
    x_valid = np.arange(len(v_valid))
    x_interp = np.linspace(0, len(v_valid) - 1, num=num_interp_points_x)

    interpolator = interp1d(
        x_valid,
        v_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )
    v_interp = interpolator(x_interp)

    return np.asanyarray(v_interp)


def extract_v_profile_meas(dataset, num_interp_points_x):
    # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
    n_t, n_x, n_branches, n_radii = dataset.shape
    dataset_x = np.zeros((n_t, num_interp_points_x, n_branches, n_radii), dtype=float)
    v_profile_fft = np.zeros(
        (n_t, num_interp_points_x // 2 + 1, n_branches, n_radii), dtype=complex
    )
    v_profile_meas_n1 = np.zeros(
        (n_t, num_interp_points_x, n_branches, n_radii), dtype=float
    )
    v_profile_meas_dc = np.zeros(
        (n_t, num_interp_points_x, n_branches, n_radii), dtype=float
    )

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for t_idx in range(n_t):
                v_profile = np.asarray(dataset[t_idx, :, branch_idx, radii_idx]) * 1e-3

                v_interp = preprocess_v_profile_meas(
                    num_interp_points_x=num_interp_points_x,
                    v_profile=v_profile,
                )
                dataset_x[t_idx, :, branch_idx, radii_idx] = v_interp

                v_fft = np.fft.rfft(np.asarray(v_interp), n=num_interp_points_x)
                v_profile_fft[t_idx, :, branch_idx, radii_idx] = v_fft

                v_meas = np.zeros_like(v_fft)
                v_meas[1] = v_fft[1]
                v_profile_meas_n1[t_idx, :, branch_idx, radii_idx] = np.fft.irfft(
                    v_meas
                )

                v_meas_dc = np.zeros_like(v_fft)
                v_meas_dc[0] = v_fft[0]
                v_profile_meas_dc[t_idx, :, branch_idx, radii_idx] = np.fft.irfft(
                    v_meas_dc
                )

    return dataset_x, v_profile_fft, v_profile_meas_n1, v_profile_meas_dc


# v_profile_meas_extraction


def preprocess_v_pulse_meas(num_interp_points_t, v_pulse):
    valid_mask = ~np.isnan(v_pulse)
    valid_indices = np.where(valid_mask)[0]

    if valid_indices.size == 0:
        return np.zeros(len(v_pulse))

    min_idx = valid_indices[0]
    max_idx = valid_indices[-1]

    v_valid = v_pulse[min_idx : max_idx + 1].copy()
    x_valid = np.arange(len(v_valid))
    x_interp = np.linspace(0, len(v_valid) - 1, num=num_interp_points_t)

    interpolator = interp1d(
        x_valid,
        v_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )
    v_interp = interpolator(x_interp)

    return np.asanyarray(v_interp)


def extract_v_pulse_meas(dataset, num_interp_points_t):
    # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
    n_t, n_x, n_branches, n_radii = dataset.shape
    v_pulse_fft = np.zeros(
        (num_interp_points_t // 2 + 1, n_x, n_branches, n_radii), dtype=complex
    )
    v_pulse_meas_n1 = np.zeros(
        (num_interp_points_t, n_x, n_branches, n_radii), dtype=float
    )
    v_pulse_meas_dc = np.zeros(
        (num_interp_points_t, n_x, n_branches, n_radii), dtype=float
    )

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for x_idx in range(n_x):
                v_pulse = np.asarray(dataset[:, x_idx, branch_idx, radii_idx])

                v_interp = preprocess_v_pulse_meas(
                    num_interp_points_t=num_interp_points_t,
                    v_pulse=v_pulse,
                )

                v_fft = np.fft.rfft(np.asarray(v_interp), n=num_interp_points_t)
                v_pulse_fft[:, x_idx, branch_idx, radii_idx] = v_fft

                v_meas = np.zeros_like(v_fft)
                v_meas[1] = v_fft[1]
                v_pulse_meas_n1[:, x_idx, branch_idx, radii_idx] = np.fft.irfft(v_meas)

                v_meas_dc = np.zeros_like(v_fft)
                v_meas_dc[0] = v_fft[0]
                v_pulse_meas_dc[:, x_idx, branch_idx, radii_idx] = np.fft.irfft(
                    v_meas_dc
                )

    return v_pulse_fft, v_pulse_meas_n1, v_pulse_meas_dc


# forward_modeling


def _abel_cell_integral(x_abs, r_left, r_right):
    lower = max(r_left, x_abs)
    upper = r_right
    lower_term = np.sqrt(max(lower**2 - x_abs**2, 0.0))
    upper_term = np.sqrt(max(upper**2 - x_abs**2, 0.0))
    return 2.0 * (upper_term - lower_term)


def apply_abel_projection(u_n):
    L = len(u_n)
    x_grid = np.linspace(0, 1, L)
    r_gird = np.linspace(0, 1, L + 1)
    K = np.zeros((L, L))

    for i, x in enumerate(x_grid):
        for j in range(L):
            K[i, j] = _abel_cell_integral(x, r_gird[j], r_gird[j + 1])
    print(f"K: {K}")
    v_model = K @ u_n
    return v_model


def psf_gaussian(fwhm, dx):
    fwhm_in_points = fwhm / dx
    sigma_in_points = fwhm_in_points / (2 * np.sqrt(2 * np.log(2)))
    kernel_radius = ceil(3 * sigma_in_points)
    kernel_size = 2 * kernel_radius + 1
    psf_kernel = windows.gaussian(kernel_size, std=sigma_in_points)
    psf_kernel /= psf_kernel.sum()
    return psf_kernel


def parabola(x, A, x0, y0):
    return A * (x - x0) ** 2 + y0


def parabola_fit(V):
    segment_data = {}
    for branch_index in range(V.shape[2]):
        for circle_index in range(V.shape[3]):
            profile_complex = V[0, :, branch_index, circle_index]
            profile = np.real(profile_complex)

            if np.all(profile == 0):
                continue

            x = np.arange(len(profile))

            try:
                A_guess = -0.1
                x0_guess = np.argmax(profile)
                y0_guess = np.max(profile)

                popt, pcov = curve_fit(
                    parabola, x, profile, p0=[A_guess, x0_guess, y0_guess]
                )

                A_fit, x0_fit, y0_fit = popt

                r0_fit = np.sqrt(-y0_fit / A_fit)

                segment_data[(branch_index, circle_index)] = {
                    "r0": r0_fit,
                    "y0": y0_fit,
                    "x0": x0_fit,
                    "A": A_fit,
                }

                print(
                    f"branch={branch_index}, "
                    f"circle={circle_index}, "
                    f"r0={r0_fit:.4f}, "
                    f"x0={x0_fit:.4f}, "
                    f"y0={y0_fit:.4f}, "
                    f"A={A_fit:.4f}"
                )

            except Exception as e:
                print(f"Fit failed for branch={branch_index}, circle={circle_index}")

                print(e)
    return segment_data


def liste_complet(liste2):
    def union_liste(liste1, liste2):
        liste = []

        for k in range(len(liste1)):
            liste.append(liste1[k])

        for k in range(len(liste2)):
            liste.append(liste2[k])

        return liste

    liste1 = []

    n = len(liste2)

    for k in range(len(liste2)):
        liste1.append(liste2[n - k - 1])

    return union_liste(liste1, liste2)


def womersley_Bn(Vn, R0, nu, omega_n, x0):
    L = len(Vn)

    n_points = int(L / 2)

    x = np.linspace(1 / (2 * n_points), 1, n_points)

    alpha_n = R0 * np.sqrt(omega_n / nu)

    lambda_n = np.exp(1j * 3 * np.pi / 4) * alpha_n

    Bn_half = 1 - jv(0, lambda_n * x) / jv(0, lambda_n)

    Bn_full = liste_complet(Bn_half)

    x_old = np.arange(L)

    x_center_ideal = (L - 1) / 2.0

    delta_x = x0 - x_center_ideal

    x_shifted = x_old - delta_x

    Bn_shifted_real = np.interp(x_shifted, x_old, np.real(Bn_full), left=0, right=0)

    Bn_shifted_imag = np.interp(x_shifted, x_old, np.imag(Bn_full), left=0, right=0)

    Bn_shifted = Bn_shifted_real + 1j * Bn_shifted_imag

    return np.array(Bn_shifted, dtype=complex)


def womersley_Psin(Vn, R0, nu, omega_n, x0):
    L = len(Vn)

    n_points = int(L / 2)

    x = np.linspace(1 / (2 * n_points), 1, n_points)

    alpha_n = R0 * np.sqrt(omega_n / nu)

    lambda_n = np.exp(1j * 3 * np.pi / 4) * alpha_n

    Psin_half = (
        (-lambda_n * jv(1, lambda_n)) / (jv(0, lambda_n) ** 2) * jv(0, lambda_n * x)
    )

    Psin_full = liste_complet(Psin_half)

    x_old = np.arange(L)

    x_center_ideal = (L - 1) / 2.0

    delta_x = x0 - x_center_ideal

    x_shifted = x_old - delta_x

    Psin_shifted_real = np.interp(x_shifted, x_old, np.real(Psin_full), left=0, right=0)

    Psin_shifted_imag = np.interp(x_shifted, x_old, np.imag(Psin_full), left=0, right=0)

    Psin_shifted = Psin_shifted_real + 1j * Psin_shifted_imag

    return np.array(Psin_shifted, dtype=complex)


def compute_Cn(Vn, model, flag):
    if flag:
        print("TODO")

    else:
        numerator = np.sum(np.conj(model) * Vn)

        denominator = np.sum(np.abs(model) ** 2)

        return numerator / denominator


def compute_Dn(Vn, model, flag):
    if flag:
        print("TODO")


def generate_harmonic_flow_profile(V, segment_data):
    v_model_freq = np.zeros(
        (V.shape[0], V.shape[1], V.shape[2], V.shape[3]), dtype=complex
    )
    for branch_index in range(V.shape[2]):
        for circle_index in range(V.shape[3]):
            if (branch_index, circle_index) not in segment_data:
                continue

            r0 = segment_data[(branch_index, circle_index)]["r0"]
            y0 = segment_data[(branch_index, circle_index)]["y0"]
            x0 = segment_data[(branch_index, circle_index)]["x0"]
            A = segment_data[(branch_index, circle_index)]["A"]

            print("\n===================================================")
            print(f"branch = {branch_index}, circle = {circle_index}")
            print(f"r0 = {r0}, y0 = {y0}, x0 = {x0}")
            print("===================================================\n")

            matrix = V[:, :, branch_index, circle_index]

            x = np.arange(matrix.shape[1])

            Cn = np.zeros(V.shape[0], dtype=complex)
            # Dn = np.zeros(V.shape[0], dtype=complex)

            for n in range(3):
                print(f"n = {n}")

                Vn = np.array(matrix[n], dtype=complex)

                R0 = r0 * pixel_size

                if n == 0:
                    model = parabola(x, A, x0, y0)
                else:
                    omega_n = n * omega_0

                    Bn = womersley_Bn(Vn, R0, nu, omega_n, x0)

                    Cn[n] = compute_Cn(Vn, Bn, flag=False)
                    model_rigidwall = Cn[n] * Bn

                    # model_movingwall = (Cn[n] * Bn) + (Dn[n] * Psin)

                    model = model_rigidwall

                u_n = model

                psf_kernel = psf_gaussian(fwhm, pixel_size)
                v_prof = apply_abel_projection(u_n)
                v_blurred = convolve(v_prof, psf_kernel, mode="same")
                v_model_freq[n, :, branch_index, circle_index] = v_blurred

    v_model = np.fft.irfft(v_model_freq, axis=0)

    return v_model


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
        dataset = obj[:]

        obj = h5file[self.b_period_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.b_period_path}, but found {type(obj)}"
            )
        # b_period = np.mean(obj[:])
        # print(f"b_period: {b_period}")

        dataset_x, v_profile_fft, v_profile_meas_n1, v_profile_meas_dc = (
            extract_v_profile_meas(
                dataset=dataset,
                num_interp_points_x=num_interp_points_x,
            )
        )

        v_pulse_fft, v_pulse_meas_n1, v_pulse_meas_dc = extract_v_pulse_meas(
            dataset=dataset_x,
            num_interp_points_t=num_interp_points_t,
        )

        # v_pulse_fft_filtered, r0_std = profile_analysis()
        segment_data = parabola_fit(v_pulse_fft)
        v_model = generate_harmonic_flow_profile(v_pulse_fft, segment_data)

        metrics: dict = {}
        metrics["dataset_x"] = np.asarray(dataset_x)
        metrics["v_profile_fft"] = np.asarray(v_profile_fft)
        metrics["v_profile_meas_n1"] = np.asarray(v_profile_meas_n1)
        metrics["v_profile_meas_dc"] = np.asarray(v_profile_meas_dc)
        metrics["v_pulse_fft"] = np.asarray(v_pulse_fft)
        metrics["v_pulse_meas_n1"] = np.asarray(v_pulse_meas_n1)
        metrics["v_pulse_meas_dc"] = np.asarray(v_pulse_meas_dc)
        metrics["v_model"] = np.asarray(v_model)
        # metrics["v_profile_fft_filtered"] = np.asarray(v_pulse_fft_filtered)
        # metrics["r0_std"] = r0_std

        return ProcessResult(metrics=metrics)
