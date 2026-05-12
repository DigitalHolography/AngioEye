import numpy as np
from numpy.ma import ceil
from scipy.signal import convolve, windows
from scipy.special import jv


def _abel_cell_integral(x_abs, r_left, r_right):
    lower = max(r_left, x_abs)
    upper = r_right
    lower_term = np.sqrt(max(lower**2 - x_abs**2, 0.0))
    upper_term = np.sqrt(max(upper**2 - x_abs**2, 0.0))
    return 2.0 * (upper_term - lower_term)


def apply_abel_projection(radial_profile, r_coord, x_coord, R0):
    r_gird = r_coord.copy() / R0
    x_grid = x_coord.copy() / R0
    K_half = np.zeros((len(x_grid), len(r_gird)))
    K = np.zeros((2 * len(x_grid), len(r_gird)))
    for i, xi in enumerate(x_grid):
        x_abs = float(xi)
        for j in range(len(r_gird) - 1):
            K_half[i, j] = _abel_cell_integral(x_abs, r_gird[j], r_gird[j + 1])
    K = np.concatenate([np.flip(K_half), K_half])
    print(f"K: {K}")
    v_model = K @ radial_profile
    return v_model


def psf_gaussian(fwhm_um, dx_um):
    fwhm_in_points = fwhm_um / dx_um
    sigma_in_points = fwhm_in_points / (2 * np.sqrt(2 * np.log(2)))
    kernel_radius = ceil(3 * sigma_in_points)
    kernel_size = 2 * kernel_radius + 1
    psf_kernel = windows.gaussian(kernel_size, std=sigma_in_points)
    psf_kernel /= psf_kernel.sum()
    return psf_kernel


def get_womersley_physics(n, b_period, R0, Nu):
    omega_n = (2.0 * np.pi * n) / b_period
    alpha_n = R0 * np.sqrt(omega_n / Nu)
    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    j0_val = jv(0, lambda_n)
    j1_val = jv(1, lambda_n)
    kn = 1.0 - (2.0 * j1_val) / (lambda_n * j0_val)
    print(f"alpha_n: {alpha_n}")
    return lambda_n, j0_val, j1_val, kn


def generate_harmonic_flow_profile(
    Cn, Dn, harmonics, b_period, R0, Nu, x_coord, r_coord, fwhm, dx, v_pulse_n_dc
):
    v_model_freq = np.zeros((len(harmonics), len(x_coord)), dtype=complex)

    psf_kernel = psf_gaussian(fwhm, dx)
    n = 1

    lambda_n, j0_val, j1_val, _ = get_womersley_physics(n, b_period, R0, Nu)
    x_rad = r_coord.copy() / R0

    bn = 1.0 - (jv(0, lambda_n * x_rad) / j0_val)
    psin = (-lambda_n * j1_val) / (j0_val**2) * jv(0, lambda_n * x_rad)
    print(f"bn: {bn}")
    print(f"psin: {psin}")

    u_n = (Cn * bn) + (Dn * psin)
    print(f"u_n: {u_n}")

    v_prof = apply_abel_projection(u_n, r_coord, x_coord, R0)
    v_blurred = convolve(v_prof, psf_kernel, mode="same")
    v_downsampled = v_blurred[::2]
    print(f"v_downsampled: {v_downsampled}")
    v_model_freq[0, :] = v_pulse_n_dc
    print(f"v_pulse_n_dc: {v_pulse_n_dc}")
    v_model_freq[n, :] = v_downsampled

    v_model = np.fft.irfft(v_model_freq, axis=0)

    return v_model, v_model_freq
