import numpy as np
from scipy.signal import convolve
from scipy.special import jv


def _abel_cell_integral(x_abs, r_left, r_right):
    lower = max(r_left, x_abs)
    upper = r_right
    lower_term = np.sqrt(max(lower**2 - x_abs**2, 0.0))
    upper_term = np.sqrt(max(upper**2 - x_abs**2, 0.0))
    return 2.0 * (upper_term - lower_term)


def apply_abel_projection(radial_profile, r_coord, x_coord, R0):
    K_half = np.zeros((len(x_coord), len(r_coord)))
    K = np.zeros((2 * len(x_coord), len(r_coord)))
    for i, xi in enumerate(x_coord):
        x_abs = float(xi)
        for j in range(len(r_coord) - 1):
            K_half[i, j] = _abel_cell_integral(x_abs, r_coord[j], r_coord[j + 1])
    K = np.concatenate([np.flip(K_half), K_half])
    print(f"K: {K}")
    v_model = K @ radial_profile
    return v_model


def get_womersley_physics(n_harmonic, b_period, R0, Nu):
    omega_n = (2.0 * np.pi * n_harmonic) / b_period
    alpha_n = R0 * np.sqrt(omega_n / Nu)
    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    j0_val = jv(0, lambda_n)
    j1_val = jv(1, lambda_n)
    kn = 1.0 - (2.0 * j1_val) / (lambda_n * j0_val)
    print(f"alpha_n: {alpha_n}")
    return lambda_n, j0_val, j1_val, kn


def generate_harmonic_flow_profile(
    Cn, Dn, n_harmonic, b_period, R0, Nu, x_coord, r_coord, psf_kernel=None
):
    lambda_n, j0_val, j1_val, _ = get_womersley_physics(n_harmonic, b_period, R0, Nu)
    x_rad = r_coord.copy() / R0
    bn = 1.0 - (jv(0, lambda_n * x_rad) / j0_val)
    psin = (-lambda_n * j1_val) / (j0_val**2) * jv(0, lambda_n * x_rad)
    u_n = (Cn * bn) + (Dn * psin)
    print(f"u_n: {u_n}")
    v_model = apply_abel_projection(u_n, r_coord, x_coord, R0)

    if psf_kernel is not None:
        v_model = convolve(v_model, psf_kernel, mode="same")

    return v_model
