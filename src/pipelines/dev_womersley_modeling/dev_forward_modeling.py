import numpy as np
from scipy.signal import convolve
from scipy.special import jv


def _abel_cell_integral(x_abs, r_left, r_right):
    lower = max(r_left, x_abs)
    upper = r_right
    lower_term = np.sqrt(max(lower**2 - x_abs**2, 0.0))
    upper_term = np.sqrt(max(upper**2 - x_abs**2, 0.0))
    return 2.0 * (upper_term - lower_term)


def apply_abel_projection(radial_profile, r_edges, x_positions, R0):
    K = np.zeros((len(x_positions), len(radial_profile)))
    for i, xi in enumerate(x_positions):
        x_abs = abs(float(xi))
        if x_abs >= R0:
            continue
        for j in range(len(radial_profile)):
            K[i, j] = _abel_cell_integral(x_abs, r_edges[j], r_edges[j + 1])
    return K @ radial_profile


def get_womersley_physics(n_harmonic, b_period, R0, Nu):
    omega_n = (2.0 * np.pi * n_harmonic) / b_period
    alpha_n = R0 * np.sqrt(omega_n / Nu)
    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    j0_val = jv(0, lambda_n)
    j1_val = jv(1, lambda_n)
    kn = 1.0 - (2.0 * j1_val) / (lambda_n * j0_val)
    return alpha_n, lambda_n, j0_val, j1_val, kn


def generate_harmonic_flow_profile(
    Cn, Dn, n_harmonic, b_period, R0, Nu, x_coord, psf_kernel=None
):
    r_edges = np.asarray(x_coord)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    alpha_n, lambda_n, j0_val, j1_val, _ = get_womersley_physics(
        n_harmonic, b_period, R0, Nu
    )

    x_rad = r_centers / R0
    bn = 1.0 - (jv(0, lambda_n * x_rad) / j0_val)
    psin = (-lambda_n * j1_val) / (j0_val**2) * jv(0, lambda_n * x_rad)

    u_n = (Cn * bn) + (Dn * psin)

    v_model = apply_abel_projection(u_n, r_edges, x_coord, R0)

    if psf_kernel is not None:
        v_model = convolve(v_model, psf_kernel, mode="same")

    return v_model
