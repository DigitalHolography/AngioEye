from __future__ import annotations

import numpy as np
from scipy.signal import convolve
from scipy.special import jv

ArrayLike = np.ndarray


def _convolve_psf(values: ArrayLike, psf_kernel: ArrayLike | None) -> ArrayLike:
    values = np.asarray(values)
    if psf_kernel is None:
        return values

    return convolve(values, psf_kernel, mode="same")


def uWom_base(alpha_n, x_coord):
    """r
    Formula (18b):
        Bn(x_coord; alpha_n) = 1 - J0(lambda_n x_coord) / J0(lambda_n)
    """

    x_coord = np.asarray(x_coord)

    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    J0 = jv(0, lambda_n)
    Bn = 1.0 - (jv(0, lambda_n * x_coord) / J0)
    return Bn


def uWom_psi(alpha_n, x_coord):
    """
    Formula (18c):
        Psin(x_coord; alpha_n) = dBn / dlnR
                           = -lambda_n J1(lambda_n) / J0(lambda_n)^2
                             * J0(lambda_n x_coord)
    """

    x_coord = np.asarray(x_coord)

    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    J0 = jv(0, lambda_n)

    Psin = (-lambda_n * jv(1, lambda_n)) / (J0**2) * jv(0, lambda_n * x_coord)
    return Psin


def get_womersley_basis(alpha_n, x_coord):
    Bn = uWom_base(alpha_n, x_coord)
    Psin = uWom_psi(alpha_n, x_coord)
    return Bn, Psin


def womersley_flow_gain(alpha_n):
    """
    Formula (18d):
        K(alpha_n) = 1 - 2 J1(lambda_n) / (lambda_n J0(lambda_n))
    """

    lambda_n = np.exp(1j * 3.0 * np.pi / 4.0) * alpha_n
    J0 = jv(0, lambda_n)
    Kn = 1.0 - (2.0 * jv(1, lambda_n)) / (lambda_n * J0)
    return Kn


def generate_womersley_profile(
    r,
    Cn,
    Dn,
    alpha_n,
    R0,
    psf_kernel,
):
    """
    Build the harmonic profile:
        model_profile = Cn * Bn + Dn * Psin
    """

    x_coord = r / R0
    Bn, Psin = get_womersley_basis(alpha_n, x_coord)

    Bn = _convolve_psf(Bn, psf_kernel)
    if Dn != 0:
        Psin = _convolve_psf(Psin, psf_kernel)

    model_profile = (Cn * Bn) + (Dn * Psin)
    return model_profile
