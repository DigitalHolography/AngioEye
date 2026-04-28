from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .geometric_correction import (
    HarmonicRadialProfile,
    LateralGrid,
    RadialGrid,
    project_harmonic_profile,
)
from .womersley_quantities import generate_womersley_profile, womersley_flow_gain

ArrayLike = np.ndarray

R0 = 0.5  # Placeholder for vessel radius in mm
Nu = 3.0e-6  # Placeholder for kinematic viscosity in mm^2/s
v_meas: ArrayLike = np.array([])  # Placeholder for measured velocity data
omega_n = (
    2.0 * np.pi * 1.0
)  # Placeholder for angular frequency of the harmonic component
weights: ArrayLike = np.ones_like(v_meas)  # Placeholder for measurement weights
r = np.linspace(0, R0, num=100)  # Radial grid for modeling
psf_kernel: ArrayLike | None = None  # Placeholder for point spread function kernel


def _apply_geometric_correction(
    model_profile: ArrayLike,
    R0: float,
    harmonic_order: int = 1,
) -> ArrayLike:
    model_profile = np.asarray(model_profile)
    radial_grid = RadialGrid.uniform(R0=R0, n_samples=model_profile.size)
    lateral_grid = LateralGrid.uniform(n_meas=v_meas.size)

    profile = HarmonicRadialProfile(
        harmonic_order=harmonic_order,
        radial_grid=radial_grid,
        values=model_profile,
    )
    return np.asarray(project_harmonic_profile(profile, lateral_grid))


def _diagonal_weighting_operator(weights) -> ArrayLike:
    weights = np.diag(weights)
    return weights


def _unpack_complex_coefficients(p) -> tuple[complex, complex]:
    p = np.asarray(p, dtype=float)
    if p.size != 4:
        raise ValueError("Expected parameter vector [Re(Cn), Im(Cn), Re(Dn), Im(Dn)].")
    Cn = p[0] + 1j * p[1]
    Dn = p[2] + 1j * p[3]
    return Cn, Dn


def _stack_complex_residual(residual: ArrayLike) -> ArrayLike:
    residual = np.asarray(residual, dtype=complex)
    return np.concatenate((np.real(residual), np.imag(residual)))


def get_alpha_n(omega_n: float, R0: float = R0, Nu: float = Nu) -> float:
    return float(R0) * np.sqrt(float(omega_n) / float(Nu))


def model_harmonic_profile(
    r,
    omega_n,
    Cn,
    Dn,
    psf_kernel=None,
    harmonic_order: int = 1,
    R0: float = R0,
    Nu: float = Nu,
) -> ArrayLike:
    alpha_n = get_alpha_n(omega_n=omega_n, R0=R0, Nu=Nu)
    model_profile = generate_womersley_profile(
        r=r,
        Cn=Cn,
        Dn=Dn,
        alpha_n=alpha_n,
        R0=R0,
        psf_kernel=psf_kernel,
    )

    return _apply_geometric_correction(
        model_profile=model_profile,
        R0=R0,
        harmonic_order=harmonic_order,
    )


def costFun(
    p,
    r,
    v_meas,
    omega_n,
    psf_kernel=None,
    weights=None,
    harmonic_order: int = 1,
    R0: float = R0,
    Nu: float = Nu,
) -> ArrayLike:
    """
    Weighted complex residual for fixed R0, Nu, and psf_kernel.

    The optimization variables are the real and imaginary parts of Cn and Dn.
    """

    Cn, Dn = _unpack_complex_coefficients(p)
    v_modeled = model_harmonic_profile(
        r=r,
        omega_n=omega_n,
        Cn=Cn,
        Dn=Dn,
        psf_kernel=psf_kernel,
        harmonic_order=harmonic_order,
        R0=R0,
        Nu=Nu,
    )

    v_meas = np.asarray(v_meas, dtype=complex)

    sqrt_weight = np.sqrt(_diagonal_weighting_operator(weights))
    residual = sqrt_weight * (v_meas - v_modeled)
    return _stack_complex_residual(residual)


def fit_Cn_Dn_least_squares(
    r,
    v_meas,
    omega_n,
    psf_kernel=None,
    weights=None,
    harmonic_order: int = 1,
    p0=None,
    bounds=(-np.inf, np.inf),
    R0: float = R0,
    Nu: float = Nu,
    **least_squares_kwargs,
):
    """
    Fit complex coefficients Cn and Dn with fixed R0, Nu, and psf_kernel.
    """

    if p0 is None:
        p0 = np.zeros(4, dtype=float)

    result = least_squares(
        fun=costFun,
        x0=np.asarray(p0, dtype=float),
        bounds=bounds,
        args=(
            r,
            v_meas,
            omega_n,
            psf_kernel,
            weights,
            harmonic_order,
            R0,
            Nu,
        ),
        **least_squares_kwargs,
    )

    Cn_fit, Dn_fit = _unpack_complex_coefficients(result.x)
    v_modeled = model_harmonic_profile(
        r=r,
        omega_n=omega_n,
        Cn=Cn_fit,
        Dn=Dn_fit,
        psf_kernel=psf_kernel,
        harmonic_order=harmonic_order,
        R0=R0,
        Nu=Nu,
    )
    v_meas = np.asarray(v_meas, dtype=complex)
    residual = v_meas - v_modeled
    weighted_residual = np.sqrt(_diagonal_weighting_operator(weights)) * residual

    return {
        "result": result,
        "Cn_fit": Cn_fit,
        "Dn_fit": Dn_fit,
        "ALPHA_N": get_alpha_n(omega_n=omega_n, R0=R0, Nu=Nu),
        "Kn": womersley_flow_gain(get_alpha_n(omega_n=omega_n, R0=R0, Nu=Nu)),
        "R0": float(R0),
        "Nu": float(Nu),
        "modeled_profile": v_modeled,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "residual_magnitude_rms": float(
            np.sqrt(np.mean(np.abs(weighted_residual) ** 2))
        ),
    }


__all__ = [
    "R0",
    "Nu",
    "costFun",
    "fit_Cn_Dn_least_squares",
    "get_alpha_n",
    "model_harmonic_profile",
]
