from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .geometric_correction import (
    HarmonicRadialProfile,
    LateralGrid,
    RadialGrid,
    project_harmonic_profile,
)
from .womersley_quantities import generate_womersley_profile

ArrayLike = np.ndarray

R0 = 50.0
Nu = 3.0e-6


def _apply_geometric_correction(
    model_profile: ArrayLike,
    radial_grid: RadialGrid | None,
    lateral_grid: LateralGrid | None,
    harmonic_order: int = 1,
) -> ArrayLike:
    model_profile = np.asarray(model_profile)
    if radial_grid is None or lateral_grid is None:
        return model_profile
    if radial_grid.centers.size != model_profile.size:
        return model_profile

    profile = HarmonicRadialProfile(
        harmonic_order=harmonic_order,
        radial_grid=radial_grid,
        values=model_profile,
    )
    return np.asarray(project_harmonic_profile(profile, lateral_grid))


def _coerce_weight_vector(weights, size: int) -> ArrayLike:
    if weights is None:
        return np.ones(size, dtype=float)

    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 2:
        weights = np.diag(weights)
    if weights.ndim != 1 or weights.size != size:
        raise ValueError(
            f"Weights must be a vector of length {size}, got shape {weights.shape}."
        )
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
    return np.concatenate((residual.real, residual.imag))


def get_alpha_n(omega_n: float, R0: float = R0, nu: float = Nu) -> float:
    return float(R0) * np.sqrt(float(omega_n) / float(nu))


def model_harmonic_profile(
    r,
    omega_n,
    Cn,
    Dn,
    psf_kernel=None,
    radial_grid: RadialGrid | None = None,
    lateral_grid: LateralGrid | None = None,
    harmonic_order: int = 1,
    R0: float = R0,
    nu: float = Nu,
) -> ArrayLike:
    alpha_n = get_alpha_n(omega_n=omega_n, R0=R0, nu=nu)
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
        radial_grid=radial_grid,
        lateral_grid=lateral_grid,
        harmonic_order=harmonic_order,
    )


def costFun(
    p,
    r,
    v_meas,
    omega_n,
    psf_kernel=None,
    weights=None,
    radial_grid: RadialGrid | None = None,
    lateral_grid: LateralGrid | None = None,
    harmonic_order: int = 1,
    R0: float = R0,
    nu: float = Nu,
) -> ArrayLike:
    """
    Weighted complex residual for fixed R0, nu, and psf_kernel.

    The optimization variables are the real and imaginary parts of Cn and Dn.
    """

    Cn, Dn = _unpack_complex_coefficients(p)
    modeled = model_harmonic_profile(
        r=r,
        omega_n=omega_n,
        Cn=Cn,
        Dn=Dn,
        psf_kernel=psf_kernel,
        radial_grid=radial_grid,
        lateral_grid=lateral_grid,
        harmonic_order=harmonic_order,
        R0=R0,
        nu=nu,
    )

    v_meas = np.asarray(v_meas, dtype=complex)
    if modeled.shape != v_meas.shape:
        raise ValueError(
            "Modeled profile and measured harmonic must have the same shape, "
            f"got {modeled.shape} and {v_meas.shape}."
        )

    sqrt_weight = np.sqrt(_coerce_weight_vector(weights, v_meas.size))
    residual = sqrt_weight * (v_meas - modeled)
    return _stack_complex_residual(residual)


def fit_Cn_Dn_least_squares(
    r,
    v_meas,
    omega_n,
    psf_kernel=None,
    weights=None,
    radial_grid: RadialGrid | None = None,
    lateral_grid: LateralGrid | None = None,
    harmonic_order: int = 1,
    p0=None,
    bounds=(-np.inf, np.inf),
    R0: float = R0,
    nu: float = Nu,
    **least_squares_kwargs,
):
    """
    Fit complex coefficients Cn and Dn with fixed R0, nu, and psf_kernel.
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
            radial_grid,
            lateral_grid,
            harmonic_order,
            R0,
            nu,
        ),
        **least_squares_kwargs,
    )

    Cn_fit, Dn_fit = _unpack_complex_coefficients(result.x)
    modeled = model_harmonic_profile(
        r=r,
        omega_n=omega_n,
        Cn=Cn_fit,
        Dn=Dn_fit,
        psf_kernel=psf_kernel,
        radial_grid=radial_grid,
        lateral_grid=lateral_grid,
        harmonic_order=harmonic_order,
        R0=R0,
        nu=nu,
    )
    v_meas = np.asarray(v_meas, dtype=complex)
    residual = v_meas - modeled
    weighted_residual = np.sqrt(_coerce_weight_vector(weights, v_meas.size)) * residual

    return {
        "result": result,
        "Cn_fit": Cn_fit,
        "Dn_fit": Dn_fit,
        "ALPHA_N": get_alpha_n(omega_n=omega_n, R0=R0, nu=nu),
        "R0": float(R0),
        "nu": float(nu),
        "modeled_profile": modeled,
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
