from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class RadialGrid:
    """
    Radial discretization for the theoretical velocity profile u_n(r).

    centers:
        Radial sample locations r_j in [0, R0].
    edges:
        Bin edges with shape (n_r + 1,). They define the integration cells.
    radius:
        Vessel radius R0.
    """

    centers: ArrayLike
    edges: ArrayLike
    radius: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        edges = np.asarray(self.edges, dtype=float)
        radius = float(self.radius)

        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "radius", radius)

    @classmethod
    def uniform(cls, radius: float, n_samples: int) -> RadialGrid:
        radius = float(radius)
        n_samples = int(n_samples)

        edges = np.linspace(0.0, radius, n_samples + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return cls(centers=centers, edges=edges, radius=radius)


@dataclass(frozen=True)
class LateralGrid:
    """
    Lateral camera sampling locations x_i.

    The Abel projection is physically defined for |x| <= R0. Values outside
    the vessel radius remain allowed, but their kernel rows become zero.
    """

    positions: ArrayLike

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class HarmonicRadialProfile:
    """
    Discrete radial velocity profile u_n(r_j) for one harmonic order n.

    values may be real or complex. In Womersley modeling they will typically
    be complex harmonic amplitudes.
    """

    harmonic_order: int
    radial_grid: RadialGrid
    values: ArrayLike

    def __post_init__(self) -> None:
        harmonic_order = int(self.harmonic_order)
        values = np.asarray(self.values)

        if values.size != self.radial_grid.centers.size:
            raise ValueError(
                "Profile length must match the number of radial grid samples."
            )

        object.__setattr__(self, "harmonic_order", harmonic_order)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class AbelProjectionOperator:
    """
    Discrete Abel projection operator K such that M_n = K u_n.

    matrix[i, j] approximates the contribution of radial cell j to the
    depth-integrated measurement at lateral position x_i.
    """

    radial_grid: RadialGrid
    lateral_grid: LateralGrid
    matrix: ArrayLike

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        expected_shape = (
            self.lateral_grid.positions.size,
            self.radial_grid.centers.size,
        )
        if matrix.ndim != 2 or matrix.shape != expected_shape:
            raise ValueError(
                f"matrix must have shape {expected_shape}, got {matrix.shape}."
            )

        object.__setattr__(self, "matrix", matrix)

    def apply(self, profile: HarmonicRadialProfile) -> ArrayLike:
        if profile.radial_grid.centers.size != self.radial_grid.centers.size:
            raise ValueError("Profile grid is not compatible with this operator.")
        return self.matrix @ profile.values


def _abel_cell_integral(x_abs: float, r_left: float, r_right: float) -> float:
    """
    Exact cell integral for piecewise-constant u(r) over [r_left, r_right]:

        2 * integral r / sqrt(r^2 - x^2) dr = 2 * sqrt(r^2 - x^2)
    """

    if r_right <= x_abs:
        return 0.0

    lower = max(r_left, x_abs)
    upper = r_right

    lower_term = np.sqrt(max(lower * lower - x_abs * x_abs, 0.0))
    upper_term = np.sqrt(max(upper * upper - x_abs * x_abs, 0.0))
    return 2.0 * (upper_term - lower_term)


def build_abel_projection_matrix(
    radial_grid: RadialGrid,
    lateral_grid: LateralGrid,
) -> ArrayLike:
    """
    Build the discrete Abel projection matrix K.

    The discretization assumes u_n is piecewise constant on each radial cell:

        M_n(x_i) ~= sum_j K[i, j] * u_n(r_j)

    with

        K[i, j] = 2 * integral_{cell_j intersect [|x_i|, R0]}
                       r / sqrt(r^2 - x_i^2) dr
    """

    x = np.asarray(lateral_grid.positions, dtype=float)
    edges = np.asarray(radial_grid.edges, dtype=float)
    radius = float(radial_grid.radius)

    K = np.zeros((x.size, radial_grid.centers.size), dtype=float)

    for i, xi in enumerate(x):
        x_abs = abs(float(xi))
        if x_abs >= radius:
            continue

        for j in range(radial_grid.centers.size):
            K[i, j] = _abel_cell_integral(
                x_abs=x_abs,
                r_left=float(edges[j]),
                r_right=float(edges[j + 1]),
            )

    return K


def project_harmonic_profile(
    profile: HarmonicRadialProfile,
    lateral_grid: LateralGrid,
) -> ArrayLike:
    """
    Convenience wrapper that constructs K from the profile grid and returns
    the projected measurement-space harmonic M_n(x_i).
    """

    matrix = build_abel_projection_matrix(
        radial_grid=profile.radial_grid,
        lateral_grid=lateral_grid,
    )
    operator = AbelProjectionOperator(
        radial_grid=profile.radial_grid,
        lateral_grid=lateral_grid,
        matrix=matrix,
    )
    return operator.apply(profile)
