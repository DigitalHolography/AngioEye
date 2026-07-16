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
    valid_mask = ~np.isnan(v_profile)
    valid_indices = np.where(valid_mask)[0]
    valid_count = np.sum(valid_mask)

    if valid_count <= 8:
        # print(f"Warning: Only {valid_count} valid points found. Skipping...")
        return np.zeros(num_interp_points_x), 0.0

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
    ratio = (num_interp_points_x - 1) / (len(v_valid) - 1)

    return np.asarray(v_interp), ratio


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

    ratio_map = np.zeros((n_branches, n_radii))

    for branch_idx in range(n_branches):
        for radii_idx in range(n_radii):
            for t_idx in range(n_t):
                v_profile = np.asarray(dataset[t_idx, :, branch_idx, radii_idx]) * 1e-3

                v_interp, ratio = preprocess_v_profile_meas(
                    num_interp_points_x=num_interp_points_x,
                    v_profile=v_profile,
                )
                dataset_x[t_idx, :, branch_idx, radii_idx] = v_interp

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

    dataset_x_symmetric = 0.5 * (
        dataset + dataset_flipped
    )

    dataset_x_antisymmetric = 0.5 * (
        dataset - dataset_flipped
    )

    invalid_profiles = np.all(
        np.isnan(dataset),
        axis=1,
        keepdims=True,
    )

    dataset_x_symmetric = np.where(
        invalid_profiles,
        np.nan,
        dataset_x_symmetric,
    )

    dataset_x_antisymmetric = np.where(
        invalid_profiles,
        np.nan,
        dataset_x_antisymmetric,
    )

    return (
        dataset_x_symmetric,
        dataset_x_antisymmetric,
    )


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
    return K @ (A * (x - x0) ** 2 + y0)


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
    # print(f"alpha_n: {alpha_n}")
    lam = np.exp(1j * 3 * np.pi / 4) * alpha_n
    # print(f"lam: {lam}")
    Bn = 1 - jv(0, lam * np.abs(x_norm)) / jv(0, lam)
    # print(f"Bn before masking: {Bn}")

    mask = np.abs(x_norm) > 1
    idx = np.where(mask)[0]

    left_idx = idx[idx < L / 2]
    for i in left_idx[::-1]:
        if i + 1 < L:
            Bn[i] = Bn[i + 1] / 4

    right_idx = idx[idx >= L / 2]
    for i in right_idx:
        if i - 1 >= 0:
            Bn[i] = Bn[i - 1] / 4

    return Bn.astype(complex)


def compute_Cn(Vn, KBn):
    numerator = np.sum(np.conj(KBn) * Vn)

    denominator = np.sum(np.abs(KBn) ** 2)

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
    v_model_fft = np.zeros(
        (V.shape[0], V.shape[1], V.shape[2], V.shape[3]), dtype=complex
    )
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
            Cn[0] = 1
            Qn = np.zeros(V.shape[0], dtype=complex)
            taun = np.zeros(V.shape[0], dtype=complex)

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


def evaluate_womersley_model(
    metrics,
    branch_index,
    circle_index,
    position_index,
    save_prefix=None,
):
    dataset_x = metrics["dataset_x_aligned"]
    v_model = metrics["v_model"]

    # ==========================================================
    # Figure 1
    # Velocity profile comparison at selected cardiac phase
    # ==========================================================

    center_idx = dataset_x.shape[1] // 2

    center_waveform = dataset_x[
        :,
        center_idx,
        branch_index,
        circle_index,
    ]

    t_peak = np.argmax(center_waveform)
    t_min = np.argmin(center_waveform)

    t_mid = (t_peak + t_min) // 2

    time_indices = [
        t_peak,
        t_mid,
        t_min,
    ]

    phase_names = [
        "Peak Systole",
        "Mid Cycle",
        "End Diastole",
    ]

    fig1, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
        sharey=True,
    )

    for ax, t_idx, phase_name in zip(
        axes,
        time_indices,
        phase_names,
        strict=True,
    ):
        raw_profile = dataset_x[
            t_idx,
            :,
            branch_index,
            circle_index,
        ]

        model_profile = v_model[
            t_idx,
            :,
            branch_index,
            circle_index,
        ]

        x = np.arange(len(raw_profile))

        ax.plot(
            x,
            raw_profile,
            "o-",
            label="Measured",
        )

        ax.plot(
            x,
            model_profile,
            "s-",
            label="Model",
        )

        rmse = np.sqrt(nanmean((raw_profile - model_profile) ** 2))

        ax.set_title(f"{phase_name}\nt={t_idx}, RMSE={rmse:.3f}")

        ax.grid(True)

        ax.set_xlabel("Radial Position")

    axes[0].set_ylabel("Velocity")

    axes[0].legend()

    fig1.suptitle(
        f"Velocity Profile Validation\nBranch={branch_index}, Circle={circle_index}"
    )

    plt.tight_layout()

    if save_prefix is not None:
        fig1.savefig(
            f"{save_prefix}_profile_validation.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ==========================================================
    # Figure 2
    # Symmetric / antisymmetric residual decomposition
    # ==========================================================

    fig2, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        sharey=True,
    )

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]

    sym_all = []
    asym_all = []

    for phase_name, t_idx, color in zip(
        phase_names,
        time_indices,
        colors,
        strict=True,
    ):
        raw_profile = dataset_x[
            t_idx,
            :,
            branch_index,
            circle_index,
        ]

        model_profile = v_model[
            t_idx,
            :,
            branch_index,
            circle_index,
        ]

        residual = (raw_profile - model_profile) / (np.abs(raw_profile) + 1e-12)

        residual_flip = residual[::-1]

        residual_sym = 0.5 * (residual + residual_flip)

        residual_asym = 0.5 * (residual - residual_flip)

        sym_all.append(residual_sym)
        asym_all.append(residual_asym)

        x = np.arange(len(residual))

        axes[0].plot(
            x,
            residual_sym,
            "o-",
            color=color,
            label=phase_name,
        )

        axes[1].plot(
            x,
            residual_asym,
            "o-",
            color=color,
            label=phase_name,
        )

    all_values = np.concatenate(sym_all + asym_all)

    ymax = nanmax(np.abs(all_values))

    axes[0].set_ylim(
        -ymax - 0.25,
        ymax + 0.25,
    )

    axes[1].set_ylim(
        -ymax - 0.25,
        ymax + 0.25,
    )

    axes[0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_ylabel("Symmetric Residual")

    axes[1].set_ylabel("Antisymmetric Residual")

    axes[1].set_xlabel("Radial Position")

    axes[0].legend()
    axes[1].legend()

    axes[0].grid(True)
    axes[1].grid(True)

    fig2.suptitle(
        f"Residual Symmetry Analysis\nBranch={branch_index}, Circle={circle_index}"
    )

    plt.tight_layout()

    if save_prefix is not None:
        fig2.savefig(
            f"{save_prefix}_residual_symmetry.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ==========================================================
    # Figure 3
    # residual_energy_fraction
    # ==========================================================

    from scipy.ndimage import gaussian_filter1d

    n_time = dataset_x.shape[0]

    SymEnergy = np.zeros(n_time)
    AsymEnergy = np.zeros(n_time)
    TotalEnergy = np.zeros(n_time)

    for t in range(n_time):
        raw_profile = dataset_x[
            t,
            :,
            branch_index,
            circle_index,
        ]

        model_profile = v_model[
            t,
            :,
            branch_index,
            circle_index,
        ]

        residual = raw_profile - model_profile

        residual_flip = residual[::-1]

        sym = 0.5 * (residual + residual_flip)

        asym = 0.5 * (residual - residual_flip)

        SymEnergy[t] = np.sum(sym**2)

        AsymEnergy[t] = np.sum(asym**2)

        TotalEnergy[t] = np.sum(residual**2)

    sigma = 2

    SymEnergy_s = gaussian_filter1d(
        SymEnergy,
        sigma=sigma,
    )

    AsymEnergy_s = gaussian_filter1d(
        AsymEnergy,
        sigma=sigma,
    )

    TotalEnergy_s = gaussian_filter1d(
        TotalEnergy,
        sigma=sigma,
    )

    phase = np.linspace(
        0,
        100,
        n_time,
    )

    fig3, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        phase,
        TotalEnergy_s,
        linewidth=3,
        label="Total Residual",
    )

    ax.plot(
        phase,
        SymEnergy_s,
        linewidth=2,
        label="Symmetric",
    )

    ax.plot(
        phase,
        AsymEnergy_s,
        linewidth=2,
        label="Antisymmetric",
    )

    ax.set_xlabel("Cardiac Phase (%)")

    ax.set_ylabel("Residual Energy")

    ax.set_title(
        f"Residual Energy Evolution\nBranch={branch_index}, Circle={circle_index}"
    )

    ax.legend()

    ax.grid(True)

    plt.tight_layout()

    if save_prefix is not None:
        fig3.savefig(
            f"{save_prefix}_residual_energy.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ==========================================================
    # Figure 4
    # Cardiac waveform comparison
    # ==========================================================

    raw_waveform = dataset_x[
        :,
        position_index,
        branch_index,
        circle_index,
    ]

    model_waveform = v_model[
        :,
        position_index,
        branch_index,
        circle_index,
    ]

    t = np.arange(len(raw_waveform))

    fig4, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        t,
        raw_waveform,
        label="Raw",
        linewidth=2,
    )

    ax.plot(
        t,
        model_waveform,
        label="Model",
        linewidth=2,
    )

    ax.set_xlabel("Cardiac Phase")
    ax.set_ylabel("Velocity")
    ax.set_title(
        f"Branch={branch_index}, Circle={circle_index}, Position={position_index}"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_prefix is not None:
        fig4.savefig(
            f"{save_prefix}_waveform_x{position_index}.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Figure 5
    # Harmonic spectrum
    # ==========================================================

    Qn = metrics["Q_n"]
    Qn_seg = Qn[:num_harmonics, branch_index, circle_index]

    n = np.arange(len(Qn_seg))

    fig5, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
    )

    axes[0].stem(
        n,
        np.abs(Qn_seg),
        basefmt=" ",
    )

    axes[0].set_ylabel(r"$|Q_n|$")
    axes[0].set_title(
        f"Harmonic Spectrum (Branch={branch_index}, Circle={circle_index})"
    )
    axes[0].grid(True)

    axes[1].stem(
        n,
        np.angle(Qn_seg),
        basefmt=" ",
    )

    axes[1].set_ylabel(r"$\angle Q_n$ (rad)")
    axes[1].set_xlabel("Harmonic order n")
    axes[1].grid(True)

    plt.tight_layout()

    if save_prefix is not None:
        fig5.savefig(
            f"{save_prefix}_harmonic_spectrum.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ==========================================================
    # Figure 6
    # Wall shear stress spectrum
    # ==========================================================

    tau_n = metrics["Tau_n"]

    tau_seg = tau_n[
        :num_harmonics,
        branch_index,
        circle_index,
    ]

    n = np.arange(len(tau_seg))

    fig6, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
    )

    axes[0].stem(
        n,
        np.abs(tau_seg),
        basefmt=" ",
    )

    axes[0].set_ylabel(r"$|\tau_n|$ (Pa)")
    axes[0].set_title("Wall Shear Stress Spectrum")
    axes[0].grid(True)

    axes[1].stem(
        n,
        np.angle(tau_seg),
        basefmt=" ",
    )

    axes[1].set_ylabel(r"$\angle \tau_n$ (rad)")
    axes[1].set_xlabel("Harmonic order n")
    axes[1].grid(True)

    plt.tight_layout()

    if save_prefix is not None:
        fig6.savefig(
            f"{save_prefix}_wall_shear_spectrum.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    

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

        # evaluate_womersley_model(
        #     metrics,
        #     branch_index=4,
        #     circle_index=5,
        #     position_index=8,
        #     save_prefix=None,  # "segment_3_2",
        # )


        return ProcessResult(metrics=metrics)
