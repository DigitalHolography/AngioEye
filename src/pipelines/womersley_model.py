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
    
    numerator = np.sum(np.conj(KBn) * Vn)
    
    denominator = np.sum(np.abs(KBn ** 2))
        
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
    disp = np.zeros((n_t, n_branches, n_radii), dtype=float)
    anti_model = np.zeros_like(dataset_x_antisymmetric, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            ratio = ratio_map[branch, circle]

            if not np.isfinite(ratio) or ratio <= 0:
                continue

            dx = pixel_size / ratio
            symmetric_gradient = np.gradient(v_model[:, :, branch, circle], dx, axis=1, edge_order=2)

            for t_idx in range(n_t):
                anti_profile = dataset_x_antisymmetric[t_idx, :, branch, circle]
                gradient_profile = symmetric_gradient[t_idx]
                valid = np.isfinite(anti_profile) & np.isfinite(gradient_profile)

                if np.sum(valid) < 3:
                    continue

                numerator = np.sum(gradient_profile[valid] * anti_profile[valid])
                denominator = np.sum(gradient_profile[valid] ** 2)

                if not np.isfinite(denominator) or denominator <= 0:
                    continue

                delta = -numerator / denominator
                disp[t_idx, branch, circle] = delta
                anti_model[t_idx, :, branch, circle] = -delta * gradient_profile

    disp_scale = np.max(np.abs(disp), axis=0, keepdims=True)
    disp = np.divide(disp, disp_scale, out=np.zeros_like(disp), where=disp_scale > 0)

    return disp, anti_model


def create_anti_jump_animation(temporal_jump_probability, mixture_reliable, overall_mean_probability, active_branch_count, high_segment_count, frame_spatial_class, temporal_event_class, global_coherence, global_motion_class, jump_threshold=0.9, coherence_threshold=0.7, animation_path="anti_disp_jump_animation.gif", fps=8):
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_transitions, n_branches, n_radii = temporal_jump_probability.shape
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1])
    matrix_axis = fig.add_subplot(grid[:, 0])
    probability_axis = fig.add_subplot(grid[0, 1])
    information_axis = fig.add_subplot(grid[1, 1])
    probability_cmap = plt.get_cmap("magma").copy()
    probability_cmap.set_bad(color="lightgray")

    probability_image = matrix_axis.imshow(np.ma.masked_invalid(temporal_jump_probability[0]), origin="lower", aspect="auto", vmin=0, vmax=1, cmap=probability_cmap)
    matrix_axis.set_xlabel("Circle")
    matrix_axis.set_ylabel("Branch")
    matrix_axis.set_xticks(np.arange(n_radii))
    matrix_axis.set_yticks(np.arange(n_branches))
    fig.colorbar(probability_image, ax=matrix_axis, label="Jump probability", fraction=0.046, pad=0.04)

    invalid_branch, invalid_circle = np.where(~mixture_reliable)
    matrix_axis.scatter(invalid_circle, invalid_branch, marker="x", color="red", s=45, linewidths=1.5, label="invalid / unreliable")
    isolated_marker = matrix_axis.scatter([], [], marker="o", facecolors="none", edgecolors="cyan", s=100, linewidths=2, label="isolated")
    clustered_marker = matrix_axis.scatter([], [], marker="s", facecolors="none", edgecolors="deepskyblue", s=100, linewidths=2, label="clustered")
    embedded_marker = matrix_axis.scatter([], [], marker="D", facecolors="none", edgecolors="lime", s=100, linewidths=2, label="embedded")
    sustained_marker = matrix_axis.scatter([], [], marker="*", facecolors="none", edgecolors="yellow", s=150, linewidths=2, label="sustained")
    matrix_axis.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=8)

    transition_axis = np.arange(n_transitions)
    probability_axis.plot(transition_axis, overall_mean_probability, color="tab:blue", label="mean probability")
    probability_axis.axhline(jump_threshold, color="tab:red", linestyle="--", label=f"jump threshold = {jump_threshold:.2f}")
    probability_cursor = probability_axis.axvline(0, color="black", linewidth=2)
    probability_axis.set_xlim(0, n_transitions - 1)
    probability_axis.set_ylim(0, 1)
    probability_axis.set_xlabel("Time transition")
    probability_axis.set_ylabel("Mean jump probability")

    count_axis = probability_axis.twinx()
    count_axis.plot(transition_axis, active_branch_count, color="tab:green", alpha=0.7, label="active branches")
    count_axis.plot(transition_axis, high_segment_count, color="tab:orange", alpha=0.7, label="high-probability segments")
    count_axis.set_ylabel("Synchronous count")
    count_axis.set_ylim(0, max(n_branches, int(np.nanmax(high_segment_count))) + 1)

    lines_1, labels_1 = probability_axis.get_legend_handles_labels()
    lines_2, labels_2 = count_axis.get_legend_handles_labels()
    probability_axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)
    information_axis.set_axis_off()
    information_text = information_axis.text(0.03, 0.95, "", va="top", ha="left", fontsize=12, transform=information_axis.transAxes)

    def marker_positions(mask):
        branch, circle = np.where(mask)
        return np.column_stack([circle, branch]) if branch.size > 0 else np.empty((0, 2))

    def update(frame):
        probability_image.set_data(np.ma.masked_invalid(temporal_jump_probability[frame]))
        probability_cursor.set_xdata([frame, frame])
        event_class = temporal_event_class[frame]
        isolated_marker.set_offsets(marker_positions(event_class == 1))
        clustered_marker.set_offsets(marker_positions(event_class == 2))
        embedded_marker.set_offsets(marker_positions(event_class == 3))
        sustained_marker.set_offsets(marker_positions(event_class == 4))
        spatial_class = frame_spatial_class[frame]
        coherence = global_coherence[frame]

        if spatial_class == 0:
            spatial_text = "no detected jump"
            border_color = "black"
        elif spatial_class == 1:
            spatial_text = "segment-local"
            border_color = "lime"
        elif spatial_class == 2:
            spatial_text = "branch-wide"
            border_color = "orange"
        elif global_motion_class[frame] == 1:
            spatial_text = "multi-branch global: coherent motion"
            border_color = "deepskyblue"
        else:
            spatial_text = "multi-branch global: incoherent degradation"
            border_color = "magenta"

        for spine in matrix_axis.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(4 if spatial_class == 3 else 2)

        coherence_text = f"{coherence:.3f}" if np.isfinite(coherence) else "nan"
        mean_text = f"{overall_mean_probability[frame]:.3f}" if np.isfinite(overall_mean_probability[frame]) else "nan"
        matrix_axis.set_title(f"Transition {frame} → {frame + 1}: {spatial_text}", color=border_color, fontsize=13)
        information_text.set_text(f"Time transition: {frame} → {frame + 1}\n\nSpatial classification:\n{spatial_text}\n\nMean jump probability: {mean_text}\nActive branches: {active_branch_count[frame]} / {n_branches}\nHigh-probability segments: {high_segment_count[frame]}\nGlobal coherence: {coherence_text}\n\nCoherence ≥ {coherence_threshold:.2f}:\ncoherent global motion\n\nCoherence < {coherence_threshold:.2f}:\nincoherent global degradation")
        return probability_image, probability_cursor, isolated_marker, clustered_marker, embedded_marker, sustained_marker, information_text

    animation = FuncAnimation(fig, update, frames=n_transitions, interval=1000 / fps, blit=False)
    animation.save(animation_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def evaluate_anti_model(disp, num_harmonics=3, num_examples=8, exclude_last=5, jump_threshold=0.9, isolation_mean_threshold=0.3, context_radius=5, event_merge_gap=3, max_isolated_duration=2, min_std_ratio=2.0, normal_std_floor_fraction=0.25, coherence_threshold=0.7, animation_fps=8, distribution_path="anti_disp_distribution.png", jump_analysis_path="anti_disp_jump_analysis.png", animation_path="anti_disp_jump_animation.gif"):
    n_t, n_branches, n_radii = disp.shape
    n_transitions = n_t - 1
    segment_valid = np.any(np.isfinite(disp) & (disp != 0), axis=0)
    disp_valid = np.isfinite(disp) & segment_valid[None, :, :]
    phase = 2 * np.pi * np.arange(n_t) / n_t
    design_matrix = np.column_stack([np.ones(n_t)] + [f(phase * n) for n in range(1, num_harmonics + 1) for f in (np.cos, np.sin)])
    disp_smooth = np.full_like(disp, np.nan, dtype=float)
    disp_residual = np.full_like(disp, np.nan, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            time_valid = disp_valid[:, branch, circle]

            if np.sum(time_valid) < design_matrix.shape[1]:
                continue

            coefficients = np.linalg.lstsq(design_matrix[time_valid], disp[time_valid, branch, circle], rcond=None)[0]
            disp_smooth[:, branch, circle] = design_matrix @ coefficients
            disp_residual[time_valid, branch, circle] = disp[time_valid, branch, circle] - disp_smooth[time_valid, branch, circle]

    temporal_difference = np.full((n_transitions, n_branches, n_radii), np.nan, dtype=float)
    difference_valid = disp_valid[1:] & disp_valid[:-1] & np.isfinite(disp_residual[1:]) & np.isfinite(disp_residual[:-1])
    raw_difference = np.diff(disp_residual, axis=0)
    temporal_difference[difference_valid] = raw_difference[difference_valid]
    fitting_valid = difference_valid.copy()

    if exclude_last > 0:
        fitting_valid[max(0, n_transitions - exclude_last):] = False

    temporal_jump_probability = np.full_like(temporal_difference, np.nan, dtype=float)

    def gaussian_density(values, std):
        return np.exp(-0.5 * (values / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    def fit_gaussian_mixture(values):
        values = values[np.isfinite(values)]

        if values.size < 10:
            return None

        scale = np.std(values)

        if not np.isfinite(scale) or scale <= 1e-12:
            return None

        normal_std = max(0.5 * scale, 1e-12)
        jump_std = max(2.0 * scale, 2.0 * normal_std)
        jump_probability = 0.1

        for _ in range(200):
            normal_density = gaussian_density(values, normal_std)
            jump_density = gaussian_density(values, jump_std)
            denominator = (1.0 - jump_probability) * normal_density + jump_probability * jump_density
            responsibility = np.divide(jump_probability * jump_density, denominator, out=np.zeros_like(values), where=denominator > 0)
            normal_weight = np.sum(1.0 - responsibility)
            jump_weight = np.sum(responsibility)
            new_jump_probability = np.mean(responsibility)
            new_normal_std = np.sqrt(np.sum((1.0 - responsibility) * values**2) / max(normal_weight, 1e-12))
            new_jump_std = np.sqrt(np.sum(responsibility * values**2) / max(jump_weight, 1e-12))

            if new_normal_std > new_jump_std:
                new_normal_std, new_jump_std = new_jump_std, new_normal_std
                new_jump_probability = 1.0 - new_jump_probability

            new_jump_probability = np.clip(new_jump_probability, 1e-6, 1.0 - 1e-6)
            new_normal_std = max(new_normal_std, 1e-12)
            new_jump_std = max(new_jump_std, new_normal_std)
            old_parameters = np.array([jump_probability, normal_std, jump_std])
            new_parameters = np.array([new_jump_probability, new_normal_std, new_jump_std])
            jump_probability, normal_std, jump_std = new_parameters

            if np.allclose(old_parameters, new_parameters, rtol=1e-6, atol=1e-8):
                break

        return jump_probability, normal_std, jump_std

    raw_jump_parameters = np.full((n_branches, n_radii, 3), np.nan, dtype=float)

    for branch in range(n_branches):
        for circle in range(n_radii):
            circle_valid = fitting_valid[:, branch, circle]
            parameters = fit_gaussian_mixture(temporal_difference[circle_valid, branch, circle])

            if parameters is not None:
                raw_jump_parameters[branch, circle] = parameters

    raw_normal_std = raw_jump_parameters[:, :, 1]
    raw_jump_std = raw_jump_parameters[:, :, 2]
    raw_std_ratio = np.divide(raw_jump_std, raw_normal_std, out=np.full((n_branches, n_radii), np.nan), where=raw_normal_std > 0)
    preliminary_reliable = np.isfinite(raw_std_ratio) & (raw_std_ratio >= min_std_ratio)
    reference_values = raw_normal_std[preliminary_reliable]
    normal_std_reference = np.median(reference_values) if reference_values.size > 0 else np.nan
    normal_std_floor = normal_std_floor_fraction * normal_std_reference if np.isfinite(normal_std_reference) else 0.0
    jump_parameters = raw_jump_parameters.copy()
    jump_parameters[:, :, 1] = np.maximum(jump_parameters[:, :, 1], normal_std_floor)
    regularized_std_ratio = np.divide(jump_parameters[:, :, 2], jump_parameters[:, :, 1], out=np.full((n_branches, n_radii), np.nan), where=jump_parameters[:, :, 1] > 0)
    mixture_reliable = np.isfinite(regularized_std_ratio) & (regularized_std_ratio >= min_std_ratio)

    for branch in range(n_branches):
        for circle in range(n_radii):
            if not mixture_reliable[branch, circle]:
                continue

            circle_valid = fitting_valid[:, branch, circle]
            jump_probability, normal_std, jump_std = jump_parameters[branch, circle]
            values = temporal_difference[circle_valid, branch, circle]
            normal_density = gaussian_density(values, normal_std)
            jump_density = gaussian_density(values, jump_std)
            denominator = (1.0 - jump_probability) * normal_density + jump_probability * jump_density
            temporal_jump_probability[circle_valid, branch, circle] = np.divide(jump_probability * jump_density, denominator, out=np.full_like(values, np.nan), where=denominator > 0)

    probability_valid = np.isfinite(temporal_jump_probability)
    high_probability = probability_valid & (temporal_jump_probability >= jump_threshold)
    valid_count = np.sum(probability_valid, axis=(1, 2))
    overall_mean_probability = np.divide(np.nansum(temporal_jump_probability, axis=(1, 2)), valid_count, out=np.full(n_transitions, np.nan), where=valid_count > 0)
    branch_high_count = np.sum(high_probability, axis=2)
    active_branch_count = np.sum(branch_high_count > 0, axis=1)
    high_segment_count = np.sum(high_probability, axis=(1, 2))
    frame_spatial_class = np.zeros(n_transitions, dtype=int)
    frame_spatial_class[high_segment_count > 0] = 1
    frame_spatial_class[np.max(branch_high_count, axis=1) >= 2] = 2
    frame_spatial_class[active_branch_count >= 2] = 3

    normal_std = jump_parameters[:, :, 1]
    standardized_difference = np.divide(temporal_difference, normal_std[None, :, :], out=np.full_like(temporal_difference, np.nan), where=normal_std[None, :, :] > 0)
    flagged_standardized_difference = np.where(high_probability, standardized_difference, np.nan)
    signed_sum = np.abs(np.nansum(flagged_standardized_difference, axis=(1, 2)))
    absolute_sum = np.nansum(np.abs(flagged_standardized_difference), axis=(1, 2))
    global_coherence = np.divide(signed_sum, absolute_sum, out=np.full(n_transitions, np.nan), where=absolute_sum > 0)
    global_motion_class = np.zeros(n_transitions, dtype=int)
    global_frame = frame_spatial_class == 3
    global_motion_class[global_frame & (global_coherence >= coherence_threshold)] = 1
    global_motion_class[global_frame & ((global_coherence < coherence_threshold) | ~np.isfinite(global_coherence))] = 2
    events = []

    for branch in range(n_branches):
        for circle in range(n_radii):
            if not mixture_reliable[branch, circle]:
                continue

            high_times = np.flatnonzero(high_probability[:, branch, circle])

            if high_times.size == 0:
                continue

            groups = []
            current_group = [int(high_times[0])]

            for time_idx in high_times[1:]:
                if time_idx - current_group[-1] <= event_merge_gap:
                    current_group.append(int(time_idx))
                else:
                    groups.append(current_group)
                    current_group = [int(time_idx)]

            groups.append(current_group)

            for group in groups:
                start = group[0]
                end = group[-1]
                duration = end - start + 1
                group_probability = temporal_jump_probability[group, branch, circle]
                peak_time = group[int(np.nanargmax(group_probability))]
                peak_probability = temporal_jump_probability[peak_time, branch, circle]
                context_start = max(0, start - context_radius)
                context_end = min(n_transitions - 1, end + context_radius)
                context_values = temporal_jump_probability[context_start:context_end + 1, branch, circle].copy()
                context_values[start - context_start:end - context_start + 1] = np.nan
                finite_context = context_values[np.isfinite(context_values)]
                context_mean = np.mean(finite_context) if finite_context.size > 0 else np.nan
                quiet_context = np.isfinite(context_mean) and context_mean < isolation_mean_threshold

                if duration <= max_isolated_duration and quiet_context:
                    temporal_type = "isolated"
                elif duration > max_isolated_duration and quiet_context:
                    temporal_type = "clustered"
                elif duration <= max_isolated_duration:
                    temporal_type = "embedded"
                else:
                    temporal_type = "sustained"

                interval_active_branches = int(np.max(active_branch_count[start:end + 1]))
                interval_branch_segments = int(np.max(branch_high_count[start:end + 1, branch]))

                if interval_active_branches >= 2:
                    spatial_type = "multi-branch global"
                    global_interval = np.arange(start, end + 1)
                    global_interval = global_interval[frame_spatial_class[global_interval] == 3]
                    coherence_weights = high_segment_count[global_interval].astype(float)
                    coherence_values = global_coherence[global_interval]
                    coherence_valid = np.isfinite(coherence_values) & (coherence_weights > 0)

                    if np.any(coherence_valid):
                        event_coherence = np.average(coherence_values[coherence_valid], weights=coherence_weights[coherence_valid])
                    else:
                        event_coherence = np.nan

                    global_direction = "coherent motion" if np.isfinite(event_coherence) and event_coherence >= coherence_threshold else "incoherent degradation"
                elif interval_branch_segments >= 2:
                    spatial_type = "branch-wide"
                    event_coherence = np.nan
                    global_direction = "not global"
                else:
                    spatial_type = "segment-local"
                    event_coherence = np.nan
                    global_direction = "not global"

                events.append({"branch": branch, "circle": circle, "start": start, "end": end, "duration": duration, "peak_time": peak_time, "peak_probability": peak_probability, "context_mean": context_mean, "temporal_type": temporal_type, "spatial_type": spatial_type, "active_branches": interval_active_branches, "branch_segments": interval_branch_segments, "global_coherence": event_coherence, "global_direction": global_direction})

    temporal_event_class = np.zeros((n_transitions, n_branches, n_radii), dtype=int)
    temporal_name_to_code = {"isolated": 1, "clustered": 2, "embedded": 3, "sustained": 4}

    for event in events:
        temporal_event_class[event["start"]:event["end"] + 1, event["branch"], event["circle"]] = temporal_name_to_code[event["temporal_type"]]

    spatial_names = ["segment-local", "branch-wide", "multi-branch global"]
    temporal_names = ["isolated", "clustered", "embedded", "sustained"]
    event_count = np.zeros((3, 4), dtype=int)
    global_direction_event_count = np.zeros((2, 4), dtype=int)

    for event in events:
        spatial_idx = spatial_names.index(event["spatial_type"])
        temporal_idx = temporal_names.index(event["temporal_type"])
        event_count[spatial_idx, temporal_idx] += 1

        if event["spatial_type"] == "multi-branch global":
            direction_idx = 0 if event["global_direction"] == "coherent motion" else 1
            global_direction_event_count[direction_idx, temporal_idx] += 1

    reliable_count = int(np.sum(mixture_reliable))
    fitted_count = int(np.sum(np.all(np.isfinite(jump_parameters), axis=2)))
    usable_transitions = max(n_transitions - exclude_last, 1)
    event_denominator = max(reliable_count * usable_transitions, 1)
    event_rate = 100.0 * event_count / event_denominator
    probability_point_count = int(np.sum(probability_valid))
    high_probability_fraction = np.sum(high_probability) / max(probability_point_count, 1)
    valid_frame = valid_count > 0
    valid_frame_count = max(int(np.sum(valid_frame)), 1)
    spatial_frame_fraction = np.array([np.sum(frame_spatial_class == index) / valid_frame_count for index in range(1, 4)])
    split_transition = usable_transitions // 2
    early_values = overall_mean_probability[:split_transition]
    late_values = overall_mean_probability[split_transition:usable_transitions]
    early_mean = np.nanmean(early_values) if np.any(np.isfinite(early_values)) else np.nan
    late_mean = np.nanmean(late_values) if np.any(np.isfinite(late_values)) else np.nan
    late_early_ratio = late_mean / early_mean if np.isfinite(early_mean) and early_mean > 0 else np.nan
    reliable_fraction = reliable_count / max(fitted_count, 1)
    coherent_global_frame_fraction = np.sum(global_motion_class == 1) / valid_frame_count
    incoherent_global_frame_fraction = np.sum(global_motion_class == 2) / valid_frame_count
    global_coherence_values = global_coherence[global_frame & np.isfinite(global_coherence)]
    mean_global_coherence = np.mean(global_coherence_values) if global_coherence_values.size > 0 else np.nan
    summary_metrics = np.array([fitted_count, reliable_count, reliable_fraction, np.nanmean(overall_mean_probability), high_probability_fraction, early_mean, late_mean, late_early_ratio, spatial_frame_fraction[0], spatial_frame_fraction[1], spatial_frame_fraction[2], len(events) / event_denominator * 100.0, normal_std_reference, normal_std_floor, mean_global_coherence, coherent_global_frame_fraction, incoherent_global_frame_fraction])

    residual_values = disp_residual[np.isfinite(disp_residual) & disp_valid]
    difference_values = temporal_difference[fitting_valid & np.isfinite(temporal_difference)]
    std_ratio = regularized_std_ratio
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if residual_values.size > 0:
        axes[0].hist(residual_values, bins=60, density=True, alpha=0.7)

    axes[0].set_xlabel("Normalized displacement residual")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("All-point residual distribution")

    if difference_values.size > 0:
        axes[1].hist(difference_values, bins=60, density=True, alpha=0.7)

    axes[1].set_xlabel("Temporal residual difference")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title(f"All branches, last {exclude_last} excluded")

    ratio_cmap = plt.get_cmap("viridis").copy()
    ratio_cmap.set_bad(color="lightgray")
    ratio_max = max(5, np.nanpercentile(std_ratio, 95)) if np.any(np.isfinite(std_ratio)) else 5
    ratio_image = axes[2].imshow(np.ma.masked_invalid(std_ratio), origin="lower", aspect="auto", vmin=1, vmax=ratio_max, cmap=ratio_cmap)
    axes[2].set_xlabel("Circle")
    axes[2].set_ylabel("Branch")
    axes[2].set_title(f"Jump std / regularized normal std\nreliable fraction = {reliable_fraction:.3f}")
    fig.colorbar(ratio_image, ax=axes[2], label="Standard-deviation ratio")

    unreliable_branch, unreliable_circle = np.where(np.isfinite(std_ratio) & ~mixture_reliable)
    axes[2].scatter(unreliable_circle, unreliable_branch, marker="x", color="red", s=40, linewidths=1.5, label=f"unreliable: ratio < {min_std_ratio:g}")

    if unreliable_branch.size > 0:
        axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    flattened_probability = temporal_jump_probability.reshape(n_transitions, n_branches * n_radii)
    events.sort(key=lambda event: event["peak_probability"], reverse=True)
    selected_events = []

    for spatial_type in spatial_names:
        for temporal_type in temporal_names:
            matching_events = [event for event in events if event["spatial_type"] == spatial_type and event["temporal_type"] == temporal_type]

            if matching_events:
                selected_events.append(matching_events[0])

    for event in events:
        if len(selected_events) >= num_examples:
            break

        if event not in selected_events:
            selected_events.append(event)

    selected_events = selected_events[:num_examples]
    fig = plt.figure(figsize=(14, 7 + 3 * max(len(selected_events), 1)))
    grid = fig.add_gridspec(max(len(selected_events), 1) + 1, 2, height_ratios=[1.7] + [1] * max(len(selected_events), 1))
    heatmap_axis = fig.add_subplot(grid[0, 0])
    heatmap_cmap = plt.get_cmap("magma").copy()
    heatmap_cmap.set_bad(color="lightgray")
    heatmap = heatmap_axis.imshow(np.ma.masked_invalid(flattened_probability), origin="lower", aspect="auto", vmin=0, vmax=1, cmap=heatmap_cmap)
    heatmap_axis.set_xlabel("Branch-circle index")
    heatmap_axis.set_ylabel("Time transition")
    heatmap_axis.set_title("Temporal jump probability, all branches")
    fig.colorbar(heatmap, ax=heatmap_axis, label="Jump probability")

    for branch in range(1, n_branches):
        heatmap_axis.axvline(branch * n_radii - 0.5, color="white", linewidth=0.7, alpha=0.7)

    heatmap_axis.set_xticks([branch * n_radii + (n_radii - 1) / 2 for branch in range(n_branches)])
    heatmap_axis.set_xticklabels([f"B{branch}" for branch in range(n_branches)])

    spatial_marker = {"segment-local": "o", "branch-wide": "s", "multi-branch global": "D"}
    temporal_color = {"isolated": "cyan", "clustered": "deepskyblue", "embedded": "lime", "sustained": "yellow"}

    for event in selected_events:
        x_position = event["branch"] * n_radii + event["circle"]
        heatmap_axis.plot(x_position, event["peak_time"], marker=spatial_marker[event["spatial_type"]], markerfacecolor="none", markeredgecolor=temporal_color[event["temporal_type"]], markersize=8, markeredgewidth=1.8)

    transition_axis = np.arange(n_transitions)
    mean_axis = fig.add_subplot(grid[0, 1])
    mean_axis.plot(transition_axis, overall_mean_probability, color="tab:blue", label="mean probability")
    mean_axis.axhline(jump_threshold, color="tab:red", linestyle="--", label=f"jump threshold = {jump_threshold:.2f}")

    if exclude_last > 0:
        mean_axis.axvspan(max(0, n_transitions - exclude_last), n_transitions - 1, color="gray", alpha=0.25, label="excluded")

    count_axis = mean_axis.twinx()
    count_axis.step(transition_axis, active_branch_count, where="mid", color="tab:green", label="active branches")
    count_axis.step(transition_axis, high_segment_count, where="mid", color="tab:orange", alpha=0.6, label="high-probability segments")
    count_axis.axhline(2, color="tab:green", linestyle=":", label="multi-branch threshold")
    count_axis.set_ylabel("Synchronous count")
    count_axis.set_ylim(0, max(n_branches, int(np.max(high_segment_count))) + 1)

    mean_axis.set_xlabel("Time transition")
    mean_axis.set_ylabel("Mean jump probability")
    mean_axis.set_ylim(0, 1)
    mean_axis.set_title(f"All-branch probability\nlate/early ratio = {late_early_ratio:.3f}")
    lines_1, labels_1 = mean_axis.get_legend_handles_labels()
    lines_2, labels_2 = count_axis.get_legend_handles_labels()
    mean_axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    if selected_events:
        plot_time = np.arange(n_t)

        for row, event in enumerate(selected_events, start=1):
            branch = event["branch"]
            circle = event["circle"]
            start = event["start"]
            end = event["end"]
            peak_time = event["peak_time"]
            normal_std = jump_parameters[branch, circle, 1]
            jump_std = jump_parameters[branch, circle, 2]
            ratio = jump_std / max(normal_std, 1e-12)
            context_text = f"{event['context_mean']:.3f}" if np.isfinite(event["context_mean"]) else "nan"
            coherence_text = f"{event['global_coherence']:.3f}" if np.isfinite(event["global_coherence"]) else "nan"
            direction_text = f" / {event['global_direction']}" if event["spatial_type"] == "multi-branch global" else ""
            axis = fig.add_subplot(grid[row, :])
            axis.plot(plot_time, disp[:, branch, circle], label="disp")
            axis.plot(plot_time, disp_smooth[:, branch, circle], label="disp smooth")
            axis.axvspan(start, end + 1, color="red", alpha=0.12)
            axis.axvline(peak_time, color="red", linestyle="--")
            axis.axvline(peak_time + 1, color="red", linestyle="--")
            axis.set_ylabel("Normalized displacement")
            axis.set_title(f"{event['spatial_type']}{direction_text} / {event['temporal_type']}: branch {branch}, circle {circle}, transitions {start} → {end + 1}, peak probability = {event['peak_probability']:.4f}, context mean = {context_text}, active branches = {event['active_branches']}, branch segments = {event['branch_segments']}, coherence = {coherence_text}, std ratio = {ratio:.2f}")
            axis.legend()

        axis.set_xlabel("Time point")

    else:
        axis = fig.add_subplot(grid[1, :])
        axis.text(0.5, 0.5, "No reliable jump events", ha="center", va="center")
        axis.set_axis_off()

    fig.tight_layout()
    fig.savefig(jump_analysis_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    create_anti_jump_animation(temporal_jump_probability, mixture_reliable, overall_mean_probability, active_branch_count, high_segment_count, frame_spatial_class, temporal_event_class, global_coherence, global_motion_class, jump_threshold=jump_threshold, coherence_threshold=coherence_threshold, animation_path=animation_path, fps=animation_fps)

    return disp_smooth, disp_residual, temporal_difference, temporal_jump_probability, jump_parameters, mixture_reliable, overall_mean_probability, active_branch_count, high_segment_count, frame_spatial_class, temporal_event_class, global_coherence, global_motion_class, event_count, global_direction_event_count, event_rate, summary_metrics


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

        disp, anti_model = fit_antisymmetric_curve(dataset_x_antisymmetric, v_model, ratio_map)
        disp_smooth, disp_residual, temporal_difference, temporal_jump_probability, jump_parameters, mixture_reliable, overall_mean_probability, active_branch_count, high_segment_count, frame_spatial_class, temporal_event_class, global_coherence, global_motion_class, jump_event_count, global_direction_event_count, jump_event_rate, jump_summary_metrics = evaluate_anti_model(disp)
        
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
        metrics["disp"] = np.asarray(disp)
        metrics["anti_model"] = np.asarray(anti_model)


        return ProcessResult(metrics=metrics)
