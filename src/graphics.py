import os
import re
import tempfile
import zipfile
from collections import defaultdict
from tkinter import Tk, filedialog
import shutil
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

GRAPHICS_SUPPORT_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/graphics_support/"
METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
SELECTED_METRICS_PNG = {
    "RI",
    "PI",
    "crest_factor",
    "t50_over_T",
    "R_VTI",
    "SF_VTI",
    "spectral_entropy",
    "mu_t_over_T",
    "sigma_t_over_T",
    "delta_phi2",
    "t_max_over_T",
    "t_min_over_T",
    "Delta_t_over_T",
    "t_up_over_T",
    "t_down_over_T",
    "S_decay",
    "Delta_DTI",
    "E_high_over_E_total",
    "E_low_over_E_total",
    "R_SD",
    "slope_fall_normalized",
    "slope_rise_normalized",
    "gamma_t",
    "mu_h",
    "sigma_h",
    "N_eff",
    "N_eff_over_T",
    "E_recon_H_MAX",
    "Q_t_skew",
    "Q_t_width",
    "Q_d_skew",
    "Q_d_width",
    "R_Q_d",
    "R_Q_t",
    "v_end_over_v_mean",
    "E_slope",
    "E_curv",
    "phase_locking_residual",
    "W50_over_T",
    "W75_over_T",
    "N_H_over_T",

}
SIGNAL_DATASET_PATH = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
METRIC_ALIASES = {
    "Hspec": "spectral_entropy",
}
EPS = 1e-12
H_MAX = 10
H_LOW_MAX = 2
H_HIGH_MIN = 4
H_HIGH_MAX = 8
LATEX_FORMULAS = {
    "RI": r"$\rm RI$",
    "crest_factor": r"$\rm CF$",
    "t50_over_T": r"$t_{50}/T$",
    "R_VTI": r"$R_{VTI}$",
    "spectral_entropy": r"$H_{spec}$",
    "mu_t_over_T": r"$\mu_t/T$",
    "PI": r"$\rm PI$",
    "SF_VTI": r"$SF_{VTI}$",
    "sigma_t_over_T": r"$\sigma_t/T$",
    "delta_phi2": r"$\Delta\phi_2$",
    "t_max_over_T": r"$t_{\mathrm{max}}/T$",
    "t_min_over_T": r"$t_{\mathrm{min}}/T$",
    "Delta_t_over_T": r"$\Delta_{\mathrm{t}}/T$",
    "t_up_over_T": r"$t_{\mathrm{up}}/T$",
    "t_down_over_T": r"$t_{\mathrm{down}}/T$",
    "S_decay": r"$S_{\mathrm{decay}}$",
    "Delta_DTI": r"$\Delta_{\mathrm{DTI}}$",
    "E_high_over_E_total": r"$E_{\mathrm{high}}/E_{\mathrm{total}}$",
    "E_low_over_E_total": r"$E_{\mathrm{low}}/E_{\mathrm{total}}$",
    "R_SD": r"$R_{SD}$",
    "slope_fall_normalized": r"$S_{\mathrm{fall}}$",
    "slope_rise_normalized": r"$S_{\mathrm{rise}}$",
    "gamma_t": r"$\gamma_t$",
    "mu_h": r"$\mu_h$",
    "sigma_h": r"$\sigma_h$",
    "N_eff": r"$N_{\mathrm{eff}}$",
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",
    "E_recon_H_MAX": r"$E_{\mathrm{recon},H_{\max}}$",
    "Q_t_skew": r"$Q_{\mathrm{t,skew}}$",
    "Q_t_width": r"$Q_{\mathrm{t,width}}$",
    "Q_d_skew": r"$Q_{\mathrm{d,skew}}$",
    "Q_d_width": r"$Q_{\mathrm{d,width}}$",
    "R_Q_t": r"$R_{\mathrm{Q_{{t}}}}$",
    "R_Q_d": r"$R_{\mathrm{Q_{{d}}}}$",
    "v_end_over_v_mean": r"$R_{EM}$",
    "E_slope": r"$E_{\mathrm{slope}}$",
    "E_curv": r"$E_{\mathrm{curv}}$",
    "phase_locking_residual": r"$E_{\phi}$",
    "W50_over_T": r"$W_{50}/T$",
    "W75_over_T": r"$W_{75}/T$",
    "N_H_over_T" : r"$N_{H}/T$",
}


def extract_graphics_support(h5_path, mode="bandlimited"):
    base = f"{GRAPHICS_SUPPORT_FOLDER}{mode}"
    out = {}

    with h5py.File(h5_path, "r") as f:
        if base not in f:
            return None
        grp = f[base]
        for key in grp.keys():
            arr = np.array(grp[key])
            out[key] = arr.item() if arr.shape == () else arr

    return out


def select_support_beat(support, beat_idx):
    out = {}
    for k, v in support.items():
        arr = np.asarray(v)
        if arr.ndim == 2:
            if k in {
                "harmonic_magnitudes",
                "harmonic_weights",
                "harmonic_phases",
                "delta_phi_all",
            }:
                out[k] = arr[beat_idx, :]
            else:
                out[k] = arr[:, beat_idx]
        elif arr.ndim == 1 and arr.shape[0] > beat_idx:
            out[k] = arr[beat_idx]
        else:
            out[k] = v
    return out


def harmonic_weights_from_signal(v, h_max=H_MAX):
    V, vb, H, w_h = harmonic_pack(v)
    if V is None or H is None or H < 1:
        return None, None, None, None

    mags = np.abs(V[1 : H + 1])
    mags = np.where(np.isfinite(mags), mags, np.nan)
    s = float(np.nansum(mags))
    if s <= 0:
        return V, vb, H, None

    w_h = mags / (s + EPS)
    return V, vb, H, w_h


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def quantile_idx_from_cumsum(C, q):
    idx = int(np.searchsorted(C, q, side="left"))
    idx = max(0, min(len(C) - 1, idx))
    return idx


def safe_rectified_signal(sig):
    sig = np.asarray(sig, dtype=float)
    return np.where(np.isfinite(sig), np.maximum(sig, 0.0), np.nan)


def rectify_keep_nan(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)


def draw_inline_formulas_ax(ax, formulas, y=0.5, fontsize=16, gap=0.03):
    if not formulas:
        return
    if isinstance(formulas, str):
        formulas = [formulas]

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x = 0.0
    for fml in formulas:
        t = ax.text(
            x, y, fml, ha="left", va="center", fontsize=fontsize, transform=ax.transAxes
        )
        t.set_clip_on(False)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        w = bbox.width / ax.bbox.width  # largeur relative dans l'axe header
        x += w + gap


def draw_formula_header(fig, formula, y=0.98, fontsize=14, pad_top=0.86):
    """
    formula: str OU list/tuple[str]
    - 1 formule -> alignée à gauche
    - n formules -> réparties horizontalement
    """
    if not formula:
        return

    fig.subplots_adjust(top=pad_top)  # réserve de la place en haut

    if isinstance(formula, (list, tuple)):
        n = len(formula)
        if n == 1:
            fig.text(0.02, y, formula[0], ha="left", va="top", fontsize=fontsize)
            return

        # positions : 2 -> (gauche, droite), 3 -> (gauche, centre, droite), etc.
        xs = np.linspace(0.02, 0.98, n)

        for i, (x, part) in enumerate(zip(xs, formula, strict=False)):
            ha = "center"
            if i == 0:
                ha = "left"
            elif i == n - 1:
                ha = "right"
            fig.text(x, y, part, ha=ha, va="top", fontsize=fontsize)
    else:
        fig.text(0.02, y, formula, ha="left", va="top", fontsize=fontsize)


def quantile_time_over_T(v, q):
    """
    Reproduit ArterialSegExample._quantile_time_over_T
    Ici pas besoin de Tbeat : t_q/T = idx/n
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0 or not np.any(np.isfinite(v)):
        return np.nan, None, None  # value, cum, idx

    w = np.where(np.isfinite(v), v, np.nan)
    m0 = float(np.nansum(w))
    if m0 <= 0:
        return np.nan, None, None

    c = np.cumsum(w) / m0
    idx = int(np.searchsorted(c, q, side="left"))
    idx = max(0, min(v.size - 1, idx))

    return float(idx / v.size), c, idx


def circular_mean(angles):
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * angles))))


def circular_std(angles):
    """
    Circular std basée sur la resultant length.
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan

    R = np.abs(np.mean(np.exp(1j * angles)))
    R = np.clip(R, EPS, 1.0)
    return float(np.sqrt(-2.0 * np.log(R)))


def compute_group_delta_phi_stats(zip_path, mode="bandlimited"):
    group_values = defaultdict(lambda: defaultdict(list))

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = [f for f in files if f.endswith(".h5")]
            if not h5_files:
                continue

            group_name = os.path.basename(root)
            if root == tmpdir:
                group_name = "all"

            for file in h5_files:
                h5_path = os.path.join(root, file)
                try:
                    support = extract_graphics_support(h5_path, mode=mode)
                    if not support:
                        continue

                    dphi = np.asarray(support.get("delta_phi_all", []), dtype=float)
                    for i, val in enumerate(dphi, start=2):
                        if np.isfinite(val):
                            group_values[group_name][i].append(val)
                except Exception:
                    continue

    group_stats = {}
    for group, harmonics_dict in group_values.items():
        hs = sorted(harmonics_dict.keys())
        mean_vals = []
        std_vals = []

        for h in hs:
            vals = np.array(harmonics_dict[h], dtype=float)
            mean_vals.append(circular_mean(vals))
            std_vals.append(circular_std(vals))

        group_stats[group] = {
            "h": np.array(hs, dtype=int),
            "mean": np.array(mean_vals, dtype=float),
            "std": np.array(std_vals, dtype=float),
            "n_files": sum(len(v) > 0 for v in harmonics_dict.values()),
        }

    return group_stats


def plot_group_delta_phi_stats(ax, group_stats, group_name):
    if group_name not in group_stats:
        ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center")
        ax.axis("off")
        return

    data = group_stats[group_name]
    hs = data["h"]
    mu = data["mean"]
    sigma = data["std"]

    ax.errorbar(
        hs,
        mu,
        yerr=sigma,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
    )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(np.pi, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(-np.pi, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlim(1.5, max(hs) + 0.5)
    ax.set_ylim(-1.1 * np.pi, 1.1 * np.pi)
    ax.set_xticks(hs)
    ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
    ax.set_ylabel(r"Group mean $\delta\phi_n$ (rad)", fontsize=14, labelpad=12)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_title(group_name, fontsize=14)


def harmonic_pack(v):
    """
    Reproduit _harmonic_pack sans Tbeat (pas nécessaire pour vb)
    """
    v = np.asarray(v, dtype=float)
    w = np.where(np.isfinite(v), v, np.nan)
    n = w.size
    if n < 2:
        return None, None, None

    Vfull = np.fft.rfft(w) / float(n)
    H = int(min(H_MAX, Vfull.size - 1))
    V = Vfull[: H + 1].copy()

    Vtrunc = np.zeros_like(Vfull)
    Vtrunc[: H + 1] = V
    vb = np.fft.irfft(Vtrunc * float(n), n=n)
    mags = np.abs(V[1 : H + 1])
    mags = np.where(np.isfinite(mags), mags, np.nan)
    s = float(np.nansum(mags))
    if s <= 0:
        return V, vb, H, None
    w = mags / (s + EPS)
    return V, vb, H, w


def crest_factor_from_vb(vb):
    if vb is None or vb.size == 0:
        return np.nan
    x = np.where(np.isfinite(vb), vb, np.nan)
    rms = float(np.sqrt(np.nanmean(x * x)))
    if rms <= 0:
        return np.nan
    return float(np.nanmax(x) / rms)


def spectral_entropy_from_harmonics(V):
    """
    Reproduit _spectral_entropy_from_harmonics (appelé spectral_entropy)
    """
    if V is None or V.size < 2:
        return np.nan
    mags = np.abs(V[1:])
    mags = np.where(np.isfinite(mags), mags, np.nan)
    s = float(np.nansum(mags))
    if s <= 0:
        return np.nan
    p = mags / s
    p = np.clip(p, EPS, 1.0)
    return float(-np.nansum(p * np.log(p)))


def find_control_group_name(groups):
    # cherche "control", "controls", "ctrl" etc.
    for g in groups:
        if g is None:
            continue
        gl = str(g).lower()
        if "control" in gl or gl in {"ctrl", "ctl", "controls"}:
            return g
    return None


def extract_graphics_support(h5_path, mode="bandlimited"):
    base = f"{GRAPHICS_SUPPORT_FOLDER}{mode}"
    out = {}

    with h5py.File(h5_path, "r") as f:
        if base not in f:
            return None

        grp = f[base]
        for key in grp.keys():
            arr = np.array(grp[key])
            if arr.shape == ():
                out[key] = arr.item()
            else:
                out[key] = arr

    return out


def plot_metric_illustration(ax, metric, support, path=None):
    if not support:
        ax.text(0.5, 0.5, "No graphics support", ha="center", va="center")
        ax.axis("off")
        return

    tau = np.asarray(support["tau"], dtype=float)
    sig = np.asarray(support["signal_mean"], dtype=float)
    C = np.asarray(support.get("cumulative", []), dtype=float)
    vb = np.asarray(support.get("vb", []), dtype=float)
    dvdt = np.asarray(support.get("dvdt", []), dtype=float)
    d2vdt2 = np.asarray(support.get("d2vdt2", []), dtype=float)
    harmonic_weights = np.asarray(support.get("harmonic_weights", []), dtype=float)
    harmonic_magnitudes = np.asarray(
        support.get("harmonic_magnitudes", []), dtype=float
    )
    harmonic_phases = np.asarray(support.get("harmonic_phases", []), dtype=float)
    delta_phi_all = np.asarray(support.get("delta_phi_all", []), dtype=float)

    n = sig.size
    if n < 2:
        ax.text(0.5, 0.5, "Signal too short", ha="center", va="center")
        ax.axis("off")
        return

    # --- helpers ---
    def _y_at(x0, x_grid, y_grid):
        y = np.asarray(y_grid, dtype=float)
        x = np.asarray(x_grid, dtype=float)
        if len(x) < 2:
            return np.nan
        y2 = np.where(np.isfinite(y), y, 0.0)
        return float(np.interp(x0, x, y2))

    def vline_to_curve(x0, x_grid, y_grid, y0=0.0, **kwargs):
        y1 = _y_at(x0, x_grid, y_grid)
        if np.isfinite(y1):
            ax.vlines(x0, y0, y1, **kwargs)
        return y1

    def hline_label(y, label, va="bottom"):
        ax.axhline(y, linestyle="--", linewidth=1, color="black")
        ax.text(
            0.98,
            y,
            f" {label}={y:.3g}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va=va,
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )

    def info_box(lines, fontsize=12):
        """
        lines: str or list[str]
        Draws the same top-left boxed annotation for all metrics.
        """
        if not lines:
            return
        if isinstance(lines, str):
            text = lines
        else:
            text = "\n".join([str(x) for x in lines if x is not None and str(x) != ""])

        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            clip_on=True,
        )

    ax.tick_params(axis="both", labelsize=12)

    def rectified(v):
        v = np.asarray(v, dtype=float)
        return np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)

    n = sig.size
    if n < 2:
        info_box("Signal too short")
        return

    tau = np.linspace(0.0, 1.0, n, endpoint=False)
    # =========================
    # RI
    # =========================
    if metric == "RI":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        ri = float(support["RI"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        info_box([f"RI = {ri:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "Delta_DTI":
        a = np.asarray(support.get("delta_dti_curve", []), dtype=float)
        delta_dti = float(support["Delta_DTI"])

        if a.size == 0:
            info_box("Missing Delta_DTI support")
            return
        x_lin = np.linspace(0, 1, n)
        ax.plot(x_lin, a, color="#EC5241")
        ax.fill_between(
            x_lin,
            0,
            a,
            where=np.isfinite(a),
            hatch="//",
            facecolor="none",
            edgecolor="#f9c2ca",
        )
        info_box([rf"$\Delta_{{DTI}} = {delta_dti:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$D(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif metric == "PI":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        vmean = float(support["vmean"])
        pi = float(support["PI"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        hline_label(vmean, "Vmean", va="bottom")
        info_box([f"PI = {pi:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "mu_t_over_T":
        mu_over_T = float(support["mu_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            mu_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$\mu_t/T = {mu_over_T:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "sigma_t_over_T":
        mu = float(support["mu_t_over_T"])
        sigma = float(support["sigma_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            mu, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1.5
        )

        left = max(0.0, mu - sigma)
        right = min(1.0, mu + sigma)
        mask = (tau >= left) & (tau <= right)
        ax.fill_between(tau, 0, sig, where=mask & np.isfinite(sig), color="#F2CCC7")

        vline_to_curve(
            mu - sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        vline_to_curve(
            mu + sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )

        info_box([rf"$\mu_t/T={mu:.3f}$", rf"$\sigma_t/T={sigma:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t50_over_T":
        t10 = float(support["t10_over_T"])
        t50 = float(support["t50_over_T"])
        t90 = float(support["t90_over_T"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq in [t10, t50, t90]:
            yq = _y_at(tq, tau, C)
            if np.isfinite(yq):
                ax.vlines(tq, 0.0, yq, linestyles="--", linewidth=1, color="#000000")
                ax.hlines(yq, 0.0, tq, linestyles="--", linewidth=1, color="#000000")

        info_box([f"t10/T = {t10:.3f}, t50/T = {t50:.3f}", f"t90/T = {t90:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$ ", fontsize=14, labelpad=12)

    elif metric == "R_VTI":
        ratio = float(support["R_VTI"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        ax.fill_between(
            tau[tau < 0.5],
            0,
            sig[tau < 0.5],
            where=np.isfinite(sig[tau < 0.5]),
            color="#f9c2ca",
        )
        ax.fill_between(
            tau[tau >= 0.5],
            0,
            sig[tau >= 0.5],
            where=np.isfinite(sig[tau >= 0.5]),
            color="#F2CCC7",
        )
        vline_to_curve(
            0.5, tau, sig, y0=0.0, color="#000000", linestyles="--", linewidth=1
        )

        d1 = float(np.nansum(sig[tau < 0.5]))
        d2 = float(np.nansum(sig[tau >= 0.5]))
        info_box([f"D1={d1:.3g}, D2={d2:.3g}", f"R_VTI={ratio:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "SF_VTI":
        sf = float(support["SF_VTI"])
        tau_k = 1.0 / 3.0

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        ax.fill_between(
            tau[tau < tau_k],
            0,
            sig[tau < tau_k],
            where=np.isfinite(sig[tau < tau_k]),
            color="#fbd3f2",
        )
        ax.fill_between(
            tau,
            0,
            sig,
            where=np.isfinite(sig),
            hatch="//",
            facecolor="none",
            edgecolor="#FB8F8F",
        )
        vline_to_curve(
            tau_k, tau, sig, y0=0.0, color="#000000", linestyles="--", linewidth=1
        )

        d1 = float(np.nansum(sig[tau < tau_k]))
        dtot = float(np.nansum(sig))
        info_box([f"D1={d1:.3g} , Dtot={dtot:.3g}", f"SF_VTI={sf:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_max_over_T":
        t_max_over_T = float(support["t_max_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_max_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{max}}/T = {t_max_over_T:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_min_over_T":
        t_min_over_T = float(support["t_min_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_min_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{min}}/T = {t_min_over_T:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "Delta_t_over_T":
        t_max_over_T = float(support["t_max_over_T"])
        t_min_over_T = float(support["t_min_over_T"])
        delta_t = float(support["Delta_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_max_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        vline_to_curve(
            t_min_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$\Delta t/T = {delta_t:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_up_over_T":
        t_up = float(support["t_up_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_up, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{up}}/T = {t_up:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_down_over_T":
        t_down = float(support["t_down_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_down, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{down}}/T = {t_down:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "S_decay":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        vmean = float(support["vmean"])
        t_max = float(support["t_max_over_T"])
        t_min = float(support["t_min_over_T"])
        s_decay = float(support["S_decay"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        a = (vmin - vmax) / ((t_min - t_max) + EPS)
        b = vmax - a * t_max
        x_line = np.linspace(0, 1, sig.size)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, color="black", linestyle="-")
        vline_to_curve(
            t_max, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        vline_to_curve(
            t_min, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        hline_label(vmean, "Vmean", va="bottom")
        info_box([rf"$S_{{decay}}= {s_decay:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_rise_normalized":
        s_rise = float(support["slope_rise_normalized"])
        idx = int(np.nanargmax(dvdt))

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{rise}}={s_rise:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_fall_normalized":
        s_fall = float(support["slope_fall_normalized"])
        idx = int(np.nanargmin(dvdt))

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{fall}}={s_fall:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "R_SD":
        ratio = float(support["R_SD"])
        vmax = float(support["vmax"])
        vend = float(support["vend"])
        i0 = int(support.get("late_window_start_idx", int(np.floor(0.75 * n))))
        i1 = int(support.get("late_window_end_idx", int(np.ceil(0.90 * n))))

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        ax.fill_between(
            tau[i0:i1], 0, sig[i0:i1], where=np.isfinite(sig[i0:i1]), color="#F2CCC7"
        )
        hline_label(vmax, "Vmax", va="bottom")
        ax.axhline(vend, linestyle="--", linewidth=1, color="black")
        ax.text(
            0,
            vend,
            rf" $\overline{{Vend}}={vend:.3g}$",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        info_box([rf"$R_{{SD}}={ratio:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "gamma_t":
        gamma_t = float(support["gamma_t"])
        mu = float(support["mu_t_over_T"])
        sigma = float(support["sigma_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        vline_to_curve(
            mu, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1.5
        )
        vline_to_curve(
            mu - sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        vline_to_curve(
            mu + sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        info_box([rf"$\gamma_t={gamma_t:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "mu_h":
        w_h = harmonic_weights
        mu_h = float(support["mu_h"])
        xh = np.arange(1, len(w_h) + 1)

        ax.bar(xh, w_h, width=0.8, color="#EC5241")
        ax.axvline(mu_h, linestyle="--", linewidth=1.2, color="black")
        info_box([rf"$\mu_h={mu_h:.3f}$", f"H={len(w_h)}"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$w_n\:(a.u.)$", fontsize=14, labelpad=12)

    elif metric == "sigma_h":
        w_h = harmonic_weights
        mu_h = float(support["mu_h"])
        sigma_h = float(support["sigma_h"])
        xh = np.arange(1, len(w_h) + 1)

        ax.bar(xh, w_h, width=0.8, color="#EC5241")
        ax.axvline(mu_h, linestyle="--", linewidth=1.2, color="black")
        ax.axvline(mu_h - sigma_h, linestyle=":", linewidth=1.0, color="black")
        ax.axvline(mu_h + sigma_h, linestyle=":", linewidth=1.0, color="black")
        info_box([rf"$\mu_h={mu_h:.3f}$", rf"$\sigma_h={sigma_h:.3f}$"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$w_n \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric in {"N_eff", "N_eff_over_T"}:
        m0 = float(support["m0"])
        p = sig / (m0 + EPS)
        n_eff_over_t = float(support["N_eff_over_T"])
        n_eff = n_eff_over_t

        ax.plot(tau, p, linewidth=3, color="#EC5241")
        ax.fill_between(tau, 0, p, where=np.isfinite(p), color="#F2CCC7")

        if metric == "N_eff":
            info_box([rf"$N_{{eff}} \approx {n_eff:.3f}$"])
        else:
            info_box([rf"$N_{{eff}}/T \approx {n_eff_over_t:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$p(\tau)\: (a.u.)$", fontsize=14, labelpad=10)

    elif metric == "delta_phi2":
        if len(harmonic_magnitudes) < 2 or len(harmonic_phases) < 2:
            info_box("Need at least 2 harmonics")
            return

        A1, A2 = float(harmonic_magnitudes[0]), float(harmonic_magnitudes[1])
        phi1, phi2 = float(harmonic_phases[0]), float(harmonic_phases[1])
        dphi2 = float(support["delta_phi2"])
        phi1_t = phi1 / (2 * np.pi)
        phi2_t = phi2 / (2 * np.pi)
        dphi2_t = dphi2 / (2 * np.pi)

        m = 500
        tau_dense = np.linspace(0.0, 1.0, m, endpoint=False)
        omega = 2.0 * np.pi
        h1 = A1 * np.cos(omega * tau_dense + phi1)
        h2 = A2 * np.cos(2.0 * omega * tau_dense + phi2)

        ax.plot(
            tau_dense,
            h1,
            linewidth=3,
            color="#EC5241",
            label=r"$A_1\cos(2\pi\tau+\phi_1)$",
        )
        ax.plot(
            tau_dense,
            h2,
            linewidth=3,
            color="#ECB341",
            label=r"$A_2\cos(4\pi\tau+\phi_2)$",
        )

        info_box(
            [
                f"φ1={phi1:.2f} rad = {phi1_t:.2f}",
                f"φ2={phi2:.2f} rad = {phi2_t:.2f}",
                f"Δφ2={dphi2:.2f} rad = {dphi2_t:.2f}",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel("Harmonic component (a.u.) ", fontsize=14, labelpad=12)
        ax.legend(
            loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=10
        )

    elif metric == "crest_factor":
        cf = float(support["crest_factor"])
        vmax = float(np.nanmax(vb))
        rms = float(np.sqrt(np.nanmean(vb**2)))
        vb_tau = np.linspace(0.0, 1.0, len(vb), endpoint=False)

        ax.plot(vb_tau, vb, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(rms, "RMS", va="top")
        info_box([f"H={len(harmonic_magnitudes)}", f"CF= {cf:.3f}"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric in {"Hspec", "spectral_entropy"}:
        p = harmonic_weights
        hn = len(p)
        ent = float(support["spectral_entropy"])
        xh = np.arange(1, hn + 1)

        ax.bar(xh, p, width=0.8, color="#EC5241")
        ymax = float(np.nanmax(p)) if np.any(np.isfinite(p)) else 1.0
        ax.set_ylim(0, ymax * 1.35)
        uniform = 1.0 / hn if hn > 0 else np.nan
        ax.axhline(uniform, linestyle="--", linewidth=1, color="#000000")

        if np.isfinite(uniform):
            ax.text(
                0.98,
                uniform,
                f" 1/H={uniform:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none"),
            )

        info_box([f"H={hn}", f"Hspec = {ent:.3f}"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$p_n \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "E_low_over_E_total":
        mags2 = np.r_[0.0, harmonic_magnitudes] ** 2
        e_low = float(np.nansum(mags2[1:H_LOW_MAX]))
        e_total = float(np.nansum(mags2))
        ratio = float(support["E_low_over_E_total"])
        xh = np.arange(0, len(mags2))

        ax.set_yscale("log")
        ax.bar(xh[1:H_LOW_MAX], mags2[1:H_LOW_MAX], color="#EC5241")
        ax.bar(xh[H_LOW_MAX:], mags2[H_LOW_MAX:], color="#cccccc")
        ax.axvline(H_LOW_MAX, linestyle="--", color="black")
        info_box(
            [
                f"E_low = {e_low:.3g}",
                f"E_total = {e_total:.3g}",
                rf"$E_{{low}}/E_{{total}} = {ratio:.3f}$",
            ]
        )
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$|V_n|^2 \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "E_high_over_E_total":
        mags2 = np.r_[0.0, harmonic_magnitudes] ** 2
        e_high = float(np.nansum(mags2[H_HIGH_MIN:H_HIGH_MAX]))
        e_total = float(np.nansum(mags2))
        ratio = float(support["E_high_over_E_total"])
        xh = np.arange(0, len(mags2))

        ax.set_yscale("log")
        ax.bar(xh[1:H_HIGH_MIN], mags2[1:H_HIGH_MIN], color="#cccccc")
        ax.bar(xh[H_HIGH_MIN:H_HIGH_MAX], mags2[H_HIGH_MIN:H_HIGH_MAX], color="#EC5241")
        ax.bar(xh[H_HIGH_MAX:], mags2[H_HIGH_MAX:], color="#cccccc")
        ax.axvline(H_HIGH_MIN, linestyle="--", color="black")
        info_box(
            [
                f"E_high = {e_high:.3g}",
                f"E_total = {e_total:.3g}",
                rf"$E_{{high}}/E_{{total}} = {ratio:.3f}$",
            ]
        )
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$|V_n|^2 \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "E_recon_H_MAX":
        e_recon = float(support["E_recon_H_MAX"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241", label="signal")
        ax.plot(
            np.linspace(0.0, 1.0, len(vb), endpoint=False),
            vb,
            linestyle="--",
            linewidth=2,
            color="black",
            label="reconstruction",
        )
        info_box(
            [rf"$E_{{recon,Hmax}}={e_recon:.3f}$", f"Hmax={len(harmonic_magnitudes)}"]
        )
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "Q_t_skew":
        t10 = float(support["t10_over_T"])
        t50 = float(support["t50_over_T"])
        t90 = float(support["t90_over_T"])
        q_t_skew = float(support["Q_t_skew"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq in [t10, t50, t90]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
            ax.hlines(yq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{t_{{skew}}}}={q_t_skew:.3f}$",
                f"t10={t10:.3f}, t50={t50:.3f}, t90={t90:.3f}",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_t_width":
        t25 = float(support["t25_over_T"])
        t75 = float(support["t75_over_T"])
        q_t_width = float(support["Q_t_width"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        y25 = _y_at(t25, tau, C)
        y75 = _y_at(t75, tau, C)
        ax.vlines(t25, 0, y25, linestyle="--", linewidth=1, color="black")
        ax.vlines(t75, 0, y75, linestyle="--", linewidth=1, color="black")
        ax.fill_between(tau, 0, C, where=(tau >= t25) & (tau <= t75), color="#F2CCC7")

        info_box(
            [rf"$Q_{{t_{{width}}}}={q_t_width:.3f}$", f"t25={t25:.3f}, t75={t75:.3f}"]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "R_Q_t":
        t10 = float(support["t10_over_T"])
        t25 = float(support["t25_over_T"])
        t50 = float(support["t50_over_T"])
        t75 = float(support["t75_over_T"])
        t90 = float(support["t90_over_T"])
        q_t_width = float(support["Q_t_width"])
        q_t_skew = float(support["Q_t_skew"])
        r_q_t = float(support["R_Q_t"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq in [t10, t25, t50, t75, t90]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
        ax.fill_between(tau, 0, C, where=(tau >= t25) & (tau <= t75), color="#F2CCC7")

        info_box(
            [
                rf"$Q_{{t_{{width}}}}={q_t_width:.3f}$",
                rf"$Q_{{t_{{skew}}}}={q_t_skew:.3f}$",
                rf"$R_{{Q_{{t}}}}={r_q_t:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_d_skew":
        d10 = float(support["d10"])
        d50 = float(support["d50"])
        d90 = float(support["d90"])
        q_d_skew = float(support["Q_d_skew"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq, dq in [(0.1, d10), (0.5, d50), (0.9, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{d_{{skew}}}}={q_d_skew:.3f}$",
                f"d10={d10:.3f}, d50={d50:.3f}, d90={d90:.3f}",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_d_width":
        d25 = float(support["d25"])
        d75 = float(support["d75"])
        q_d_width = float(support["Q_d_width"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        ax.vlines(0.25, 0, d25, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.75, 0, d75, linestyle="--", linewidth=1, color="black")
        ax.hlines(d25, 0, 0.25, linestyle="--", linewidth=1, color="black")
        ax.hlines(d75, 0, 0.75, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color="#F2CCC7")

        info_box(
            [rf"$Q_{{d_{{width}}}}={q_d_width:.3f}$", f"d25={d25:.3f}, d75={d75:.3f}"]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "R_Q_d":
        d10 = float(support["d10"])
        d25 = float(support["d25"])
        d50 = float(support["d50"])
        d75 = float(support["d75"])
        d90 = float(support["d90"])
        q_d_width = float(support["Q_d_width"])
        q_d_skew = float(support["Q_d_skew"])
        r_q_d = float(support["R_Q_d"])

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq, dq in [(0.10, d10), (0.25, d25), (0.50, d50), (0.75, d75), (0.90, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color="#F2CCC7")

        info_box(
            [
                rf"$Q_{{d_{{width}}}}={q_d_width:.3f}$",
                rf"$Q_{{d_{{skew}}}}={q_d_skew:.3f}$",
                rf"$R_{{Q_{{d}}}}={r_q_d:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "v_end_over_v_mean":
        vmean = float(support["vmean"])
        vend = float(support["vend"])
        ratio = float(support["v_end_over_v_mean"])
        i0 = int(support.get("late_window_start_idx", int(np.floor(0.75 * n))))
        i1 = int(support.get("late_window_end_idx", int(np.ceil(0.90 * n))))

        ax.plot(tau, sig, linewidth=3, color="#EC5241")
        ax.fill_between(
            tau[i0:i1], 0, sig[i0:i1], where=np.isfinite(sig[i0:i1]), color="#F2CCC7"
        )
        hline_label(vmean, "Vmean", va="bottom")
        ax.axhline(vend, linestyle="--", linewidth=1, color="black")
        ax.text(
            0,
            vend,
            rf" $\overline{{Vend}}={vend:.3g}$",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        info_box([rf"$R_{{EM}}={ratio:.3f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_slope":
        e_slope = float(support["E_slope"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241", label="signal")
        ax2 = ax.twinx()
        ax2.plot(
            tau,
            dvdt**2,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=r"$\dot v^2$",
        )
        ax2.set_ylabel(r"$\dot v^2$", fontsize=12)
        ax2.set_yticks([])
        info_box([rf"$E_{{slope}}={e_slope:.4f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_curv":
        e_curv =  float(support["E_curv"])

        ax.plot(tau, sig, linewidth=3, color="#EC5241", label="signal")
        ax2 = ax.twinx()
        ax2.plot(
            tau,
            d2vdt2**2,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=r"$\ddot v^2$",
        )
        ax2.set_yticks([])
        ax2.set_ylabel(r"$\ddot v^2$", fontsize=12)
        info_box([rf"$E_{{curv}}={e_curv:.4f}$"])
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    

    elif metric == "W50_over_T":
        vmax = float(support["vmax"])
        threshold = 0.5 * vmax
        W50_over_T = float(support["W50_over_T"])
        
        ax.plot(tau, sig, linewidth=3, color="#EC5241")

                
        hline_label(vmax, "Vmax", va="bottom")
        ax.axhline(threshold, linestyle="--", linewidth=1, color="black")
        ax.text(
            0,
            threshold,
            "",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        ax.fill_between(
            tau, threshold, sig, 
            where=(sig >= threshold),
            color="#F2CCC7", 
            label=r"Area $\geq 0.5 V_{max}$"
        )
        info_box([rf"$Vmax = {vmax:.2f}$",rf"$W50/T = {W50_over_T:.3f}$"])
        

        vline_to_curve(
            W50_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    
    elif metric == "W75_over_T":
        vmax = float(support["vmax"])
        threshold = 0.75 * vmax
        W75_over_T = float(support["W75_over_T"])
        
        ax.plot(tau, sig, linewidth=3, color="#EC5241")

                
        hline_label(vmax, "Vmax", va="bottom")
        ax.axhline(threshold, linestyle="--", linewidth=1, color="black")
        ax.text(
            0,
            threshold,
            "",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        ax.fill_between(
            tau, threshold, sig, 
            where=(sig >= threshold),
            color="#F2CCC7", 
            label=r"Area $\geq 075 V_{max}$"
        )
        info_box([rf"$Vmax = {vmax:.2f}$",rf"$W50/T = {W75_over_T:.3f}$"])
        

        vline_to_curve(
            W75_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "N_H_over_T":
        m0 = float(support["m0"])
        p = sig / (m0 + EPS)
        n_h_over_t = float(support["N_H_over_T"])
        

        ax.plot(tau, p, linewidth=3, color="#EC5241")
        ax.fill_between(tau, 0, p, where=np.isfinite(p), color="#F2CCC7")

        
        info_box([rf"$N_{{H}}/T \approx {n_h_over_t:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$p(\tau)\: (a.u.)$", fontsize=14, labelpad=10)

    else:
        info_box(f"No illustration for {metric}")


def export_selected_metric_pngs_bandlimited(
    all_results, zip_path, out_dir, dataset_path=SIGNAL_DATASET_PATH
):
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # index des chemins .h5 par groupe / fichier
        h5_index = build_h5_path_index_from_extracted_tree(tmpdir)
        group_delta_phi_stats = compute_group_delta_phi_stats(zip_path)
        for metric in sorted(SELECTED_METRICS_PNG):
            metric_key = METRIC_ALIASES.get(metric, metric)
            if metric_key not in all_results["bandlimited"]:
                continue

            df = pd.DataFrame(all_results["bandlimited"][metric_key]).copy()

            # ordre des groupes (control en dernier)
            groups = sorted(df["group"].dropna().unique().tolist())
            control_name = find_control_group_name(groups)
            if control_name in groups:
                groups = [g for g in groups if g != control_name] + [control_name]

            # positions x pour le scatter
            x_pos = {g: i for i, g in enumerate(groups)}

            # stats par groupe (sur les patients)
            grp = df.groupby("group")["mean"]
            grp_mean = grp.mean()
            grp_std = grp.std()

            rep_file = select_representative_file_per_group(df, value_col="mean")

            # ===== Layout figure (gauche scatter + droite 2x2) =====
            n_groups = len(groups)

            fig = plt.figure(figsize=(15, 6.2), dpi=200)

            outer = gridspec.GridSpec(
                1,
                2,
                width_ratios=[
                    0.7,
                    1.0,
                ],  # un peu moins de place au scatter => + place à droite
                wspace=0.15,  # rapproche fortement gauche/droite
            )

            # marges globales (ENLÈVE les bandes blanches inutiles)
            fig.subplots_adjust(left=0.04, right=0.995, bottom=0.08, top=0.86)

            ax_header = fig.add_axes([0.04, 0.88, 0.955, 0.11])  # x, y, w, h
            ax_header.axis("off")
            # formula = LATEX_FORMULAS.get(metric, "")

            # ===== Gauche: scatter =====
            ax_top = fig.add_subplot(outer[0, 0])

            if control_name in x_pos:
                cx = x_pos[control_name]
                ax_top.axvspan(cx - 0.5, cx + 0.5, color="#E0E0E0")

            rng = np.random.default_rng(0)
            shapes = ["D", "o", "s", "^"]
            for i, g in enumerate(groups):
                gdf = df[df["group"] == g]
                x = np.full(len(gdf), x_pos[g], dtype=float) + rng.normal(
                    0, 0.06, size=len(gdf)
                )

                ax_top.scatter(
                    x,
                    gdf["mean"].values,
                    color="black",
                    s=20,
                    edgecolors="none",
                )

                if g in grp_mean.index:
                    ax_top.errorbar(
                        [x_pos[g]],
                        [grp_mean.loc[g]],
                        color="black",
                        yerr=[grp_std.loc[g] if pd.notna(grp_std.loc[g]) else 0],
                        fmt=shapes[i],
                        capsize=5,
                        markersize=12,
                        linewidth=1.2,
                        markerfacecolor="none",
                        markeredgecolor="black",
                        markeredgewidth=3,
                    )

            ax_top.set_title(
                f"{LATEX_FORMULAS[metric]} (bandlimited waveform) ", fontsize=20, pad=20
            )
            ax_top.set_xticks([x_pos[g] for g in groups])
            ax_top.set_xticklabels(groups, rotation=0)
            ax_top.tick_params(axis="both", labelsize=16)
            ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
            ax_top.grid(True, axis="y")

            # ===== Droite: 2x2 illustrations =====
            right = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer[0, 1],
                hspace=0.5,  # <-- réduit l'écart vertical entre les 4
                wspace=0.28,  # <-- réduit l'écart horizontal entre les 2 colonnes d'illustrations
            )

            for i, g in enumerate(groups[:4]):  # on affiche 4 max
                r = i // 2
                c = i % 2
                ax = fig.add_subplot(right[r, c])

                chosen = rep_file.get(g, None)
                path = h5_index.get(g, {}).get(chosen, None) if chosen else None
                if metric == "phase_locking_residual":
                    plot_group_delta_phi_stats(ax, group_delta_phi_stats, g)
                    ax.set_title(f"{g}", fontsize=14)
                elif path and os.path.exists(path):
                    support = extract_graphics_support(path, mode="bandlimited")
                    support_beat = select_support_beat(support, 0)
                    plot_metric_illustration(ax, metric, support_beat, path)

                    ax.set_title(f" {g} ", fontsize=14)

                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(np.minimum(0, ymin), ymax * 1.4)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No representative file for {g}",
                        ha="center",
                        va="center",
                    )
                    ax.axis("off")

            # si <4 groupes, masque les cases vides

            for j in range(len(groups[:4]), 4):
                r = j // 2
                c = j % 2
                ax_empty = fig.add_subplot(right[r, c])
                ax_empty.axis("off")

            png_path = os.path.join(out_dir, f"{metric}_bandlimited.png")
            fig.savefig(png_path)
            plt.close(fig)


def replace_folder_in_zip(zip_path: str, folder_path: str, arc_folder: str):
    """
    Remplace complètement un dossier dans un zip.
    Supprime toute ancienne version de arc_folder/ puis ajoute folder_path.
    """
    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if not item.filename.startswith(arc_folder + "/"):
                    buffer = zin.read(item.filename)
                    zout.writestr(item, buffer)

            for root, _, files in os.walk(folder_path):
                for fn in files:
                    fullpath = os.path.join(root, fn)
                    rel = os.path.relpath(fullpath, folder_path)
                    arcname = os.path.join(arc_folder, rel).replace("\\", "/")
                    zout.write(fullpath, arcname)

    os.replace(temp_zip, zip_path)


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def replace_file_in_zip(zip_path, file_to_add):

    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w") as zout:
            for item in zin.infolist():
                if item.filename != os.path.basename(file_to_add):
                    buffer = zin.read(item.filename)
                    zout.writestr(item, buffer)

            zout.write(file_to_add)

    os.replace(temp_zip, zip_path)


def load_first_m0_image(zip_path):

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for f in sorted(files):
                if f.endswith(".h5"):
                    h5_path = os.path.join(root, f)

                    with h5py.File(h5_path, "r") as h5:
                        img = h5["/Maps/M0_ff_img/value"][()]

                    return img

    return None


def build_heatmap(img):

    # transpose
    img = img.T

    h, w = img.shape

    # centre
    cy, cx = h // 2, w // 2

    # rayon = moitié du carré
    r = min(cx, cy)

    # grille coordonnées
    Y, X = np.ogrid[:h, :w]

    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2

    # appliquer masque circulaire
    img_circle = np.full_like(img, np.nan, dtype=float)
    img_circle[mask] = img[mask]

    # heatmap
    fig = px.imshow(img_circle, color_continuous_scale="inferno", origin="lower")

    # cacher axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_layout(
        width=150,
        height=150,
        margin=dict(t=10, b=0, l=0, r=0),
        coloraxis_showscale=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def extract_sort_key(filename):

    name = os.path.basename(filename)

    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return (date, hd_index)


def extract_metrics(h5_path):

    results = {}

    with h5py.File(h5_path, "r") as f:
        metrics_root = f[METRIC_FOLDER]

        for mode in metrics_root.keys():
            if mode not in VALID_METRIC_FOLDERS:
                continue

            results[mode] = {}

            group = metrics_root[mode]

            for metric_name in group.keys():
                dataset = group[metric_name]

                data = np.array(dataset)

                latex_formula = dataset.attrs.get("latex_formula", "")
                results[mode][metric_name] = {
                    "mean": np.median(data),
                    "std": np.std(data),
                    "latex_formula": latex_formula,
                }

    return results


def select_representative_file_per_group(df_metric: pd.DataFrame, value_col="mean"):
    """
    Renvoie un dict: {group -> filename} du patient le plus proche de la médiane du groupe.
    df_metric doit contenir au moins: ["group", "file", value_col]
    """
    rep = {}
    for g, gdf in df_metric.groupby("group"):
        vals = gdf[value_col].astype(float).values
        if len(vals) == 0 or not np.any(np.isfinite(vals)):
            continue
        med = float(np.nanmedian(vals))
        # index du patient le plus proche de la médiane
        idx = int(np.nanargmin(np.abs(vals - med)))
        rep[g] = gdf.iloc[idx]["file"]
    return rep


def build_h5_path_index_from_extracted_tree(tmpdir: str):
    """
    Construit un index: {group_name -> {filename -> fullpath}}
    group_name = nom du dossier parent (comme dans analyze_zip)
    """
    index = defaultdict(dict)
    for root, _, files in os.walk(tmpdir):
        h5_files = [f for f in files if f.endswith(".h5")]
        if not h5_files:
            continue
        group_name = os.path.basename(root)
        if root == tmpdir:
            group_name = "all"
        for f in h5_files:
            index[group_name][f] = os.path.join(root, f)
    return index


def analyze_zip(zip_path):

    all_results = defaultdict(lambda: defaultdict(list))
    detected_groups = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = sorted(f for f in files if f.endswith(".h5"))
            if not h5_files:
                continue

            group_name = os.path.basename(root)

            if root == tmpdir:
                group_name = "all"

            detected_groups.add(group_name)

            for file in h5_files:
                filepath = os.path.join(root, file)

                metrics = extract_metrics(filepath)

                for mode, metric_dict in metrics.items():
                    for metric_name, values in metric_dict.items():
                        all_results[mode][metric_name].append(
                            {
                                "file": file,
                                "group": group_name,
                                "mean": values["mean"],
                                "std": values["std"],
                                "latex_formula": values.get("latex_formula", ""),
                            }
                        )

    single_group = len(detected_groups) < 1

    return dict(all_results), single_group


def build_metric_figure(df, metric, mode, ymin, ymax, single_group):

    groups = sorted(df["group"].unique())
    control_name = find_control_group_name(groups)
    if control_name in groups:
        groups = [g for g in groups if g != control_name] + [control_name]

    color_map = {
        g: c
        for g, c in zip(
            groups,
            ["royalblue", "firebrick", "seagreen", "orange", "purple"],
            strict=False,
        )
    }

    fig = go.Figure()

    fig.update_layout(autosize=True, height=400, margin=dict(t=10, b=10, l=10, r=10))

    xmin = df["index"].min()
    xmax = df["index"].max()

    current = xmin + 0.5
    toggle = True

    while current <= xmax:
        if toggle:
            fig.add_vrect(
                x0=current - 1,
                x1=current,
                fillcolor="lightblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        toggle = not toggle
        current += 1

    for g in groups:
        group_df = df[df["group"] == g]

        fig.add_trace(
            go.Scatter(
                x=group_df["index"],
                y=group_df["mean"],
                mode="markers",
                marker=dict(color="black", size=7, opacity=0.6),
                showlegend=False,
            )
        )

    for g in groups:
        group_df = df[df["group"] == g]

        fig.add_trace(
            go.Scatter(
                x=[group_df["index"].mean()],
                y=[group_df["mean"].mean()],
                mode="markers",
                marker=dict(
                    size=25,
                    color="white",  # intérieur creux
                    line=dict(
                        color="black",  # bordure noire
                        width=2,
                    ),
                ),
                error_y=dict(
                    type="data",
                    array=[group_df["mean"].std()],
                    visible=True,
                    thickness=3,
                    width=8,
                ),
                showlegend=False,
            )
        )

    if not single_group:
        tickvals = []
        ticktext = []

        for g in groups:
            group_indices = df[df["group"] == g]["index"]
            center = group_indices.mean()

            color = color_map[g]

            tickvals.append(center)
            ticktext.append(f'<span style="color:{color}; font-weight:bold">{g}</span>')

        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title="Patient Group",
        )
    else:
        fig.update_xaxes(showticklabels=False, title="")

    fig.update_yaxes(range=[ymin, ymax])

    fig.update_layout(yaxis_title=metric, yaxis_title_font=dict(size=15))

    return fig


def extract_mean_support_per_file(h5_path, mode="bandlimited"):
    support = extract_graphics_support(h5_path, mode)
    if not support:
        return None

    out = {}

    for k, v in support.items():
        arr = np.asarray(v)

        if arr.ndim == 2:
            if k in {
                "harmonic_magnitudes",
                "harmonic_weights",
                "harmonic_phases",
                "delta_phi_all",
            }:
                out[k] = np.nanmean(arr, axis=0)
            else:
                out[k] = np.nanmean(arr, axis=1)

        elif arr.ndim == 1:
            out[k] = np.nanmean(arr)

        else:
            out[k] = v

    return out


def extract_mean_signal_per_file(h5_path, dataset_path):
    with h5py.File(h5_path, "r") as f:
        signal = np.array(f[dataset_path])  # shape (time, beats)

    # moyenne sur les beats
    mean_signal = signal.mean(axis=1)

    return mean_signal


def compute_group_mean_signals(zip_path, dataset_path):

    group_signals = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = [f for f in files if f.endswith(".h5")]
            if not h5_files:
                continue

            group_name = os.path.basename(root)
            if root == tmpdir:
                group_name = "all"

            for file in h5_files:
                h5_path = os.path.join(root, file)
                signal = extract_mean_signal_per_file(h5_path, dataset_path)
                group_signals[group_name].append(signal)

    group_curves = {}

    for group, signals in group_signals.items():
        min_len = min(len(s) for s in signals)
        aligned = np.array([s[:min_len] for s in signals])

        group_mean = np.mean(aligned, axis=0)

        group_curves[group] = {
            "x": np.arange(min_len),
            "mean": group_mean,
        }

    return group_curves


def build_comparison_signal_figure(group_curves):

    fig = go.Figure()

    groups = sorted(group_curves.keys())
    max_len = max(len(group_curves[g]["x"]) for g in groups)

    x_common = np.arange(max_len)

    global_max = 0

    color_map = {
        g: c
        for g, c in zip(
            groups,
            ["royalblue", "firebrick", "seagreen", "orange", "purple"],
            strict=False,
        )
    }

    for group in groups:
        data = group_curves[group]

        y_old = data["mean"]

        y_interp = np.interp(x_common, np.linspace(0, max_len - 1, len(y_old)), y_old)

        global_max = max(global_max, np.max(y_interp))

        fig.add_trace(
            go.Scatter(
                x=x_common,
                y=y_interp,
                mode="lines",
                name=group,
                line=dict(color=color_map[group], width=3),
            )
        )

    fig.update_yaxes(range=[0, global_max * 1.05])

    fig.update_layout(
        height=550,
        xaxis_title="Time index",
        yaxis_title="Signal amplitude",
        template="simple_white",
        legend_title="Group",
    )

    return fig


def build_group_signal_figure(group_name, data):

    fig = go.Figure()

    x = data["x"]
    mean = data["mean"]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(width=3),
        )
    )
    y_max = np.max(mean)

    fig.update_yaxes(range=[0, y_max])

    fig.update_layout(
        height=450,
        xaxis_title="Time",
        yaxis_title="Velocity",
        template="simple_white",
    )

    return fig


def save_dashboard(all_results, original_zip, single_group):

    dashboard_file = "metric_dashboard.html"

    with open(dashboard_file, "w") as f:
        f.write("""
<html>
<head>
<title>Metrics Dashboard</title>
<script>
MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }
};
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>

body {
    margin: 20px;
    font-family: Arial, sans-serif;
}

/* ===== HEADER ===== */
.header {
    display: flex;
    align-items: center;
    gap: 40px;
    margin-bottom: 40px;
}

.header img {
    height: 180px;
    border-radius: 10px;
}

.header h1 {
    font-size: 25px;
    margin: 0;
}

/* ===== METRIC BLOCK ===== */
.metric-block {
    margin-top: 5px;
    padding-top: 5px;
    border-top: 3px solid #ddd;
}

/* ===== metric title ===== */
.metric-title {
    font-size: 15px;
    font-weight: bold;
    margin-bottom: 5px;
}

@media (max-width: 900px) {
    .row {
        flex-direction: column;
    }
}
/* ===== RAW/BANDLIMITED ROW ===== */
.row {
    display: flex;
    flex-direction: row;
    gap: 5px;
    width: 100%;
    align-items: flex-start eliminar;
}
.plotly-graph-div {
    width: 100% !important;
}

/* ===== each plot ===== */
.plot {
    flex: 1 1 50%;
    width: 100%;
}

/* ===== mode titles ===== */
.mode-title {
    font-size:10px;
    font-weight:bold;
    margin-bottom:5px;
    letter-spacing:1px;
}
.signal-grid {
    display: grid;
    grid-template-columns: 1fr 1fr; /* 2 colonnes */
    gap: 20px;
    width: 100%;
    margin-bottom: 40px;
}

.signal-plot {
    width: 100%;
}

</style>
</head>
<body>
""")

    all_metrics = set()
    for mode in all_results:
        all_metrics.update(all_results[mode].keys())
    dashboard_file = "metric_dashboard.html"
    img = load_first_m0_image(original_zip)
    if img is not None:
        heatmap_fig = build_heatmap(img)
        heatmap_html = heatmap_fig.to_html(full_html=False, include_plotlyjs="cdn")
        with open(dashboard_file, "a") as f:
            f.write(f"""
                    <div class = "header">
                    {heatmap_html}
                    <h1>Metrics Analysis</h1>
                    </div>""")

    dataset_path = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    dataset_path_bl = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"

    group_curves = compute_group_mean_signals(original_zip, dataset_path)
    group_curves_bl = compute_group_mean_signals(original_zip, dataset_path_bl)
    group_comparison_curves = build_comparison_signal_figure(group_curves)
    group_comparison_curves_bl = build_comparison_signal_figure(group_curves_bl)
    png_dir = os.path.join(os.path.dirname(dashboard_file), "export_png")
    export_selected_metric_pngs_bandlimited(
        all_results, original_zip, png_dir, dataset_path=dataset_path_bl
    )
    print("PNGs exportés dans :", png_dir)
    replace_folder_in_zip(original_zip, png_dir, arc_folder="export_png")
    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)
    with open(dashboard_file, "a") as f:
        f.write('<div class="signal-grid">')
    for group, data in group_curves.items():
        fig_signal = build_group_signal_figure(group, data)

        fig_html = fig_signal.to_html(
            full_html=False, include_plotlyjs=False, config={"responsive": True}
        )

        with open(dashboard_file, "a") as f:
            f.write(f"""
    <div class="signal-plot">
        <div class="metric-title">Signal raw {group}</div>
        {fig_html}
    </div>
    """)
    for group, data in group_curves_bl.items():
        fig_signal = build_group_signal_figure(group, data)

        fig_html = fig_signal.to_html(
            full_html=False, include_plotlyjs=False, config={"responsive": True}
        )

        with open(dashboard_file, "a") as f:
            f.write(f"""
    <div class="signal-plot">
        <div class="metric-title">Signal bandlimited {group}</div>
        {fig_html}
    </div>
    """)
    fig_html = group_comparison_curves.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    with open(dashboard_file, "a") as f:
        f.write(f"""<div class="signal-plot">
        <div class="metric-title">Signal comparison</div>
        {fig_html}
    </div>
    """)
    fig_html_bl = group_comparison_curves_bl.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    with open(dashboard_file, "a") as f:
        f.write(f"""<div class="signal-plot">
        <div class="metric-title">Signal comparison bandlimited</div>
        {fig_html_bl}
    </div>""")

    with open(dashboard_file, "a") as f:
        f.write("</div>")
    for metric in sorted(all_metrics):
        definition = all_results["raw"][metric][0].get("latex_formula", "")
        for mode in ["raw", "bandlimited"]:
            if mode in all_results and metric in all_results[mode]:
                definition = all_results[mode][metric][0].get("latex_formula", "")
                break
        y_values = []

        for mode in ["raw", "bandlimited"]:
            if mode in all_results and metric in all_results[mode]:
                df_tmp = pd.DataFrame(all_results[mode][metric])
                y_values.extend(df_tmp["mean"].values)

        ymin = min(y_values)
        ymax = max(y_values)

        margin = 0.05 * (ymax - ymin if ymax != ymin else 1)
        ymax += margin

        # ----- HTML metric header -----
        with open(dashboard_file, "a") as f:
            f.write('<div class="metric-block">')
            f.write(f'<div class="metric-title">{metric + " = " + definition[0]}</div>')
            f.write('<div class="row">')

        # ======================
        # LOOP MODES
        # ======================
        for mode in ["raw", "bandlimited"]:
            if mode not in all_results:
                continue
            if metric not in all_results[mode]:
                continue

            data = all_results[mode][metric]

            df = pd.DataFrame(data)

            df["group_order"] = df["group"].astype("category").cat.codes
            df = df.sort_values(["group_order", "file"])
            df["index"] = range(len(df))

            fig = build_metric_figure(
                df,
                metric,
                mode,
                ymin,
                ymax,
                single_group,
            )

            fig_html = fig.to_html(
                full_html=False, include_plotlyjs="cdn", config={"responsive": True}
            )

            with open(dashboard_file, "a") as f:
                f.write(f"""
                <div class="plot">
                    <div class="mode-title">{mode.upper()}</div>
                    {fig_html}
                </div>
                """)

        with open(dashboard_file, "a") as f:
            f.write("</div></div>")
    with open(dashboard_file, "a") as f:
        f.write("""
    <script>
    window.addEventListener("load", function() {
        setTimeout(function() {
            document.querySelectorAll('.plotly-graph-div')
            .forEach(function(el) {
                Plotly.Plots.resize(el);
            });
        }, 300);
    });
    </script>
    """)
    with open(dashboard_file, "a") as f:
        f.write("</body></html>")

    replace_file_in_zip(
        original_zip,
        dashboard_file,
    )

    print("Dashboard ajouté à:", original_zip)


if __name__ == "__main__":
    zip_path = choose_zip()
    dataset_path_bl = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"

    results, single_group = analyze_zip(zip_path)
    dashboard_file = "metric_dashboard.html"
    png_dir = os.path.join(os.path.dirname(dashboard_file), "export_png")
    export_selected_metric_pngs_bandlimited(
        results, zip_path, png_dir, dataset_path=dataset_path_bl
    )
    replace_folder_in_zip(zip_path, png_dir, arc_folder="export_png")
    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)
    # save_dashboard(results, zip_path, single_group)
