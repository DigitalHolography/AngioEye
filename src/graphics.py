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
    "G_t",
    "R_pre_post",
    "R_slope",
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
    "G_t": r"$G_t$",
    "R_pre_post": r"$R_{{pre/post}}$",
    "R_slope": r"$R_{\mathrm{slope}}$",
    "E_recon_H_MAX": r"$E_{\mathrm{recon},H_{\max}}$",
    "Q_t_skew": r"$Q_{\mathrm{t_{{skew}}}}$",
    "Q_t_width": r"$Q_{\mathrm{t_{{width}}}}$",
    "Q_d_skew": r"$Q_{\mathrm{d_{{skew}}}}$",
    "Q_d_width": r"$Q_{\mathrm{d_{{width}}}}$",
    "R_Q_t": r"$R_{\mathrm{Q_{{t}}}}$",
    "R_Q_d": r"$R_{\mathrm{Q_{{d}}}}$",
    "v_end_over_v_mean": r"$R_{EM}$",
    "E_slope": r"$E_{\mathrm{slope}}$",
    "E_curv": r"$E_{\mathrm{curv}}$",
}


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


def plot_metric_illustration(ax, metric, sig_control, path):
    sig = np.asarray(sig_control, dtype=float)
    sig = np.where(np.isfinite(sig), sig, np.nan)

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
        w = rectified(sig)
        vmax = float(np.nanmax(w))
        vmin = float(np.nanmin(w))
        RI = 1.0 - (vmin / vmax) if vmax > 0 else np.nan

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")

        info_box([f"RI  = {RI:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # mu_t_over_T
    # =========================
    elif metric == "mu_h":
        V, vb, H, w_h = harmonic_weights_from_signal(sig)
        if w_h is None:
            info_box("Invalid harmonics")
            return

        xh = np.arange(1, len(w_h) + 1)
        mu_h = float(np.nansum(xh * w_h))

        ax.bar(xh, w_h, width=0.8, color="#EC5241")
        ax.axvline(mu_h, linestyle="--", linewidth=1.2, color="black")
        info_box([rf"$\mu_h={mu_h:.3f}$", f"H={len(w_h)}"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$w_n\:(a.u.)$", fontsize=14, labelpad=12)

    elif metric == "sigma_h":
        V, vb, H, w_h = harmonic_weights_from_signal(sig)
        if w_h is None:
            info_box("Invalid harmonics")
            return

        xh = np.arange(1, len(w_h) + 1)
        mu_h = float(np.nansum(xh * w_h))
        sigma_h = float(np.sqrt(np.nansum(((xh - mu_h) ** 2) * w_h)))

        ax.bar(xh, w_h, width=0.8, color="#EC5241")
        ax.axvline(mu_h, linestyle="--", linewidth=1.2, color="black")
        ax.axvline(mu_h - sigma_h, linestyle=":", linewidth=1.0, color="black")
        ax.axvline(mu_h + sigma_h, linestyle=":", linewidth=1.0, color="black")
        info_box([rf"$\mu_h={mu_h:.3f}$", rf"$\sigma_h={sigma_h:.3f}$"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$w_n \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric in {"N_eff", "N_eff_over_T"}:
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        p = w / (m0 + EPS)
        dtau = 1.0 / n
        integral_p2 = float(np.nansum(p**2) * dtau)

        if integral_p2 <= 0:
            info_box("Invalid density")
            return

        n_eff_over_t = float(1.0 / (integral_p2 + EPS))
        n_eff = n_eff_over_t  # ici T normalisé à 1 dans l’illustration

        ax.plot(tau, p, linewidth=3, color="#EC5241")
        ax.fill_between(tau, 0, p, where=np.isfinite(p), color="#F2CCC7")

        if metric == "N_eff":
            info_box([rf"$N_{{eff}} \approx {n_eff:.3f}$"])
        else:
            info_box([rf"$N_{{eff}}/T \approx {n_eff_over_t:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$p(\tau)\: (a.u.)$", fontsize=14, labelpad=10)

    elif metric == "G_t":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        gt = float(1.0 - 2.0 * np.nansum(C) / n)

        ax.plot(tau, C, linewidth=3, color="#EC5241", label=r"$C(\tau)$")
        ax.plot(tau, tau, linestyle="--", linewidth=1.2, color="black", label=r"$\tau$")
        ax.fill_between(tau, C, tau, where=np.isfinite(C), color="#F2CCC7", alpha=0.8)
        info_box([rf"$G_t={gt:.3f}$"])
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "R_pre_post":
        w = rectified(sig)
        idx_peak = int(np.nanargmax(w))
        t_up = float(idx_peak / n)
        t_down = float(1.0 - t_up)
        r_ud = float(t_up / (t_down + EPS))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        vline_to_curve(
            t_up, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1.2
        )
        info_box(
            [
                rf"$t_\uparrow/T={t_up:.3f}$",
                rf"$t_\downarrow/T={t_down:.3f}$",
                rf"$R_{{pre/post}}={r_ud:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "R_slope":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        dt = 1.0 / n
        dvdt = np.gradient(w, dt)
        s_rise = float(np.nanmax(dvdt))
        s_fall = float(np.abs(np.nanmin(dvdt)))
        r_slope = float(s_rise / (s_fall + EPS))

        i_rise = int(np.nanargmax(dvdt))
        i_fall = int(np.nanargmin(dvdt))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        vline_to_curve(
            tau[i_rise], tau, w, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        vline_to_curve(
            tau[i_fall], tau, w, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box(
            [
                rf"$S_{{rise}}={s_rise:.3f}$",
                rf"$S_{{fall}}={s_fall:.3f}$",
                rf"$R_{{slope}}={r_slope:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_recon_H_MAX":
        w = rectified(sig)
        V, vb, H, w_h = harmonic_pack(w)
        if vb is None:
            info_box("Invalid reconstruction")
            return

        num = float(np.nansum((w - vb) ** 2))
        den = float(np.nansum(w**2))
        e_recon = float(num / (den + EPS))

        ax.plot(tau, w, linewidth=3, color="#EC5241", label="signal")
        ax.plot(
            np.linspace(0.0, 1.0, len(vb), endpoint=False),
            vb,
            linestyle="--",
            linewidth=2,
            color="black",
            label="reconstruction",
        )
        info_box([rf"$E_{{recon,Hmax}}={e_recon:.3f}$", f"Hmax={H}"])
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "Q_t_skew":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        i10 = quantile_idx_from_cumsum(C, 0.10)
        i50 = quantile_idx_from_cumsum(C, 0.50)
        i90 = quantile_idx_from_cumsum(C, 0.90)

        t10 = float(i10 / n)
        t50 = float(i50 / n)
        t90 = float(i90 / n)

        Q_t_skew = float(((t90 - t50) - (t50 - t10)) / (t90 - t10 + EPS))

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq, lab in [(t10, "t10"), (t50, "t50"), (t90, "t90")]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
            ax.hlines(yq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{t_{{skew}}}}={Q_t_skew:.3f}$",
                f"t10={t10:.3f}, t50={t50:.3f}, t90={t90:.3f}",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "Q_d_skew":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        t10 = int(w.size / 10)
        t50 = int(w.size / 2)
        t90 = int(9 * w.size / 10)
        d10 = C[t10]
        d50 = C[t50]
        d90 = C[t90]

        Q_d_skew = float(((d90 - d50) - (d50 - d10)) / (d90 - d10 + EPS))

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        for tq, dq in [(0.1, d10), (0.5, d50), (0.9, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{d_{{skew}}}}={Q_d_skew:.3f}$",
                f"d10={d10:.3f}, d50={d50:.3f}, d90={d90:.3f}",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_t_width":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        i25 = quantile_idx_from_cumsum(C, 0.25)
        i75 = quantile_idx_from_cumsum(C, 0.75)

        t25 = float(i25 / n)
        t75 = float(i75 / n)
        Q_t_width = float(t75 - t25)

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        y25 = _y_at(t25, tau, C)
        y75 = _y_at(t75, tau, C)
        ax.vlines(t25, 0, y25, linestyle="--", linewidth=1, color="black")
        ax.vlines(t75, 0, y75, linestyle="--", linewidth=1, color="black")
        ax.fill_between(
            tau, 0, C, where=(tau >= t25) & (tau <= t75), color="#F2CCC7", alpha=0.7
        )

        info_box(
            [rf"$Q_{{t_{{width}}}}={Q_t_width:.3f}$", f"t25={t25:.3f}, t75={t75:.3f}"]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "Q_d_width":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        t25 = int(w.size / 4)
        t75 = int(3 * w.size / 4)
        d25 = C[t25]
        d75 = C[t75]
        Q_d_width = float(d75 - d25)

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        ax.vlines(0.25, 0, d25, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.75, 0, d75, linestyle="--", linewidth=1, color="black")
        ax.hlines(d25, 0, 0.25, linestyle="--", linewidth=1, color="black")
        ax.hlines(d75, 0, 0.75, linestyle="--", linewidth=1, color="black")
        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)

        ax.fill_betweenx(y_fill, 0, x_curve, color="#F2CCC7", alpha=0.7)
        info_box(
            [rf"$Q_{{d_{{width}}}}={Q_d_width:.3f}$", f"d25={d25:.3f}, d75={d75:.3f}"]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "R_Q_d":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        t25 = int(w.size / 4)
        t75 = int(3 * w.size / 4)
        t10 = int(w.size / 10)
        t50 = int(w.size / 2)
        t90 = int(9 * w.size / 10)
        d10 = C[t10]
        d50 = C[t50]
        d90 = C[t90]
        d25 = C[t25]
        d75 = C[t75]
        Q_d_width = float(d75 - d25)
        Q_d_skew = float(((d90 - d50) - (d50 - d10)) / (d90 - d10 + EPS))
        R_Q_d = Q_d_skew / Q_d_width
        ax.plot(tau, C, linewidth=3, color="#EC5241")
        ax.vlines(0.10, 0, d10, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.25, 0, d25, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.50, 0, d50, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.75, 0, d75, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.90, 0, d90, linestyle="--", linewidth=1, color="black")
        ax.hlines(d10, 0, 0.10, linestyle="--", linewidth=1, color="black")
        ax.hlines(d25, 0, 0.25, linestyle="--", linewidth=1, color="black")
        ax.hlines(d50, 0, 0.50, linestyle="--", linewidth=1, color="black")
        ax.hlines(d75, 0, 0.75, linestyle="--", linewidth=1, color="black")
        ax.hlines(d90, 0, 0.90, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color="#F2CCC7", alpha=0.7)

        info_box(
            [
                rf"$Q_{{d_{{width}}}}={Q_d_width:.3f}$",
                rf"$Q_{{d_{{skew}}}}={Q_d_skew:.3f}$",
                rf"$R_{{Q_{{d}}}}={R_Q_d:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "R_Q_t":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / (m0 + EPS)
        i25 = quantile_idx_from_cumsum(C, 0.25)
        i75 = quantile_idx_from_cumsum(C, 0.75)
        i10 = quantile_idx_from_cumsum(C, 0.10)
        i50 = quantile_idx_from_cumsum(C, 0.50)
        i90 = quantile_idx_from_cumsum(C, 0.90)

        t10 = float(i10 / n)
        t50 = float(i50 / n)
        t90 = float(i90 / n)
        t25 = float(i25 / n)
        t75 = float(i75 / n)

        Q_t_width = float(t75 - t25)
        Q_t_skew = float(((t90 - t50) - (t50 - t10)) / (t90 - t10 + EPS))
        R_Q_t = Q_t_skew / Q_t_width

        ax.plot(tau, C, linewidth=3, color="#EC5241")
        ax.vlines(t10, 0, C[i10], linestyle="--", linewidth=1, color="black")
        ax.vlines(t25, 0, C[i25], linestyle="--", linewidth=1, color="black")
        ax.vlines(t50, 0, C[i50], linestyle="--", linewidth=1, color="black")
        ax.vlines(t75, 0, C[i75], linestyle="--", linewidth=1, color="black")
        ax.vlines(t90, 0, C[i90], linestyle="--", linewidth=1, color="black")
        ax.fill_between(
            tau, 0, C, where=(tau >= t25) & (tau <= t75), color="#F2CCC7", alpha=0.7
        )

        info_box(
            [
                rf"$Q_{{t_{{width}}}}={Q_t_width:.3f}$",
                rf"$Q_{{t_{{skew}}}}={Q_t_skew:.3f}$",
                rf"$R_{{Q_{{t}}}}={R_Q_t:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "v_end_over_v_mean":
        w = rectified(sig)
        vmean = float(np.nanmean(w))
        i0 = int(np.floor(0.75 * n))
        i1 = int(np.ceil(0.90 * n))
        i1 = max(i1, i0 + 1)

        vend = float(np.nanmean(w[i0:i1]))
        ratio = float(vend / (vmean + EPS))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        ax.fill_between(
            tau[i0:i1], 0, w[i0:i1], where=np.isfinite(w[i0:i1]), color="#F2CCC7"
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
        ax.vlines(
            tau[i0],
            0,
            _y_at(tau[i0], tau, w),
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax.vlines(
            tau[min(i1 - 1, n - 1)],
            0,
            _y_at(tau[min(i1 - 1, n - 1)], tau, w),
            linestyle="--",
            linewidth=1,
            color="black",
        )
        info_box(
            [
                rf"$R_{{EM}}={ratio:.3f}$",
            ]
        )
        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_slope":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        dt = 1.0 / n
        dvdt = np.gradient(w, dt)
        e_slope = float(np.nansum(dvdt**2) * dt / ((m0 + EPS) ** 2))

        ax.plot(tau, w, linewidth=3, color="#EC5241", label="signal")
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
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        dt = 1.0 / n
        dvdt = np.gradient(w, dt)
        d2vdt2 = np.gradient(dvdt, dt)
        e_curv = float(np.nansum(d2vdt2**2) * dt / ((m0 + EPS) ** 2))

        ax.plot(tau, w, linewidth=3, color="#EC5241", label="signal")
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
    elif metric == "mu_t_over_T":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if not np.isfinite(m0) or m0 <= 0:
            info_box("Invalid signal for mu/T")
            return

        mu_over_T = float(np.nansum(w * tau) / m0)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            mu_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )

        info_box([rf"$\mu_t/T = {mu_over_T:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # PI
    # =========================
    elif metric == "PI":
        w = rectified(sig)
        vmax = float(np.nanmax(w))
        vmin = float(np.nanmin(w))
        vmean = float(np.nanmean(w))

        PI = (
            ((vmax - vmin) / (vmean + EPS))
            if (np.isfinite(vmean) and vmean > 0)
            else np.nan
        )

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        hline_label(vmean, "Vmean", va="bottom")

        info_box([f"PI = {PI:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # SF_VTI
    # =========================
    elif metric == "SF_VTI":
        w = rectified(sig)

        k = int(np.ceil(n / 3.0))
        k = max(0, min(n, k))
        tau_k = float(k / n)

        D1 = float(np.nansum(w[:k])) if k > 0 else np.nan
        Dtot = float(np.nansum(w))
        sf = D1 / (Dtot + EPS)

        ax.plot(tau, w, linewidth=3, color="#EC5241")

        ax.fill_between(tau[:k], 0, w[:k], where=np.isfinite(w[:k]), color="#fbd3f2")
        ax.fill_between(
            tau,
            0,
            w,
            where=np.isfinite(w),
            hatch="//",
            facecolor="none",
            edgecolor="#FB8F8F",
        )

        vline_to_curve(
            tau_k, tau, w, y0=0.0, color="#000000", linestyles="--", linewidth=1
        )

        info_box([f"D1={D1:.3g} , Dtot={Dtot:.3g}", f"SF_VTI={sf:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # sigma_t_over_T
    # =========================
    elif metric == "sigma_t_over_T":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if not np.isfinite(m0) or m0 <= 0:
            info_box("Invalid signal")
            return

        mu = float(np.nansum(w * tau) / m0)
        var = float(np.nansum(w * (tau - mu) ** 2) / (m0 + EPS))
        sigma = float(np.sqrt(max(var, 0.0)))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        vline_to_curve(
            mu, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1.5
        )

        left = max(0.0, mu - sigma)
        right = min(1.0, mu + sigma)
        idx_left = int(np.floor(left * n))
        idx_right = int(np.ceil(right * n))
        ax.fill_between(
            tau[idx_left:idx_right],
            0,
            w[idx_left:idx_right],
            where=np.isfinite(w[idx_left:idx_right]),
            color="#F2CCC7",
        )
        vline_to_curve(
            mu - sigma, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        vline_to_curve(
            mu + sigma, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1
        )

        info_box([rf"$\mu_t/T={mu:.3f}$", rf"$\sigma_t/T={sigma:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # delta_phi2
    # =========================
    elif metric == "delta_phi2":
        w = rectify_keep_nan(sig)
        w = np.where(np.isfinite(w), w, np.nan)

        Vfull = np.fft.rfft(w) / float(n)
        if Vfull.size < 3:
            info_box("Need at least 2 harmonics")
            return

        V1, V2 = Vfull[1], Vfull[2]
        A1, A2 = float(np.abs(V1)), float(np.abs(V2))
        phi1, phi2 = float(np.angle(V1)), float(np.angle(V2))
        phi1_t = phi1 / (2 * np.pi)
        phi2_t = phi2 / (2 * np.pi)

        dphi2 = float((phi2 - 2.0 * phi1 + np.pi) % (2.0 * np.pi) - np.pi)
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
                f"φ1={phi1:.2f} rad = {phi1_t:.2f} ",
                f"φ2={phi2:.2f} rad= {phi2_t:.2f}",
                f"Δφ2={dphi2:.2f} rad= {dphi2_t:.2f}",
            ]
        )

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel("Harmonic component (a.u.) ", fontsize=14, labelpad=12)
        ax.legend(
            loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=10
        )

    # =========================
    # crest_factor
    # =========================
    elif metric == "crest_factor":
        V, vb, H, __ = harmonic_pack(sig)
        if vb is None or vb.size < 2:
            info_box("Invalid vb")
            return

        vb = np.asarray(vb, dtype=float)
        vb_tau = np.linspace(0.0, 1.0, vb.size, endpoint=False)

        rms = float(np.sqrt(np.nanmean(vb**2)))
        vmax = float(np.nanmax(vb))
        cf = crest_factor_from_vb(vb)

        ax.plot(vb_tau, vb, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(rms, "RMS", va="top")

        info_box([f"H={H}", f"CF= {cf:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # t50_over_T
    # =========================
    elif metric == "t50_over_T":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        C = np.nancumsum(w) / m0
        x_norm = tau
        d = C - x_norm

        idx_50 = int(np.searchsorted(C, 0.5, side="left"))
        idx_50 = max(0, min(n - 1, idx_50))
        t50 = float(x_norm[idx_50])

        idx_10 = int(np.searchsorted(C, 0.1, side="left"))
        idx_10 = max(0, min(n - 1, idx_10))
        t10 = float(x_norm[idx_10])

        idx_90 = int(np.searchsorted(C, 0.9, side="left"))
        idx_90 = max(0, min(n - 1, idx_90))
        t90 = float(x_norm[idx_90])

        ax.plot(x_norm, C, linewidth=3, color="#EC5241")
        y_t50 = _y_at(t50, x_norm, C)
        y_t10 = _y_at(t10, x_norm, C)
        y_t90 = _y_at(t90, x_norm, C)
        if np.isfinite(y_t50):
            ax.vlines(t50, 0.0, y_t50, linestyles="--", linewidth=1, color="#000000")
            ax.hlines(y_t50, 0.0, t50, linestyles="--", linewidth=1, color="#000000")

        if np.isfinite(y_t10):
            ax.vlines(t10, 0.0, y_t10, linestyles="--", linewidth=1, color="#000000")
            ax.hlines(y_t10, 0.0, t10, linestyles="--", linewidth=1, color="#000000")

        if np.isfinite(y_t90):
            ax.vlines(t90, 0.0, y_t90, linestyles="--", linewidth=1, color="#000000")
            ax.hlines(y_t90, 0.0, t90, linestyles="--", linewidth=1, color="#000000")

        info_box([f"t10/T = {t10:.3f}, t50/T = {t50:.3f}", f"t90/T = {t90:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$C(\tau) \: (a.u.)$ ", fontsize=14, labelpad=12)
    # =========================
    # R_VTI
    # =========================
    elif metric == "R_VTI":
        w = rectified(sig)

        k = int(np.ceil(n * 0.5))
        k = max(0, min(n, k))

        D1 = float(np.nansum(w[:k])) if k > 0 else np.nan
        D2 = float(np.nansum(w[k:])) if k < n else np.nan
        R = D1 / (D2 + EPS)

        ax.plot(tau, w, linewidth=3, color="#EC5241")

        ax.fill_between(tau[:k], 0, w[:k], where=np.isfinite(w[:k]), color="#f9c2ca")
        ax.fill_between(tau[k:], 0, w[k:], where=np.isfinite(w[k:]), color="#F2CCC7")

        vline_to_curve(
            0.5,
            tau,
            w,
            y0=0.0,
            colors="k",
            linestyles="--",
            linewidth=1,
            color="#000000",
        )

        info_box([f"D1={D1:.3g}, D2={D2:.3g}", f"R_VTI={R:.3f}"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    # =========================
    # Hspec / spectral_entropy
    # =========================
    elif metric in {"Hspec", "spectral_entropy"}:
        V, vb, H, __ = harmonic_pack(sig)
        if V is None or V.size < 2:
            info_box("Invalid harmonics")
            return

        mags = np.abs(V[1:])
        mags = np.where(np.isfinite(mags), mags, np.nan)
        s = float(np.nansum(mags))
        if s <= 0:
            info_box("Invalid harmonics")
            return

        p = mags / s
        Hn = len(p)
        ent = spectral_entropy_from_harmonics(V)

        xh = np.arange(1, Hn + 1)
        ax.bar(xh, p, width=0.8, color="#EC5241")

        ymax = float(np.nanmax(p)) if np.any(np.isfinite(p)) else 1.0
        ax.set_ylim(0, ymax * 1.35)

        uniform = 1.0 / Hn
        ax.axhline(uniform, linestyle="--", linewidth=1, color="#000000")
        ax.text(
            0.98,
            uniform,
            f" 1/H={uniform:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none"),
        )

        info_box([f"H={Hn}", f"Hspec = {ent:.3f}"])

        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$p_n \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "E_high_over_E_total":
        V, vb, H, __ = harmonic_pack(sig)

        if V is None:
            info_box("Invalid harmonics")
            return

        mags = np.abs(V[0:]) ** 2
        mags = np.where(np.isfinite(mags), mags, np.nan)

        Hn = len(mags)

        E_high = np.nansum(mags[H_HIGH_MIN:H_HIGH_MAX])
        E_total = np.nansum(mags)

        ratio = E_high / (E_total + EPS)

        xh = np.arange(0, Hn)
        ax.set_yscale("log")
        ax.bar(xh[1:H_HIGH_MIN], mags[1:H_HIGH_MIN], color="#cccccc")
        ax.bar(xh[H_HIGH_MIN:H_HIGH_MAX], mags[H_HIGH_MIN:H_HIGH_MAX], color="#EC5241")
        ax.bar(xh[H_HIGH_MAX:], mags[H_HIGH_MAX:], color="#cccccc")

        ax.axvline(H_HIGH_MIN, linestyle="--", color="black")
        lines = [
            f"E_high = {E_high:.3g}",
            f"E_total = {E_total:.3g}",
            rf"$E_{{high}}/E_{{total}} = {ratio:.3f}$",
        ]
        text = "\n".join([str(x) for x in lines if x is not None and str(x) != ""])
        ax.text(
            0.50,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            clip_on=True,
        )

        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$|V_n|^2 \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "E_low_over_E_total":
        V, vb, H, __ = harmonic_pack(sig)

        if V is None:
            info_box("Invalid harmonics")
            return

        mags = np.abs(V[0:]) ** 2
        mags = np.where(np.isfinite(mags), mags, np.nan)

        Hn = len(mags)

        E_low = np.nansum(mags[1:H_LOW_MAX])
        E_total = np.nansum(mags)

        ratio = E_low / (E_total + EPS)

        xh = np.arange(0, Hn)
        ax.set_yscale("log")
        ax.bar(xh[1:H_LOW_MAX], mags[1:H_LOW_MAX], color="#EC5241")
        ax.bar(xh[H_LOW_MAX:], mags[H_LOW_MAX:], color="#cccccc")

        ax.axvline(H_LOW_MAX, linestyle="--", color="black")
        lines = [
            f"E_low = {E_low:.3g}",
            f"E_total = {E_total:.3g}",
            rf"$E_{{low}}/E_{{total}} = {ratio:.3f}$",
        ]
        text = "\n".join([str(x) for x in lines if x is not None and str(x) != ""])
        ax.text(
            0.50,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            clip_on=True,
        )

        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$|V_n|^2 \: (a.u.)$", fontsize=14, labelpad=12)
    elif metric == "t_max_over_T":
        w = rectified(sig)
        idx = int(np.nanargmax(w))

        t_max_over_T = float(idx / w.size)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            t_max_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )

        info_box([rf"$t_{{max}}/T = {t_max_over_T:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "t_min_over_T":
        w = rectified(sig)
        idx = int(np.nanargmin(w))

        t_min_over_T = float(idx / w.size)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            t_min_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )

        info_box([rf"$t_{{min}}/T = {t_min_over_T:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_down_over_T":
        w = rectified(sig)

        dt = 1 / w.size
        dvdt = np.gradient(w, dt)

        idx_min = np.argmin(dvdt)
        t_down_over_T = float(idx_min / w.size)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            t_down_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )

        info_box([rf"$t_{{down}}/T = {t_down_over_T:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_up_over_T":
        w = rectified(sig)

        dt = 1 / w.size
        dvdt = np.gradient(w, dt)

        idx_max = np.argmax(dvdt)
        t_up_over_T = float(idx_max / w.size)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            t_up_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )

        info_box([rf"$t_{{up}}/T = {t_up_over_T:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "S_decay":
        w = rectified(sig)
        vmax = float(np.nanmax(w))
        vmin = float(np.nanmin(w))
        vmean = float(np.nanmean(w))
        norm_w = w / vmean

        idx_min = int(np.nanargmin(w))
        idx_max = int(np.nanargmax(w))
        t_min_over_T = float(idx_min / w.size)
        t_max_over_T = float(idx_max / w.size)
        delta_t = t_min_over_T - t_max_over_T
        a, b = np.polyfit([t_max_over_T, t_min_over_T], [w[idx_max], w[idx_min]], 1)
        s_decay = (vmin - vmax) / ((delta_t + EPS) * (vmean + EPS))
        ax.plot(tau, w, linewidth=3, color="#EC5241")
        x_line = np.linspace(0, 1, w.size)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, color="black", linestyle="-")
        vline_to_curve(
            t_min_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        vline_to_curve(
            t_max_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        hline_label(vmean, "Vmean", va="bottom")
        info_box([rf"$S_{{decay}}= {s_decay:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "Delta_t_over_T":
        w = rectified(sig)
        idx_min = int(np.nanargmin(w))
        idx_max = int(np.nanargmax(w))
        t_min_over_T = float(idx_min / w.size)
        t_max_over_T = float(idx_max / w.size)
        delta_t = idx_min - idx_max
        delta_t_over_t = float(delta_t / w.size)
        ax.plot(tau, w, linewidth=3, color="#EC5241")

        vline_to_curve(
            t_min_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        vline_to_curve(
            t_max_over_T, tau, w, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        # ax.hlines(5, xmin=t_max_over_T, xmax=t_min_over_T, color="black", linewidth=1)
        ax.annotate(
            "",
            xy=(t_max_over_T, 5),
            xytext=(t_min_over_T, 5),
            arrowprops=dict(arrowstyle="<->"),
        )
        ax.text((t_max_over_T + t_min_over_T) / 2, 5 + 2, r"$\Delta_t$", ha="center")

        info_box([rf"$\Delta_t/T = {delta_t_over_t:.3f}$"])

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_rise_normalized":
        w = rectified(sig)

        meanv = float(np.nanmean(w))

        dt = 1 / w.size
        dvdt = np.gradient(w, dt)

        idx_max = int(np.nanargmax(dvdt))
        tau_max = tau[idx_max]
        w_max = w[idx_max]

        s_up = float(np.nanmax(dvdt))
        slope_rise_tot = s_up / (meanv + EPS)
        T_max_slope_rise = s_up

        x_norm = tau
        x_norm_line = np.linspace(0, 1, 200)
        y_norm_line = slope_rise_tot * x_norm_line

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        ax.plot(
            x_norm_line,
            y_norm_line,
            linestyle="--",
            color="black",
            linewidth=2,
            label=rf"$S_{{rise}}= {slope_rise_tot:.3f}$",
        )

        x_tan = np.linspace(0, 1, 400)
        y_tan = T_max_slope_rise * (x_tan - tau_max) + w_max

        mask = (y_tan >= 0) & (y_tan <= ax.get_ylim()[1])
        ax.plot(
            x_tan[mask],
            y_tan[mask],
            linestyle=":",
            color="black",
            linewidth=2,
            label=rf"$S_{{rise}} = {T_max_slope_rise:.3f}$",
        )

        ax.legend(fontsize=11)

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_fall_normalized":
        w = rectified(sig)

        meanv = float(np.nanmean(w))

        dt = 1 / w.size
        dvdt = np.gradient(w, dt)

        idx_min = int(np.nanargmin(dvdt))
        tau_min = tau[idx_min]
        w_min = w[idx_min]

        s_up = float(np.nanmin(dvdt))
        slope_rise_tot = s_up / (meanv + EPS)
        T_min_slope_rise = s_up

        x_norm = tau
        x_norm_line = np.linspace(0, 1, 200)
        y_norm_line = slope_rise_tot * x_norm_line

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        ax.plot(
            x_norm_line,
            y_norm_line,
            linestyle="--",
            color="black",
            linewidth=2,
            label=rf"$S_{{rise}}= {slope_rise_tot:.3f}$",
        )

        x_tan = np.linspace(0, 1, 400)
        y_tan = T_min_slope_rise * (x_tan - tau_min) + w_min

        mask = (y_tan >= 0) & (y_tan <= ax.get_ylim()[1])
        ax.plot(
            x_tan[mask],
            y_tan[mask],
            linestyle=":",
            color="black",
            linewidth=2,
            label=rf"$S_{{rise}} = {T_min_slope_rise:.3f}$",
        )

        ax.legend(fontsize=11)

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "R_SD":
        w = rectified(sig)
        vmax = float(np.nanmax(w))
        idx_start = int(np.ceil(0.75 * w.size))
        idx_end = int(np.ceil(0.90 * w.size))

        if idx_start == idx_end:
            idx_end = idx_start + 1
        tail = w[idx_start:idx_end]

        vend = np.nanmedian(tail)
        D = float(np.nansum(tail))

        r_sd = float(vmax / (vend + EPS))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        hline_label(vmax, "Vmax", va="bottom")
        ax.axhline(vend, linestyle="--", linewidth=1, color="black")
        ax.text(
            0,
            vend,
            rf"$ \overline{{Vend}}={vend:.3g}$",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        ax.fill_between(
            tau[idx_start:idx_end], 0, tail, where=np.isfinite(tail), color="#F2CCC7"
        )

        info_box([f"D={D:.3g} ", f"R_SD={r_sd:.3f}"])

        ax.vlines(
            tau[idx_start], 0, w[idx_start], color="black", linestyles="--", linewidth=1
        )

        ax.vlines(
            tau[idx_end], 0, w[idx_end], color="black", linestyles="--", linewidth=1
        )

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "gamma_t":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if not np.isfinite(m0) or m0 <= 0:
            info_box("Invalid signal")
            return

        mu = float(np.nansum(w * tau) / m0)
        var = float(np.nansum(w * (tau - mu) ** 2) / (m0 + EPS))
        sigma = float(np.sqrt(max(var, 0.0)))
        z = (tau - mu) / (sigma + EPS)
        gamma = float(np.nansum(w * (z**3)) / (m0 + EPS))

        ax.plot(tau, w, linewidth=3, color="#EC5241")
        vline_to_curve(
            mu, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1.5
        )

        left = max(0.0, mu - sigma)
        right = min(1.0, mu + sigma)
        idx_left = int(np.floor(left * n))
        idx_right = int(np.ceil(right * n))
        ax.fill_between(
            tau[idx_left:idx_right],
            0,
            w[idx_left:idx_right],
            where=np.isfinite(w[idx_left:idx_right]),
            color="#F2CCC7",
        )
        vline_to_curve(
            mu - sigma, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        vline_to_curve(
            mu + sigma, tau, w, y0=0, color="#000000", linestyles="--", linewidth=1
        )

        info_box(
            [
                rf"$\mu_t/T={mu:.3f}$",
                rf"$\sigma_t/T={sigma:.3f}$",
                rf"$\gamma_t ={gamma:.3f}$",
            ]
        )

        ax.set_xlabel("rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v_b\: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "Delta_DTI":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if m0 <= 0:
            info_box("Invalid signal")
            return

        d = np.nancumsum(w)

        # normalisation
        d_star = d / (m0 + EPS)
        d0_star = tau

        a = d_star - d0_star

        delta_dti = np.sum(a)

        ax.plot(d0_star, a, linewidth=3, color="#EC5241")
        ax.fill_between(
            d0_star,
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
    else:
        info_box(f"No illustration for {metric}")


def export_selected_metric_pngs_bandlimited(
    all_results, zip_path, out_dir, dataset_path=SIGNAL_DATASET_PATH
):
    os.makedirs(out_dir, exist_ok=True)

    if "bandlimited" not in all_results:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # index des chemins .h5 par groupe / fichier
        h5_index = build_h5_path_index_from_extracted_tree(tmpdir)

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

                if path and os.path.exists(path):
                    sig = extract_mean_signal_per_file(path, dataset_path)
                    plot_metric_illustration(ax, metric, sig, path)
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
