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

METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
SELECTED_METRICS_PNG = {
    "RI",
    "PI",
    "crest_factor",
    "t50_over_T",
    "R_VTI",
    "SF_VTI",
    "Hspec",
    "mu_t_over_T",
    "sigma_t_over_T",
    "delta_phi2",
}
SIGNAL_DATASET_PATH = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
METRIC_ALIASES = {
    "Hspec": "spectral_entropy",
}
EPS = 1e-12
H_MAX = 10
LATEX_FORMULAS = {
    "RI": r"$RI = 1 - \frac{V_{\min}}{V_{\max}}$",
    "crest_factor": [
        r"$CF=\frac{\max_{t}\, v_b(t)}{\mathrm{RMS}(v_b(t))}$",
        r"$\mathrm{RMS}(x)=\sqrt{\frac{1}{N}\sum x^2}$",
    ],
    "t50_over_T": r"$\frac{t_{50}}{T}$"
    r" with $C(t)=\frac{\sum_{\tau\leq t} v(\tau)}{\sum_{\tau} v(\tau)}$"
    r" and $C(t_{50})=0.5$",
    "R_VTI": r"$R_{VTI}=\frac{\sum_{t\leq T/2} v(t)}{\sum_{t> T/2} v(t)+\varepsilon}$",
    "Hspec": [
        r"$H_{spec}=-\sum_{n=1}^{H} p_n\log(p_n+\varepsilon)$",
        r"$p_n=\frac{|V_n|}{\sum_{k=1}^{H}|V_k|}$",
    ],
    "mu_t_over_T": [
        r"$\mu/T$",
        r"$\mu=\frac{\sum_{i=1}^{N} w_i\,t_i}{\sum_{i=1}^{N} w_i}$",
        r"$w_i=\max(v_i,0)$",
    ],
    "PI": r"$PI=\frac{V_{\max}-V_{\min}}{V_{\mathrm{mean}}}$",
    "SF_VTI": r"$SF_{VTI}=\frac{\sum_{t\leq T/3} v(t)}{\sum_{t\leq T} v(t)}$",
    "sigma_t_over_T": [
        r"$\sigma/T$",
        r"$\sigma=\sqrt{\frac{\sum w_i (t_i-\mu)^2}{\sum w_i}}$",
        r"$\mu=\frac{\sum w_i t_i}{\sum w_i}$",
    ],
    "delta_phi2": r"$\Delta\phi_2=\mathrm{wrap}(\phi_2-2\phi_1)$",
}


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

    vv = np.where(np.isfinite(v), v, np.nan)
    m0 = float(np.nansum(vv))
    if m0 <= 0:
        return np.nan, None, None

    c = np.cumsum(vv) / m0
    idx = int(np.searchsorted(c, q, side="left"))
    idx = max(0, min(v.size - 1, idx))

    return float(idx / v.size), c, idx


def harmonic_pack(v):
    """
    Reproduit _harmonic_pack sans Tbeat (pas nécessaire pour vb)
    """
    v = np.asarray(v, dtype=float)
    vv = np.where(np.isfinite(v), v, np.nan)
    n = vv.size
    if n < 2:
        return None, None, None

    Vfull = np.fft.rfft(vv) / float(n)
    H = int(min(H_MAX, Vfull.size - 1))
    V = Vfull[: H + 1].copy()

    Vtrunc = np.zeros_like(Vfull)
    Vtrunc[: H + 1] = V
    vb = np.fft.irfft(Vtrunc * float(n), n=n)

    return V, vb, H


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


def plot_metric_illustration(ax, metric, sig_control):
    sig = np.asarray(sig_control, dtype=float)
    sig = np.where(np.isfinite(sig), sig, np.nan)

    # --- helpers ---
    def _y_at(x0, x_grid, y_grid):
        # Interpolation linéaire, ignore NaN en remplaçant par 0 pour l'interp local
        y = np.asarray(y_grid, dtype=float)
        x = np.asarray(x_grid, dtype=float)
        if len(x) < 2:
            return np.nan
        # pour éviter np.interp qui n'aime pas les NaN
        y2 = np.where(np.isfinite(y), y, 0.0)
        return float(np.interp(x0, x, y2))

    def vline_to_curve(x0, x_grid, y_grid, y0=0.0, **kwargs):
        y1 = _y_at(x0, x_grid, y_grid)
        if np.isfinite(y1):
            ax.vlines(x0, y0, y1, **kwargs)
        return y1

    def hline_label(y, label, va="bottom"):
        ax.axhline(y, linestyle="--", linewidth=1)
        ax.text(
            0.98,
            y,
            f" {label}={y:.3g}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va=va,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Rectification cohérente avec ton pipeline quand nécessaire
    # (RI/PI utilisent vmin sur v rectifié_keep_nan → vmin>=0 typiquement, mais on garde la logique simple)
    def rectified(v):
        v = np.asarray(v, dtype=float)
        return np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)

    n = sig.size
    if n < 2:
        ax.text(0.5, 0.5, "Signal too short", ha="center", va="center")
        return

    # τ normalisé 0..1 (endpoint=False comme ton pipeline: dt=T/n)
    tau = np.linspace(0.0, 1.0, n, endpoint=False)

    # =========================
    # RI
    # =========================
    if metric == "RI":
        vv = rectified(sig)
        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(vv))
        RI = 1.0 - (vmin / vmax) if vmax > 0 else np.nan

        ax.plot(tau, vv, linewidth=2)

        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")

        ax.text(
            0.01,
            0.95,
            f"RI = 1 - Vmin/Vmax = {RI:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel("Velocity")

    # =========================
    # mu_t_over_T   (mu en ratio, pas index)
    # =========================
    elif metric == "mu_t_over_T":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if not np.isfinite(m0) or m0 <= 0:
            ax.text(0.5, 0.5, "Invalid signal for mu/T", ha="center", va="center")
            return

        mu_over_T = float(np.nansum(w * tau) / m0)  # directement en ratio
        ax.plot(tau, w, linewidth=2)

        # ligne verticale qui s'arrête sur la courbe
        vline_to_curve(
            mu_over_T, tau, w, y0=0.0, colors="k", linestyles="--", linewidth=1.5
        )

        ax.text(
            mu_over_T,
            0.98,
            r"$\mu/T$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.text(
            0.01,
            0.95,
            f"mu/T = {mu_over_T:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel(r"$w(\tau)=\max(v(\tau),0)$")

    # =========================
    # PI
    # =========================
    elif metric == "PI":
        vv = rectified(sig)

        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(vv))
        vmean = float(np.nanmean(vv))

        PI = (
            (vmax - vmin) / (vmean + EPS)
            if (np.isfinite(vmean) and vmean > 0)
            else np.nan
        )

        ax.plot(tau, vv, linewidth=2)

        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        hline_label(vmean, "Vmean", va="bottom")

        ax.text(
            0.01,
            0.95,
            f"PI = (Vmax-Vmin)/Vmean = {PI:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel("Velocity")

    # =========================
    # SF_VTI  (split à 1/3) + aire sous courbe seulement
    # =========================
    elif metric == "SF_VTI":
        vv = rectified(sig)

        k = int(np.ceil(n * (1.0 / 3.0)))
        k = max(0, min(n, k))
        tau_k = float(k / n)  # ratio correspondant (proche de 1/3)

        D1 = float(np.nansum(vv[:k])) if k > 0 else np.nan
        Dtot = float(np.nansum(vv))
        sf = D1 / (Dtot + EPS)

        ax.plot(tau, vv, linewidth=2)

        ax.fill_between(
            tau[:k],
            0,
            vv[:k],
            where=np.isfinite(vv[:k]),
            color="#f7b6d2",
            alpha=0.55,
            label=r"$0 \rightarrow 1/3$",
        )
        ax.fill_between(
            tau[k:],
            0,
            vv[k:],
            where=np.isfinite(vv[k:]),
            color="#fdd0a2",
            alpha=0.35,
            label=r"$1/3 \rightarrow 1$",
        )

        # ligne verticale qui s'arrête sur la courbe
        vline_to_curve(tau_k, tau, vv, y0=0.0, colors="k", linestyles="--", linewidth=1)

        ax.text(
            0.01,
            0.95,
            f"D1={D1:.3g}, Dtot={Dtot:.3g}\nSF_VTI={sf:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel("Velocity")
        ax.legend(loc="upper right", frameon=False)

    # =========================
    # sigma_t_over_T  (mu et sigma en ratio)
    # =========================
    elif metric == "sigma_t_over_T":
        w = rectified(sig)
        m0 = float(np.nansum(w))
        if not np.isfinite(m0) or m0 <= 0:
            ax.text(0.5, 0.5, "Invalid signal", ha="center", va="center")
            return

        mu = float(np.nansum(w * tau) / m0)  # ratio
        var = float(np.nansum(w * (tau - mu) ** 2) / (m0 + EPS))
        sigma = float(np.sqrt(max(var, 0.0)))  # ratio
        sigma_over_T = sigma  # déjà "over_T"

        ax.plot(tau, w, linewidth=2)

        # mu : verticale arrêtée sur la courbe
        vline_to_curve(mu, tau, w, y0=0.0, colors="k", linestyles="--", linewidth=1.5)

        # bande ±sigma (en abscisse), pas rectangle pleine hauteur
        left = max(0.0, mu - sigma)
        right = min(1.0, mu + sigma)
        ax.axvspan(left, right, alpha=0.15)

        ax.text(
            0.01,
            0.95,
            f"mu/T={mu:.3f}\nsigma/T={sigma_over_T:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel(r"$w(\tau)=\max(v(\tau),0)$")

    # =========================
    # delta_phi2 : tracer H1 et H2 "tels quels" + noter Δφ2
    # (tu demandais de juste visualiser le déphasage entre les 2 harmoniques)
    # =========================
    elif metric == "delta_phi2":
        vv = rectify_keep_nan(sig)  # ton helper: garde NaN, rectifie
        vv = np.where(np.isfinite(vv), vv, np.nan)
        if n < 2:
            return

        Vfull = np.fft.rfft(vv) / float(n)
        if Vfull.size < 3:
            ax.text(0.5, 0.5, "Need at least 2 harmonics", ha="center", va="center")
            return

        V1, V2 = Vfull[1], Vfull[2]
        A1, A2 = float(np.abs(V1)), float(np.abs(V2))
        phi1, phi2 = float(np.angle(V1)), float(np.angle(V2))

        dphi2 = float((phi2 - 2.0 * phi1 + np.pi) % (2.0 * np.pi) - np.pi)

        m = 500
        tau_dense = np.linspace(0.0, 1.0, m, endpoint=False)
        omega = 2.0 * np.pi

        h1 = A1 * np.cos(omega * tau_dense + phi1)
        h2 = A2 * np.cos(2.0 * omega * tau_dense + phi2)

        ax.plot(tau_dense, h1, linewidth=2, label=r"$A_1\cos(2\pi\tau+\phi_1)$")
        ax.plot(tau_dense, h2, linewidth=2, label=r"$A_2\cos(4\pi\tau+\phi_2)$")

        ax.text(
            0.01,
            0.95,
            f"A1={A1:.2g}, φ1={phi1:.2f} rad\nA2={A2:.2g}, φ2={phi2:.2f} rad\nΔφ2 = wrap(φ2 − 2φ1) = {dphi2:.2f} rad",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel("Harmonic component")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    # =========================
    # crest_factor : RMS + Vmax (horizontales OK) + x normalisé (optionnel)
    # Ici on garde x en τ aussi (plus cohérent visuellement)
    # =========================
    elif metric == "crest_factor":
        V, vb, H = harmonic_pack(sig)
        if vb is None or vb.size < 2:
            ax.text(0.5, 0.5, "Invalid vb", ha="center", va="center")
            return

        vb = np.asarray(vb, dtype=float)
        vb_tau = np.linspace(0.0, 1.0, vb.size, endpoint=False)

        rms = float(np.sqrt(np.nanmean(vb**2)))
        vmax = float(np.nanmax(vb))
        cf = crest_factor_from_vb(vb)

        ax.plot(vb_tau, vb, linewidth=2)
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(rms, "RMS", va="top")

        ax.text(
            0.01,
            0.95,
            f"H={H}\nCF = Vmax/RMS = {cf:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel(r"$v_b(\tau)$")

    # =========================
    # t50_over_T : x normalisé + traits stoppés sur le point utile
    # (on garde ton idée D(t)=C(t)-x + tangente)
    # =========================
    elif metric == "t50_over_T":
        vv = rectified(sig)
        m0 = float(np.nansum(vv))
        if m0 <= 0:
            return

        C = np.nancumsum(vv) / m0
        if len(C) < 2:
            return

        # x_norm = tau (mêmes longueurs)
        x_norm = tau
        d = C - x_norm

        idx = int(np.searchsorted(C, 0.5, side="left"))
        idx = max(0, min(n - 1, idx))
        t50 = float(x_norm[idx])  # ratio

        ax.plot(x_norm, d, linewidth=2)

        # repères t50: verticale arrêtée à d(t50)
        y_t50 = _y_at(t50, x_norm, d)
        if np.isfinite(y_t50):
            ax.vlines(t50, 0.0, y_t50, linestyles="--", linewidth=1.2)

        # tangente sur les premiers points (comme toi)
        k = min(4, n)
        x_fit = x_norm[:k]
        y_fit = d[:k]
        mask = np.isfinite(x_fit) & np.isfinite(y_fit)

        if np.sum(mask) >= 2:
            a, b = np.polyfit(x_fit[mask], y_fit[mask], 1)
            y_line = a * x_norm + b
            ax.plot(x_norm, y_line, linestyle="--", linewidth=1.5)

            ax.text(
                0.01,
                0.95,
                f"t50/T = {t50:.3f}\ntangent slope ≈ {a:.3g}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        else:
            ax.text(
                0.01,
                0.95,
                f"t50/T = {t50:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax.set_xlabel(r"Normalized time $x=\tau=t/T$")
        ax.set_ylabel(r"$D(\tau)=C(\tau)-\tau$")

    # =========================
    # R_VTI : aires sous courbe, split à 0.5 en τ, ligne verticale stoppée
    # =========================
    elif metric == "R_VTI":
        vv = rectified(sig)

        k = int(np.ceil(n * 0.5))
        k = max(0, min(n, k))
        tau_k = float(k / n)

        D1 = float(np.nansum(vv[:k])) if k > 0 else np.nan
        D2 = float(np.nansum(vv[k:])) if k < n else np.nan
        R = D1 / (D2 + EPS)

        ax.plot(tau, vv, linewidth=2)

        ax.fill_between(
            tau[:k],
            0,
            vv[:k],
            where=np.isfinite(vv[:k]),
            color="#f7b6d2",
            alpha=0.55,
            label=r"$0 \rightarrow 1/2$",
        )
        ax.fill_between(
            tau[k:],
            0,
            vv[k:],
            where=np.isfinite(vv[k:]),
            color="#fdd0a2",
            alpha=0.55,
            label=r"$1/2 \rightarrow 1$",
        )

        vline_to_curve(0.5, tau, vv, y0=0.0, colors="k", linestyles="--", linewidth=1)

        ax.text(
            0.01,
            0.95,
            f"D1={D1:.3g}, D2={D2:.3g}\nR_VTI={R:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(r"Normalized time $\tau=t/T$")
        ax.set_ylabel("Velocity")
        ax.legend(loc="upper right", frameon=False)

    # =========================
    # Hspec / spectral_entropy : barres p_n + ligne uniforme 1/H
    # =========================
    elif metric in {"Hspec", "spectral_entropy"}:
        V, vb, H = harmonic_pack(sig)
        if V is None or V.size < 2:
            ax.text(0.5, 0.5, "Invalid harmonics", ha="center", va="center")
            return

        mags = np.abs(V[1:])
        mags = np.where(np.isfinite(mags), mags, np.nan)
        s = float(np.nansum(mags))
        if s <= 0:
            return

        p = mags / s
        Hn = len(p)
        ent = spectral_entropy_from_harmonics(V)

        xh = np.arange(1, Hn + 1)
        ax.bar(xh, p, width=0.8)

        uniform = 1.0 / Hn
        ax.axhline(uniform, linestyle="--", linewidth=1)
        ax.text(
            0.98,
            uniform,
            f" 1/H={uniform:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.text(
            0.01,
            0.95,
            f"H={Hn}\nHspec = -Σ p log(p+ε) = {ent:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel("Harmonic n")
        ax.set_ylabel(r"$p_n = |V_n|/\sum_{k=1}^{H}|V_k|$")

    else:
        ax.text(0.5, 0.5, f"No illustration for {metric}", ha="center", va="center")


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

            # patient représentatif (le plus proche de la médiane)
            rep_file = select_representative_file_per_group(df, value_col="mean")

            # ===== Layout figure =====
            n_groups = len(groups)
            ncols = 2
            nrows = int(np.ceil(n_groups / ncols))

            fig = plt.figure(figsize=(12, 6 + 3.2 * nrows), dpi=200)

            # un GridSpec global : 1 ligne pour le scatter + nrows pour les illustrations
            gs = gridspec.GridSpec(
                1 + nrows,
                ncols,
                height_ratios=[1.1] + [1.0] * nrows,
                hspace=0.55,
                wspace=0.35,
            )

            # réserve une bande en haut pour le header latex
            fig.subplots_adjust(top=0.88)

            # ===== Header latex =====
            ax_header = fig.add_axes([0.06, 0.91, 0.88, 0.08])
            ax_header.axis("off")
            formula = LATEX_FORMULAS.get(metric, "")
            if isinstance(formula, (list, tuple)):
                draw_inline_formulas_ax(
                    ax_header, formula, y=0.5, fontsize=16, gap=0.04
                )
            elif formula:
                ax_header.text(
                    0.0,
                    0.5,
                    formula,
                    ha="left",
                    va="center",
                    fontsize=14,
                    transform=ax_header.transAxes,
                )

            # ===== Panel du haut : scatter (prend 2 colonnes) =====
            ax_top = fig.add_subplot(gs[0, :])

            if control_name in x_pos:
                cx = x_pos[control_name]
                ax_top.axvspan(cx - 0.5, cx + 0.5, alpha=0.10)

            rng = np.random.default_rng(0)
            for g in groups:
                gdf = df[df["group"] == g]
                x = np.full(len(gdf), x_pos[g], dtype=float) + rng.normal(
                    0, 0.06, size=len(gdf)
                )
                is_control = g == control_name
                ax_top.scatter(
                    x,
                    gdf["mean"].values,
                    s=18 if not is_control else 28,
                    alpha=0.75 if not is_control else 0.95,
                    edgecolors="none",
                )

                if g in grp_mean.index:
                    ax_top.errorbar(
                        [x_pos[g]],
                        [grp_mean.loc[g]],
                        yerr=[grp_std.loc[g] if pd.notna(grp_std.loc[g]) else 0],
                        fmt="D",
                        capsize=5,
                        markersize=5 if not is_control else 7,
                        linewidth=1.2,
                    )

            ax_top.set_title(f"{metric} (bandlimited) — per group")
            ax_top.set_xticks([x_pos[g] for g in groups])
            ax_top.set_xticklabels(groups, rotation=0)
            ax_top.grid(True, axis="y", alpha=0.25)

            # ===== Illustrations : 1 subplot par groupe =====
            for i, g in enumerate(groups):
                r = 1 + (i // ncols)
                c = i % ncols
                ax = fig.add_subplot(gs[r, c])

                chosen = rep_file.get(g, None)
                path = h5_index.get(g, {}).get(chosen, None) if chosen else None

                if path and os.path.exists(path):
                    sig = extract_mean_signal_per_file(path, dataset_path)
                    plot_metric_illustration(ax, metric, sig)
                    ax.set_title(f"Illustration — {g}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No representative file for {g}",
                        ha="center",
                        va="center",
                    )
                    ax.axis("off")

            # si nombre impair : masque l'axe vide
            if n_groups % ncols == 1:
                ax_empty = fig.add_subplot(gs[1 + (n_groups // ncols), 1])
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
                marker=dict(color=color_map[g], size=7, opacity=0.6),
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
                marker=dict(color=color_map[g], size=18),
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
        ymin -= margin
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

    results, single_group = analyze_zip(zip_path)

    save_dashboard(results, zip_path, single_group)
