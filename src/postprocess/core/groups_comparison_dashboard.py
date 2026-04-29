import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from tkinter import Tk, filedialog
import base64
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from angioeye_io.hdf5_io import find_first_existing_path, read_array
from angioeye_io.hdf5_schema import pipeline_path_candidates
from angioeye_io.archive_io import (
    extracted_zip_tree,
    reset_output_dir,
    replace_folder_in_zip,
)
from .grouped_batch import (
    build_group_order,
    build_grouped_h5_index,
    find_control_group_name,
    iter_grouped_h5_files_in_zip,
)

WAVEFORM_SHAPE_METRICS_PIPELINE = "waveform_shape_metrics"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
VALID_VESSELS = ["artery", "vein"]
PIPELINE_BASE_CANDIDATES_WINDKESSEL = pipeline_path_candidates(
    "Windkessel_RC", "bandlimited"
)

METHODS_WINDKESSEL = ["arx", "freq", "time_integral"]
METRICS_WINDKESSEL = ["tau", "Deltat"]
POSTSCRIPT_BACKEND_MODULE = "matplotlib.backends.backend_ps"

METHOD_MARKERS_WINDKESSEL = {
    "arx": "D",
    "freq": "o",
    "time_integral": "^",
}

def _run_optional_eps_export(export_func, output_dir: str) -> bool:
    try:
        export_func()
    except ModuleNotFoundError as exc:
        if exc.name != POSTSCRIPT_BACKEND_MODULE:
            raise
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        print(
            "[WARN] EPS export skipped because the Matplotlib PostScript backend "
            f"'{POSTSCRIPT_BACKEND_MODULE}' is unavailable in this build."
        )
        return False
    return True

def get_metrics_base_candidates(vessel: str) -> list[str]:
    return pipeline_path_candidates(WAVEFORM_SHAPE_METRICS_PIPELINE, vessel, "global")


def get_mode_path_candidates(vessel: str, mode: str) -> list[str]:
    return pipeline_path_candidates(
        WAVEFORM_SHAPE_METRICS_PIPELINE, vessel, "global", mode
    )


def extract_windkessel_rows_from_h5(h5_path, group_name):
    rows = []

    with h5py.File(h5_path, "r") as f:
        base = find_first_existing_path(f, PIPELINE_BASE_CANDIDATES_WINDKESSEL)
        if base is None:
            return rows

        for method in METHODS_WINDKESSEL:
            for metric in METRICS_WINDKESSEL:
                dataset_path = f"{base}/{method}/{metric}"
                values = read_array(f, dataset_path, dtype=float)
                if values is None:
                    continue

                values = values[np.isfinite(values)]

                for v in values:
                    rows.append(
                        {
                            "file": os.path.basename(h5_path),
                            "group": group_name,
                            "method": method,
                            "metric": metric,
                            "value": float(v),
                        }
                    )

    return rows


SELECTED_METRICS_PNG = {
    "mu_t_over_T",
    "RI",
    "PI",
    "R_VTI",
    "SF_VTI",
    "sigma_t_over_T",
    "W50_over_T",
    "W80_over_T",
    "E_low_over_E_total",
    "t_max_over_T",
    "t_min_over_T",
    "Delta_t_over_T",
    "slope_rise_normalized",
    "slope_fall_normalized",
    "t_up_over_T",
    "t_down_over_T",
    "crest_factor",
    "Delta_DTI",
    "gamma_t",
    "N_eff_over_T",
    "N_t_over_T",
    "s_t",
    "w_t",
    "s_d",
    "w_d",
    "v_end_over_v_mean",
    "E_slope",
    "E_curv",
    "t50_over_T",
    "t_phi_over_T",
    "t_phi_n_over_T",
    "rho_h",
    "w_h",
    "N_h_over_H_minus_1",
    "D_phi",
    "s_phi_over_T",
    "eta_h",
}
METRIC_ALIASES = {
    "Hspec": "spectral_entropy",
}
EPS = 1e-12
LATEX_FORMULAS = {
    "RI": r"$\rm RI$",
    "rho_h_90": r"$\rho_{h,90}$",
    "rho_h_95": r"$\rho_{h,95}$",
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
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",
    "E_recon_H_MAX": r"$E_{\mathrm{recon},H_{\max}}$",
    "s_t": r"$s_{\mathrm{t}}$",
    "w_t": r"$w_{\mathrm{t}}$",
    "s_d": r"$s_{\mathrm{d}}$",
    "w_d": r"$w_{\mathrm{d}}$",
    "v_end_over_v_mean": r"$R_{EM}$",
    "E_slope": r"$E_{\mathrm{slope}}$",
    "phase_locking_residual": r"$E_{\phi}$",
    "W50_over_T": r"$W_{50}/T$",
    "W80_over_T": r"$W_{80}/T$",
    "N_t_over_T": r"$N_t/T$",
    "t_phi_n_over_T": r"$t_{\Delta\phi_n}/T$",
    "t_phi_over_T": r"$t_{\phi}/T$",
    "D_phi": r"$D_{\phi}$",
    "s_phi_over_T": r"$s_{\Delta\phi}/T$",
    "eta_h": r"$\eta_h$",
    "rho_h": r"$\rho_{h}$",
    "w_h": r"$w_{h}$",
    "N_h_over_H_minus_1": r"$N_{H}/(H-1)$",
}


def extract_graphics_support(h5_path, vessel="artery", mode="bandlimited"):
    base_candidates = get_mode_path_candidates(vessel, mode)
    out = {}

    with h5py.File(h5_path, "r") as f:
        base = find_first_existing_path(f, base_candidates)
        if base is None:
            return None
        if base not in f:
            return None

        grp = f[base]
        for key in grp.keys():
            arr = np.array(grp[key])

            out[key] = arr.item() if arr.shape == () else arr
    return out


def analyze_zip_windkessel(zip_path):
    rows = []

    for grouped_file in iter_grouped_h5_files_in_zip(zip_path):
        try:
            rows.extend(
                extract_windkessel_rows_from_h5(
                    grouped_file.file_path,
                    grouped_file.group_name,
                )
            )
        except Exception as e:
            print(f"Erreur avec {grouped_file.file_path}: {e}")

    return pd.DataFrame(rows)


def plot_windkessel_metric_for_method(df, metric, method, out_path):
    sub = df[(df["metric"] == metric) & (df["method"] == method)].copy()

    if sub.empty:
        print(f"Aucune donnée pour metric={metric}, method={method}")
        return

    groups = build_group_order(sub["group"].dropna().unique().tolist())
    x_pos = {g: i for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    ax.set_facecolor("#f2f2f2")

    control_name = find_control_group_name(groups)
    if control_name in x_pos:
        cx = x_pos[control_name]
        ax.axvspan(cx - 0.5, cx + 0.5, color="#d9d9d9", zorder=0)

    rng = np.random.default_rng(0)

    for group in groups:
        gdf = sub[sub["group"] == group]
        if gdf.empty:
            continue

        x = np.full(len(gdf), x_pos[group], dtype=float) + rng.normal(
            0, 0.05, size=len(gdf)
        )

        ax.scatter(
            x,
            gdf["value"].values,
            s=22,
            color="black",
            zorder=2,
        )

    stats = sub.groupby("group")["value"].agg(["mean", "std"]).reset_index()

    for _, row in stats.iterrows():
        group = row["group"]
        mean_val = row["mean"]
        std_val = row["std"] if np.isfinite(row["std"]) else 0.0

        ax.errorbar(
            x_pos[group],
            mean_val,
            yerr=std_val,
            fmt=METHOD_MARKERS_WINDKESSEL[method],
            color="black",
            ecolor="black",
            elinewidth=1.8,
            capsize=6,
            markersize=13,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2.2,
            zorder=3,
        )

    ax.set_xticks([x_pos[g] for g in groups])
    ax.set_xticklabels(groups, fontsize=17)

    ylabel = "tau (s)" if metric == "tau" else "Deltat (s)"
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"Windkessel bandlimited - {metric} - {method}", fontsize=18, pad=15)

    ax.grid(True, axis="y", color="gray", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_windkessel_figures(zip_path, out_dir, format="png"):
    os.makedirs(out_dir, exist_ok=True)

    df = analyze_zip_windkessel(zip_path)

    if df.empty:
        print("Aucune donnée Windkessel trouvée dans le zip.")
        return

    for metric in METRICS_WINDKESSEL:
        for method in METHODS_WINDKESSEL:
            filename = f"windkessel_{metric}_{method}.{format}"
            out_path = os.path.join(out_dir, filename)

            plot_windkessel_metric_for_method(
                df,
                metric,
                method,
                out_path,
            )

    # CSV uniquement une fois (png par exemple)
    if format == "png":
        csv_path = os.path.join(out_dir, "windkessel_values.csv")
        df.to_csv(csv_path, index=False)


def _safe_norm(v):
    v = np.asarray(v, dtype=float)
    s = np.nansum(v)
    if not np.isfinite(s) or s <= EPS:
        return np.full_like(v, np.nan, dtype=float)
    return v / s


def _higher_harmonic_weights_from_support(support):
    """
    Retourne les poids normalisés des harmoniques n=2..H.
    On part de harmonic_energies si disponible, sinon harmonic_energies_weights.
    """
    e = np.asarray(support.get("harmonic_energies", []), dtype=float)

    if e.size == 0:
        w = np.asarray(support.get("harmonic_energies_weights", []), dtype=float)
        if w.size == 0:
            return np.array([], dtype=float)
        # w supposé sur n=1..H -> on enlève la fondamentale
        hh = w[1:] if w.size >= 2 else np.array([], dtype=float)
        return _safe_norm(hh)

    # harmonic_energies supposé sur n=1..H
    hh = e[1:] if e.size >= 2 else np.array([], dtype=float)
    return _safe_norm(hh)



def _phase_delay_equivalents_from_support(support):
    """
    t_{Δφ,n}/T = Δφ_n / (2π n), avec n à partir de 2.
    """
    dphi = np.asarray(support.get("delta_phi_all", []), dtype=float)
    if dphi.ndim == 0 or dphi.size == 0:
        return np.array([], dtype=float)

    # après select_support_beat, on s'attend à un vecteur 1D sur n=2..H
    dphi = np.ravel(dphi).astype(float)
    n_vals = np.arange(2, 2 + len(dphi), dtype=float)
    return dphi / (2.0 * np.pi * n_vals)


def select_support_beat(support, beat_idx):
    out = {}
    for k, v in support.items():
        arr = np.asarray(v)
        if arr.ndim == 2:
            if k in {
                "harmonic_magnitudes",
                "harmonic_weights",
                "harmonic_phases",
                "harmonic_energies",
                "harmonic_energies_weights",
                "harmonic_energy_cumsum",
                "harmonic_energy_cumsum_h",
                "harmonic_energy_cumsum_interp",
                "harmonic_energy_cumsum_h_interp",
                "delta_phi_all",
                "A2_cumsum_interp",
                "A2_m_interp",
            }:
                out[k] = arr[beat_idx, :]

            else:
                out[k] = arr[:, beat_idx]
        elif arr.ndim == 1 and arr.shape[0] > beat_idx:
            out[k] = arr[beat_idx]
        else:
            out[k] = v

    return out




def plot_metric_illustration(ax, metric, support, path=None, vessel="artery"):
    main_color = "#EC5241" if vessel == "artery" else "#414CEC"
    fill_color1 = "#f9c2ca" if vessel == "artery" else "#BDDBE7"
    fill_color2 = "#F2CCC7" if vessel == "artery" else "#A1B2F2"
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
    harmonic_energies = np.asarray(support.get("harmonic_energies", []), dtype=float)
    harmonic_energies_weights = np.asarray(
        support.get("harmonic_energies_weights", []), dtype=float
    )
    harmonic_phases = np.asarray(support.get("harmonic_phases", []), dtype=float)
    delta_phi_all = np.asarray(support.get("delta_phi_all", []), dtype=float)
    H_MAX = int(np.asarray(support.get("H_MAX", 10)).item())
    H_LOW_MAX = int(np.asarray(support.get("H_LOW_MAX", 3)).item())
    H_HIGH_MIN = int(np.asarray(support.get("H_HIGH_MIN", 4)).item())
    H_HIGH_MAX = int(np.asarray(support.get("H_HIGH_MAX", 8)).item())
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

    # =========================
    # RI
    # =========================
    if metric == "RI":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        ri = float(support["RI"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        info_box([f"RI = {ri:.3f}"])
        ax.set_xlabel(r"rectified time :  t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "Delta_DTI":
        a = np.asarray(support.get("delta_dti_curve", []), dtype=float)
        delta_dti = float(support["Delta_DTI"])

        if a.size == 0:
            info_box("Missing Delta_DTI support")
            return
        x_lin = np.linspace(0, 1, n)
        ax.plot(x_lin, a, color=main_color, linewidth=3)
        ax.fill_between(
            x_lin,
            0,
            a,
            where=np.isfinite(a),
            hatch="//",
            facecolor="none",
            edgecolor=fill_color1,
        )
        info_box([rf"$\Delta_{{DTI}} = {delta_dti:.3f}$"])

        ax.set_xlabel(r"rectified time :  t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) - t/T \: (a.u.)$", fontsize=14, labelpad=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif metric == "PI":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        vmean = float(support["vmean"])
        pi = float(support["PI"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(vmin, "Vmin", va="top")
        hline_label(vmean, r"$\overline{{v}}$", va="bottom")
        info_box([f"PI = {pi:.3f}"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "rho_h":
        A2 = np.asarray(support.get("A2_cumsum", []), dtype=float)
        m_vals = np.asarray(support.get("A2_m", []), dtype=float)

        A2i = np.asarray(support.get("A2_cumsum_interp", []), dtype=float)
        mi = np.asarray(support.get("A2_m_interp", []), dtype=float)

        m80 = float(support.get("m_80", np.nan))
        rho_h = float(support.get("rho_h", np.nan))

        # retire le padding éventuel en NaN
        mask_disc = np.isfinite(A2) & np.isfinite(m_vals)
        A2 = A2[mask_disc]
        m_vals = m_vals[mask_disc]

        mask_interp = np.isfinite(A2i) & np.isfinite(mi)
        A2i = A2i[mask_interp]
        mi = mi[mask_interp]

        if A2.size == 0 and A2i.size == 0:
            info_box("Missing rho_h cumulative support")
            return

        ax.plot(mi, A2i, color=main_color, linewidth=3)

        # quantile 80%
        ax.axhline(0.80, linestyle="--", color="black", linewidth=1)

        if np.isfinite(m80):
            ax.axvline(m80, linestyle="--", color="black", linewidth=1)
            ax.plot(m80, 0.80, "o", color="black", markersize=5)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel(r"Interpolated cumulative index $m$", fontsize=14)
        ax.set_ylabel(r"$A^{(2)}(m)$", fontsize=14)

        info_box(
            [
                (
                    rf"$m_{{0.8}}={m80:.3f}$"
                    if np.isfinite(m80)
                    else r"$m_{0.8}=\mathrm{NaN}$"
                ),
                (
                    rf"$\rho_h={rho_h:.3f}$"
                    if np.isfinite(rho_h)
                    else r"$\rho_h=\mathrm{NaN}$"
                ),
            ]
        )
    elif metric == "w_h":
        A2i = np.asarray(support.get("A2_cumsum_interp", []), dtype=float)
        mi = np.asarray(support.get("A2_m_interp", []), dtype=float)
        m50 = float(support.get("m_50", np.nan))
        m80 = float(support.get("m_80", np.nan))
        w_h = float(support.get("w_h", np.nan))

        ax.plot(mi, A2i, color=main_color, linewidth=3)

        ax.axhline(0.50, linestyle="--", color="black", linewidth=1)
        ax.axhline(0.80, linestyle="--", color="black", linewidth=1)

        if np.isfinite(m50):
            ax.axvline(m50, linestyle="--", color="black", linewidth=1)
            ax.plot(m50, 0.50, "o", color="black", markersize=5)

        if np.isfinite(m80):
            ax.axvline(m80, linestyle="--", color="black", linewidth=1)
            ax.plot(m80, 0.80, "o", color="black", markersize=5)

        if np.isfinite(m50) and np.isfinite(m80):
            ax.axvspan(m50, m80, color="#cccccc")

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel(r"Interpolated cumulative index $m$", fontsize=14)
        ax.set_ylabel(r"$A^{(2)}(m)$", fontsize=14)

        info_box(
            [
                (
                    rf"$m_{{0.5}}={m50:.3f}$"
                    if np.isfinite(m50)
                    else r"$m_{0.5}=\mathrm{NaN}$"
                ),
                (
                    rf"$m_{{0.8}}={m80:.3f}$"
                    if np.isfinite(m80)
                    else r"$m_{0.8}=\mathrm{NaN}$"
                ),
                rf"$w_h={w_h:.3f}$" if np.isfinite(w_h) else r"$w_h=\mathrm{NaN}$",
            ]
        )
    elif metric == "N_h_over_H_minus_1":
        b = _higher_harmonic_weights_from_support(support)
        if b.size == 0:
            info_box("Missing higher-harmonic weights")
            return

        hspec = -np.nansum(np.where(b > 0, b * np.log(b), 0.0))
        nh_spec = float(np.exp(hspec))
        nh_spec_norm = float(support.get("N_h_over_H_minus_1", np.nan))
        if not np.isfinite(nh_spec_norm):
            nh_spec_norm = nh_spec / max(len(b), 1)

        xk = np.arange(1, len(b) + 1)
        ax.bar(xk, b, color=main_color, width=0.8)
        ax.set_yscale("log")

        info_box(
            [
                rf"$N_{{H,spec}}={nh_spec:.3f}$",
                rf"$N_{{H,spec}}/(H-1)={nh_spec_norm:.3f}$",
            ]
        )
        ax.set_xlabel(r"Higher-harmonic index $k=n-1$", fontsize=14)
        ax.set_ylabel(r"$b_k$ (a.u.)", fontsize=14, labelpad=12)
    elif metric == "D_phi":
        dphi = np.asarray(support.get("delta_phi_all", []), dtype=float)
        if dphi.size == 0:
            info_box("Missing phase data")
            return

        dphi = np.ravel(dphi).astype(float)
        n_vals = np.arange(2, 2 + len(dphi))
        wphi = _higher_harmonic_weights_from_support(support)
        if wphi.size != dphi.size:
            wphi = np.ones_like(dphi, dtype=float)
            wphi = _safe_norm(wphi)

        R_phi = np.abs(np.nansum(wphi * np.exp(1j * dphi)))
        D_phi = float(support.get("D_phi", np.nan))
        if not np.isfinite(D_phi):
            D_phi = 1.0 - R_phi

        ax.bar(n_vals, dphi, color=main_color, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axhline(np.pi, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(-np.pi, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        info_box([rf"$R_{{\phi}}={R_phi:.3f}$", rf"$D_{{\phi}}={D_phi:.3f}$"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$\Delta \phi_n$ (rad)", fontsize=14, labelpad=12)
    elif metric == "s_phi_over_T":
        tdphi_n = _phase_delay_equivalents_from_support(support)
        if tdphi_n.size == 0:
            info_box("Missing phase-delay equivalents")
            return

        n_vals = np.arange(2, 2 + len(tdphi_n))
        t_delta = float(support.get("t_phi_over_T", np.nan))
        if not np.isfinite(t_delta):
            t_delta = float(np.nanmedian(tdphi_n))

        s_delta = float(support.get("s_phi_over_T", np.nan))
        if not np.isfinite(s_delta):
            s_delta = float(np.nanmedian(np.abs(tdphi_n - t_delta)))

        ax.bar(n_vals, tdphi_n, color=main_color, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axhline(t_delta, color="black", linestyle="--", linewidth=1.0)
        ax.axhspan(
            t_delta - s_delta,
            t_delta + s_delta,
            facecolor="none",
            hatch="////",
            edgecolor="black",
            alpha=0.3,
            linewidth=0,
        )

        info_box(
            [
                rf"$t_{{\Delta\phi}}/T={t_delta:.3f}$",
                rf"$s_{{\Delta\phi}}/T={s_delta:.3f}$",
            ]
        )
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$t_{\Delta\phi,n}/T$ (a.u.)", fontsize=14, labelpad=12)
    elif metric == "eta_h":
        eta_h = float(support.get("eta_h", np.nan))
        if not np.isfinite(eta_h):
            # fallback si le support ne donne pas directement la métrique
            resid = (
                np.nansum((sig - vb[: len(sig)]) ** 2)
                if len(vb) == len(sig)
                else np.nan
            )
            denom = np.nansum((sig - np.nanmean(sig)) ** 2)
            eta_h = 1.0 - resid / max(denom, EPS)

        ax.plot(tau, sig, linewidth=3, color=main_color, label="signal")
        if len(vb) > 0:
            ax.plot(
                np.linspace(0.0, 1.0, len(vb), endpoint=False),
                vb,
                linestyle="--",
                linewidth=2,
                color="black",
                label="reconstruction",
            )

        info_box([rf"$\eta_h={eta_h:.3f}$", f"H={len(harmonic_magnitudes)}"])
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "mu_t_over_T":
        mu_over_T = float(support["mu_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            mu_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$\mu_t/T = {mu_over_T:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "sigma_t_over_T":
        mu = float(support["mu_t_over_T"])
        sigma = float(support["sigma_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            mu, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1.5
        )

        left = max(0.0, mu - sigma)
        right = min(1.0, mu + sigma)
        mask = (tau >= left) & (tau <= right)
        ax.fill_between(tau, 0, sig, where=mask & np.isfinite(sig), color=fill_color2)

        vline_to_curve(
            mu - sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )
        vline_to_curve(
            mu + sigma, tau, sig, y0=0, color="#000000", linestyles="--", linewidth=1
        )

        info_box([rf"$\mu_t/T={mu:.3f}$", rf"$\sigma_t/T={sigma:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t50_over_T":
        t10 = float(support["t10_over_T"])
        t50 = float(support["t50_over_T"])
        t90 = float(support["t90_over_T"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq in [t10, t50, t90]:
            yq = _y_at(tq, tau, C)
            if np.isfinite(yq):
                ax.vlines(tq, 0.0, yq, linestyles="--", linewidth=1, color="#000000")
                ax.hlines(yq, 0.0, tq, linestyles="--", linewidth=1, color="#000000")

        info_box(
            [
                rf"$t_{{10}}/T = {t10:.3f}, t_{{50}}/T = {t50:.3f}$",
                rf"$t_{{90}}/T = {t90:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$ ", fontsize=14, labelpad=12)

    elif metric == "R_VTI":
        ratio = float(support["R_VTI"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.fill_between(
            tau[tau < 0.5],
            0,
            sig[tau < 0.5],
            where=np.isfinite(sig[tau < 0.5]),
            color=fill_color1,
        )
        ax.fill_between(
            tau[tau >= 0.5],
            0,
            sig[tau >= 0.5],
            where=np.isfinite(sig[tau >= 0.5]),
            color=fill_color2,
        )
        vline_to_curve(
            0.5, tau, sig, y0=0.0, color="#000000", linestyles="--", linewidth=1
        )

        d1 = float(np.nansum(sig[tau < 0.5]))
        d2 = float(np.nansum(sig[tau >= 0.5]))
        info_box([rf"$D_1={d1:.3g} , D_2={d2:.3g}$", rf"$R_{{VTI}}={ratio:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "SF_VTI":
        sf = float(support["SF_VTI"])
        tau_k = 1.0 / 3.0

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.fill_between(
            tau[tau < tau_k],
            0,
            sig[tau < tau_k],
            where=np.isfinite(sig[tau < tau_k]),
            color=fill_color1,
        )
        ax.fill_between(
            tau,
            0,
            sig,
            where=np.isfinite(sig),
            hatch="//",
            facecolor="none",
            edgecolor=fill_color2,
        )
        vline_to_curve(
            tau_k, tau, sig, y0=0.0, color="#000000", linestyles="--", linewidth=1
        )

        d1 = float(np.nansum(sig[tau < tau_k]))
        dtot = float(np.nansum(sig))
        info_box([rf"$D_1={d1:.3g} , D_1 + D_2={dtot:.3g}$", rf"$SF_{{VTI}}={sf:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_max_over_T":
        t_max_over_T = float(support["t_max_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_max_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{max}}/T = {t_max_over_T:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_min_over_T":
        t_min_over_T = float(support["t_min_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_min_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{min}}/T = {t_min_over_T:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "Delta_t_over_T":
        t_max_over_T = float(support["t_max_over_T"])
        t_min_over_T = float(support["t_min_over_T"])
        delta_t = float(support["Delta_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_max_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        vline_to_curve(
            t_min_over_T, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$\Delta_t/T = {delta_t:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_up_over_T":
        t_up = float(support["t_up_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_up, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{up}}/T = {t_up:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_down_over_T":
        t_down = float(support["t_down_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_down, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{down}}/T = {t_down:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "S_decay":
        vmax = float(support["vmax"])
        vmin = float(support["vmin"])
        vmean = float(support["vmean"])
        t_max = float(support["t_max_over_T"])
        t_min = float(support["t_min_over_T"])
        s_decay = float(support["S_decay"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
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
        info_box([rf"$S_{{decay}}= {s_decay:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_rise_normalized":
        s_rise = float(support["slope_rise_normalized"])
        idx = int(np.nanargmax(dvdt))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{rise}}={s_rise:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "slope_fall_normalized":
        s_fall = float(support["slope_fall_normalized"])
        idx = int(np.nanargmin(dvdt))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{fall}}={s_fall:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "R_SD":
        ratio = float(support["R_SD"])
        vmax = float(support["vmax"])
        vend = float(support["vend"])
        i0 = int(support.get("late_window_start_idx", int(np.floor(0.75 * n))))
        i1 = int(support.get("late_window_end_idx", int(np.ceil(0.90 * n))))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.fill_between(
            tau[i0:i1], 0, sig[i0:i1], where=np.isfinite(sig[i0:i1]), color=fill_color2
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
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "gamma_t":
        gamma_t = float(support["gamma_t"])
        mu = float(support["mu_t_over_T"])
        sigma = float(support["sigma_t_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
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
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "rho_h_90":
        cumsum = np.asarray(support.get("harmonic_energy_cumsum", []), dtype=float)
        cumsum_h = np.asarray(support.get("harmonic_energy_cumsum_h", []), dtype=float)

        cumsum_interp = np.asarray(
            support.get("harmonic_energy_cumsum_interp", []), dtype=float
        )
        cumsum_h_interp = np.asarray(
            support.get("harmonic_energy_cumsum_h_interp", []), dtype=float
        )

        h90 = float(support.get("h_90", np.nan))
        rho90 = float(support.get("rho_h_90", np.nan))

        mask_i = np.isfinite(cumsum_interp) & np.isfinite(cumsum_h_interp)
        mask_d = np.isfinite(cumsum) & np.isfinite(cumsum_h)

        ax.plot(
            cumsum_h_interp[mask_i],
            cumsum_interp[mask_i],
            color=main_color,
            linewidth=2,
        )

        ax.plot(
            cumsum_h[mask_d],
            cumsum[mask_d],
            "o",
            color="black",
            markersize=4,
        )

        ax.axhline(0.90, linestyle="--", color="black", linewidth=1)
        if np.isfinite(h90):
            ax.axvline(h90, linestyle="--", color="black", linewidth=1)
            ax.plot(h90, 0.90, "o", color="black", markersize=5)

        ax.set_xlabel("Harmonic index $h$ (a.u.)", fontsize=14)
        ax.set_ylabel(r"$C(h)$", fontsize=14)
    elif metric == "rho_h_95":
        cumsum = np.asarray(support.get("harmonic_energy_cumsum", []), dtype=float)
        cumsum_h = np.asarray(support.get("harmonic_energy_cumsum_h", []), dtype=float)

        cumsum_interp = np.asarray(
            support.get("harmonic_energy_cumsum_interp", []), dtype=float
        )
        cumsum_h_interp = np.asarray(
            support.get("harmonic_energy_cumsum_h_interp", []), dtype=float
        )

        h95 = float(support.get("h_95", np.nan))
        rho95 = float(support.get("rho_h_95", np.nan))

        mask_i = np.isfinite(cumsum_interp) & np.isfinite(cumsum_h_interp)
        mask_d = np.isfinite(cumsum) & np.isfinite(cumsum_h)

        ax.plot(
            cumsum_h_interp[mask_i],
            cumsum_interp[mask_i],
            color=main_color,
            linewidth=2,
        )

        ax.plot(
            cumsum_h[mask_d],
            cumsum[mask_d],
            "o",
            color="black",
            markersize=4,
        )

        ax.axhline(0.95, linestyle="--", color="black", linewidth=1)
        if np.isfinite(h95):
            ax.axvline(h95, linestyle="--", color="black", linewidth=1)
            ax.plot(h95, 0.95, "o", color="black", markersize=5)

        ax.set_xlabel("Harmonic index $h$ (a.u.)", fontsize=14)
        ax.set_ylabel(r"$C(h)$", fontsize=14)
    elif metric == "mu_h":
        w_h = harmonic_energies_weights
        mu_h = float(support["mu_h"])
        xh = np.arange(1, len(w_h) + 1)
        ax.set_yscale("log")
        ax.bar(xh, w_h, width=0.8, color=main_color)
        ax.axvline(mu_h, linestyle="--", linewidth=1.2, color="black")
        info_box([rf"$\mu_h={mu_h:.3f}$", f"H={len(w_h)}"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$w_n\:(a.u.)$", fontsize=14, labelpad=12)

    elif metric == "sigma_h":
        w_h = harmonic_energies_weights
        mu_h = float(support["mu_h"])
        sigma_h = float(support["sigma_h"])
        xh = np.arange(1, len(w_h) + 1)
        ax.set_yscale("log")
        ax.bar(xh, w_h, width=0.8, color=main_color)
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

        ax.plot(tau, p**2, linewidth=3, color=main_color)
        ax.fill_between(tau, 0, p**2, where=np.isfinite(p**2), color=fill_color2)

        if metric == "N_eff":
            info_box([rf"$N_{{eff}} \approx {n_eff:.3f}$"])
        else:
            info_box([rf"$N_{{eff}}/T \approx {n_eff_over_t:.3f}$"])

        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$p^2(t)\: (a.u.)$", fontsize=14, labelpad=10)

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
            color=main_color,
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
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel("Harmonic component (a.u.) ", fontsize=14, labelpad=12)
        ax.legend(
            loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=10
        )

    elif metric == "crest_factor":
        cf = float(support["crest_factor"])
        vmax = float(np.nanmax(vb))
        rms = float(np.sqrt(np.nanmean(vb**2)))
        vb_tau = np.linspace(0.0, 1.0, len(vb), endpoint=False)

        ax.plot(vb_tau, vb, linewidth=3, color=main_color)
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(rms, "RMS", va="top")
        info_box([f"CF= {cf:.3f}"])
        ax.set_xlabel(r"rectified time :  t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric in {"Hspec", "spectral_entropy"}:
        p = harmonic_energies_weights
        hn = len(p)
        ent = float(support["spectral_entropy"])
        xh = np.arange(1, hn + 1)

        ax.bar(xh, p, width=0.8, color=main_color)
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
        ax.set_yscale("log")
        info_box([f"H={hn}", f"Hspec = {ent:.3f}"])
        ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
        ax.set_ylabel(r"$\tilde a_n$ (a.u.)", fontsize=14, labelpad=12)

    elif metric == "E_low_over_E_total":
        mags2 = harmonic_energies[1:]
        e_low = float(support["E_low"])
        e_total = float(support["E_total"])
        ratio = float(support["E_low_over_E_total"])
        xh = np.arange(1, len(mags2) + 1)

        ax.set_yscale("log")
        ax.bar(xh[: H_LOW_MAX + 1], mags2[: H_LOW_MAX + 1], color=main_color)
        ax.bar(xh[H_LOW_MAX:], mags2[H_LOW_MAX:], color="#cccccc")
        lines = [
            rf"$E_{{low}} = {e_low:.3g}$",
            rf"$E_{{total}} = {e_total:.3g}$",
            rf"$E_{{low}}/E_{{total}} = {ratio:.3f}$",
        ]
        text = "\n".join([str(x) for x in lines if x is not None and str(x) != ""])

        ax.text(
            0.5,
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

    elif metric == "E_high_over_E_total":
        mags2 = harmonic_energies
        ax.set_yscale("log")
        e_high = float(support["E_high"])
        e_total = float(support["E_total"])
        ratio = float(support["E_high_over_E_total"])
        xh = np.arange(0, len(mags2))

        ax.bar(xh[1:H_HIGH_MIN], mags2[1:H_HIGH_MIN], color="#cccccc")
        ax.bar(
            xh[H_HIGH_MIN : H_HIGH_MAX + 1],
            mags2[H_HIGH_MIN : H_HIGH_MAX + 1],
            color=main_color,
        )

        lines = [
            f"E_high = {e_high:.3g}",
            f"E_total = {e_total:.3g}",
            rf"$E_{{high}}/E_{{total}} = {ratio:.3f}$",
        ]
        text = "\n".join([str(x) for x in lines if x is not None and str(x) != ""])

        ax.text(
            0.5,
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

    elif metric == "E_recon_H_MAX":
        e_recon = float(support["E_recon_H_MAX"])

        ax.plot(tau, sig, linewidth=3, color=main_color, label="signal")
        ax.plot(
            np.linspace(0.0, 1.0, len(vb), endpoint=False),
            vb,
            linestyle="--",
            linewidth=2,
            color="black",
            label="reconstruction",
        )
        info_box(
            [rf"$E_{{recon,Hmax}}={e_recon:.4f}$", f"Hmax={len(harmonic_magnitudes)}"]
        )
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "s_t":
        t10 = float(support["t10_over_T"])
        t50 = float(support["t50_over_T"])
        t90 = float(support["t90_over_T"])
        s_t = float(support["s_t"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq in [t10, t50, t90]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
            ax.hlines(yq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$s_{{t}}={s_t:.3f}$",
                rf"$t_{{10}}={t10:.3f}, t_{{50}}={t50:.3f}, t_{{90}}={t90:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "w_t":
        t25 = float(support["t25_over_T"])
        t75 = float(support["t75_over_T"])
        w_t = float(support["w_t"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        y25 = _y_at(t25, tau, C)
        y75 = _y_at(t75, tau, C)
        ax.vlines(t25, 0, y25, linestyle="--", linewidth=1, color="black")
        ax.vlines(t75, 0, y75, linestyle="--", linewidth=1, color="black")
        ax.hlines(y25, 0, t25, linestyle="--", linewidth=1, color="black")
        ax.hlines(y75, 0, t75, linestyle="--", linewidth=1, color="black")
        ax.fill_between(tau, 0, C, where=(tau >= t25) & (tau <= t75), color=fill_color2)

        info_box(
            [rf"$w_{{t}}={w_t:.3f}$", rf"$t_{{25}}={t25:.3f}, t_{{75}}={t75:.3f}$"]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "R_Q_t":
        t10 = float(support["t10_over_T"])
        t25 = float(support["t25_over_T"])
        t50 = float(support["t50_over_T"])
        t75 = float(support["t75_over_T"])
        t90 = float(support["t90_over_T"])
        w_t = float(support["w_t"])
        s_t = float(support["s_t"])
        r_q_t = float(support["R_Q_t"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq in [t10, t25, t50, t75, t90]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
            ax.hlines(yq, 0, tq, linestyle="--", linewidth=1, color="black")
        ax.fill_between(tau, 0, C, where=(tau >= t25) & (tau <= t75), color=fill_color2)

        info_box(
            [
                rf"$w_{{t}}={w_t:.3f}$",
                rf"$s_{{t}}={s_t:.3f}$",
                rf"$R_{{Q_{{t}}}}={r_q_t:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "s_d":
        d10 = float(support["d10"])
        d50 = float(support["d50"])
        d90 = float(support["d90"])
        s_d = float(support["s_d"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq, dq in [(0.1, d10), (0.5, d50), (0.9, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$s_{{d}}={s_d:.3f}$",
                rf"$d_{{10}}={d10:.3f}, d_{{50}}={d50:.3f}, d_{{90}}={d90:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "w_d":
        d25 = float(support["d25"])
        d75 = float(support["d75"])
        w_d = float(support["w_d"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        ax.vlines(0.25, 0, d25, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.75, 0, d75, linestyle="--", linewidth=1, color="black")
        ax.hlines(d25, 0, 0.25, linestyle="--", linewidth=1, color="black")
        ax.hlines(d75, 0, 0.75, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color=fill_color2)

        info_box(
            [rf"$W_{{d}}={w_d:.3f}$", rf"$d_{{25}}={d25:.3f}, d_{{75}}={d75:.3f}$"]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "R_Q_d":
        d10 = float(support["d10"])
        d25 = float(support["d25"])
        d50 = float(support["d50"])
        d75 = float(support["d75"])
        d90 = float(support["d90"])
        w_d = float(support["w_d"])
        s_d = float(support["s_d"])
        r_q_d = float(support["R_Q_d"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq, dq in [(0.10, d10), (0.25, d25), (0.50, d50), (0.75, d75), (0.90, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color=fill_color2)

        info_box(
            [
                rf"$Q_{{d_{{width}}}}={w_d:.3f}$",
                rf"$Q_{{d_{{skew}}}}={s_d:.3f}$",
                rf"$R_{{Q_{{d}}}}={r_q_d:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "v_end_over_v_mean":
        vmean = float(support["vmean"])
        vend = float(support["vend"])
        ratio = float(support["v_end_over_v_mean"])
        i0 = int(support.get("late_window_start_idx", int(np.floor(0.75 * n))))
        i1 = int(support.get("late_window_end_idx", int(np.ceil(0.90 * n))))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.fill_between(
            tau[i0:i1], 0, sig[i0:i1], where=np.isfinite(sig[i0:i1]), color=fill_color2
        )
        hline_label(vmean, r"$\overline{{v}}$", va="bottom")
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
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_slope":
        e_slope = float(support["E_slope"])
        dvdt_norm = support["dvdt_norm"]

        ax.plot(tau, sig, linewidth=3, color=main_color, label="signal")
        ax2 = ax.twinx()
        ax2.plot(
            tau,
            dvdt_norm,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=r"$\dot v^2$",
        )
        ax2.set_ylabel(r"$\dot v^2$", fontsize=12)
        ax2.set_yticks([])
        info_box([rf"$E_{{slope}}={e_slope:.4f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_curv":
        e_curv = float(support["E_curv"])
        d2vdt2_norm = support["d2vdt2_norm"]
        ax.plot(tau, sig, linewidth=3, color=main_color, label="signal")
        ax2 = ax.twinx()
        ax2.plot(
            tau,
            d2vdt2_norm,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=r"$\ddot v^2$",
        )
        ax2.set_yticks([])
        ax2.set_ylabel(r"$\ddot v^2$", fontsize=12)
        info_box([rf"$E_{{curv}}={e_curv:.4f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "W50_over_T":
        w50 = float(support["W50_over_T"])
        vmax = float(support["vmax"])
        thr = 0.5 * vmax

        mask = np.isfinite(sig) & (sig >= thr)

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.axhline(thr, linestyle="--", linewidth=1, color="black")
        ax.fill_between(
            tau,
            0,
            sig,
            where=mask,
            color=fill_color2,
            interpolate=True,
        )

        info_box(
            [
                rf"$W_{{50}}/T = {w50:.3f}$",
                rf"$0.5\,V_{{max}} = {thr:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "W80_over_T":
        w80 = float(support["W80_over_T"])
        vmax = float(support["vmax"])
        thr = 0.8 * vmax

        mask = np.isfinite(sig) & (sig >= thr)

        ax.plot(tau, sig, linewidth=3, color=main_color)
        ax.axhline(thr, linestyle="--", linewidth=1, color="black")
        ax.fill_between(
            tau,
            0,
            sig,
            where=mask,
            color=fill_color2,
            interpolate=True,
        )

        info_box(
            [
                rf"$W_{{80}}/T = {w80:.3f}$",
                rf"$0.8\,V_{{max}} = {thr:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)
    elif metric == "N_t_over_T":
        m0 = float(support["m0"])
        nt_over_t = float(support["N_t_over_T"])

        p = sig / (m0 + EPS)

        ax.plot(tau, p, linewidth=3, color=main_color)
        ax.fill_between(
            tau,
            0,
            p,
            where=np.isfinite(p),
            color=fill_color2,
            interpolate=True,
        )

        info_box([rf"$N_t/T = {nt_over_t:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$p(t)\: (a.u.)$", fontsize=14, labelpad=10)

    else:
        info_box(f"No illustration for {metric}")


def export_selected_metric_pngs_bandlimited(
    all_results, zip_path, out_dir, format="png", show_right_panel=True
):
    os.makedirs(out_dir, exist_ok=True)

    with extracted_zip_tree(zip_path) as extracted_root:
        h5_index = build_grouped_h5_index(extracted_root)

        for vessel in VALID_VESSELS:
            if "bandlimited" not in all_results:
                continue
            if vessel not in all_results["bandlimited"]:
                continue

            for metric in sorted(SELECTED_METRICS_PNG):
                metric_key = METRIC_ALIASES.get(metric, metric)

                if metric_key not in all_results["bandlimited"][vessel]:
                    continue

                df = pd.DataFrame(all_results["bandlimited"][vessel][metric_key]).copy()
                if df.empty:
                    continue

                groups = sorted(df["group"].dropna().unique().tolist())
                control_name = find_control_group_name(groups)
                if control_name in groups:
                    groups = [g for g in groups if g != control_name] + [control_name]

                x_pos = {g: i for i, g in enumerate(groups)}

                grp = df.groupby("group")["mean"]
                grp_mean = grp.mean()
                grp_std = grp.std()
                rep_file = select_representative_file_per_group(df, value_col="mean")

                if show_right_panel:
                    fig = plt.figure(figsize=(15, 6.2), dpi=200)
                else:
                    fig = plt.figure(figsize=(8, 6.2), dpi=200)

                if show_right_panel:
                    outer = gridspec.GridSpec(
                        1,
                        2,
                        width_ratios=[0.7, 1.0],
                        wspace=0.15,
                    )
                else:
                    outer = gridspec.GridSpec(
                        1,
                        1,
                    )

                fig.subplots_adjust(left=0.04, right=0.995, bottom=0.08, top=0.86)

                ax_header = fig.add_axes([0.04, 0.88, 0.955, 0.11])
                ax_header.axis("off")

                # ===== Gauche: scatter =====
                if show_right_panel:
                    ax_top = fig.add_subplot(outer[0, 0])
                else:
                    ax_top = fig.add_subplot(outer[0])

                if control_name in x_pos:
                    cx = x_pos[control_name]
                    ax_top.axvspan(cx - 0.5, cx + 0.5, color="#E0E0E0")

                rng = np.random.default_rng(0)
                shapes = ["D", "o", "s", "^", "v", "P", "X"]

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
                            fmt=shapes[i % len(shapes)],
                            capsize=5,
                            markersize=12,
                            linewidth=1.2,
                            markerfacecolor="none",
                            markeredgecolor="red",
                            markeredgewidth=3,
                            )
                        

                ax_top.set_title(
                    f"{LATEX_FORMULAS.get(metric, metric)} (bandlimited waveform, {vessel})",
                    fontsize=20,
                    pad=20,
                )
                ax_top.set_xticks([x_pos[g] for g in groups])
                ax_top.set_xticklabels(groups, rotation=0)
                ax_top.tick_params(axis="both", labelsize=16)
                ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
                ax_top.grid(True, axis="y")

                # ===== Droite: illustrations =====
                if show_right_panel:
                    right = gridspec.GridSpecFromSubplotSpec(
                        2,
                        2,
                        subplot_spec=outer[0, 1],
                        hspace=0.5,
                        wspace=0.28,
                    )

                    for i, g in enumerate(groups[:4]):
                        r = i // 2
                        c = i % 2
                        ax = fig.add_subplot(right[r, c])

                        chosen = rep_file.get(g, None)
                        path = h5_index.get(g, {}).get(chosen, None) if chosen else None

                        if metric == "D_phi":
                            if path and os.path.exists(path):
                                support = extract_graphics_support(
                                    path, vessel=vessel, mode="bandlimited"
                                )
                                if support:
                                    support_beat = select_support_beat(support, 0)
                                    plot_metric_illustration(
                                        ax, metric, support_beat, path, vessel
                                    )
                                    ax.set_title(f"{g}", fontsize=14)
                                else:
                                    ax.text(
                                        0.5,
                                        0.5,
                                        f"No support for {g} ({vessel})",
                                        ha="center",
                                        va="center",
                                    )
                                    ax.axis("off")
                            else:
                                ax.text(
                                    0.5,
                                    0.5,
                                    f"No representative file for {g}",
                                    ha="center",
                                    va="center",
                                )
                                ax.axis("off")

                        elif path and os.path.exists(path):
                            support = extract_graphics_support(
                                path,
                                vessel=vessel,
                                mode="bandlimited",
                            )

                            if support:
                                support_beat = select_support_beat(support, 0)

                                plot_metric_illustration(
                                    ax, metric, support_beat, path, vessel
                                )
                                ax.set_title(f"{g}", fontsize=14)

                                ymin, ymax = ax.get_ylim()
                                ax.set_ylim(np.minimum(0, ymin), ymax * 1.4)
                            else:
                                ax.text(
                                    0.5,
                                    0.5,
                                    f"No support for {g} ({vessel})",
                                    ha="center",
                                    va="center",
                                )
                                ax.axis("off")
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"No representative file for {g}",
                                ha="center",
                                va="center",
                            )
                            ax.axis("off")

                    for j in range(len(groups[:4]), 4):
                        r = j // 2
                        c = j % 2
                        ax_empty = fig.add_subplot(right[r, c])
                        ax_empty.axis("off")
                if format == "png":
                    png_path = os.path.join(
                        out_dir, f"{metric}_bandlimited_{vessel}.png"
                    )
                    fig.savefig(png_path, bbox_inches="tight")
                if format == "eps":
                    eps_path = os.path.join(
                        out_dir, f"{metric}_bandlimited_{vessel}.eps"
                    )
                    fig.savefig(eps_path, bbox_inches="tight")

                plt.close(fig)


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def extract_metrics(h5_path):
    """
    Retourne:
    results[mode][vessel][metric_name] = {
        "mean": ...,
        "std": ...,
        "latex_formula": ...
    }
    """
    results = defaultdict(lambda: defaultdict(dict))

    with h5py.File(h5_path, "r") as f:
        for vessel in VALID_VESSELS:
            metrics_root_path = find_first_existing_path(
                f, get_metrics_base_candidates(vessel)
            )

            if metrics_root_path is None or metrics_root_path not in f:
                continue

            metrics_root = f[metrics_root_path]

            for mode in metrics_root.keys():
                if mode not in VALID_METRIC_FOLDERS:
                    continue

                group = metrics_root[mode]

                for metric_name in group.keys():
                    dataset = group[metric_name]
                    data = np.array(dataset)

                    latex_formula = dataset.attrs.get("latex_formula", "")
                    results[mode][vessel][metric_name] = {
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

METRIC_GROUPS = {
    "Timing and displacement - distribution metrics": {
        "mu_t_over_T",
        "sigma_t_over_T",
        "gamma_t",
    },
    "Near peak crest witdh": {
        "W50_over_T",
        "W80_over_T",
    },
    "Excursion and pulsability metrics": {
        "PI",
        "RI",
    },
    "Displacement partitioning and cumulative - displacement geometry": {
        "R_VTI",
        "SF_VTI",
        "Delta_DTI",
        "t50_over_T",
        "s_t",
        "w_t",
        "s_d",
        "w_d",
    },
    "Temporal kinetics and persistence metrics": {
        "t_max_over_T",
        "t_min_over_T",
        "Delta_t_over_T",
        "t_up_over_T",
        "t_down_over_T",
        "crest_factor",
        "slope_rise_normalized",
        "slope_fall_normalized",
        "v_end_over_v_mean",
    },
    
    "Harmonic - domain organization metrics": {
        "E_low_over_E_total",
        "rho_h",
        "w_h",
        "N_h_over_H_minus_1",
        "eta_h",
        
        
    },
    
    "Derivative - energy metrics": {
        "E_slope",
        
    },
    "Temporal support and concentration metrics": {
        "N_t_over_T",
        "N_eff_over_T",
        
    },
}
def generate_html_gallery(image_dir, html_dir, html_name="metric_dashboard.html"):
    png_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
    )

    html = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Waveform Metrics Dashboard</title>",
        "    <script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
        "    <style>",
        "        .group-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }",
        "        .group-toggle { cursor: pointer; font-size: 14px; color: #666; width: 18px; text-align: center; }",
        "        .filter-group-content.collapsed { display: none; }",
        "        .image-thumbnail { width: 100%; border: 1px solid #cccccc; border-radius: 8px; cursor: pointer; outline: none; transition: transform 0.2s ease; }",
        "        .image-thumbnail:focus, .image-thumbnail:active { outline: none; border: 1px solid #cccccc; }",
        "        .image-thumbnail:hover { transform: scale(1.02); }",
        "        .image-modal { display: none; position: fixed; z-index: 9999; left: 0; top: 0; width: 100vw; height: 100vh; background-color: rgba(0,0,0,0.9); justify-content: center; align-items: center; padding: 20px; box-sizing: border-box; }",
        "        .image-modal.open { display: flex; }",
        "        .image-modal img { display: block; max-width: 90vw; max-height: 90vh; width: auto; height: auto; object-fit: contain; border-radius: 10px; background: white; }",
        "        .image-modal-close { position: absolute; top: 20px; right: 35px; color: white; font-size: 40px; font-weight: bold; cursor: pointer; }",
        "        .card img { cursor: zoom-in; transition: transform 0.2s ease; }",
        "        .card img:hover { transform: scale(1.02); }",
        "        .toolbar { background: white; border-radius: 12px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
        "        .toolbar-top { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 15px; }",
        "        .toolbar input[type='text'] { flex: 1; min-width: 250px; padding: 10px; border: 1px solid #ccc; border-radius: 8px; font-size: 14px; }",
        "        .toolbar button { padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; background: #e0e0e0; }",
        "        .toolbar button:hover { background: #d0d0d0; }",
        "        .filter-panel { border-top: 1px solid #ddd; padding-top: 15px; display: none; }",
        "        .filter-panel.open { display: block; }",
        "        .filter-grid { display: flex; flex-direction: column; gap: 12px; }",
        "        .filter-item { display: flex; align-items: center; gap: 8px; background: #fafafa; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }",
        "        .filter-item input { cursor: pointer; }",
        "        .filter-group-box { background: #f8f8f8; border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 12px; }",
        "        .filter-group-title { font-weight: bold; margin-bottom: 10px; font-size: 15px; color: #222; border-bottom: 1px solid #ddd; padding-bottom: 6px; }",
        "        .filter-group-content { display: flex; flex-direction: column; gap: 6px; }",
        "        .hidden { display: none !important; }",
        "        .group-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }",
        "        .group-header input { cursor: pointer; }",
        "        .group-header label { font-weight: bold; font-size: 15px; color: #222; cursor: pointer; }",
        "        .search-help { font-size: 13px; color: #666; margin-top: 8px; }",
        "        .search-help code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }",
        "        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }",
        "        h1 { text-align: center; }",
        "        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1600px; margin: 0 auto; }",
        "        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }", 
        "        .card { background: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
        "        .card h2 { font-size: 16px; margin-bottom: 10px; }",
        "        img { width: 100%; border-radius: 8px; border: 1px solid #ddd; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Waveform Shape Metrics Dashboard</h1>",
        "    <div class='toolbar'>",
        "        <div class='toolbar-top'>",
        "            <input type='text' id='searchBox' placeholder='Search for a metric...'>",
        "            <button onclick='toggleFilters()'>Show / Hide Filters</button>",
        "            <button onclick='selectAllFilters()'>Select All</button>",
        "            <button onclick='clearAllFilters()'>Clear All</button>",
        "            <button onclick='collapseAllGroups()'>Collapse All Groups</button>",
        "            <button onclick='expandAllGroups()'>Expand All Groups</button>",
        "            <button onclick='invertSelection()'>Invert Selection</button>",
        "            <label><input type='checkbox' id='showArtery' checked onchange='applyFilters()'> Artery</label>",
        "            <label><input type='checkbox' id='showVein' checked onchange='applyFilters()'> Vein</label>",
        "        </div>",
        
        "               <div class='search-help'>",
        "            Search by metric name : <code>RI</code>, <code>PI</code>, <code>mu_t_over_T</code>, <code>W50</code>, <code>rho_h</code>, <code>phi</code>, <code>slope</code>, etc.",
        "        </div>",
        "        <div id='filterPanel' class='filter-panel'>",
        "            <div class='filter-grid'>",
    ]

    seen_metrics = set()

    assigned_metrics = set()

    for metrics in METRIC_GROUPS.values():
        assigned_metrics.update(metrics)

    for group_name, group_metrics in METRIC_GROUPS.items():
        group_id = (
            group_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        html.extend([
            "        <div class='filter-group-box'>",
            "            <div class='group-header'>",
            f"                <span class='group-toggle' onclick=\"toggleCollapse('{group_id}', this)\">▼</span>",
            f"                <input type='checkbox' checked onchange=\"toggleGroup('{group_id}', this.checked)\">",
            f"                <label>{group_name}</label>",
            "            </div>",
            f"            <div class='filter-group-content' data-group='{group_id}'>",
        ])

        for png in sorted(png_files):
            title = os.path.splitext(png)[0]

            parts = title.split("_bandlimited_")
            metric_name = parts[0]
            vessel = parts[1] if len(parts) > 1 else "unknown"

            if metric_name not in group_metrics:
                continue

            filter_key = f"{metric_name}_{vessel}"

            if filter_key in seen_metrics:
                continue

            seen_metrics.add(filter_key)

            display_title = LATEX_FORMULAS.get(metric_name, metric_name)
            display_title = display_title.replace("$", "")
            filter_id = f"{metric_name}_{vessel}".replace("/", "_").replace(" ", "_").lower()

            html.extend([
                "        <div class='filter-item'>",
                f"            <input type='checkbox' class='metric-filter' id='filter_{filter_id}' value='{filter_key.lower()}' checked onchange='applyFilters()'>",
                f"            <label for='filter_{filter_id}'>\\({display_title}\\) ({vessel})</label>",
                "        </div>",
            ])
        html.extend([
        "            </div>",
        "        </div>",
        ])

    remaining_metrics = []

    for png in sorted(png_files):
        title = os.path.splitext(png)[0]

        parts = title.split("_bandlimited_")
        metric_name = parts[0]
        vessel = parts[1] if len(parts) > 1 else "unknown"

        if metric_name in assigned_metrics:
            continue

        filter_key = f"{metric_name}_{vessel}"

        if filter_key in seen_metrics:
            continue

        seen_metrics.add(filter_key)
        remaining_metrics.append((metric_name, vessel, filter_key))

    if remaining_metrics:
        html.extend([
            "        <div class='filter-group-box'>",
            "            <div class='group-header'>",
            "                <span class='group-toggle' onclick=\"toggleCollapse('other', this)\">▼</span>",
            "                <input type='checkbox' checked onchange=\"toggleGroup('other', this.checked)\">",
            "                <label>Other</label>",
            "            </div>",
            "            <div class='filter-group-content' data-group='other'>",
        ])

        for metric_name, vessel, filter_key in remaining_metrics:
            display_title = LATEX_FORMULAS.get(metric_name, metric_name)
            display_title = display_title.replace("$", "")

            filter_id = f"{metric_name}_{vessel}".replace("/", "_").replace(" ", "_").lower()

            html.extend([
                "        <div class='filter-item'>",
                f"            <input type='checkbox' class='metric-filter' id='filter_{filter_id}' value='{filter_key.lower()}' checked onchange='applyFilters()'>",
                f"            <label for='filter_{filter_id}'>\\({display_title}\\) ({vessel})</label>",
                "        </div>",
            ])
        html.extend([
        "            </div>",
        "        </div>",
        ])
        
    html.extend([
    "            </div>",
    "        </div>",
    "    <div class='grid' id='metricsGrid'>",
])

    for png in sorted(png_files):
        title = os.path.splitext(png)[0]

        parts = title.split("_bandlimited_")
        metric_name = parts[0]
        vessel = parts[1] if len(parts) > 1 else "unknown"

        display_title = LATEX_FORMULAS.get(metric_name, metric_name)
        display_title = display_title.replace("$", "")

        filter_key = f"{metric_name}_{vessel}".lower()

        search_text = f"{metric_name.lower()} {display_title.lower()} {vessel.lower()} {filter_key}"
        search_text = search_text.replace("\\", "").replace("{", "").replace("}", "")
        png_path = os.path.join(image_dir, png)

        with open(png_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        html.extend([
            f"        <div class='card metric-card' data-metric='{filter_key}' data-search='{search_text}' data-vessel='{vessel.lower()}'>",
            f"            <h2>\\({display_title}\\) - {vessel.capitalize()}</h2>",
            f"            <img class='image-thumbnail' src='data:image/png;base64,{encoded}' alt='{title}' onclick=\"openImageModal(this.src)\">",
            "        </div>",
        ])

    html.extend([
    "    </div>",
    "    <div id='imageModal' class='image-modal' onclick='closeImageModal()'>",
    "        <span class='image-modal-close'>&times;</span>",
    "        <img id='modalImage' src='' onclick='closeImageModal(); event.stopPropagation()'>",
    "    </div>",
    "    <script>",
    "        function toggleFilters() {",
    "            document.getElementById('filterPanel').classList.toggle('open');",
    "        }",
    "",
    "        function toggleCollapse(groupName, element) {",
    "            const container = document.querySelector(`.filter-group-content[data-group='${groupName}']`);",
    "            if (!container) return;",
    "",
    "            container.classList.toggle('collapsed');",
    "            element.textContent = container.classList.contains('collapsed') ? '▶' : '▼';",
    "        }",
    "",
    "        function openImageModal(src) {",
    "            document.getElementById('modalImage').src = src;",
    "            document.getElementById('imageModal').classList.add('open');",
    "        }",
    "",
    "        function closeImageModal() {",
    "            document.getElementById('imageModal').classList.remove('open');",
    "        }",
    "        function applyFilters() {",
    "            const search = document.getElementById('searchBox').value.toLowerCase();",
    "            const checked = Array.from(document.querySelectorAll('.metric-filter:checked')).map(cb => cb.value.toLowerCase());",
    "            const cards = document.querySelectorAll('.metric-card');",
    "            const showArtery = document.getElementById('showArtery').checked;",
    "            const showVein = document.getElementById('showVein').checked;",
    "",
    "            cards.forEach(card => {",
    "                const metric = card.dataset.metric.toLowerCase();",
    "                const searchText = card.dataset.search.toLowerCase();",
    "",
    "                const vessel = card.dataset.vessel.toLowerCase();",
    "",
    "                const visibleByCheckbox = checked.includes(metric);",
    "                const visibleBySearch = search === '' || searchText.includes(search);",
    "",
    "                const visibleByVessel =",
    "                    (vessel === 'artery' && showArtery) ||",
    "                    (vessel === 'vein' && showVein);",    "",
    "                if (visibleByCheckbox && visibleBySearch && visibleByVessel) {",
    "                    card.classList.remove('hidden');",
    "                } else {",
    "                    card.classList.add('hidden');",
    "                }",
    "            });",
    "        }",
    "",
    "        function toggleGroup(groupName, checked) {",
    "            const container = document.querySelector(`.filter-group-content[data-group='${groupName}']`);",
    "            if (!container) return;",
    "",
    "            container.querySelectorAll('.metric-filter').forEach(cb => {",
    "                cb.checked = checked;",
    "            });",
    "",
    "            applyFilters();",
    "        }",
    "        function collapseAllGroups() {",
    "            document.querySelectorAll('.filter-group-content').forEach(group => {",
    "                group.classList.add('collapsed');",
    "            });",
    "",
    "            document.querySelectorAll('.group-toggle').forEach(toggle => {",
    "                toggle.textContent = '▶';",
    "            });",
    "        }",
    "",
    "        function expandAllGroups() {",
    "            document.querySelectorAll('.filter-group-content').forEach(group => {",
    "                group.classList.remove('collapsed');",
    "            });",
    "",
    "            document.querySelectorAll('.group-toggle').forEach(toggle => {",
    "                toggle.textContent = '▼';",
    "            });",
    "        }",
    "",
    "        function invertSelection() {",
    "            document.querySelectorAll('.metric-filter').forEach(cb => {",
    "                cb.checked = !cb.checked;",
    "            });",
    "",
    "            document.querySelectorAll('.filter-group-content').forEach(group => {",
    "                const checkboxes = Array.from(group.querySelectorAll('.metric-filter'));",
    "                const groupCheckbox = group.parentElement.querySelector('.group-header input[type=\"checkbox\"]');",
    "",
    "                if (groupCheckbox) {",
    "                    groupCheckbox.checked = checkboxes.every(cb => cb.checked);",
    "                }",
    "            });",
    "",
    "            applyFilters();",
    "        }",
    "        function selectAllFilters() {",
    "            document.querySelectorAll('.metric-filter').forEach(cb => cb.checked = true);",
    "            document.querySelectorAll('.group-header input[type=\"checkbox\"]').forEach(cb => cb.checked = true);",
    "            applyFilters();",
    "        }",
    "",
    "        function clearAllFilters() {",
    "            document.querySelectorAll('.metric-filter').forEach(cb => cb.checked = false);",
    "            document.querySelectorAll('.group-header input[type=\"checkbox\"]').forEach(cb => cb.checked = false);",
    "            applyFilters();",
    "        }",
    "",
    "        document.getElementById('searchBox').addEventListener('input', applyFilters);",
    "        window.addEventListener('load', applyFilters);",
    "    </script>",
    "</body>",
    "</html>",
])
    html_path = os.path.join(html_dir, html_name)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def add_html_to_zip(zip_path: str, html_path: str, arc_name: str = "waveform_metrics_dashboard.html"):
    import zipfile
    import os

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(html_path, arc_name)


def analyze_zip(zip_path):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    detected_groups = set()

    for grouped_file in iter_grouped_h5_files_in_zip(zip_path):
        detected_groups.add(grouped_file.group_name)
        metrics = extract_metrics(grouped_file.file_path)

        for mode, vessel_dict in metrics.items():
            for vessel, metric_dict in vessel_dict.items():
                for metric_name, values in metric_dict.items():
                    all_results[mode][vessel][metric_name].append(
                        {
                            "file": grouped_file.file_name,
                            "group": grouped_file.group_name,
                            "mean": values["mean"],
                            "std": values["std"],
                            "latex_formula": values.get("latex_formula", ""),
                            "vessel": vessel,
                        }
                    )

    single_group = len(detected_groups) <= 1
    return dict(all_results), single_group


def save_dashboard(all_results, zip_path, single_group):
    # -----------------------------
    # export PNGs
    # -----------------------------
    png_dir = "export_png"
    eps_dir = "export_eps"

    reset_output_dir(png_dir)
    reset_output_dir(eps_dir)

    # --- Windkessel ---
    export_windkessel_figures(zip_path, png_dir, format="png")

    eps_supported = _run_optional_eps_export(
        lambda: export_windkessel_figures(zip_path, eps_dir, format="eps"),
        eps_dir,
    )

    # --- Metrics illustrations ---
    export_selected_metric_pngs_bandlimited(
        all_results,
        zip_path,
        png_dir,
        "png",
        show_right_panel=False,
    )

    replace_folder_in_zip(zip_path, png_dir, arc_folder="export_png")

    
    

    # --- EPS ---
    if eps_supported:
        eps_supported = _run_optional_eps_export(
            lambda: export_selected_metric_pngs_bandlimited(
                all_results,
                zip_path,
                eps_dir,
                "eps",
                show_right_panel=False,
            ),
            eps_dir,
        )

    if eps_supported:
        replace_folder_in_zip(zip_path, eps_dir, arc_folder="export_eps")


    #HTML 
    temp_html_dir = tempfile.mkdtemp()
    generate_html_gallery(
        image_dir=png_dir,
        html_dir=temp_html_dir,
        html_name="waveform_metrics_dashboard.html",
    )

    html_path = os.path.join(
        temp_html_dir,
        "waveform_metrics_dashboard.html"
    )

    add_html_to_zip(
        zip_path,
        html_path,
        arc_name="waveform_metrics_dashboard.html",
    )

    if os.path.isdir(temp_html_dir):
            shutil.rmtree(temp_html_dir)
    
    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)

    if os.path.isdir(eps_dir):
        shutil.rmtree(eps_dir)

if __name__ == "__main__":
    zip_path = choose_zip()

    results, single_group = analyze_zip(zip_path)

    save_dashboard(results, zip_path, single_group)
