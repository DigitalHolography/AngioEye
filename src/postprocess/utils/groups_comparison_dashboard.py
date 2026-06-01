import os
import shutil
from collections import defaultdict
from tkinter import Tk, filedialog
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from input_output.hdf5_io import find_first_existing_path, read_array
from input_output.hdf5_schema import pipeline_path_candidates
from input_output.archive_io import (
    extracted_zip_tree,
    reset_output_dir,
    replace_folder_in_zip,
)
from ..core.grouped_batch import (
    build_group_order,
    build_grouped_h5_index,
    find_control_group_name,
    iter_grouped_h5_files_in_zip,
)

WAVEFORM_SHAPE_METRICS_PIPELINE = "waveform_shape_metrics"
VALID_MODE = ["raw", "bandlimited"]
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


SELECTED_METRICS = {
    "mu_t_over_T",
    "RI",
    "PI",
    "R_VTI",
    "SF_VTI",
    "sigma_t_over_T",
    "W50_over_T",
    "W80_over_T",
    "E_LF_over_E_HF",
    "t_max_over_T",
    "t_min_over_T",
    "S_rise",
    "S_fall",
    "t_rise_over_T",
    "t_fall_over_T",
    "CF",
    "Delta_DTI",
    "gamma_t",
    "N_eff_over_T",
    "N_t_over_T",
    "Q_t_skew",
    "Q_t_width",
    "Q_d_skew",
    "Q_d_width",
    "v_end_over_vbar",
    "E_slope",
    "t50_over_T",
    "eta_h",
}

EPS = 1e-12
LATEX_FORMULAS = {
    "RI": r"$\rm RI$",
    "CF": r"$\rm CF$",
    "t50_over_T": r"$t_{50}/T$",
    "R_VTI": r"$R_{\mathrm{VTI}}$",    
    "mu_t_over_T": r"$\mu_t/T$",
    "PI": r"$\rm PI$",
    "SF_VTI": r"$SF_{\mathrm{VTI}}$",
    "sigma_t_over_T": r"$\sigma_t/T$",    
    "t_max_over_T": r"$t_{\mathrm{max}}/T$",
    "t_min_over_T": r"$t_{\mathrm{min}}/T$",   
    "t_rise_over_T": r"$t_{\mathrm{rise}}/T$",
    "t_fall_over_T": r"$t_{\mathrm{fall}}/T$",    
    "Delta_DTI": r"$\Delta_{\mathrm{DTI}}$",
    "E_LF_over_E_HF": r"$E_{\mathrm{LF}}/E_{\mathrm{HF}}$",
    "S_fall": r"$S_{\mathrm{fall}}$",
    "S_rise": r"$S_{\mathrm{rise}}$",
    "gamma_t": r"$\gamma_t$",    
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",    
    "Q_t_skew": r"$Q_{\mathrm{t,skew}}$",
    "Q_t_width": r"$Q_{\mathrm{t,width}}$",
    "Q_d_skew": r"$Q_{\mathrm{d,skew}}$",
    "Q_d_width": r"$Q_{\mathrm{d,width}}$",
    "v_end_over_vbar": r"$\bar{\mathrm{v}}_{\mathrm{end}}/\bar{\mathrm{v}}$",
    "E_slope": r"$E_{\mathrm{slope}}$",   
    "W50_over_T": r"$W_{50}/T$",
    "W80_over_T": r"$W_{80}/T$",
    "N_t_over_T": r"$N_t/T$",    
    "eta_h": r"$\eta_h$",
}


def extract_graphics_support(h5_path, vessel="artery", mode="bandlimited"):

    if mode == "all":
        modes_to_load = VALID_MODE
    else:
        modes_to_load = [mode]

    out = {}

    with h5py.File(h5_path, "r") as f:

        for current_mode in modes_to_load:

            base_candidates = get_mode_path_candidates(vessel, current_mode)
            base = find_first_existing_path(f, base_candidates)

            if base is None or base not in f:
                continue

            grp = f[base]

            mode_dict = {}

            for key in grp.keys():
                arr = np.array(grp[key])
                mode_dict[key] = arr.item() if arr.shape == () else arr

            out[current_mode] = mode_dict

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
        print(f"Aucune donnÃ©e pour metric={metric}, method={method}")
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

    stats = sub.groupby("group")["value"].agg(["median", "std"]).reset_index()

    for _, row in stats.iterrows():
        group = row["group"]
        median_val = row["median"]
        std_val = row["std"] if np.isfinite(row["std"]) else 0.0

        ax.errorbar(
            x_pos[group],
            median_val,
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
        print("Aucune donnÃ©e Windkessel trouvÃ©e dans le zip.")
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

    if format == "png":
        csv_path = os.path.join(out_dir, "windkessel_values.csv")
        df.to_csv(csv_path, index=False)



def select_support_beat(support, beat_idx):
    out = {}
    for k, v in support.items():
        arr = np.asarray(v)
        if arr.ndim == 2:
            if k in {
                "harmonic_magnitudes", 
                "harmonic_energies",
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
    fill_color1 = "#f9c2ca" if vessel == "artery" else "#A1B2F2"
    fill_color2 = "#F2CCC7" if vessel == "artery" else "#BDDBE7"
    if not support:
        ax.text(0.5, 0.5, "No graphics support", ha="center", va="center")
        ax.axis("off")
        return

    tau = np.asarray(support["tau"], dtype=float)
    sig = np.asarray(support["signal_mean"], dtype=float)
    C = np.asarray(support.get("cumulative", []), dtype=float)
    vb = np.asarray(support.get("vb", []), dtype=float)
    dvdt = np.asarray(support.get("dvdt", []), dtype=float)
    harmonic_magnitudes = np.asarray(
        support.get("harmonic_magnitudes", []), dtype=float
    )
    harmonic_energies = np.asarray(support.get("harmonic_energies", []), dtype=float)

    H_LOW_MAX = int(np.asarray(support.get("H_LOW_MAX", 3)).item())

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

    n = sig.size
    if n < 2:
        info_box("Signal too short")
        return

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

    elif metric == "eta_h":
        eta_h = float(support.get("eta_h", np.nan))
        if not np.isfinite(eta_h):
            # fallback si le support ne donne pas directement la mÃ©trique
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

        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=2)

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


    elif metric == "t_rise_over_T":
        t_rise = float(support["t_rise_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_rise, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{rise}}/T = {t_rise:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "t_fall_over_T":
        t_fall = float(support["t_fall_over_T"])

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            t_fall, tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1
        )
        info_box([rf"$t_{{fall}}/T = {t_fall:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)


    elif metric == "S_rise":
        s_rise = float(support["S_rise"])
        idx = int(np.nanargmax(dvdt))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{rise}}={s_rise:.3f}$"])
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "S_fall":
        s_fall = float(support["S_fall"])
        idx = int(np.nanargmin(dvdt))

        ax.plot(tau, sig, linewidth=3, color=main_color)
        vline_to_curve(
            tau[idx], tau, sig, y0=0.0, color="black", linestyles="--", linewidth=1.0
        )
        info_box([rf"$S_{{fall}}={s_fall:.3f}$"])
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

    
    elif metric == "CF":
        cf = float(support["CF"])
        vmax = float(np.nanmax(vb))
        rms = float(np.sqrt(np.nanmean(vb**2)))
        vb_tau = np.linspace(0.0, 1.0, len(vb), endpoint=False)

        ax.plot(vb_tau, vb, linewidth=3, color=main_color)
        hline_label(vmax, "Vmax", va="bottom")
        hline_label(rms, "RMS", va="top")
        info_box([f"CF= {cf:.3f}"])
        ax.set_xlabel(r"rectified time :  t/T", fontsize=14)
        ax.set_ylabel(r"$v(t) \: (mm/s)$", fontsize=14, labelpad=12)

    elif metric == "E_LF_over_E_HF":
        mags2 = harmonic_energies[1:]
        e_lf_over_e_hf = float(support["E_LF_over_E_HF"])
        
        xh = np.arange(1, len(mags2) + 1)

        ax.set_yscale("log")
        ax.bar(xh[: H_LOW_MAX + 1], mags2[: H_LOW_MAX + 1], color=main_color)
        ax.bar(xh[H_LOW_MAX:], mags2[H_LOW_MAX:], color="#cccccc")
        lines = [
            
            rf"$E_{{LF}}/E_{{HF}} = {e_lf_over_e_hf:.3f}$",
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


    elif metric == "Q_t_skew":
        t10 = float(support["t10_over_T"])
        t50 = float(support["t50_over_T"])
        t90 = float(support["t90_over_T"])
        Q_t_skew = float(support["Q_t_skew"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq in [t10, t50, t90]:
            yq = _y_at(tq, tau, C)
            ax.vlines(tq, 0, yq, linestyle="--", linewidth=1, color="black")
            ax.hlines(yq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{t,skew}}={Q_t_skew:.3f}$",
                rf"$t_{{10}}/T={t10:.3f}, t_{{50}}/T={t50:.3f}, t_{{90}}/T={t90:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_t_width":
        t25 = float(support["t25_over_T"])
        t75 = float(support["t75_over_T"])
        Q_t_width = float(support["Q_t_width"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        y25 = _y_at(t25, tau, C)
        y75 = _y_at(t75, tau, C)
        ax.vlines(t25, 0, y25, linestyle="--", linewidth=1, color="black")
        ax.vlines(t75, 0, y75, linestyle="--", linewidth=1, color="black")
        ax.hlines(y25, 0, t25, linestyle="--", linewidth=1, color="black")
        ax.hlines(y75, 0, t75, linestyle="--", linewidth=1, color="black")
        ax.fill_between(tau, 0, C, where=(tau >= t25) & (tau <= t75), color=fill_color2)

        info_box(
            [rf"$Q_{{t,width}}={Q_t_width:.3f}$", rf"$t_{{25}}/T={t25:.3f}, t_{{75}}/T={t75:.3f}$"]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)


    elif metric == "Q_d_skew":
        d10 = float(support["d10_over_D"])
        d50 = float(support["d50_over_D"])
        d90 = float(support["d90_over_D"])
        Q_d_skew = float(support["Q_d_skew"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        for tq, dq in [(0.1, d10), (0.5, d50), (0.9, d90)]:
            ax.vlines(tq, 0, dq, linestyle="--", linewidth=1, color="black")
            ax.hlines(dq, 0, tq, linestyle="--", linewidth=1, color="black")

        info_box(
            [
                rf"$Q_{{d,skew}}={Q_d_skew:.3f}$",
                rf"$d_{{10}}/D={d10:.3f}, d_{{50}}/D={d50:.3f}$", 
                rf"$d_{{90}}/D={d90:.3f}$",
            ]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)

    elif metric == "Q_d_width":
        d25 = float(support["d25_over_D"])
        d75 = float(support["d75_over_D"])
        Q_d_width = float(support["Q_d_width"])

        ax.plot(tau, C, linewidth=3, color=main_color)
        ax.vlines(0.25, 0, d25, linestyle="--", linewidth=1, color="black")
        ax.vlines(0.75, 0, d75, linestyle="--", linewidth=1, color="black")
        ax.hlines(d25, 0, 0.25, linestyle="--", linewidth=1, color="black")
        ax.hlines(d75, 0, 0.75, linestyle="--", linewidth=1, color="black")

        y_fill = np.linspace(d25, d75, 300)
        x_curve = np.interp(y_fill, C, tau)
        ax.fill_betweenx(y_fill, 0, x_curve, color=fill_color2)

        info_box(
            [rf"$Q_{{d,width}}={Q_d_width:.3f}$", rf"$d_{{25}}/D={d25:.3f}, d_{{75}}/D={d75:.3f}$"]
        )
        ax.set_xlabel(r"rectified time : t/T", fontsize=14)
        ax.set_ylabel(r"$d(t) \: (a.u.)$", fontsize=14, labelpad=12)


    elif metric == "v_end_over_vbar":
        vmean = float(support["vmean"])
        vend = float(support["vend"])
        ratio = float(support["v_end_over_vbar"])
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
            rf" $\overline{{v}}_{{end}}={vend:.3g}$",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none"),
        )
        info_box([rf"$\bar{{v}}_{{end}}/\bar{{v}}={ratio:.3f}$"])
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


def export_selected_metric(
    all_results, zip_path, out_dir, format="png", show_group_illustrations=True, vessel="artery", mode="bandlimited"
):
    os.makedirs(out_dir, exist_ok=True)

    if mode == "all":
        modes_to_process = VALID_MODE
    else:
        modes_to_process = [mode]

    with extracted_zip_tree(zip_path) as extracted_root:
        h5_index = build_grouped_h5_index(extracted_root)

        for current_mode in modes_to_process:

            if current_mode not in all_results:
                continue

            for vessel in VALID_VESSELS:

                main_color = "#EC5241" if vessel == "artery" else "#414CEC"

                if vessel not in all_results[current_mode]:
                    continue

                for metric in sorted(SELECTED_METRICS):
                    metric_key = metric

                    if metric_key not in all_results[current_mode][vessel]:
                        continue

                    df = pd.DataFrame(all_results[current_mode][vessel][metric_key]).copy()
                    if df.empty:
                        continue

                    groups = sorted(df["group"].dropna().unique().tolist())
                    control_name = find_control_group_name(groups)
                    if control_name in groups:
                        groups = [g for g in groups if g != control_name] + [control_name]

                    x_pos = {g: i for i, g in enumerate(groups)}

                    grp = df.groupby("group")["median"]
                    grp_mean = grp.mean()
                    grp_std = grp.std()
                    rep_file = select_representative_file_per_group(df, value_col="median")

                    if show_group_illustrations:
                        fig = plt.figure(figsize=(15, 6.2), dpi=200)
                    else:
                        fig = plt.figure(figsize=(8, 6.2), dpi=200)

                    if show_group_illustrations:
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
                    if show_group_illustrations:
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
                            gdf["median"].values,
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
                                markersize=15,
                                linewidth=1.2,
                                markerfacecolor="none",
                                markeredgecolor=main_color,
                                markeredgewidth=3,
                                )
                            

                    ax_top.set_title(
                        f"{LATEX_FORMULAS.get(metric, metric)} ({current_mode} waveform, {vessel})",
                        fontsize=20,
                        pad=20,
                    )
                    ax_top.set_xticks([x_pos[g] for g in groups])
                    ax_top.set_xticklabels(groups, rotation=0)
                    ax_top.tick_params(axis="both", labelsize=16)
                    ax_top.set_xlim(-0.5, len(groups) - 0.5)
                    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
                    ax_top.grid(True, axis="y")

                    # ===== Droite: illustrations =====
                    if show_group_illustrations:
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

                            if path and os.path.exists(path):
                                support = extract_graphics_support(
                                    path,
                                    vessel=vessel,
                                    mode=current_mode,
                                )

                                if support:
                                    support_mode = support[current_mode]
                                    support_beat = select_support_beat(support_mode, 0)

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
                            out_dir, f"{metric}_{current_mode}_{vessel}.png"
                        )
                        fig.savefig(png_path, bbox_inches="tight")
                    if format == "eps":
                        eps_path = os.path.join(
                            out_dir, f"{metric}_{current_mode}_{vessel}.eps"
                        )
                        fig.savefig(eps_path, bbox_inches="tight")

                    plt.close(fig)


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def extract_group_metrics(group, results_dict, prefix=""):
    """
    Parcourt rÃ©cursivement les groupes/datasets HDF5
    """

    for metric_name in group.keys():

        item = group[metric_name]

        full_name = f"{prefix}/{metric_name}" if prefix else metric_name

        if isinstance(item, h5py.Group):

            extract_group_metrics(
                item,
                results_dict,
                prefix=full_name
            )

        elif isinstance(item, h5py.Dataset):

            try:
                data = np.array(item, dtype=float)

                latex_formula = item.attrs.get("latex_formula", "")

                results_dict[full_name] = {
                    "median": float(np.nanmedian(data)),
                    "std": float(np.nanstd(data)),
                    "latex_formula": latex_formula,
                }

            except (ValueError, TypeError):
                print(f"Skipping non numeric dataset: {full_name}")


def extract_metrics(h5_path):

    results = defaultdict(lambda: defaultdict(dict))

    with h5py.File(h5_path, "r") as f:

        for vessel in VALID_VESSELS:

            metrics_root_path = find_first_existing_path(
                f,
                get_metrics_base_candidates(vessel)
            )

            if metrics_root_path is None or metrics_root_path not in f:
                continue

            metrics_root = f[metrics_root_path]

            for mode in metrics_root.keys():

                if mode not in VALID_MODE:
                    continue

                group = metrics_root[mode]

                extract_group_metrics(
                    group,
                    results[mode][vessel]
                )

    return results


def select_representative_file_per_group(df_metric: pd.DataFrame, value_col="median"):
    """
    Renvoie un dict: {group -> filename} du patient le plus proche de la mÃ©diane du groupe.
    df_metric doit contenir au moins: ["group", "file", value_col]
    """
    rep = {}
    for g, gdf in df_metric.groupby("group"):
        vals = gdf[value_col].astype(float).values
        if len(vals) == 0 or not np.any(np.isfinite(vals)):
            continue
        med = float(np.nanmedian(vals))
        # index du patient le plus proche de la mÃ©diane
        idx = int(np.nanargmin(np.abs(vals - med)))
        rep[g] = gdf.iloc[idx]["file"]

    return rep


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
                            "median": values["median"],
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
    export_selected_metric(
        all_results,
        zip_path,
        png_dir,
        "png",
        show_group_illustrations=True,
        mode="bandlimited"
    )

    replace_folder_in_zip(zip_path, png_dir, arc_folder="export_png")

    # --- EPS ---
    if eps_supported:
        eps_supported = _run_optional_eps_export(
            lambda: export_selected_metric(
                all_results,
                zip_path,
                eps_dir,
                "eps",
                show_group_illustrations=True,
                mode="bandlimited"
            ),
            eps_dir,
        )

    if eps_supported:
        replace_folder_in_zip(zip_path, eps_dir, arc_folder="export_eps")

    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)

    if os.path.isdir(eps_dir):
        shutil.rmtree(eps_dir)


if __name__ == "__main__":
    zip_path = choose_zip()

    results, single_group = analyze_zip(zip_path)

    save_dashboard(results, zip_path, single_group)

