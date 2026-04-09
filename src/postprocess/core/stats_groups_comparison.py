import os
import tempfile
import zipfile
from collections import defaultdict
import shutil
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from tkinter import Tk, filedialog

PIPELINE_ROOT = "/Pipelines/waveform_shape_metrics"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
VALID_VESSELS = ["artery", "vein"]

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
    "N_H_over_T",
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
    "N_H_over_T": r"$N_H/T$",
    "t_phi_n_over_T": r"$t_{\Delta\phi_n}/T$",
    "t_phi_over_T": r"$t_{\phi}/T$",
    "D_phi": r"$D_{\phi}$",
    "s_phi_over_T": r"$s_{\Delta\phi}/T$",
    "eta_h": r"$\eta_h$",
    "rho_h": r"$\rho_{h}$",
    "w_h": r"$w_{h}$",
    "N_h_over_H_minus_1": r"$N_{H}/(H-1)$",
}


def find_control_group_name(groups):
    for g in groups:
        if g is None:
            continue
        gl = str(g).lower()
        if "control" in gl or gl in {"ctrl", "ctl", "controls"}:
            return g
    return None


def get_metrics_base_path(vessel: str) -> str:
    return f"{PIPELINE_ROOT}/{vessel}/global"


def extract_metrics(h5_path):
    results = defaultdict(lambda: defaultdict(dict))

    with h5py.File(h5_path, "r") as f:
        for vessel in VALID_VESSELS:
            metrics_root_path = get_metrics_base_path(vessel)

            if metrics_root_path not in f:
                continue

            metrics_root = f[metrics_root_path]

            for mode in metrics_root.keys():
                if mode not in VALID_METRIC_FOLDERS:
                    continue

                group = metrics_root[mode]

                for metric_name in group.keys():
                    dataset = group[metric_name]
                    data = np.array(dataset, dtype=float)

                    results[mode][vessel][metric_name] = {
                        "mean": np.nanmedian(data),
                        "std": np.nanstd(data),
                    }

    return results


def analyze_zip(zip_path):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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

            for file in h5_files:
                filepath = os.path.join(root, file)
                metrics = extract_metrics(filepath)

                for mode, vessel_dict in metrics.items():
                    for vessel, metric_dict in vessel_dict.items():
                        for metric_name, values in metric_dict.items():
                            all_results[mode][vessel][metric_name].append(
                                {
                                    "file": file,
                                    "group": group_name,
                                    "mean": values["mean"],
                                    "std": values["std"],
                                    "vessel": vessel,
                                }
                            )

    return dict(all_results)


def reset_output_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def plot_group_statistics(df, metric, vessel, out_path):
    if df.empty:
        return

    groups = sorted(df["group"].dropna().unique().tolist())
    control_name = find_control_group_name(groups)
    if control_name in groups:
        groups = [g for g in groups if g != control_name] + [control_name]

    x_pos = {g: i for i, g in enumerate(groups)}

    grp_mean = df.groupby("group")["mean"].mean()
    grp_std = df.groupby("group")["mean"].std()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_facecolor("#f2f2f2")

    if control_name in x_pos:
        cx = x_pos[control_name]
        ax.axvspan(cx - 0.5, cx + 0.5, color="#E0E0E0", zorder=0)

    rng = np.random.default_rng(0)
    shapes = ["D", "o", "s", "^", "v", "P", "X"]

    for i, g in enumerate(groups):
        gdf = df[df["group"] == g]
        if gdf.empty:
            continue

        x = np.full(len(gdf), x_pos[g], dtype=float) + rng.normal(
            0, 0.06, size=len(gdf)
        )

        ax.scatter(
            x,
            gdf["mean"].values,
            color="black",
            s=20,
            edgecolors="none",
            zorder=2,
        )

        if g in grp_mean.index:
            ax.errorbar(
                [x_pos[g]],
                [grp_mean.loc[g]],
                yerr=[grp_std.loc[g] if pd.notna(grp_std.loc[g]) else 0.0],
                fmt=shapes[i % len(shapes)],
                color="black",
                ecolor="black",
                capsize=5,
                markersize=12,
                linewidth=1.2,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2.5,
                zorder=3,
            )

    ax.set_xticks([x_pos[g] for g in groups])
    ax.set_xticklabels(groups, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.grid(True, axis="y", color="gray", linewidth=0.8)

    title = LATEX_FORMULAS.get(metric, metric)
    ax.set_title(f"{title} ({vessel})", fontsize=18, pad=15)
    ax.set_ylabel(LATEX_FORMULAS[metric], fontsize=14)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_group_statistics_figures(all_results, out_dir, formats=("png", "eps")):
    os.makedirs(out_dir, exist_ok=True)

    if "bandlimited" not in all_results:
        print("Aucune donnée bandlimited trouvée.")
        return

    for vessel in VALID_VESSELS:
        if vessel not in all_results["bandlimited"]:
            continue

        for metric in sorted(SELECTED_METRICS_PNG):
            metric_key = METRIC_ALIASES.get(metric, metric)

            if metric_key not in all_results["bandlimited"][vessel]:
                continue

            df = pd.DataFrame(all_results["bandlimited"][vessel][metric_key]).copy()
            if df.empty:
                continue

            for fmt in formats:
                out_path = os.path.join(
                    out_dir, f"{metric_key}_bandlimited_{vessel}.{fmt}"
                )
                plot_group_statistics(df, metric_key, vessel, out_path)


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


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


def save_dashboard(zip_path, export_png_dir="export_png", export_eps_dir="export_eps"):
    all_results = analyze_zip(zip_path)
    reset_output_dir(export_png_dir)
    reset_output_dir(export_eps_dir)
    export_group_statistics_figures(
        all_results,
        out_dir=export_png_dir,
        formats=("png",),
    )

    export_group_statistics_figures(
        all_results,
        out_dir=export_eps_dir,
        formats=("eps",),
    )

    replace_folder_in_zip(zip_path, export_png_dir, arc_folder="export_png")
    replace_folder_in_zip(zip_path, export_eps_dir, arc_folder="export_eps")


if __name__ == "__main__":
    zip_path = choose_zip()

    save_dashboard(zip_path)
