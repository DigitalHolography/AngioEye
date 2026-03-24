import os
import re
import tempfile
import zipfile
from collections import defaultdict
from tkinter import Tk, filedialog
import shutil
import h5py
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

GRAPHICS_SUPPORT_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
SEGMENT_METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/by_segment/"
SEGMENT_VALID_MODES = ["raw", "bandlimited"]
SEGMENT_METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/by_segment/"
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12
SEGMENT_SELECTED_METRICS = {
    "mu_t_over_T",
    "RI",
    "PI",
    "R_VTI",
    "SF_VTI",
    "sigma_t_over_T",
    "W50_over_T",
    "E_low_over_E_total",
    "E_high_over_E_total",
    "t_max_over_T",
    "t_min_over_T",
    "Delta_t_over_T",
    "slope_rise_normalized",
    "slope_fall_normalized",
    "t_up_over_T",
    "t_down_over_T",
    "S_decay",
    "crest_factor",
    "R_SD",
    "Delta_DTI",
    "gamma_t",
    "spectral_entropy",
    "delta_phi2",
    "rho_h_90",
    "mu_h",
    "sigma_h",
    "N_eff_over_T",
    "N_H_over_T",
    "phase_locking_residual",
    "E_recon_H_MAX",
    "Q_t_skew",
    "Q_t_width",
    "R_Q_t",
    "Q_d_skew",
    "Q_d_width",
    "R_Q_d",
    "v_end_over_v_mean",
    "E_slope",
    "E_curv",
}

HIGHER_LEVEL_METRICS = {
    "CV",
    "IQR",
    "MAD",
    "H_seg",
    "H_rad",
    "H_branch",
}

EPS = 1e-12
H_MAX = 10
H_LOW_MAX = 3
H_HIGH_MIN = 4
H_HIGH_MAX = 8


def extract_segment_metric(h5_path, metric_name, mode="bandlimited_segment"):
    dataset_path = f"{SEGMENT_METRIC_FOLDER}{mode}/{metric_name}"
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            return None
        arr = np.array(f[dataset_path], dtype=float)

    if arr.ndim != 3:
        return None

    return arr
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

def make_higher_metric_key(base_metric, higher_metric):
    return f"{higher_metric}__{base_metric}"


def make_higher_metric_label(base_metric, higher_metric):
    return f"{higher_metric}({base_metric})"
def compute_segment_higher_level_metrics(arr, eps=EPS):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # A. variability across beats for each segment
        seg_mean = np.nanmean(arr, axis=0)
        seg_std = np.nanstd(arr, axis=0, ddof=1)
        seg_cv = seg_std / (np.abs(seg_mean) + eps)

        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        seg_iqr = q75 - q25

        seg_med = np.nanmedian(arr, axis=0)
        seg_mad = np.nanmedian(np.abs(arr - seg_med[None, :, :]), axis=0)

        # central segment map used for spatial heterogeneity
        seg_value = np.nanmedian(arr, axis=0)

    # B. H_seg across all valid segments
    flat_seg = seg_value[np.isfinite(seg_value)]
    if flat_seg.size == 0:
        H_seg = np.nan
    elif flat_seg.size == 1:
        H_seg = 0.0
    else:
        H_seg = float(
            np.nanstd(flat_seg, ddof=1) / (np.abs(np.nanmean(flat_seg)) + eps)
        )

    # C. H_rad per branch, then median across branches
    n_branch = seg_value.shape[0]
    H_rad_per_branch = np.full(n_branch, np.nan, dtype=float)

    for j in range(n_branch):
        row = seg_value[j, :]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        elif row.size == 1:
            H_rad_per_branch[j] = 0.0
        else:
            H_rad_per_branch[j] = float(
                np.nanstd(row, ddof=1) / (np.abs(np.nanmean(row)) + eps)
            )

    H_rad = (
        float(np.nanmedian(H_rad_per_branch))
        if np.any(np.isfinite(H_rad_per_branch))
        else np.nan
    )

    # D. H_branch from branch summaries
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        branch_summary = np.nanmedian(seg_value, axis=1)

    valid_branch = branch_summary[np.isfinite(branch_summary)]

    if valid_branch.size == 0:
        H_branch = np.nan
    elif valid_branch.size == 1:
        H_branch = 0.0
    else:
        H_branch = float(
            np.nanstd(valid_branch, ddof=1) / (np.abs(np.nanmean(valid_branch)) + eps)
        )

    # global summaries for beat-to-beat metrics
    CV = float(np.nanmedian(seg_cv)) if np.any(np.isfinite(seg_cv)) else np.nan
    IQR = float(np.nanmedian(seg_iqr)) if np.any(np.isfinite(seg_iqr)) else np.nan
    MAD = float(np.nanmedian(seg_mad)) if np.any(np.isfinite(seg_mad)) else np.nan

    return {
        "seg_value": seg_value,
        "seg_cv": seg_cv,
        "seg_iqr": seg_iqr,
        "seg_mad": seg_mad,
        "CV": CV,
        "IQR": IQR,
        "MAD": MAD,
        "H_seg": H_seg,
        "H_rad": H_rad,
        "H_branch": H_branch,
        "H_rad_per_branch": H_rad_per_branch,
        "branch_summary": branch_summary,
    }
def analyze_zip(zip_path, mode=SEGMENT_MODE):
    results = defaultdict(list)
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
                h5_path = os.path.join(root, file)

                for base_metric in SEGMENT_SELECTED_METRICS:
                    arr = extract_segment_metric(h5_path, base_metric, mode=mode)
                    if arr is None:
                        continue

                    metric_dict = compute_segment_higher_level_metrics(arr)
                    if metric_dict is None:
                        continue

                    for high_metric in HIGHER_LEVEL_METRICS:
                        value = metric_dict.get(high_metric, np.nan)
                        metric_key = make_higher_metric_key(base_metric, high_metric)

                        results[metric_key].append(
                            {
                                "file": file,
                                "group": group_name,
                                "mean": value,
                                "std": 0.0,
                                "base_metric": base_metric,
                                "higher_metric": high_metric,
                                "metric_label": make_higher_metric_label(base_metric, high_metric),
                            }
                        )

    single_group = len(detected_groups) <= 1
    return dict(results), single_group
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
                "delta_phi_all",
            }:
    
                out[k] = arr[beat_idx,:]
                
            else:
                out[k] = arr[:, beat_idx]
        elif arr.ndim == 1 and arr.shape[0] > beat_idx:
            out[k] = arr[beat_idx]
        else:
            out[k] = v
   
    return out


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




def plot_higher_level_metric_illustration(ax, higher_metric, metric_dict):
    seg_value = metric_dict["seg_value"]
    seg_cv = metric_dict["seg_cv"]
    seg_iqr = metric_dict["seg_iqr"]
    seg_mad = metric_dict["seg_mad"]
    H_rad_per_branch = metric_dict["H_rad_per_branch"]
    branch_summary = metric_dict["branch_summary"]

    def info_box(lines, fontsize=11):
        text = "\n".join(lines)
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )

    if higher_metric == "CV":
        im = ax.imshow(seg_cv, aspect="auto", origin="lower")
        ax.set_title("Segment beat-to-beat CV")
        ax.set_xlabel("Disk")
        ax.set_ylabel("Branch")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        info_box([f"CV = {metric_dict['CV']:.3f}"])

    elif higher_metric == "IQR":
        im = ax.imshow(seg_iqr, aspect="auto", origin="lower")
        ax.set_title("Segment beat-to-beat IQR")
        ax.set_xlabel("Disk")
        ax.set_ylabel("Branch")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        info_box([f"IQR = {metric_dict['IQR']:.3f}"])

    elif higher_metric == "MAD":
        im = ax.imshow(seg_mad, aspect="auto", origin="lower")
        ax.set_title("Segment beat-to-beat MAD")
        ax.set_xlabel("Disk")
        ax.set_ylabel("Branch")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        info_box([f"MAD = {metric_dict['MAD']:.3f}"])

    elif higher_metric == "H_seg":
        im = ax.imshow(seg_value, aspect="auto", origin="lower")
        ax.set_title("Segment median map")
        ax.set_xlabel("Disk")
        ax.set_ylabel("Branch")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        info_box([f"H_seg = {metric_dict['H_seg']:.3f}"])

    elif higher_metric == "H_rad":
        x = np.arange(len(H_rad_per_branch))
        ax.bar(x, H_rad_per_branch, color="#EC5241", edgecolor="black")
        ax.set_title(r"Per-branch $H^{rad}$")
        ax.set_xlabel("Branch")
        ax.set_ylabel("H_rad")
        for i, v in enumerate(H_rad_per_branch):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        info_box([f"H_rad = {metric_dict['H_rad']:.3f}"])

    elif higher_metric == "H_branch":
        x = np.arange(len(branch_summary))
        ax.bar(x, branch_summary, color="black")
        ax.set_title("Branch summaries")
        ax.set_xlabel("Branch")
        ax.set_ylabel("Median across disks")
        info_box([f"H_branch = {metric_dict['H_branch']:.3f}"])

    else:
        ax.text(0.5, 0.5, f"No illustration for {higher_metric}", ha="center", va="center")
        ax.axis("off")

def export_segment_higher_level_pngs(results, zip_path, out_dir, mode=SEGMENT_MODE):
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        h5_index = build_h5_path_index_from_extracted_tree(tmpdir)

        for metric_key in sorted(results.keys()):
            df = pd.DataFrame(results[metric_key]).copy()
            if df.empty:
                continue

            base_metric = df["base_metric"].iloc[0]
            higher_metric = df["higher_metric"].iloc[0]
            metric_label = df["metric_label"].iloc[0]

            groups = sorted(df["group"].dropna().unique().tolist())
            control_name = find_control_group_name(groups)
            if control_name in groups:
                groups = [g for g in groups if g != control_name] + [control_name]

            x_pos = {g: i for i, g in enumerate(groups)}
            grp = df.groupby("group")["mean"]
            grp_mean = grp.mean()
            grp_std = grp.std()

            rep_file = select_representative_file_per_group(df, value_col="mean")

            fig = plt.figure(figsize=(15, 6.2), dpi=140)
            outer = gridspec.GridSpec(1, 2, width_ratios=[0.7, 1.0], wspace=0.15)
            fig.subplots_adjust(left=0.04, right=0.995, bottom=0.08, top=0.86)

            # left panel
            ax_top = fig.add_subplot(outer[0, 0])

            if control_name in x_pos:
                cx = x_pos[control_name]
                ax_top.axvspan(cx - 0.5, cx + 0.5, color="#E0E0E0")

            rng = np.random.default_rng(0)
            shapes = ["D", "o", "s", "^"]

            for i, g in enumerate(groups):
                gdf = df[df["group"] == g]
                x = np.full(len(gdf), x_pos[g], dtype=float) + rng.normal(0, 0.06, size=len(gdf))

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
                        markeredgecolor="black",
                        markeredgewidth=3,
                    )

            ax_top.set_title(metric_label, fontsize=20, pad=20)
            ax_top.set_xticks([x_pos[g] for g in groups])
            ax_top.set_xticklabels(groups, rotation=0)
            ax_top.tick_params(axis="both", labelsize=16)
            ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
            ax_top.grid(True, axis="y")

            # right panel
            right = gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=outer[0, 1], hspace=0.5, wspace=0.28
            )

            for i, g in enumerate(groups[:4]):
                r = i // 2
                c = i % 2
                ax = fig.add_subplot(right[r, c])

                chosen = rep_file.get(g, None)
                path = h5_index.get(g, {}).get(chosen, None) if chosen else None

                if path and os.path.exists(path):
                    arr = extract_segment_metric(path, base_metric, mode=mode)
                    if arr is not None:
                        metric_dict = compute_segment_higher_level_metrics(arr)
                        plot_higher_level_metric_illustration(ax, higher_metric, metric_dict)
                        ax.set_title(f"{g}", fontsize=14)
                    else:
                        ax.text(0.5, 0.5, f"No by_segment for {g}", ha="center", va="center")
                        ax.axis("off")
                else:
                    ax.text(0.5, 0.5, f"No representative file for {g}", ha="center", va="center")
                    ax.axis("off")

            for j in range(len(groups[:4]), 4):
                r = j // 2
                c = j % 2
                ax_empty = fig.add_subplot(right[r, c])
                ax_empty.axis("off")

            png_path = os.path.join(out_dir, f"{metric_key}_{mode}.png")
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)


def find_control_group_name(groups):
    # cherche "control", "controls", "ctrl" etc.
    for g in groups:
        if g is None:
            continue
        gl = str(g).lower()
        if "control" in gl or gl in {"ctrl", "ctl", "controls"}:
            return g
    return None


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




def save_dashboard(original_zip):
    results, single_group = analyze_zip(original_zip, mode=SEGMENT_MODE)

    segment_png_dir = os.path.join(os.path.dirname(original_zip), "export_segment_png")
    export_segment_higher_level_pngs(results, original_zip, segment_png_dir, mode=SEGMENT_MODE)

    replace_folder_in_zip(original_zip, segment_png_dir, arc_folder="export_segment_png")
    if os.path.isdir(segment_png_dir):
        shutil.rmtree(segment_png_dir)




if __name__ == "__main__":
    zip_path = choose_zip()
    save_dashboard(zip_path)
