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

GRAPHICS_SUPPORT_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
SELECTED_METRICS_PNG = {
}
EPS = 1e-12
H_MAX = 10
H_LOW_MAX = 3
H_HIGH_MIN = 4
H_HIGH_MAX = 8
LATEX_FORMULAS = {
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



def export_selected_metric_pngs_bandlimited(all_results, zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # index des chemins .h5 par groupe / fichier
        h5_index = build_h5_path_index_from_extracted_tree(tmpdir)
        for metric in sorted(SELECTED_METRICS_PNG):
            
            if metric not in all_results["bandlimited"]:
                continue

            df = pd.DataFrame(all_results["bandlimited"][metric]).copy()

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

def plot_group_delta_phi_stats(ax, group_stats, group_name):
    if group_name not in group_stats:
        ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center")
        ax.axis("off")
        return

    data = group_stats[group_name]
    hs = data["h"]
    mu = data["mean"]
    sigma = data["std"]
    ax.bar(
        hs,
        mu,
        width=0.7,
        color="#EC5241",
        edgecolor="black",
    )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(np.pi, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(-np.pi, color="black", linewidth=0.8, linestyle="--")

    for h, m, s in zip(hs, mu, sigma, strict=False):
        if not np.isfinite(m):
            continue

        va = "bottom" if m >= 0 else "top"
        offset = 0.08 if m >= 0 else -0.08

        ax.text(
            h,
            m + offset,
            f"{m:.2f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    ax.set_xlim(1.5, max(hs) + 0.5)
    ax.set_ylim(-1.1 * np.pi, 1.1 * np.pi)
    ax.set_xticks(hs)

    ax.set_xlabel("Harmonic n (a.u.)", fontsize=14)
    ax.set_ylabel(r"Mean $\delta\phi_n$ (rad)", fontsize=14, labelpad=12)

    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(
        [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],
        fontsize=12,
    )

    ax.set_title(group_name, fontsize=14)


def build_group_signal_figure(group_name, data):
    fig = go.Figure()

    x = np.asarray(data["x"], dtype=float)
    mean = np.asarray(data["mean"], dtype=float)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(width=3),
            name=group_name,
        )
    )

    y_max = np.nanmax(mean) if np.any(np.isfinite(mean)) else 1.0

    fig.update_yaxes(range=[0, y_max * 1.05])

    fig.update_layout(
        height=450,
        xaxis_title="Time",
        yaxis_title="Velocity",
        template="simple_white",
        showlegend=False,
    )

    return fig


def find_control_group_name(groups):
    # cherche "control", "controls", "ctrl" etc.
    for g in groups:
        if g is None:
            continue
        gl = str(g).lower()
        if "control" in gl or gl in {"ctrl", "ctl", "controls"}:
            return g
    return None


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
    harmonic_energies = np.asarray(support.get("harmonic_energies", []), dtype=float)
    harmonic_energies_weights = np.asarray(support.get("harmonic_energies_weights", []), dtype=float)
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
                marker=dict(
                    size=20,
                    color=color_map[g],  # intérieur creux
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


def compute_group_mean_signals(zip_path, mode="bandlimited"):
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

                support_mean = extract_mean_support_per_file(h5_path, mode=mode)
                if support_mean is None or "signal_mean" not in support_mean:
                    continue

                signal = np.asarray(support_mean["signal_mean"], dtype=float)
                if signal.ndim != 1 or signal.size == 0:
                    continue

                group_signals[group_name].append(signal)

    group_curves = {}

    for group, signals in group_signals.items():
        min_len = min(len(s) for s in signals)
        aligned = np.array([s[:min_len] for s in signals], dtype=float)

        group_mean = np.nanmean(aligned, axis=0)

        group_curves[group] = {
            "x": np.arange(min_len),
            "mean": group_mean,
        }

    return group_curves


def build_comparison_signal_figure(group_curves):
    fig = go.Figure()

    groups = sorted(group_curves.keys())
    if not groups:
        return fig

    max_len = max(len(group_curves[g]["x"]) for g in groups)
    x_common = np.arange(max_len)
    global_max = 0.0

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
        y_old = np.asarray(data["mean"], dtype=float)

        y_interp = np.interp(
            x_common,
            np.linspace(0, max_len - 1, len(y_old)),
            y_old,
        )

        if np.any(np.isfinite(y_interp)):
            global_max = max(global_max, float(np.nanmax(y_interp)))

        fig.add_trace(
            go.Scatter(
                x=x_common,
                y=y_interp,
                mode="lines",
                name=group,
                line=dict(color=color_map.get(group, "black"), width=3),
            )
        )

    if global_max <= 0:
        global_max = 1.0

    fig.update_yaxes(range=[0, global_max * 1.05])

    fig.update_layout(
        height=550,
        xaxis_title="Time index",
        yaxis_title="Signal amplitude",
        template="simple_white",
        legend_title="Group",
    )

    return fig


def save_dashboard(all_results, original_zip, single_group):

    all_metrics = set()
    for mode in all_results:
        all_metrics.update(all_results[mode].keys())

    group_curves = compute_group_mean_signals(original_zip, mode="raw")
    group_curves_bl = compute_group_mean_signals(original_zip, mode="bandlimited")

    group_comparison_curves = build_comparison_signal_figure(group_curves)
    group_comparison_curves_bl = build_comparison_signal_figure(group_curves_bl)

    png_dir = os.path.join(os.path.dirname(original_zip), "export_png")
    export_selected_metric_pngs_bandlimited(all_results, original_zip, png_dir)

    print("PNGs exportés dans :", png_dir)
    replace_folder_in_zip(original_zip, png_dir, arc_folder="export_png")
    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)




if __name__ == "__main__":
    zip_path = choose_zip()
    dataset_path_bl = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"

    results, single_group = analyze_zip(zip_path)
    save_dashboard(results, zip_path, single_group)
