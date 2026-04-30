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
import base64

PIPELINE_ROOT = "/AngioEye/Processing/waveform_shape_metrics"
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
                        "median": np.nanmedian(data),
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
                                    "median": values["median"],
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

    grp_mean = df.groupby("group")["median"].mean()
    grp_std = df.groupby("group")["median"].std()

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
            gdf["median"].values,
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


def export_group_statistics_figures(all_results, out_dir, formats=("png")):
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

def add_file_to_zip(zip_path: str, file_path: str, arc_name: str):
    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            already_seen = set()

            for item in zin.infolist():
                normalized = item.filename.replace("\\", "/").strip("/")

                
                if normalized == arc_name.strip("/"):
                    continue

                
                if normalized in already_seen:
                    continue

                already_seen.add(normalized)
                zout.writestr(item, zin.read(item.filename))

            zout.write(file_path, arc_name)

    os.replace(temp_zip, zip_path)

def save_dashboard(zip_path, export_png_dir="export_png_html"):
    all_results = analyze_zip(zip_path)
    reset_output_dir(export_png_dir)

    export_group_statistics_figures(
        all_results,
        out_dir=export_png_dir,
        formats=("png",),
    )

    temp_html_dir = tempfile.mkdtemp()

    generate_html_gallery(
        image_dir=export_png_dir,
        html_dir=temp_html_dir,
        html_name="waveform_metrics_dashboard.html",
    )

    html_path = os.path.join(
        temp_html_dir,
        "waveform_metrics_dashboard.html"
    )

    replace_folder_in_zip(zip_path, export_png_dir, arc_folder="export_png_html")

    add_file_to_zip(
        zip_path,
        html_path,
        arc_name="waveform_metrics_dashboard.html",
    )

    if os.path.isdir(export_png_dir):
        shutil.rmtree(export_png_dir)
    
    if os.path.isdir(temp_html_dir):
        shutil.rmtree(temp_html_dir)
    
    

if __name__ == "__main__":
    zip_path = choose_zip()

    save_dashboard(zip_path)
