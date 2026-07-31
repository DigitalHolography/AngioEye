import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

import base64
import html
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt

from input_output.archive_io import (
    reset_output_dir,
)
from input_output.hdf5_io import find_eyeflow_dataset, find_first_existing_path
from input_output.hdf5_schema import find_pipeline_group, pipeline_path_candidates
from input_output.output_paths import (
    PNG_OUTPUT_DIRNAME,
    dataset_stem_from_path,
    find_companion_file,
)
from math_utils import nanmedian, nanstd

WAVEFORM_SHAPE_METRICS_PIPELINE = "waveform_shape_metrics"
TOPOLOGICAL_METRICS_PIPELINE = "topological_metrics"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
VALID_VESSELS = ["artery", "vein"]
VALID_VESSEL_TYPES = ("artery", "vein")
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


def get_metrics_base_candidates(
    vessel: str,
    *parts: str,
) -> list[str]:
    return pipeline_path_candidates(
        WAVEFORM_SHAPE_METRICS_PIPELINE,
        vessel,
        *parts,
    )


def extract_group_metrics(group, results_dict, prefix=""):

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

                results_dict[full_name] = {
                    "median": nanmedian(data),
                    "std": nanstd(data),
                }

            except (ValueError, TypeError):
                print(f"Skipping non numeric dataset: {full_name}")


def _iter_metric_mode_groups(group):
    """Yield raw/bandlimited groups at any depth below a metric region."""
    for name, item in group.items():
        if not isinstance(item, h5py.Group):
            continue
        if name in VALID_METRIC_FOLDERS:
            yield name, item
            continue
        yield from _iter_metric_mode_groups(item)


def _extract_metric_modes(group):
    global_group = group.get("global")
    if isinstance(global_group, h5py.Group):
        return _extract_metric_modes(global_group)

    modes = defaultdict(dict)
    for mode, mode_group in _iter_metric_mode_groups(group):
        extract_group_metrics(mode_group, modes[mode])
    return modes


def extract_topological_metrics(h5_path):

    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
        )
    )

    with h5py.File(h5_path, "r") as f:
        pipeline_group = find_pipeline_group(f, TOPOLOGICAL_METRICS_PIPELINE)
        if pipeline_group is None:
            return results

        for vessel in VALID_VESSELS:
            vessel_group = pipeline_group.get(vessel)
            if not isinstance(vessel_group, h5py.Group):
                continue

            for zone in vessel_group:
                zone_group = vessel_group.get(zone)
                if not isinstance(zone_group, h5py.Group):
                    continue
                bandlimited_group = zone_group.get("global/bandlimited")
                if not isinstance(bandlimited_group, h5py.Group):
                    continue

                extract_group_metrics(
                    bandlimited_group,
                    results["bandlimited"][zone][vessel]
                )

    return results


def extract_waveform_shape_metrics(h5_path):

    results = defaultdict(lambda: defaultdict(dict))

    with h5py.File(h5_path, "r") as f:

        for vessel in VALID_VESSELS:

            metrics_root_path = find_first_existing_path(
                f,
                get_metrics_base_candidates(vessel, "global"),
            )

            if metrics_root_path is not None and metrics_root_path in f:
                metrics_root = f[metrics_root_path]
                for mode, mode_group in _iter_metric_mode_groups(metrics_root):
                    extract_group_metrics(mode_group, results[mode][vessel])

            hemifield_root_path = find_first_existing_path(
                f,
                get_metrics_base_candidates(vessel, "hemifield"),
            )
            if hemifield_root_path is None or hemifield_root_path not in f:
                continue

            hemifield_root = f[hemifield_root_path]
            direct_modes = {
                mode: mode_group
                for mode, mode_group in hemifield_root.items()
                if mode in VALID_METRIC_FOLDERS
                and isinstance(mode_group, h5py.Group)
            }
            if direct_modes:
                for mode, mode_group in direct_modes.items():
                    metrics = {}
                    extract_group_metrics(mode_group, metrics)
                    results["hemifield"].setdefault("all", {}).setdefault(
                        mode,
                        {},
                    )[vessel] = metrics
                continue

            for region_name, region_group in hemifield_root.items():
                if not isinstance(region_group, h5py.Group):
                    continue
                for mode, metrics in _extract_metric_modes(region_group).items():
                    results["hemifield"].setdefault(region_name, {}).setdefault(
                        mode,
                        {},
                    )[vessel] = metrics

    return results

def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])

def build_metrics_table_for_file(metrics_dict):
    rows = []

    for metric in sorted(SELECTED_METRICS):
        metric_key = metric

        artery_values = metrics_dict.get("artery", {}).get(metric_key, {})
        vein_values = metrics_dict.get("vein", {}).get(metric_key, {})

        artery_median = artery_values.get("median", np.nan)
        artery_std = artery_values.get("std", np.nan)
        vein_median = vein_values.get("median", np.nan)
        vein_std = vein_values.get("std", np.nan)

        latex_metric = LATEX_FORMULAS.get(metric_key, metric_key)

        if latex_metric.startswith("$") and latex_metric.endswith("$"):
            latex_metric = latex_metric[1:-1]

        display_metric = f"\\({latex_metric}\\)"

        rows.append(
            {
                "metric": display_metric,
                "artery_median": artery_median,
                "artery_std": artery_std,
                "vein_median": vein_median,
                "vein_std": vein_std,
            }
        )

    return pd.DataFrame(rows)

def _append_metrics_table(html_parts, df):
    html_parts.append("""
    <table>
        <thead>
            <tr>
                <th>Metric (mode : bandlimited)</th>
                <th class="artery-col">Median (Artery)</th>
                <th class="artery-col">Std (Artery)</th>
                <th class="vein-col">Median (Vein)</th>
                <th class="vein-col">Std (Vein)</th>
            </tr>
        </thead>
        <tbody>
    """)

    for _, row in df.iterrows():
        html_parts.append(f"""
        <tr>
            <td>{row['metric']}</td>
            <td class="artery-col">{row['artery_median']:.6g}</td>
            <td class="artery-col">{row['artery_std']:.6g}</td>
            <td class="vein-col">{row['vein_median']:.6g}</td>
            <td class="vein-col">{row['vein_std']:.6g}</td>
        </tr>
        """)

    html_parts.append("""
        </tbody>
    </table>
    """)

def _find_topology_branch_label_map(h5file, vessel_type):
    source_dataset = find_eyeflow_dataset(
        h5file,
        f"/Segmentation/{vessel_type.title()}/BranchLabelMap/value",
    )
    if isinstance(source_dataset, h5py.Dataset):
        return source_dataset

    pipeline_group = find_pipeline_group(h5file, TOPOLOGICAL_METRICS_PIPELINE)
    if pipeline_group is None:
        return None
    dataset = pipeline_group.get(f"topology/{vessel_type}/branch_label_map")
    return dataset if isinstance(dataset, h5py.Dataset) else None


def _branch_label_map_to_base64(
    source_path,
    *,
    vessel_type,
    image_dir,
    base_name,
    fallback_path=None,
):
    data = None
    for candidate in (source_path, fallback_path):
        if (
            candidate is None
            or not Path(candidate).exists()
            or not h5py.is_hdf5(candidate)
        ):
            continue
        with h5py.File(candidate, "r") as f:
            dataset = _find_topology_branch_label_map(f, vessel_type)
            if dataset is not None:
                data = np.array(dataset)
                break
    if data is None:
        return None

    image_path = os.path.join(
        image_dir,
        f"{base_name}_{vessel_type}_branch_label_map.png",
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.flip(data.T,axis=0),cmap="viridis")  
    ax.axis("off")
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)

    return image_file_to_base64(image_path)
    
def dataframe_to_html_table(
    waveform_tables,
    topological_tables,
    hemifield_tables=None,
    title="Metrics Table",
    M_0_path=None,
    mask_vein_path=None,
    mask_artery_path=None,
    f_AVG_mean_path=None,
    artery_velocity_signal_path=None,
    vein_velocity_signal_path=None,
    artery_branch_label_map_path=None,
    vein_branch_label_map_path=None,
):
    html_parts = []
    hemifield_tables = hemifield_tables or []

    html_parts.append("""
    <html>
    <head>
        <meta charset='utf-8'>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {
            font-family: Arial, sans-serif;
            font-family: Arial, sans-serif;
                margin: 30px;
                background-color: #f8f8f8;
            }
            h1 {
                color: #222;
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 30px;
            }
            h2 {
                font-weight: normal;
            }

            h2 mjx-container {
                font-weight: normal !important;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                background-color: white;
            }
            th, td {
            border: 1px solid #cccccc;
            padding: 8px 12px;
            text-align: center;
        }

        .artery-col {
            background-color: white;
            color: black;
            font-weight: bold;
        }

        .vein-col {
            background-color: white;
            color: black ;
            font-weight: bold;
        }
            th {
                background-color: #eaeaea;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f4f4f4;
            }
        </style>
    </head>
    <body>
    """)

    html_parts.append(f"<h1>{html.escape(title)}</h1>")

    html_parts.append("""
    <style>
    .image-thumbnail {
        width: 100%;
        max-width: 900px;
        border: 1px solid #cccccc;
        border-radius: 8px;
        cursor: pointer;
        outline: none;
    }
    .image-thumbnail:focus,
    .image-thumbnail:active {
        outline: none;
        border: 1px solid #cccccc;
    }

    .image-thumbnail:hover {
        transform: scale(1.02);
    }

    .image-container {
        text-align: center;
    }

    .image-thumbnail {
        width: 100%;
        max-width: 900px;
        border: 1px solid #cccccc;
        border-radius: 8px;
        cursor: pointer;
        outline: none;
        display: inline-block;
    }

    .image-title {
        text-align: center;
        font-weight: normal;
        margin-bottom: 10px;
    }
    .pipeline-title {
    font-size: 30px;   
    font-weight: bold;
    text-decoration: underline;
    margin-top: 30px;
    margin-bottom: 20px;
    }

    .zone-selector {
    text-align: center;
    margin: 25px 0;
    }

    .zone-selector label {
        font-size: 40px;
        font-weight: bold;
        margin-right: 12px;
    }

    .zone-selector select {
        font-size: 38px;
        font-weight: bold;
        padding: 8px 16px;
        min-width: 180px;
    }
    .image-modal {
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0,0,0,0.9);
        justify-content: center;
        align-items: center;
    }

    .image-modal img {
        display: block;
        margin: auto;
        width: 80vw;
        height: auto;
        max-height: 90vh;
        object-fit: contain;
    }

    .image-modal-close {
        position: absolute;
        top: 20px;
        right: 35px;
        color: white;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
    }

    .vessel-image-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
        margin-bottom: 30px;
        align-items: start;
    }
    </style>

    <script>
    function openImageModal(src) {
        const modal = document.getElementById("image-modal");
        const modalImg = document.getElementById("image-modal-content");

        modal.style.display = "flex";
        modalImg.src = src;
    }

    function showZone(zone) {
        document.querySelectorAll(".zone-table").forEach(div => {
            div.style.display = "none";
        });

        document.getElementById(zone).style.display = "block";
    }

    function showHemifield(region) {
        document.querySelectorAll(".hemifield-table").forEach(div => {
            div.style.display = "none";
        });

        document.getElementById(region).style.display = "block";
    }

    function closeImageModal() {
        document.getElementById("image-modal").style.display = "none";
    }
    </script>

    <div id="image-modal" class="image-modal" onclick="closeImageModal()">
        <img id="image-modal-content">
    </div>
    """)

    html_parts.append("""
    <div class="vessel-image-grid">
    """)

    if mask_artery_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Artery Segmentation</h2>
            <img
                src="{mask_artery_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    if mask_vein_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Vein Segmentation</h2>
            <img
                src="{mask_vein_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    if artery_velocity_signal_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Artery Velocity Signal</h2>
            <img
                src="{artery_velocity_signal_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    if vein_velocity_signal_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Vein Velocity Signal</h2>
            <img
                src="{vein_velocity_signal_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    if artery_branch_label_map_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Artery Branch Label Map</h2>
            <img
                src="{artery_branch_label_map_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    if vein_branch_label_map_path is not None:
        html_parts.append(f"""
        <div class="image-container">
            <h2 class="image-title">Vein Branch Label Map</h2>
            <img
                src="{vein_branch_label_map_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)
    else:
        html_parts.append("<div></div>")

    html_parts.append("</div>")

    if waveform_tables:

        html_parts.append('<h2 class="pipeline-title">Waveform Shape Metrics</h2>')
        for _, df in waveform_tables:
                _append_metrics_table(html_parts, df)
                html_parts.append("<br><br>")

    if hemifield_tables:
        html_parts.append('<h2 class="pipeline-title">Hemifield Analysis</h2>')
        html_parts.append("""
        <label for="hemifield-select"><b>Region :</b></label>
        <select id="hemifield-select" onchange="showHemifield(this.value)">
        """)

        for index, (region, _) in enumerate(hemifield_tables):
            display_region = str(region).replace("_", " ").title()
            selected = " selected" if index == 0 else ""
            html_parts.append(
                f'<option value="hemifield-{index}"{selected}>'
                f"{html.escape(display_region)}</option>"
            )

        html_parts.append("""
        </select>
        <br><br>
        """)

        for index, (region, df) in enumerate(hemifield_tables):
            display = "block" if index == 0 else "none"
            html_parts.append(
                f'<div id="hemifield-{index}" class="hemifield-table" '
                f'style="display:{display};">'
            )
            html_parts.append(
                f"<h3>{html.escape(str(region).replace('_', ' ').title())}</h3>"
            )
            _append_metrics_table(html_parts, df)
            html_parts.append("<br><br>")
            html_parts.append("</div>")
         

    if topological_tables:
        
        html_parts.append('<h2 class="pipeline-title">Topological Metrics</h2>')
        html_parts.append("""
        <label for="zone-select"><b>Zone :</b></label>
        <select id="zone-select" onchange="showZone(this.value)">
        """)

        for i, (zone, _) in enumerate(topological_tables):
            display_zone = zone.replace("_", " ").title()
            selected = " selected" if i == 0 else ""
            html_parts.append(
                f'<option value="{zone}"{selected}>{display_zone}</option>'
            )

        html_parts.append("""
        </select>
        <br><br>
        """)

        for i, (zone, df) in enumerate(topological_tables):

            display = "block" if i == 0 else "none"

            html_parts.append(
                f'<div id="{zone}" class="zone-table" style="display:{display};">'
            )


            _append_metrics_table(html_parts, df)

            html_parts.append("<br>")
            html_parts.append("</div>")

    return "".join(html_parts)

def replace_folder_in_zip(zip_path: str, folder_path: str, arc_folder: str):
    
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

def image_file_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_vessel_type(vessel_type):
    vessel = vessel_type.strip().lower()
    if vessel not in VALID_VESSEL_TYPES:
        raise ValueError(
            f"Unknown vessel type: {vessel_type!r}. "
            f"Expected one of {', '.join(VALID_VESSEL_TYPES)}."
        )
    return vessel


def _stem_for_source_path(path, stem=None):
    if stem is not None:
        return stem
    path_obj = Path(path)
    for parent in (path_obj.parent, *path_obj.parents):
        if parent.name.endswith("_EF") and len(parent.name) > len("_EF"):
            return parent.name.removesuffix("_EF")
    try:
        return dataset_stem_from_path(path)
    except ValueError:
        return path_obj.stem


def velocity_signal_png_filename(*, stem, vessel_type):
    vessel = normalize_vessel_type(vessel_type)
    return f"{stem}_RI_v_{vessel}.png"


def segmentation_map_png_filename(*, stem, vessel_type):
    vessel = normalize_vessel_type(vessel_type)
    return f"{stem}_{vessel}_seg_map_bkg.png"


def find_velocity_signal_png(path, *, vessel_type, stem=None):
    filename = velocity_signal_png_filename(
        stem=_stem_for_source_path(path, stem),
        vessel_type=vessel_type,
    )
    return find_companion_file(
        path,
        app_suffix="EF",
        query_type=PNG_OUTPUT_DIRNAME,
        filename=filename,
    )


def find_segmentation_map_png(path, *, vessel_type, stem=None):
    filename = segmentation_map_png_filename(
        stem=_stem_for_source_path(path, stem),
        vessel_type=vessel_type,
    )
    return find_companion_file(
        path,
        app_suffix="EF",
        query_type=PNG_OUTPUT_DIRNAME,
        filename=filename,
    )


def _velocity_signal_image_to_base64(source_path, *, vessel_type):
    if source_path is None:
        return None
    try:
        image_path = find_velocity_signal_png(
            source_path,
            vessel_type=vessel_type,
        )
    except ValueError:
        return None
    if image_path is None:
        return None
    return image_file_to_base64(image_path)


def _segmentation_map_image_to_base64(source_path, *, vessel_type):
    if source_path is None:
        return None
    try:
        image_path = find_segmentation_map_png(
            source_path,
            vessel_type=vessel_type,
        )
    except ValueError:
        return None
    if image_path is None:
        return None
    return image_file_to_base64(image_path)


def _array_image_to_base64(data, image_dir, filename, *, cmap):
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, filename)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(data.T, cmap=cmap)
    ax.axis("off")
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    return image_file_to_base64(image_path)


def _signal_image_to_base64(data, image_dir, filename, *, color, title):
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, filename)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data, linewidth=2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Velocity")
    ax.grid(True)
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    return image_file_to_base64(image_path)


def _build_single_file_html(filepath, *, image_dir, source_path=None):
    filepath = Path(filepath)
    source_h5_path = filepath
    if source_path is not None:
        candidate = Path(source_path)
        if candidate.suffix.lower() in {".h5", ".hdf5"} and candidate.exists():
            source_h5_path = candidate
    waveform = extract_waveform_shape_metrics(filepath)
    topological = extract_topological_metrics(filepath)
    
    waveform_tables = []
    topological_tables = []
    hemifield_tables = []


    # Waveform Shape Metrics
    if waveform and "bandlimited" in waveform:
        waveform_tables.append(
            (
                "Waveform Shape Metrics",
                build_metrics_table_for_file(
                    waveform["bandlimited"]
                ),
            )
        )

    hemifield = waveform.get("hemifield", {}) if waveform else {}
    for region, region_modes in sorted(hemifield.items()):
        mode = "bandlimited" if "bandlimited" in region_modes else "raw"
        if mode not in region_modes:
            continue
        hemifield_tables.append(
            (
                region,
                build_metrics_table_for_file(region_modes[mode]),
            )
        )

    # Topological Metrics
    if topological and "bandlimited" in topological:
        for zone, zone_metrics in sorted(topological["bandlimited"].items()):
            df = build_metrics_table_for_file(zone_metrics)

            topological_tables.append((zone, df))

    if not waveform_tables and not hemifield_tables and not topological_tables:
        raise ValueError(
            "No compatible pipeline metrics were found for the dashboard."
        )

    base_name = filepath.stem

    artery_branch_label_map_path = _branch_label_map_to_base64(
        source_h5_path,
        vessel_type="artery",
        image_dir=image_dir,
        base_name=base_name,
        fallback_path=filepath,
    )

    vein_branch_label_map_path = _branch_label_map_to_base64(
        source_h5_path,
        vessel_type="vein",
        image_dir=image_dir,
        base_name=base_name,
        fallback_path=filepath,
    )

    M_0_rel_path = None
    mask_rel_path_vein = _segmentation_map_image_to_base64(
        source_path,
        vessel_type="vein",
    )
    mask_rel_path_artery = _segmentation_map_image_to_base64(
        source_path,
        vessel_type="artery",
    )
    f_AVG_mean_rel_path = None
    artery_velocity_signal_path = _velocity_signal_image_to_base64(
        source_path,
        vessel_type="artery",
    )
    vein_velocity_signal_path = _velocity_signal_image_to_base64(
        source_path,
        vessel_type="vein",
    )

    with h5py.File(filepath, "r") as f:
        M_0_dataset = find_eyeflow_dataset(f, "Maps/M0_ff_img/value")
        if M_0_rel_path is None and M_0_dataset is not None:
            M_0_rel_path = _array_image_to_base64(
                np.array(M_0_dataset),
                image_dir,
                f"{base_name}_M_0.png",
                cmap="viridis",
            )

        vein_mask = find_eyeflow_dataset(f, "/Segmentation/Vein/Mask/value")
        if mask_rel_path_vein is None and isinstance(vein_mask, h5py.Dataset):
            mask_rel_path_vein = _array_image_to_base64(
                np.array(vein_mask),
                image_dir,
                f"{base_name}_vein_mask.png",
                cmap="gray",
            )

        artery_mask = find_eyeflow_dataset(f, "/Segmentation/Artery/Mask/value")
        if mask_rel_path_artery is None and isinstance(artery_mask, h5py.Dataset):
            mask_rel_path_artery = _array_image_to_base64(
                np.array(artery_mask),
                image_dir,
                f"{base_name}_artery_mask.png",
                cmap="gray",
            )

        frms_map = find_eyeflow_dataset(
            f,
            "/Processing/FrequencyMaps/fRMS_avg/value",
        )
        if f_AVG_mean_rel_path is None and isinstance(frms_map, h5py.Dataset):
            f_AVG_mean_rel_path = _array_image_to_base64(
                np.array(frms_map),
                image_dir,
                f"{base_name}_f_AVG_mean.png",
                cmap="viridis",
            )

        artery_velocity = find_eyeflow_dataset(
            f,
            "/Processing/Velocity/Artery/Raw/value",
        )
        if artery_velocity_signal_path is None and isinstance(
            artery_velocity, h5py.Dataset
        ):
            artery_velocity_signal_path = _signal_image_to_base64(
                np.array(artery_velocity),
                image_dir,
                f"{base_name}_artery_velocity_signal.png",
                color="#EC5241",
                title="Artery Velocity Signal",
            )

        vein_velocity = find_eyeflow_dataset(
            f,
            "/Processing/Velocity/Vein/Raw/value",
        )
        if vein_velocity_signal_path is None and isinstance(
            vein_velocity, h5py.Dataset
        ):
            vein_velocity_signal_path = _signal_image_to_base64(
                np.array(vein_velocity),
                image_dir,
                f"{base_name}_vein_velocity_signal.png",
                color="#414CEC",
                title="Vein Velocity Signal",
            )

    return dataframe_to_html_table(
        waveform_tables=waveform_tables,
        topological_tables=topological_tables,
        hemifield_tables=hemifield_tables,
        title=f"Metrics for {filepath.name}",
        M_0_path=M_0_rel_path,
        mask_vein_path=mask_rel_path_vein,
        mask_artery_path=mask_rel_path_artery,
        f_AVG_mean_path=f_AVG_mean_rel_path,
        artery_velocity_signal_path=artery_velocity_signal_path,
        vein_velocity_signal_path=vein_velocity_signal_path,
        artery_branch_label_map_path=artery_branch_label_map_path,
        vein_branch_label_map_path=vein_branch_label_map_path,
    )


def generate_metric_table_html_for_file(filepath, html_path, *, source_path=None):
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as image_dir:
        html_content = _build_single_file_html(
            filepath,
            image_dir=image_dir,
            source_path=source_path or filepath,
        )
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def generate_metric_tables_html(zip_path, output_dir="html"):
    reset_output_dir(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = sorted(f for f in files if f.endswith(".h5"))
            if not h5_files:
                continue

            relative_root = os.path.relpath(root, tmpdir)
            if relative_root == ".":
                relative_root = ""

            output_subdir = os.path.join(output_dir, relative_root)
            os.makedirs(output_subdir, exist_ok=True)

            for file in h5_files:
                filepath = os.path.join(root, file)
                html_path = os.path.join(
                    output_subdir,
                    f"{Path(file).stem}.html",
                )

                generate_metric_table_html_for_file(
                    filepath,
                    html_path,
                    source_path=filepath,
                )

                

def save_dashboard(
    zip_path,
    output_dir="HTML summary",
):

    generate_metric_tables_html(
        zip_path,
        output_dir=output_dir,
    )

    replace_folder_in_zip(
        zip_path,
        output_dir,
        arc_folder="HTML summary",
    )

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    
    
    
if __name__ == "__main__":
    zip_path = choose_zip()
    
    save_dashboard(zip_path)

