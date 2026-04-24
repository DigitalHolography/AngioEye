import os
import tempfile
import zipfile
from collections import defaultdict
import shutil
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tkinter import Tk, filedialog
import html
import base64

PIPELINE_ROOT = "/AngioEye/Processing/waveform_shape_metrics"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]
VALID_VESSELS = ["artery", "vein"]

SELECTED_METRICS = {
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

def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])

def build_metrics_table_for_file(metrics_dict):
    rows = []

    for metric in sorted(SELECTED_METRICS):
        metric_key = METRIC_ALIASES.get(metric, metric)

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

def dataframe_to_html_table(
    df,
    title="Metrics Table",
    M_0_path=None,
    mask_vein_path=None,
    mask_artery_path=None,
    f_AVG_mean_path=None,
    artery_velocity_signal_path=None,
    vein_velocity_signal_path=None

):
    html_parts = []

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
    </style>

    <script>
    function openImageModal(src) {
        const modal = document.getElementById("image-modal");
        const modalImg = document.getElementById("image-modal-content");

        modal.style.display = "flex";
        modalImg.src = src;
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
    <div style="
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: repeat(2, auto);
        gap: 20px;
        margin-bottom: 30px;
        align-items: start;
    ">
    """)

    if M_0_path is not None:
        html_parts.append(f"""
        <div>
            <h2>\\(M_0\\)</h2>
            <img
                src="{M_0_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)

    if mask_artery_path is not None:
        html_parts.append(f"""
        <div>
            <h2>Mask Artery</h2>
            <img
                src="{mask_artery_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >        </div>
        """)

    if artery_velocity_signal_path is not None:
            html_parts.append(f"""
            <div>
                <h2>Artery Velocity Signal</h2>
            <img
                src="{artery_velocity_signal_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
           </div>
            """)

    if f_AVG_mean_path is not None:
            html_parts.append(f"""
            <div>
                <h2>f AVG mean</h2>
            <img
                src="{f_AVG_mean_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >                              
            </div>
            """)

    if mask_vein_path is not None:
        html_parts.append(f"""
        <div>
            <h2>Mask Vein</h2>
            <img
                src="{mask_vein_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)

    if vein_velocity_signal_path is not None:
        html_parts.append(f"""
        <div>
            <h2>Vein Velocity Signal</h2>
            <img
                src="{vein_velocity_signal_path}"
                class="image-thumbnail"
                onclick="openImageModal(this.src)"
            >
        </div>
        """)

    html_parts.append("</div>")

    html_parts.append("""
    <table>
        <thead>
            <tr>
                <th>Metric</th>
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
    </body>
    </html>
    """)

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

def generate_metric_tables_html(zip_path, output_dir="html_metric_tables"):
    reset_output_dir(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
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

                extracted = extract_metrics(filepath)

                merged_metrics = defaultdict(dict)

                for vessel in extracted.get("bandlimited", {}):
                    for metric_name, values in extracted["bandlimited"][vessel].items():
                        merged_metrics[vessel][metric_name] = values

                df = build_metrics_table_for_file(merged_metrics)

                base_name = os.path.splitext(file)[0]
                html_path = os.path.join(output_subdir, f"{base_name}.html")

        

                with h5py.File(filepath, "r") as f:
                    png_dir_name = f"{base_name}_png"
                    png_dir = os.path.join(output_subdir, png_dir_name)
                    os.makedirs(png_dir, exist_ok=True)

                    M_0_rel_path = None
                    mask_rel_path_vein = None
                    mask_rel_path_artery = None
                    f_AVG_mean_rel_path = None
                    artery_velocity_signal_path = None
                    vein_velocity_signal_path = None

                    # M_0
                    if "/Maps/M0_ff_img/value" in f:
                        M_0_data = np.array(f["/Maps/M0_ff_img/value"])

                        M_0_filename = f"{base_name}_M_0.png"
                        M_0_path = os.path.join(png_dir, M_0_filename)

                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(M_0_data.T, cmap="viridis")
                        ax.axis("off")
                        fig.savefig(M_0_path, bbox_inches="tight")
                        plt.close(fig)

                        M_0_rel_path = image_file_to_base64(M_0_path)

                    # Vein mask
                    if "/Vein/Segmentation/Mask/value" in f:
                        mask_vein_data = np.array(f["/Vein/Segmentation/Mask/value"])

                        mask_vein_filename = f"{base_name}_vein_mask.png"
                        mask_vein_path = os.path.join(png_dir, mask_vein_filename)

                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(mask_vein_data.T, cmap="gray")
                        ax.axis("off")
                        fig.savefig(mask_vein_path, bbox_inches="tight")
                        plt.close(fig)

                        mask_rel_path_vein = image_file_to_base64(mask_vein_path)

                    # Artery mask
                    if "/Artery/Segmentation/Mask/value" in f:
                        mask_artery_data = np.array(f["/Artery/Segmentation/Mask/value"])

                        mask_artery_filename = f"{base_name}_artery_mask.png"
                        mask_artery_path = os.path.join(png_dir, mask_artery_filename)

                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(mask_artery_data.T, cmap="gray")
                        ax.axis("off")
                        fig.savefig(mask_artery_path, bbox_inches="tight")
                        plt.close(fig)

                        mask_rel_path_artery = image_file_to_base64(mask_artery_path)

                    # f_AVG_mean map
                    if "/Maps/f_AVG_mean/value" in f:
                        f_AVG_mean_data = np.array(f["/Maps/f_AVG_mean/value"])

                        f_AVG_mean_filename = f"{base_name}_f_AVG_mean.png"
                        f_AVG_mean_path = os.path.join(png_dir, f_AVG_mean_filename)

                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(f_AVG_mean_data.T, cmap="viridis")
                        ax.axis("off")
                        fig.savefig(f_AVG_mean_path, bbox_inches="tight")
                        plt.close(fig)

                        f_AVG_mean_rel_path = image_file_to_base64(f_AVG_mean_path)

                    # Artery velocity signal
                    if "/Artery/Velocity/VelocitySignal/value" in f:
                        artery_velocity_signal_data = np.array(
                            f["/Artery/Velocity/VelocitySignal/value"]
                        )

                        artery_velocity_signal_filename = (
                            f"{base_name}_artery_velocity_signal.png"
                        )
                        artery_velocity_signal_png_path = os.path.join(
                            png_dir,
                            artery_velocity_signal_filename
                        )

                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(artery_velocity_signal_data, linewidth=2,color="#EC5241")
                        ax.set_title("Artery Velocity Signal")
                        ax.set_xlabel("Sample")
                        ax.set_ylabel("Velocity")
                        ax.grid(True)

                        fig.savefig(artery_velocity_signal_png_path, bbox_inches="tight")
                        plt.close(fig)

                        artery_velocity_signal_path = image_file_to_base64(
                            artery_velocity_signal_png_path
                        )
                    
                    # Vein velocity signal
                    if "/Vein/Velocity/VelocitySignal/value" in f:
                        vein_velocity_signal_data = np.array(
                            f["/Vein/Velocity/VelocitySignal/value"]
                        )

                        vein_velocity_signal_filename = (
                            f"{base_name}_vein_velocity_signal.png"
                        )
                        vein_velocity_signal_png_path = os.path.join(
                            png_dir,
                            vein_velocity_signal_filename
                        )

                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(vein_velocity_signal_data, linewidth=2, color="#414CEC")
                        ax.set_title("Vein Velocity Signal")
                        ax.set_xlabel("Sample")
                        ax.set_ylabel("Velocity")
                        ax.grid(True)

                        fig.savefig(vein_velocity_signal_png_path, bbox_inches="tight")
                        plt.close(fig)

                        vein_velocity_signal_path = image_file_to_base64(
                            vein_velocity_signal_png_path
                        )

                html_content = dataframe_to_html_table(
                    df,
                    title=f"Metrics for {file}",
                    M_0_path=M_0_rel_path,
                    mask_vein_path=mask_rel_path_vein,
                    mask_artery_path=mask_rel_path_artery,
                    f_AVG_mean_path=f_AVG_mean_rel_path,
                    artery_velocity_signal_path=artery_velocity_signal_path,
                    vein_velocity_signal_path=vein_velocity_signal_path
                )

                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                

def save_dashboard(zip_path, export_png_dir="export_png", export_eps_dir="export_eps"):
    generate_metric_tables_html(
    zip_path,
    output_dir="html_metric_tables"
    )

    replace_folder_in_zip(
        zip_path,
        "html_metric_tables",
        arc_folder="html_metric_tables"
    )

    if os.path.isdir("html_metric_tables"):
        shutil.rmtree("html_metric_tables")
    
    
    
if __name__ == "__main__":
    zip_path = choose_zip()
    
    save_dashboard(zip_path)
