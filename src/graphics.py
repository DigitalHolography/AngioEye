import zipfile
import tempfile
import os
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tkinter import Tk, filedialog
import shutil
import re

# ======================
# dossier contenant les métriques
# ======================
METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]


# ======================
# Choisir ZIP
# ======================
def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


# ======================
# Lire toutes les métriques d'un h5
# ======================
def extract_index(filename):
    """
    Trouve le numéro d'expérience dans le nom.
    Pas de numéro = 0.
    """

    match = re.search(r"PRESSURE_(\d+)", filename)

    if match:
        return int(match.group(1))

    # aucun numéro -> fichier 0
    return 0


def extract_metrics(h5_path):

    results = {}

    with h5py.File(h5_path, "r") as f:
        metrics_root = f[METRIC_FOLDER]

        for mode in metrics_root.keys():
            if mode not in VALID_METRIC_FOLDERS:
                continue  # ignore params ou autre dossier

            results[mode] = {}

            group = metrics_root[mode]

            for metric_name in group.keys():
                data = np.array(group[metric_name])

                results[mode][metric_name] = {
                    "mean": np.mean(data),
                    "std": np.std(data),
                }

    return results


# ======================
# Analyse du ZIP
# ======================
def analyze_zip(zip_path):

    all_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for file in files:
                if not file.endswith(".h5"):
                    continue

                filepath = os.path.join(root, file)

                metrics = extract_metrics(filepath)

                for mode, metrics_dict in metrics.items():
                    if mode not in all_results:
                        all_results[mode] = {}

                    for metric, values in metrics_dict.items():
                        if metric not in all_results[mode]:
                            all_results[mode][metric] = []

                        all_results[mode][metric].append(
                            {
                                "file": file,
                                "index": extract_index(file),
                                "mean": values["mean"],
                                "std": values["std"],
                            }
                        )

    return all_results


# ======================
# Figure HTML par métrique
# ======================
def save_dashboard(all_results, original_zip):

    dashboard_file = "metric_dashboard.html"

    with open(dashboard_file, "w") as f:
        f.write("""
<html>
<head>
<title>Metrics Dashboard</title>
</head>
<body>
<h1>Metrics Analysis</h1>
""")

    # ======================
    # POUR CHAQUE MODE
    # ======================
    for mode, metrics in all_results.items():
        with open(dashboard_file, "a") as f:
            f.write(f"<h1>{mode.upper()}</h1><hr>")

        for metric, data in metrics.items():
            df = pd.DataFrame(data).sort_values("index")

            fig = go.Figure()
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
            fig.add_trace(
                go.Scatter(
                    x=df["index"],
                    y=df["mean"],
                    error_y=dict(type="data", array=df["std"], visible=True),
                    mode="markers",
                )
            )

            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title=metric,
                showlegend=False,
            )

            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

            with open(dashboard_file, "a") as f:
                f.write(f"<h2>{metric}</h2>")
                f.write(fig_html)

    with open(dashboard_file, "a") as f:
        f.write("</body></html>")

    # ======================
    # nouveau zip
    # ======================
    new_zip = original_zip.replace(".zip", "_graphics.zip")

    shutil.copy(original_zip, new_zip)

    with zipfile.ZipFile(new_zip, "a") as z:
        z.write(dashboard_file)

    print("Dashboard créé :", new_zip)


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    zip_path = choose_zip()

    results = analyze_zip(zip_path)

    save_dashboard(results, zip_path)
