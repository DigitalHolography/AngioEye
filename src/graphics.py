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
METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/bandlimited"


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

    metrics = {}

    with h5py.File(h5_path, "r") as f:
        group = f[METRIC_FOLDER]

        for metric_name in group.keys():
            data = np.array(group[metric_name])

            metrics[metric_name] = {"mean": np.mean(data), "std": np.std(data)}

    return metrics


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

                try:
                    metrics = extract_metrics(filepath)

                    for metric, values in metrics.items():
                        if metric not in all_results:
                            all_results[metric] = []

                        all_results[metric].append(
                            {
                                "file": file,
                                "index": extract_index(file),
                                "mean": values["mean"],
                                "std": values["std"],
                            }
                        )

                except Exception as e:
                    print("Erreur :", file, e)

    return all_results


# ======================
# Figure HTML par métrique
# ======================
def save_dashboard(all_results, original_zip):

    dashboard_file = "metric_dashboard.html"

    # ==========================
    # Création HTML
    # ==========================
    with open(dashboard_file, "w") as f:
        f.write("""
<html>
<head>
<title>Metrics Dashboard</title>
</head>
<body>
<h1>Metrics Analysis Bandlimited</h1>
""")

    # ==========================
    # Ajouter une figure par métrique
    # ==========================
    for metric, data in all_results.items():
        df = pd.DataFrame(data)
        df = df.sort_values("index")

        fig = go.Figure()
        xmin = df["index"].min()
        xmax = df["index"].max()

        current = xmin
        toggle = True

        while current <= xmax:
            if toggle:
                fig.add_vrect(
                    x0=current - 0.5,
                    x1=current + 0.5,
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
            xaxis_title="Experiments",
            yaxis_title=metric,
            showlegend=False,
        )

        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        with open(dashboard_file, "a") as f:
            f.write(f"<h2>{metric}</h2>")
            f.write(fig_html)

    with open(dashboard_file, "a") as f:
        f.write("</body></html>")

    # ==========================
    # Création nouveau ZIP
    # ==========================
    new_zip = original_zip.replace(".zip", "_graphics.zip")

    # copier ZIP original
    shutil.copy(original_zip, new_zip)

    # ajouter dashboard dedans
    with zipfile.ZipFile(new_zip, "a") as z:
        z.write(dashboard_file)

    print(f"Nouveau ZIP créé : {new_zip}")


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    zip_path = choose_zip()

    results = analyze_zip(zip_path)

    save_dashboard(results, zip_path)
