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
import base64
from pathlib import Path


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


def replace_file_in_zip(zip_path, file_to_add):

    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w") as zout:
            # copier tous les fichiers sauf l'ancien html
            for item in zin.infolist():
                if item.filename != os.path.basename(file_to_add):
                    buffer = zin.read(item.filename)
                    zout.writestr(item, buffer)

            # ajouter la nouvelle version
            zout.write(file_to_add)

    # remplacer ancien zip
    os.replace(temp_zip, zip_path)


# ======================
# Lire toutes les métriques d'un h5
# ======================


def extract_sort_key(filename):

    name = os.path.basename(filename)

    # 1️⃣ date = premier nombre à 6 chiffres
    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    # 2️⃣ index avant HD
    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return (date, hd_index)


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
            group_name = os.path.basename(root)

            for file in files:
                if not file.endswith(".h5"):
                    continue

                filepath = os.path.join(root, file)

                metrics = extract_metrics(filepath)
                for metric_group, group_values in metrics.items():
                    if "mean" in group_values:
                        metric_name = metric_group
                        mean = group_values["mean"]
                        std = group_values["std"]

                        all_results.setdefault(metric_name, []).append(
                            {
                                "file": file,
                                "group": group_name,
                                "mean": mean,
                                "std": std,
                            }
                        )

                    # cas nouveau : sous-dossier (bandlimited, raw, etc.)
                    else:
                        for metric_name, values in group_values.items():
                            mean = values["mean"]
                            std = values["std"]

                            all_results.setdefault(metric_name, []).append(
                                {
                                    "file": file,
                                    "group": group_name,
                                    "mean": mean,
                                    "std": std,
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
<style>
body {
    margin-left: 80px;
}
</style>
</head>
<body>
<h1>Metrics Analysis</h1>
""")

    # ======================
    # POUR CHAQUE MODE
    # ======================
    for mode, metrics in all_results.items():
        metrics_by_name = defaultdict(list)

        for entry in metrics:
            metric_name = entry["metric"]
            metrics_by_name[metric_name].append(entry)
        with open(dashboard_file, "a") as f:
            f.write(f"<h1>{mode.upper()}</h1><hr>")
        dashboard_file = "metric_dashboard.html"
        BASE_DIR = Path(__file__).parent
        image_path = BASE_DIR / "images.jpg"
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        html_img = f'<img src="data:image/jpeg;base64,{encoded}" width="300">'

        with open(dashboard_file, "a") as f:
            f.write(html_img)
        for metric, data in metrics.items():
            df = pd.DataFrame(data).sort_values("index")
            df["group_order"] = df["group"].astype("category").cat.codes
            df = df.sort_values(["group_order", "file"])

            df["index"] = range(len(df))

            fig = go.Figure()
            fig.update_layout(width=800, height=500, margin=dict(t=20, b=20))
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
                    marker=dict(color="black"),
                )
            )
            tickvals = []
            ticktext = []

            for g in df["group"].unique():
                group_indices = df[df["group"] == g]["index"]

                center = group_indices.mean()

                tickvals.append(center)
                ticktext.append(g)

            fig.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    title="Patient Group",
                ),
                xaxis_title_font=dict(size=30),
                yaxis_title_font=dict(size=30),
                xaxis_title="Epoch",
                yaxis_title=metric,
                showlegend=False,
            )
            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            with open(dashboard_file, "a") as f:
                f.write(f"<h2>{metric + '_' + mode}</h2>")
                f.write(fig_html)

    # ======================
    # nouveau zip

    replace_file_in_zip(original_zip, dashboard_file)

    print("Dashboard ajouté à:", original_zip)


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    zip_path = choose_zip()

    results = analyze_zip(zip_path)

    save_dashboard(results, zip_path)
