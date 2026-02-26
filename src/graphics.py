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
from collections import defaultdict
import plotly.express as px


METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/global/"
VALID_METRIC_FOLDERS = ["raw", "bandlimited"]


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

    img = img.T

    fig = px.imshow(
        img,
        color_continuous_scale="inferno",
        origin="lower"
    )


    fig.update_xaxes(
        visible=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        visible=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    fig.update_layout(
        width=300,
        height=300,
        margin=dict(t=20, b=0, l=0, r=0),
        coloraxis_showscale=False
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

                definition = dataset.attrs.get(
                    "definition", dataset.attrs.get("formula", "")
                )
                latex_formula = dataset.attrs.get("latex_formula", "")
                results[mode][metric_name] = {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "latex_formula": latex_formula,
                }

    return results

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

                for mode, metric_dict in metrics.items():
                    all_results.setdefault(mode, {})

                    for metric_name, values in metric_dict.items():
                        all_results[mode].setdefault(metric_name, []).append(
                            {
                                "file": file,
                                "group": group_name,
                                "mean": values["mean"],
                                "std": values["std"],
                                "latex_formula": values.get("latex_formula", ""),
                            }
                        )

    return all_results

def save_dashboard(all_results, original_zip):
    all_metrics = set()
    for mode in all_results:
        all_metrics.update(all_results[mode].keys())
    dashboard_file = "metric_dashboard.html"

    with open(dashboard_file, "w") as f:
        f.write("""
<html>
<head>
<title>Metrics Dashboard</title>
<script>
MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }
};
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
.definition {
    max-width:900px;
    font-style:italic;
    color:#555;
    font-size:20px;
    margin-top:-10px;
    margin-bottom:20px;
}
.mode-title {
    text-align:center;
    font-size:20px;
    font-weight:bold;
    margin-bottom:5px;
}
.row {
    display: flex;
    flex-direction: row;
    gap: 40px;
    margin-bottom: 40px;
}
.plot {
    flex: 1;
}
body {
    margin-left: 80px;
}
</style>
</head>
<body>
<h1>Metrics Analysis</h1>
""")

    img = load_first_m0_image(original_zip)

    if img is not None:

        heatmap_fig = build_heatmap(img)

        heatmap_html = heatmap_fig.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        )

        with open(dashboard_file, "a") as f:
            f.write(heatmap_html)
    for metric in sorted(all_metrics):
        definition = all_results["raw"][metric][0].get("latex_formula", "")
        with open(dashboard_file, "a") as f:
            f.write(f"<h2>{metric + ' = ' + definition[0]}</h2>")
            f.write('<div class="row">')
        y_values=[]
        for mode in ["raw", "bandlimited"]:
            if mode not in all_results:
                continue
            if metric not in all_results[mode]:
                continue

            data = all_results[mode][metric]
            df = pd.DataFrame(data)
            y_values.extend(df["mean"].values)
            ymin = min(y_values)
            ymax = max(y_values)
            margin = 0.05 * (ymax - ymin)

            ymin -= margin
            ymax += margin
            df["group_order"] = df["group"].astype("category").cat.codes
            df = df.sort_values(["group_order", "file"])
            df["index"] = range(len(df))

            groups = df["group"].unique()

            color_map = {
                g: c
                for g, c in zip(
                    groups, ["royalblue", "firebrick", "seagreen", "orange", "purple"]
                )
            }

            fig = go.Figure()
            fig.update_layout(width=600, height=300, margin=dict(t=20, b=20))
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
                        name=g,
                        marker=dict(color=color_map[g], size=7, opacity=0.6),
                        showlegend=True,
                    )
                )
            for g in groups:
                group_df = df[df["group"] == g]
                x_center = group_df["index"].mean()
                y_mean = group_df["mean"].mean()
                y_std = group_df["mean"].std()
                fig.add_trace(
                    go.Scatter(
                        x=[x_center],
                        y=[y_mean],
                        mode="markers",
                        name=f"{g} mean",
                        marker=dict(color=color_map[g], size=18),
                        error_y=dict(
                            type="data",
                            array=[y_std],
                            visible=True,
                            thickness=3,
                            width=8,
                        ),
                    )
                )

            tickvals = []
            ticktext = []

            for g in df["group"].unique():
                group_indices = df[df["group"] == g]["index"]

                center = group_indices.mean()

                tickvals.append(center)
                ticktext.append(g)
            fig.update_yaxes(range=[ymin, ymax])
            fig.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    title="Patient Group",
                ),
                yaxis_title_font=dict(size=30),
                xaxis_title="Epoch",
                yaxis_title=metric,
                showlegend=False,
            )
            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            with open(dashboard_file, "a") as f:
                f.write(f"""
                        <div class="plot">
                            <div class="mode-title">{mode.upper()}</div>
                            {fig_html}
                        </div>
                        """)
        with open(dashboard_file, "a") as f:
            f.write("</div>")
    with open(dashboard_file, "a") as f:
        f.write("</body></html>")
    replace_file_in_zip(original_zip, dashboard_file)

    print("Dashboard ajouté à:", original_zip)

if __name__ == "__main__":
    zip_path = choose_zip()

    results = analyze_zip(zip_path)

    save_dashboard(results, zip_path)
