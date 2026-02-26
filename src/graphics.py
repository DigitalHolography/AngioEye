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

    # transpose
    img = img.T

    h, w = img.shape

    # centre
    cy, cx = h // 2, w // 2

    # rayon = moitié du carré
    r = min(cx, cy)

    # grille coordonnées
    Y, X = np.ogrid[:h, :w]

    mask = (X - cx)**2 + (Y - cy)**2 <= r**2

    # appliquer masque circulaire
    img_circle = np.full_like(img, np.nan, dtype=float)
    img_circle[mask] = img[mask]

    # heatmap
    fig = px.imshow(
        img_circle,
        color_continuous_scale="inferno",
        origin="lower"
    )

    # cacher axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_layout(
        width=150,
        height=150,
        margin=dict(t=10, b=0, l=0, r=0),
        coloraxis_showscale=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
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
                                "latex_formula": values.get(
                                    "latex_formula", ""
                                ),
                            }
                        )

    single_group = len(detected_groups) < 1

    return dict(all_results), single_group
def build_metric_figure(df, metric, mode, ymin, ymax, single_group):

    groups = df["group"].unique()

    color_map = {
        g: c for g, c in zip(
            groups,
            ["royalblue", "firebrick", "seagreen", "orange", "purple"]
        )
    }

    fig = go.Figure()

    fig.update_layout(
        
        width=600,
        height=300,
        margin=dict(t=20, b=20)
    )

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
                marker=dict(
                    color=color_map[g],
                    size=7,
                    opacity=0.6
                ),
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
                    color=color_map[g],
                    size=18
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
            tickvals.append(group_indices.mean())
            ticktext.append(g)

        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title="Patient Group",
            
        )
    else:
        fig.update_xaxes(showticklabels=False, title="")

    fig.update_yaxes(range=[ymin, ymax])

    fig.update_layout(
        yaxis_title=metric,
        yaxis_title_font=dict(size=15)
    )

    return fig
def save_dashboard(all_results, original_zip, single_group):

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

body {
    margin-left: 60px;
    font-family: Arial, sans-serif;
}

/* ===== HEADER ===== */
.header {
    display: flex;
    align-items: center;
    gap: 40px;
    margin-bottom: 40px;
}

.header img {
    height: 180px;
    border-radius: 10px;
}

.header h1 {
    font-size: 25px;
    margin: 0;
}

/* ===== METRIC BLOCK ===== */
.metric-block {
    margin-top: 5px;
    padding-top: 5px;
    border-top: 3px solid #ddd;
}

/* ===== metric title ===== */
.metric-title {
    font-size: 15px;
    font-weight: bold;
    margin-bottom: 5px;
}


/* ===== RAW/BANDLIMITED ROW ===== */
.row {
    display: flex;
    flex-direction: row;
    gap: 60px;
    align-items: flex-start eliminar;
}

/* ===== each plot ===== */
.plot {
    flex: 1;
    text-align:center;
}

/* ===== mode titles ===== */
.mode-title {
    font-size:10px;
    font-weight:bold;
    margin-bottom:5px;
    letter-spacing:1px;
}

</style>
</head>
<body>
""")

    all_metrics = set()
    for mode in all_results:
        all_metrics.update(all_results[mode].keys())
    dashboard_file = "metric_dashboard.html"

    img = load_first_m0_image(original_zip)

    if img is not None:

        heatmap_fig = build_heatmap(img)

        heatmap_html = heatmap_fig.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        )

        with open(dashboard_file, "a") as f:
            f.write(f"""
<div class="header">
    {heatmap_html}
    <h1>Metrics Analysis</h1>
</div>
""")

    for metric in sorted(all_metrics):

        definition = all_results["raw"][metric][0].get("latex_formula", "")
        for mode in ["raw", "bandlimited"]:
            if mode in all_results and metric in all_results[mode]:
                definition = all_results[mode][metric][0].get(
                    "latex_formula", ""
                )
                break
        y_values = []

        for mode in ["raw", "bandlimited"]:
            if mode in all_results and metric in all_results[mode]:
                df_tmp = pd.DataFrame(all_results[mode][metric])
                y_values.extend(df_tmp["mean"].values)

        ymin = min(y_values)
        ymax = max(y_values)

        margin = 0.05 * (ymax - ymin if ymax != ymin else 1)
        ymin -= margin
        ymax += margin

        # ----- HTML metric header -----
        with open(dashboard_file, "a") as f:
            f.write('<div class="metric-block">')
            f.write(f'<div class="metric-title">{metric + " = " + definition[0]}</div>')
            f.write('<div class="row">')

        # ======================
        # LOOP MODES
        # ======================
        for mode in ["raw", "bandlimited"]:

            if mode not in all_results:
                continue
            if metric not in all_results[mode]:
                continue

            data = all_results[mode][metric]

            df = pd.DataFrame(data)

            df["group_order"] = df["group"].astype("category").cat.codes
            df = df.sort_values(["group_order", "file"])
            df["index"] = range(len(df))

            fig = build_metric_figure(
                df,
                metric,
                mode,
                ymin,
                ymax,
                single_group,
            )

            fig_html = fig.to_html(
                full_html=False,
                include_plotlyjs="cdn"
            )

            with open(dashboard_file, "a") as f:
                f.write(f"""
                <div class="plot">
                    <div class="mode-title">{mode.upper()}</div>
                    {fig_html}
                </div>
                """)

        with open(dashboard_file, "a") as f:
            f.write("</div></div>")

    with open(dashboard_file, "a") as f:
        f.write("</body></html>")

    replace_file_in_zip(original_zip, dashboard_file)

    print("Dashboard ajouté à:", original_zip)
    
        
    '''for metric in sorted(all_metrics):
        definition = all_results["raw"][metric][0].get("latex_formula", "")
        with open(dashboard_file, "a") as f:
            f.write(f'<div class="metric-block">')
            f.write(f'<div class="metric-title">{metric + " = " +definition[0]}</div>')
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
            if single_group:
                fig.update_xaxes(showticklabels=False, title="")
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

    print("Dashboard ajouté à:", original_zip)'''

if __name__ == "__main__":
    zip_path = choose_zip()

    results, single_group = analyze_zip(zip_path)

    save_dashboard(results, zip_path, single_group)
