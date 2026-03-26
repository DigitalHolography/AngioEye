import os
import re
import tempfile
import zipfile
from collections import defaultdict
from tkinter import Tk, filedialog
import shutil
import h5py
import numpy as np
import pandas as pd

SEGMENT_METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/artery/by_segment/"
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12

INPUT_METRICS = [
    "mu_t_over_T",
    "RI",
    "PI",
    "R_VTI",
    "SF_VTI",
    "sigma_t_over_T",
    "W50_over_T",
    "E_low_over_E_total",
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
    
]
METRIC_LABELS = {
    "mu_t_over_T": r"$\mu_t/T$",
    "RI": r"$\mathrm{RI}$",
    "PI": r"$\mathrm{PI}$",
    "R_VTI": r"$R_{VTI}$",
    "SF_VTI": r"$SF_{VTI}$",
    "sigma_t_over_T": r"$\sigma_t/T$",
    "W50_over_T": r"$W_{50}/T$",
    "E_low_over_E_total": r"$E_{\mathrm{low}}/E_{\mathrm{total}}$",
    "E_high_over_E_total": r"$E_{\mathrm{high}}/E_{\mathrm{total}}$",
    "t_max_over_T": r"$t_{\max}/T$",
    "t_min_over_T": r"$t_{\min}/T$",
    "Delta_t_over_T": r"$\Delta_t/T$",
    "slope_rise_normalized": r"$S_{\mathrm{rise}}$",
    "slope_fall_normalized": r"$S_{\mathrm{fall}}$",
    "t_up_over_T": r"$t_{\mathrm{up}}/T$",
    "t_down_over_T": r"$t_{\mathrm{down}}/T$",
    "S_decay": r"$S_{\mathrm{decay}}$",
    "crest_factor": r"$\mathrm{CF}$",
    "R_SD": r"$R_{SD}$",
    "Delta_DTI": r"$\Delta_{DTI}$",
    "gamma_t": r"$\gamma_t$",
    "spectral_entropy": r"$H_{\mathrm{spec}}$",
    "delta_phi2": r"$\Delta\phi_2$",
    "rho_h_90": r"$\rho_{h,90}$",
    "mu_h": r"$\mu_h$",
    "sigma_h": r"$\sigma_h$",
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",
    "N_H_over_T": r"$N_H/T$",
    "phase_locking_residual": r"$E_{\phi}$",
    "E_recon_H_MAX": r"$E_{\mathrm{recon},H_{\max}}$",
    "Q_t_skew": r"$Q_{t,\mathrm{skew}}$",
    "Q_t_width": r"$Q_{t,\mathrm{width}}$",
    "R_Q_t": r"$R_{Q_t}$",
    "Q_d_skew": r"$Q_{d,\mathrm{skew}}$",
    "Q_d_width": r"$Q_{d,\mathrm{width}}$",
    "R_Q_d": r"$R_{Q_d}$",
    "v_end_over_v_mean": r"$R_{EM}$",
    "E_slope": r"$E_{\mathrm{slope}}$",
    
}
COLUMN_LABELS = {
    "MED_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{med}_{seg})$",
    "STD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{STD}_{seg})$",
    "IQR_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{IQR}_{seg})$",
    "MAD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{MAD}_{seg})$",
    "CV_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{CV}_{seg})$",
    "CV_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{CV}_{b})$",
}


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def extract_sort_key(filename):
    name = os.path.basename(filename)

    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return (date, hd_index)


def extract_segment_metric(h5_path, metric_name, mode=SEGMENT_MODE):
    dataset_path = f"{SEGMENT_METRIC_FOLDER}{mode}/{metric_name}"
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            return None
        arr = np.array(f[dataset_path], dtype=float)

    if arr.ndim != 3:
        return None

    return arr


def iqr_1d(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q25 = np.nanpercentile(x, 25)
    q75 = np.nanpercentile(x, 75)
    return float(q75 - q25)


def mad_1d(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def cv_1d(x, eps=EPS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return float(sd / (np.abs(mu) + eps))

def median_1d(x, eps=EPS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(med)

def std_1d(x, eps=EPS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    std = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return float(std)


def compute_file_higher_metrics_from_segment_array(arr, eps=EPS):
    """
    arr shape = (n_beat, n_branch, n_disk)

    Returns
    -------
    dict with:
      - IQR_seg_medbeat : IQR sur segments à chaque beat, puis médiane sur beats
      - MAD_seg_medbeat : MAD sur segments à chaque beat, puis médiane sur beats
      - CV_seg_medbeat  : CV sur segments à chaque beat, puis médiane sur beats
      - CV_beat_medseg  : CV sur beats pour chaque segment, puis médiane sur segments
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        return None

    # 1) Dispersion spatiale à chaque beat, puis médiane sur beats
    beat_iqr = []
    beat_mad = []
    beat_cv_seg = []
    beat_std=[]
    beat_median=[]

    for b in range(arr.shape[0]):
        x = arr[b, :, :]
        x = x[np.isfinite(x)]

        beat_iqr.append(iqr_1d(x))
        beat_mad.append(mad_1d(x))
        beat_cv_seg.append(cv_1d(x, eps=eps))
        beat_std.append(std_1d(x))
        beat_median.append(median_1d(x))
        

    beat_iqr = np.asarray(beat_iqr, dtype=float)
    beat_mad = np.asarray(beat_mad, dtype=float)
    beat_cv_seg = np.asarray(beat_cv_seg, dtype=float)
    beat_median = np.asarray(beat_median, dtype=float)
    beat_std = np.asarray(beat_std, dtype=float)
    

    # 2) Variabilité temporelle par segment, puis médiane sur segments
    seg_cv_beat = []

    for j in range(arr.shape[1]):
        for r in range(arr.shape[2]):
            x = arr[:, j, r]
            x = x[np.isfinite(x)]
            seg_cv_beat.append(cv_1d(x, eps=eps))

    seg_cv_beat = np.asarray(seg_cv_beat, dtype=float)

    return {"MED_seg_medbeat" : (
            float(np.nanmedian(beat_median)) if np.any(np.isfinite(beat_median)) else np.nan
        ),
        "STD_seg_medbeat" : (
            float(np.nanmedian(beat_std)) if np.any(np.isfinite(beat_std)) else np.nan
        ),
        "IQR_seg_medbeat": (
            float(np.nanmedian(beat_iqr)) if np.any(np.isfinite(beat_iqr)) else np.nan
        ),
        "MAD_seg_medbeat": (
            float(np.nanmedian(beat_mad)) if np.any(np.isfinite(beat_mad)) else np.nan
        ),
        "CV_seg_medbeat": (
            float(np.nanmedian(beat_cv_seg))
            if np.any(np.isfinite(beat_cv_seg))
            else np.nan
        ),
        "CV_beat_medseg": (
            float(np.nanmedian(seg_cv_beat))
            if np.any(np.isfinite(seg_cv_beat))
            else np.nan
        ),
    }


def analyze_zip(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE):
    """
    results[group][metric][higher_metric] = [values over files]
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = sorted(
                [f for f in files if f.endswith(".h5")],
                key=extract_sort_key,
            )
            if not h5_files:
                continue

            group_name = os.path.basename(root)
            if root == tmpdir:
                group_name = "all"

            for file in h5_files:
                h5_path = os.path.join(root, file)

                for metric_name in metrics:
                    arr = extract_segment_metric(h5_path, metric_name, mode=mode)
                    if arr is None:
                        continue

                    high = compute_file_higher_metrics_from_segment_array(arr, eps=EPS)
                    if high is None:
                        continue

                    for high_name, value in high.items():
                        results[group_name][metric_name][high_name].append(value)

    return results


def format_mean_std(values, digits=3):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "NA"
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def build_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    higher_metric_order = [
        "MED_seg_medbeat",
        "STD_seg_medbeat",
        "IQR_seg_medbeat",
        "MAD_seg_medbeat",
        "CV_seg_medbeat",
        "CV_beat_medseg",
    ]

    rows = []
    for metric_name in metrics:
        metric_block = results_for_group.get(metric_name, {})
        safe_metric = metric_name.replace("_", r"\_")
        row = {
            "Metric": METRIC_LABELS.get(metric_name, metric_name.replace("_", r"\_"))
        }

        for high_name in higher_metric_order:
            vals = metric_block.get(high_name, [])
            row[COLUMN_LABELS[high_name]] = format_mean_std(vals, digits=digits)

        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_latex_table(df, caption=None, label=None):
    latex = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        column_format="l" + "c" * (df.shape[1] - 1),
    )

    if caption or label:
        lines = latex.splitlines()
        if lines and lines[0].startswith("\\begin{tabular}"):
            new_lines = ["\\begin{table}[ht]"]
            new_lines.append("\\centering")
            if caption:
                new_lines.append(f"\\caption{{{caption}}}")
            if label:
                new_lines.append(f"\\label{{{label}}}")
            new_lines.extend(lines)
            new_lines.append("\\end{table}")
            latex = "\n".join(new_lines)

    return latex


def export_group_tables(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE, digits=3):
    out_dir = os.path.join(os.path.dirname(zip_path), "latex_tables")
    os.makedirs(out_dir, exist_ok=True)

    results = analyze_zip(zip_path, metrics=metrics, mode=mode)

    print("Groups found:", list(results.keys()))

    generated = []

    for group_name in sorted(results.keys()):
        print("Building table for group:", group_name)
        df = build_group_table(results[group_name], metrics=metrics, digits=digits)

        safe_group = re.sub(r"[^A-Za-z0-9_-]+", "_", group_name)

        csv_path = os.path.join(out_dir, f"{safe_group}_variability_table.csv")
        tex_path = os.path.join(out_dir, f"{safe_group}_variability_table.tex")

        df.to_csv(csv_path, index=False)

        latex = dataframe_to_latex_table(
            df,
            caption=f"Higher-level variability metrics for group {group_name}",
            label=f"tab:{safe_group}_variability",
        )
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)

        generated.extend([csv_path, tex_path])
    replace_folder_in_zip(zip_path, arc_folder="latex_tables")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)


def replace_folder_in_zip(zip_path: str, arc_folder: str):
    temp_zip = zip_path + ".tmp"
    out_dir = os.path.join(os.path.dirname(zip_path), "latex_tables")

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if not item.filename.startswith(arc_folder + "/"):
                    buffer = zin.read(item.filename)
                    zout.writestr(item, buffer)

            for root, _, files in os.walk(out_dir):
                for fn in files:
                    fullpath = os.path.join(root, fn)
                    rel = os.path.relpath(fullpath, out_dir)
                    arcname = os.path.join(arc_folder, rel).replace("\\", "/")
                    zout.write(fullpath, arcname)

    os.replace(temp_zip, zip_path)


if __name__ == "__main__":
    zip_path = choose_zip()

    export_group_tables(zip_path)
