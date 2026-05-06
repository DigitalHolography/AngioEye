import re
from collections import defaultdict
from tkinter import Tk, filedialog
from pathlib import Path
import shutil
import h5py
import numpy as np
import pandas as pd
from angioeye_io.hdf5_io import find_first_existing_path
from angioeye_io.archive_io import replace_folder_in_zip
from ..core.grouped_batch import iter_grouped_h5_files_in_zip
from angioeye_io.archive_io import extract_folder_from_zip, temporary_zip_from_tree
from angioeye_io.hdf5_io import MetricsTree, append_metrics_trees_to_h5, read_dataset
from angioeye_io.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT, find_pipeline_group

SEGMENT_METRIC_FOLDER = "/AngioEye/Processing/waveform_shape_metrics/artery/by_segment/"
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
    "rho_h",
    "w_h",
    "N_h_over_H_minus_1",
    "D_phi",
    "s_phi_over_T",
    "eta_h",
]
METRIC_LABELS = {
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
COLUMN_LABELS = {
    "MED_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{med}_{seg})$",
    "STD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{STD}_{seg})$",
    "IQR_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{IQR}_{seg})$",
    "MAD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{MAD}_{seg})$",
    "CV_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{CV}_{seg})$",
    "MED_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{med}_{b})$",
    "STD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{STD}_{b})$",
    "IQR_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{IQR}_{b})$",
    "MAD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{MAD}_{b})$",
    "CV_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{CV}_{b})$",
}

CONTROL_ALIASES = [
    "ctrl",
    "control",
    "controls",
    "CTRL",
    "Control"
]

def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def extract_sort_key(filename):
    name = Path(filename).name

    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return (date, hd_index)


def extract_segment_metric(h5_path, metric_name, mode=SEGMENT_MODE):
    suffix = f"{mode}/{metric_name}"
    candidate_paths = [
        f"{SEGMENT_METRIC_FOLDER.rstrip('/')}/{suffix}"
    ]

    with h5py.File(h5_path, "r") as f:
        dataset_path = find_first_existing_path(
            f,
            candidate_paths,
        )
        if dataset_path is None:
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
    beat_std = []
    beat_median = []

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
    seg_iqr=[]
    seg_mad=[]
    seg_std=[]
    seg_median=[]

    for j in range(arr.shape[1]):
        for r in range(arr.shape[2]):
            x = arr[:, j, r]
            x = x[np.isfinite(x)]
            seg_cv_beat.append(cv_1d(x, eps=eps))
            seg_iqr.append(iqr_1d(x))
            seg_mad.append(mad_1d(x))
            seg_std.append(std_1d(x))
            seg_median.append(median_1d(x))

    seg_cv_beat = np.asarray(seg_cv_beat, dtype=float)
    seg_iqr = np.asarray(seg_iqr, dtype=float)
    seg_mad = np.asarray(seg_mad, dtype=float)
    seg_median = np.asarray(seg_median, dtype=float)
    seg_std = np.asarray(seg_std, dtype=float)

    return {
        "MED_seg_medbeat": (
            float(np.nanmedian(beat_median))
            if np.any(np.isfinite(beat_median))
            else np.nan
        ),
        "STD_seg_medbeat": (
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
        "MED_beat_medseg": (
            float(np.nanmedian(seg_median))
            if np.any(np.isfinite(seg_median))
            else np.nan
        ),
        "STD_beat_medseg": (
            float(np.nanmedian(seg_std)) if np.any(np.isfinite(seg_std)) else np.nan
        ),
        "IQR_beat_medseg": (
            float(np.nanmedian(seg_iqr)) if np.any(np.isfinite(seg_iqr)) else np.nan
        ),
        "MAD_beat_medseg": (
            float(np.nanmedian(seg_mad)) if np.any(np.isfinite(seg_mad)) else np.nan
        ),
        "CV_beat_medseg": (
            float(np.nanmedian(seg_cv_beat))
            if np.any(np.isfinite(seg_cv_beat))
            else np.nan
        ),
    }

def write_variability_tree(file_path):
        metrics = {}

        for metric_name in INPUT_METRICS:
            arr = extract_segment_metric(file_path, metric_name)

            if arr is None:
                continue

            high = compute_file_higher_metrics_from_segment_array(arr)

            if high is None:
                continue

            for high_name, value in high.items():
                key = f"{high_name}/{metric_name}"
                metrics[key] = np.asarray(value, dtype=float)

        if not metrics:
            return None

        return MetricsTree(
            name="Variability",
            metrics=metrics,
            attrs={
                "kind": "postprocess",
                "source": "segment_metrics",
            },
        )

def get_clean_values(results_for_group, metric, col):
    vals = results_for_group.get(metric, {}).get(col, [])
    x = np.asarray(vals, dtype=float)
    return x[np.isfinite(x)]


def analyze_zip(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE):
    """
    results[group][metric][higher_metric] = [values over files]
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for grouped_file in iter_grouped_h5_files_in_zip(
        zip_path,
        sort_key=lambda record: (record.group_name, extract_sort_key(record.file_name)),
    ):
        for metric_name in metrics:
            arr = extract_segment_metric(grouped_file.file_path, metric_name, mode=mode)
            if arr is None:
                continue

            high = compute_file_higher_metrics_from_segment_array(arr, eps=EPS)
            
            if high is None:
                continue
            
            for high_name, value in high.items():
                results[grouped_file.group_name][metric_name][high_name].append(value)

    return results


def format_mean_std(values, digits=3):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "NA"
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def build_group_table(results_for_group, metrics=INPUT_METRICS, digits=3, mode="spatial"):
    if mode == "spatial":
        higher_metric_order = [
            "MED_seg_medbeat",
            "STD_seg_medbeat",
            "IQR_seg_medbeat",
            "MAD_seg_medbeat",
            "CV_seg_medbeat",
        ]
    elif mode == "temporal":
        higher_metric_order = [
            "MED_beat_medseg",
            "STD_beat_medseg",
            "IQR_beat_medseg",
            "MAD_beat_medseg",
            "CV_beat_medseg",
        ]
    else:
        raise ValueError("mode must be 'spatial' or 'temporal'")

    rows = []
    for metric_name in metrics:
        metric_block = results_for_group.get(metric_name, {})
        row = {
            "Metric": METRIC_LABELS.get(metric_name, metric_name.replace("_", r"\_"))
        }

        for high_name in higher_metric_order:
            vals = metric_block.get(high_name, [])
            row[COLUMN_LABELS[high_name]] = format_mean_std(vals, digits=digits)

        rows.append(row)

    return pd.DataFrame(rows)

def format_group_name(name):
    if name.lower() == "ctrl":
        return "Control"
    return name.capitalize()

def format_group_list(groups, ctrl_group):
    others = [format_group_name(g) for g in groups if g != ctrl_group]

    if len(others) == 0:
        return ""

    if len(others) == 1:
        return others[0]

    return ", ".join(others[:-1]) + " and " + others[-1]

def dataframe_to_latex_table(
    df,
    caption=None,
    label=None,
    column_format=None,
    longtable=False
):
    if column_format is None:
        column_format = "l" + "c" * (df.shape[1] - 1)

    latex = df.to_latex(
        index=False,
        escape=False,
        longtable=longtable,
        column_format=column_format,
    )

    if not caption and not label:
        return latex

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



def compute_stability_scores(results_for_group, metrics, columns):
    rows = []

    for metric in metrics:
        means = []

        for col in columns:
            x = get_clean_values(results_for_group, metric, col)
            if x.size == 0:
                break

            means.append(np.nanmean(x))

        if len(means) != len(columns):
            continue

        rows.append({
            "metric": metric,
            "score": float(np.mean(means))
        })

    df = pd.DataFrame(rows).sort_values("score")
    return df

def get_interfile_mean(results_for_group, metric, col):
    vals = results_for_group.get(metric, {}).get(col, [])
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    return float(np.nanmean(x))

def find_control_group(results):
    for g in results.keys():
        if g.lower() in CONTROL_ALIASES:
            return g
    raise ValueError("No control group found")


def compute_variability_score(results_for_group, metric, mode="spatial", eps=EPS):

    if mode == "spatial":
        cols = [
            "MED_seg_medbeat",
            "STD_seg_medbeat",
            "IQR_seg_medbeat",
            "MAD_seg_medbeat",
            "CV_seg_medbeat",
        ]
    elif mode == "temporal":
        cols = [
            "MED_seg_medbeat",
            "STD_beat_medseg",
            "IQR_beat_medseg",
            "MAD_beat_medseg",
            "CV_beat_medseg",
        ]
    else:
        raise ValueError

    data = {col: get_interfile_mean(results_for_group, metric, col) for col in cols}

    if any(np.isnan(v) for v in data.values()):
        return np.nan

    denom = abs(data["MED_seg_medbeat"]) + eps

    score = (
        data[cols[1]] / denom +
        data[cols[2]] / denom +
        data[cols[3]] / denom +
        data[cols[4]] 
    ) / 4

    return float(score)

def build_variability_tables(scores, metrics, groups, mode, top_n=10):

    rows = []

    for metric in metrics:
        row = {"Metric": METRIC_LABELS.get(metric, metric)}

        vals = []
        for g in groups:
            v = scores[g][metric]
            row[f"$V_{{{g}}}^{{{mode}}}$"] = v
            vals.append(v)

        row["mean"] = np.nanmean(vals)
        rows.append(row)

    df = pd.DataFrame(rows)

    # tri global
    df = df.sort_values("mean", ascending=False).drop(columns="mean")

    df_high = df.head(top_n)
    df_low = df.tail(top_n).iloc[::-1]

    return df_low, df_high

def format_sig(x, sig=3):
    if np.isnan(x):
        return "NA"
    return f"{x:.{sig}g}"
    
def precompute_scores(results, metrics, mode):
    scores = defaultdict(dict)

    for g in results:
        for m in metrics:
            scores[g][m] = compute_variability_score(results[g], m, mode)

    return scores

def build_directional_ratio_tables(scores, metrics, groups, ctrl_group, mode, top_n=10):

    tables = {}

    other_groups = [g for g in groups if g != ctrl_group]

    for g in other_groups:

        rows = []

        for m in metrics:

            v_ctrl = scores[ctrl_group].get(m, np.nan)
            v_other = scores[g].get(m, np.nan)

            if np.isnan(v_ctrl) or np.isnan(v_other):
                ratio = np.nan
                log_ratio = np.nan
                trend = "NA"
                more_variable = "NA"
            else:
                ratio = v_other / (v_ctrl + EPS)
                log_ratio = np.log(ratio)

                if v_other > v_ctrl:
                    more_variable = g
                elif v_other < v_ctrl:
                    more_variable = ctrl_group
                else:
                    more_variable = "Equal"

                trend = "↑" if ratio > 1 else "↓" if ratio < 1 else "≈"

            rows.append({
                "Metric": METRIC_LABELS.get(m, m),
                "More variable group": more_variable,
                f"$V_{{{ctrl_group}}}^{{{mode}}}$": v_ctrl,
                f"$V_{{{g}}}^{{{mode}}}$": v_other,
                "Ratio": ratio,
                "Log-ratio": log_ratio,
                "Trend": trend,
            })

        df = pd.DataFrame(rows)

        # tri par importance
        df["abs_log"] = df["Log-ratio"].abs()
        df = df.sort_values("abs_log", ascending=False).drop(columns="abs_log")

        df.insert(0, "Rank", range(1, len(df) + 1))

        tables[g] = df.head(top_n)

    return tables

def format_dataframe(df, sig=3):
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: format_sig(x, sig))

    return df


def export_group_tables(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE, digits=3):
    base_dir = Path(zip_path).parent / "latex_tables"
    csv_dir = base_dir / "csv"
    tex_dir = base_dir / "tex"

    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_zip(zip_path, metrics=metrics, mode=mode)

    print("Groups found:", list(results.keys()))

    ctrl_group = find_control_group(results)
    groups = list(results.keys())

    modes = ["spatial", "temporal"]

    scores = {
        mode: precompute_scores(results, metrics, mode)
        for mode in modes
    }

    # =========================
    # TABLES PAR GROUPE
    # =========================
    for group_name in sorted(results.keys()):

        safe_group = re.sub(r"[^A-Za-z0-9_-]+", "_", group_name)

        for mode in modes:

            df = build_group_table(results[group_name], metrics, mode=mode)
            df = format_dataframe(df)

            name = f"{safe_group}_{mode}"

            df.to_csv(csv_dir/f"{name}.csv", index=False)

            with open(tex_dir/f"{name}.tex", "w", encoding="utf-8") as f:
                f.write(dataframe_to_latex_table(
                    df,
                    caption=f"{mode.capitalize()} variability ({group_name})",
                    label=f"tab:{name}",
                ))

    # =========================
    # TABLES GLOBAL VARIABILITY
    # =========================
    for mode in modes:

        df_low, df_high = build_variability_tables(
            scores[mode],
            metrics,
            groups,
            mode=mode,
            top_n=10
        )

        df_low = format_dataframe(df_low)
        df_high = format_dataframe(df_high)

        df_low.to_csv(csv_dir/f"{mode}_low.csv", index=False)
        df_high.to_csv(csv_dir/f"{mode}_high.csv", index=False)


        with open(tex_dir/f"{mode}_low.tex", "w", encoding="utf-8") as f:
            f.write(dataframe_to_latex_table(
                df_low,
                caption=f"Least {mode} variability (most stable)",
                label=f"tab:{mode}_low"
            ))

        with open(tex_dir/f"{mode}_high.tex", "w", encoding="utf-8") as f:
            f.write(dataframe_to_latex_table(
                df_high,
                caption=f"Most {mode} variability",
                label=f"tab:{mode}_high"
            ))

    # =========================
    # TABLES RATIO (direction)
    # =========================
    for mode in modes:

        ratio_tables = build_directional_ratio_tables(
            scores[mode],
            metrics,
            groups,
            ctrl_group,
            mode=mode,
            top_n=10
        )

        for g, df_ratio in ratio_tables.items():

            df_ratio = format_dataframe(df_ratio)

            name = f"{mode}_ratio_{g}"

            df_ratio.to_csv(csv_dir/f"{name}.csv", index=False)

            with open(tex_dir/f"{name}.tex", "w", encoding="utf-8") as f:
                f.write(dataframe_to_latex_table(
                    df_ratio,
                    caption=(
                        f"{mode.capitalize()} variability: {g} vs {ctrl_group}"
                    ),
                    label=f"tab:{name}"
                ))


    
    replace_folder_in_zip(zip_path, str(base_dir), arc_folder="latex_tables")

    
    if base_dir.is_dir():
        shutil.rmtree(base_dir)


if __name__ == "__main__":
    zip_path = choose_zip()

    export_group_tables(zip_path)
