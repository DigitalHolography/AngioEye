import re
import shutil
from collections import defaultdict
from pathlib import Path
from tkinter import Tk, filedialog

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except ImportError as exc:
    raise ImportError(
        "This script requires scipy for Mann-Whitney tests. Install it with: pip install scipy"
    ) from exc

from angioeye_io.hdf5_io import find_first_existing_path
from angioeye_io.archive_io import replace_folder_in_zip
from ..core.grouped_batch import iter_grouped_h5_files_in_zip
from angioeye_io.hdf5_io import MetricsTree


SEGMENT_METRIC_FOLDER = "/AngioEye/Processing/waveform_shape_metrics/artery/by_segment/"
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12

DEFAULT_TOP_N = 10

CONTROL_GROUP_PATTERNS = [
    r"^control$",
    r"^controle$",
    r"^ctrl$",
    r"^ctl$",
    r"^controls$",
    r"^healthy_control$",
    r"^healthy_controls$",
]

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
    # Spatial variability
    "STD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{STD}_{seg})$",
    "IQR_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{IQR}_{seg})$",
    "MAD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{MAD}_{seg})$",
    "CV_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{CV}_{seg})$",
    # Temporal variability
    "STD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{STD}_{b})$",
    "IQR_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{IQR}_{b})$",
    "MAD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{MAD}_{b})$",
    "CV_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{CV}_{b})$",
}

SPATIAL_VARIABILITY_COLUMNS = [
    "STD_seg_medbeat",
    "IQR_seg_medbeat",
    "MAD_seg_medbeat",
    "CV_seg_medbeat",
]

TEMPORAL_VARIABILITY_COLUMNS = [
    "STD_beat_medseg",
    "MAD_beat_medseg",
    "CV_beat_medseg",
]

SPATIAL_RAW_COLUMNS = ["MED_seg_medbeat", *SPATIAL_VARIABILITY_COLUMNS]
TEMPORAL_RAW_COLUMNS = ["MED_seg_medbeat", *TEMPORAL_VARIABILITY_COLUMNS]

SUMMARY_PVALUE_METRICS = [
    "RI",
    "PI",
    "N_t_over_T",
    "N_eff_over_T",
]


# -----------------------------------------------------------------------------
# Basic IO and metric extraction
# -----------------------------------------------------------------------------

def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")


def extract_sort_key(filename):
    name = Path(filename).name

    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return date, hd_index


def extract_segment_metric(h5_path, metric_name, mode=SEGMENT_MODE):
    suffix = f"{mode}/{metric_name}"
    candidate_paths = [f"{SEGMENT_METRIC_FOLDER.rstrip('/')}/{suffix}"]

    with h5py.File(h5_path, "r") as f:
        dataset_path = find_first_existing_path(f, candidate_paths)

        if dataset_path is None:
            return None

        arr = np.array(f[dataset_path], dtype=float)

    if arr.ndim != 3:
        return None

    return arr


# -----------------------------------------------------------------------------
# Robust 1D statistics
# -----------------------------------------------------------------------------

def finite_1d(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def iqr_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    q25 = np.nanpercentile(x, 25)
    q75 = np.nanpercentile(x, 75)
    return float(q75 - q25)


def mad_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def cv_1d(x, eps=EPS):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return float(sd / (np.abs(mu) + eps))


def median_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.nanmedian(x))


def std_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.nanstd(x, ddof=1) if x.size > 1 else 0.0)


def nanmedian_or_nan(x):
    x = np.asarray(x, dtype=float)
    if np.any(np.isfinite(x)):
        return float(np.nanmedian(x))
    return np.nan


def clean_values(values):
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


# -----------------------------------------------------------------------------
# Per-file higher-order metrics
# -----------------------------------------------------------------------------

def compute_file_higher_metrics_from_segment_array(arr, eps=EPS):
    """
    Parameters
    ----------
    arr : np.ndarray
        Shape = (n_beat, n_branch, n_disk)

    Returns
    -------
    dict
        Central level:
          - MED_seg_medbeat:
            median across segments at each beat, then median across beats.

        Spatial variability:
          - STD_seg_medbeat:
            STD across segments at each beat, then median across beats.
          - IQR_seg_medbeat:
            IQR across segments at each beat, then median across beats.
          - MAD_seg_medbeat:
            MAD across segments at each beat, then median across beats.
          - CV_seg_medbeat:
            CV across segments at each beat, then median across beats.

        Temporal variability:
          - STD_beat_medseg:
            STD across beats for each segment, then median across segments.
          - IQR_beat_medseg:
            IQR across beats for each segment, then median across segments.
          - MAD_beat_medseg:
            MAD across beats for each segment, then median across segments.
          - CV_beat_medseg:
            CV across beats for each segment, then median across segments.
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 3:
        return None

    beat_median = []
    beat_std = []
    beat_iqr = []
    beat_mad = []
    beat_cv_seg = []

    for b in range(arr.shape[0]):
        x = arr[b, :, :]
        x = finite_1d(x)

        beat_median.append(median_1d(x))
        beat_std.append(std_1d(x))
        beat_iqr.append(iqr_1d(x))
        beat_mad.append(mad_1d(x))
        beat_cv_seg.append(cv_1d(x, eps=eps))

    beat_median = np.asarray(beat_median, dtype=float)
    beat_std = np.asarray(beat_std, dtype=float)
    beat_iqr = np.asarray(beat_iqr, dtype=float)
    beat_mad = np.asarray(beat_mad, dtype=float)
    beat_cv_seg = np.asarray(beat_cv_seg, dtype=float)

    seg_std_beat = []
    seg_iqr_beat = []
    seg_mad_beat = []
    seg_cv_beat = []

    for j in range(arr.shape[1]):
        for r in range(arr.shape[2]):
            x = arr[:, j, r]
            x = finite_1d(x)

            seg_std_beat.append(std_1d(x))
            seg_iqr_beat.append(iqr_1d(x))
            seg_mad_beat.append(mad_1d(x))
            seg_cv_beat.append(cv_1d(x, eps=eps))

    seg_std_beat = np.asarray(seg_std_beat, dtype=float)
    seg_iqr_beat = np.asarray(seg_iqr_beat, dtype=float)
    seg_mad_beat = np.asarray(seg_mad_beat, dtype=float)
    seg_cv_beat = np.asarray(seg_cv_beat, dtype=float)

    return {
        "MED_seg_medbeat": nanmedian_or_nan(beat_median),
        "STD_seg_medbeat": nanmedian_or_nan(beat_std),
        "IQR_seg_medbeat": nanmedian_or_nan(beat_iqr),
        "MAD_seg_medbeat": nanmedian_or_nan(beat_mad),
        "CV_seg_medbeat": nanmedian_or_nan(beat_cv_seg),
        "STD_beat_medseg": nanmedian_or_nan(seg_std_beat),
        "IQR_beat_medseg": nanmedian_or_nan(seg_iqr_beat),
        "MAD_beat_medseg": nanmedian_or_nan(seg_mad_beat),
        "CV_beat_medseg": nanmedian_or_nan(seg_cv_beat),
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


# -----------------------------------------------------------------------------
# Zip analysis
# -----------------------------------------------------------------------------

def analyze_zip(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE):
    """
    Returns
    -------
    results : dict
        results[group][metric][higher_metric] = list of values over files.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for grouped_file in iter_grouped_h5_files_in_zip(
        zip_path,
        sort_key=lambda record: (
            record.group_name,
            extract_sort_key(record.file_name),
        ),
    ):
        for metric_name in metrics:
            arr = extract_segment_metric(
                grouped_file.file_path,
                metric_name,
                mode=mode,
            )

            if arr is None:
                continue

            high = compute_file_higher_metrics_from_segment_array(
                arr,
                eps=EPS,
            )

            if high is None:
                continue

            for high_name, value in high.items():
                results[grouped_file.group_name][metric_name][high_name].append(value)

    return results


def normalize_group_name(group_name):
    s = str(group_name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def is_control_group(group_name, patterns=CONTROL_GROUP_PATTERNS):
    s = normalize_group_name(group_name)
    return any(re.match(pattern, s, flags=re.IGNORECASE) for pattern in patterns)


def find_control_group(results):
    groups = list(results.keys())
    candidates = [g for g in groups if is_control_group(g)]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer exact short names if several candidates exist.
        priority = {"control": 0, "controle": 1, "ctrl": 2, "ctl": 3}
        candidates = sorted(
            candidates,
            key=lambda g: priority.get(normalize_group_name(g), 100),
        )
        return candidates[0]

    raise ValueError(
        "No control group found. Expected a group folder named like: "
        "control, controle, ctrl, ctl, healthy_control. "
        f"Groups found: {groups}"
    )


# -----------------------------------------------------------------------------
# Formatting and raw tables
# -----------------------------------------------------------------------------

def format_mean_std(values, digits=3):
    x = clean_values(values)
    if x.size == 0:
        return "NA"

    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def metric_label(metric_name):
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", r"\_"))


def build_group_table_with_columns(
    results_for_group,
    selected_higher_metrics,
    metrics=INPUT_METRICS,
    digits=3,
):
    rows = []

    for metric_name in metrics:
        metric_block = results_for_group.get(metric_name, {})

        row = {
            "Metric": metric_label(metric_name),
        }

        for high_name in selected_higher_metrics:
            vals = metric_block.get(high_name, [])
            row[COLUMN_LABELS[high_name]] = format_mean_std(vals, digits=digits)

        rows.append(row)

    return pd.DataFrame(rows)


def build_spatial_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    return build_group_table_with_columns(
        results_for_group=results_for_group,
        selected_higher_metrics=SPATIAL_RAW_COLUMNS,
        metrics=metrics,
        digits=digits,
    )


def build_temporal_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    return build_group_table_with_columns(
        results_for_group=results_for_group,
        selected_higher_metrics=TEMPORAL_RAW_COLUMNS,
        metrics=metrics,
        digits=digits,
    )


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------

def combine_variability_score(
    results_for_group,
    metric_name,
    higher_metrics,
    eps=EPS,
):
    """
    Combines several higher-order variability descriptors into one dimensionless score.

    Each variability descriptor is normalized by the central median level of the
    corresponding metric:

        normalized_variability = variability / (abs(MED_seg_medbeat) + eps)

    This is especially important for STD and IQR, otherwise metrics with naturally
    larger numerical values dominate the score.

    Notes
    -----
    - CV columns are already normalized by construction, so they are kept as-is.
    - STD/IQR/MAD columns are divided file-by-file by MED_seg_medbeat.
    - The final score is the mean of the available normalized variability columns.
    """
    metric_block = results_for_group.get(metric_name, {})
    median_level = np.asarray(metric_block.get("MED_seg_medbeat", []), dtype=float)

    arrays = []

    for high_name in higher_metrics:
        x = np.asarray(metric_block.get(high_name, []), dtype=float)
        if x.size == 0:
            continue

        if high_name.startswith("CV_"):
            normalized = x
        else:
            min_len = min(len(x), len(median_level))
            if min_len == 0:
                continue

            normalized = x[:min_len] / (np.abs(median_level[:min_len]) + eps)

        arrays.append(np.asarray(normalized, dtype=float))

    if not arrays:
        return np.asarray([], dtype=float)

    min_len = min(len(x) for x in arrays)
    if min_len == 0:
        return np.asarray([], dtype=float)

    matrix = np.vstack([x[:min_len] for x in arrays]).T
    values = np.nanmean(matrix, axis=1)
    return clean_values(values)


def single_higher_metric_values(results_for_group, metric_name, high_name):
    return clean_values(results_for_group.get(metric_name, {}).get(high_name, []))


def summarize_values(values):
    x = clean_values(values)
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }

    return {
        "n": int(x.size),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1) if x.size > 1 else 0.0),
        "median": float(np.nanmedian(x)),
        "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
    }


def mann_whitney_pvalue(control_values, group_values):
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan

    try:
        res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except ValueError:
        return np.nan


def rank_biserial_effect_size(control_values, group_values):
    """
    Rank-biserial correlation derived from the Mann-Whitney U statistic.
    Positive value means the compared group tends to have larger values than control.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan

    try:
        res = mannwhitneyu(y, x, alternative="two-sided", method="auto")
        u = float(res.statistic)
        return float((2.0 * u) / (x.size * y.size) - 1.0)
    except ValueError:
        return np.nan


def build_variability_ranking_table(
    results_for_group,
    higher_metrics,
    metrics=INPUT_METRICS,
    n=DEFAULT_TOP_N,
    ascending=False,
    digits=4,
):
    rows = []

    for metric_name in metrics:
        values = combine_variability_score(
            results_for_group,
            metric_name,
            higher_metrics,
        )
        stats = summarize_values(values)

        rows.append(
            {
                "Metric": metric_label(metric_name),
                "metric_name": metric_name,
                "n": stats["n"],
                "score_mean": stats["mean"],
                "score_std": stats["std"],
                "score_median": stats["median"],
                "score_iqr": stats["iqr"],
            }
        )

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["score_mean"])]
    df = df.sort_values("score_mean", ascending=ascending).head(n)

    df["Score mean ± SD"] = df.apply(
        lambda r: (
            "NA"
            if not np.isfinite(r["score_mean"])
            else f"{r['score_mean']:.{digits}g} $\\pm$ {r['score_std']:.{digits}g}"
        ),
        axis=1,
    )
    df["Score median [IQR]"] = df.apply(
        lambda r: (
            "NA"
            if not np.isfinite(r["score_median"])
            else f"{r['score_median']:.{digits}g} [{r['score_iqr']:.{digits}g}]"
        ),
        axis=1,
    )

    return df[["Metric", "n", "Score mean ± SD", "Score median [IQR]"]]


def build_contrast_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics=INPUT_METRICS,
    n=DEFAULT_TOP_N,
    digits=4,
):
    rows = []

    for metric_name in metrics:
        x = combine_variability_score(control_results, metric_name, higher_metrics)
        y = combine_variability_score(group_results, metric_name, higher_metrics)
        sx = summarize_values(x)
        sy = summarize_values(y)

        diff = sy["median"] - sx["median"]
        ratio = sy["median"] / (abs(sx["median"]) + EPS) if np.isfinite(sx["median"]) else np.nan

        rows.append(
            {
                "Metric": metric_label(metric_name),
                "metric_name": metric_name,
                f"n {control_name}": sx["n"],
                f"n {group_name}": sy["n"],
                f"Median {control_name}": sx["median"],
                f"Median {group_name}": sy["median"],
                "Median difference": diff,
                "Median ratio": ratio,
                "Abs median difference": abs(diff) if np.isfinite(diff) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["Abs median difference"])]
    df = df.sort_values("Abs median difference", ascending=False).head(n)

    for col in [f"Median {control_name}", f"Median {group_name}", "Median difference", "Median ratio"]:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    return df[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            f"Median {control_name}",
            f"Median {group_name}",
            "Median difference",
            "Median ratio",
        ]
    ]


def build_mannwhitney_ranking_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics=INPUT_METRICS,
    n=None,
    digits=4,
):
    rows = []

    for metric_name in metrics:
        x = combine_variability_score(control_results, metric_name, higher_metrics)
        y = combine_variability_score(group_results, metric_name, higher_metrics)
        sx = summarize_values(x)
        sy = summarize_values(y)

        p = mann_whitney_pvalue(x, y)
        rbc = rank_biserial_effect_size(x, y)
        diff = sy["median"] - sx["median"]

        rows.append(
            {
                "Metric": metric_label(metric_name),
                "metric_name": metric_name,
                f"n {control_name}": sx["n"],
                f"n {group_name}": sy["n"],
                f"Median {control_name}": sx["median"],
                f"Median {group_name}": sy["median"],
                "Median difference": diff,
                "Mann-Whitney p-value": p,
                "Rank-biserial effect": rbc,
            }
        )

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["Mann-Whitney p-value"])]
    df = df.sort_values("Mann-Whitney p-value", ascending=True)

    if n is not None:
        df = df.head(n)

    for col in [f"Median {control_name}", f"Median {group_name}", "Median difference", "Rank-biserial effect"]:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    df["Mann-Whitney p-value"] = df["Mann-Whitney p-value"].apply(
        lambda v: format_float(v, digits=digits)
    )

    return df[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            f"Median {control_name}",
            f"Median {group_name}",
            "Median difference",
            "Rank-biserial effect",
            "Mann-Whitney p-value",
        ]
    ]


def get_descriptor_values_for_test(
    results_for_group,
    metric_name,
    high_name,
    eps=EPS,
):
    """
    Returns the per-file values used for one descriptor-specific Mann-Whitney test.

    STD, IQR and MAD are normalized by MED_seg_medbeat to remove the influence of
    the absolute metric level. CV is already normalized and is therefore kept as-is.
    """
    metric_block = results_for_group.get(metric_name, {})
    x = np.asarray(metric_block.get(high_name, []), dtype=float)

    if x.size == 0:
        return np.asarray([], dtype=float)

    if high_name.startswith("CV_"):
        return clean_values(x)

    median_level = np.asarray(metric_block.get("MED_seg_medbeat", []), dtype=float)
    min_len = min(len(x), len(median_level))

    if min_len == 0:
        return np.asarray([], dtype=float)

    normalized = x[:min_len] / (np.abs(median_level[:min_len]) + eps)
    return clean_values(normalized)


def build_descriptor_pvalue_summary_table(
    control_results,
    group_results,
    descriptor_map,
    control_name,
    group_name,
    metrics=SUMMARY_PVALUE_METRICS,
    digits=4,
):
    """
    Builds a compact table with one Mann-Whitney p-value per variability descriptor.

    Parameters
    ----------
    descriptor_map : dict
        Maps display descriptor names to higher metric names.
        Example for spatial:
            {
                "STD": "STD_seg_medbeat",
                "IQR": "IQR_seg_medbeat",
                "MAD": "MAD_seg_medbeat",
                "CV": "CV_seg_medbeat",
            }

    Notes
    -----
    - STD, IQR and MAD are normalized by MED_seg_medbeat before testing.
    - CV is kept as-is because it is already normalized.
    - The column "Mean p-value" is not the arithmetic mean of the descriptor
      p-values. It is computed like the previous composite Mann-Whitney table:
      first average the normalized descriptors file-by-file, then run one
      Mann-Whitney test on this mean descriptor score between groups.
    """
    rows = []
    higher_metrics = list(descriptor_map.values())

    for metric_name in metrics:
        row = {"Metric": metric_label(metric_name)}
        n_control_values = []
        n_group_values = []

        for descriptor_name, high_name in descriptor_map.items():
            x = get_descriptor_values_for_test(control_results, metric_name, high_name)
            y = get_descriptor_values_for_test(group_results, metric_name, high_name)
            p = mann_whitney_pvalue(x, y)

            row[f"{descriptor_name} p-value"] = p
            n_control_values.append(len(x))
            n_group_values.append(len(y))

        mean_score_control = combine_variability_score(
            control_results,
            metric_name,
            higher_metrics=higher_metrics,
        )
        mean_score_group = combine_variability_score(
            group_results,
            metric_name,
            higher_metrics=higher_metrics,
        )
        row["Mean p-value"] = mann_whitney_pvalue(
            mean_score_control,
            mean_score_group,
        )

        row[f"n {control_name}"] = int(max(n_control_values)) if n_control_values else 0
        row[f"n {group_name}"] = int(max(n_group_values)) if n_group_values else 0
        rows.append(row)

    df = pd.DataFrame(rows)

    p_cols = [f"{name} p-value" for name in descriptor_map.keys()] + ["Mean p-value"]
    for col in p_cols:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    return df[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            *[f"{name} p-value" for name in descriptor_map.keys()],
            "Mean p-value",
        ]
    ]


SPATIAL_DESCRIPTOR_MAP = {
    "STD": "STD_seg_medbeat",
    "IQR": "IQR_seg_medbeat",
    "MAD": "MAD_seg_medbeat",
    "CV": "CV_seg_medbeat",
}

TEMPORAL_DESCRIPTOR_MAP = {
    "STD": "STD_beat_medseg",
    "MAD": "MAD_beat_medseg",
    "CV": "CV_beat_medseg",
}


def descriptor_axis_label(descriptor_name, high_name):
    if high_name.startswith("CV_"):
        return f"{descriptor_name}"
    return f"{descriptor_name} / |median metric value|"


def export_variability_value_plots(
    results,
    out_dir,
    descriptor_map,
    domain_name,
    metrics=SUMMARY_PVALUE_METRICS,
    dpi=300,
):
    """
    Exports PNG figures showing individual variability values by cohort.

    One PNG is generated for each metric and each variability descriptor.
    Example:
        spatial_IQR_RI_by_group.png

    Values are exactly the same values as those used in the descriptor-specific
    Mann-Whitney tests:
        - STD, IQR and MAD are normalized by MED_seg_medbeat.
        - CV is kept as-is.

    Parameters
    ----------
    results : dict
        Output of analyze_zip.
    out_dir : Path-like
        Folder where PNG files are written.
    descriptor_map : dict
        Example spatial map:
            {"STD": "STD_seg_medbeat", "IQR": "IQR_seg_medbeat", ...}
    domain_name : str
        Usually "spatial" or "temporal".
    metrics : list[str]
        Base metrics to plot, e.g. RI, PI, N_t_over_T, N_eff_over_T.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    group_names = sorted(results.keys())
    rng = np.random.default_rng(12345)

    for metric_name in metrics:
        for descriptor_name, high_name in descriptor_map.items():
            group_values = []
            non_empty_group_names = []

            for group_name in group_names:
                values = get_descriptor_values_for_test(
                    results[group_name],
                    metric_name,
                    high_name,
                )
                values = clean_values(values)

                if values.size == 0:
                    continue

                group_values.append(values)
                non_empty_group_names.append(group_name)

            if not group_values:
                continue

            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(group_values)), 4.5))

            positions = np.arange(1, len(group_values) + 1)

            ax.boxplot(
                group_values,
                positions=positions,
                widths=0.45,
                showfliers=False,
            )

            for pos, values in zip(positions, group_values):
                jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    s=18,
                    alpha=0.75,
                )

            xlabels = [
                "{}\n(n={})".format(name, len(values))
                for name, values in zip(non_empty_group_names, group_values)
            ]
            ax.set_xticks(positions)
            ax.set_xticklabels(xlabels, rotation=0)
            ax.set_ylabel(descriptor_axis_label(descriptor_name, high_name))
            ax.set_xlabel("Cohort")
            ax.set_title(
                f"{domain_name.capitalize()} {descriptor_name} variability for {metric_name}"
            )
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()

            filename = (
                f"{safe_name(domain_name)}_"
                f"{safe_name(descriptor_name)}_"
                f"{safe_name(metric_name)}_by_group.png"
            )
            path = out_dir / filename
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)

    return generated


# -----------------------------------------------------------------------------
# LaTeX export
# -----------------------------------------------------------------------------

def dataframe_to_latex_table(
    df,
    caption=None,
    label=None,
    font_size=r"\scriptsize",
):
    r"""
    Requires in Overleaf preamble:
        \usepackage{float}
        \usepackage{booktabs}
    """
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        column_format="l" + "c" * (df.shape[1] - 1),
    )

    lines = [
        r"\begin{table}[H]",
        r"\raggedright",
        font_size,
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.9}",
    ]

    if caption:
        lines.append(f"\\caption{{{caption}}}")

    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append(latex_tabular)
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_table(df, csv_path, tex_path, caption, label, digits=3):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    latex = dataframe_to_latex_table(
        df,
        caption=caption,
        label=label,
        font_size=r"\scriptsize"
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return [csv_path, tex_path]


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------

def export_group_tables(
    zip_path,
    metrics=INPUT_METRICS,
    mode=SEGMENT_MODE,
    digits=3,
    top_n=DEFAULT_TOP_N,
):
    """
    Creates inside the ZIP:

    latex_tables/
      spatial/
        raw/
          <group>_spatial_variability_table.{csv,tex}
        comparisons_vs_control/
          <group>_vs_<control>_n_most_spatially_variable_metrics.{csv,tex}
          <group>_vs_<control>_n_least_spatially_variable_metrics.{csv,tex}
          <group>_vs_<control>_strongest_spatial_variability_contrast.{csv,tex}
          <group>_vs_<control>_best_spatial_variability_mannwhitney.{csv,tex}
      temporal/
        raw/
          <group>_temporal_variability_table.{csv,tex}
        comparisons_vs_control/
          <group>_vs_<control>_n_most_temporally_variable_metrics.{csv,tex}
          <group>_vs_<control>_n_least_temporally_variable_metrics.{csv,tex}
          <group>_vs_<control>_strongest_temporal_variability_contrast.{csv,tex}
          <group>_vs_<control>_best_temporal_variability_mannwhitney.{csv,tex}
    """
    zip_path = Path(zip_path)
    out_dir = zip_path.parent / "latex_tables"

    spatial_raw_dir = out_dir / "spatial" / "raw"
    temporal_raw_dir = out_dir / "temporal" / "raw"
    spatial_cmp_dir = out_dir / "spatial" / "comparisons_vs_control"
    temporal_cmp_dir = out_dir / "temporal" / "comparisons_vs_control"
    spatial_fig_dir = out_dir / "spatial" / "figures"
    temporal_fig_dir = out_dir / "temporal" / "figures"

    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    for d in [
        spatial_raw_dir,
        temporal_raw_dir,
        spatial_cmp_dir,
        temporal_cmp_dir,
        spatial_fig_dir,
        temporal_fig_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    results = analyze_zip(zip_path, metrics=metrics, mode=mode)
    print("Groups found:", list(results.keys()))

    control_group = find_control_group(results)
    safe_control = safe_name(control_group)
    print("Control group detected:", control_group)

    generated = []

    # ------------------------------------------------------------------
    # PNG figures: individual variability values by cohort.
    # ------------------------------------------------------------------
    generated.extend(
        export_variability_value_plots(
            results,
            spatial_fig_dir,
            descriptor_map=SPATIAL_DESCRIPTOR_MAP,
            domain_name="spatial",
            metrics=SUMMARY_PVALUE_METRICS,
        )
    )
    generated.extend(
        export_variability_value_plots(
            results,
            temporal_fig_dir,
            descriptor_map=TEMPORAL_DESCRIPTOR_MAP,
            domain_name="temporal",
            metrics=SUMMARY_PVALUE_METRICS,
        )
    )

    # ------------------------------------------------------------------
    # Raw tables for every group, including control.
    # ------------------------------------------------------------------
    for group_name in sorted(results.keys()):
        print("Building raw spatial and temporal tables for group:", group_name)
        safe_group = safe_name(group_name)

        df_spatial = build_spatial_group_table(
            results[group_name],
            metrics=metrics,
            digits=digits,
        )
        generated.extend(
            save_table(
                df_spatial,
                spatial_raw_dir / f"{safe_group}_spatial_variability_table.csv",
                spatial_raw_dir / f"{safe_group}_spatial_variability_table.tex",
                caption=f"Raw spatial variability metrics for group {group_name}",
                label=f"tab:{safe_group}_spatial_variability_raw",
                digits=digits,
            )
        )

        df_temporal = build_temporal_group_table(
            results[group_name],
            metrics=metrics,
            digits=digits,
        )
        generated.extend(
            save_table(
                df_temporal,
                temporal_raw_dir / f"{safe_group}_temporal_variability_table.csv",
                temporal_raw_dir / f"{safe_group}_temporal_variability_table.tex",
                caption=f"Raw temporal variability metrics for group {group_name}",
                label=f"tab:{safe_group}_temporal_variability_raw",
                digits=digits,
            )
        )

    # ------------------------------------------------------------------
    # Control vs every other group.
    # ------------------------------------------------------------------
    control_results = results[control_group]

    for group_name in sorted(results.keys()):
        if group_name == control_group:
            continue

        print(f"Building comparison tables: {group_name} vs {control_group}")
        group_results = results[group_name]
        safe_group = safe_name(group_name)
        pair = f"{safe_group}_vs_{safe_control}"

        # ------------------------------
        # Spatial comparison tables
        # ------------------------------
        df = build_variability_ranking_table(
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            metrics=metrics,
            n=top_n,
            ascending=False,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_n_most_spatially_variable_metrics.csv",
                spatial_cmp_dir / f"{pair}_n_most_spatially_variable_metrics.tex",
                caption=f"Top {top_n} most spatially variable metrics in group {group_name}",
                label=f"tab:{pair}_most_spatially_variable",
                digits=digits,
            )
        )

        df = build_variability_ranking_table(
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            metrics=metrics,
            n=top_n,
            ascending=True,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_n_least_spatially_variable_metrics.csv",
                spatial_cmp_dir / f"{pair}_n_least_spatially_variable_metrics.tex",
                caption=f"Top {top_n} least spatially variable metrics in group {group_name}",
                label=f"tab:{pair}_least_spatially_variable",
                digits=digits,
            )
        )

        df = build_contrast_table(
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_strongest_spatial_variability_contrast.csv",
                spatial_cmp_dir / f"{pair}_strongest_spatial_variability_contrast.tex",
                caption=f"Top {top_n} strongest spatial variability contrasts between {group_name} and {control_group}",
                label=f"tab:{pair}_strongest_spatial_contrast",
                digits=digits,
            )
        )

        df = build_mannwhitney_ranking_table(
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=None,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_best_spatial_variability_mannwhitney.csv",
                spatial_cmp_dir / f"{pair}_best_spatial_variability_mannwhitney.tex",
                caption=f"Best spatial variability metrics between {control_group} and {group_name}, ranked by Mann-Whitney p-value",
                label=f"tab:{pair}_best_spatial_mannwhitney",
                digits=digits,
            )
        )

        df = build_descriptor_pvalue_summary_table(
            control_results,
            group_results,
            descriptor_map=SPATIAL_DESCRIPTOR_MAP,
            control_name=control_group,
            group_name=group_name,
            metrics=SUMMARY_PVALUE_METRICS,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_spatial_descriptor_pvalue_summary_RI_PI_Nt_Neff.csv",
                spatial_cmp_dir / f"{pair}_spatial_descriptor_pvalue_summary_RI_PI_Nt_Neff.tex",
                caption=(
                    f"Spatial descriptor-specific Mann-Whitney p-values between "
                    f"{control_group} and {group_name} for $\rm RI$, $\rm PI$, "
                    f"$N_t/T$ and $N_{{\mathrm{{eff}}}}/T$"
                ),
                label=f"tab:{pair}_spatial_descriptor_pvalue_summary",
                digits=digits,
            )
        )

        # ------------------------------
        # Temporal comparison tables
        # ------------------------------
        df = build_variability_ranking_table(
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            metrics=metrics,
            n=top_n,
            ascending=False,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_n_most_temporally_variable_metrics.csv",
                temporal_cmp_dir / f"{pair}_n_most_temporally_variable_metrics.tex",
                caption=f"Top {top_n} most temporally variable metrics in group {group_name}",
                label=f"tab:{pair}_most_temporally_variable",
                digits=digits,
            )
        )

        df = build_variability_ranking_table(
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            metrics=metrics,
            n=top_n,
            ascending=True,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_n_least_temporally_variable_metrics.csv",
                temporal_cmp_dir / f"{pair}_n_least_temporally_variable_metrics.tex",
                caption=f"Top {top_n} least temporally variable metrics in group {group_name}",
                label=f"tab:{pair}_least_temporally_variable",
                digits=digits,
            )
        )

        df = build_contrast_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_strongest_temporal_variability_contrast.csv",
                temporal_cmp_dir / f"{pair}_strongest_temporal_variability_contrast.tex",
                caption=f"Top {top_n} strongest temporal variability contrasts between {group_name} and {control_group}",
                label=f"tab:{pair}_strongest_temporal_contrast",
                digits=digits,
            )
        )

        df = build_mannwhitney_ranking_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=None,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_best_temporal_variability_mannwhitney.csv",
                temporal_cmp_dir / f"{pair}_best_temporal_variability_mannwhitney.tex",
                caption=f"Best temporal variability metrics between {control_group} and {group_name}, ranked by Mann-Whitney p-value",
                label=f"tab:{pair}_best_temporal_mannwhitney",
                digits=digits,
            )
        )

        df = build_descriptor_pvalue_summary_table(
            control_results,
            group_results,
            descriptor_map=TEMPORAL_DESCRIPTOR_MAP,
            control_name=control_group,
            group_name=group_name,
            metrics=SUMMARY_PVALUE_METRICS,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_temporal_descriptor_pvalue_summary_RI_PI_Nt_Neff.csv",
                temporal_cmp_dir / f"{pair}_temporal_descriptor_pvalue_summary_RI_PI_Nt_Neff.tex",
                caption=(
                    f"Temporal descriptor-specific Mann-Whitney p-values between "
                    f"{control_group} and {group_name} for $\rm RI$, $\rm PI$, "
                    f"$N_t/T$ and $N_{{\mathrm{{eff}}}}/T$"
                ),
                label=f"tab:{pair}_temporal_descriptor_pvalue_summary",
                digits=digits,
            )
        )

    replace_folder_in_zip(zip_path, out_dir, arc_folder="latex_tables")

    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    print(f"Generated {len(generated)} files and inserted them into {zip_path} under latex_tables/.")
    return generated


if __name__ == "__main__":
    zip_path = choose_zip()
    export_group_tables(zip_path, top_n=DEFAULT_TOP_N)
