import re
import shutil
from collections import defaultdict
from pathlib import Path
from tkinter import Tk, filedialog
import h5py
import numpy as np
import pandas as pd
from math_utils import nanmad, nanmean, nanmedian, nanpercentile, nanstd, nanvar

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, norm
except ImportError as exc:
    raise ImportError(
        "This script requires scipy for Mann-Whitney tests. Install it with: pip install scipy"
    ) from exc

from input_output.archive_io import replace_folder_in_zip
from input_output.hdf5_io import MetricsTree, iter_h5_arrays

from ..core.grouped_batch import extract_group_name, iter_grouped_h5_files_in_zip


SEGMENT_METRIC_FOLDERS = (
    "/AngioEye/Processing/waveform_shape_metrics_denoised/artery/by_segment/",
    "/AngioEye/Processing/waveform_shape_metrics/artery/by_segment/",
)
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12

DEFAULT_TOP_N = 10
PLOT_STYLE = "default"
PNG_PIL_KWARGS = {"compress_level": 1}

CONTROL_GROUP_PATTERNS = [
    r"^control$",
    r"^controle$",
    r"^controls$",
    r"^ctrl$",
    r"^ctl$",
    r"^BL$",
    r"^healthy$",
    r"^healthy_control$",
    r"^healthy_controls$",
    r"^BL$",
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
]


METRIC_LABELS = {
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
    # "IQR_beat_medseg",
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

SPATIAL_SELECTED_METRICS = [
    "RI",
    "PI",
    "t50_over_T",
    "v_end_over_vbar",
]

TEMPORAL_SELECTED_METRICS = [
    "N_t_over_T",
    "N_eff_over_T",
    "RI",
    "t50_over_T",
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


def iter_segment_metrics(
    h5_path,
    metric_names,
    mode=SEGMENT_MODE,
    *,
    metric_folders=SEGMENT_METRIC_FOLDERS,
):
    """Yield available 3D metric arrays while opening the HDF5 file only once."""
    group_paths = [
        f"{folder.rstrip('/')}/{mode}" for folder in metric_folders
    ]
    yield from iter_h5_arrays(
        h5_path,
        metric_names,
        group_paths=group_paths,
        dtype=float,
        ndim=3,
    )


def extract_segment_metric(h5_path, metric_name, mode=SEGMENT_MODE):
    for _metric_name, arr in iter_segment_metrics(
        h5_path,
        (metric_name,),
        mode=mode,
    ):
        return arr
    return None


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
    q25 = nanpercentile(x, 25)
    q75 = nanpercentile(x, 75)
    return float(q75 - q25)


def mad_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(nanmad(x))


def cv_1d(x, eps=EPS):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    mu = nanmean(x)
    sd = nanstd(x, ddof=1) if x.size > 1 else 0.0
    return float(sd / (np.abs(mu) + eps))


def median_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(nanmedian(x))


def std_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(nanstd(x, ddof=1) if x.size > 1 else 0.0)


def nanmedian_or_nan(x):
    x = np.asarray(x, dtype=float)
    if np.any(np.isfinite(x)):
        return float(nanmedian(x))
    return np.nan


def clean_values(values):
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _compute_axis_statistics(values, axis, eps=EPS):
    """Compute robust statistics for every slice along one matrix axis."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Axis statistics require a two-dimensional array.")

    samples = np.moveaxis(values, axis, -1)
    samples = np.where(np.isfinite(samples), samples, np.nan)
    result_size = samples.shape[0]
    result = {
        name: np.full(result_size, np.nan, dtype=float)
        for name in ("median", "std", "iqr", "mad", "cv")
    }

    if result_size == 0 or samples.shape[1] == 0:
        return result

    counts = np.sum(np.isfinite(samples), axis=1)
    valid = counts > 0
    if not np.any(valid):
        return result

    valid_samples = samples[valid]
    valid_counts = counts[valid]
    medians = nanmedian(valid_samples, axis=1)
    quartiles = nanpercentile(valid_samples, (25, 75), axis=1)
    means = nanmean(valid_samples, axis=1)

    stds = np.zeros(len(valid_samples), dtype=float)
    multiple_values = valid_counts > 1
    if np.any(multiple_values):
        stds[multiple_values] = nanstd(
            valid_samples[multiple_values],
            axis=1,
            ddof=1,
        )

    result["median"][valid] = medians
    result["std"][valid] = stds
    result["iqr"][valid] = quartiles[1] - quartiles[0]
    result["mad"][valid] = nanmedian(
        np.abs(valid_samples - medians[:, np.newaxis]),
        axis=1,
    )
    result["cv"][valid] = stds / (np.abs(means) + eps)
    return result


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

    segment_count = arr.shape[1] * arr.shape[2]
    beat_by_segment = arr.reshape(arr.shape[0], segment_count)
    spatial = _compute_axis_statistics(beat_by_segment, axis=1, eps=eps)
    temporal = _compute_axis_statistics(beat_by_segment, axis=0, eps=eps)

    return {
        "MED_seg_medbeat": nanmedian_or_nan(spatial["median"]),
        "STD_seg_medbeat": nanmedian_or_nan(spatial["std"]),
        "IQR_seg_medbeat": nanmedian_or_nan(spatial["iqr"]),
        "MAD_seg_medbeat": nanmedian_or_nan(spatial["mad"]),
        "CV_seg_medbeat": nanmedian_or_nan(spatial["cv"]),
        "STD_beat_medseg": nanmedian_or_nan(temporal["std"]),
        "IQR_beat_medseg": nanmedian_or_nan(temporal["iqr"]),
        "MAD_beat_medseg": nanmedian_or_nan(temporal["mad"]),
        "CV_beat_medseg": nanmedian_or_nan(temporal["cv"]),
    }


def compute_file_higher_metric_blocks(
    file_path,
    metrics=INPUT_METRICS,
    mode=SEGMENT_MODE,
):
    blocks = {}

    for metric_name, arr in iter_segment_metrics(file_path, metrics, mode=mode):
        high = compute_file_higher_metrics_from_segment_array(arr)
        if high is None:
            continue

        blocks[metric_name] = high

    return blocks


def add_file_blocks_to_results(results, group_name, blocks):
    for metric_name, high in blocks.items():
        for high_name, value in high.items():
            results[group_name][metric_name][high_name].append(value)


def variability_tree_from_blocks(blocks):
    metrics = {}

    for metric_name, high in blocks.items():
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


def write_variability_tree(file_path):
    blocks = compute_file_higher_metric_blocks(file_path)
    return variability_tree_from_blocks(blocks)


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
        blocks = compute_file_higher_metric_blocks(
            grouped_file.file_path,
            metrics=metrics,
            mode=mode,
        )
        add_file_blocks_to_results(results, grouped_file.group_name, blocks)

    return results


def analyze_files(file_paths, output_dir, metrics=INPUT_METRICS, mode=SEGMENT_MODE):
    """
    Analyze already-extracted/processed HDF5 outputs directly.

    This avoids creating a temporary ZIP and repeatedly extracting members during
    AngioEye postprocessing runs.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    records = []
    for file_path in file_paths:
        path = Path(file_path).expanduser().resolve()
        records.append(
            (
                extract_group_name(path.parent, output_dir),
                path.name,
                path,
            )
        )

    records.sort(key=lambda item: (item[0], extract_sort_key(item[1])))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for group_name, _file_name, file_path in records:
        blocks = compute_file_higher_metric_blocks(
            file_path,
            metrics=metrics,
            mode=mode,
        )
        add_file_blocks_to_results(results, group_name, blocks)

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

    mu = nanmean(x)
    sd = nanstd(x, ddof=1) if x.size > 1 else 0.0
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def format_pvalue_latex(value, sig_digits=3, threshold=1e-3):
    """
    Formats p-values for LaTeX tables.

    Examples
    --------
    1.03e-10 -> $1.03 \times 10^{-10}$
    0.0441   -> 0.0441
    """
    if value is None or not np.isfinite(value):
        return "NA"

    value = float(value)

    if value == 0.0:
        return r"$<10^{-300}$"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"${mantissa:.{sig_digits}g} \times 10^{{{exponent}}}$"


def latex_escape_text(value):
    """
    Escapes plain text for LaTeX while leaving math-mode strings untouched.

    This is needed because DataFrame.to_latex(..., escape=False) is used to keep
    metric labels such as $N_t/T$ valid. Therefore, any non-math text containing
    underscores, percent signs, ampersands, etc. must be escaped manually.
    """
    if value is None:
        return ""

    s = str(value)

    # Already math-mode or already a LaTeX command/table fragment: leave unchanged.
    if "$" in s or s.startswith("\\"):
        return s

    replacements = {
        "\\": r"	extbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"	extasciitilde{}",
        "^": r"	extasciicircum{}",
    }

    return "".join(replacements.get(ch, ch) for ch in s)


def metric_label(metric_name):
    return METRIC_LABELS.get(metric_name, latex_escape_text(metric_name))


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
    values = nanmean(matrix, axis=1)
    return clean_values(values)



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
        "mean": float(nanmean(x)),
        "std": float(nanstd(x, ddof=1) if x.size > 1 else 0.0),
        "median": float(nanmedian(x)),
        "iqr": float(nanpercentile(x, 75) - nanpercentile(x, 25)),
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


def build_variability_ranking_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics=INPUT_METRICS,
    n=DEFAULT_TOP_N,
    ascending=False,
    digits=3,
    domain_name="spatial",
):
    """
    Builds the Top-N most/least variable metrics table between one group and control.

    Output columns:
      Rank, Metric, V_group, V_ctrl, V_global

    V_group  = median composite variability score in compared group
    V_ctrl   = median composite variability score in control group
    V_global = mean(V_group, V_ctrl)

    Sorting is done on V_global:
      ascending=False -> most variable
      ascending=True  -> least variable
    """
    rows = []

    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    v_group_col = f"$V^{{{domain_name}}}_{{{group_tex}}}$"
    v_ctrl_col = f"$V^{{{domain_name}}}_{{{control_tex}}}$"
    v_global_col = f"$V^{{{domain_name}}}_{{global}}$"
    for metric_name in metrics:
        x_ctrl = combine_variability_score(
            control_results,
            metric_name,
            higher_metrics,
        )
        x_group = combine_variability_score(
            group_results,
            metric_name,
            higher_metrics,
        )

        s_ctrl = summarize_values(x_ctrl)
        s_group = summarize_values(x_group)

        v_ctrl = s_ctrl["median"]
        v_group = s_group["median"]

        if not np.isfinite(v_ctrl) or not np.isfinite(v_group):
            continue

        v_global = 0.5 * (v_ctrl + v_group)

        rows.append(
            {
                "Metric": metric_label(metric_name),
                v_group_col: v_group,
                v_ctrl_col: v_ctrl,
                v_global_col: v_global,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(v_global_col, ascending=ascending).head(n)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    value_cols = [
        v_group_col,
        v_ctrl_col,
        v_global_col,
    ]

    for col in value_cols:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    return df[
        [
            "Rank",
            "Metric",
            v_group_col,
            v_ctrl_col,
            v_global_col,
        ]
    ]


def build_contrast_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics=INPUT_METRICS,
    n=DEFAULT_TOP_N,
    digits=3,
    domain_name="spatial",
):
    """
    Builds the strongest variability contrast table.

    Output columns:
      Rank, Metric, More variable group, V_group, V_ctrl, Ratio

    Ratio = max(V_group, V_ctrl) / min(V_group, V_ctrl)

    The table is sorted by Ratio in descending order.
    """
    rows = []

    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    v_group_col = f"$V^{{{domain_name}}}_{{{group_tex}}}$"
    v_ctrl_col = f"$V^{{{domain_name}}}_{{{control_tex}}}$"

    for metric_name in metrics:
        x_ctrl = combine_variability_score(
            control_results,
            metric_name,
            higher_metrics,
        )
        x_group = combine_variability_score(
            group_results,
            metric_name,
            higher_metrics,
        )

        s_ctrl = summarize_values(x_ctrl)
        s_group = summarize_values(x_group)

        v_ctrl = s_ctrl["median"]
        v_group = s_group["median"]

        if not np.isfinite(v_ctrl) or not np.isfinite(v_group):
            continue

        if abs(v_ctrl) < EPS and abs(v_group) < EPS:
            ratio = np.nan
        else:
            ratio = max(abs(v_ctrl), abs(v_group)) / (
                min(abs(v_ctrl), abs(v_group)) + EPS
            )

        if not np.isfinite(ratio):
            continue

        more_variable_group = group_tex if v_group >= v_ctrl else control_tex

        rows.append(
            {
                "Metric": metric_label(metric_name),
                "More variable group": more_variable_group,
                v_group_col: v_group,
                v_ctrl_col: v_ctrl,
                "Ratio": ratio,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("Ratio", ascending=False).head(n)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    value_cols = [
        v_group_col,
        v_ctrl_col,
        "Ratio",
    ]

    for col in value_cols:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    return df[
        [
            "Rank",
            "Metric",
            "More variable group",
            v_group_col,
            v_ctrl_col,
            "Ratio",
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
            }
        )

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["Mann-Whitney p-value"])]
    df = df.sort_values("Mann-Whitney p-value", ascending=True)

    if n is not None:
        df = df.head(n)

    for col in [
        f"Median {control_name}",
        f"Median {group_name}",
        "Median difference",
    ]:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    df["Mann-Whitney p-value"] = df["Mann-Whitney p-value"].apply(
        lambda v: format_pvalue_latex(v, sig_digits=digits)
    )

    return df[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            f"Median {control_name}",
            f"Median {group_name}",
            "Median difference",
            "Mann-Whitney p-value",
        ]
    ]


DESCRIPTOR_LABELS = {
    "STD": r"$\mathrm{STD}$",
    "IQR": r"$\mathrm{IQR}$",
    "MAD": r"$\mathrm{MAD}$",
    "CV": r"$\mathrm{CV}$",
}


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
        df[col] = df[col].apply(lambda v: format_pvalue_latex(v, sig_digits=digits))

    return df[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            *[f"{name} p-value" for name in descriptor_map.keys()],
            "Mean p-value",
        ]
    ]


def cohen_d(control_values, group_values):
    """
    Cohen's d using pooled standard deviation.

    Positive values mean that the compared group has a larger mean than control.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size < 2 or y.size < 2:
        return np.nan

    sx = nanstd(x, ddof=1)
    sy = nanstd(y, ddof=1)
    pooled_var = ((x.size - 1) * sx**2 + (y.size - 1) * sy**2) / (x.size + y.size - 2)

    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    return float((nanmean(y) - nanmean(x)) / np.sqrt(pooled_var))


def mean_difference_ci95(control_values, group_values):
    """
    Approximate 95% CI for the mean difference group - control.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan

    diff = float(nanmean(y) - nanmean(x))
    se = np.sqrt(nanvar(x, ddof=1) / x.size + nanvar(y, ddof=1) / y.size)

    if not np.isfinite(se):
        return diff, np.nan, np.nan

    return diff, float(diff - 1.96 * se), float(diff + 1.96 * se)



def auc_from_scores(control_values, group_values):
    """
    ROC AUC computed from Mann-Whitney ranks.

    AUC is oriented so that higher scores predict the compared group.
    If AUC < 0.5, the separability is in the opposite direction; for practical
    discrimination strength, use max(AUC, 1 - AUC).
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan

    try:
        u = mannwhitneyu(y, x, alternative="two-sided", method="auto").statistic
        return float(u / (x.size * y.size))
    except ValueError:
        return np.nan


def best_threshold_sensitivity_specificity_cumulative_sweep(
    control_values,
    group_values,
    *,
    evaluate_both_directions=False,
):
    """Find the Youden-optimal threshold with one sorted cumulative sweep.

    By default, classification direction follows the cohort medians, matching
    ``best_threshold_sensitivity_specificity``. When ``evaluate_both_directions``
    is true, the opposite direction is also evaluated and selected only when its
    Youden index is strictly better.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan, np.nan, np.nan, "NA"

    scores = np.concatenate([x, y])
    labels = np.concatenate(
        [
            np.zeros(x.size, dtype=np.int64),
            np.ones(y.size, dtype=np.int64),
        ]
    )
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    values, starts, counts = np.unique(
        sorted_scores,
        return_index=True,
        return_counts=True,
    )

    if values.size == 1:
        return float(values[0]), np.nan, np.nan, "NA"

    group_counts = np.add.reduceat(sorted_labels, starts)
    control_counts = counts - group_counts
    cumulative_group = np.cumsum(group_counts)[:-1]
    cumulative_control = np.cumsum(control_counts)[:-1]
    thresholds = (values[:-1] + values[1:]) / 2.0

    def best_for_direction(direction):
        if direction == ">=":
            true_positive = y.size - cumulative_group
            true_negative = cumulative_control
        else:
            true_positive = cumulative_group
            true_negative = x.size - cumulative_control

        sensitivity = true_positive / y.size
        specificity = true_negative / x.size
        youden = sensitivity + specificity - 1.0
        best_index = int(np.argmax(youden))
        return (
            float(youden[best_index]),
            float(thresholds[best_index]),
            float(sensitivity[best_index]),
            float(specificity[best_index]),
            direction,
        )

    preferred_direction = ">=" if nanmedian(y) >= nanmedian(x) else "<="
    best = best_for_direction(preferred_direction)

    if evaluate_both_directions:
        opposite_direction = "<=" if preferred_direction == ">=" else ">="
        opposite = best_for_direction(opposite_direction)
        if opposite[0] > best[0]:
            best = opposite

    _, threshold, sensitivity, specificity, direction = best
    return threshold, sensitivity, specificity, direction


def overlap_from_cohen_d(d):
    """
    Gaussian equal-variance overlap approximation: OVL = 2 Phi(-|d|/2).
    """
    if d is None or not np.isfinite(d):
        return np.nan
    return float(2.0 * norm.cdf(-abs(float(d)) / 2.0))


def format_decision_rule(threshold, direction, group_name, digits=4):
    """
    Formats the optimal threshold as a readable LaTeX decision rule.
    Example output: score <= threshold -> group, with LaTeX symbols.
    """
    if direction == "NA" or threshold is None or not np.isfinite(threshold):
        return "NA"

    bs = chr(92)
    op = "$" + bs + "geq$" if direction == ">=" else "$" + bs + "leq$"
    threshold_str = format_float(threshold, digits=digits)
    group_str = latex_escape_text(group_name)
    return f"score {op} {threshold_str} $" + bs + f"rightarrow$ {group_str}"


def build_group_separation_metrics_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics,
    digits=4,
    evaluate_both_directions=False,
):
    """
    Builds a transposed statistical interpretation table for selected composite
    variability scores.

    Layout is optimized for Overleaf:
        rows    = statistical estimators
        columns = selected metrics, e.g. RI / PI or N_t/T / N_eff/T
    """
    metric_results = {}
    bs = chr(92)

    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)

    n_control_label = f"$n_{{{bs}mathrm{{{control_tex}}}}}$"
    n_group_label = f"$n_{{{bs}mathrm{{{group_tex}}}}}$"
    ci_label = "Mean difference 95" + bs + "% CI"

    for metric_name in metrics:
        x = combine_variability_score(
            control_results,
            metric_name,
            higher_metrics=higher_metrics,
        )
        y = combine_variability_score(
            group_results,
            metric_name,
            higher_metrics=higher_metrics,
        )

        sx = summarize_values(x)
        sy = summarize_values(y)
        p = mann_whitney_pvalue(x, y)
        d = cohen_d(x, y)
        diff, ci_low, ci_high = mean_difference_ci95(x, y)
        auc = auc_from_scores(x, y)
        auc_sep = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        threshold, sensitivity, specificity, direction = (
            best_threshold_sensitivity_specificity_cumulative_sweep(
                x,
                y,
                evaluate_both_directions=evaluate_both_directions,
            )
        )
        ovl = overlap_from_cohen_d(d)

        more_variable_group = group_tex if sy["median"] > sx["median"] else control_tex
        decision_rule = format_decision_rule(
            threshold,
            direction,
            group_name=group_name,
            digits=digits,
        )

        metric_results[metric_label(metric_name)] = {
            n_control_label: str(sx["n"]),
            n_group_label: str(sy["n"]),
            f"Median {control_tex}": format_float(sx["median"], digits=digits),
            f"Median {group_tex}": format_float(sy["median"], digits=digits),
            "More variable group": more_variable_group,
            "Mann--Whitney p-value": format_pvalue_latex(p, sig_digits=digits),
            "Cohen's $d$": format_float(d, digits=digits),
            ci_label: (
                f"[{format_float(ci_low, digits=digits)}, "
                f"{format_float(ci_high, digits=digits)}]"
            ),
            "AUC separability": format_float(auc_sep, digits=digits),
            "Sensitivity": format_float(sensitivity, digits=digits),
            "Specificity": format_float(specificity, digits=digits),
            "Overlap OVL": format_float(ovl, digits=digits),
        }

    estimator_order = [
        n_control_label,
        n_group_label,
        f"Median {control_tex}",
        f"Median {group_tex}",
        "More variable group",
        "Mann--Whitney p-value",
        "Cohen's $d$",
        ci_label,
        "AUC separability",
        "Sensitivity",
        "Specificity",
        "Overlap OVL",
    ]

    rows = []
    for estimator in estimator_order:
        row = {"Estimator": estimator}
        for metric_col, values in metric_results.items():
            row[metric_col] = values.get(estimator, "NA")
        rows.append(row)

    return pd.DataFrame(rows)


def build_auc_separability_ranking_table(
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics=INPUT_METRICS,
    digits=4,
    evaluate_both_directions=False,
):
    """
    Ranks all metrics by AUC separability for the composite variability score.

    AUC separability is max(AUC, 1 - AUC), so it measures separation strength
    independently of direction. The direction is reported through the more variable
    group, Cohen's d and the mean difference.
    """
    rows = []
    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)

    for metric_name in metrics:
        x = combine_variability_score(
            control_results,
            metric_name,
            higher_metrics=higher_metrics,
        )
        y = combine_variability_score(
            group_results,
            metric_name,
            higher_metrics=higher_metrics,
        )

        sx = summarize_values(x)
        sy = summarize_values(y)

        if sx["n"] == 0 or sy["n"] == 0:
            continue

        p = mann_whitney_pvalue(x, y)
        d = cohen_d(x, y)
        auc = auc_from_scores(x, y)
        auc_sep = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        threshold, sensitivity, specificity, direction = (
            best_threshold_sensitivity_specificity_cumulative_sweep(
                x,
                y,
                evaluate_both_directions=evaluate_both_directions,
            )
        )
        more_variable_group = group_tex if sy["median"] > sx["median"] else control_tex


        rows.append(
            {
                "Metric": metric_label(metric_name),
                f"Median variability {control_tex}": sx["median"],
                f"Median variability {group_tex}": sy["median"],
                "More variable group": more_variable_group,
                "AUC separability": auc_sep,
                "Mann--Whitney p-value": p,
                "Cohen's $d$": d,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[np.isfinite(df["AUC separability"])]
    df = df.sort_values("AUC separability", ascending=False)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    numeric_cols = [
        f"Median variability {control_tex}",
        f"Median variability {group_tex}",
        "AUC separability",
        "Cohen's $d$",
        "Sensitivity",
        "Specificity",
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(lambda v: format_float(v, digits=digits))

    df["Mann--Whitney p-value"] = df["Mann--Whitney p-value"].apply(
        lambda v: format_pvalue_latex(v, sig_digits=digits)
    )

    return df[
        [
            "Rank",
            "Metric",
            f"Median variability {control_tex}",
            f"Median variability {group_tex}",
            "More variable group",
            "AUC separability",
            "Mann--Whitney p-value",
            "Cohen's $d$",
            "Sensitivity",
            "Specificity",
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
    # "IQR": "IQR_beat_medseg",
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

            fig, ax = plt.subplots(
                figsize=(max(6, 1.2 * len(group_values)), 4.5),
                layout="constrained",
            )

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

            filename = (
                f"{safe_name(domain_name)}_"
                f"{safe_name(descriptor_name)}_"
                f"{safe_name(metric_name)}_by_group.png"
            )
            path = out_dir / filename
            fig.savefig(path, dpi=dpi, pil_kwargs=PNG_PIL_KWARGS)
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
        r"\centering",
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
        df, caption=caption, label=label, font_size=r"\scriptsize"
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return [csv_path, tex_path]

def pretty_table_title(csv_file):
    """
    Converts generated CSV filenames into readable HTML titles.
    """

    name = csv_file.stem

    name = name.replace("_", " ")

    # AUC tables
    if "spatial auc separability ranking all metrics" in name:
        groups = name.split(" spatial auc")[0]
        groups = groups.replace(" vs ", " and ")

        return (
            f"Spatial variability metrics between {groups}, "
            "ranked by AUC separability"
        )

    if "temporal auc separability ranking all metrics" in name:
        groups = name.split(" temporal auc")[0]
        groups = groups.replace(" vs ", " and ")

        return (
            f"Temporal variability metrics between {groups}, "
            "ranked by AUC separability"
        )


    # strongest contrast
    if "strongest spatial variability contrast" in name:
        groups = name.split(" strongest")[0]
        groups = groups.replace(" vs ", " and ")

        return (
            f"Top 10 strongest spatial variability contrasts between {groups}"
        )


    if "strongest temporal variability contrast" in name:
        groups = name.split(" strongest")[0]
        groups = groups.replace(" vs ", " and ")

        return (
            f"Top 10 strongest temporal variability contrasts between {groups}"
        )


    # most / least
    if "n most spatially variable metrics" in name:
        group = name.split(" vs ")[0]

        return (
            f"Top 10 most spatially variable metrics in group {group}"
        )


    if "n least spatially variable metrics" in name:
        group = name.split(" vs ")[0]

        return (
            f"Top 10 least spatially variable metrics in group {group}"
        )


    if "n most temporally variable metrics" in name:
        group = name.split(" vs ")[0]

        return (
            f"Top 10 most temporally variable metrics in group {group}"
        )


    if "n least temporally variable metrics" in name:
        group = name.split(" vs ")[0]

        return (
            f"Top 10 least temporally variable metrics in group {group}"
        )


    # raw tables
    if "spatial variability table" in name:
        group = name.replace(" spatial variability table", "")

        return (
            f"Raw spatial variability metrics for group {group}"
        )


    if "temporal variability table" in name:
        group = name.replace(" temporal variability table", "")

        return (
            f"Raw temporal variability metrics for group {group}"
        )


    return name

def comparison_order(csv_file):
    name = csv_file.stem.lower()

    if "most_spatially_variable" in name or "most_temporally_variable" in name:
        return 0

    if "least_spatially_variable" in name or "least_temporally_variable" in name:
        return 1

    if "strongest_spatial_variability_contrast" in name:
        return 2

    if "strongest_temporal_variability_contrast" in name:
        return 2

    if "spatial_auc_separability" in name:
        return 3

    if "temporal_auc_separability" in name:
        return 3

    return 99

def card_header(csv_file):
    """
    Returns the short title displayed in bold on each dashboard card.
    """

    name = csv_file.stem.lower()

    # ---------- Raw ----------
    if "spatial_variability_table" in name:
        group = name.replace("_spatial_variability_table", "")
        return f"Raw - {group.replace('_', ' ').title()}"

    if "temporal_variability_table" in name:
        group = name.replace("_temporal_variability_table", "")
        return f"Raw - {group.replace('_', ' ').title()}"

    # ---------- Spatial ----------
    if "most_spatially_variable" in name:
        return "Most spatially"

    if "least_spatially_variable" in name:
        return "Least spatially"

    if "strongest_spatial_variability_contrast" in name:
        return "Strongest contrast"

    if "spatial_auc_separability" in name:
        return "AUC separability"

    # ---------- Temporal ----------
    if "most_temporally_variable" in name:
        return "Most temporally"

    if "least_temporally_variable" in name:
        return "Least temporally"

    if "strongest_temporal_variability_contrast" in name:
        return "Strongest contrast"

    if "temporal_auc_separability" in name:
        return "AUC separability"

    return "Table"

def save_html_report(output_dir, title="Variability Report"):
    """
    Creates one HTML page containing all generated CSV tables
    with LaTeX rendering using MathJax.
    """

    output_dir = Path(output_dir)

    html_path = output_dir / "variability_report.html"

    csv_files = sorted(output_dir.rglob("*.csv"))

    spatial_raw = []
    spatial_cmp = []

    temporal_raw = []
    temporal_cmp = []

    for f in csv_files:

        name = f.stem.lower()

        if "spatial" in name:

            if "variability_table" in name:
                spatial_raw.append(f)
            else:
                spatial_cmp.append(f)

        elif "temporal" in name:

            if "variability_table" in name:
                temporal_raw.append(f)
            else:
                temporal_cmp.append(f)

    sections = []
    spatial_cmp.sort(key=comparison_order)
    temporal_cmp.sort(key=comparison_order)

    sections.append(
        f"""
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<title>{title}</title>

<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
    }},
    svg: {{
        fontCache: 'global'
    }}
}};
</script>

<script 
src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>


<style>

body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    background: #fafafa;
}}

h1 {{
    text-align:center;
    color:#000000;
}}

h2 {{
    font-size:24px;
    font-weight:700;
    margin-top:45px;
    margin-bottom:15px;
    border-bottom:2px solid #d9d9d9;
    padding-bottom:8px;
}}

h3 {{
    font-size:17px;
    font-weight:600;
    color:#555;
    margin-top:25px;
    margin-bottom:12px;
}}

table {{
    border-collapse: collapse;
    width:100%;
    background:white;
    margin-bottom:40px;
}}

th {{
    background:#e6e6e6;
    color:black;
    padding:8px;
    text-align:center;
    vertical-align:middle;
}}

td {{
    border:1px solid #ddd;
    padding:6px;
    text-align:center;
    vertical-align:middle;
}}

tr:nth-child(even) {{
    background:#f5f5f5;
}}

.container {{
    overflow-x:auto;
}}


/* ===== DASHBOARD MENU ===== */

.dashboard-grid {{
    display:grid;
    grid-template-columns:repeat(
        auto-fit,
        minmax(300px,1fr)
    );
    gap:18px;
    margin-top:25px;
    margin-bottom:50px;
}}


.dashboard-card {{
    background: #fff;
    border: 1px solid #d8d8d8;
    border-radius: 10px;

    padding: 18px;

    text-decoration: none;
    color: #222;

    box-shadow: 0 2px 5px rgba(0,0,0,.08);

    transition: all .2s ease;
}}

.dashboard-card:hover {{
    transform: translateY(-2px);
    border-color: #999;
    box-shadow: 0 5px 12px rgba(0,0,0,.15);
}}


.card-type {{
    font-size:15px;
    font-weight:700;
    color:#111;
    margin-bottom:5px;

}}


.card-title {{
    font-size:14px;
    line-height:1.45;
    color:#666;
}}


.scroll-top {{
    position:fixed;

    right:25px;
    bottom:25px;

    width:45px;
    height:45px;

    background:#e6e6e6;
    color:#000000;

    border-radius:50%;

    display:flex;
    align-items:center;
    justify-content:center;

    text-decoration:none;

    font-size:26px;
    font-weight:bold;

    box-shadow:
        0 3px 10px rgba(0,0,0,0.25);

    transition:0.2s;
}}


.scroll-top:hover {{

    transform:translateY(-4px);

    background:#d0d0d0;
}}

</style>


</head>

<body id="top">

<h1>{title}</h1>


"""
    )

    # Dashboard table of contents

    sections.append("<h2>SPATIAL</h2>")

    sections.append("<h3>Raw tables</h3>")
    sections.append('<div class="dashboard-grid">')

    index = 0

    for csv_file in spatial_raw:

        sections.append(f"""
    <a class="dashboard-card" href="#table{index}">

    <div class="card-type">
    <b>{card_header(csv_file)}</b>
    </div>

    <div class="card-title">
    {pretty_table_title(csv_file)}
    </div>

    </a>
    """)

        index += 1

    sections.append("</div>")


    sections.append("<h3>Comparison tables</h3>")
    sections.append('<div class="dashboard-grid">')

    for csv_file in spatial_cmp:

        sections.append(f"""
    <a class="dashboard-card" href="#table{index}">

    <div class="card-type">
    <b>{card_header(csv_file)}</b>
    </div>

    <div class="card-title">
    {pretty_table_title(csv_file)}
    </div>

    </a>
    """)

        index += 1

    sections.append("</div>")


    sections.append("<h2>TEMPORAL</h2>")

    sections.append("<h3>Raw tables</h3>")
    sections.append('<div class="dashboard-grid">')

    for csv_file in temporal_raw:

        sections.append(f"""
    <a class="dashboard-card" href="#table{index}">

    <div class="card-type">
    <b>{card_header(csv_file)}</b>
    </div>

    <div class="card-title">
    {pretty_table_title(csv_file)}
    </div>

    </a>
    """)

        index += 1

    sections.append("</div>")


    sections.append("<h3>Comparison tables</h3>")
    sections.append('<div class="dashboard-grid">')

    for csv_file in temporal_cmp:

        sections.append(f"""
    <a class="dashboard-card" href="#table{index}">

    <div class="card-type">
    <b>{card_header(csv_file)}</b>
    </div>

    <div class="card-title">
    {pretty_table_title(csv_file)}
    </div>

    </a>
    """)

        index += 1

    sections.append("</div>")



    # Tables
    ordered_files = (
        spatial_raw +
        spatial_cmp +
        temporal_raw +
        temporal_cmp
    )

    for i, csv_file in enumerate(ordered_files):

        df = pd.read_csv(csv_file)

        html_table = df.to_html(
            index=False,
            escape=False
        )

        table_title = pretty_table_title(csv_file)

        sections.append(
            f"""

<h2 id="table{i}">
{table_title}
</h2>

<div class="container">

{html_table}

</div>

"""
        )


    sections.append(
        """

<script>
MathJax.typeset();
</script>


<a href="#top" class="scroll-top">
↑
</a>


</body>
</html>
"""
    )


    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(sections))


    print(
        "HTML dashboard created:",
        html_path
    )

    return html_path
# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------


def export_group_tables_from_results(
    results,
    output_dir,
    metrics=INPUT_METRICS,
    digits=3,
    top_n=DEFAULT_TOP_N,
    idle_callback=None,
    evaluate_both_directions=True,
):
    """
    Creates table and figure files in output_dir:

    spatial/raw, spatial/comparisons_vs_control, temporal/raw,
    temporal/comparisons_vs_control, and figure subfolders.
    """
    out_dir = Path(output_dir)

    spatial_raw_dir = out_dir / "spatial" / "raw"
    temporal_raw_dir = out_dir / "temporal" / "raw"
    spatial_cmp_dir = out_dir / "spatial" / "comparisons_vs_control"
    temporal_cmp_dir = out_dir / "temporal" / "comparisons_vs_control"
    spatial_fig_dir = out_dir / "spatial" / "figures"
    temporal_fig_dir = out_dir / "temporal" / "figures"

    plt.style.use(PLOT_STYLE)

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
            metrics=SPATIAL_SELECTED_METRICS,
        )
    )
    if idle_callback is not None:
        idle_callback()
    generated.extend(
        export_variability_value_plots(
            results,
            temporal_fig_dir,
            descriptor_map=TEMPORAL_DESCRIPTOR_MAP,
            domain_name="temporal",
            metrics=TEMPORAL_SELECTED_METRICS,
        )
    )
    if idle_callback is not None:
        idle_callback()

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
                caption=f"Raw spatial variability metrics for group {latex_escape_text(group_name)}",
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
                caption=f"Raw temporal variability metrics for group {latex_escape_text(group_name)}",
                label=f"tab:{safe_group}_temporal_variability_raw",
                digits=digits,
            )
        )
        if idle_callback is not None:
            idle_callback()

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
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            ascending=False,
            digits=digits,
            domain_name="spatial",
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_n_most_spatially_variable_metrics.csv",
                spatial_cmp_dir / f"{pair}_n_most_spatially_variable_metrics.tex",
                caption=f"Top {top_n} most spatially variable metrics in group {latex_escape_text(group_name)}",
                label=f"tab:{pair}_most_spatially_variable",
                digits=digits,
            )
        )
        if idle_callback is not None:
            idle_callback()

        df = build_variability_ranking_table(
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            ascending=True,
            digits=digits,
            domain_name="spatial",
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_n_least_spatially_variable_metrics.csv",
                spatial_cmp_dir / f"{pair}_n_least_spatially_variable_metrics.tex",
                caption=f"Top {top_n} least spatially variable metrics in group {latex_escape_text(group_name)}",
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
            domain_name="spatial",
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_strongest_spatial_variability_contrast.csv",
                spatial_cmp_dir / f"{pair}_strongest_spatial_variability_contrast.tex",
                caption=f"Top {top_n} strongest spatial variability contrasts between {latex_escape_text(group_name)} and {latex_escape_text(control_group)}",
                label=f"tab:{pair}_strongest_spatial_contrast",
                digits=digits,
            )
        )

        df = build_descriptor_pvalue_summary_table(
            control_results,
            group_results,
            descriptor_map=SPATIAL_DESCRIPTOR_MAP,
            control_name=control_group,
            group_name=group_name,
            metrics=SPATIAL_SELECTED_METRICS,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_spatial_descriptor_pvalue_summary_RI_PI.csv",
                spatial_cmp_dir / f"{pair}_spatial_descriptor_pvalue_summary_RI_PI.tex",
                caption=(
                    f"Spatial descriptor-specific Mann-Whitney p-values between "
                    f"{control_group} and {group_name}"
                ),
                label=f"tab:{pair}_spatial_descriptor_pvalue_summary",
                digits=digits,
            )
        )

        df = build_group_separation_metrics_table(
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=SPATIAL_SELECTED_METRICS,
            digits=digits,
            evaluate_both_directions=evaluate_both_directions,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir / f"{pair}_spatial_group_separation_metrics_RI_PI.csv",
                spatial_cmp_dir / f"{pair}_spatial_group_separation_metrics_RI_PI.tex",
                caption=(
                    f"Spatial group-separation metrics between {control_group} and "
                    f"{group_name}"
                ),
                label=f"tab:{pair}_spatial_group_separation_metrics",
                digits=digits,
            )
        )

        df = build_auc_separability_ranking_table(
            control_results,
            group_results,
            higher_metrics=SPATIAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            digits=digits,
            evaluate_both_directions=evaluate_both_directions,
        )
        generated.extend(
            save_table(
                df,
                spatial_cmp_dir
                / f"{pair}_spatial_auc_separability_ranking_all_metrics.csv",
                spatial_cmp_dir
                / f"{pair}_spatial_auc_separability_ranking_all_metrics.tex",
                caption=(
                    f"Spatial variability metrics between {latex_escape_text(control_group)} "
                    f"and {latex_escape_text(group_name)}, ranked by AUC separability"
                ),
                label=f"tab:{pair}_spatial_auc_separability_ranking",
                digits=digits,
            )
        )

        # ------------------------------
        # Temporal comparison tables
        # ------------------------------
        df = build_variability_ranking_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            ascending=False,
            digits=digits,
            domain_name="temporal",
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_n_most_temporally_variable_metrics.csv",
                temporal_cmp_dir / f"{pair}_n_most_temporally_variable_metrics.tex",
                caption=f"Top {top_n} most temporally variable metrics in group {latex_escape_text(group_name)}",
                label=f"tab:{pair}_most_temporally_variable",
                digits=digits,
            )
        )

        df = build_variability_ranking_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            ascending=True,
            digits=digits,
            domain_name="temporal",
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_n_least_temporally_variable_metrics.csv",
                temporal_cmp_dir / f"{pair}_n_least_temporally_variable_metrics.tex",
                caption=f"Top {top_n} least temporally variable metrics in group {latex_escape_text(group_name)}",
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
            domain_name="temporal",
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir
                / f"{pair}_strongest_temporal_variability_contrast.csv",
                temporal_cmp_dir
                / f"{pair}_strongest_temporal_variability_contrast.tex",
                caption=f"Top {top_n} strongest temporal variability contrasts between {latex_escape_text(group_name)} and {latex_escape_text(control_group)}",
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
        temporal_mannwhitney_caption = (
            "Best temporal variability metrics between "
            f"{latex_escape_text(control_group)} and {latex_escape_text(group_name)}, "
            "ranked by Mann-Whitney p-value"
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir / f"{pair}_best_temporal_variability_mannwhitney.csv",
                temporal_cmp_dir / f"{pair}_best_temporal_variability_mannwhitney.tex",
                caption=temporal_mannwhitney_caption,
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
            metrics=TEMPORAL_SELECTED_METRICS,
            digits=digits,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir
                / f"{pair}_temporal_descriptor_pvalue_summary_Nt_Neff.csv",
                temporal_cmp_dir
                / f"{pair}_temporal_descriptor_pvalue_summary_Nt_Neff.tex",
                caption=(
                    f"Temporal descriptor-specific Mann-Whitney p-values between "
                    f"{control_group} and {group_name} "
                ),
                label=f"tab:{pair}_temporal_descriptor_pvalue_summary",
                digits=digits,
            )
        )

        df = build_group_separation_metrics_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=TEMPORAL_SELECTED_METRICS,
            digits=digits,
            evaluate_both_directions=evaluate_both_directions,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir
                / f"{pair}_temporal_group_separation_metrics_Nt_Neff.csv",
                temporal_cmp_dir
                / f"{pair}_temporal_group_separation_metrics_Nt_Neff.tex",
                caption=(
                    f"Temporal group-separation metrics between {control_group} and "
                    f"{group_name}"
                ),
                label=f"tab:{pair}_temporal_group_separation_metrics",
                digits=digits,
            )
        )

        df = build_auc_separability_ranking_table(
            control_results,
            group_results,
            higher_metrics=TEMPORAL_VARIABILITY_COLUMNS,
            control_name=control_group,
            group_name=group_name,
            metrics=metrics,
            digits=digits,
            evaluate_both_directions=evaluate_both_directions,
        )
        generated.extend(
            save_table(
                df,
                temporal_cmp_dir
                / f"{pair}_temporal_auc_separability_ranking_all_metrics.csv",
                temporal_cmp_dir
                / f"{pair}_temporal_auc_separability_ranking_all_metrics.tex",
                caption=(
                    f"Temporal variability metrics between {latex_escape_text(control_group)} "
                    f"and {latex_escape_text(group_name)}, ranked by AUC separability"
                ),
                label=f"tab:{pair}_temporal_auc_separability_ranking",
                digits=digits,
            )
        )
        if idle_callback is not None:
            idle_callback()

    print(
        f"Generated {len(generated)} variability/heterogeneity file(s) in {out_dir}."
    )

    html_file = save_html_report(out_dir)
    generated.append(html_file)

    return generated


def export_group_tables(
    zip_path,
    metrics=INPUT_METRICS,
    mode=SEGMENT_MODE,
    digits=3,
    top_n=DEFAULT_TOP_N,
    evaluate_both_directions=False,
):
    """
    Backward-compatible ZIP export entry point.
    """
    zip_path = Path(zip_path)
    out_dir = zip_path.parent / "Variability and heterogeneity"
    results = analyze_zip(zip_path, metrics=metrics, mode=mode)
    generated = export_group_tables_from_results(
        results,
        out_dir,
        metrics=metrics,
        digits=digits,
        top_n=top_n,
        evaluate_both_directions=evaluate_both_directions,
    )

    replace_folder_in_zip(zip_path, out_dir, arc_folder="Variability and heterogeneity")

    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    print(
        f"Generated {len(generated)} files and inserted them into {zip_path} under Variability and heterogeneity/."
    )
    return generated


if __name__ == "__main__":
    zip_path = choose_zip()
    export_group_tables(zip_path, top_n=DEFAULT_TOP_N)
