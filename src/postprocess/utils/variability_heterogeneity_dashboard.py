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
from angioeye_io.hdf5_io import MetricsTree


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

    # --------------------------------------------------
    # 1. Spatial variability across segments, per beat
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. Temporal variability across beats, per segment
    # --------------------------------------------------
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
        # Central level
        "MED_seg_medbeat": nanmedian_or_nan(beat_median),
        # Spatial variability
        "STD_seg_medbeat": nanmedian_or_nan(beat_std),
        "IQR_seg_medbeat": nanmedian_or_nan(beat_iqr),
        "MAD_seg_medbeat": nanmedian_or_nan(beat_mad),
        "CV_seg_medbeat": nanmedian_or_nan(beat_cv_seg),
        # Temporal variability
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


def format_mean_std(values, digits=3):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return "NA"

    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0

    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


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
            "Metric": METRIC_LABELS.get(
                metric_name,
                metric_name.replace("_", r"\_"),
            )
        }

        for high_name in selected_higher_metrics:
            vals = metric_block.get(high_name, [])
            row[COLUMN_LABELS[high_name]] = format_mean_std(vals, digits=digits)

        rows.append(row)

    return pd.DataFrame(rows)


def build_spatial_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    spatial_columns = [
        "MED_seg_medbeat",
        "STD_seg_medbeat",
        "IQR_seg_medbeat",
        "MAD_seg_medbeat",
        "CV_seg_medbeat",
    ]

    return build_group_table_with_columns(
        results_for_group=results_for_group,
        selected_higher_metrics=spatial_columns,
        metrics=metrics,
        digits=digits,
    )


def build_temporal_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    temporal_columns = [
        "MED_seg_medbeat",
        "STD_beat_medseg",
        "IQR_beat_medseg",
        "MAD_beat_medseg",
        "CV_beat_medseg",
    ]

    return build_group_table_with_columns(
        results_for_group=results_for_group,
        selected_higher_metrics=temporal_columns,
        metrics=metrics,
        digits=digits,
    )


def dataframe_to_latex_table(
    df,
    caption=None,
    label=None,
    font_size=r"\scriptsize",
    max_width=r"\textwidth",
):
    """
    Requires in Overleaf preamble:
        \\usepackage{float}
        \\usepackage{booktabs}
        \\usepackage{adjustbox}
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

    lines.append(rf"\begin{{adjustbox}}{{max width={max_width}}}")
    lines.append(latex_tabular)
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def export_group_tables(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE, digits=3):
    out_dir = Path(zip_path).parent / "latex_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_zip(zip_path, metrics=metrics, mode=mode)

    print("Groups found:", list(results.keys()))

    generated = []

    for group_name in sorted(results.keys()):
        print("Building spatial and temporal tables for group:", group_name)

        safe_group = re.sub(r"[^A-Za-z0-9_-]+", "_", group_name)

        # --------------------------------------------------
        # Spatial variability table
        # --------------------------------------------------
        df_spatial = build_spatial_group_table(
            results[group_name],
            metrics=metrics,
            digits=digits,
        )

        spatial_csv_path = out_dir / f"{safe_group}_spatial_variability_table.csv"
        spatial_tex_path = out_dir / f"{safe_group}_spatial_variability_table.tex"

        df_spatial.to_csv(spatial_csv_path, index=False)

        spatial_latex = dataframe_to_latex_table(
            df_spatial,
            caption=f"Spatial variability metrics for group {group_name}",
            label=f"tab:{safe_group}_spatial_variability",
            font_size=r"\scriptsize",
            max_width=r"\textwidth",
        )

        with open(spatial_tex_path, "w", encoding="utf-8") as f:
            f.write(spatial_latex)

        # --------------------------------------------------
        # Temporal variability table
        # --------------------------------------------------
        df_temporal = build_temporal_group_table(
            results[group_name],
            metrics=metrics,
            digits=digits,
        )

        temporal_csv_path = out_dir / f"{safe_group}_temporal_variability_table.csv"
        temporal_tex_path = out_dir / f"{safe_group}_temporal_variability_table.tex"

        df_temporal.to_csv(temporal_csv_path, index=False)

        temporal_latex = dataframe_to_latex_table(
            df_temporal,
            caption=f"Temporal variability metrics for group {group_name}",
            label=f"tab:{safe_group}_temporal_variability",
            font_size=r"\scriptsize",
            max_width=r"\textwidth",
        )

        with open(temporal_tex_path, "w", encoding="utf-8") as f:
            f.write(temporal_latex)

        generated.extend(
            [
                spatial_csv_path,
                spatial_tex_path,
                temporal_csv_path,
                temporal_tex_path,
            ]
        )

    replace_folder_in_zip(zip_path, out_dir, arc_folder="latex_tables")

    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    return generated


if __name__ == "__main__":
    zip_path = choose_zip()

    export_group_tables(zip_path)
