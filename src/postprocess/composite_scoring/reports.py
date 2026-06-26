from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Iterable

from .dataclasses import MetricContributionRecord
from .optimal_split import SplitStats


METRIC_LATEX_NAMES: dict[str, str] = {
    "mu_t_over_T": r"$\mu_t/T$",
    "sigma_t_over_T": r"$\sigma_t/T$",
    "gamma_t": r"$\gamma_t$",
    "t10_over_T": r"$t_{10}/T$",
    "t25_over_T": r"$t_{25}/T$",
    "t50_over_T": r"$t_{50}/T$",
    "t75_over_T": r"$t_{75}/T$",
    "t90_over_T": r"$t_{90}/T$",
    "Qt_width": r"$Q_{t,\mathrm{width}}$",
    "Qt_skew": r"$Q_{t,\mathrm{skew}}$",
    "Delta_DTI": r"$\Delta_{\mathrm{DTI}}$",
    "d10_over_D": r"$d_{10}/D$",
    "d25_over_D": r"$d_{25}/D$",
    "d50_over_D": r"$d_{50}/D$",
    "d75_over_D": r"$d_{75}/D$",
    "d90_over_D": r"$d_{90}/D$",
    "Qd_width": r"$Q_{d,\mathrm{width}}$",
    "Qd_skew": r"$Q_{d,\mathrm{skew}}$",
    "R_VTI": r"$R_{\mathrm{VTI}}$",
    "SF_VTI": r"$\mathrm{SF}_{\mathrm{VTI}}$",
    "W50_over_T": r"$W_{50}/T$",
    "W80_over_T": r"$W_{80}/T$",
    "RI": r"$\mathrm{RI}$",
    "PI": r"$\mathrm{PI}$",
    "CF": r"$\mathrm{CF}$",
    "tmax_over_T": r"$t_{\max}/T$",
    "tmin_over_T": r"$t_{\min}/T$",
    "Srise": r"$S_{\mathrm{rise}}$",
    "Sfall": r"$S_{\mathrm{fall}}$",
    "trise_over_T": r"$t_{\mathrm{rise}}/T$",
    "tfall_over_T": r"$t_{\mathrm{fall}}/T$",
    "Eslope": r"$E_{\mathrm{slope}}$",
    "v_end_over_vbar": r"$\bar{v}_{\mathrm{end}}/\bar{v}$",
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",
    "N_t_over_T": r"$N_t/T$",
    "E_LF_over_E_HF": r"$E_{\mathrm{LF}}/E_{\mathrm{HF}}$",
    "E_low_over_E_total": r"$E_{\mathrm{low}}/E_{\mathrm{total}}$",
    "eta_h": r"$\eta_h$",
}


def write_optimal_split_report(
    split_stats: Iterable[SplitStats],
    output_dir: Path,
) -> list[str]:
    report_dir = output_dir / "composite_scoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(stat) for stat in split_stats]
    csv_path = report_dir / "optimal_split_calibration.csv"
    _write_csv(csv_path, rows)
    return [str(csv_path)]


def write_selected_metric_panel_report(
    selected_split_stats: Iterable[SplitStats],
    output_dir: Path,
) -> list[str]:
    """Write the final metric panel actually used to compute WAS/WAS-c."""
    report_dir = output_dir / "composite_scoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, stat in enumerate(selected_split_stats, start=1):
        row = asdict(stat)
        row["rank_by_separability_auc"] = rank
        rows.append(row)

    csv_path = report_dir / "selected_metric_panel_top_auc.csv"
    tex_path = report_dir / "selected_metric_panel_top_auc.tex"
    _write_csv(csv_path, rows)
    _write_selected_metric_panel_tex(tex_path, rows)
    return [str(csv_path), str(tex_path)]


def write_metric_contribution_reports(
    contribution_records: Iterable[MetricContributionRecord],
    split_stats: Iterable[SplitStats],
    output_dir: Path,
) -> list[str]:
    """Write detailed and summarized reports of metric usefulness.

    A metric is counted as active for a subject when z > 0, meaning it crossed
    the automatically selected threshold in the pathological direction.
    """
    report_dir = output_dir / "composite_scoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = [asdict(record) for record in contribution_records]
    detail_csv = report_dir / "metric_contributions_by_subject.csv"
    _write_csv(detail_csv, detail_rows)

    summary_rows = _summarize_contributions(detail_rows, split_stats)
    summary_csv = report_dir / "metric_usefulness_summary.csv"
    _write_csv(summary_csv, summary_rows)

    return [str(detail_csv), str(summary_csv)]


def _summarize_contributions(
    rows: list[dict],
    split_stats: Iterable[SplitStats],
) -> list[dict]:
    split_by_metric = {stat.metric_key: asdict(stat) for stat in split_stats}
    groups: dict[tuple[str, str, str, str], list[dict]] = {}

    for row in rows:
        keys = [
            ("ALL", row["vessel_type"], row["representation"], row["metric_key"]),
            (row["cohort"], row["vessel_type"], row["representation"], row["metric_key"]),
        ]
        for key in keys:
            groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for (cohort, vessel_type, representation, metric_key), vals in groups.items():
        if not vals:
            continue
        z_values = [float(v["z"]) for v in vals]
        zc_values = [float(v["z_capped"]) for v in vals]
        was_points = [float(v["was_points"]) for v in vals]
        was_c_points = [float(v["was_c_points"]) for v in vals]
        active = [z > 0.0 for z in z_values]
        split = split_by_metric.get(metric_key, {})

        out.append(
            {
                "cohort": cohort,
                "vessel_type": vessel_type,
                "representation": representation,
                "metric_key": metric_key,
                "metric_name": vals[0]["metric_name"],
                "n_subjects": len(vals),
                "n_active": int(sum(active)),
                "active_fraction": float(sum(active) / len(vals)),
                "mean_z": float(sum(z_values) / len(z_values)),
                "median_z": float(median(z_values)),
                "sum_z": float(sum(z_values)),
                "mean_z_capped": float(sum(zc_values) / len(zc_values)),
                "sum_z_capped": float(sum(zc_values)),
                "mean_was_points": float(sum(was_points) / len(was_points)),
                "sum_was_points": float(sum(was_points)),
                "mean_was_c_points": float(sum(was_c_points) / len(was_c_points)),
                "sum_was_c_points": float(sum(was_c_points)),
                "threshold": vals[0]["threshold"],
                "direction": vals[0]["direction"],
                "control_std": vals[0]["control_std"],
                "split_sensitivity": split.get("sensitivity"),
                "split_specificity": split.get("specificity"),
                "split_balanced_accuracy": split.get("balanced_accuracy"),
                "split_youden_j": split.get("youden_j"),
                "split_auc_greater": split.get("auc_greater"),
                "split_auc_less": split.get("auc_less"),
                "split_separability_auc": split.get("separability_auc"),
                "p_value_mannwhitney": split.get("p_value_mannwhitney"),
                "selected_for_score": split.get("selected_for_score"),
            }
        )

    out.sort(
        key=lambda r: (
            r["cohort"] != "ALL",
            r["cohort"],
            r["vessel_type"],
            r["representation"],
            -float(r["active_fraction"]),
            -float(r["sum_was_c_points"]),
            -float(r["split_youden_j"] or 0.0),
        )
    )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        if not rows:
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_metric_panel_tex(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("\\begin{table}[htbp]\n")
        file.write("\\centering\n")
        file.write("\\caption{Top metrics selected for WAS/WAS-c by separability AUC.}\n")
        file.write("\\label{tab:selected_metric_panel_top_auc}\n")
        file.write("\\begin{tabular}{lrrrrr}\n")
        file.write("\\hline\n")
        file.write(
            "Metric & Threshold & Control SD & Balanced accuracy & "
            "AUC separability & Mann--Whitney $p$ \\\\\n"
        )
        file.write("\\hline\n")
        for row in rows:
            metric_name = METRIC_LATEX_NAMES.get(
                str(row.get("metric_name", "")),
                _latex_escape(str(row.get("metric_name", ""))),
            )
            file.write(
                f"{metric_name} & "
                f"{_format_number(row.get('threshold'))} & "
                f"{_format_number(row.get('control_std'))} & "
                f"{_format_number(row.get('balanced_accuracy'))} & "
                f"{_format_number(row.get('separability_auc'))} & "
                f"{_format_p_value(row.get('p_value_mannwhitney'))} \\\\\n"
            )
        file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write("\\end{table}\n")


def _format_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{number:.{digits}g}"


def _format_p_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number < 1e-3:
        return f"{number:.2e}"
    return f"{number:.3f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
