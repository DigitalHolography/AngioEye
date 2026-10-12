import numpy as np
import pandas as pd

from math_utils import (
    auc_from_scores,
    cohen_d,
    mann_whitney_pvalue,
    summarize_values,
)

from .constants import (
    COLUMN_LABELS,
    DEFAULT_TOP_N,
    EPS,
    INPUT_METRICS,
    SPATIAL_RAW_COLUMNS,
    TEMPORAL_RAW_COLUMNS,
)
from .formatting import (
    format_float,
    format_mean_std,
    format_pvalue_latex,
    latex_escape_text,
    metric_label,
)
from .statistics import (
    best_threshold_sensitivity_specificity_cumulative_sweep,
    combine_variability_score,
)


def build_group_table_with_columns(
    results_for_group,
    selected_higher_metrics,
    metrics=INPUT_METRICS,
    digits=3,
):
    rows = []
    for metric_name in metrics:
        metric_block = results_for_group.get(metric_name, {})
        row = {"Metric": metric_label(metric_name)}
        for high_name in selected_higher_metrics:
            row[COLUMN_LABELS[high_name]] = format_mean_std(
                metric_block.get(high_name, []),
                digits=digits,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_spatial_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    return build_group_table_with_columns(
        results_for_group,
        SPATIAL_RAW_COLUMNS,
        metrics=metrics,
        digits=digits,
    )


def build_temporal_group_table(results_for_group, metrics=INPUT_METRICS, digits=3):
    return build_group_table_with_columns(
        results_for_group,
        TEMPORAL_RAW_COLUMNS,
        metrics=metrics,
        digits=digits,
    )


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
    rows = []
    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    group_col = f"$V^{{{domain_name}}}_{{{group_tex}}}$"
    control_col = f"$V^{{{domain_name}}}_{{{control_tex}}}$"
    global_col = f"$V^{{{domain_name}}}_{{global}}$"
    for metric_name in metrics:
        control = summarize_values(
            combine_variability_score(control_results, metric_name, higher_metrics)
        )["median"]
        group = summarize_values(
            combine_variability_score(group_results, metric_name, higher_metrics)
        )["median"]
        if np.isfinite(control) and np.isfinite(group):
            rows.append(
                {
                    "Metric": metric_label(metric_name),
                    group_col: group,
                    control_col: control,
                    global_col: 0.5 * (control + group),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(global_col, ascending=ascending).head(n)
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    for column in (group_col, control_col, global_col):
        frame[column] = frame[column].apply(
            lambda value: format_float(value, digits=digits)
        )
    return frame[["Rank", "Metric", group_col, control_col, global_col]]


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
    rows = []
    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    group_col = f"$V^{{{domain_name}}}_{{{group_tex}}}$"
    control_col = f"$V^{{{domain_name}}}_{{{control_tex}}}$"
    for metric_name in metrics:
        control = summarize_values(
            combine_variability_score(control_results, metric_name, higher_metrics)
        )["median"]
        group = summarize_values(
            combine_variability_score(group_results, metric_name, higher_metrics)
        )["median"]
        if not np.isfinite(control) or not np.isfinite(group):
            continue
        if abs(control) < EPS and abs(group) < EPS:
            ratio = np.nan
        else:
            ratio = max(abs(control), abs(group)) / (
                min(abs(control), abs(group)) + EPS
            )
        if np.isfinite(ratio):
            rows.append(
                {
                    "Metric": metric_label(metric_name),
                    "More variable group": (
                        group_tex if group >= control else control_tex
                    ),
                    group_col: group,
                    control_col: control,
                    "Ratio": ratio,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("Ratio", ascending=False).head(n)
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    for column in (group_col, control_col, "Ratio"):
        frame[column] = frame[column].apply(
            lambda value: format_float(value, digits=digits)
        )
    return frame[
        ["Rank", "Metric", "More variable group", group_col, control_col, "Ratio"]
    ]


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
    rows = []
    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    control_median_column = f"Median variability {control_tex}"
    group_median_column = f"Median variability {group_tex}"
    for metric_name in metrics:
        control = combine_variability_score(
            control_results, metric_name, higher_metrics
        )
        group = combine_variability_score(group_results, metric_name, higher_metrics)
        control_summary = summarize_values(control)
        group_summary = summarize_values(group)
        if control_summary["n"] == 0 or group_summary["n"] == 0:
            continue
        auc = auc_from_scores(control, group)
        auc_separability = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        _, sensitivity, specificity, _ = (
            best_threshold_sensitivity_specificity_cumulative_sweep(
                control,
                group,
                evaluate_both_directions=evaluate_both_directions,
            )
        )
        rows.append(
            {
                "Metric": metric_label(metric_name),
                control_median_column: control_summary["median"],
                group_median_column: group_summary["median"],
                "More variable group": (
                    group_tex
                    if group_summary["median"] > control_summary["median"]
                    else control_tex
                ),
                "AUC separability": auc_separability,
                "Mann--Whitney p-value": mann_whitney_pvalue(control, group),
                "Cohen's $d$": cohen_d(control, group),
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame[np.isfinite(frame["AUC separability"])]
    frame = frame.sort_values("AUC separability", ascending=False)
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    for column in (
        control_median_column,
        group_median_column,
        "AUC separability",
        "Cohen's $d$",
        "Sensitivity",
        "Specificity",
    ):
        frame[column] = frame[column].apply(
            lambda value: format_float(value, digits=digits)
        )
    frame["Mann--Whitney p-value"] = frame["Mann--Whitney p-value"].apply(
        lambda value: format_pvalue_latex(value, sig_digits=digits)
    )
    return frame[
        [
            "Rank",
            "Metric",
            control_median_column,
            group_median_column,
            "More variable group",
            "AUC separability",
            "Mann--Whitney p-value",
            "Cohen's $d$",
            "Sensitivity",
            "Specificity",
        ]
    ]
