import numpy as np
import pandas as pd

from math_utils import (
    auc_from_scores,
    cohen_d,
    mann_whitney_pvalue,
    mean_difference_ci95,
    overlap_from_cohen_d,
    summarize_values,
)

from .constants import (
    COLUMN_LABELS,
    DEFAULT_TOP_N,
    EPS,
    INPUT_METRICS,
    SPATIAL_RAW_COLUMNS,
    SUMMARY_PVALUE_METRICS,
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
    get_descriptor_values_for_test,
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
        control = combine_variability_score(
            control_results, metric_name, higher_metrics
        )
        group = combine_variability_score(group_results, metric_name, higher_metrics)
        control_summary = summarize_values(control)
        group_summary = summarize_values(group)
        rows.append(
            {
                "Metric": metric_label(metric_name),
                f"n {control_name}": control_summary["n"],
                f"n {group_name}": group_summary["n"],
                f"Median {control_name}": control_summary["median"],
                f"Median {group_name}": group_summary["median"],
                "Median difference": (
                    group_summary["median"] - control_summary["median"]
                ),
                "Mann-Whitney p-value": mann_whitney_pvalue(control, group),
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame[np.isfinite(frame["Mann-Whitney p-value"])]
    frame = frame.sort_values("Mann-Whitney p-value")
    if n is not None:
        frame = frame.head(n)
    for column in (
        f"Median {control_name}",
        f"Median {group_name}",
        "Median difference",
    ):
        frame[column] = frame[column].apply(
            lambda value: format_float(value, digits=digits)
        )
    frame["Mann-Whitney p-value"] = frame["Mann-Whitney p-value"].apply(
        lambda value: format_pvalue_latex(value, sig_digits=digits)
    )
    return frame[
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


def build_descriptor_pvalue_summary_table(
    control_results,
    group_results,
    descriptor_map,
    control_name,
    group_name,
    metrics=SUMMARY_PVALUE_METRICS,
    digits=4,
):
    rows = []
    higher_metrics = list(descriptor_map.values())
    for metric_name in metrics:
        row = {"Metric": metric_label(metric_name)}
        control_counts = []
        group_counts = []
        for descriptor_name, high_name in descriptor_map.items():
            control = get_descriptor_values_for_test(
                control_results, metric_name, high_name
            )
            group = get_descriptor_values_for_test(
                group_results, metric_name, high_name
            )
            row[f"{descriptor_name} p-value"] = mann_whitney_pvalue(
                control, group
            )
            control_counts.append(len(control))
            group_counts.append(len(group))
        control_score = combine_variability_score(
            control_results, metric_name, higher_metrics
        )
        group_score = combine_variability_score(
            group_results, metric_name, higher_metrics
        )
        row["Mean p-value"] = mann_whitney_pvalue(control_score, group_score)
        row[f"n {control_name}"] = max(control_counts, default=0)
        row[f"n {group_name}"] = max(group_counts, default=0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    pvalue_columns = [
        *(f"{name} p-value" for name in descriptor_map),
        "Mean p-value",
    ]
    for column in pvalue_columns:
        frame[column] = frame[column].apply(
            lambda value: format_pvalue_latex(value, sig_digits=digits)
        )
    return frame[
        [
            "Metric",
            f"n {control_name}",
            f"n {group_name}",
            *pvalue_columns,
        ]
    ]


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
    metric_results = {}
    backslash = chr(92)
    control_tex = latex_escape_text(control_name)
    group_tex = latex_escape_text(group_name)
    control_count_label = f"$n_{{{backslash}mathrm{{{control_tex}}}}}$"
    group_count_label = f"$n_{{{backslash}mathrm{{{group_tex}}}}}$"
    confidence_label = "Mean difference 95" + backslash + "% CI"
    for metric_name in metrics:
        control = combine_variability_score(
            control_results, metric_name, higher_metrics
        )
        group = combine_variability_score(group_results, metric_name, higher_metrics)
        control_summary = summarize_values(control)
        group_summary = summarize_values(group)
        pvalue = mann_whitney_pvalue(control, group)
        effect_size = cohen_d(control, group)
        _, ci_low, ci_high = mean_difference_ci95(control, group)
        auc = auc_from_scores(control, group)
        auc_separability = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        _, sensitivity, specificity, _ = (
            best_threshold_sensitivity_specificity_cumulative_sweep(
                control,
                group,
                evaluate_both_directions=evaluate_both_directions,
            )
        )
        more_variable_group = (
            group_tex
            if group_summary["median"] > control_summary["median"]
            else control_tex
        )
        metric_results[metric_label(metric_name)] = {
            control_count_label: str(control_summary["n"]),
            group_count_label: str(group_summary["n"]),
            f"Median {control_tex}": format_float(
                control_summary["median"], digits=digits
            ),
            f"Median {group_tex}": format_float(
                group_summary["median"], digits=digits
            ),
            "More variable group": more_variable_group,
            "Mann--Whitney p-value": format_pvalue_latex(
                pvalue, sig_digits=digits
            ),
            "Cohen's $d$": format_float(effect_size, digits=digits),
            confidence_label: (
                f"[{format_float(ci_low, digits=digits)}, "
                f"{format_float(ci_high, digits=digits)}]"
            ),
            "AUC separability": format_float(auc_separability, digits=digits),
            "Sensitivity": format_float(sensitivity, digits=digits),
            "Specificity": format_float(specificity, digits=digits),
            "Overlap OVL": format_float(
                overlap_from_cohen_d(effect_size), digits=digits
            ),
        }
    estimator_order = [
        control_count_label,
        group_count_label,
        f"Median {control_tex}",
        f"Median {group_tex}",
        "More variable group",
        "Mann--Whitney p-value",
        "Cohen's $d$",
        confidence_label,
        "AUC separability",
        "Sensitivity",
        "Specificity",
        "Overlap OVL",
    ]
    return pd.DataFrame(
        [
            {
                "Estimator": estimator,
                **{
                    metric_name: values.get(estimator, "NA")
                    for metric_name, values in metric_results.items()
                },
            }
            for estimator in estimator_order
        ]
    )


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
