from dataclasses import replace

from postprocess.core.base import PostprocessContext, PostprocessResult
from postprocess.core.grouped_batch import extract_group_name

from .dataclasses import MetricContributionRecord, ScoreRecord
from .metrics import METRIC_PANEL
from .optimal_split import SplitStats, calibrate_metrics_from_processed_files
from .plots import write_score_plots
from .reports import (
    write_metric_contribution_reports,
    write_optimal_split_report,
    write_selected_metric_panel_report,
)
from .scoring import (
    append_scores_to_file,
    contribution_records_for_tree,
    score_records_for_tree,
)

# Set these explicitly if the automatic detection is ambiguous.
CONTROL_GROUP_NAME = None
PATHOLOGY_GROUP_NAME = None

# This defines where thresholds/directions/AUC are learned. Scores are then
# written for all configured representations/vessel types using the selected panel.
OPTIMIZE_VESSEL_TYPE = "artery"
OPTIMIZE_REPRESENTATION = "bandlimited"
AGGREGATION_FOR_SPLIT = "median"

# Keep only the N metrics with the best threshold-independent separability AUC.
N_METRICS_FOR_SCORE = 10


def _select_top_auc_metrics(
    metric_specs: dict,
    split_stats: list[SplitStats],
    *,
    n_metrics: int,
) -> tuple[dict, list[SplitStats], list[SplitStats]]:
    """Return metric_specs restricted to top-N separability AUC metrics.

    `separability_auc = max(AUC_GREATER, AUC_LESS)` ranks metrics independently
    from the exact threshold. The threshold/direction used for z are still the
    optimal split parameters stored in SplitStats/Metric.
    """
    ranked = sorted(
        split_stats,
        key=lambda stat: (
            stat.separability_auc,
            stat.youden_j,
            stat.balanced_accuracy,
        ),
        reverse=True,
    )
    selected_keys = {stat.metric_key for stat in ranked[:n_metrics]}

    selected_specs = {
        key: metric
        for key, metric in metric_specs.items()
        if key in selected_keys
    }

    selected_stats = []
    annotated_stats = []
    for stat in split_stats:
        annotated = replace(
            stat,
            selected_for_score=stat.metric_key in selected_keys,
        )
        annotated_stats.append(annotated)
        if annotated.selected_for_score:
            selected_stats.append(annotated)

    # Preserve ranking order for selected stats/reports.
    selected_stats.sort(
        key=lambda stat: (
            stat.separability_auc,
            stat.youden_j,
            stat.balanced_accuracy,
        ),
        reverse=True,
    )
    return selected_specs, selected_stats, annotated_stats


def run_composite_scoring(context: PostprocessContext) -> PostprocessResult:
    updated_paths: list[str] = []
    score_records: list[ScoreRecord] = []
    contribution_records: list[MetricContributionRecord] = []
    failures: list[str] = []

    try:
        all_metric_specs, all_split_stats = calibrate_metrics_from_processed_files(
            context.processed_files,
            context.output_dir,
            metric_panel=METRIC_PANEL,
            control_group=CONTROL_GROUP_NAME,
            pathology_group=PATHOLOGY_GROUP_NAME,
            optimize_vessel_type=OPTIMIZE_VESSEL_TYPE,
            optimize_representation=OPTIMIZE_REPRESENTATION,
            aggregation=AGGREGATION_FOR_SPLIT,
        )
        metric_specs, selected_split_stats, annotated_split_stats = _select_top_auc_metrics(
            all_metric_specs,
            all_split_stats,
            n_metrics=N_METRICS_FOR_SCORE,
        )
        if not metric_specs:
            raise ValueError("No metrics were selected for WAS/WAS-c scoring.")
    except Exception as exc:  # noqa: BLE001
        return PostprocessResult(
            summary=(
                "Composite Scoring optimal-split/AUC calibration failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            generated_paths=[],
            metadata={"failures": [str(exc)]},
        )

    report_paths = write_optimal_split_report(annotated_split_stats, context.output_dir)
    report_paths.extend(
        write_selected_metric_panel_report(
            selected_split_stats,
            context.output_dir,
        )
    )

    for file_path in context.processed_files:
        try:
            tree = append_scores_to_file(file_path, metric_specs=metric_specs)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"Composite Scoring skipped {file_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        cohort = extract_group_name(file_path.parent, context.output_dir)
        score_records.extend(
            score_records_for_tree(
                tree,
                cohort=cohort,
                file_path=file_path,
            )
        )
        contribution_records.extend(
            contribution_records_for_tree(
                tree,
                cohort=cohort,
                file_path=file_path,
                metric_specs=metric_specs,
            )
        )
        updated_paths.append(str(file_path))

    png_paths = write_score_plots(score_records, context.output_dir)
    report_paths.extend(
        write_metric_contribution_reports(
            contribution_records,
            selected_split_stats,
            context.output_dir,
        )
    )

    return PostprocessResult(
        summary=(
            f"Appended paper-style WAS/WAS-c using the top "
            f"{len(metric_specs)} metric(s) ranked by separability AUC to "
            f"{len(updated_paths)} processed HDF5 file(s). Generated "
            f"{len(png_paths)} PNG plot(s) and {len(report_paths)} report file(s). "
            f"Skipped {len(failures)} file(s)."
        ),
        generated_paths=[*updated_paths, *png_paths, *report_paths],
        metadata={
            "failures": failures,
            "selected_metric_panel": [stat.metric_key for stat in selected_split_stats],
            "n_metrics_for_score": len(metric_specs),
        },
    )
