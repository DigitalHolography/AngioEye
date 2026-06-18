from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Iterable

from .dataclasses import MetricContributionRecord
from .optimal_split import SplitStats


def write_optimal_split_report(
    split_stats: Iterable[SplitStats],
    output_dir: Path,
) -> list[str]:
    report_dir = output_dir / "composite_scoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(stat) for stat in split_stats]
    json_path = report_dir / "optimal_split_calibration.json"
    csv_path = report_dir / "optimal_split_calibration.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    _write_csv(csv_path, rows)
    return [str(json_path), str(csv_path)]




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
    json_path = report_dir / "selected_metric_panel_top_auc.json"
    _write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    return [str(csv_path), str(json_path)]


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
    summary_json = report_dir / "metric_usefulness_summary.json"
    _write_csv(summary_csv, summary_rows)
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2, ensure_ascii=False)

    return [str(detail_csv), str(summary_csv), str(summary_json)]


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
