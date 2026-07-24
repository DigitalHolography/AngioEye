"""
Turns a batch of already-processed AngioEye output files into the low-rank
flicker paper's confound-controlled replication report (manuscript Sec. V),
for the project's confirmed candidate endpoints (total_pulsatile_rms, rho1,
rho2, rho3), grouped by participant/eye and epoch.

Reads lowrank_pulsatility_metrics.py's *already-computed* output metrics
(the "raw" representation's per-beat arrays and acquisition-level mean beat
period) straight out of each processed .h5 file via the standard
pipeline_path_candidates/find_first_existing_path convention -- unlike every
flicker-detection analysis script that came before this
(flicker_epoch_report.py, flicker_denoising_separability.py, and the
langevin-internship check_*.py scripts), which all bypassed the pipeline
framework's own output and re-loaded the raw acquisition file to recompute
_compute_representation from scratch.

Grouping: uses grouped_batch.iter_grouped_h5_files(..., group_depth=2), so
this works unmodified on both known folder layouts -- BOM0753's flat
{epoch}/*.h5 (group_name ends up just "baseline1"/"flicker"/"baseline2")
and 260720_OSS's nested {eye}/{epoch}/.../*.h5 (group_name becomes
"L/baseline1" etc). The last path segment of group_name is always treated
as the epoch; everything before it (if any) is the participant/eye prefix.
Each distinct prefix is analyzed completely separately and never pooled
with another -- two eyes of one participant, or two different
participants, are never treated as one bigger N, per this project's
established rule (see project_flicker_detection_findings.md finding 17).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from input_output.hdf5_io import find_first_existing_path, read_array, read_dataset
from input_output.hdf5_schema import pipeline_path_candidates

from ..core.grouped_batch import iter_grouped_h5_files
from . import flicker_stats

PIPELINE_NAME = "lowrank_pulsatility_metrics"
REPRESENTATION = "raw"

# name -> the beatwise/ key lowrank_pulsatility_metrics.py writes for it.
CANDIDATE_METRICS = {
    "total_pulsatile_rms": "median_kr_total_rms",
    "rho1": "rho1_beatwise",
    "rho2": "rho2_beatwise",
    "rho3": "rho3_beatwise",
}

CONTROL_METRIC = "total_pulsatile_rms"


def _resolve(h5file: h5py.File, *parts: str) -> str | None:
    return find_first_existing_path(
        h5file, pipeline_path_candidates(PIPELINE_NAME, REPRESENTATION, *parts)
    )


def read_acquisition(h5_path: Path) -> dict | None:
    """
    One processed acquisition's per-beat arrays for each CANDIDATE_METRICS
    entry plus its mean beat period, or None if lowrank_pulsatility_metrics
    didn't run (or its SVD wasn't available) for this file.
    """
    with h5py.File(h5_path, "r") as f:
        svd_path = _resolve(f, "qc", "svd_available")
        if svd_path is None:
            return None
        svd_available = read_dataset(f, svd_path)
        if svd_available is None or not bool(np.asarray(svd_available).ravel()[0]):
            return None

        period_path = _resolve(f, "acquisition_dots", "beat_period", "mean")
        if period_path is None:
            return None
        beat_period = read_dataset(f, period_path)
        if beat_period is None:
            return None

        beatwise: dict[str, np.ndarray] = {}
        for name, key in CANDIDATE_METRICS.items():
            arr_path = _resolve(f, "beatwise", key)
            if arr_path is None:
                return None
            arr = read_array(f, arr_path, dtype=float)
            if arr is None:
                return None
            beatwise[name] = arr

    return {"beatwise": beatwise, "beat_period": float(np.asarray(beat_period).ravel()[0])}


def split_group_name(group_name: str) -> tuple[str, str]:
    """(participant_prefix, epoch) -- epoch is always the last path segment."""
    parts = group_name.split("/")
    return "/".join(parts[:-1]), parts[-1]


def _empty_prefix_bucket() -> dict[str, dict[str, list]]:
    bucket = {name: {epoch: [] for epoch in flicker_stats.EPOCH_ORDER} for name in CANDIDATE_METRICS}
    bucket["beat_period"] = {epoch: [] for epoch in flicker_stats.EPOCH_ORDER}
    return bucket


def collect_prefixed_epoch_data(output_dir: Path) -> dict[str, dict[str, dict[str, list]]]:
    """
    results[prefix][metric_name][epoch] -> list of per-acquisition per-beat
    arrays; results[prefix]["beat_period"][epoch] -> matching list of mean
    beat periods. One independent bucket per participant/eye prefix
    ("" if the batch has no such prefix, i.e. a flat single-participant
    layout like BOM0753's).
    """
    results: dict[str, dict[str, dict[str, list]]] = {}

    for grouped_file in iter_grouped_h5_files(output_dir, group_depth=2):
        prefix, epoch = split_group_name(grouped_file.group_name)
        if epoch not in flicker_stats.EPOCH_ORDER:
            continue

        acquisition = read_acquisition(grouped_file.file_path)
        if acquisition is None:
            continue

        if prefix not in results:
            results[prefix] = _empty_prefix_bucket()
        bucket = results[prefix]

        for name in CANDIDATE_METRICS:
            bucket[name][epoch].append(acquisition["beatwise"][name])
        bucket["beat_period"][epoch].append(acquisition["beat_period"])

    return results


def _native_values(beatwise_values: dict[str, list]) -> dict[str, np.ndarray]:
    """Median-across-beats reduction, matching the manuscript's own
    "median across vessel locations and beats" acquisition-dot convention."""
    return {
        epoch: np.array([flicker_stats.beat_combine(v, "median") for v in beatwise_values[epoch]])
        for epoch in flicker_stats.EPOCH_ORDER
    }


def build_replication_report(output_dir: str | Path) -> dict[str, dict]:
    """
    One flicker_stats.run_confound_battery report per (participant/eye
    prefix, metric) pair. total_pulsatile_rms is never residualized against
    itself; rho1/rho2/rho3 are each residualized against
    total_pulsatile_rms, matching the confound-control pattern used
    throughout this project's history.
    """
    prefixed_data = collect_prefixed_epoch_data(Path(output_dir))
    report: dict[str, dict] = {}

    for prefix, metrics in prefixed_data.items():
        beat_periods = metrics["beat_period"]
        control_native = _native_values(metrics[CONTROL_METRIC])

        prefix_report: dict[str, dict] = {}
        for name in CANDIDATE_METRICS:
            beatwise_values = metrics[name]
            control_values = None if name == CONTROL_METRIC else control_native
            prefix_report[name] = flicker_stats.run_confound_battery(
                _native_values(beatwise_values),
                beatwise_values=beatwise_values,
                beat_periods=beat_periods,
                control_values=control_values,
                control_name=CONTROL_METRIC,
            )
        report[prefix or "all"] = prefix_report

    return report


def format_report_text(report: dict[str, dict]) -> str:
    lines: list[str] = []
    for prefix, metrics in sorted(report.items()):
        lines.append(f"=== {prefix} ===")
        for name, battery in metrics.items():
            vb = battery.get("vb_report") or {}
            resid = battery.get("residualized")
            line = (
                f"  {name:<20} raw p={battery['raw_p']:.4g} AUC={battery['raw_auc']:.3f}"
                f"  V.B retained={vb.get('retained')}"
            )
            if resid is not None:
                line += (
                    f"  residualized(vs {resid['control_name']}) p={resid['p']:.4g}"
                    f" R^2={resid['r_squared']:.3f} survives={resid['survives']}"
                )
            lines.append(line)
        lines.append("")
    return "\n".join(lines)
