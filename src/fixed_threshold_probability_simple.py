"""Fixed-threshold patient pathology index with simple interactive file choice.

Usage
-----
Simple mode, recommended:
    python fixed_threshold_probability_simple.py

A file-selection window opens. Pick one patient .h5/.hdf5 file. Outputs are
written next to that file, in a folder named:
    <patient_file_stem>_fixed_threshold_probability/

Defaults:
    vessel_type = "artery"
    representation = "bandlimited"

Optional CLI mode still works:
    python fixed_threshold_probability_simple.py --h5 /path/to/patient.h5
    python fixed_threshold_probability_simple.py --h5 /path/to/patient.h5 --vessel vein --representation raw
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import csv

import h5py
import numpy as np

try:
    from input_output.hdf5_io import read_dataset
    from input_output.hdf5_schema import find_pipeline_group
except Exception:  # allows importing this file outside AngioEye for inspection
    read_dataset = None
    find_pipeline_group = None


GREATER = 1
LESS = -1
DEFAULT_VESSEL_TYPE = "artery"
DEFAULT_REPRESENTATION = "bandlimited"
METRIC_PATH = "{vessel_type}/global/{representation}/{metric_name}"


@dataclass(frozen=True)
class FixedThresholdMetric:
    key: str
    name: str
    latex_name: str
    threshold: float
    direction: int
    control_std: float | None = None
    weight: float = 1.0
    numerator_name: str | None = None
    denominator_name: str | None = None

    def path(self, vessel_type: str, representation: str) -> str:
        return METRIC_PATH.format(
            vessel_type=vessel_type,
            representation=representation,
            metric_name=self.name,
        )

    def derived_paths(self, vessel_type: str, representation: str) -> tuple[str, str] | None:
        if self.numerator_name is None or self.denominator_name is None:
            return None
        return (
            METRIC_PATH.format(
                vessel_type=vessel_type,
                representation=representation,
                metric_name=self.numerator_name,
            ),
            METRIC_PATH.format(
                vessel_type=vessel_type,
                representation=representation,
                metric_name=self.denominator_name,
            ),
        )


@dataclass(frozen=True)
class MetricContribution:
    metric_key: str
    metric_name: str
    latex_name: str
    value: float
    threshold: float
    direction: int
    decision_rule: str
    control_std: float | None
    weight: float
    abnormal: bool
    z: float
    z_capped: float
    weighted_capped_contribution: float


@dataclass(frozen=True)
class PatientProbabilityResult:
    file_name: str
    vessel_type: str
    representation: str
    n_metrics_total: int
    n_metrics_valid: int
    n_metrics_abnormal: int
    abnormal_fraction: float
    was: float
    was_c: float
    pathology_probability_like: float


# ---------------------------------------------------------------------------
# EDIT THIS PANEL
# ---------------------------------------------------------------------------
# Define your fixed-threshold key metrics here.
# direction=GREATER means abnormal if value >= threshold.
# direction=LESS means abnormal if value <= threshold.
# If control_std is None, the metric contributes 0/1. If control_std is set,
# it contributes a normalized threshold-excess z-score.
KEY_METRICS: tuple[FixedThresholdMetric, ...] = (
    FixedThresholdMetric(
        key="PI",
        name="PI",
        latex_name=r"$\mathrm{PI}$",
        threshold=1.30,
        direction=GREATER,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="t50_over_T",
        name="t50_over_T",
        latex_name=r"$t_{50}/T$",
        threshold=0.36,
        direction=LESS,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="SF_VTI",
        name="SF_VTI",
        latex_name=r"$\mathrm{SF}_{\mathrm{VTI}}$",
        threshold=0.48,
        direction=GREATER,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="RI",
        name="RI",
        latex_name=r"$\mathrm{RI}$",
        threshold=0.70,
        direction=GREATER,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="v_end_over_vbar",
        name="v_end_over_vbar",
        latex_name=r"$\bar v_{\mathrm{end}}/\bar v$",
        threshold=0.59,
        direction=LESS,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="N_eff_over_T",
        name="N_eff_over_T",
        latex_name=r"$N_{\mathrm{eff}}/T$",
        threshold=0.90,
        direction=LESS,
        control_std=None,
    ),
    FixedThresholdMetric(
        key="W50_over_T",
        name="W50_over_T",
        latex_name=r"$W_{50}/T$",
        threshold=0.60,
        direction=LESS,
        control_std=None,
    ),
)


def choose_h5_file() -> Path:
    """Open a file chooser when possible, otherwise ask for a path in terminal."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Choose patient HDF5 file",
            filetypes=(
                ("HDF5 files", "*.h5 *.hdf5"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    selected = input("Path to patient .h5/.hdf5 file: ").strip().strip('"')
    if not selected:
        raise SystemExit("No file selected.")
    return Path(selected)


def default_output_dir_for_h5(file_path: str | Path) -> Path:
    file_path = Path(file_path)
    return file_path.parent / f"{file_path.stem}_fixed_threshold_probability"


def _finite_values(value: Any) -> np.ndarray:
    values = np.asarray(value, dtype=float).ravel()
    return values[np.isfinite(values)]


def _finite_scalar(value: Any) -> float | None:
    values = _finite_values(value)
    if values.size == 0:
        return None
    return float(np.nanmedian(values))


def _read_metric_value(
    source_group: h5py.Group,
    metric: FixedThresholdMetric,
    *,
    vessel_type: str,
    representation: str,
) -> float | None:
    if read_dataset is None:
        raise RuntimeError("This script must be run inside the AngioEye environment.")

    derived = metric.derived_paths(vessel_type, representation)
    if derived is None:
        value = read_dataset(
            source_group,
            metric.path(vessel_type, representation),
            default=None,
        )
        return _finite_scalar(value)

    numerator = read_dataset(source_group, derived[0], default=None)
    denominator = read_dataset(source_group, derived[1], default=None)
    if numerator is None or denominator is None:
        return None

    numerator_values = np.asarray(numerator, dtype=float)
    denominator_values = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(denominator_values) & (denominator_values != 0),
            numerator_values / denominator_values,
            np.nan,
        )
    return _finite_scalar(ratio)


def _metric_contribution(value: float, metric: FixedThresholdMetric) -> MetricContribution:
    signed_excess = metric.direction * (value - metric.threshold)
    abnormal = bool(signed_excess >= 0)

    if not abnormal:
        z = 0.0
    elif metric.control_std is None or not np.isfinite(metric.control_std) or metric.control_std <= 0:
        z = 1.0
    else:
        z = max(0.0, float(signed_excess / metric.control_std))

    z_capped = min(1.0, z)
    weighted = metric.weight * z_capped
    decision_rule = ">= threshold" if metric.direction == GREATER else "<= threshold"

    return MetricContribution(
        metric_key=metric.key,
        metric_name=metric.name,
        latex_name=metric.latex_name,
        value=float(value),
        threshold=float(metric.threshold),
        direction=int(metric.direction),
        decision_rule=decision_rule,
        control_std=None if metric.control_std is None else float(metric.control_std),
        weight=float(metric.weight),
        abnormal=abnormal,
        z=float(z),
        z_capped=float(z_capped),
        weighted_capped_contribution=float(weighted),
    )


def score_patient_h5(
    file_path: str | Path,
    *,
    metrics: tuple[FixedThresholdMetric, ...] = KEY_METRICS,
    vessel_type: str = DEFAULT_VESSEL_TYPE,
    representation: str = DEFAULT_REPRESENTATION,
    source_pipeline: str = "waveform_shape_metrics",
) -> tuple[PatientProbabilityResult, list[MetricContribution]]:
    file_path = Path(file_path)

    if find_pipeline_group is None:
        raise RuntimeError("This script must be run inside the AngioEye environment.")

    contributions: list[MetricContribution] = []
    with h5py.File(file_path, "r") as h5:
        source_group = find_pipeline_group(h5, source_pipeline)
        if source_group is None:
            raise ValueError(
                f"Expected '{source_pipeline}' pipeline group not found in {file_path}"
            )

        for metric in metrics:
            value = _read_metric_value(
                source_group,
                metric,
                vessel_type=vessel_type,
                representation=representation,
            )
            if value is None:
                continue
            contributions.append(_metric_contribution(value, metric))

    total_weight = float(sum(item.weight for item in contributions))
    if total_weight <= 0 or not contributions:
        raise ValueError(
            "No valid key metric could be read. Check metric names, vessel_type, "
            f"and representation for {file_path}."
        )

    sum_z_weighted = float(sum(item.weight * item.z for item in contributions))
    sum_z_capped_weighted = float(sum(item.weight * item.z_capped for item in contributions))

    was = 10.0 * sum_z_weighted / total_weight
    was_c = 10.0 * sum_z_capped_weighted / total_weight
    probability_like = was_c / 10.0

    n_abnormal = int(sum(item.abnormal for item in contributions))
    result = PatientProbabilityResult(
        file_name=file_path.name,
        vessel_type=vessel_type,
        representation=representation,
        n_metrics_total=len(metrics),
        n_metrics_valid=len(contributions),
        n_metrics_abnormal=n_abnormal,
        abnormal_fraction=float(n_abnormal / len(contributions)),
        was=float(was),
        was_c=float(was_c),
        pathology_probability_like=float(probability_like),
    )
    return result, contributions


def write_patient_reports(
    result: PatientProbabilityResult,
    contributions: list[MetricContribution],
    output_dir: str | Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "patient_pathology_probability_summary.csv"
    contribution_path = output_dir / "patient_metric_threshold_contributions.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

    rows = [asdict(item) for item in contributions]
    if rows:
        with open(contribution_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return [str(summary_path), str(contribution_path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score one patient HDF5 file with fixed thresholds. "
            "Run without arguments to choose the H5 file interactively."
        )
    )
    parser.add_argument(
        "--h5",
        required=False,
        help="Path to patient HDF5 file. If omitted, a file chooser opens.",
    )
    parser.add_argument(
        "--out",
        required=False,
        help=(
            "Output directory. If omitted, outputs are written next to the H5 "
            "inside <h5_stem>_fixed_threshold_probability/."
        ),
    )
    parser.add_argument("--vessel", default=DEFAULT_VESSEL_TYPE, choices=["artery", "vein"])
    parser.add_argument(
        "--representation",
        default=DEFAULT_REPRESENTATION,
        choices=["raw", "bandlimited"],
    )
    args = parser.parse_args()

    h5_path = Path(args.h5) if args.h5 else choose_h5_file()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    output_dir = Path(args.out) if args.out else default_output_dir_for_h5(h5_path)

    result, contributions = score_patient_h5(
        h5_path,
        vessel_type=args.vessel,
        representation=args.representation,
    )
    paths = write_patient_reports(result, contributions, output_dir)

    print("\nFixed-threshold patient score")
    print(f"H5 file: {h5_path}")
    print(f"Vessel: {result.vessel_type}")
    print(f"Representation: {result.representation}")
    print(f"Pathology probability-like index: {result.pathology_probability_like:.3f}")
    print(f"WAS-c: {result.was_c:.3f}/10")
    print(f"Abnormal metrics: {result.n_metrics_abnormal}/{result.n_metrics_valid}")
    print(f"Output folder: {output_dir}")
    print("Generated:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
