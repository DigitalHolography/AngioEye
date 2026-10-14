from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

METRIC_PATH = "{vessel_type}/global/{representation}/{metric_name}"


@dataclass(frozen=True)
class FixedMetric:
    """One fixed-threshold metric used for patient evaluation."""

    key: str
    name: str
    threshold: float
    direction: int
    weight: float = 1.0
    control_std: float | None = None
    latex_name: str | None = None
    numerator_name: str | None = None
    denominator_name: str | None = None

    def path(self, vessel_type: str, representation: str) -> str:
        return METRIC_PATH.format(
            vessel_type=vessel_type,
            representation=representation,
            metric_name=self.name,
        )

    def derived_paths(
        self,
        vessel_type: str,
        representation: str,
    ) -> tuple[str, str] | None:
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
class MetricEvaluation:
    patient_id: str
    source_file: str
    archive_member: str | None
    vessel_type: str
    representation: str
    metric_key: str
    metric_name: str
    latex_name: str
    available: bool
    value: float
    threshold: float
    direction: int
    direction_label: str
    abnormal: bool
    control_std: float | None
    z: float
    z_capped: float
    weight: float
    weighted_contribution: float
    message: str = ""


@dataclass(frozen=True)
class PatientEvaluation:
    patient_id: str
    source_file: str
    archive_member: str | None
    h5_file_name: str
    vessel_type: str
    representation: str
    aggregation: str
    pathology_index: float
    pathology_index_percent: float
    was_c_equivalent: float
    abnormal_fraction: float
    n_metrics_configured: int
    n_metrics_available: int
    n_metrics_abnormal: int
    coverage_fraction: float
    evaluation_label: str
    metric_evaluations: tuple[MetricEvaluation, ...]


@dataclass(frozen=True)
class EvaluationFailure:
    source_file: str
    archive_member: str | None
    patient_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EvaluationRunResult:
    evaluations: tuple[PatientEvaluation, ...]
    failures: tuple[EvaluationFailure, ...]
    generated_paths: tuple[str, ...]
    output_dir: Path
