from __future__ import annotations

from dataclasses import dataclass, field

METRIC_PATH = "{vessel_type}/global/{representation}/{metric_name}"


@dataclass(frozen=True)
class Metric:
    """Metric path definition plus calibrated scoring parameters.

    In the automatic version, the user only defines `name` and optionally
    `numerator_name` / `denominator_name`. `threshold`, `direction`, and
    `control_std` are filled by optimal split calibration.
    """

    name: str
    threshold: float = float("nan")
    direction: int = 0
    control_std: dict[str, float] = field(default_factory=dict)
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
class ScoreRecord:
    cohort: str
    file_name: str
    representation: str
    was: float
    was_c: float


@dataclass(frozen=True)
class MetricContributionRecord:
    cohort: str
    file_name: str
    vessel_type: str
    representation: str
    metric_key: str
    metric_name: str
    z: float
    z_capped: float
    was_points: float
    was_c_points: float
    threshold: float
    direction: int
    control_std: float
