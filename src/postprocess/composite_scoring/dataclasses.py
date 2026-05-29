from dataclasses import dataclass

METRIC_PATH = "{vessel_type}/global/{representation}/{metric_name}"

@dataclass(frozen=True)
class Metric:
    name: str
    threshold: float
    direction: int
    control_std: dict[str, float]
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
class Domain:
    metrics: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class ScoreRecord:
    cohort: str
    file_name: str
    representation: str
    rwas: float
    rwas4: float