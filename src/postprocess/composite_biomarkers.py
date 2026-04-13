from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from input_output.hdf5_io import (
    MetricsTree,
    append_metrics_trees_to_h5,
    read_dataset,
    safe_h5_key,
)
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT, find_pipeline_group

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)
from .core.grouped_batch import build_group_order, extract_group_name


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


METRIC_PATH = "{vessel_type}/global/{representation}/{metric_name}"
POSTPROCESS_GROUP = "composite_scoring"
PLOT_VESSEL_TYPE = "artery"
GREATER = 1
LESS = -1


def _finite_values(value: Any) -> np.ndarray:
    values = np.asarray(value, dtype=float).ravel()
    return values[np.isfinite(values)]


def _has_abnormal_value(value: Any, metric: Metric) -> bool:
    values = _finite_values(value)
    if values.size == 0:
        return False
    if metric.direction == GREATER:
        return bool(np.any(values >= metric.threshold))
    return bool(np.any(values <= metric.threshold))


def _severity(value: Any, metric: Metric, vessel_type: str) -> float:
    values = _finite_values(value)
    if values.size == 0:
        return 0.0

    deviation = metric.direction * (values - metric.threshold)
    normalized = np.maximum(0.0, deviation / metric.control_std[vessel_type])
    return float(np.nanmax(normalized))


@registerPostprocess(
    name="Composite Scoring",
    description=(
        "Appends composite RWAS/RWAS4 scores from dimensionless retinal waveform "
        "shape metrics and writes cohort score visualizations under png/."
    ),
    required_deps=["matplotlib>=3.8"],
    required_pipelines=["waveform_shape_metrics"],
)
class CompositeScoringPostprocess(BatchPostprocess):
    REPRESENTATIONS = ("raw", "bandlimited")
    VESSEL_TYPES = ("artery", "vein")

    M_STROKE_FRACTION = "stroke_fraction"
    M_MED_DISPLACEMENT_TIMING = "med_displacement_timing"
    M_LOW_FREQ_SPECTRAL_FRACTION = "low_freq_spectral_fraction"
    M_LATE_CYCLE_MEAN_FRACTION = "late_cycle_mean_fraction"
    M_PARTICIPATION_RATIO_EFF_SUPP = "participation_ratio_eff_supp"
    M_RESISTIVITY_INDEX = "resistivity_index"
    M_PULSATILITY_INDEX = "pulsatility_index"
    M_NEAR_PEAK_CREST_WIDTH = "near_peak_crest_width"

    METRICS: dict[str, Metric] = {
        M_STROKE_FRACTION: Metric(
            name="SF_VTI",
            threshold=0.5,
            direction=GREATER,
            control_std={"artery": 0.02130616468479075, "vein": 0.012238142379889327},
        ),
        M_MED_DISPLACEMENT_TIMING: Metric(
            name="t50_over_T",
            threshold=0.36,
            direction=LESS,
            control_std={"artery": 0.02170372191846459, "vein": 0.011114571157947383},
        ),
        M_LOW_FREQ_SPECTRAL_FRACTION: Metric(
            name="E_low_over_E_total",
            threshold=0.76,
            direction=GREATER,
            control_std={"artery": 0.0669936550252201, "vein": 0.06952508377563454},
            numerator_name="E_low",
            denominator_name="E_total",
        ),
        M_LATE_CYCLE_MEAN_FRACTION: Metric(
            name="v_end_over_vbar",
            threshold=0.59,
            direction=LESS,
            control_std={"artery": 0.06584969658865907, "vein": 0.0382835422777238},
        ),
        M_PARTICIPATION_RATIO_EFF_SUPP: Metric(
            name="N_eff_over_T",
            threshold=0.90,
            direction=LESS,
            control_std={"artery": 0.02584899777470274, "vein": 0.0055469432407653264},
        ),
        M_RESISTIVITY_INDEX: Metric(
            name="RI",
            threshold=0.75,
            direction=GREATER,
            control_std={"artery": 0.08357600130504828, "vein": 0.063285564839462},
        ),
        M_PULSATILITY_INDEX: Metric(
            name="PI",
            threshold=1.30,
            direction=GREATER,
            control_std={"artery": 0.2003058879383459, "vein": 0.07992210522433744},
        ),
        M_NEAR_PEAK_CREST_WIDTH: Metric(
            name="W50_over_T",
            threshold=0.60,
            direction=LESS,
            control_std={"artery": 0.13043404441873044, "vein": 0.003223005055989731},
        ),
    }

    DOMAINS: dict[str, Domain] = {
        "timing": Domain(
            metrics=(M_STROKE_FRACTION, M_MED_DISPLACEMENT_TIMING),
            weight=1,
        ),
        "spectral": Domain(metrics=(M_LOW_FREQ_SPECTRAL_FRACTION,), weight=1.5),
        "persistence": Domain(
            metrics=(
                M_LATE_CYCLE_MEAN_FRACTION,
                M_PARTICIPATION_RATIO_EFF_SUPP,
                M_NEAR_PEAK_CREST_WIDTH,
            ),
            weight=1,
        ),
        "pulsatility": Domain(
            metrics=(M_RESISTIVITY_INDEX, M_PULSATILITY_INDEX),
            weight=1,
        ),
    }

    def _build_scores_tree(self, file_path: Path) -> MetricsTree:
        with h5py.File(file_path, "r") as h5:
            source_group = find_pipeline_group(h5, "waveform_shape_metrics")
            if source_group is None:
                raise ValueError(
                    "Expected 'waveform_shape_metrics' pipeline group not found "
                    f"in {file_path}"
                )

            metrics: dict[str, Any] = {}
            for representation in self.REPRESENTATIONS:
                values: dict[tuple[str, str], Any] = {}
                missing_input = False
                for vessel_type in self.VESSEL_TYPES:
                    for metric_key, metric in self.METRICS.items():
                        paths = metric.derived_paths(vessel_type, representation)
                        if paths is None:
                            value = read_dataset(
                                source_group,
                                metric.path(vessel_type, representation),
                                default=None,
                            )
                        else:
                            numerator = read_dataset(
                                source_group,
                                paths[0],
                                default=None,
                            )
                            denominator = read_dataset(
                                source_group,
                                paths[1],
                                default=None,
                            )
                            if numerator is None or denominator is None:
                                value = None
                            else:
                                numerator = np.asarray(numerator, dtype=float)
                                denominator = np.asarray(denominator, dtype=float)
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    value = np.where(
                                        np.isfinite(denominator) & (denominator != 0),
                                        numerator / denominator,
                                        np.nan,
                                    )
                        if value is None:
                            missing_input = True
                            break
                        values[(vessel_type, metric_key)] = value
                    if missing_input:
                        break

                if missing_input:
                    continue

                weighted_scores = {"artery": 0.0, "vein": 0.0}
                rwas4_score = 0
                for domain in self.DOMAINS.values():
                    domain_has_abnormality = False
                    for vessel_type in self.VESSEL_TYPES:
                        domain_severity = max(
                            _severity(
                                values[(vessel_type, metric_key)],
                                self.METRICS[metric_key],
                                vessel_type,
                            )
                            for metric_key in domain.metrics
                        )
                        weighted_scores[vessel_type] += domain.weight * domain_severity
                        domain_has_abnormality = domain_has_abnormality or any(
                            _has_abnormal_value(
                                values[(vessel_type, metric_key)],
                                self.METRICS[metric_key],
                            )
                            for metric_key in domain.metrics
                        )
                    rwas4_score += int(domain_has_abnormality)

                rwas_score = max(weighted_scores.values())

                for vessel_type in self.VESSEL_TYPES:
                    base = f"{vessel_type}/global/{representation}"

                    metrics[f"{base}/RWAS"] = np.asarray(rwas_score, dtype=float)
                    metrics[f"{base}/RWAS4"] = np.asarray(rwas4_score, dtype=int)

            return MetricsTree(
                name=POSTPROCESS_GROUP,
                metrics=metrics,
                attrs={
                    "kind": "postprocess",
                    "source_pipeline": str(source_group.name),
                },
            )

    def _append_scores_to_file(self, file_path: Path) -> MetricsTree:
        tree = self._build_scores_tree(file_path)
        append_metrics_trees_to_h5(
            file_path,
            ANGIOEYE_POSTPROCESS_ROOT,
            [tree],
            overwrite=True,
        )
        with h5py.File(file_path, "r+") as h5:
            composite_group = h5[f"{ANGIOEYE_POSTPROCESS_ROOT}/{POSTPROCESS_GROUP}"]
            for vessel_type in self.VESSEL_TYPES:
                for representation in self.REPRESENTATIONS:
                    composite_group.require_group(
                        f"{vessel_type}/by_segment/{representation}"
                    )
        return tree

    def _score_records_for_tree(
        self,
        tree: MetricsTree,
        *,
        cohort: str,
        file_path: Path,
    ) -> list[ScoreRecord]:
        records: list[ScoreRecord] = []
        for representation in self.REPRESENTATIONS:
            base = f"{PLOT_VESSEL_TYPE}/global/{representation}"
            rwas = _finite_scalar(tree.metrics.get(f"{base}/RWAS"))
            rwas4 = _finite_scalar(tree.metrics.get(f"{base}/RWAS4"))
            if rwas is None or rwas4 is None:
                continue
            records.append(
                ScoreRecord(
                    cohort=cohort,
                    file_name=file_path.name,
                    representation=representation,
                    rwas=rwas,
                    rwas4=rwas4,
                )
            )
        return records

    def _write_score_plots(
        self,
        records: list[ScoreRecord],
        output_dir: Path,
    ) -> list[str]:
        if not records:
            return []

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        png_dir = output_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        created_paths: list[str] = []

        for representation in self.REPRESENTATIONS:
            representation_records = [
                record
                for record in records
                if record.representation == representation
            ]
            if not representation_records:
                continue
            created_paths.extend(
                self._plot_score_by_cohort(
                    representation_records,
                    score_name="RWAS",
                    value_getter=lambda record: record.rwas,
                    representation=representation,
                    png_dir=png_dir,
                    plt=plt,
                )
            )
            created_paths.append(
                self._plot_score_violin_by_cohort(
                    representation_records,
                    score_name="RWAS",
                    value_getter=lambda record: record.rwas,
                    representation=representation,
                    png_dir=png_dir,
                    plt=plt,
                )
            )
            created_paths.extend(
                self._plot_score_by_cohort(
                    representation_records,
                    score_name="RWAS4",
                    value_getter=lambda record: record.rwas4,
                    representation=representation,
                    png_dir=png_dir,
                    plt=plt,
                )
            )
            created_paths.append(
                self._plot_score_violin_by_cohort(
                    representation_records,
                    score_name="RWAS4",
                    value_getter=lambda record: record.rwas4,
                    representation=representation,
                    png_dir=png_dir,
                    plt=plt,
                )
            )
            created_paths.append(
                self._plot_rwas_scatter(
                    representation_records,
                    representation=representation,
                    png_dir=png_dir,
                    plt=plt,
                )
            )

        return created_paths

    def _plot_score_by_cohort(
        self,
        records: list[ScoreRecord],
        *,
        score_name: str,
        value_getter,
        representation: str,
        png_dir: Path,
        plt,
    ) -> list[str]:
        cohorts = build_group_order({record.cohort for record in records})
        positions = {cohort: index for index, cohort in enumerate(cohorts)}

        fig, ax = plt.subplots(figsize=(max(6.0, len(cohorts) * 1.1), 4.6))
        cmap = plt.get_cmap("tab10")
        for cohort_index, cohort in enumerate(cohorts):
            cohort_records = [
                record for record in records if record.cohort == cohort
            ]
            if not cohort_records:
                continue
            values = [value_getter(record) for record in cohort_records]
            x_values = [
                positions[cohort] + _deterministic_jitter(index)
                for index, _record in enumerate(cohort_records)
            ]
            ax.scatter(
                x_values,
                values,
                label=cohort,
                color=cmap(cohort_index % 10),
                alpha=0.82,
                edgecolors="black",
                linewidths=0.35,
                s=42,
            )

        ax.set_title(f"{score_name} by cohort ({representation})")
        ax.set_xlabel("Cohort")
        ax.set_ylabel(score_name)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohorts, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if score_name == "RWAS4":
            ax.set_yticks(range(5))
        fig.tight_layout()

        output_path = png_dir / (
            f"composite_scoring_{safe_h5_key(representation)}_"
            f"{safe_h5_key(score_name)}_by_cohort.png"
        )
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return [str(output_path)]

    def _plot_score_violin_by_cohort(
        self,
        records: list[ScoreRecord],
        *,
        score_name: str,
        value_getter,
        representation: str,
        png_dir: Path,
        plt,
    ) -> str:
        cohorts = build_group_order({record.cohort for record in records})

        fig, ax = plt.subplots(figsize=(max(6.4, len(cohorts) * 1.3), 4.9))
        cmap = plt.get_cmap("tab10")
        for cohort_index, cohort in enumerate(cohorts):
            cohort_records = [
                record for record in records if record.cohort == cohort
            ]
            if not cohort_records:
                continue

            position = cohort_index
            color = cmap(cohort_index % 10)
            values = np.asarray(
                [value_getter(record) for record in cohort_records],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            if values.size > 1 and float(np.nanmax(values) - np.nanmin(values)) > 0:
                parts = ax.violinplot(
                    [values],
                    positions=[position],
                    widths=0.78,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.34)
                    body.set_linewidth(0.8)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(1.4)
            else:
                ax.scatter(
                    [position],
                    [float(values[0])],
                    color=color,
                    marker="_",
                    s=520,
                    linewidths=2.0,
                    zorder=3,
                )

            x_values = [
                position + _centered_jitter(index, values.size, width=0.26)
                for index in range(values.size)
            ]
            ax.scatter(
                x_values,
                values,
                color=color,
                alpha=0.42,
                edgecolors="black",
                linewidths=0.25,
                s=24,
                zorder=4,
            )

            q1, median, q3 = np.percentile(values, [25, 50, 75])
            ax.vlines(position, q1, q3, color="black", linewidth=2.0, zorder=5)
            ax.scatter(
                [position],
                [median],
                color="white",
                edgecolors="black",
                linewidths=0.8,
                s=38,
                zorder=6,
            )

        ax.set_title(f"{score_name} concentration by cohort ({representation})")
        ax.set_xlabel("Cohort")
        ax.set_ylabel(score_name)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(
            [
                f"{cohort}\nn={sum(1 for record in records if record.cohort == cohort)}"
                for cohort in cohorts
            ],
        )
        ax.grid(axis="y", alpha=0.25)
        if score_name == "RWAS":
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylabel("RWAS (symlog)")
        else:
            ax.set_yticks(range(5))
        fig.tight_layout()

        output_path = png_dir / (
            f"composite_scoring_{safe_h5_key(representation)}_"
            f"{safe_h5_key(score_name)}_violin_by_cohort.png"
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return str(output_path)

    def _plot_rwas_scatter(
        self,
        records: list[ScoreRecord],
        *,
        representation: str,
        png_dir: Path,
        plt,
    ) -> str:
        cohorts = build_group_order({record.cohort for record in records})

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        cmap = plt.get_cmap("tab10")
        for cohort_index, cohort in enumerate(cohorts):
            cohort_records = [
                record for record in records if record.cohort == cohort
            ]
            if not cohort_records:
                continue
            ax.scatter(
                [record.rwas for record in cohort_records],
                [record.rwas4 for record in cohort_records],
                label=cohort,
                color=cmap(cohort_index % 10),
                alpha=0.82,
                edgecolors="black",
                linewidths=0.35,
                s=46,
            )

        ax.set_title(f"RWAS vs RWAS4 ({representation})")
        ax.set_xlabel("RWAS")
        ax.set_ylabel("RWAS4")
        ax.set_yticks(range(5))
        ax.grid(alpha=0.25)
        if len(cohorts) > 1:
            ax.legend(title="Cohort", loc="best", frameon=False)
        fig.tight_layout()

        output_path = png_dir / (
            f"composite_scoring_{safe_h5_key(representation)}_"
            "rwas_vs_rwas4_by_cohort.png"
        )
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return str(output_path)

    def run(self, context: PostprocessContext) -> PostprocessResult:
        updated_paths: list[str] = []
        score_records: list[ScoreRecord] = []
        failures: list[str] = []
        for file_path in context.processed_files:
            try:
                tree = self._append_scores_to_file(file_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"Composite Scoring skipped {file_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            cohort = extract_group_name(file_path.parent, context.output_dir)
            score_records.extend(
                self._score_records_for_tree(
                    tree,
                    cohort=cohort,
                    file_path=file_path,
                )
            )
            updated_paths.append(str(file_path))

        png_paths = self._write_score_plots(score_records, context.output_dir)
        return PostprocessResult(
            summary=(
                f"Appended Composite Scoring to {len(updated_paths)} processed HDF5 "
                f"file(s). Generated {len(png_paths)} PNG plot(s). "
                f"Skipped {len(failures)} file(s)."
            ),
            generated_paths=[*updated_paths, *png_paths],
            metadata={"failures": failures},
        )


def _finite_scalar(value: Any) -> float | None:
    values = _finite_values(value)
    if values.size == 0:
        return None
    return float(values[0])


def _deterministic_jitter(index: int) -> float:
    return ((index % 9) - 4) * 0.035


def _centered_jitter(index: int, count: int, *, width: float) -> float:
    if count <= 1:
        return 0.0
    slots = min(count, 31)
    slot = index % slots
    return ((slot / max(slots - 1, 1)) - 0.5) * width


CompositeWaveformBiomarkersPostprocess = CompositeScoringPostprocess

