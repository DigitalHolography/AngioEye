import operator
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)

@dataclass(frozen=True)
class MetricConfig:
    name: str
    threshold: float
    threshold_type: Callable[[float, float], bool]
    control_std: dict[str, float]

    def path(self, vessel_type: str, representation: str) -> str:
        return METRIC_PATH.format(
            vessel_type=vessel_type,
            representation=representation,
            metric_name=self.name,
        )

    def is_value_abnormal(self, *values: float) -> bool:
        for value in values:
            arr = np.asarray(value, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size and np.any(self.threshold_type(arr, self.threshold)):
                return True
        return False


@dataclass(frozen=True)
class DomainConfig:
    metrics: tuple[str, ...]
    weight: float

    def weighted(self, artery: float, vein: float) -> tuple[float, float]:
        return (self.weight * artery, self.weight * vein)

@dataclass
class BiomarkersMetadata:
    # Pathology oriented thresholds. The type indicate where the range for abnormal values is located.
    threshold: float
    threshold_type: Callable[[float, float], bool]
    artery_value: Any = None
    vein_value: Any = None


METRIC_PATH = "/Pipelines/waveform_shape_metrics/{vessel_type}/global/{representation}/{metric_name}"

@registerPostprocess(
    name="Composite Waveform Biomarkers",
    description=(
        "Appends a composite biomarker from dimensionless retinal waveform shape metrics "
        "under /Pipelines/composite_biomarker" # TODO - update naming of biomarker and this description
    ),
    required_pipelines=["waveform_shape_metrics"],
)
class CompositeWaveformBiomarkersPostprocess(BatchPostprocess):
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

    METRICS: dict[str, MetricConfig] = {
        M_STROKE_FRACTION: MetricConfig(
            name="SF_VTI",
            threshold=0.5,
            threshold_type=operator.ge,
            control_std={"artery": 0.02130616468479075, "vein": 0.012238142379889327},
        ),
        M_MED_DISPLACEMENT_TIMING: MetricConfig(
            name="t50_over_T",
            threshold=0.36,
            threshold_type=operator.le,
            control_std={"artery": 0.02170372191846459, "vein": 0.011114571157947383},
        ),
        M_LOW_FREQ_SPECTRAL_FRACTION: MetricConfig(
            name="E_low_over_E_total",
            threshold=0.76,
            threshold_type=operator.ge,
            control_std={"artery": 0.0669936550252201, "vein": 0.06952508377563454},
        ),
        M_LATE_CYCLE_MEAN_FRACTION: MetricConfig(
            name="v_end_over_v_mean",
            threshold=0.59,
            threshold_type=operator.le,
            control_std={"artery": 0.06584969658865907, "vein": 0.0382835422777238},
        ),
        M_PARTICIPATION_RATIO_EFF_SUPP: MetricConfig(
            name="N_eff_over_T",
            threshold=0.90,
            threshold_type=operator.le,
            control_std={"artery": 0.02584899777470274, "vein": 0.0055469432407653264},
        ),
        M_RESISTIVITY_INDEX: MetricConfig(
            name="RI",
            threshold=0.75,
            threshold_type=operator.ge,
            control_std={"artery": 0.08357600130504828, "vein": 0.063285564839462},
        ),
        M_PULSATILITY_INDEX: MetricConfig(
            name="PI",
            threshold=1.30,
            threshold_type=operator.ge,
            control_std={"artery": 0.2003058879383459, "vein": 0.07992210522433744},
        ),
        M_NEAR_PEAK_CREST_WIDTH: MetricConfig(
            name="W50_over_T",
            threshold=0.60,
            threshold_type=operator.le,
            control_std={"artery": 0.13043404441873044, "vein": 0.003223005055989731},
        ),
    }

    DOMAINS: dict[str, DomainConfig] = {
        "timing": DomainConfig(metrics=(M_STROKE_FRACTION, M_MED_DISPLACEMENT_TIMING), weight=1),
        "spectral": DomainConfig(metrics=(M_LOW_FREQ_SPECTRAL_FRACTION,), weight=1.5),
        "persistence": DomainConfig(
            metrics=(M_LATE_CYCLE_MEAN_FRACTION, M_PARTICIPATION_RATIO_EFF_SUPP, M_NEAR_PEAK_CREST_WIDTH),
            weight=1,
        ),
        "pulsatility": DomainConfig(metrics=(M_RESISTIVITY_INDEX, M_PULSATILITY_INDEX), weight=1),
    }

    def _evaluate_threshold_excess_severity(
        self,
        metric_name: str,
        metric: BiomarkersMetadata,
    ) -> tuple[float, float]:
        """
        Evaluates the severity of the threshold excess for a given metric.
        Returns a tuple of (artery_excess, vein_excess) where each excess is a value between 0 and 1. 
        """
        metric_config = self.METRICS[metric_name]
        artery_cntrl_std = metric_config.control_std["artery"]
        vein_cntrl_std = metric_config.control_std["vein"]
        direction = 1 if metric_config.threshold_type in (operator.gt, operator.ge) else -1

        def _excess(value: Any, cntrl_std: float) -> float:
            arr = np.asarray(value, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0

            threshold_deviation = direction * (arr - metric.threshold)
            normalized = np.maximum(0.0, threshold_deviation / cntrl_std)
            return float(np.nanmax(normalized))

        return (
            _excess(metric.artery_value, artery_cntrl_std),
            _excess(metric.vein_value, vein_cntrl_std),
        )
    
    def _get_domain_collapsed_contiuous_composite(
        self,
        bmk_dict: dict[str, BiomarkersMetadata],
    ) -> dict[str, tuple[float, float]]:
        def _domain_max(domain: DomainConfig) -> tuple[float, float]:
            severities = [
                self._evaluate_threshold_excess_severity(name, bmk_dict[name])
                for name in domain.metrics
            ]
            return (
                max(severity[0] for severity in severities),
                max(severity[1] for severity in severities),
            )

        return {
            domain_name: _domain_max(domain)
            for domain_name, domain in self.DOMAINS.items()
        }
    
    def _calculate_rwas_score(self, composite_scores: dict[str, tuple[float, float]]) -> float:
        """Calculates RWAS based on weighted domain composite scores."""
        artery, vein = zip(*(
            self.DOMAINS[domain_name].weighted(*scores)
            for domain_name, scores in composite_scores.items()
        ))
        return max(sum(artery), sum(vein))
    
    def _calculate_rwas4(self, bmk_dict: dict[str, BiomarkersMetadata]) -> float:
        """Returns the sum of scores from the timing, spectral, persistence, and pulsatility abnormality binary-domains.
        Hence RWAS4 is a discrete score from 0 to 4 indicating the number of domains in which the patient has an abnormality."""

        def _is_abnormal(metric_names: list[str]) -> bool:
            return any(
                self.METRICS[metric_name].is_value_abnormal(
                    bmk_dict[metric_name].artery_value,
                    bmk_dict[metric_name].vein_value,
                )
                for metric_name in metric_names
            )
        
        return sum(
            _is_abnormal(list(domain.metrics))
            for domain in self.DOMAINS.values()
        )


    def _find_waveform_shape_metrics_group(self, pipelines: h5py.Group):
        for name, group in pipelines.items():
            if isinstance(group, h5py.Group) and group.attrs.get("pipeline", "") == "waveform_shape_metrics":
                return group
        # fallback if a user manually created an exact-case group
        if "waveform_shape_metrics" in pipelines and isinstance(pipelines["waveform_shape_metrics"], h5py.Group):
            return pipelines["waveform_shape_metrics"]
        return None

    def _resolve_target(self, root_group: h5py.Group, key: str) -> tuple[h5py.Group, str]:
        parts = [part for part in str(key).replace("\\", "/").strip("/").split("/") if part]
        if not parts:
            raise ValueError("Empty target key.")
        parent = root_group
        for part in parts[:-1]:
            parent = parent[part] if part in parent else parent.create_group(part)
            if not isinstance(parent, h5py.Group):
                raise ValueError(f"Cannot create subgroup under non-group path: {key}")
        return parent, parts[-1]

    def _write_value(self, root_group: h5py.Group, key: str, value: Any) -> None:
        parent, leaf = self._resolve_target(root_group, key)
        if leaf in parent:
            del parent[leaf]

        str_dtype = h5py.string_dtype(encoding="utf-8")
        if isinstance(value, str):
            parent.create_dataset(leaf, data=value, dtype=str_dtype)
            return
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            parent.create_dataset(leaf, data=np.asarray(value, dtype=str_dtype))
            return
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
            parent.create_dataset(leaf, data=value.astype(str_dtype))
            return

        try:
            parent.create_dataset(leaf, data=value)
        except (TypeError, ValueError):
            parent.create_dataset(leaf, data=str(value), dtype=str_dtype)

    def _collect_representation_biomarkers(
        self,
        h5: h5py.File,
        representation: str,
    ) -> dict[str, BiomarkersMetadata] | None:
        bmk_dict: dict[str, BiomarkersMetadata] = {}
        for metric_name, metric_config in self.METRICS.items():
            artery_path = metric_config.path("artery", representation)
            vein_path = metric_config.path("vein", representation)
            if artery_path not in h5 or vein_path not in h5:
                return None

            artery_value = h5[artery_path][()]
            vein_value = h5[vein_path][()]
            bmk_dict[metric_name] = BiomarkersMetadata(
                threshold=metric_config.threshold,
                threshold_type=metric_config.threshold_type,
                artery_value=artery_value,
                vein_value=vein_value,
            )
        return bmk_dict

    def _append_scores_to_file(self, file_path: Path) -> None:
        with h5py.File(file_path, "r+") as h5:
            pipelines = h5["Pipelines"] if "Pipelines" in h5 else h5.create_group("Pipelines")
            source_group = self._find_waveform_shape_metrics_group(pipelines)
            if source_group is None:
                raise ValueError(f"Expected 'waveform_shape_metrics' pipeline group not found in {file_path}")

            if "composite_biomarker" in pipelines:
                del pipelines["composite_biomarker"]
            composite_group = pipelines.create_group("composite_biomarker")
            composite_group.attrs["pipeline"] = "composite_biomarker"
            composite_group.attrs["source_pipeline"] = str(source_group.name)

            # Create placeholder by_segment groups for each vessel and representation
            for vessel_type in self.VESSEL_TYPES:
                for representation in self.REPRESENTATIONS:
                    composite_group.require_group(f"{vessel_type}/by_segment/{representation}")

            for representation in self.REPRESENTATIONS:
                bmk_dict = self._collect_representation_biomarkers(h5, representation)
                if bmk_dict is None:
                    continue

                composite_scores = self._get_domain_collapsed_contiuous_composite(bmk_dict)
                rwas_score = self._calculate_rwas_score(composite_scores)
                rwas4_score = self._calculate_rwas4(bmk_dict)

                for vessel_type in self.VESSEL_TYPES:
                    base = f"{vessel_type}/global/{representation}"

                    self._write_value(composite_group, f"{base}/RWAS", np.asarray(rwas_score, dtype=float))
                    self._write_value(composite_group, f"{base}/RWAS4", np.asarray(rwas4_score, dtype=int))

                    for domain_name, (artery_score, vein_score) in composite_scores.items():
                        vessel_score = artery_score if vessel_type == "artery" else vein_score
                        weighted_score = self.DOMAINS[domain_name].weight * vessel_score
                        self._write_value(composite_group, f"{base}/domain_scores/{domain_name}/score", np.asarray(vessel_score, dtype=float))
                        self._write_value(composite_group, f"{base}/domain_scores/{domain_name}/weighted_score", np.asarray(weighted_score, dtype=float))

                    for metric_name, metric_config in self.METRICS.items():
                        metric_data = bmk_dict[metric_name]
                        vessel_value = metric_data.artery_value if vessel_type == "artery" else metric_data.vein_value
                        abnormal = metric_config.is_value_abnormal(vessel_value)
                        self._write_value(composite_group, f"{base}/inputs/{metric_name}/value", np.asarray(vessel_value, dtype=float))
                        self._write_value(composite_group, f"{base}/inputs/{metric_name}/is_abnormal", np.asarray(abnormal, dtype=bool))
                        self._write_value(composite_group, f"{base}/inputs/{metric_name}/source_path", metric_config.path(vessel_type, representation))

            # for metric_name, metric_config in self.METRICS.items():
            #     self._write_value(composite_group, f"params/metrics/{metric_name}/h5_metric", metric_config.name)
            #     self._write_value(composite_group, f"params/metrics/{metric_name}/threshold", np.asarray(metric_config.threshold, dtype=float))
            #     self._write_value(composite_group, f"params/metrics/{metric_name}/threshold_direction", metric_config.threshold_type.__name__)
            #     self._write_value(composite_group, f"params/metrics/{metric_name}/control_std/artery", np.asarray(metric_config.control_std["artery"], dtype=float))
            #     self._write_value(composite_group, f"params/metrics/{metric_name}/control_std/vein", np.asarray(metric_config.control_std["vein"], dtype=float))
            #     for representation in self.REPRESENTATIONS:
            #         self._write_value(composite_group, f"params/metrics/{metric_name}/source_path/{representation}/artery", metric_config.path("artery", representation))
            #         self._write_value(composite_group, f"params/metrics/{metric_name}/source_path/{representation}/vein", metric_config.path("vein", representation))

            # for domain_name, domain in self.DOMAINS.items():
            #     self._write_value(composite_group, f"params/domains/{domain_name}/weight", np.asarray(domain.weight, dtype=float))
            #     self._write_value(composite_group, f"params/domains/{domain_name}/metrics", list(domain.metrics))

    
    def run(self, context: PostprocessContext) -> PostprocessResult:
        updated_paths: list[str] = []
        for file_path in context.processed_files:
            self._append_scores_to_file(file_path)
            updated_paths.append(str(file_path))

        return PostprocessResult(
            summary=(
                f"Appended Composite Waveform Biomarkers to {len(updated_paths)} "
                "processed HDF5 file(s)."
            ),
            generated_paths=updated_paths,
        )