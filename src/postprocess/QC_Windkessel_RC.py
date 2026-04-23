from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="QC_Windkessel_RC",
    description=(
        "Append interpreted QC for Windkessel_RC directly into each processed HDF5 file "
        "under /AngioEye/QC_Windkessel_RC."
    ),
    required_pipelines=["Windkessel_RC"],
)
class QCWindkesselRC(BatchPostprocess):
    """
    Interpreted QC layer for Windkessel_RC.

    This postprocess consumes the intrinsic QC primitives stored by Windkessel_RC and
    appends a new HDF5 group per processed file.
    """

    min_valid_beats_per_method = 3
    min_consensus_fraction = 0.50
    min_reasonable_fraction = 0.50
    max_tau_intermethod_rel_range_median = 0.50
    max_delay_intermethod_range_median = 0.050
    max_residual_norm_median = 1.00
    max_freq_self_reldiff_median = 0.50

    def run(self, context: PostprocessContext) -> PostprocessResult:
        updated_paths: list[str] = []
        for file_path in context.processed_files:
            self._append_qc_to_file(file_path)
            updated_paths.append(str(file_path))

        return PostprocessResult(
            summary=(
                f"Appended QC_Windkessel_RC to {len(updated_paths)} processed HDF5 file(s)."
            ),
            generated_paths=updated_paths,
        )

    def _append_qc_to_file(self, file_path: Path) -> None:
        with h5py.File(file_path, "r+") as h5:
            pipelines = (
                h5["AngioEye"] if "AngioEye" in h5 else h5.create_group("AngioEye")
            )
            wind_group = self._find_windkessel_group(pipelines)
            if wind_group is None:
                raise ValueError(
                    f"Windkessel_RC pipeline group not found in processed file: {file_path}"
                )

            if "QC_Windkessel_RC" in pipelines:
                del pipelines["QC_Windkessel_RC"]
            qc_group = pipelines.create_group("QC_Windkessel_RC")
            qc_group.attrs["pipeline"] = "QC_Windkessel_RC"
            qc_group.attrs["source_pipeline"] = str(wind_group.name)

            rep_statuses: list[str] = []
            available_reps = 0
            for representation in ("raw", "bandlimited"):
                report = self._analyze_representation_group(wind_group, representation)
                if report is None:
                    continue
                available_reps += 1
                rep_statuses.append(report["status"])
                self._write_report_group(qc_group, representation, report)

            overall_status = self._overall_status(rep_statuses, available_reps)
            self._write_value(qc_group, "status", overall_status)
            self._write_value(
                qc_group,
                "available_representation_count",
                np.asarray(available_reps, dtype=int),
            )
            self._write_value(
                qc_group,
                "representation_statuses",
                np.asarray(rep_statuses, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            self._write_value(
                qc_group,
                "params/min_valid_beats_per_method",
                np.asarray(self.min_valid_beats_per_method, dtype=int),
            )
            self._write_value(
                qc_group,
                "params/min_consensus_fraction",
                np.asarray(self.min_consensus_fraction, dtype=float),
            )
            self._write_value(
                qc_group,
                "params/min_reasonable_fraction",
                np.asarray(self.min_reasonable_fraction, dtype=float),
            )
            self._write_value(
                qc_group,
                "params/max_tau_intermethod_rel_range_median",
                np.asarray(self.max_tau_intermethod_rel_range_median, dtype=float),
            )
            self._write_value(
                qc_group,
                "params/max_delay_intermethod_range_median",
                np.asarray(self.max_delay_intermethod_range_median, dtype=float),
            )
            self._write_value(
                qc_group,
                "params/max_residual_norm_median",
                np.asarray(self.max_residual_norm_median, dtype=float),
            )
            self._write_value(
                qc_group,
                "params/max_freq_self_reldiff_median",
                np.asarray(self.max_freq_self_reldiff_median, dtype=float),
            )

    def _find_windkessel_group(self, pipelines: h5py.Group):
        for name, group in pipelines.items():
            if (
                isinstance(group, h5py.Group)
                and group.attrs.get("pipeline", "") == "Windkessel_RC"
            ):
                return group
        # fallback if a user manually created an exact-case group
        if "Windkessel_RC" in pipelines and isinstance(
            pipelines["Windkessel_RC"], h5py.Group
        ):
            return pipelines["Windkessel_RC"]
        return None

    def _read(self, group: h5py.Group, path: str):
        try:
            return group[path][()]
        except Exception:
            return None

    def _safe_median(self, x) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmedian(x))

    def _safe_mean(self, x) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmean(x))

    def _safe_mad(self, x) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        med = float(np.nanmedian(x))
        return float(np.nanmedian(np.abs(x - med)))

    def _count_valid(self, x) -> int:
        x = np.asarray(x, dtype=float)
        return int(np.sum(np.isfinite(x)))

    def _analyze_representation_group(
        self, group: h5py.Group, representation: str
    ) -> dict[str, Any] | None:
        probe = self._read(group, f"{representation}/qc/qa_input_valid_fraction")
        if probe is None:
            return None

        methods: dict[str, dict[str, Any]] = {}
        flags: list[str] = []

        qa_input_valid_fraction = np.asarray(
            self._read(group, f"{representation}/qc/qa_input_valid_fraction"),
            dtype=float,
        )
        qv_input_valid_fraction = np.asarray(
            self._read(group, f"{representation}/qc/qv_input_valid_fraction"),
            dtype=float,
        )
        qa_prepared_valid_fraction = np.asarray(
            self._read(group, f"{representation}/qc/qa_prepared_valid_fraction"),
            dtype=float,
        )
        qv_prepared_valid_fraction = np.asarray(
            self._read(group, f"{representation}/qc/qv_prepared_valid_fraction"),
            dtype=float,
        )

        for method in ("freq", "time_integral", "arx"):
            tau = self._read(group, f"{representation}/{method}/tau")
            delay = self._read(group, f"{representation}/{method}/Deltat")
            residual_norm = self._read(
                group, f"{representation}/{method}/residual_norm"
            )
            accepted = self._read(group, f"{representation}/{method}/accepted")
            if tau is None or delay is None:
                continue

            tau = np.asarray(tau, dtype=float)
            delay = np.asarray(delay, dtype=float)
            residual_norm = (
                np.asarray(residual_norm, dtype=float)
                if residual_norm is not None
                else np.full_like(tau, np.nan)
            )
            accepted = (
                np.asarray(accepted, dtype=float)
                if accepted is not None
                else np.isfinite(tau).astype(float)
            )

            info: dict[str, Any] = {
                "n_valid_tau": self._count_valid(tau),
                "tau_median": self._safe_median(tau),
                "tau_mad": self._safe_mad(tau),
                "delay_median": self._safe_median(delay),
                "delay_mad": self._safe_mad(delay),
                "residual_norm_median": self._safe_median(residual_norm),
                "accepted_fraction": self._safe_mean(accepted),
            }

            if method == "freq":
                info["tau_phase_rel_diff_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/tau_phase_rel_diff")
                )
                info["tau_amp_rel_diff_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/tau_amp_rel_diff")
                )
                info["harmonics_used_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/harmonics_used")
                )
            if method == "arx":
                info["stability_margin_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/stability_margin")
                )
                info["rows_used_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/rows_used")
                )
            if method == "time_integral":
                info["rows_used_median"] = self._safe_median(
                    self._read(group, f"{representation}/{method}/rows_used")
                )

            methods[method] = info

        qc_summary = {
            "qa_input_valid_fraction_median": self._safe_median(
                qa_input_valid_fraction
            ),
            "qv_input_valid_fraction_median": self._safe_median(
                qv_input_valid_fraction
            ),
            "qa_prepared_valid_fraction_median": self._safe_median(
                qa_prepared_valid_fraction
            ),
            "qv_prepared_valid_fraction_median": self._safe_median(
                qv_prepared_valid_fraction
            ),
            "consensus_available_fraction": self._coalesce_scalar(
                group,
                representation,
                "consensus_available_fraction",
                "tau_consensus_available",
                mean_fallback=True,
            ),
            "methods_valid_count_median": self._coalesce_scalar(
                group,
                representation,
                "methods_valid_count_median",
                "methods_valid_count",
                mean_fallback=False,
            ),
            "tau_intermethod_rel_range_median": self._coalesce_scalar(
                group,
                representation,
                "tau_intermethod_rel_range_median",
                "tau_intermethod_rel_range",
                mean_fallback=False,
            ),
            "delay_intermethod_range_median": self._coalesce_scalar(
                group,
                representation,
                "delay_intermethod_range_median",
                "delay_intermethod_range",
                mean_fallback=False,
            ),
            "freq_tau_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "freq_tau_reasonable_fraction",
                "freq_tau_reasonable",
            ),
            "time_integral_tau_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "time_integral_tau_reasonable_fraction",
                "time_integral_tau_reasonable",
            ),
            "arx_tau_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "arx_tau_reasonable_fraction",
                "arx_tau_reasonable",
            ),
            "freq_delay_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "freq_delay_reasonable_fraction",
                "freq_delay_reasonable",
            ),
            "time_integral_delay_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "time_integral_delay_reasonable_fraction",
                "time_integral_delay_reasonable",
            ),
            "arx_delay_reasonable_fraction": self._coalesce_summary(
                group,
                representation,
                "arx_delay_reasonable_fraction",
                "arx_delay_reasonable",
            ),
        }

        if qc_summary["consensus_available_fraction"] < self.min_consensus_fraction:
            flags.append("low_consensus_fraction")
        if (
            np.isfinite(qc_summary["tau_intermethod_rel_range_median"])
            and qc_summary["tau_intermethod_rel_range_median"]
            > self.max_tau_intermethod_rel_range_median
        ):
            flags.append("high_tau_intermethod_disagreement")
        if (
            np.isfinite(qc_summary["delay_intermethod_range_median"])
            and qc_summary["delay_intermethod_range_median"]
            > self.max_delay_intermethod_range_median
        ):
            flags.append("high_delay_intermethod_disagreement")

        for method_name, info in methods.items():
            if info["n_valid_tau"] < self.min_valid_beats_per_method:
                flags.append(f"{method_name}_too_few_valid_beats")
            if (
                np.isfinite(info["residual_norm_median"])
                and info["residual_norm_median"] > self.max_residual_norm_median
            ):
                flags.append(f"{method_name}_high_residual_norm")
            if method_name == "freq":
                if (
                    np.isfinite(info.get("tau_phase_rel_diff_median", np.nan))
                    and info["tau_phase_rel_diff_median"]
                    > self.max_freq_self_reldiff_median
                ):
                    flags.append("freq_high_phase_self_disagreement")
                if (
                    np.isfinite(info.get("tau_amp_rel_diff_median", np.nan))
                    and info["tau_amp_rel_diff_median"]
                    > self.max_freq_self_reldiff_median
                ):
                    flags.append("freq_high_amplitude_self_disagreement")

        for name in (
            "freq_tau_reasonable_fraction",
            "time_integral_tau_reasonable_fraction",
            "arx_tau_reasonable_fraction",
            "freq_delay_reasonable_fraction",
            "time_integral_delay_reasonable_fraction",
            "arx_delay_reasonable_fraction",
        ):
            value = qc_summary[name]
            if np.isfinite(value) and value < self.min_reasonable_fraction:
                flags.append(name.replace("_fraction", "") + "_low_reasonable_fraction")

        status = self._status_from_flags(flags)
        return {
            "status": status,
            "flags": flags,
            "method_summaries": methods,
            "summary": qc_summary,
        }

    def _coalesce_scalar(
        self,
        group: h5py.Group,
        representation: str,
        summary_name: str,
        fallback_name: str,
        mean_fallback: bool,
    ) -> float:
        v = self._read(group, f"{representation}/qc/summary/{summary_name}")
        if v is not None:
            return float(np.asarray(v, dtype=float))
        arr = self._read(group, f"{representation}/qc/cross_method/{fallback_name}")
        return self._safe_mean(arr) if mean_fallback else self._safe_median(arr)

    def _coalesce_summary(
        self,
        group: h5py.Group,
        representation: str,
        summary_name: str,
        fallback_name: str,
    ) -> float:
        v = self._read(group, f"{representation}/qc/summary/{summary_name}")
        if v is not None:
            return float(np.asarray(v, dtype=float))
        arr = self._read(group, f"{representation}/qc/plausibility/{fallback_name}")
        return self._safe_mean(arr)

    def _status_from_flags(self, flags: list[str]) -> str:
        if not flags:
            return "ok"
        severe = any(
            flag
            in {
                "low_consensus_fraction",
                "high_tau_intermethod_disagreement",
                "high_delay_intermethod_disagreement",
            }
            or flag.endswith("too_few_valid_beats")
            or flag.endswith("high_residual_norm")
            for flag in flags
        )
        return "fail" if severe else "warn"

    def _overall_status(self, rep_statuses: list[str], available_reps: int) -> str:
        if available_reps == 0:
            return "missing_representations"
        if any(status == "fail" for status in rep_statuses):
            return "fail"
        if any(status == "warn" for status in rep_statuses):
            return "warn"
        return "ok"

    def _resolve_target(
        self, root_group: h5py.Group, key: str
    ) -> tuple[h5py.Group, str]:
        parts = [
            part for part in str(key).replace("\\", "/").strip("/").split("/") if part
        ]
        if not parts:
            raise ValueError("Empty target key.")
        parent = root_group
        for part in parts[:-1]:
            parent = parent[part] if part in parent else parent.create_group(part)
            if not isinstance(parent, h5py.Group):
                raise ValueError(f"Cannot create subgroup under non-group path: {key}")
        return parent, parts[-1]

    def _write_value(self, root_group: h5py.Group, key: str, value) -> None:
        parent, leaf = self._resolve_target(root_group, key)
        if leaf in parent:
            del parent[leaf]
        str_dtype = h5py.string_dtype(encoding="utf-8")
        if isinstance(value, str):
            parent.create_dataset(leaf, data=value, dtype=str_dtype)
            return
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, str) for v in value)
        ):
            parent.create_dataset(leaf, data=np.asarray(value, dtype=str_dtype))
            return
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
            parent.create_dataset(leaf, data=value.astype(str_dtype))
            return
        try:
            parent.create_dataset(leaf, data=value)
        except (TypeError, ValueError):
            parent.create_dataset(leaf, data=str(value), dtype=str_dtype)

    def _write_report_group(
        self, qc_group: h5py.Group, representation: str, report: dict[str, Any]
    ) -> None:
        base = f"{representation}"
        self._write_value(qc_group, f"{base}/status", report["status"])
        self._write_value(qc_group, f"{base}/flags", report["flags"])
        for key, value in report["summary"].items():
            self._write_value(
                qc_group, f"{base}/summary/{key}", np.asarray(value, dtype=float)
            )
        for method, info in report["method_summaries"].items():
            for key, value in info.items():
                self._write_value(
                    qc_group,
                    f"{base}/method_summaries/{method}/{key}",
                    np.asarray(value, dtype=float),
                )
