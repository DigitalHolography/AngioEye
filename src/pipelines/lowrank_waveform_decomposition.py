from pathlib import Path

import h5py
import numpy as np

from input_output.hdf5_io import create_h5_file, write_metrics_trees_to_h5
from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT

from .lowrank_svd_math import LowRankSVDMath
from .core.base import (
    ProcessPipeline,
    ProcessResult,
    process_result_to_metrics_tree,
    registerPipeline,
)


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformDecomposition(LowRankSVDMath, ProcessPipeline):
    """
    Low-rank SVD decomposition for beat-aligned arterial (and optionally
    venous) segment waveforms.

    For each acquisition, each enabled vessel, and each configured
    waveform source, the pipeline removes the local temporal mean,
    performs one joint SVD over all valid beat-location waveforms, and
    reports the four primary endpoints A1, rho1, A2, and rho2 -- plus,
    independently, a per-beat SVD robustness variant of the same
    endpoints (see "SVD METHOD -- beat-by-beat with endpoints" below).
    Beat period is shared across vessels (a single per-acquisition value
    used for both artery and vein), matching the convention in
    waveform_harmonic_organization.py.

    Vein processing is opt-in via ``process_veins`` (default False): set
    ``helper.process_veins = True`` (or subclass / construct with that
    attribute) before calling ``run`` / ``compute_acquisition_endpoints`` /
    ``write_acquisition_h5`` to also process venous segments.
    """

    description = (
        "Joint low-rank waveform decomposition from beat-aligned arterial "
        "(and optionally venous) segment waveforms, reporting A1, rho1, A2, "
        "rho2, and TPR per acquisition and per beat, for raw and bandlimited "
        "signals."
    )

    # When False (default), only artery segment sources are resolved and
    # processed. Set True to also include vein/raw and vein/bandlimited.
    process_veins = False

    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    v_band_segment_input = (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_raw_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )

    T_input_eyeflow = "EyeFlow/Processing/VelocityPerBeat/BeatPeriodSeconds/value"
    v_raw_segment_input_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/Raw/value"
    )
    v_band_segment_input_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
    )
    v_raw_segment_input_vein_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/Raw/value"
    )
    v_band_segment_input_vein_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
    )

    # =====================================================================
    # Schema resolution
    #
    # The pure-numpy SVD/endpoint math (safe-stat helpers, aggregation
    # reducers, singular-spectrum diagnostics, input shaping, the joint and
    # per-beat SVD, and the endpoint calculations built on them) lives in
    # LowRankSVDMath (lowrank_svd_math.py), mixed into this class below so
    # every `self.xxx(...)` call site here is unchanged -- only schema
    # resolution and metrics-tree export, which need h5py, stay here.
    # =====================================================================

    def _enabled_vessels(self) -> tuple[str, ...]:
        """Vessel compartments this instance will process (always includes
        artery; vein only when ``process_veins`` is True)."""
        return ("artery", "vein") if self.process_veins else ("artery",)

    def _filter_vessel_candidates(self, candidates: dict[str, str]) -> dict[str, str]:
        """Drop vein/* candidates when ``process_veins`` is False."""
        if self.process_veins:
            return candidates
        return {k: v for k, v in candidates.items() if not k.startswith("vein/")}

    def _resolve_vessel_sources(self, h5file) -> tuple[dict[str, str], str] | None:
        """Resolves the (vessel, representation) -> dataset_path candidate
        map for whichever schema family this file uses (current
        EyeFlow/... export schema first, falling back to the older
        Artery/Vein VelocityPerBeat schema), plus the shared beat-period
        path -- beat period is not vessel-specific. Vein candidates are
        omitted unless ``process_veins`` is True. Remaining candidate paths
        are returned whether or not they're actually present in h5file
        (e.g. artery/bandlimited might be missing); callers check
        membership themselves the same way run() always has, so a
        per-combo QC/missing flag can still be reported. Returns None if
        neither schema's beat-period path is present at all."""
        if self.T_input_eyeflow in h5file:
            candidates = {
                "artery/raw": self.v_raw_segment_input_eyeflow,
                "artery/bandlimited": self.v_band_segment_input_eyeflow,
                "vein/raw": self.v_raw_segment_input_vein_eyeflow,
                "vein/bandlimited": self.v_band_segment_input_vein_eyeflow,
            }
            return self._filter_vessel_candidates(candidates), self.T_input_eyeflow
        if self.T_input in h5file:
            candidates = {
                "artery/raw": self.v_raw_segment_input,
                "artery/bandlimited": self.v_band_segment_input,
                "vein/raw": self.v_raw_segment_input_vein,
                "vein/bandlimited": self.v_band_segment_input_vein,
            }
            return self._filter_vessel_candidates(candidates), self.T_input
        return None

    # =====================================================================
    # Metrics export
    # =====================================================================

    @staticmethod
    def _mode_label(m: int) -> str:
        return f"mode{m}"

    def _append_nan_mode_metrics(
        self,
        metrics: dict,
        prefix: str,
        mode_number: int,
        rep: dict,
    ) -> None:
        sh = rep["shape"]
        n_t = int(sh["n_t"])
        n_beats = int(sh["n_beats"])
        n_branches = int(sh["n_branches"])
        n_radii = int(sh["n_radii"])
        mode_key = self._mode_label(mode_number)

        metrics[f"{prefix}/decomposition/u_{mode_key}"] = np.full(
            (n_t,), np.nan, dtype=float
        )
        metrics[f"{prefix}/decomposition/scores_{mode_key}_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/rms/{mode_key}_amplitude_rms_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/residuals/r{mode_number}_t_bkr"] = np.full(
            (n_t, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/residuals/rms_r{mode_number}_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/A{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/R{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/rho{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/median_abs_a{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )

        for endpoint in ("A", "R", "rho", "median_abs_a"):
            metrics[f"{prefix}/endpoints/{endpoint}{mode_number}"] = np.asarray(
                np.nan, dtype=float
            )
        for stem in (
            "sigma_A",
            "mad_A",
            "cv_A",
            "sigma_R",
            "mad_R",
            "cv_R",
            "sigma_rho",
            "mad_rho",
            "cv_rho",
        ):
            metrics[f"{prefix}/variability/{stem}{mode_number}_beat"] = np.asarray(
                np.nan, dtype=float
            )
        metrics[
            f"{prefix}/variability/spatial_mad_A{mode_number}_median_over_beats"
        ] = np.asarray(np.nan, dtype=float)
        metrics[
            f"{prefix}/variability/spatial_mad_R{mode_number}_median_over_beats"
        ] = np.asarray(np.nan, dtype=float)

    def _append_config_metrics(
        self, metrics: dict, prefix: str, source_name: str, dataset_path: str
    ) -> None:
        metrics[f"{prefix}/config/signal_source"] = source_name
        metrics[f"{prefix}/config/input_dataset_path"] = dataset_path
        metrics[f"{prefix}/config/svd_method"] = "joint (t,bkr) SVD"
        metrics[f"{prefix}/config/aggregation"] = (
            "median over (k,r), then median over b"
        )
        metrics[f"{prefix}/config/max_exported_modes"] = np.asarray(
            self.exported_modes, dtype=int
        )
        metrics[f"{prefix}/config/min_valid_samples_fraction"] = np.asarray(
            self.min_valid_samples_fraction, dtype=float
        )
        metrics[f"{prefix}/config/min_valid_columns"] = np.asarray(
            self.min_valid_columns, dtype=int
        )

    def _append_input_metrics(self, metrics: dict, prefix: str, rep: dict) -> None:
        sh = rep["shape"]
        metrics[f"{prefix}/inputs/n_t"] = np.asarray(sh["n_t"], dtype=int)
        metrics[f"{prefix}/inputs/n_beats"] = np.asarray(sh["n_beats"], dtype=int)
        metrics[f"{prefix}/inputs/n_branches"] = np.asarray(
            sh["n_branches"], dtype=int
        )
        metrics[f"{prefix}/inputs/n_radii"] = np.asarray(sh["n_radii"], dtype=int)
        metrics[f"{prefix}/inputs/n_total_columns"] = np.asarray(
            sh["n_total_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs/n_valid_columns"] = np.asarray(
            sh["n_valid_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs/valid_fraction_columns"] = np.asarray(
            sh["n_valid_columns"] / float(max(1, sh["n_total_columns"])), dtype=float
        )
        metrics[f"{prefix}/inputs/finite_fraction_per_column_bkr"] = rep[
            "finite_fraction_per_column"
        ]
        metrics[f"{prefix}/inputs/valid_column_mask_bkr"] = rep[
            "valid_column_mask"
        ].astype(np.uint8)
        metrics[f"{prefix}/inputs/valid_columns_per_beat"] = rep[
            "valid_counts_per_beat"
        ]
        metrics[f"{prefix}/inputs/valid_fraction_columns_per_beat"] = rep[
            "valid_fraction_per_beat"
        ]
        metrics[f"{prefix}/inputs/beat_period_valid_b"] = rep[
            "beat_period_valid"
        ].astype(np.uint8)

    def _append_baseline_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
        """Non-modal endpoints (mu, beat period, TPR, MPR, mpr_prime) --
        these characterize the raw signal and don't depend on whether the
        SVD below is available."""
        metrics[f"{prefix}/baseline/mu_bkr"] = rep["mu"]
        metrics[f"{prefix}/baseline/mu_b"] = rep["beatwise"]["mu_b"]
        metrics[f"{prefix}/baseline/mu_acq"] = np.asarray(
            acq["mu_acq"], dtype=float
        )
        metrics[f"{prefix}/baseline/sigma_mu_beat"] = np.asarray(
            acq["sigma_mu_beat"], dtype=float
        )
        metrics[f"{prefix}/baseline/mad_mu_beat"] = np.asarray(
            acq["mad_mu_beat"], dtype=float
        )

        metrics[f"{prefix}/beat_period/mean"] = np.asarray(
            acq["beat_period_mean"], dtype=float
        )
        metrics[f"{prefix}/beat_period/median"] = np.asarray(
            acq["beat_period_median"], dtype=float
        )
        metrics[f"{prefix}/beat_period/std"] = np.asarray(
            acq["beat_period_std"], dtype=float
        )

        metrics[f"{prefix}/rms/total_pulsatile_rms_bkr"] = rep["total_rms_bkr"]
        metrics[f"{prefix}/beatwise/TPR_b"] = rep["beatwise"]["TPR_b"]
        metrics[f"{prefix}/endpoints/TPR"] = np.asarray(acq["TPR"], dtype=float)
        metrics[f"{prefix}/variability/sigma_TPR_beat"] = np.asarray(
            acq["sigma_TPR_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/mad_TPR_beat"] = np.asarray(
            acq["mad_TPR_beat"], dtype=float
        )

        metrics[f"{prefix}/rms/mean_pulsatile_ratio_bkr"] = rep[
            "mean_pulsatile_ratio_bkr"
        ]
        metrics[f"{prefix}/beatwise/mpr_b"] = rep["beatwise"]["mpr_b"]
        metrics[f"{prefix}/endpoints/mpr"] = np.asarray(acq["mpr"], dtype=float)
        metrics[f"{prefix}/variability/sigma_mpr_beat"] = np.asarray(
            acq["sigma_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/mad_mpr_beat"] = np.asarray(
            acq["mad_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/cv_mpr_beat"] = np.asarray(
            acq["cv_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/baseline/abs_mu_acq"] = np.asarray(
            acq["abs_mu_acq"], dtype=float
        )
        metrics[f"{prefix}/endpoints/mpr_prime"] = np.asarray(
            acq["mpr_prime"], dtype=float
        )

    def _append_qc_unavailable_metrics(
        self, metrics: dict, prefix: str, rep: dict
    ) -> None:
        metrics[f"{prefix}/qc/svd_available"] = np.asarray(0, dtype=np.uint8)
        metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "unknown"))
        metrics[f"{prefix}/qc/n_modes"] = np.asarray(0, dtype=int)
        metrics[f"{prefix}/qc/sign_flips_mode1to2"] = np.zeros(
            (self.exported_modes,), dtype=int
        )
        for m in range(1, self.exported_modes + 1):
            metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = np.asarray(
                1, dtype=np.uint8
            )
            self._append_nan_mode_metrics(metrics, prefix, m, rep)

    def _append_qc_available_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
        metrics[f"{prefix}/qc/svd_available"] = np.asarray(1, dtype=np.uint8)
        metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "ok"))
        metrics[f"{prefix}/qc/n_modes"] = np.asarray(rep["n_modes_panel"], dtype=int)
        sign_flips = np.zeros((self.exported_modes,), dtype=int)
        available_sign_flips = rep["sign_flips"][: self.exported_modes]
        sign_flips[: available_sign_flips.size] = available_sign_flips
        metrics[f"{prefix}/qc/sign_flips_mode1to2"] = sign_flips
        for m in range(1, self.exported_modes + 1):
            metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = np.asarray(
                int(not np.isfinite(acq.get(f"rho{m}", np.nan))), dtype=np.uint8
            )

    def _append_decomposition_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
        """Singular-spectrum diagnostics plus the per-mode panels (u,
        scores, amplitude/residual RMS, beatwise/endpoint/variability
        values) for every exported mode. Only called once QC has confirmed
        the SVD is available."""
        metrics[f"{prefix}/decomposition/singular_values"] = rep["s"]
        metrics[f"{prefix}/decomposition/singular_energy"] = rep["energy"]
        metrics[f"{prefix}/decomposition/singular_energy_fraction"] = rep[
            "energy_fraction"
        ]
        metrics[f"{prefix}/decomposition/effective_rank"] = np.asarray(
            acq["effective_rank"], dtype=float
        )
        metrics[f"{prefix}/decomposition/participation_ratio"] = np.asarray(
            acq["participation_ratio"], dtype=float
        )
        metrics[f"{prefix}/decomposition/alpha"] = np.asarray(
            acq["alpha"], dtype=float
        )
        metrics[f"{prefix}/decomposition/G1"] = np.asarray(
            acq["G1"], dtype=float
        )
        metrics[f"{prefix}/decomposition/eta1"] = np.asarray(acq["eta1"], dtype=float)
        metrics[f"{prefix}/decomposition/eta2"] = np.asarray(acq["eta2"], dtype=float)
        metrics[f"{prefix}/decomposition/eta12"] = np.asarray(
            acq["eta12"], dtype=float
        )

        for m in range(1, self.exported_modes + 1):
            idx = m - 1
            mode_key = self._mode_label(m)
            if rep["n_modes_panel"] < m:
                self._append_nan_mode_metrics(metrics, prefix, m, rep)
                continue

            metrics[f"{prefix}/decomposition/u_{mode_key}"] = rep["U_panel"][:, idx]
            metrics[f"{prefix}/decomposition/scores_{mode_key}_bkr"] = rep[
                "score_panel_bkr"
            ][idx]
            metrics[f"{prefix}/rms/{mode_key}_amplitude_rms_bkr"] = rep[
                "rms_mode_panel"
            ][idx]
            metrics[f"{prefix}/residuals/r{m}_t_bkr"] = rep["residual_t_bkr_panel"][
                idx
            ]
            metrics[f"{prefix}/residuals/rms_r{m}_bkr"] = rep["residual_rms_panel"][
                idx
            ]

            metrics[f"{prefix}/beatwise/A{m}_b"] = rep["beatwise"][f"A{m}_b"]
            metrics[f"{prefix}/beatwise/R{m}_b"] = rep["beatwise"][f"R{m}_b"]
            metrics[f"{prefix}/beatwise/rho{m}_b"] = rep["beatwise"][f"rho{m}_b"]
            metrics[f"{prefix}/beatwise/median_abs_a{m}_b"] = rep["beatwise"][
                f"median_abs_a{m}_b"
            ]

            metrics[f"{prefix}/endpoints/A{m}"] = np.asarray(
                acq[f"A{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/R{m}"] = np.asarray(
                acq[f"R{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/rho{m}"] = np.asarray(
                acq[f"rho{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/median_abs_a{m}"] = np.asarray(
                acq[f"median_abs_a{m}"], dtype=float
            )

            metrics[f"{prefix}/variability/sigma_A{m}_beat"] = np.asarray(
                acq[f"sigma_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_A{m}_beat"] = np.asarray(
                acq[f"mad_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_A{m}_beat"] = np.asarray(
                acq[f"cv_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/sigma_R{m}_beat"] = np.asarray(
                acq[f"sigma_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_R{m}_beat"] = np.asarray(
                acq[f"mad_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_R{m}_beat"] = np.asarray(
                acq[f"cv_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/sigma_rho{m}_beat"] = np.asarray(
                acq[f"sigma_rho{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_rho{m}_beat"] = np.asarray(
                acq[f"mad_rho{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_rho{m}_beat"] = np.asarray(
                acq[f"cv_rho{m}_beat"], dtype=float
            )
            metrics[
                f"{prefix}/variability/spatial_mad_A{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_A{m}_median_over_beats"], dtype=float)
            metrics[
                f"{prefix}/variability/spatial_mad_R{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_R{m}_median_over_beats"], dtype=float)

    def _append_representation_metrics(
        self,
        metrics: dict,
        source_name: str,
        dataset_path: str,
        rep: dict,
    ) -> None:
        """Orchestrates the config/inputs/baseline/QC/decomposition
        sub-methods above into the full metrics tree for one (vessel,
        representation) source: config -> inputs -> baseline -> QC (+
        decomposition, only if the SVD was available)."""
        prefix = source_name
        acq = rep["acq"]

        self._append_config_metrics(metrics, prefix, source_name, dataset_path)
        self._append_input_metrics(metrics, prefix, rep)
        self._append_baseline_metrics(metrics, prefix, rep, acq)

        if not rep.get("svd_available", False):
            self._append_qc_unavailable_metrics(metrics, prefix, rep)
            return

        self._append_qc_available_metrics(metrics, prefix, rep, acq)
        self._append_decomposition_metrics(metrics, prefix, rep, acq)

    def _append_per_beat_metrics(
        self,
        metrics: dict,
        source_name: str,
        per_beat_endpoints: dict,
    ) -> None:
        """Writes per_beat_svd_panels + _compute_per_beat_endpoints' per-beat
        endpoint sequences (A1_b_pb, A2_b_pb, R1_b_pb, R2_b_pb, rho1_b_pb,
        rho2_b_pb, TPR_b_pb, MPR_b_pb, effective_rank_b_pb,
        participation_ratio_b_pb, alpha_b_pb -- each a (n_beats,) array)
        into the metrics tree under
        {source_name}/per_beat/{key}, alongside the acquisition-level
        metrics _append_representation_metrics writes for the same
        source_name."""
        prefix = source_name
        for key, arr in per_beat_endpoints.items():
            metrics[f"{prefix}/per_beat/{key}"] = arr

    # =====================================================================
    # Entry points
    #
    # run() is the framework entry point (engine hands us an open h5file);
    # compute_acquisition_endpoints and write_acquisition_h5 are the
    # standalone, path-opening callers for use outside the pipeline_engine.
    # _build_attrs and _compute_source_metrics are shared by run() and
    # write_acquisition_h5(), which otherwise differ only in how they open
    # the file and what they do with the finished metrics dict.
    # =====================================================================

    def _build_attrs(self, representations: list[str], input_beat_period_path: str) -> dict:
        return {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD + per-beat (t,kr) SVD robustness variant",
            "aggregation": "median over (k,r), then median over b",
            "mode_panel_max": int(self.exported_modes),
            "vessels": list(self._enabled_vessels()),
            "process_veins": bool(self.process_veins),
            "representations": representations,
            "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
            "context_endpoint": "TPR",
            "input_beat_period_path": input_beat_period_path,
        }

    def _compute_source_metrics(
        self, h5file, candidates: dict[str, str], T: np.ndarray
    ) -> tuple[dict, list[str]]:
        """For every (vessel, representation) candidate present in h5file,
        computes the joint-SVD and per-beat-SVD metrics and appends them to
        one metrics dict. Returns (metrics, resolved), where resolved is the
        source names that were actually present and processed."""
        metrics: dict = {}
        resolved: list[str] = []
        for source_name, dataset_path in candidates.items():
            if dataset_path not in h5file:
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
                continue

            metrics[f"{source_name}/qc/input_available"] = np.asarray(
                1, dtype=np.uint8
            )
            v_block = np.asarray(h5file[dataset_path], dtype=float)

            # _mean_subtract already suppresses the "Mean of empty slice"
            # warning internally, so no outer catch_warnings is needed here.
            rep = self._compute_representation(v_block=v_block, T=T)
            self._append_representation_metrics(
                metrics=metrics,
                source_name=source_name,
                dataset_path=dataset_path,
                rep=rep,
            )

            panels = self.per_beat_svd_panels(v_block, T)
            per_beat_endpoints = self._compute_per_beat_endpoints(panels)
            self._append_per_beat_metrics(
                metrics=metrics,
                source_name=source_name,
                per_beat_endpoints=per_beat_endpoints,
            )

            resolved.append(source_name)

        return metrics, resolved

    def run(self, h5file) -> ProcessResult:
        """Framework entry point: the engine hands us an already-open
        h5file (never a path -- see compute_acquisition_endpoints/
        write_acquisition_h5 for the standalone, path-opening callers).
        Resolves the vessel/representation schema (current EyeFlow/...
        first, legacy Artery/Vein/... fallback) and, for every combination
        the file has, computes both the joint-SVD and per-beat-SVD
        endpoints, returning them as the pipeline's metrics tree."""
        legacy_candidates = self._filter_vessel_candidates(
            {
                "artery/raw": self.v_raw_segment_input,
                "artery/bandlimited": self.v_band_segment_input,
                "vein/raw": self.v_raw_segment_input_vein,
                "vein/bandlimited": self.v_band_segment_input_vein,
            }
        )
        schema = self._resolve_vessel_sources(h5file)
        if schema is None:
            metrics = {}
            for source_name, dataset_path in legacy_candidates.items():
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
            attrs = self._build_attrs([], self.T_input)
            return ProcessResult(metrics=metrics, attrs=attrs)

        candidates, t_path = schema
        T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))
        metrics, resolved = self._compute_source_metrics(h5file, candidates, T)
        attrs = self._build_attrs(resolved, t_path)
        return ProcessResult(metrics=metrics, attrs=attrs)

    def compute_acquisition_endpoints(self, h5_path) -> dict | None:
        """Standalone entry point for callers outside the AngioEye
        pipeline_engine framework (e.g. an external reproduction script):
        given a path to one acquisition's raw .h5 file, resolves whichever
        known schema it uses and, for each enabled vessel (artery always;
        vein only when ``process_veins`` is True), runs both the joint and
        per-beat SVDs on that vessel's raw segment.

        Returns None if the file has no known beat-period path or no
        enabled vessel's raw segment at all. Otherwise returns a dict keyed
        by enabled vessel name (``{"artery": {...} | None}``, and also
        ``"vein"`` when ``process_veins``), where a vessel is None when its
        raw segment is absent or its SVD was unavailable (too few valid
        columns). Each vessel dict carries the endpoints plus the raw "mu"
        and "energy_fraction" arrays some callers need directly."""
        with h5py.File(h5_path, "r") as h5file:
            schema = self._resolve_vessel_sources(h5file)
            if schema is None:
                return None
            candidates, t_path = schema
            T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))

            vessel_blocks: dict[str, np.ndarray | None] = {}
            for vessel in self._enabled_vessels():
                raw_path = candidates.get(f"{vessel}/raw")
                if raw_path is not None and raw_path in h5file:
                    vessel_blocks[vessel] = np.asarray(h5file[raw_path], dtype=float)
                else:
                    vessel_blocks[vessel] = None

        if all(v_block is None for v_block in vessel_blocks.values()):
            return None

        result: dict[str, dict | None] = {}
        for vessel, v_block in vessel_blocks.items():
            if v_block is None:
                result[vessel] = None
                continue

            v_block = self._ensure_segment_shape(v_block, T)

            # _mean_subtract already suppresses the "Mean of empty slice"
            # warning internally, so no outer catch_warnings is needed here.
            rep = self._compute_representation(v_block=v_block, T=T)
            if not rep.get("svd_available", False):
                result[vessel] = None
                continue

            per_beat_panels = self.per_beat_svd_panels(v_block, T)
            per_beat_svd = self._compute_per_beat_endpoints(per_beat_panels)
            valid_fraction_per_beat = np.asarray(
                rep.get("valid_fraction_per_beat", []), dtype=float
            )
            beat_period_arr = np.asarray(T[0], dtype=float)

            result[vessel] = {
                "acq": rep["acq"],
                "beatwise": rep["beatwise"],
                "per_beat_svd": per_beat_svd,
                "mu": rep["mu"],
                "energy_fraction": np.asarray(rep.get("energy_fraction", []), dtype=float),
                "beat_period_mean": (
                    float(np.nanmean(beat_period_arr))
                    if beat_period_arr.size
                    else float("nan")
                ),
                "beat_period_sd": (
                    float(np.nanstd(beat_period_arr, ddof=1))
                    if beat_period_arr.size > 1
                    else float("nan")
                ),
                "beat_period_b": beat_period_arr,
                "valid_fraction_per_beat": valid_fraction_per_beat,
                "n_valid_columns": int(rep["shape"]["n_valid_columns"]),
                "n_total_columns": int(rep["shape"]["n_total_columns"]),
            }

        return result

    def write_acquisition_h5(self, h5_path, out_path: Path | str) -> bool:
        """Standalone counterpart to run() for callers outside the
        pipeline_engine framework: writes the same metrics run() would
        produce -- acquisition-level joint-SVD plus per-beat SVD, for every
        enabled (vessel, representation) the file has (vein only when
        ``process_veins`` is True) -- to a new .h5 at out_path.

        Uses the same MetricsTree/ProcessResult conversion and
        ANGIOEYE_PROCESSING_ROOT group as the normal batch workflow, so the
        output is found where a downstream reader expects. Only the computed
        metrics are written, not the (often large) source waveform arrays.

        Returns False (and writes nothing) if the file has no known schema.
        """
        with h5py.File(h5_path, "r") as h5file:
            schema = self._resolve_vessel_sources(h5file)
            if schema is None:
                return False
            candidates, t_path = schema
            T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))
            metrics, resolved = self._compute_source_metrics(h5file, candidates, T)

        if not resolved:
            return False

        attrs = self._build_attrs(resolved, t_path)

        create_h5_file(out_path, source_file=str(h5_path), trim_source=True)
        metric_tree = process_result_to_metrics_tree(
            self.name, ProcessResult(metrics=metrics, attrs=attrs)
        )
        write_metrics_trees_to_h5(
            out_path, ANGIOEYE_PROCESSING_ROOT, [metric_tree], overwrite=False
        )
        return True
