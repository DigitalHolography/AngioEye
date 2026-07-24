import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs
from .lowrank_pulsatility_metrics import LowRankPulsatilityMetrics
from .waveform_shape_metrics import ArterialSegExample as WaveformShapeMetrics


@registerPipeline(name="canonical_mode_shape_metrics")
class CanonicalModeShapeMetrics(ProcessPipeline):
    """
    Shape descriptors of the canonical mode-m pulse template
    a_m(b,k,r)*u_m(t), stripped of both baseline mu and every other mode --
    as opposed to the same descriptors computed on the raw waveform in
    waveform_shape_metrics.py. Covers m=1 (Sec. VI.A) and m=2,3 (the
    Mode-2/3 residual-fraction finding under Sec. VI.C), since all three
    reconstructions come from the same SVD panel already computed by
    LowRankPulsatilityMetrics.

    Reuses LowRankPulsatilityMetrics._compute_representation for the SVD
    (the same decomposition already used for the A1/rho1/rho2/rho3
    endpoints) and ArterialSegExample._compute_metrics_1d from
    waveform_shape_metrics.py for the shape formulas themselves, so neither
    is reimplemented here.

    Note: _compute_metrics_1d rectifies its input (negative samples clipped
    to 0) before computing shape descriptors, since it was written for
    physically non-negative velocity waveforms. Each mode reconstruction is
    a mean-removed SVD component and can go negative, so any negative lobe
    of a_m*u_m is clipped away here, the same way it would be for any other
    consumer of _compute_metrics_1d.
    """

    description = (
        "Gain-invariant shape descriptors (W80/T, dicrotic-region width, "
        "crest factor, low/high-frequency energy ratio) evaluated on the "
        "pure mode-1/2/3 low-rank reconstructions rather than the raw "
        "waveform."
    )

    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    v_band_segment_input = (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )

    shape_metric_names = ("W80_over_T", "Q_d_width", "CF", "E_LF_over_E_HF")
    mode_indices = (1, 2, 3)

    def __init__(self) -> None:
        super().__init__()
        self._lowrank = LowRankPulsatilityMetrics()
        self._shape = WaveformShapeMetrics()

    def _mode_reconstruction_block(
        self, v_block: np.ndarray, T: np.ndarray, mode_index: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the pure mode-m reconstruction a_m(b,k,r)*u_m(t) as a
        (n_t, n_beats, n_branches, n_radii) block -- Eq. (4)-(6) restricted
        to the given mode_index, with mu and every other mode left out.
        """
        rep = self._lowrank._compute_representation(v_block=v_block, T=T)
        n_t, n_beats, n_branches, n_radii = v_block.shape
        valid_column_mask = np.zeros((n_beats, n_branches, n_radii), dtype=bool)
        recon = np.full((n_t, n_beats, n_branches, n_radii), np.nan, dtype=float)

        if not rep.get("svd_available", False):
            return recon, valid_column_mask
        if mode_index > rep.get("n_modes_panel", 0):
            return recon, valid_column_mask

        u_m = rep["U_panel"][:, mode_index - 1]
        a_m = rep["score_panel_flat"][mode_index - 1, :]
        valid_column_mask = rep["valid_column_mask"]
        recon[:, valid_column_mask] = np.outer(u_m, a_m)
        return recon, valid_column_mask

    def _compute_block(
        self, v_block: np.ndarray, T: np.ndarray, mode_index: int = 1
    ) -> dict:
        """
        Returns per-segment (beat, branch, radius) arrays for each shape
        descriptor, computed on the mode-m reconstruction of every valid
        column.
        """
        recon, valid_column_mask = self._mode_reconstruction_block(
            v_block, T, mode_index
        )
        n_beats, n_branches, n_radii = valid_column_mask.shape

        out = {
            name: np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            for name in self.shape_metric_names
        }

        for beat_idx, branch_idx, radius_idx in np.argwhere(valid_column_mask):
            Tbeat = float(T[0][beat_idx])
            pulse = recon[:, beat_idx, branch_idx, radius_idx]
            m = self._shape._compute_metrics_1d(pulse, Tbeat)
            for name in self.shape_metric_names:
                out[name][beat_idx, branch_idx, radius_idx] = m[name]

        return out

    def run(self, h5file) -> ProcessResult:
        metrics = {}
        T = self._lowrank._normalize_T(np.asarray(h5file[self.T_input]))

        rep_map = {
            "raw": self.v_raw_segment_input,
            "bandlimited": self.v_band_segment_input,
        }

        for rep_name, dataset_path in rep_map.items():
            if dataset_path not in h5file:
                metrics[f"{rep_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                continue

            metrics[f"{rep_name}/qc/input_available"] = np.asarray(1, dtype=np.uint8)
            v_block = np.asarray(h5file[dataset_path], dtype=float)

            for mode_index in self.mode_indices:
                seg = self._compute_block(v_block, T, mode_index)
                # Mode-1 keeps its original (unnamespaced) key paths for
                # backward compatibility with existing consumers (Sec. VI.A);
                # modes 2/3 are namespaced under mode{m}/ (Sec. VI.C).
                segment_prefix = (
                    f"{rep_name}/by_segment"
                    if mode_index == 1
                    else f"{rep_name}/by_segment/mode{mode_index}"
                )
                dots_prefix = (
                    f"{rep_name}/acquisition_dots"
                    if mode_index == 1
                    else f"{rep_name}/acquisition_dots/mode{mode_index}"
                )
                for name, arr in seg.items():
                    metrics[f"{segment_prefix}/{name}"] = with_attrs(
                        arr,
                        {
                            "definition": [
                                f"{name} of the canonical Mode-{mode_index} "
                                f"reconstruction a{mode_index}(b,k,r)*"
                                f"u{mode_index}(t), stored as "
                                "(beat, branch, radius)"
                            ],
                        },
                    )
                    metrics[f"{dots_prefix}/{name}"] = np.asarray(
                        self._lowrank._safe_nanmedian(arr), dtype=float
                    )

        attrs = {
            "pipeline_family": "low_rank_pulsatility",
            "reference": "Sec. VI.A canonical template shape change / "
            "Sec. VI.C mode-2/3 residual fraction",
            "representations": ["raw", "bandlimited"],
            "mode_indices": list(self.mode_indices),
            "input_beat_period_path": self.T_input,
            "input_raw_segment_path": self.v_raw_segment_input,
            "input_bandlimited_segment_path": self.v_band_segment_input,
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
