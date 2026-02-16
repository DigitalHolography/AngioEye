import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="segment_waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Waveform-shape metrics on per-beat, per-branch, per-radius velocity waveforms.

    Expected v_block layout:
        v_block[:, beat_idx, branch_idx, radius_idx]
    i.e. v_block shape: (n_t, n_beats, n_branches, n_radii)

    Outputs
    -------
    A) Per-segment (flattened branch×radius):
        *_segment : shape (n_beats, n_segments)
        where n_segments = n_branches * n_radii and
              seg_idx = branch_idx * n_radii + radius_idx   (branch-major)

    B) Aggregated:
        *_branch : shape (n_beats, n_branches)   (median over radii)
        *_global : shape (n_beats,)              (mean over all branches & radii)

    Metric definitions
    ------------------
    - Rectification: v <- max(v, 0) (NaNs preserved)
    - tau_M1: first moment time / zeroth moment on rectified waveform
        tau_M1 = M1/M0,  M0 = sum(v), M1 = sum(v * t_k), t_k = k * (Tbeat/n_t)
    - tau_M1_over_T: (tau_M1 / Tbeat)
    - RI (robust): RI = 1 - vmin/vmax with guards for vmax<=0 or all-NaN
    - R_VTI_*: kept dataset name for compatibility, but uses PAPER convention:
        RVTI = D1 / (D2 + eps)
        D1 = sum(v[0:k]), D2 = sum(v[k:n_t]), k = ceil(n_t * ratio), ratio=0.5
    """

    description = "Segment waveform shape metrics (tau, RI, RVTI) + branch/global aggregates."

    v_raw_input = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    v_bandlimited_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    @staticmethod
    def _rectify_keep_nan(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)

    @staticmethod
    def _safe_nanmean(x: np.ndarray) -> float:
        if x.size == 0 or not np.any(np.isfinite(x)):
            return 0.0
        return float(np.nanmean(x))

    @staticmethod
    def _safe_nanmedian(x: np.ndarray) -> float:
        if x.size == 0 or not np.any(np.isfinite(x)):
            return 0.0
        return float(np.nanmedian(x))

    @staticmethod
    def _metrics_from_waveform(
        v: np.ndarray,
        Tbeat: float,
        ratio: float = 0.5,
        eps: float = 1e-12,
    ):
        v = ArterialSegExample._rectify_keep_nan(v)

        n = int(v.size)
        if n <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # tau_M1 and tau_M1/T (PER WAVEFORM ONLY)
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            tau_M1 = 0.0
            tau_M1_over_T = 0.0
        else:
            m0 = np.nansum(v)
            if (not np.isfinite(m0)) or m0 <= 0:
                tau_M1 = 0.0
                tau_M1_over_T = 0.0
            else:
                dt = Tbeat / n
                t = np.arange(n, dtype=float) * dt
                m1 = np.nansum(v * t)
                tau_M1 = (m1 / m0) if np.isfinite(m1) else 0.0
                tau_M1_over_T = tau_M1 / Tbeat

        # RI robust
        if not np.any(np.isfinite(v)):
            RI = 0.0
        else:
            vmax = np.nanmax(v)
            if (not np.isfinite(vmax)) or vmax <= 0:
                RI = 0.0
            else:
                vmin = np.nanmin(v)
                RI = 1.0 - (vmin / vmax)
                if not np.isfinite(RI):
                    RI = 0.0
                else:
                    RI = float(np.clip(RI, 0.0, 1.0))

        # RVTI (paper): D1/(D2+eps)
        k = int(np.ceil(n * ratio))
        k = max(0, min(n, k))
        D1 = np.nansum(v[:k]) if k > 0 else 0.0
        D2 = np.nansum(v[k:]) if k < n else 0.0
        if not np.isfinite(D1):
            D1 = 0.0
        if not np.isfinite(D2):
            D2 = 0.0
        RVTI = float(D1 / (D2 + eps))

        return float(tau_M1), float(tau_M1_over_T), float(RI), RVTI

    def _compute_block(self, v_block: np.ndarray, T: np.ndarray, ratio: float):
        if v_block.ndim != 4:
            raise ValueError(f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}")

        n_t, n_beats, n_branches, n_radii = v_block.shape
        n_segments = n_branches * n_radii

        # Per-segment flattened (beat, segment)
        tau_seg = np.zeros((n_beats, n_segments), dtype=float)
        tauT_seg = np.zeros((n_beats, n_segments), dtype=float)
        RI_seg = np.zeros((n_beats, n_segments), dtype=float)
        RVTI_seg = np.zeros((n_beats, n_segments), dtype=float)

        # Aggregated
        tau_branch = np.zeros((n_beats, n_branches), dtype=float)
        tauT_branch = np.zeros((n_beats, n_branches), dtype=float)
        RI_branch = np.zeros((n_beats, n_branches), dtype=float)
        RVTI_branch = np.zeros((n_beats, n_branches), dtype=float)

        tau_global = np.zeros((n_beats,), dtype=float)
        tauT_global = np.zeros((n_beats,), dtype=float)
        RI_global = np.zeros((n_beats,), dtype=float)
        RVTI_global = np.zeros((n_beats,), dtype=float)

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])

            # Global accumulators for this beat
            tau_vals = []
            tauT_vals = []
            RI_vals = []
            RVTI_vals = []

            for branch_idx in range(n_branches):
                # Branch accumulators across radii
                tau_b = []
                tauT_b = []
                RI_b = []
                RVTI_b = []

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    tM1, tM1T, ri, rvti = self._metrics_from_waveform(
                        v=v, Tbeat=Tbeat, ratio=ratio, eps=1e-12
                    )

                    seg_idx = branch_idx * n_radii + radius_idx
                    tau_seg[beat_idx, seg_idx] = tM1
                    tauT_seg[beat_idx, seg_idx] = tM1T
                    RI_seg[beat_idx, seg_idx] = ri
                    RVTI_seg[beat_idx, seg_idx] = rvti

                    tau_b.append(tM1)
                    tauT_b.append(tM1T)
                    RI_b.append(ri)
                    RVTI_b.append(rvti)

                    tau_vals.append(tM1)
                    tauT_vals.append(tM1T)
                    RI_vals.append(ri)
                    RVTI_vals.append(rvti)

                # Branch aggregates: MEDIAN over radii
                tau_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(tau_b))
                tauT_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(tauT_b))
                RI_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(RI_b))
                RVTI_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(RVTI_b))

            # Global aggregates: MEAN over all branches & radii
            tau_global[beat_idx] = self._safe_nanmean(np.asarray(tau_vals))
            tauT_global[beat_idx] = self._safe_nanmean(np.asarray(tauT_vals))
            RI_global[beat_idx] = self._safe_nanmean(np.asarray(RI_vals))
            RVTI_global[beat_idx] = self._safe_nanmean(np.asarray(RVTI_vals))

        return (
            tau_seg, tauT_seg, RI_seg, RVTI_seg,
            tau_branch, tauT_branch, RI_branch, RVTI_branch,
            tau_global, tauT_global, RI_global, RVTI_global,
            n_branches, n_radii
        )

    def run(self, h5file) -> ProcessResult:
        v_raw = np.asarray(h5file[self.v_raw_input])
        v_band = np.asarray(h5file[self.v_bandlimited_input])
        T = np.asarray(h5file[self.T_input])

        v_raw = self._rectify_keep_nan(v_raw)
        v_band = self._rectify_keep_nan(v_band)

        ratio_systole_diastole_R_VTI = 0.5

        (
            tau_seg_b, tauT_seg_b, RI_seg_b, RVTI_seg_b,
            tau_br_b, tauT_br_b, RI_br_b, RVTI_br_b,
            tau_gl_b, tauT_gl_b, RI_gl_b, RVTI_gl_b,
            n_branches_b, n_radii_b
        ) = self._compute_block(v_band, T, ratio_systole_diastole_R_VTI)

        (
            tau_seg_r, tauT_seg_r, RI_seg_r, RVTI_seg_r,
            tau_br_r, tauT_br_r, RI_br_r, RVTI_br_r,
            tau_gl_r, tauT_gl_r, RI_gl_r, RVTI_gl_r,
            n_branches_r, n_radii_r
        ) = self._compute_block(v_raw, T, ratio_systole_diastole_R_VTI)

        # Consistency attributes (optional but useful)
        seg_order_note = "seg_idx = branch_idx * n_radii + radius_idx (branch-major flattening)"
        if n_radii_b != n_radii_r or n_branches_b != n_branches_r:
            seg_order_note += " | WARNING: raw/bandlimited branch/radius dims differ."

        metrics = {
            # --- Existing datasets (unchanged names/shapes) ---
            "tau_M1_bandlimited_segment": with_attrs(
                tau_seg_b,
                {"unit": ["s"], "definition": ["tau_M1 = M1/M0 on rectified waveform"], "segment_indexing": [seg_order_note]},
            ),
            "tau_M1_over_T_bandlimited_segment": with_attrs(
                tauT_seg_b,
                {"unit": [""], "definition": ["tau_M1_over_T = (M1/M0)/T"], "segment_indexing": [seg_order_note]},
            ),
            "RI_bandlimited_segment": with_attrs(
                RI_seg_b,
                {"unit": [""], "definition": ["RI = 1 - vmin/vmax (robust, rectified)"], "segment_indexing": [seg_order_note]},
            ),
            "R_VTI_bandlimited_segment": with_attrs(
                RVTI_seg_b,
                {"unit": [""], "definition": ["paper RVTI = D1/(D2+eps)"], "segment_indexing": [seg_order_note]},
            ),

            "tau_M1_raw_segment": with_attrs(
                tau_seg_r,
                {"unit": ["s"], "definition": ["tau_M1 = M1/M0 on rectified waveform"], "segment_indexing": [seg_order_note]},
            ),
            "tau_M1_over_T_raw_segment": with_attrs(
                tauT_seg_r,
                {"unit": [""], "definition": ["tau_M1_over_T = (M1/M0)/T"], "segment_indexing": [seg_order_note]},
            ),
            "RI_raw_segment": with_attrs(
                RI_seg_r,
                {"unit": [""], "definition": ["RI = 1 - vmin/vmax (robust, rectified)"], "segment_indexing": [seg_order_note]},
            ),
            "R_VTI_raw_segment": with_attrs(
                RVTI_seg_r,
                {"unit": [""], "definition": ["paper RVTI = D1/(D2+eps)"], "segment_indexing": [seg_order_note]},
            ),

            "ratio_systole_diastole_R_VTI": np.asarray(ratio_systole_diastole_R_VTI, dtype=float),

            # --- New aggregated outputs ---
            "tau_M1_bandlimited_branch": with_attrs(
                tau_br_b, {"unit": ["s"], "definition": ["median over radii: tau_M1 per branch"]}
            ),
            "tau_M1_over_T_bandlimited_branch": with_attrs(
                tauT_br_b, {"unit": [""], "definition": ["median over radii: tau_M1/T per branch"]}
            ),
            "RI_bandlimited_branch": with_attrs(
                RI_br_b, {"unit": [""], "definition": ["median over radii: RI per branch"]}
            ),
            "R_VTI_bandlimited_branch": with_attrs(
                RVTI_br_b, {"unit": [""], "definition": ["median over radii: paper RVTI per branch"]}
            ),
            "tau_M1_bandlimited_global": with_attrs(
                tau_gl_b, {"unit": ["s"], "definition": ["mean over branches & radii: tau_M1 global"]}
            ),
            "tau_M1_over_T_bandlimited_global": with_attrs(
                tauT_gl_b, {"unit": [""], "definition": ["mean over branches & radii: tau_M1/T global"]}
            ),
            "RI_bandlimited_global": with_attrs(
                RI_gl_b, {"unit": [""], "definition": ["mean over branches & radii: RI global"]}
            ),
            "R_VTI_bandlimited_global": with_attrs(
                RVTI_gl_b, {"unit": [""], "definition": ["mean over branches & radii: paper RVTI global"]}
            ),

            "tau_M1_raw_branch": with_attrs(
                tau_br_r, {"unit": ["s"], "definition": ["median over radii: tau_M1 per branch"]}
            ),
            "tau_M1_over_T_raw_branch": with_attrs(
                tauT_br_r, {"unit": [""], "definition": ["median over radii: tau_M1/T per branch"]}
            ),
            "RI_raw_branch": with_attrs(
                RI_br_r, {"unit": [""], "definition": ["median over radii: RI per branch"]}
            ),
            "R_VTI_raw_branch": with_attrs(
                RVTI_br_r, {"unit": [""], "definition": ["median over radii: paper RVTI per branch"]}
            ),
            "tau_M1_raw_global": with_attrs(
                tau_gl_r, {"unit": ["s"], "definition": ["mean over branches & radii: tau_M1 global"]}
            ),
            "tau_M1_over_T_raw_global": with_attrs(
                tauT_gl_r, {"unit": [""], "definition": ["mean over branches & radii: tau_M1/T global"]}
            ),
            "RI_raw_global": with_attrs(
                RI_gl_r, {"unit": [""], "definition": ["mean over branches & radii: RI global"]}
            ),
            "R_VTI_raw_global": with_attrs(
                RVTI_gl_r, {"unit": [""], "definition": ["mean over branches & radii: paper RVTI global"]}
            ),
        }

        return ProcessResult(metrics=metrics)
