import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="old_waveform_shape_metrics")
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

    description = (
        "Segment waveform shape metrics (tau, RI, RVTI) + branch/global aggregates."
    )

    v_raw_global_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_bandlimited_global_input = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )
    v_bandlimited_global_max_input = (
        "/Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value"
    )
    v_bandlimited_global_min_input = (
        "/Artery/VelocityPerBeat/VminPerBeatBandLimited/value"
    )
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    @staticmethod
    def _rectify_keep_nan(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)

    @staticmethod
    def _safe_nanmean(x: np.ndarray) -> float:
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmean(x))

    @staticmethod
    def _safe_nanmedian(x: np.ndarray) -> float:
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
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
            tau_M1 = np.nan
            tau_M1_over_T = np.nan
        else:
            m0 = np.nansum(v)
            if (not np.isfinite(m0)) or m0 <= 0:
                tau_M1 = np.nan
                tau_M1_over_T = np.nan
            else:
                dt = Tbeat / n
                t = np.arange(n, dtype=float) * dt
                m1 = np.nansum(v * t)
                tau_M1 = (m1 / m0) if np.isfinite(m1) else np.nan
                tau_M1_over_T = tau_M1 / Tbeat

        # RI robust
        if not np.any(np.isfinite(v)):
            RI = np.nan
            PI = np.nan
        else:
            vmax = np.nanmax(v)
            mean = np.nanmean(v)
            if (not np.isfinite(vmax)) or vmax <= 0:
                RI = np.nan
                PI = np.nan
            else:
                vmin = np.nanmin(v)
                RI = 1.0 - (vmin / vmax)
                PI = (vmax - vmin) / mean
                if not np.isfinite(RI):
                    RI = np.nan
                    PI = np.nan
                else:
                    RI = float(np.clip(RI, 0.0, 1.0))
                    PI = float(PI)
        # RVTI (paper): D1/(D2+eps)
        k = int(np.ceil(n * ratio))
        k = max(0, min(n, k))
        D1 = np.nansum(v[:k]) if k > 0 else np.nan
        D2 = np.nansum(v[k:]) if k < n else np.nan
        if not np.isfinite(D1) or D1 == 0.0:
            D1 = np.nan
        if not np.isfinite(D2) or D2 == 0.0:
            D2 = np.nan
        RVTI = float(D1 / (D2 + eps))

        return float(tau_M1), float(tau_M1_over_T), float(RI), RVTI, float(PI)

    def _compute_block(self, v_block: np.ndarray, T: np.ndarray, ratio: float):
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}"
            )

        n_t, n_beats, n_branches, n_radii = v_block.shape
        n_segments = n_branches * n_radii

        # Per-segment flattened (beat, segment)
        tau_seg = np.zeros((n_beats, n_segments), dtype=float)
        tauT_seg = np.zeros((n_beats, n_segments), dtype=float)
        RI_seg = np.zeros((n_beats, n_segments), dtype=float)
        PI_seg = np.zeros((n_beats, n_segments), dtype=float)
        RVTI_seg = np.zeros((n_beats, n_segments), dtype=float)

        # Aggregated
        tau_branch = np.zeros((n_beats, n_branches), dtype=float)
        tauT_branch = np.zeros((n_beats, n_branches), dtype=float)
        RI_branch = np.zeros((n_beats, n_branches), dtype=float)
        PI_branch = np.zeros((n_beats, n_branches), dtype=float)
        RVTI_branch = np.zeros((n_beats, n_branches), dtype=float)

        tau_global = np.zeros((n_beats,), dtype=float)
        tauT_global = np.zeros((n_beats,), dtype=float)
        RI_global = np.zeros((n_beats,), dtype=float)
        PI_global = np.zeros((n_beats,), dtype=float)
        RVTI_global = np.zeros((n_beats,), dtype=float)

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])

            # Global accumulators for this beat
            tau_vals = []
            tauT_vals = []
            RI_vals = []
            PI_vals = []
            RVTI_vals = []

            for branch_idx in range(n_branches):
                # Branch accumulators across radii
                tau_b = []
                tauT_b = []
                RI_b = []
                PI_b = []
                RVTI_b = []

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    tM1, tM1T, ri, rvti, pi = self._metrics_from_waveform(
                        v=v, Tbeat=Tbeat, ratio=ratio, eps=1e-12
                    )

                    seg_idx = branch_idx * n_radii + radius_idx
                    tau_seg[beat_idx, seg_idx] = tM1
                    tauT_seg[beat_idx, seg_idx] = tM1T
                    RI_seg[beat_idx, seg_idx] = ri
                    RVTI_seg[beat_idx, seg_idx] = rvti
                    PI_seg[beat_idx, seg_idx] = pi

                    tau_b.append(tM1)
                    tauT_b.append(tM1T)
                    RI_b.append(ri)
                    RVTI_b.append(rvti)
                    PI_b.append(pi)

                    tau_vals.append(tM1)
                    tauT_vals.append(tM1T)
                    RI_vals.append(ri)
                    RVTI_vals.append(rvti)
                    PI_vals.append(pi)

                # Branch aggregates: MEDIAN over radii
                tau_branch[beat_idx, branch_idx] = self._safe_nanmedian(
                    np.asarray(tau_b)
                )
                tauT_branch[beat_idx, branch_idx] = self._safe_nanmedian(
                    np.asarray(tauT_b)
                )
                RI_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(RI_b))
                PI_branch[beat_idx, branch_idx] = self._safe_nanmedian(np.asarray(PI_b))
                RVTI_branch[beat_idx, branch_idx] = self._safe_nanmedian(
                    np.asarray(RVTI_b)
                )

            # Global aggregates: MEAN over all branches & radii
            tau_global[beat_idx] = self._safe_nanmean(np.asarray(tau_vals))
            tauT_global[beat_idx] = self._safe_nanmean(np.asarray(tauT_vals))
            RI_global[beat_idx] = self._safe_nanmean(np.asarray(RI_vals))
            RVTI_global[beat_idx] = self._safe_nanmean(np.asarray(RVTI_vals))
            PI_global[beat_idx] = self._safe_nanmean(np.asarray(PI_vals))

        return (
            tau_seg,
            tauT_seg,
            RI_seg,
            PI_seg,
            RVTI_seg,
            tau_branch,
            tauT_branch,
            RI_branch,
            PI_branch,
            RVTI_branch,
            tau_global,
            tauT_global,
            RI_global,
            PI_global,
            RVTI_global,
            n_branches,
            n_radii,
        )

    def run(self, h5file) -> ProcessResult:
        T = np.asarray(h5file[self.T_input])
        ratio_systole_diastole_R_VTI = 0.5

        try:
            v_raw_input = (
                "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
            )
            v_bandlimited_input = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"

            v_raw = np.asarray(h5file[v_raw_input])
            v_band = np.asarray(h5file[v_bandlimited_input])
            v_raw = self._rectify_keep_nan(v_raw)
            v_band = self._rectify_keep_nan(v_band)

            (
                tau_seg_b,
                tauT_seg_b,
                RI_seg_b,
                PI_seg_b,
                RVTI_seg_b,
                tau_br_b,
                tauT_br_b,
                RI_br_b,
                PI_br_b,
                RVTI_br_b,
                tau_gl_b,
                tauT_gl_b,
                RI_gl_b,
                PI_gl_b,
                RVTI_gl_b,
                n_branches_b,
                n_radii_b,
            ) = self._compute_block(v_band, T, ratio_systole_diastole_R_VTI)

            (
                tau_seg_r,
                tauT_seg_r,
                RI_seg_r,
                PI_seg_r,
                RVTI_seg_r,
                tau_br_r,
                tauT_br_r,
                RI_br_r,
                PI_br_r,
                RVTI_br_r,
                tau_gl_r,
                tauT_gl_r,
                RI_gl_r,
                PI_gl_r,
                RVTI_gl_r,
                n_branches_r,
                n_radii_r,
            ) = self._compute_block(v_raw, T, ratio_systole_diastole_R_VTI)

            # Consistency attributes (optional but useful)
            seg_order_note = (
                "seg_idx = branch_idx * n_radii + radius_idx (branch-major flattening)"
            )
            if n_radii_b != n_radii_r or n_branches_b != n_branches_r:
                seg_order_note += (
                    " | WARNING: raw/bandlimited branch/radius dims differ."
                )

            metrics = {
                # --- Existing datasets (unchanged names/shapes) ---
                "by_segment/tau_M1_bandlimited_segment": with_attrs(
                    tau_seg_b,
                    {
                        "unit": ["s"],
                        "definition": ["tau_M1 = M1/M0 on rectified waveform"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/tau_M1_over_T_bandlimited_segment": with_attrs(
                    tauT_seg_b,
                    {
                        "unit": [""],
                        "definition": ["tau_M1_over_T = (M1/M0)/T"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/RI_bandlimited_segment": with_attrs(
                    RI_seg_b,
                    {
                        "unit": [""],
                        "definition": ["RI = 1 - vmin/vmax (robust, rectified)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/PI_bandlimited_segment": with_attrs(
                    PI_seg_b,
                    {
                        "unit": [""],
                        "definition": ["RI = 1 - vmin/vmax (robust, rectified)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/R_VTI_bandlimited_segment": with_attrs(
                    RVTI_seg_b,
                    {
                        "unit": [""],
                        "definition": ["paper RVTI = D1/(D2+eps)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/tau_M1_raw_segment": with_attrs(
                    tau_seg_r,
                    {
                        "unit": ["s"],
                        "definition": ["tau_M1 = M1/M0 on rectified waveform"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/tau_M1_over_T_raw_segment": with_attrs(
                    tauT_seg_r,
                    {
                        "unit": [""],
                        "definition": ["tau_M1_over_T = (M1/M0)/T"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/RI_raw_segment": with_attrs(
                    RI_seg_r,
                    {
                        "unit": [""],
                        "definition": ["RI = 1 - vmin/vmax (robust, rectified)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/PI_raw_segment": with_attrs(
                    PI_seg_r,
                    {
                        "unit": [""],
                        "definition": ["RI = 1 - vmin/vmax (robust, rectified)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/R_VTI_raw_segment": with_attrs(
                    RVTI_seg_r,
                    {
                        "unit": [""],
                        "definition": ["paper RVTI = D1/(D2+eps)"],
                        "segment_indexing": [seg_order_note],
                    },
                ),
                "by_segment/ratio_systole_diastole_R_VTI": np.asarray(
                    ratio_systole_diastole_R_VTI, dtype=float
                ),
                # --- New aggregated outputs ---
                "by_segment/tau_M1_bandlimited_branch": with_attrs(
                    tau_br_b,
                    {
                        "unit": ["s"],
                        "definition": ["median over radii: tau_M1 per branch"],
                    },
                ),
                "by_segment/tau_M1_over_T_bandlimited_branch": with_attrs(
                    tauT_br_b,
                    {
                        "unit": [""],
                        "definition": ["median over radii: tau_M1/T per branch"],
                    },
                ),
                "by_segment/RI_bandlimited_branch": with_attrs(
                    RI_br_b,
                    {"unit": [""], "definition": ["median over radii: RI per branch"]},
                ),
                "by_segment/PI_bandlimited_branch": with_attrs(
                    PI_br_b,
                    {"unit": [""], "definition": ["median over radii: RI per branch"]},
                ),
                "by_segment/R_VTI_bandlimited_branch": with_attrs(
                    RVTI_br_b,
                    {
                        "unit": [""],
                        "definition": ["median over radii: paper RVTI per branch"],
                    },
                ),
                "by_segment/tau_M1_bandlimited_global": with_attrs(
                    tau_gl_b,
                    {
                        "unit": ["s"],
                        "definition": ["mean over branches & radii: tau_M1 global"],
                    },
                ),
                "by_segment/tau_M1_over_T_bandlimited_global": with_attrs(
                    tauT_gl_b,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: tau_M1/T global"],
                    },
                ),
                "by_segment/RI_bandlimited_global": with_attrs(
                    RI_gl_b,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: RI global"],
                    },
                ),
                "by_segment/PI_bandlimited_global": with_attrs(
                    PI_gl_b,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: RI global"],
                    },
                ),
                "by_segment/R_VTI_bandlimited_global": with_attrs(
                    RVTI_gl_b,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: paper RVTI global"],
                    },
                ),
                "by_segment/tau_M1_raw_branch": with_attrs(
                    tau_br_r,
                    {
                        "unit": ["s"],
                        "definition": ["median over radii: tau_M1 per branch"],
                    },
                ),
                "by_segment/tau_M1_over_T_raw_branch": with_attrs(
                    tauT_br_r,
                    {
                        "unit": [""],
                        "definition": ["median over radii: tau_M1/T per branch"],
                    },
                ),
                "by_segment/RI_raw_branch": with_attrs(
                    RI_br_r,
                    {"unit": [""], "definition": ["median over radii: RI per branch"]},
                ),
                "by_segment/PI_raw_branch": with_attrs(
                    PI_br_r,
                    {"unit": [""], "definition": ["median over radii: RI per branch"]},
                ),
                "by_segment/R_VTI_raw_branch": with_attrs(
                    RVTI_br_r,
                    {
                        "unit": [""],
                        "definition": ["median over radii: paper RVTI per branch"],
                    },
                ),
                "by_segment/tau_M1_raw_global": with_attrs(
                    tau_gl_r,
                    {
                        "unit": ["s"],
                        "definition": ["mean over branches & radii: tau_M1 global"],
                    },
                ),
                "by_segment/tau_M1_over_T_raw_global": with_attrs(
                    tauT_gl_r,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: tau_M1/T global"],
                    },
                ),
                "by_segment/RI_raw_global": with_attrs(
                    RI_gl_r,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: RI global"],
                    },
                ),
                "by_segment/PI_raw_global": with_attrs(
                    PI_gl_r,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: RI global"],
                    },
                ),
                "by_segment/R_VTI_raw_global": with_attrs(
                    RVTI_gl_r,
                    {
                        "unit": [""],
                        "definition": ["mean over branches & radii: paper RVTI global"],
                    },
                ),
            }

        except Exception:  # noqa: BLE001
            metrics = {}
        v_raw = np.asarray(h5file[self.v_raw_global_input])
        v_raw = np.maximum(v_raw, 0)
        v_bandlimited = np.asarray(h5file[self.v_bandlimited_global_input])
        v_bandlimited = np.maximum(v_bandlimited, 0)
        v_bandlimited_max = np.asarray(h5file[self.v_bandlimited_global_max_input])
        v_bandlimited_max = np.maximum(v_bandlimited_max, 0)
        v_bandlimited_min = np.asarray(h5file[self.v_bandlimited_global_min_input])
        v_bandlimited_min = np.maximum(v_bandlimited_min, 0)
        tau_M1_raw = []
        tau_M1_over_T_raw = []
        tau_M1_bandlimited = []
        tau_M1_over_T_bandlimited = []

        R_VTI_bandlimited = []
        R_VTI_raw = []

        RI_bandlimited = []
        RI_raw = []
        PI_bandlimited = []
        PI_raw = []

        ratio_systole_diastole_R_VTI = 0.5

        for beat_idx in range(len(T[0])):
            t = T[0][beat_idx] / len(v_raw.T[beat_idx])
            D1_raw = np.sum(
                v_raw.T[beat_idx][
                    : int(np.ceil(len(v_raw.T[0]) * ratio_systole_diastole_R_VTI))
                ]
            )
            D2_raw = np.sum(
                v_raw.T[beat_idx][
                    int(np.ceil(len(v_raw.T[0]) * ratio_systole_diastole_R_VTI)) :
                ]
            )
            D1_bandlimited = np.sum(
                v_bandlimited.T[beat_idx][
                    : int(
                        np.ceil(len(v_bandlimited.T[0]) * ratio_systole_diastole_R_VTI)
                    )
                ]
            )
            D2_bandlimited = np.sum(
                v_bandlimited.T[beat_idx][
                    int(
                        np.ceil(len(v_bandlimited.T[0]) * ratio_systole_diastole_R_VTI)
                    ) :
                ]
            )
            R_VTI_bandlimited.append(D1_bandlimited / (D2_bandlimited + 10 ** (-12)))
            R_VTI_raw.append(D1_raw / (D2_raw + 10 ** (-12)))
            M_0 = np.sum(v_raw.T[beat_idx])
            M_1 = 0
            for time_idx in range(len(v_raw.T[beat_idx])):
                M_1 += v_raw[time_idx][beat_idx] * time_idx * t
            TM1 = M_1 / M_0
            tau_M1_raw.append(TM1)
            tau_M1_over_T_raw.append(TM1 / T[0][beat_idx])

        for beat_idx in range(len(T[0])):
            t = T[0][beat_idx] / len(v_raw.T[beat_idx])
            M_0 = np.sum(v_bandlimited.T[beat_idx])
            M_1 = 0
            for time_idx in range(len(v_raw.T[beat_idx])):
                M_1 += v_bandlimited[time_idx][beat_idx] * time_idx * t
            TM1 = M_1 / M_0
            tau_M1_bandlimited.append(TM1)
            tau_M1_over_T_bandlimited.append(TM1 / T[0][beat_idx])

        for beat_idx in range(len(v_bandlimited_max[0])):
            RI_bandlimited_temp = 1 - (
                np.min(v_bandlimited.T[beat_idx]) / np.max(v_bandlimited.T[beat_idx])
            )
            RI_bandlimited.append(RI_bandlimited_temp)
            PI_bandlimited_temp = (
                np.max(v_bandlimited.T[beat_idx]) - np.min(v_bandlimited.T[beat_idx])
            ) / np.mean(v_bandlimited.T[beat_idx])
            PI_bandlimited.append(PI_bandlimited_temp)

        for beat_idx in range(len(v_bandlimited_max[0])):
            RI_raw_temp = 1 - (np.min(v_raw.T[beat_idx]) / np.max(v_raw.T[beat_idx]))
            RI_raw.append(RI_raw_temp)
            PI_raw_temp = (
                np.max(v_raw.T[beat_idx]) - np.min(v_raw.T[beat_idx])
            ) / np.mean(v_raw.T[beat_idx])
            PI_raw.append(PI_raw_temp)
        metrics.update(
            {
                "global/tau_M1_raw": with_attrs(np.asarray(tau_M1_raw), {"unit": [""]}),
                "global/tau_M1_bandlimited": np.asarray(tau_M1_bandlimited),
                "global/tau_M1_over_T_raw": with_attrs(
                    np.asarray(tau_M1_over_T_raw), {"unit": [""]}
                ),
                "global/tau_M1_over_T_bandlimited": np.asarray(
                    tau_M1_over_T_bandlimited
                ),
                "global/RI_bandlimited": np.asarray(RI_bandlimited),
                "global/RI_raw": np.asarray(RI_raw),
                "global/PI_raw": np.asarray(PI_raw),
                "global/PI_bandlimited": np.asarray(PI_bandlimited),
                "global/R_VTI_bandlimited": np.asarray(R_VTI_bandlimited),
                "global/R_VTI_raw": np.asarray(R_VTI_raw),
                "global/ratio_systole_diastole_R_VTI": np.asarray(
                    ratio_systole_diastole_R_VTI
                ),
            }
        )
        return ProcessResult(metrics=metrics)
