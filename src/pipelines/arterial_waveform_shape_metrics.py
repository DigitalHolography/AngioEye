import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterial_waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Waveform-shape metrics on per-beat, per-branch, per-radius velocity waveforms.

    Expected segment layout:
        v_seg[t, beat, branch, radius]
    i.e. v_seg shape: (n_t, n_beats, n_branches, n_radii)

    Outputs
    -------
    A) Per-segment (flattened branch x radius):
        by_segment/*_segment : shape (n_beats, n_segments)
        n_segments = n_branches * n_radii
        seg_idx = branch_idx * n_radii + radius_idx   (branch-major)

    B) Aggregated:
        by_segment/*_branch : shape (n_beats, n_branches)  (median over radii)
        by_segment/*_global : shape (n_beats,)             (mean over all branches & radii)

    C) Independent global metrics (from global waveform path):
        global/* : shape (n_beats,)

    Definitions (gain-invariant / shape metrics)
    --------------------------------------------
    Rectification:
        v <- max(v, 0) with NaNs preserved

    Basic:
        tau_M1        = M1 / M0
        tau_M1_over_T = (M1/M0) / T

        RI = 1 - vmin/vmax (robust)
        PI = (vmax - vmin) / mean(v) (robust)

        RVTI (paper) = D1 / (D2 + eps), split at 1/2 T (ratio_rvti = 0.5)

    New:
        SF_VTI (systolic fraction) = D1_1/3 / (D1_1/3 + D2_2/3 + eps)
            where D1_1/3 is integral over first 1/3 of samples, D2_2/3 over remaining 2/3

        Normalized central moments (shape, not scale):
            mu2_norm = mu2 / (M0 * T^2 + eps)   (variance-like)


            with central moments around t_bar = tau_M1:
                mu2 = sum(v * (t - t_bar)^2)


        Quantile timing (on cumulative integral):
            C(t) = cumsum(v) / sum(v)
            t10_over_T,t25_over_T, t50_over_T,t75_over_T, t90_over_T

        Spectral shape ratios (per beat):
            Compute FFT power P(f) of v(t). Define harmonic index h = f * T (cycles/beat).
            E_total = sum_{h>=0} P
            E_low   = sum_{h in [1..H_LOW]} P
            E_high  = sum_{h in [H_HIGH1..H_HIGH2]} P
            Return E_low_over_E_total and E_high_over_E_total

            Default bands:
                low:  1..3 harmonics
                high: 4..8 harmonics
    """

    description = "Waveform shape metrics (segment + aggregates + global), gain-invariant and robust."

    # Segment inputs
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"

    # Global inputs
    v_raw_global_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )

    # Beat period
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    # Parameters
    eps = 1e-12
    ratio_rvti = 0.5  # split for RVTI
    ratio_sf_vti = 1.0 / 3.0  # split for SF_VTI

    # Spectral bands (harmonic indices, inclusive)
    H_LOW_MAX = 3
    H_HIGH_MIN = 4
    H_HIGH_MAX = 8

    # -------------------------
    # Helpers
    # -------------------------
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
    def _ensure_time_by_beat(v2: np.ndarray, n_beats: int) -> np.ndarray:
        """
        Ensure v2 is shaped (n_t, n_beats). If it is (n_beats, n_t), transpose.
        """
        v2 = np.asarray(v2, dtype=float)
        if v2.ndim != 2:
            raise ValueError(f"Expected 2D global waveform, got shape {v2.shape}")

        if v2.shape[1] == n_beats:
            return v2
        if v2.shape[0] == n_beats and v2.shape[1] != n_beats:
            return v2.T

        # Fallback: if ambiguous, assume (n_t, n_beats)
        return v2

    def _quantile_time_over_T(self, v: np.ndarray, Tbeat: float, q: float) -> float:
        """
        v: rectified 1D waveform (NaNs allowed)
        Returns t_q / Tbeat where C(t_q) >= q, with C = cumsum(v)/sum(v).
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        m0 = float(np.sum(vv))
        if m0 <= 0:
            return np.nan

        c = np.cumsum(vv) / m0
        idx = int(np.searchsorted(c, q, side="left"))
        idx = max(0, min(v.size - 1, idx))

        dt = Tbeat / v.size
        t_q = idx * dt
        return float(t_q / Tbeat)

    def _spectral_ratios(self, v: np.ndarray, Tbeat: float) -> tuple[float, float]:
        """
        Return (E_low/E_total, E_high/E_total) using harmonic-index bands.
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan, np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan, np.nan

        vv = np.where(np.isfinite(v), v, 0.0)

        n = vv.size
        if n < 2:
            return np.nan, np.nan

        # Remove DC? For "shape" we typically keep DC in total energy but exclude it from low/high
        # Here: total includes all bins (including DC). Low/high exclude DC by construction (harmonics >= 1).
        fs = n / Tbeat  # Hz
        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        A = np.abs(X)
        f = np.fft.rfftfreq(n, d=1.0 / fs)  # Hz
        h = f * Tbeat  # cycles per beat (harmonic index, continuous)
        # vv_no_mean = vv - np.mean(vv)
        # idx_fund = np.argmax(A[1:]) + 1
        # f1 = f[idx_fund]
        # V1 = A[idx_fund]
        # if V1 <= 0:
        # return np.nan, np.nan, np.nan
        # HRI_2_10 = float(np.nan)
        """for k in range(2, 11):
            target_freq = k * f1

            # trouver le bin le plus proche
            idx = np.argmin(np.abs(f - target_freq))

            # éviter de sortir du spectre
            if idx < len(A):
                HRI_2_10 += A[idx] / V1"""
        E_total = float(np.sum(P))
        if not np.isfinite(E_total) or E_total <= 0:
            return np.nan, np.nan, np.nan

        low_mask = (h >= 1.0) & (h <= float(self.H_LOW_MAX))
        high_mask = (h >= float(self.H_HIGH_MIN)) & (h <= float(self.H_HIGH_MAX))

        E_low = float(np.sum(P[low_mask]))
        E_high = float(np.sum(P[high_mask]))

        return float(E_low / E_total), float(E_high / E_total)

    def _compute_metrics_1d(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Canonical metric kernel: compute all waveform-shape metrics from a single 1D waveform v(t).
        Returns a dict of scalar metrics (floats).
        """
        v = self._rectify_keep_nan(v)
        n = int(v.size)
        if n <= 0:
            return {k: np.nan for k in self._metric_keys()}

        # If Tbeat invalid, many metrics become NaN
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {k: np.nan for k in self._metric_keys()}

        vv = np.where(
            np.isfinite(v), v, np.nan
        )  # vv = np.where(np.isfinite(v), v, 0.0)
        m0 = float(np.nansum(vv))
        if m0 <= 0:
            return {k: np.nan for k in self._metric_keys()}

        dt = Tbeat / n
        t = np.arange(n, dtype=float) * dt

        # First moment
        m1 = float(np.nansum(vv * t))
        tau_M1 = m1 / m0
        tau_M1_over_T = tau_M1 / Tbeat

        # RI / PI robust
        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(v))  # vmin = float(np.min(vv))
        meanv = float(self._safe_nanmean(v))  # meanv = float(np.mean(vv))

        if vmax <= 0:
            RI = np.nan
            PI = np.nan
        else:
            RI = 1.0 - (vmin / vmax)
            RI = float(np.clip(RI, 0.0, 1.0)) if np.isfinite(RI) else np.nan

            if (not np.isfinite(meanv)) or meanv <= 0:
                PI = np.nan
            else:
                PI = (vmax - vmin) / meanv
                PI = float(PI) if np.isfinite(PI) else np.nan

        # RVTI (paper, split 1/2)
        k_rvti = int(np.ceil(n * self.ratio_rvti))
        k_rvti = max(0, min(n, k_rvti))
        D1_rvti = float(np.sum(vv[:k_rvti])) if k_rvti > 0 else np.nan
        D2_rvti = float(np.sum(vv[k_rvti:])) if k_rvti < n else np.nan
        RVTI = D1_rvti / (D2_rvti + self.eps)

        # SF_VTI (split 1/3 vs 2/3)
        k_sf = int(np.ceil(n * self.ratio_sf_vti))
        k_sf = max(0, min(n, k_sf))
        D1_sf = float(np.nansum(vv[:k_sf])) if k_sf > 0 else np.nan
        D2_sf = float(np.nansum(vv[k_sf:])) if k_sf < n else np.nan
        SF_VTI = D1_sf / (D1_sf + D2_sf + self.eps)

        # Central moments around tau_M1 (t_bar)
        # mu2 = sum(v*(t-tau)^2)
        dtau = t - tau_M1
        mu2 = float(np.nansum(vv * (dtau**2)))
        tau_M2 = np.sqrt(mu2 / m0 + self.eps)
        tau_M2_over_T = tau_M2 / Tbeat

        # Quantile timing features (on cumulative integral)
        t10_over_T = self._quantile_time_over_T(vv, Tbeat, 0.10)
        t25_over_T = self._quantile_time_over_T(vv, Tbeat, 0.25)
        t50_over_T = self._quantile_time_over_T(vv, Tbeat, 0.50)
        t75_over_T = self._quantile_time_over_T(vv, Tbeat, 0.75)
        t90_over_T = self._quantile_time_over_T(vv, Tbeat, 0.90)

        # Spectral ratios
        E_low_over_E_total, E_high_over_E_total = self._spectral_ratios(vv, Tbeat)

        return {
            "tau_M1": float(tau_M1),
            "tau_M1_over_T": float(tau_M1_over_T),
            "RI": float(RI) if np.isfinite(RI) else np.nan,
            "PI": float(PI) if np.isfinite(PI) else np.nan,
            "R_VTI": float(RVTI),
            "SF_VTI": float(SF_VTI),
            "tau_M2_over_T": float(tau_M2_over_T),
            "tau_M2": float(tau_M2),
            "t10_over_T": float(t10_over_T),
            "t25_over_T": float(t25_over_T),
            "t50_over_T": float(t50_over_T),
            "t75_over_T": float(t75_over_T),
            "t90_over_T": float(t90_over_T),
            "E_low_over_E_total": float(E_low_over_E_total),
            "E_high_over_E_total": float(E_high_over_E_total),
            # "HRI_2_10_total": float(HRI_2_10_total),
        }

    @staticmethod
    def _metric_keys() -> list[str]:
        return [
            "tau_M1",
            "tau_M1_over_T",
            "RI",
            "PI",
            "R_VTI",
            "SF_VTI",
            "tau_M2_over_T",
            "tau_M2",
            "t10_over_T",
            "t25_over_T",
            "t50_over_T",
            "t75_over_T",
            "t90_over_T",
            "E_low_over_E_total",
            "E_high_over_E_total",
            # "HRI_2_10_total",
        ]

    def _compute_block_segment(self, v_block: np.ndarray, T: np.ndarray):
        """
        v_block: (n_t, n_beats, n_branches, n_radii)
        Returns:
          per-segment arrays: (n_beats, n_segments)
          per-branch arrays:  (n_beats, n_branches)   (median over radii)
          global arrays:      (n_beats,)              (mean over all branches & radii)
        """
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}"
            )

        n_t, n_beats, n_branches, n_radii = v_block.shape
        n_segments = n_branches * n_radii

        # Allocate per metric
        seg = {
            k: np.full((n_beats, n_segments), np.nan, dtype=float)
            for k in self._metric_keys()
        }
        br = {
            k: np.full((n_beats, n_branches), np.nan, dtype=float)
            for k in self._metric_keys()
        }
        gl = {k: np.full((n_beats,), np.nan, dtype=float) for k in self._metric_keys()}

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])

            # For global aggregate at this beat
            gl_vals = {k: [] for k in self._metric_keys()}

            for branch_idx in range(n_branches):
                # For branch aggregate over radii
                br_vals = {k: [] for k in self._metric_keys()}

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    m = self._compute_metrics_1d(v, Tbeat)

                    seg_idx = branch_idx * n_radii + radius_idx
                    for k in self._metric_keys():
                        seg[k][beat_idx, seg_idx] = m[k]
                        br_vals[k].append(m[k])
                        gl_vals[k].append(m[k])

                # Branch aggregates: median over radii (nanmedian)
                for k in self._metric_keys():
                    br[k][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[k], dtype=float)
                    )

            # Global aggregates: mean over all branches & radii (nanmean)
            for k in self._metric_keys():
                gl[k][beat_idx] = self._safe_nanmean(
                    np.asarray(gl_vals[k], dtype=float)
                )

        seg_order_note = (
            "seg_idx = branch_idx * n_radii + radius_idx (branch-major flattening)"
        )
        return seg, br, gl, n_branches, n_radii, seg_order_note

    def _compute_block_global(self, v_global: np.ndarray, T: np.ndarray):
        """
        v_global: (n_t, n_beats) after _ensure_time_by_beat
        Returns dict of arrays each shaped (n_beats,)
        """
        n_beats = int(T.shape[1])
        v_global = self._ensure_time_by_beat(v_global, n_beats)
        v_global = self._rectify_keep_nan(v_global)

        out = {k: np.full((n_beats,), np.nan, dtype=float) for k in self._metric_keys()}

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])
            v = v_global[:, beat_idx]
            m = self._compute_metrics_1d(v, Tbeat)
            for k in self._metric_keys():
                out[k][beat_idx] = m[k]

        return out

    # -------------------------
    # Pipeline entrypoint
    # -------------------------
    def run(self, h5file) -> ProcessResult:
        T = np.asarray(h5file[self.T_input])
        metrics = {}

        # -------------------------
        # Segment metrics (raw + bandlimited)
        # -------------------------
        have_seg = (self.v_raw_segment_input in h5file) and (
            self.v_band_segment_input in h5file
        )
        if have_seg:
            v_raw_seg = np.asarray(h5file[self.v_raw_segment_input])
            v_band_seg = np.asarray(h5file[self.v_band_segment_input])

            seg_b, br_b, gl_b, nb_b, nr_b, seg_note_b = self._compute_block_segment(
                v_band_seg, T
            )
            seg_r, br_r, gl_r, nb_r, nr_r, seg_note_r = self._compute_block_segment(
                v_raw_seg, T
            )

            seg_note = seg_note_b
            if (nb_b != nb_r) or (nr_b != nr_r):
                seg_note = (
                    seg_note_b + " | WARNING: raw/band branch/radius dims differ."
                )

            # Helper to pack dict-of-arrays into HDF5 metric keys
            def pack(prefix: str, d: dict, attrs_common: dict):
                for k, arr in d.items():
                    metrics[f"{prefix}/{k}"] = with_attrs(arr, attrs_common)

            # Per-segment outputs (compat dataset names)
            pack(
                "by_segment/bandlimited_segment",
                {
                    "tau_M1": seg_b["tau_M1"],
                    "tau_M1_over_T": seg_b["tau_M1_over_T"],
                    "RI": seg_b["RI"],
                    "PI": seg_b["PI"],
                    "R_VTI": seg_b["R_VTI"],
                    "SF_VTI": seg_b["SF_VTI"],
                    "tau_M2_over_T": seg_b["tau_M2_over_T"],
                    "tau_M2": seg_b["tau_M2"],
                    "t10_over_T": seg_b["t10_over_T"],
                    "t25_over_T": seg_b["t25_over_T"],
                    "t50_over_T": seg_b["t50_over_T"],
                    "t75_over_T": seg_b["t75_over_T"],
                    "t90_over_T": seg_b["t90_over_T"],
                    "E_low_over_E_total": seg_b["E_low_over_E_total"],
                    "E_high_over_E_total": seg_b["E_high_over_E_total"],
                    # "HRI_2_10_total": seg_b["HRI_2_10_total"],
                },
                {
                    "segment_indexing": [seg_note],
                },
            )
            pack(
                "by_segment/raw_segment",
                {
                    "tau_M1": seg_r["tau_M1"],
                    "tau_M1_over_T": seg_r["tau_M1_over_T"],
                    "RI": seg_r["RI"],
                    "PI": seg_r["PI"],
                    "R_VTI": seg_r["R_VTI"],
                    "SF_VTI": seg_r["SF_VTI"],
                    "tau_M2_over_T": seg_r["tau_M2_over_T"],
                    "tau_M2": seg_r["tau_M2"],
                    "t10_over_T": seg_r["t10_over_T"],
                    "t25_over_T": seg_r["t25_over_T"],
                    "t50_over_T": seg_r["t50_over_T"],
                    "t75_over_T": seg_r["t75_over_T"],
                    "t90_over_T": seg_r["t90_over_T"],
                    "E_low_over_E_total": seg_r["E_low_over_E_total"],
                    "E_high_over_E_total": seg_r["E_high_over_E_total"],
                    # "HRI_2_10_total": seg_r["HRI_2_10_total"],
                },
                {
                    "segment_indexing": [seg_note],
                },
            )

            # Branch aggregates (median over radii)
            pack(
                "by_segment/bandlimited_branch",
                br_b,
                {"definition": ["median over radii per branch"]},
            )
            pack(
                "by_segment/raw_branch",
                br_r,
                {"definition": ["median over radii per branch"]},
            )

            # Global aggregates (mean over all branches & radii)
            pack(
                "by_segment/bandlimited_global",
                gl_b,
                {"definition": ["mean over branches and radii"]},
            )
            pack(
                "by_segment/raw_global",
                gl_r,
                {"definition": ["mean over branches and radii"]},
            )

            # Store parameters used (for provenance)
            metrics["by_segment/params/ratio_rvti"] = np.asarray(
                self.ratio_rvti, dtype=float
            )
            metrics["by_segment/params/ratio_sf_vti"] = np.asarray(
                self.ratio_sf_vti, dtype=float
            )
            metrics["by_segment/params/eps"] = np.asarray(self.eps, dtype=float)
            metrics["by_segment/params/H_LOW_MAX"] = np.asarray(
                self.H_LOW_MAX, dtype=int
            )
            metrics["by_segment/params/H_HIGH_MIN"] = np.asarray(
                self.H_HIGH_MIN, dtype=int
            )
            metrics["by_segment/params/H_HIGH_MAX"] = np.asarray(
                self.H_HIGH_MAX, dtype=int
            )

        # -------------------------
        # Independent global metrics (raw + bandlimited)
        # -------------------------
        have_glob = (self.v_raw_global_input in h5file) and (
            self.v_band_global_input in h5file
        )
        if have_glob:
            v_raw_gl = np.asarray(h5file[self.v_raw_global_input])
            v_band_gl = np.asarray(h5file[self.v_band_global_input])

            out_raw = self._compute_block_global(v_raw_gl, T)
            out_band = self._compute_block_global(v_band_gl, T)

            for k in self._metric_keys():
                metrics[f"global/raw/{k}"] = out_raw[k]
                metrics[f"global/bandlimited/{k}"] = out_band[k]

            # provenance
            metrics["global/params/ratio_rvti"] = np.asarray(
                self.ratio_rvti, dtype=float
            )
            metrics["global/params/ratio_sf_vti"] = np.asarray(
                self.ratio_sf_vti, dtype=float
            )
            metrics["global/params/eps"] = np.asarray(self.eps, dtype=float)
            metrics["global/params/H_LOW_MAX"] = np.asarray(self.H_LOW_MAX, dtype=int)
            metrics["global/params/H_HIGH_MIN"] = np.asarray(self.H_HIGH_MIN, dtype=int)
            metrics["global/params/H_HIGH_MAX"] = np.asarray(self.H_HIGH_MAX, dtype=int)

        return ProcessResult(metrics=metrics)
