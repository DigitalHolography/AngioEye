import warnings
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="absolute_waveform_metrics")
class AbsoluteWaveformMetrics(ProcessPipeline):
    """
    Absolute waveform metrics on per-beat, per-branch, per-radius velocity waveforms.

    Notes
    -----
    - This pipeline computes amplitude-bearing, gain-sensitive quantities from both raw and
      bandlimited arterial and venous waveforms.
    - Metrics are reported globally, locally per segment, as branch medians, and as
      segment-derived global medians, following the organization of
      ``waveform_shape_metrics``.
    - Raw-vs-bandlimited agreement metrics are also reported when both
      representations are available.
    """

    description = (
        "Absolute waveform metrics (artery + vein; segment + aggregates + global), "
        "gain-sensitive beat-wise velocity, VTI, derivative, energy, harmonic, "
        "and raw-vs-bandlimited QC metrics."
    )

    # ----------------------------
    # Arterial inputs
    # ----------------------------
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input = (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_global_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )

    # ----------------------------
    # Venous inputs
    # ----------------------------
    v_raw_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_global_input_vein = "/Vein/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input_vein = (
        "/Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )

    # Beat period input. Kept identical to waveform_shape_metrics.
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    eps = 1e-12

    # Window definitions reused from waveform_shape_metrics where appropriate.
    ratio_vend_start = 0.75
    ratio_vend_end = 0.90

    H_LOW_MAX = 1
    H_MAX = 10

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _rectify_keep_nan(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)

    @staticmethod
    def _safe_nanmean(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmean(x))

    @staticmethod
    def _safe_nanmedian(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmedian(x))

    @staticmethod
    def _safe_nanmedian_array(x: np.ndarray, axis: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            shape = list(x.shape)
            if len(shape) == 0:
                return np.asarray(np.nan, dtype=float)
            del shape[axis]
            return np.full(tuple(shape), np.nan, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(x, axis=axis)

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

        return v2

    @staticmethod
    def _n_beats_from_T(T: np.ndarray) -> int:
        T = np.asarray(T)
        if T.ndim == 1:
            return int(T.shape[0])
        if T.ndim >= 2:
            return int(T.shape[1])
        raise ValueError(f"Expected beat-period array with at least 1 dimension, got {T.shape}")

    @staticmethod
    def _get_Tbeat(T: np.ndarray, beat_idx: int) -> float:
        T = np.asarray(T)
        if T.ndim == 1:
            return float(T[beat_idx])
        return float(T[0][beat_idx])

    @staticmethod
    def _window_indices(n: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
        if n <= 0:
            return 0, 0

        a = float(start_ratio)
        b = float(end_ratio)

        if (
            not np.isfinite(a)
            or not np.isfinite(b)
            or a < 0
            or b <= a
            or b > 1
        ):
            return 0, 0

        k0 = int(np.floor(a * n))
        k1 = int(np.ceil(b * n))

        k0 = max(0, min(n - 1, k0))
        k1 = max(k0 + 1, min(n, k1))
        return k0, k1

    def _late_window_indices(self, n: int) -> tuple[int, int]:
        return self._window_indices(n, self.ratio_vend_start, self.ratio_vend_end)

    def _nan_metrics_1d(self) -> dict:
        out = {k[0]: np.nan for k in self._scalar_metric_keys()}
        for k in self._array_metric_keys():
            out[k[0]] = np.full((int(k[3]),), np.nan, dtype=float)
        return out

    def _nan_qc_metrics_1d(self) -> dict:
        return {k[0]: np.nan for k in self._qc_metric_keys()}

    # ---------------------------------------------------------------------
    # 1D absolute metric kernel
    # ---------------------------------------------------------------------
    def _window_sum(self, v: np.ndarray, dt: float, a: float, b: float) -> float:
        k0, k1 = self._window_indices(v.size, a, b)
        if k1 <= k0:
            return np.nan
        return float(np.nansum(v[k0:k1]) * dt)

    def _window_mean(self, v: np.ndarray, a: float, b: float) -> float:
        k0, k1 = self._window_indices(v.size, a, b)
        if k1 <= k0:
            return np.nan
        return self._safe_nanmean(v[k0:k1])

    def _window_max(self, v: np.ndarray, a: float, b: float) -> float:
        k0, k1 = self._window_indices(v.size, a, b)
        if k1 <= k0 or not np.any(np.isfinite(v[k0:k1])):
            return np.nan
        return float(np.nanmax(v[k0:k1]))

    def _window_min(self, v: np.ndarray, a: float, b: float) -> float:
        k0, k1 = self._window_indices(v.size, a, b)
        if k1 <= k0 or not np.any(np.isfinite(v[k0:k1])):
            return np.nan
        return float(np.nanmin(v[k0:k1]))

    def _compute_metrics_1d(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Canonical metric kernel: compute all absolute metrics from a single 1D
        waveform v(t). Returns scalar metrics and harmonic arrays.
        """
        out = self._nan_metrics_1d()

        v = self._rectify_keep_nan(v)
        n = int(v.size)
        if n <= 0 or (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        if not np.any(np.isfinite(v)):
            return out

        dt = float(Tbeat) / float(n)
        vv = np.where(np.isfinite(v), v, np.nan)
        vf = np.where(np.isfinite(v), v, 0.0)

        finite_values = vv[np.isfinite(vv)]
        if finite_values.size == 0:
            return out

        # -------------------------------------------------------------
        # Absolute velocity level / extrema
        # -------------------------------------------------------------
        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(vv))
        vmean = float(np.nanmean(vv))
        vmedian = float(np.nanmedian(vv))
        vrms = float(np.sqrt(np.nanmean(vv * vv)))
        vstd = float(np.nanstd(vv))
        v_range = float(vmax - vmin)

        out.update(
            {
                "vmax": vmax,
                "vmin": vmin,
                "vmean": vmean,
                "vmedian": vmedian,
                "vrms": vrms,
                "vstd": vstd,
                "v_range": v_range,
                "v_peak_above_mean": float(vmax - vmean),
                "v_mean_above_min": float(vmean - vmin),
            }
        )

        percentiles = {
            "v_p05": 5,
            "v_p10": 10,
            "v_p25": 25,
            "v_p50": 50,
            "v_p75": 75,
            "v_p90": 90,
            "v_p95": 95,
        }
        for key, q in percentiles.items():
            out[key] = float(np.nanpercentile(vv, q))

        out["v_iqr"] = float(out["v_p75"] - out["v_p25"])
        out["v_mad"] = float(np.nanmedian(np.abs(vv - vmedian)))

        # -------------------------------------------------------------
        # Absolute VTI / displacement-like metrics
        # -------------------------------------------------------------
        vti_total = float(np.nansum(vv) * dt)
        out["vti_total"] = vti_total
        out["vti_0_10"] = self._window_sum(vv, dt, 0.00, 0.10)
        out["vti_0_25"] = self._window_sum(vv, dt, 0.00, 0.25)
        out["vti_0_33"] = self._window_sum(vv, dt, 0.00, 1.0 / 3.0)
        out["vti_0_50"] = self._window_sum(vv, dt, 0.00, 0.50)
        out["vti_0_75"] = self._window_sum(vv, dt, 0.00, 0.75)
        out["vti_75_90"] = self._window_sum(vv, dt, 0.75, 0.90)
        out["vti_90_100"] = self._window_sum(vv, dt, 0.90, 1.00)
        out["vti_late_half"] = self._window_sum(vv, dt, 0.50, 1.00)

        out["vti_above_min"] = float(np.nansum(np.maximum(vv - vmin, 0.0)) * dt)
        out["vti_above_mean_pos"] = float(np.nansum(np.maximum(vv - vmean, 0.0)) * dt)
        out["vti_below_mean_abs"] = float(np.nansum(np.maximum(vmean - vv, 0.0)) * dt)

        # -------------------------------------------------------------
        # Windowed absolute velocity levels
        # -------------------------------------------------------------
        out["v_start_mean"] = self._window_mean(vv, 0.00, 0.10)
        out["v_early_mean"] = self._window_mean(vv, 0.00, 1.0 / 3.0)
        out["v_mid_mean"] = self._window_mean(vv, 1.0 / 3.0, 2.0 / 3.0)
        out["v_late_mean"] = self._window_mean(vv, 2.0 / 3.0, 1.00)
        out["vend_mean"] = self._window_mean(vv, self.ratio_vend_start, self.ratio_vend_end)
        out["v_final_mean"] = self._window_mean(vv, 0.90, 1.00)
        out["v_early_max"] = self._window_max(vv, 0.00, 1.0 / 3.0)
        out["v_late_min"] = self._window_min(vv, 0.50, 1.00)

        # -------------------------------------------------------------
        # Absolute pulsatile-component metrics
        # -------------------------------------------------------------
        vac = vv - vmean
        out["v_ac_rms"] = float(np.sqrt(np.nanmean(vac * vac)))
        out["v_ac_abs_mean"] = float(np.nanmean(np.abs(vac)))
        out["v_ac_abs_integral"] = float(np.nansum(np.abs(vac)) * dt)
        out["v_positive_pulsatile_integral"] = float(
            np.nansum(np.maximum(vac, 0.0)) * dt
        )
        out["v_negative_pulsatile_integral"] = float(
            np.nansum(np.maximum(-vac, 0.0)) * dt
        )
        out["v_peak_to_peak"] = v_range

        # -------------------------------------------------------------
        # Absolute-time event metrics
        # -------------------------------------------------------------
        idx_peak = int(np.nanargmax(vv))
        idx_min = int(np.nanargmin(vv))

        out["t_vmax"] = float(idx_peak * dt)
        out["t_vmin"] = float(idx_min * dt)
        out["beat_period"] = float(Tbeat)

        if n > 0 and idx_peak >= 0 and idx_min >= 0:
            delta_idx = int((idx_min - idx_peak) % n)
            out["peak_to_trough_time"] = (
                float(delta_idx * dt) if delta_idx != 0 else np.nan
            )

        # -------------------------------------------------------------
        # Absolute derivative / kinetic metrics
        # -------------------------------------------------------------
        if n >= 2:
            dvdt = np.gradient(vf, dt)

            out["dvdt_max"] = float(np.nanmax(dvdt))
            out["dvdt_min"] = float(np.nanmin(dvdt))
            out["dvdt_fall_abs_max"] = float(abs(np.nanmin(dvdt)))
            out["dvdt_rms"] = float(np.sqrt(np.nanmean(dvdt * dvdt)))
            out["dvdt_abs_mean"] = float(np.nanmean(np.abs(dvdt)))
            out["dvdt_std"] = float(np.nanstd(dvdt))
            out["dvdt_energy"] = float(np.nansum(dvdt * dvdt) * dt)
            out["total_variation"] = float(np.nansum(np.abs(dvdt)) * dt)
            out["positive_variation"] = float(np.nansum(np.maximum(dvdt, 0.0)) * dt)
            out["negative_variation"] = float(np.nansum(np.maximum(-dvdt, 0.0)) * dt)

            idx_up = int(np.nanargmax(dvdt))
            idx_down = int(np.nanargmin(dvdt))
            out["t_upstroke_max"] = float(idx_up * dt)
            out["t_downstroke_max"] = float(idx_down * dt)

        # -------------------------------------------------------------
        # Absolute harmonic-amplitude metrics
        # -------------------------------------------------------------
        if n >= 2:
            Vfull = np.fft.rfft(vf) / float(n)
            H = int(min(self.H_MAX, Vfull.size - 1))

            coeff_abs = np.full((self.H_MAX + 1,), np.nan, dtype=float)
            harmonic_power = np.full((self.H_MAX + 1,), np.nan, dtype=float)
            harmonic_amp = np.full((self.H_MAX,), np.nan, dtype=float)

            coeff = np.abs(Vfull[: H + 1])
            power = coeff * coeff

            coeff_abs[: H + 1] = coeff
            harmonic_power[: H + 1] = power

            if H >= 1:
                harmonic_amp[:H] = 2.0 * coeff[1 : H + 1]

            out["harmonic_coeff_abs"] = coeff_abs
            out["harmonic_amp"] = harmonic_amp
            out["harmonic_power"] = harmonic_power

            out["dc_level"] = float(np.real(Vfull[0]))
            out["fundamental_amp"] = float(2.0 * coeff[1]) if H >= 1 else np.nan
            out["second_harmonic_amp"] = float(2.0 * coeff[2]) if H >= 2 else np.nan
            out["third_harmonic_amp"] = float(2.0 * coeff[3]) if H >= 3 else np.nan

            pulsatile_harmonic_power = (
                float(np.nansum(power[1 : H + 1])) if H >= 1 else np.nan
            )
            higher_harmonic_power = (
                float(np.nansum(power[2 : H + 1])) if H >= 2 else np.nan
            )
            Hlow = int(min(H, self.H_LOW_MAX))
            low_harmonic_power = (
                float(np.nansum(power[1 : Hlow + 1])) if Hlow >= 1 else np.nan
            )

            out["pulsatile_harmonic_power"] = pulsatile_harmonic_power
            out["higher_harmonic_power"] = higher_harmonic_power
            out["low_harmonic_power"] = low_harmonic_power

            if np.isfinite(pulsatile_harmonic_power) and pulsatile_harmonic_power >= 0:
                out["bandlimited_ac_rms_from_harmonics"] = float(
                    np.sqrt(2.0 * pulsatile_harmonic_power)
                )

        # -------------------------------------------------------------
        # Absolute signal-energy metrics
        # -------------------------------------------------------------
        out["signal_energy"] = float(np.nansum(vv * vv) * dt)
        out["signal_mean_square"] = float(np.nanmean(vv * vv))
        out["pulsatile_energy"] = float(np.nansum(vac * vac) * dt)
        out["pulsatile_mean_square"] = float(np.nanmean(vac * vac))
        out["absolute_deviation_energy"] = float(np.nansum(np.abs(vac)) * dt)
        out["vti_squared_over_T"] = (
            float((vti_total * vti_total) / Tbeat) if Tbeat > 0 else np.nan
        )

        return out

    def _compute_qc_metrics_1d(
        self, v_raw: np.ndarray, v_band: np.ndarray, Tbeat: float
    ) -> dict:
        """
        Compute raw-vs-bandlimited agreement metrics from a pair of aligned waveforms.
        """
        out = self._nan_qc_metrics_1d()

        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        v_raw = self._rectify_keep_nan(v_raw)
        v_band = self._rectify_keep_nan(v_band)

        n = int(min(v_raw.size, v_band.size))
        if n <= 0:
            return out

        raw = v_raw[:n]
        band = v_band[:n]

        mask = np.isfinite(raw) & np.isfinite(band)
        if not np.any(mask):
            return out

        dt = float(Tbeat) / float(n)

        r = np.full((n,), np.nan, dtype=float)
        r[mask] = raw[mask] - band[mask]
        rf = np.where(np.isfinite(r), r, 0.0)

        out["raw_minus_band_mean"] = self._safe_nanmean(r)
        out["raw_minus_band_bias"] = float(np.nansum(rf) * dt / Tbeat)
        out["raw_minus_band_rms"] = float(np.sqrt(np.nanmean(r * r)))
        out["raw_minus_band_mae"] = float(np.nanmean(np.abs(r)))
        out["raw_minus_band_max_abs"] = float(np.nanmax(np.abs(r)))
        out["raw_minus_band_energy"] = float(np.nansum(rf * rf) * dt)
        out["raw_minus_band_vti_abs"] = float(np.nansum(np.abs(rf)) * dt)
        out["raw_band_vti_difference"] = float(np.nansum(rf) * dt)

        paired_raw = raw[mask]
        paired_band = band[mask]
        if paired_raw.size >= 2:
            raw_std = float(np.nanstd(paired_raw))
            band_std = float(np.nanstd(paired_band))
            if raw_std > self.eps and band_std > self.eps:
                corr = np.corrcoef(paired_raw, paired_band)[0, 1]
                out["raw_band_corr"] = float(corr) if np.isfinite(corr) else np.nan

        return out

    # ---------------------------------------------------------------------
    # Block processors
    # ---------------------------------------------------------------------
    def _compute_block_segment(self, v_block: np.ndarray, T: np.ndarray):
        """
        v_block: (n_t, n_beats, n_branches, n_radii)

        Returns:
          per-segment arrays: (n_beats, n_branches, n_radii[, n_h])
          per-branch arrays:  (n_beats, n_branches[, n_h])
          global arrays:      (n_beats[, n_h])
        """
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}"
            )

        _, n_beats, n_branches, n_radii = v_block.shape

        seg = {
            k[0]: np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            for k in self._scalar_metric_keys()
        }
        br = {
            k[0]: np.full((n_beats, n_branches), np.nan, dtype=float)
            for k in self._scalar_metric_keys()
        }
        gl = {
            k[0]: np.full((n_beats,), np.nan, dtype=float)
            for k in self._scalar_metric_keys()
        }

        for k in self._array_metric_keys():
            name = k[0]
            dim = int(k[3])
            seg[name] = np.full((n_beats, n_branches, n_radii, dim), np.nan, dtype=float)
            br[name] = np.full((n_beats, n_branches, dim), np.nan, dtype=float)
            gl[name] = np.full((n_beats, dim), np.nan, dtype=float)

        for beat_idx in range(n_beats):
            Tbeat = self._get_Tbeat(T, beat_idx)

            gl_vals = {k[0]: [] for k in self._scalar_metric_keys()}
            gl_arr_vals = {k[0]: [] for k in self._array_metric_keys()}

            for branch_idx in range(n_branches):
                br_vals = {k[0]: [] for k in self._scalar_metric_keys()}
                br_arr_vals = {k[0]: [] for k in self._array_metric_keys()}

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    m = self._compute_metrics_1d(v, Tbeat)

                    for k in self._scalar_metric_keys():
                        key = k[0]
                        seg[key][beat_idx, branch_idx, radius_idx] = m[key]
                        br_vals[key].append(m[key])
                        gl_vals[key].append(m[key])

                    for k in self._array_metric_keys():
                        key = k[0]
                        seg[key][beat_idx, branch_idx, radius_idx, :] = m[key]
                        br_arr_vals[key].append(m[key])
                        gl_arr_vals[key].append(m[key])

                for k in self._scalar_metric_keys():
                    key = k[0]
                    br[key][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[key], dtype=float)
                    )

                for k in self._array_metric_keys():
                    key = k[0]
                    br[key][beat_idx, branch_idx, :] = self._safe_nanmedian_array(
                        np.asarray(br_arr_vals[key], dtype=float), axis=0
                    )

            for k in self._scalar_metric_keys():
                key = k[0]
                gl[key][beat_idx] = self._safe_nanmedian(
                    np.asarray(gl_vals[key], dtype=float)
                )

            for k in self._array_metric_keys():
                key = k[0]
                gl[key][beat_idx, :] = self._safe_nanmedian_array(
                    np.asarray(gl_arr_vals[key], dtype=float), axis=0
                )

        seg_order_note = "segment arrays are stored as (beat, branch, radius[, harmonic])"
        return seg, br, gl, n_branches, n_radii, seg_order_note

    def _compute_qc_block_segment(
        self, v_raw_block: np.ndarray, v_band_block: np.ndarray, T: np.ndarray
    ):
        """
        Raw-vs-bandlimited QC for segment waveforms.

        Returns:
          per-segment arrays: (n_beats, n_branches, n_radii)
          per-branch arrays:  (n_beats, n_branches)
          global arrays:      (n_beats,)
        """
        if v_raw_block.ndim != 4 or v_band_block.ndim != 4:
            raise ValueError(
                "Expected raw and bandlimited segment blocks shaped "
                "(n_t,n_beats,n_branches,n_radii)"
            )

        n_beats = min(v_raw_block.shape[1], v_band_block.shape[1])
        n_branches = min(v_raw_block.shape[2], v_band_block.shape[2])
        n_radii = min(v_raw_block.shape[3], v_band_block.shape[3])

        seg = {
            k[0]: np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            for k in self._qc_metric_keys()
        }
        br = {
            k[0]: np.full((n_beats, n_branches), np.nan, dtype=float)
            for k in self._qc_metric_keys()
        }
        gl = {
            k[0]: np.full((n_beats,), np.nan, dtype=float)
            for k in self._qc_metric_keys()
        }

        for beat_idx in range(n_beats):
            Tbeat = self._get_Tbeat(T, beat_idx)
            gl_vals = {k[0]: [] for k in self._qc_metric_keys()}

            for branch_idx in range(n_branches):
                br_vals = {k[0]: [] for k in self._qc_metric_keys()}

                for radius_idx in range(n_radii):
                    v_raw = v_raw_block[:, beat_idx, branch_idx, radius_idx]
                    v_band = v_band_block[:, beat_idx, branch_idx, radius_idx]
                    m = self._compute_qc_metrics_1d(v_raw, v_band, Tbeat)

                    for k in self._qc_metric_keys():
                        key = k[0]
                        seg[key][beat_idx, branch_idx, radius_idx] = m[key]
                        br_vals[key].append(m[key])
                        gl_vals[key].append(m[key])

                for k in self._qc_metric_keys():
                    key = k[0]
                    br[key][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[key], dtype=float)
                    )

            for k in self._qc_metric_keys():
                key = k[0]
                gl[key][beat_idx] = self._safe_nanmedian(
                    np.asarray(gl_vals[key], dtype=float)
                )

        seg_order_note = "segment arrays are stored as (beat, branch, radius)"
        return seg, br, gl, n_branches, n_radii, seg_order_note

    def _compute_block_global(self, v_global: np.ndarray, T: np.ndarray):
        """
        v_global: (n_t, n_beats) after _ensure_time_by_beat.
        Returns dict of arrays shaped (n_beats,) or (n_beats, n_h).
        """
        n_beats = self._n_beats_from_T(T)
        v_global = self._ensure_time_by_beat(v_global, n_beats)
        v_global = self._rectify_keep_nan(v_global)

        out = {
            k[0]: np.full((n_beats,), np.nan, dtype=float)
            for k in self._scalar_metric_keys()
        }
        for k in self._array_metric_keys():
            out[k[0]] = np.full((n_beats, int(k[3])), np.nan, dtype=float)

        for beat_idx in range(n_beats):
            Tbeat = self._get_Tbeat(T, beat_idx)
            v = v_global[:, beat_idx]
            m = self._compute_metrics_1d(v, Tbeat)

            for k in self._scalar_metric_keys():
                out[k[0]][beat_idx] = m[k[0]]

            for k in self._array_metric_keys():
                out[k[0]][beat_idx, :] = m[k[0]]

        return out

    def _compute_qc_block_global(
        self, v_raw_global: np.ndarray, v_band_global: np.ndarray, T: np.ndarray
    ):
        """
        Raw-vs-bandlimited QC for global waveforms.
        """
        n_beats = self._n_beats_from_T(T)
        v_raw_global = self._ensure_time_by_beat(v_raw_global, n_beats)
        v_band_global = self._ensure_time_by_beat(v_band_global, n_beats)

        out = {
            k[0]: np.full((n_beats,), np.nan, dtype=float)
            for k in self._qc_metric_keys()
        }

        for beat_idx in range(n_beats):
            Tbeat = self._get_Tbeat(T, beat_idx)
            v_raw = v_raw_global[:, beat_idx]
            v_band = v_band_global[:, beat_idx]
            m = self._compute_qc_metrics_1d(v_raw, v_band, Tbeat)

            for k in self._qc_metric_keys():
                out[k[0]][beat_idx] = m[k[0]]

        return out

    # ---------------------------------------------------------------------
    # Packing helpers
    # ---------------------------------------------------------------------
    def _metric_meta(self) -> dict:
        meta = {}
        for k in self._scalar_metric_keys():
            meta[k[0]] = {
                "definition": [k[1]],
                "unit": [k[2]],
                "latex_formula": [k[4]],
                "metric_type": ["absolute_gain_sensitive"],
            }
        for k in self._array_metric_keys():
            meta[k[0]] = {
                "definition": [k[1]],
                "unit": [k[2]],
                "latex_formula": [k[4]],
                "metric_type": ["absolute_gain_sensitive"],
                "array_axis": [k[5]],
            }
        for k in self._qc_metric_keys():
            meta[k[0]] = {
                "definition": [k[1]],
                "unit": [k[2]],
                "latex_formula": [k[3]],
                "metric_type": ["raw_vs_bandlimited_qc"],
            }
        return meta

    def _pack_dict(
        self,
        metrics: dict,
        path_prefix: str,
        d: dict,
        attrs_common: dict | None = None,
    ) -> None:
        meta = self._metric_meta()
        attrs_common = attrs_common or {}

        for key, arr in d.items():
            attrs = {}
            attrs.update(meta.get(key, {}))
            attrs.update(attrs_common)
            metrics[f"{path_prefix}/{key}"] = with_attrs(arr, attrs)

    def _pack_params(self, metrics: dict, vessel_prefix: str, scope: str) -> None:
        metrics[f"{vessel_prefix}/{scope}/params/eps"] = np.asarray(self.eps, dtype=float)
        metrics[f"{vessel_prefix}/{scope}/params/ratio_vend_start"] = np.asarray(
            self.ratio_vend_start, dtype=float
        )
        metrics[f"{vessel_prefix}/{scope}/params/ratio_vend_end"] = np.asarray(
            self.ratio_vend_end, dtype=float
        )
        metrics[f"{vessel_prefix}/{scope}/params/H_LOW_MAX"] = np.asarray(
            self.H_LOW_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/{scope}/params/H_MAX"] = np.asarray(
            self.H_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/{scope}/params/is_gain_sensitive"] = np.asarray(
            1, dtype=int
        )

    def _pack_segment_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_raw_seg: np.ndarray,
        v_band_seg: np.ndarray,
        T: np.ndarray,
    ) -> None:
        seg_b, br_b, gl_b, nb_b, nr_b, seg_note_b = self._compute_block_segment(
            v_band_seg, T
        )
        seg_r, br_r, gl_r, nb_r, nr_r, seg_note_r = self._compute_block_segment(
            v_raw_seg, T
        )

        seg_note = seg_note_b
        if (nb_b != nb_r) or (nr_b != nr_r):
            seg_note = seg_note_b + " | WARNING: raw/band branch/radius dims differ."

        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/bandlimited_segment",
            seg_b,
            {
                "definition_scope": ["per-segment metrics stored as (beat, branch, radius[, harmonic])"],
                "segment_indexing": [seg_note],
                "waveform_representation": ["bandlimited"],
            },
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_segment",
            seg_r,
            {
                "definition_scope": ["per-segment metrics stored as (beat, branch, radius[, harmonic])"],
                "segment_indexing": [seg_note],
                "waveform_representation": ["raw"],
            },
        )

        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/bandlimited_branch",
            br_b,
            {
                "definition_scope": ["median over radii per branch"],
                "waveform_representation": ["bandlimited"],
            },
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_branch",
            br_r,
            {
                "definition_scope": ["median over radii per branch"],
                "waveform_representation": ["raw"],
            },
        )

        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/bandlimited_global",
            gl_b,
            {
                "definition_scope": ["median over all branch-radius segment values per beat"],
                "waveform_representation": ["bandlimited"],
            },
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_global",
            gl_r,
            {
                "definition_scope": ["median over all branch-radius segment values per beat"],
                "waveform_representation": ["raw"],
            },
        )

        qc_seg, qc_br, qc_gl, _, _, qc_note = self._compute_qc_block_segment(
            v_raw_seg, v_band_seg, T
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_vs_bandlimited_segment",
            qc_seg,
            {
                "definition_scope": ["per-segment raw-vs-bandlimited QC metrics"],
                "segment_indexing": [qc_note],
            },
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_vs_bandlimited_branch",
            qc_br,
            {"definition_scope": ["median over radii per branch"]},
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/raw_vs_bandlimited_global",
            qc_gl,
            {"definition_scope": ["median over all branch-radius segment values per beat"]},
        )

        self._pack_params(metrics, vessel_prefix, "by_segment")

    def _pack_global_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_raw_gl: np.ndarray,
        v_band_gl: np.ndarray,
        T: np.ndarray,
    ) -> None:
        out_raw = self._compute_block_global(v_raw_gl, T)
        out_band = self._compute_block_global(v_band_gl, T)
        out_qc = self._compute_qc_block_global(v_raw_gl, v_band_gl, T)

        self._pack_dict(
            metrics,
            f"{vessel_prefix}/global/raw",
            out_raw,
            {"definition_scope": ["global waveform metrics"], "waveform_representation": ["raw"]},
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/global/bandlimited",
            out_band,
            {
                "definition_scope": ["global waveform metrics"],
                "waveform_representation": ["bandlimited"],
            },
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/global/raw_vs_bandlimited",
            out_qc,
            {"definition_scope": ["global raw-vs-bandlimited QC metrics"]},
        )

        self._pack_params(metrics, vessel_prefix, "global")

    # ---------------------------------------------------------------------
    # Metric registries
    # ---------------------------------------------------------------------
    @staticmethod
    def _scalar_metric_keys() -> list[list]:
        return [
            # Absolute velocity level / extrema
            ["vmax", "max_t v(t)", "velocity", None, r"$v_{\max}=\max_t v(t)$"],
            ["vmin", "min_t v(t)", "velocity", None, r"$v_{\min}=\min_t v(t)$"],
            ["vmean", "mean_t v(t)", "velocity", None, r"$\bar v=T^{-1}\int_0^T v(t)\,dt$"],
            ["vmedian", "median_t v(t)", "velocity", None, r"$\mathrm{median}_t\,v(t)$"],
            ["vrms", "sqrt(mean_t v(t)^2)", "velocity", None, r"$v_{\mathrm{RMS}}=\sqrt{T^{-1}\int_0^T v(t)^2\,dt}$"],
            ["vstd", "standard deviation of v(t)", "velocity", None, r"$\sigma_v=\mathrm{std}_t(v(t))$"],
            ["v_range", "vmax-vmin", "velocity", None, r"$v_{\max}-v_{\min}$"],
            ["v_peak_above_mean", "vmax-vmean", "velocity", None, r"$v_{\max}-\bar v$"],
            ["v_mean_above_min", "vmean-vmin", "velocity", None, r"$\bar v-v_{\min}$"],
            ["v_p05", "5th percentile of v(t)", "velocity", None, r"$P_5[v]$"],
            ["v_p10", "10th percentile of v(t)", "velocity", None, r"$P_{10}[v]$"],
            ["v_p25", "25th percentile of v(t)", "velocity", None, r"$P_{25}[v]$"],
            ["v_p50", "50th percentile of v(t)", "velocity", None, r"$P_{50}[v]$"],
            ["v_p75", "75th percentile of v(t)", "velocity", None, r"$P_{75}[v]$"],
            ["v_p90", "90th percentile of v(t)", "velocity", None, r"$P_{90}[v]$"],
            ["v_p95", "95th percentile of v(t)", "velocity", None, r"$P_{95}[v]$"],
            ["v_iqr", "v_p75-v_p25", "velocity", None, r"$P_{75}[v]-P_{25}[v]$"],
            ["v_mad", "median absolute deviation from vmedian", "velocity", None, r"$\mathrm{median}_t|v(t)-\mathrm{median}(v)|$"],

            # Absolute VTI / displacement-like metrics
            ["vti_total", "int_0^T v(t) dt", "velocity*s", None, r"$\int_0^T v(t)\,dt$"],
            ["vti_0_10", "int_0^{0.10T} v(t) dt", "velocity*s", None, r"$\int_0^{0.10T}v(t)\,dt$"],
            ["vti_0_25", "int_0^{0.25T} v(t) dt", "velocity*s", None, r"$\int_0^{0.25T}v(t)\,dt$"],
            ["vti_0_33", "int_0^{T/3} v(t) dt", "velocity*s", None, r"$\int_0^{T/3}v(t)\,dt$"],
            ["vti_0_50", "int_0^{0.50T} v(t) dt", "velocity*s", None, r"$\int_0^{0.50T}v(t)\,dt$"],
            ["vti_0_75", "int_0^{0.75T} v(t) dt", "velocity*s", None, r"$\int_0^{0.75T}v(t)\,dt$"],
            ["vti_75_90", "int_{0.75T}^{0.90T} v(t) dt", "velocity*s", None, r"$\int_{0.75T}^{0.90T}v(t)\,dt$"],
            ["vti_90_100", "int_{0.90T}^{T} v(t) dt", "velocity*s", None, r"$\int_{0.90T}^{T}v(t)\,dt$"],
            ["vti_late_half", "int_{0.50T}^{T} v(t) dt", "velocity*s", None, r"$\int_{0.50T}^{T}v(t)\,dt$"],
            ["vti_above_min", "int_0^T max(v(t)-vmin,0) dt", "velocity*s", None, r"$\int_0^T\max(v(t)-v_{\min},0)\,dt$"],
            ["vti_above_mean_pos", "int_0^T max(v(t)-vmean,0) dt", "velocity*s", None, r"$\int_0^T\max(v(t)-\bar v,0)\,dt$"],
            ["vti_below_mean_abs", "int_0^T max(vmean-v(t),0) dt", "velocity*s", None, r"$\int_0^T\max(\bar v-v(t),0)\,dt$"],

            # Windowed absolute velocity levels
            ["v_start_mean", "mean v(t) over [0,0.10T]", "velocity", None, r"$\langle v\rangle_{[0,0.10T]}$"],
            ["v_early_mean", "mean v(t) over [0,T/3]", "velocity", None, r"$\langle v\rangle_{[0,T/3]}$"],
            ["v_mid_mean", "mean v(t) over [T/3,2T/3]", "velocity", None, r"$\langle v\rangle_{[T/3,2T/3]}$"],
            ["v_late_mean", "mean v(t) over [2T/3,T]", "velocity", None, r"$\langle v\rangle_{[2T/3,T]}$"],
            ["vend_mean", "mean v(t) over [ratio_vend_start*T, ratio_vend_end*T]", "velocity", None, r"$\langle v\rangle_{[\alpha T,\beta T]}$"],
            ["v_final_mean", "mean v(t) over [0.90T,T]", "velocity", None, r"$\langle v\rangle_{[0.90T,T]}$"],
            ["v_early_max", "max v(t) over [0,T/3]", "velocity", None, r"$\max_{t\in[0,T/3]}v(t)$"],
            ["v_late_min", "min v(t) over [0.50T,T]", "velocity", None, r"$\min_{t\in[0.50T,T]}v(t)$"],

            # Absolute pulsatile-component metrics
            ["v_ac_rms", "RMS of v(t)-vmean", "velocity", None, r"$\sqrt{T^{-1}\int_0^T(v(t)-\bar v)^2\,dt}$"],
            ["v_ac_abs_mean", "mean absolute deviation from vmean", "velocity", None, r"$T^{-1}\int_0^T|v(t)-\bar v|\,dt$"],
            ["v_ac_abs_integral", "int_0^T |v(t)-vmean| dt", "velocity*s", None, r"$\int_0^T|v(t)-\bar v|\,dt$"],
            ["v_positive_pulsatile_integral", "int_0^T max(v(t)-vmean,0) dt", "velocity*s", None, r"$\int_0^T\max(v(t)-\bar v,0)\,dt$"],
            ["v_negative_pulsatile_integral", "int_0^T max(vmean-v(t),0) dt", "velocity*s", None, r"$\int_0^T\max(\bar v-v(t),0)\,dt$"],
            ["v_peak_to_peak", "vmax-vmin", "velocity", None, r"$v_{\max}-v_{\min}$"],

            # Absolute-time event metrics
            ["t_vmax", "time of maximum v(t)", "seconds", None, r"$t_{v_{\max}}$"],
            ["t_vmin", "time of minimum v(t)", "seconds", None, r"$t_{v_{\min}}$"],
            ["t_upstroke_max", "time of maximum dv/dt", "seconds", None, r"$t_{\max(dv/dt)}$"],
            ["t_downstroke_max", "time of minimum dv/dt", "seconds", None, r"$t_{\min(dv/dt)}$"],
            ["peak_to_trough_time", "circular forward time from peak to trough", "seconds", None, r"$(t_{\min}-t_{\max})\bmod T$"],
            ["beat_period", "beat period T", "seconds", None, r"$T$"],

            # Absolute derivative / kinetic metrics
            ["dvdt_max", "max_t dv/dt", "velocity/s", None, r"$\max_t\,dv/dt$"],
            ["dvdt_min", "min_t dv/dt", "velocity/s", None, r"$\min_t\,dv/dt$"],
            ["dvdt_fall_abs_max", "|min_t dv/dt|", "velocity/s", None, r"$|\min_t\,dv/dt|$"],
            ["dvdt_rms", "RMS of dv/dt", "velocity/s", None, r"$\sqrt{T^{-1}\int_0^T(dv/dt)^2\,dt}$"],
            ["dvdt_abs_mean", "mean absolute dv/dt", "velocity/s", None, r"$T^{-1}\int_0^T|dv/dt|\,dt$"],
            ["dvdt_std", "standard deviation of dv/dt", "velocity/s", None, r"$\mathrm{std}_t(dv/dt)$"],
            ["dvdt_energy", "int_0^T (dv/dt)^2 dt", "velocity^2/s", None, r"$\int_0^T(dv/dt)^2\,dt$"],
            ["total_variation", "int_0^T |dv/dt| dt", "velocity", None, r"$\int_0^T|dv/dt|\,dt$"],
            ["positive_variation", "int_0^T max(dv/dt,0) dt", "velocity", None, r"$\int_0^T\max(dv/dt,0)\,dt$"],
            ["negative_variation", "int_0^T max(-dv/dt,0) dt", "velocity", None, r"$\int_0^T\max(-dv/dt,0)\,dt$"],

            # Absolute harmonic-amplitude scalar metrics
            ["dc_level", "DC Fourier coefficient V0", "velocity", None, r"$V_0$"],
            ["fundamental_amp", "2*abs(V1)", "velocity", None, r"$2|V_1|$"],
            ["second_harmonic_amp", "2*abs(V2)", "velocity", None, r"$2|V_2|$"],
            ["third_harmonic_amp", "2*abs(V3)", "velocity", None, r"$2|V_3|$"],
            ["pulsatile_harmonic_power", "sum_{h=1}^H |Vh|^2", "velocity^2", None, r"$\sum_{h=1}^H |V_h|^2$"],
            ["higher_harmonic_power", "sum_{h=2}^H |Vh|^2", "velocity^2", None, r"$\sum_{h=2}^H |V_h|^2$"],
            ["low_harmonic_power", "sum_{h=1}^{H_LOW_MAX} |Vh|^2", "velocity^2", None, r"$\sum_{h=1}^{H_{\mathrm{low}}}|V_h|^2$"],
            ["bandlimited_ac_rms_from_harmonics", "sqrt(2*sum_{h=1}^H |Vh|^2)", "velocity", None, r"$\sqrt{2\sum_{h=1}^H |V_h|^2}$"],

            # Absolute signal-energy metrics
            ["signal_energy", "int_0^T v(t)^2 dt", "velocity^2*s", None, r"$\int_0^T v(t)^2\,dt$"],
            ["signal_mean_square", "mean_t v(t)^2", "velocity^2", None, r"$T^{-1}\int_0^T v(t)^2\,dt$"],
            ["pulsatile_energy", "int_0^T (v(t)-vmean)^2 dt", "velocity^2*s", None, r"$\int_0^T(v(t)-\bar v)^2\,dt$"],
            ["pulsatile_mean_square", "mean_t (v(t)-vmean)^2", "velocity^2", None, r"$T^{-1}\int_0^T(v(t)-\bar v)^2\,dt$"],
            ["absolute_deviation_energy", "int_0^T |v(t)-vmean| dt", "velocity*s", None, r"$\int_0^T|v(t)-\bar v|\,dt$"],
            ["vti_squared_over_T", "(int_0^T v(t)dt)^2/T", "velocity^2*s", None, r"$\left(\int_0^T v(t)\,dt\right)^2/T$"],
        ]

    def _array_metric_keys(self) -> list[list]:
        return [
            [
                "harmonic_coeff_abs",
                "abs(Vh) for h=0..H_MAX, V=rfft(v)/n",
                "velocity",
                self.H_MAX + 1,
                r"$|V_h|,\ h=0,\ldots,H_{\max}$",
                "harmonic index h=0..H_MAX",
            ],
            [
                "harmonic_amp",
                "2*abs(Vh) for h=1..H_MAX",
                "velocity",
                self.H_MAX,
                r"$2|V_h|,\ h=1,\ldots,H_{\max}$",
                "harmonic index h=1..H_MAX stored at index h-1",
            ],
            [
                "harmonic_power",
                "abs(Vh)^2 for h=0..H_MAX",
                "velocity^2",
                self.H_MAX + 1,
                r"$|V_h|^2,\ h=0,\ldots,H_{\max}$",
                "harmonic index h=0..H_MAX",
            ],
        ]

    @staticmethod
    def _qc_metric_keys() -> list[list]:
        return [
            [
                "raw_minus_band_mean",
                "mean_t(raw(t)-bandlimited(t))",
                "velocity",
                r"$\langle v_{\mathrm{raw}}-v_{\mathrm{band}}\rangle$",
            ],
            [
                "raw_minus_band_bias",
                "T^{-1} int_0^T (raw(t)-bandlimited(t)) dt",
                "velocity",
                r"$T^{-1}\int_0^T(v_{\mathrm{raw}}-v_{\mathrm{band}})\,dt$",
            ],
            [
                "raw_minus_band_rms",
                "RMS of raw(t)-bandlimited(t)",
                "velocity",
                r"$\sqrt{\langle (v_{\mathrm{raw}}-v_{\mathrm{band}})^2\rangle}$",
            ],
            [
                "raw_minus_band_mae",
                "mean_t |raw(t)-bandlimited(t)|",
                "velocity",
                r"$\langle |v_{\mathrm{raw}}-v_{\mathrm{band}}|\rangle$",
            ],
            [
                "raw_minus_band_max_abs",
                "max_t |raw(t)-bandlimited(t)|",
                "velocity",
                r"$\max_t |v_{\mathrm{raw}}-v_{\mathrm{band}}|$",
            ],
            [
                "raw_minus_band_energy",
                "int_0^T (raw(t)-bandlimited(t))^2 dt",
                "velocity^2*s",
                r"$\int_0^T(v_{\mathrm{raw}}-v_{\mathrm{band}})^2\,dt$",
            ],
            [
                "raw_minus_band_vti_abs",
                "int_0^T |raw(t)-bandlimited(t)| dt",
                "velocity*s",
                r"$\int_0^T|v_{\mathrm{raw}}-v_{\mathrm{band}}|\,dt$",
            ],
            [
                "raw_band_corr",
                "Pearson correlation between raw and bandlimited waveforms",
                "",
                r"$\rho(v_{\mathrm{raw}},v_{\mathrm{band}})$",
            ],
            [
                "raw_band_vti_difference",
                "int_0^T raw(t)dt - int_0^T bandlimited(t)dt",
                "velocity*s",
                r"$\int_0^T v_{\mathrm{raw}}\,dt-\int_0^T v_{\mathrm{band}}\,dt$",
            ],
        ]

    # ---------------------------------------------------------------------
    # Entrypoint
    # ---------------------------------------------------------------------
    def run(self, h5file) -> ProcessResult:
        T = np.asarray(h5file[self.T_input])
        metrics = {}

        vessel_configs = [
            {
                "prefix": "artery",
                "v_raw_segment_input": self.v_raw_segment_input,
                "v_band_segment_input": self.v_band_segment_input,
                "v_raw_global_input": self.v_raw_global_input,
                "v_band_global_input": self.v_band_global_input,
            },
            {
                "prefix": "vein",
                "v_raw_segment_input": self.v_raw_segment_input_vein,
                "v_band_segment_input": self.v_band_segment_input_vein,
                "v_raw_global_input": self.v_raw_global_input_vein,
                "v_band_global_input": self.v_band_global_input_vein,
            },
        ]

        for cfg in vessel_configs:
            vessel_prefix = cfg["prefix"]

            have_seg = (
                cfg["v_raw_segment_input"] in h5file
                and cfg["v_band_segment_input"] in h5file
            )
            if have_seg:
                v_raw_seg = np.asarray(h5file[cfg["v_raw_segment_input"]])
                v_band_seg = np.asarray(h5file[cfg["v_band_segment_input"]])
                self._pack_segment_outputs(
                    metrics, vessel_prefix, v_raw_seg, v_band_seg, T
                )

            have_glob = (
                cfg["v_raw_global_input"] in h5file
                and cfg["v_band_global_input"] in h5file
            )
            if have_glob:
                v_raw_gl = np.asarray(h5file[cfg["v_raw_global_input"]])
                v_band_gl = np.asarray(h5file[cfg["v_band_global_input"]])
                self._pack_global_outputs(
                    metrics, vessel_prefix, v_raw_gl, v_band_gl, T
                )

        return ProcessResult(metrics=metrics)
