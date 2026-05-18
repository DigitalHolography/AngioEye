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

        raise ValueError(
            "Expected global waveform with one axis matching the beat-period "
            f"count ({n_beats}), got shape {v2.shape}"
        )

    @staticmethod
    def _Tbeat_values(T: np.ndarray) -> np.ndarray:
        original_shape = np.asarray(T).shape
        T = np.squeeze(np.asarray(T, dtype=float))
        if T.ndim == 0:
            return np.asarray([float(T)], dtype=float)
        if T.ndim == 1:
            return T.astype(float, copy=False)
        raise ValueError(
            "Expected beat-period array to be scalar, 1D, or singleton-padded "
            f"1D, got shape {original_shape}"
        )

    @classmethod
    def _n_beats_from_T(cls, T: np.ndarray) -> int:
        return int(cls._Tbeat_values(T).size)

    @classmethod
    def _get_Tbeat(cls, T: np.ndarray, beat_idx: int) -> float:
        Tbeats = cls._Tbeat_values(T)
        if beat_idx < 0 or beat_idx >= Tbeats.size:
            raise IndexError(
                f"Beat index {beat_idx} is outside beat-period array of length {Tbeats.size}"
            )
        return float(Tbeats[beat_idx])

    @staticmethod
    def _gradient_keep_nan(v: np.ndarray, dt: float) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        dvdt = np.full(v.shape, np.nan, dtype=float)
        finite = np.isfinite(v)
        if np.count_nonzero(finite) < 2:
            return dvdt

        idx = np.flatnonzero(finite)
        t = idx.astype(float) * float(dt)
        dvdt[finite] = np.gradient(v[finite], t)
        return dvdt

    @staticmethod
    def _rfft_amplitude_factors(n: int, H: int) -> np.ndarray:
        factors = np.ones((H + 1,), dtype=float)
        if H >= 1:
            factors[1:] = 2.0
            if n % 2 == 0 and H >= n // 2:
                factors[n // 2] = 1.0
        return factors

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
        window = v[k0:k1]
        if not np.any(np.isfinite(window)):
            return np.nan
        return float(np.nansum(window) * dt)

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
            dvdt = self._gradient_keep_nan(vv, dt)
            if np.any(np.isfinite(dvdt)):
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
            amp_factors = self._rfft_amplitude_factors(n, H)

            coeff_abs = np.full((self.H_MAX + 1,), np.nan, dtype=float)
            harmonic_power = np.full((self.H_MAX + 1,), np.nan, dtype=float)
            harmonic_amp = np.full((self.H_MAX,), np.nan, dtype=float)

            coeff = np.abs(Vfull[: H + 1])
            power = coeff * coeff

            coeff_abs[: H + 1] = coeff
            harmonic_power[: H + 1] = power

            if H >= 1:
                harmonic_amp[:H] = amp_factors[1 : H + 1] * coeff[1 : H + 1]

            out["harmonic_coeff_abs"] = coeff_abs
            out["harmonic_amp"] = harmonic_amp
            out["harmonic_power"] = harmonic_power

            out["dc_level"] = float(np.real(Vfull[0]))
            out["fundamental_amp"] = (
                float(amp_factors[1] * coeff[1]) if H >= 1 else np.nan
            )
            out["second_harmonic_amp"] = (
                float(amp_factors[2] * coeff[2]) if H >= 2 else np.nan
            )
            out["third_harmonic_amp"] = (
                float(amp_factors[3] * coeff[3]) if H >= 3 else np.nan
            )

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
                ac_mean_square = float(
                    np.nansum(amp_factors[1 : H + 1] * power[1 : H + 1])
                )
                out["bandlimited_ac_rms_from_harmonics"] = float(
                    np.sqrt(ac_mean_square)
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
        self,
        v_raw: np.ndarray,
        v_band: np.ndarray,
        Tbeat: float,
        *,
        rectify: bool = False,
    ) -> dict:
        """
        Compute raw-vs-bandlimited agreement metrics from a pair of aligned waveforms.
        """
        out = self._nan_qc_metrics_1d()

        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        v_raw = np.asarray(v_raw, dtype=float)
        v_band = np.asarray(v_band, dtype=float)
        if v_raw.size != v_band.size:
            raise ValueError(
                f"Raw and bandlimited waveforms have different lengths: "
                f"{v_raw.size} vs {v_band.size}. Resample before QC."
            )

        if rectify:
            v_raw = self._rectify_keep_nan(v_raw)
            v_band = self._rectify_keep_nan(v_band)

        n = int(v_raw.size)
        if n <= 0:
            return out

        raw = v_raw
        band = v_band

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
        self,
        v_raw_block: np.ndarray,
        v_band_block: np.ndarray,
        T: np.ndarray,
        *,
        rectify: bool = False,
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
                    m = self._compute_qc_metrics_1d(
                        v_raw, v_band, Tbeat, rectify=rectify
                    )

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
        self,
        v_raw_global: np.ndarray,
        v_band_global: np.ndarray,
        T: np.ndarray,
        *,
        rectify: bool = False,
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
            m = self._compute_qc_metrics_1d(
                v_raw, v_band, Tbeat, rectify=rectify
            )

            for k in self._qc_metric_keys():
                out[k[0]][beat_idx] = m[k[0]]

        return out

    # ---------------------------------------------------------------------
    # Packing helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _audit_metric_names() -> dict[str, str]:
        return {
            "vmax": "maximum_velocity",
            "vmin": "minimum_velocity",
            "vmean": "mean_velocity",
            "vmedian": "median_velocity",
            "vrms": "rms_velocity",
            "vstd": "velocity_standard_deviation",
            "v_range": "velocity_range",
            "v_peak_above_mean": "peak_velocity_above_mean",
            "v_mean_above_min": "mean_velocity_above_minimum",
            "v_p05": "velocity_percentile_05",
            "v_p10": "velocity_percentile_10",
            "v_p25": "velocity_percentile_25",
            "v_p50": "velocity_percentile_50",
            "v_p75": "velocity_percentile_75",
            "v_p90": "velocity_percentile_90",
            "v_p95": "velocity_percentile_95",
            "v_iqr": "velocity_interquartile_range",
            "v_mad": "velocity_median_absolute_deviation",
            "vti_total": "total_velocity_time_integral",
            "vti_0_10": "velocity_time_integral_first_10_percent",
            "vti_0_25": "velocity_time_integral_first_25_percent",
            "vti_0_33": "velocity_time_integral_first_third",
            "vti_0_50": "velocity_time_integral_first_half",
            "vti_0_75": "velocity_time_integral_first_75_percent",
            "vti_75_90": "velocity_time_integral_75_to_90_percent",
            "vti_90_100": "velocity_time_integral_final_10_percent",
            "vti_late_half": "velocity_time_integral_late_half",
            "vti_above_min": "velocity_time_integral_above_minimum",
            "vti_above_mean_pos": "positive_velocity_time_integral_above_mean",
            "vti_below_mean_abs": "negative_velocity_time_integral_below_mean",
            "v_start_mean": "mean_velocity_first_10_percent",
            "v_early_mean": "mean_velocity_first_third",
            "v_mid_mean": "mean_velocity_middle_third",
            "v_late_mean": "mean_velocity_final_third",
            "vend_mean": "mean_velocity_end_window",
            "v_final_mean": "mean_velocity_final_10_percent",
            "v_early_max": "maximum_velocity_first_third",
            "v_late_min": "minimum_velocity_late_half",
            "v_ac_rms": "rms_pulsatile_velocity",
            "v_ac_abs_mean": "mean_absolute_pulsatile_velocity",
            "v_ac_abs_integral": "integrated_absolute_pulsatile_velocity",
            "v_positive_pulsatile_integral": "positive_pulsatile_velocity_integral",
            "v_negative_pulsatile_integral": "negative_pulsatile_velocity_integral",
            "v_peak_to_peak": "peak_to_peak_velocity",
            "t_vmax": "time_of_maximum_velocity",
            "t_vmin": "time_of_minimum_velocity",
            "t_upstroke_max": "time_of_maximum_acceleration",
            "t_downstroke_max": "time_of_maximum_deceleration",
            "peak_to_trough_time": "time_from_peak_velocity_to_trough",
            "beat_period": "beat_period",
            "dvdt_max": "maximum_acceleration",
            "dvdt_min": "minimum_acceleration",
            "dvdt_fall_abs_max": "maximum_deceleration_magnitude",
            "dvdt_rms": "rms_acceleration",
            "dvdt_abs_mean": "mean_absolute_acceleration",
            "dvdt_std": "acceleration_standard_deviation",
            "dvdt_energy": "acceleration_energy",
            "total_variation": "total_velocity_change",
            "positive_variation": "integrated_positive_acceleration",
            "negative_variation": "integrated_deceleration_magnitude",
            "dc_level": "fourier_dc_velocity_level",
            "fundamental_amp": "fundamental_harmonic_velocity_amplitude",
            "second_harmonic_amp": "second_harmonic_velocity_amplitude",
            "third_harmonic_amp": "third_harmonic_velocity_amplitude",
            "pulsatile_harmonic_power": "pulsatile_harmonic_power",
            "higher_harmonic_power": "higher_harmonic_power",
            "low_harmonic_power": "low_harmonic_power",
            "bandlimited_ac_rms_from_harmonics": "rms_pulsatile_velocity_from_harmonics",
            "signal_energy": "velocity_signal_energy",
            "signal_mean_square": "velocity_mean_square",
            "pulsatile_energy": "pulsatile_velocity_energy",
            "pulsatile_mean_square": "pulsatile_velocity_mean_square",
            "absolute_deviation_energy": "absolute_pulsatile_deviation_integral",
            "vti_squared_over_T": "velocity_time_integral_squared_per_period",
            "harmonic_coeff_abs": "harmonic_velocity_coefficient_magnitude",
            "harmonic_amp": "harmonic_velocity_amplitude",
            "harmonic_power": "harmonic_velocity_power",
            "raw_minus_band_mean": "mean_raw_minus_bandlimited_velocity",
            "raw_minus_band_bias": "raw_minus_bandlimited_velocity_bias",
            "raw_minus_band_rms": "rms_raw_minus_bandlimited_velocity",
            "raw_minus_band_mae": "mean_absolute_raw_minus_bandlimited_velocity",
            "raw_minus_band_max_abs": "maximum_absolute_raw_minus_bandlimited_velocity",
            "raw_minus_band_energy": "raw_minus_bandlimited_residual_energy",
            "raw_minus_band_vti_abs": "absolute_raw_minus_bandlimited_vti",
            "raw_band_corr": "raw_bandlimited_velocity_correlation",
            "raw_band_vti_difference": "raw_minus_bandlimited_vti_difference",
        }

    @staticmethod
    def _rectified_qc_metric_names() -> dict[str, str]:
        return {
            "raw_minus_band_mean": (
                "mean_rectified_raw_minus_rectified_bandlimited_velocity"
            ),
            "raw_minus_band_bias": (
                "rectified_raw_minus_rectified_bandlimited_velocity_bias"
            ),
            "raw_minus_band_rms": (
                "rms_rectified_raw_minus_rectified_bandlimited_velocity"
            ),
            "raw_minus_band_mae": (
                "mean_absolute_rectified_raw_minus_rectified_bandlimited_velocity"
            ),
            "raw_minus_band_max_abs": (
                "maximum_absolute_rectified_raw_minus_rectified_bandlimited_velocity"
            ),
            "raw_minus_band_energy": (
                "rectified_raw_minus_rectified_bandlimited_residual_energy"
            ),
            "raw_minus_band_vti_abs": (
                "absolute_rectified_raw_minus_rectified_bandlimited_vti"
            ),
            "raw_band_corr": "rectified_raw_bandlimited_velocity_correlation",
            "raw_band_vti_difference": (
                "rectified_raw_minus_rectified_bandlimited_vti_difference"
            ),
        }

    @staticmethod
    def _expected_range_note(key: str) -> str:
        nonnegative = (
            "Expected mathematical range: >= 0 or NaN. This is not a "
            "physiological reference interval."
        )
        signed = (
            "Expected mathematical range: any finite signed value or NaN. "
            "This is not a physiological reference interval."
        )

        if key in {"raw_band_corr"}:
            return "Expected mathematical range: [-1, 1] or NaN."
        if key in {"beat_period"}:
            return "Expected mathematical range: > 0 seconds."
        if key.startswith("t_") or key in {"peak_to_trough_time"}:
            return (
                "Expected mathematical range: [0, beat_period) seconds or NaN; "
                "peak_to_trough_time is NaN when peak and trough coincide."
            )
        if key in {
            "dvdt_max",
            "dvdt_min",
            "raw_minus_band_mean",
            "raw_minus_band_bias",
            "raw_band_vti_difference",
        }:
            return signed
        return nonnegative

    def _rectified_qc_meta(self) -> dict:
        meta = {}
        output_names = self._rectified_qc_metric_names()
        for k in self._qc_metric_keys():
            meta[k[0]] = {
                "output_name": [output_names.get(k[0], k[0])],
                "source_metric_id": [k[0]],
                "definition": [
                    k[1].replace("raw", "rectified raw").replace(
                        "bandlimited", "rectified bandlimited"
                    )
                ],
                "unit": [k[2]],
                "latex_formula": [
                    k[3]
                    .replace(r"v_{\mathrm{raw}}", r"\max(v_{\mathrm{raw}},0)")
                    .replace(
                        r"v_{\mathrm{band}}",
                        r"\max(v_{\mathrm{band}},0)",
                    )
                ],
                "expected_range": [self._expected_range_note(k[0])],
                "range_basis": [
                    "Mathematical sanity range for rectified raw-vs-rectified "
                    "bandlimited agreement; not a clinical normal range."
                ],
                "metric_type": ["raw_vs_bandlimited_qc_rectified"],
                "qc_waveform_comparison": [
                    "Compares finite raw and bandlimited samples after clipping "
                    "negative values to zero; NaNs are excluded from paired comparisons."
                ],
            }
        return meta

    def _metric_meta(self) -> dict:
        meta = {}
        output_names = self._audit_metric_names()
        for k in self._scalar_metric_keys():
            meta[k[0]] = {
                "output_name": [output_names.get(k[0], k[0])],
                "source_metric_id": [k[0]],
                "definition": [k[1]],
                "unit": [k[2]],
                "latex_formula": [k[4]],
                "expected_range": [self._expected_range_note(k[0])],
                "range_basis": [
                    "Mathematical sanity range after rectifying finite velocities to >= 0; "
                    "not a clinical normal range."
                ],
                "metric_type": ["absolute_gain_sensitive"],
            }
        for k in self._array_metric_keys():
            meta[k[0]] = {
                "output_name": [output_names.get(k[0], k[0])],
                "source_metric_id": [k[0]],
                "definition": [k[1]],
                "unit": [k[2]],
                "latex_formula": [k[4]],
                "expected_range": [self._expected_range_note(k[0])],
                "range_basis": [
                    "Mathematical sanity range after rectifying finite velocities to >= 0; "
                    "not a clinical normal range."
                ],
                "metric_type": ["absolute_gain_sensitive"],
                "array_axis": [k[5]],
            }
        for k in self._qc_metric_keys():
            meta[k[0]] = {
                "output_name": [output_names.get(k[0], k[0])],
                "source_metric_id": [k[0]],
                "definition": [f"unrectified {k[1]}"],
                "unit": [k[2]],
                "latex_formula": [k[3]],
                "expected_range": [self._expected_range_note(k[0])],
                "range_basis": [
                    "Mathematical sanity range for unrectified raw-vs-bandlimited "
                    "agreement; not a clinical normal range."
                ],
                "metric_type": ["raw_vs_bandlimited_qc_unrectified"],
                "qc_waveform_comparison": [
                    "Compares original finite raw and bandlimited samples directly; "
                    "negative values are preserved and NaNs are excluded from paired "
                    "comparisons."
                ],
            }
        return meta

    def _pack_dict(
        self,
        metrics: dict,
        path_prefix: str,
        d: dict,
        attrs_common: dict | None = None,
        meta_overrides: dict | None = None,
    ) -> None:
        meta = self._metric_meta()
        attrs_common = attrs_common or {}
        meta_overrides = meta_overrides or {}

        for key, arr in d.items():
            attrs = {}
            attrs.update(meta.get(key, {}))
            attrs.update(meta_overrides.get(key, {}))
            attrs.update(attrs_common)
            output_name = attrs.get("output_name", [key])
            if isinstance(output_name, (list, tuple, np.ndarray)):
                output_name = output_name[0]
            metrics[f"{path_prefix}/{output_name}"] = with_attrs(arr, attrs)

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
            v_raw_seg, v_band_seg, T, rectify=False
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

        rectified_qc_meta = self._rectified_qc_meta()
        qc_seg, qc_br, qc_gl, _, _, qc_note = self._compute_qc_block_segment(
            v_raw_seg, v_band_seg, T, rectify=True
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/rectified_raw_vs_rectified_bandlimited_segment",
            qc_seg,
            {
                "definition_scope": [
                    "per-segment rectified raw-vs-rectified bandlimited QC metrics"
                ],
                "segment_indexing": [qc_note],
            },
            meta_overrides=rectified_qc_meta,
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/rectified_raw_vs_rectified_bandlimited_branch",
            qc_br,
            {"definition_scope": ["median over radii per branch"]},
            meta_overrides=rectified_qc_meta,
        )
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/by_segment/rectified_raw_vs_rectified_bandlimited_global",
            qc_gl,
            {"definition_scope": ["median over all branch-radius segment values per beat"]},
            meta_overrides=rectified_qc_meta,
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
        out_qc = self._compute_qc_block_global(v_raw_gl, v_band_gl, T, rectify=False)
        out_qc_rectified = self._compute_qc_block_global(
            v_raw_gl, v_band_gl, T, rectify=True
        )

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
        self._pack_dict(
            metrics,
            f"{vessel_prefix}/global/rectified_raw_vs_rectified_bandlimited",
            out_qc_rectified,
            {
                "definition_scope": [
                    "global rectified raw-vs-rectified bandlimited QC metrics"
                ]
            },
            meta_overrides=self._rectified_qc_meta(),
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
            ["t_upstroke_max", "time of maximum acceleration dv/dt", "seconds", None, r"$t_{\max(dv/dt)}$"],
            ["t_downstroke_max", "time of maximum deceleration (minimum dv/dt)", "seconds", None, r"$t_{\min(dv/dt)}$"],
            ["peak_to_trough_time", "circular forward time from peak to trough", "seconds", None, r"$(t_{\min}-t_{\max})\bmod T$"],
            ["beat_period", "beat period T", "seconds", None, r"$T$"],

            # Absolute derivative / kinetic metrics
            ["dvdt_max", "maximum acceleration dv/dt", "velocity/s", None, r"$\max_t\,dv/dt$"],
            ["dvdt_min", "minimum acceleration dv/dt", "velocity/s", None, r"$\min_t\,dv/dt$"],
            ["dvdt_fall_abs_max", "maximum deceleration magnitude |min_t dv/dt|", "velocity/s", None, r"$|\min_t\,dv/dt|$"],
            ["dvdt_rms", "RMS acceleration from dv/dt", "velocity/s", None, r"$\sqrt{T^{-1}\int_0^T(dv/dt)^2\,dt}$"],
            ["dvdt_abs_mean", "mean absolute acceleration |dv/dt|", "velocity/s", None, r"$T^{-1}\int_0^T|dv/dt|\,dt$"],
            ["dvdt_std", "standard deviation of acceleration dv/dt", "velocity/s", None, r"$\mathrm{std}_t(dv/dt)$"],
            ["dvdt_energy", "integral of squared acceleration", "velocity^2/s", None, r"$\int_0^T(dv/dt)^2\,dt$"],
            ["total_variation", "total velocity change accumulated from |dv/dt|", "velocity", None, r"$\int_0^T|dv/dt|\,dt$"],
            ["positive_variation", "velocity gain accumulated from positive acceleration", "velocity", None, r"$\int_0^T\max(dv/dt,0)\,dt$"],
            ["negative_variation", "velocity drop accumulated from deceleration magnitude", "velocity", None, r"$\int_0^T\max(-dv/dt,0)\,dt$"],

            # Absolute harmonic-amplitude scalar metrics
            ["dc_level", "DC Fourier coefficient V0", "velocity", None, r"$V_0$"],
            [
                "fundamental_amp",
                "one-sided amplitude of V1",
                "velocity",
                None,
                r"$a_1$",
            ],
            [
                "second_harmonic_amp",
                "one-sided amplitude of V2",
                "velocity",
                None,
                r"$a_2$",
            ],
            [
                "third_harmonic_amp",
                "one-sided amplitude of V3",
                "velocity",
                None,
                r"$a_3$",
            ],
            ["pulsatile_harmonic_power", "sum_{h=1}^H |Vh|^2", "velocity^2", None, r"$\sum_{h=1}^H |V_h|^2$"],
            ["higher_harmonic_power", "sum_{h=2}^H |Vh|^2", "velocity^2", None, r"$\sum_{h=2}^H |V_h|^2$"],
            ["low_harmonic_power", "sum_{h=1}^{H_LOW_MAX} |Vh|^2", "velocity^2", None, r"$\sum_{h=1}^{H_{\mathrm{low}}}|V_h|^2$"],
            [
                "bandlimited_ac_rms_from_harmonics",
                "sqrt(sum_{h=1}^H c_h |Vh|^2), "
                "with c_h=2 except c_h=1 for Nyquist",
                "velocity",
                None,
                r"$\sqrt{\sum_{h=1}^H c_h|V_h|^2}$",
            ],

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
                "one-sided amplitude for h=1..H_MAX; Nyquist is not doubled",
                "velocity",
                self.H_MAX,
                r"$a_h,\ h=1,\ldots,H_{\max}$",
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
