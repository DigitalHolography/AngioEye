import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterial_waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Waveform-shape metrics on per-beat, per-branch, per-radius velocity waveforms.
    Gain-invariant: all metrics are invariant to scaling of the waveform by a positive constant.
    """

    description = "Waveform shape metrics (segment + aggregates + global), gain-invariant and robust."

    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"

    v_raw_global_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )

    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    eps = 1e-12
    ratio_R_VTI = 0.5
    ratio_SF_VTI = 1.0 / 3.0

    ratio_vend_start = 0.75
    ratio_vend_end = 0.90

    H_LOW_MAX = 3
    H_HIGH_MIN = 4
    H_HIGH_MAX = 8

    H_MAX = 10
    H_PHASE_RESIDUAL = 10

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

        return v2

    @staticmethod
    def _wrap_pi(x: float) -> float:
        """Wrap angle to [-pi, pi]."""
        if not np.isfinite(x):
            return np.nan
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    def _late_window_indices(self, n: int) -> tuple[int, int]:
        """
        Return [k0:k1) corresponding to [ratio_vend_start*T, ratio_vend_end*T].
        """
        if n <= 0:
            return 0, 0

        a = float(self.ratio_vend_start)
        b = float(self.ratio_vend_end)

        if (not np.isfinite(a)) or (not np.isfinite(b)) or a < 0 or b <= a or b > 1:
            return 0, 0

        k0 = int(np.floor(a * n))
        k1 = int(np.ceil(b * n))

        k0 = max(0, min(n - 1, k0))
        k1 = max(k0 + 1, min(n, k1))
        return k0, k1

    def _quantile_time_over_T(self, v: np.ndarray, Tbeat: float, q: float) -> float:
        """
        v: rectified 1D waveform (NaNs allowed)
        Returns t_q / Tbeat where C(t_q) >= q, with C = cumsum(v)/sum(v).
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vv = np.where(np.isfinite(v), v, np.nan)
        m0 = float(np.nansum(vv))
        if m0 <= 0:
            return np.nan

        c = np.nancumsum(vv) / m0
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

        fs = n / Tbeat
        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        h = f * Tbeat

        E_total = float(np.sum(P))
        if not np.isfinite(E_total) or E_total <= 0:
            return np.nan, np.nan

        low_mask = (h >= 0.9) & (h <= float(self.H_LOW_MAX) + 0.1)
        high_mask = (h >= float(self.H_HIGH_MIN) - 0.1) & (
            h <= float(self.H_HIGH_MAX) + 0.1
        )

        E_low = float(np.sum(P[low_mask]))
        E_high = float(np.sum(P[high_mask]))

        return float(E_low / E_total), float(E_high / E_total)

    def _harmonic_pack(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Compute complex harmonic coefficients Vn for n=0..H, with H=min(H_MAX, n_rfft-1),
        and synthesize band-limited waveform vb(t) using harmonics 0..H.

        Returns:
          - V (complex array length H+1)
          - H (int)
          - vb (float array length n)
          - Vmax (float) = max_t vb(t)
          - omega0 (float) = 2*pi/Tbeat
          - Vfull (full rfft/n coefficient array)
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {
                "V": None,
                "H": 0,
                "vb": None,
                "Vmax": np.nan,
                "omega0": np.nan,
                "Vfull": None,
            }

        if v.size == 0 or not np.any(np.isfinite(v)):
            return {
                "V": None,
                "H": 0,
                "vb": None,
                "Vmax": np.nan,
                "omega0": np.nan,
                "Vfull": None,
            }

        vv = np.where(np.isfinite(v), v, 0.0)
        n = vv.size
        if n < 2:
            return {
                "V": None,
                "H": 0,
                "vb": None,
                "Vmax": np.nan,
                "omega0": np.nan,
                "Vfull": None,
            }

        Vfull = np.fft.rfft(vv) / float(n)
        H = int(min(self.H_MAX, Vfull.size - 1))
        V = Vfull[: H + 1].copy()

        Vtrunc = np.zeros_like(Vfull)
        Vtrunc[: H + 1] = V
        vb = np.fft.irfft(Vtrunc * float(n), n=n)

        Vmax = float(np.nanmax(vb)) if vb.size else np.nan
        omega0 = float(2.0 * np.pi / Tbeat)

        return {
            "V": V,
            "H": H,
            "vb": vb,
            "Vmax": Vmax,
            "omega0": omega0,
            "Vfull": Vfull,
        }

    def _tauH_from_harmonics(self, V: np.ndarray, Vmax: float, omega0: float) -> float:
        """
        Magnitude-only damping proxy tauH:
          Xn = Vn / Vmax
          tau_|H|,n = (1/omega_n) * sqrt(1/|Xn|^2 - 1)  for |Xn| in (0,1]
          tauH = sum_{n=1..H} |Vn| * tau_n / sum_{n=1..H} |Vn|
        """
        if (
            V is None
            or (not np.isfinite(Vmax))
            or Vmax <= 0
            or (not np.isfinite(omega0))
            or omega0 <= 0
        ):
            return np.nan

        H = int(V.size - 1)
        if H < 1:
            return np.nan

        weights = []
        taus = []

        for n in range(1, H + 1):
            Vn = V[n]
            an = float(np.abs(Vn))
            if not np.isfinite(an) or an <= 0:
                continue

            Xn = an / Vmax
            if (not np.isfinite(Xn)) or Xn <= 0:
                continue

            if Xn > 1.0 + 1e-9:
                continue

            omega_n = float(n) * omega0
            inside = (1.0 / (Xn * Xn + self.eps)) - 1.0
            if inside <= 0:
                continue

            tau_n = (1.0 / omega_n) * float(np.sqrt(inside))
            if np.isfinite(tau_n) and tau_n > 0:
                weights.append(an)
                taus.append(tau_n)

        if len(weights) == 0:
            return np.nan

        w = np.asarray(weights, dtype=float)
        t = np.asarray(taus, dtype=float)
        return float(np.sum(w * t) / (np.sum(w) + self.eps))

    def _spectral_entropy_from_harmonics(self, V: np.ndarray) -> float:
        """
        Spectral entropy of harmonic magnitude distribution over n=1..H:
          p_n = |Vn| / sum_{k=1..H} |Vk|
          Hspec = - sum p_n log(p_n)
        """
        if V is None:
            return np.nan
        H = int(V.size - 1)
        if H < 1:
            return np.nan

        mags = np.abs(V[1:])
        mags = np.where(np.isfinite(mags), mags, np.nan)
        s = float(np.nansum(mags))
        if s <= 0:
            return np.nan

        p = mags / s
        p = np.clip(p, self.eps, 1.0)
        return float(-np.nansum(p * np.log(p)))

    def _crest_factor(self, v: np.ndarray) -> float:
        """
        Crest factor on the current waveform representation:
          CF = max(v) / rms(v)
        """
        if v is None or v.size == 0:
            return np.nan
        v = np.asarray(v, dtype=float)
        if not np.any(np.isfinite(v)):
            return np.nan
        x = np.where(np.isfinite(v), v, np.nan)
        rms = float(np.sqrt(self._safe_nanmean(x * x)))
        if rms <= 0:
            return np.nan
        return float(np.nanmax(x) / rms)

    def _harmonic_phases(self, V: np.ndarray) -> dict:
        """
        Return delta_phi2 only.
        delta_phi2 = wrap(phi2 - 2*phi1)
        """
        out = {
            "delta_phi2": np.nan,
        }
        if V is None:
            return out

        H = int(V.size - 1)
        if H < 2:
            return out

        def phase_if_strong(Vn: complex) -> float:
            if not np.isfinite(Vn.real) or not np.isfinite(Vn.imag):
                return np.nan
            if np.abs(Vn) <= self.eps:
                return np.nan
            return self._wrap_pi(float(np.angle(Vn)))

        phi1 = phase_if_strong(V[1])
        phi2 = phase_if_strong(V[2])

        if np.isfinite(phi1) and np.isfinite(phi2):
            out["delta_phi2"] = self._wrap_pi(phi2 - 2.0 * phi1)

        return out

    def _spectral_centroid_spread(self, V: np.ndarray) -> tuple[float, float]:
        """
        mu_h = sum n w_n, sigma_h = sqrt(sum (n-mu_h)^2 w_n), with w_n ∝ |V_n| for n=1..H.
        """
        if V is None:
            return np.nan, np.nan
        H = int(V.size - 1)
        if H < 1:
            return np.nan, np.nan

        n_idx = np.arange(1, H + 1, dtype=float)
        mags = np.abs(V[1:])
        mags = np.where(np.isfinite(mags), mags, np.nan)
        s = float(np.nansum(mags))
        if not np.isfinite(s) or s <= 0:
            return np.nan, np.nan

        w = mags / s
        mu_h = float(np.nansum(n_idx * w))
        sigma_h = float(np.sqrt(np.nansum(((n_idx - mu_h) ** 2) * w)))
        return mu_h, sigma_h

    def _n_eff_over_T(self, v: np.ndarray, Tbeat: float, m0: float) -> float:
        """
        Normalized effective support duration:
          p(t) = v(t)/M0,  N_eff = 1 / ∫ p(t)^2 dt,  returns N_eff/T
        """
        if (
            v.size == 0
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan
        if (not np.isfinite(m0)) or m0 <= 0:
            return np.nan

        dt = Tbeat / v.size
        p = np.where(np.isfinite(v), v, 0.0) / (m0 + self.eps)

        int_p2 = float(np.sum(p * p) * dt)
        if not np.isfinite(int_p2) or int_p2 <= 0:
            return np.nan

        n_eff = 1.0 / int_p2
        return float(n_eff / Tbeat)

    def _phase_locking_residual(self, V: np.ndarray) -> float:
        """
        E_phi = weighted mean squared wrapped residual of phi_n - n phi_1 over n=2..H_PHASE_RESIDUAL,
        using weights |V_n|.
        """
        if V is None:
            return np.nan

        H = int(V.size - 1)
        Huse = int(min(H, self.H_PHASE_RESIDUAL))
        if Huse < 2:
            return np.nan

        if np.abs(V[1]) <= self.eps:
            return np.nan
        phi1 = self._wrap_pi(float(np.angle(V[1])))

        weights = []
        residuals2 = []
        for n in range(2, Huse + 1):
            if np.abs(V[n]) <= self.eps:
                continue
            phin = self._wrap_pi(float(np.angle(V[n])))
            dphi = self._wrap_pi(phin - n * phi1)
            w = float(np.abs(V[n]))
            if np.isfinite(dphi) and np.isfinite(w) and w > 0:
                weights.append(w)
                residuals2.append(dphi * dphi)

        if len(weights) == 0:
            return np.nan

        w = np.asarray(weights, dtype=float)
        r2 = np.asarray(residuals2, dtype=float)
        return float(np.sum(w * r2) / (np.sum(w) + self.eps))

    def _reconstruction_error_from_vb(self, v: np.ndarray, vb: np.ndarray) -> float:
        """
        Reconstruction error from the existing harmonic truncation used in _harmonic_pack,
        i.e. from DC + first H_MAX harmonics (or fewer if limited by sampling).
        """
        if v is None or vb is None:
            return np.nan
        if v.size < 2 or vb.size != v.size:
            return np.nan
        if not np.any(np.isfinite(v)) or not np.any(np.isfinite(vb)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        vbb = np.where(np.isfinite(vb), vb, 0.0)

        num = float(np.sum((vv - vbb) ** 2))
        den = float(np.sum(vv**2))
        if (not np.isfinite(den)) or den <= 0:
            return np.nan
        return float(num / (den + self.eps))

    def _peak_trough_times(self, v: np.ndarray) -> tuple[float, float, int, int]:
        """
        Returns:
          t_max_over_T, t_min_over_T, idx_peak, idx_min
        """
        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan, np.nan, -1, -1

        idx_peak = int(np.nanargmax(v))
        idx_min = int(np.nanargmin(v))

        return float(idx_peak / v.size), float(idx_min / v.size), idx_peak, idx_min

    def _normalized_slopes_and_times(
        self, v: np.ndarray, Tbeat: float
    ) -> tuple[float, float, float, float]:
        """
        Returns:
          S_rise, S_fall, t_up_over_T, t_down_over_T
        """
        if (
            v.size < 2
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan, np.nan, np.nan, np.nan

        meanv = self._safe_nanmean(v)
        if (not np.isfinite(meanv)) or meanv <= 0:
            return np.nan, np.nan, np.nan, np.nan

        dt = Tbeat / v.size
        dvdt = np.gradient(np.where(np.isfinite(v), v, 0.0), dt)
        if not np.any(np.isfinite(dvdt)):
            return np.nan, np.nan, np.nan, np.nan

        idx_up = int(np.nanargmax(dvdt))
        idx_down = int(np.nanargmin(dvdt))

        s_up = float(np.nanmax(dvdt))
        s_down = float(np.nanmin(dvdt))

        return (
            float(Tbeat * s_up / (meanv + self.eps)),
            float(Tbeat * np.abs(s_down) / (meanv + self.eps)),
            float(idx_up / v.size),
            float(idx_down / v.size),
        )

    def _peak_to_trough_interval(self, idx_peak: int, idx_min: int, n: int) -> float:
        """
        Delta_t_over_T = (t_min - t_max)/T, assuming min occurs after peak in the beat.
        Returns NaN if ordering is inconsistent.
        """
        if n <= 0 or idx_peak < 0 or idx_min < 0:
            return np.nan
        if idx_min < idx_peak:
            return np.nan
        return float((idx_min - idx_peak) / n)

    def _normalized_decay_slope(
        self,
        vmin: float,
        vmax: float,
        meanv: float,
        delta_t_over_T: float,
    ) -> float:
        """
        S_decay = ((v_min - v_max) T) / (Delta_t * v_mean)
                = (v_min - v_max) / ( (Delta_t/T) * v_mean )
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.nan
        if (not np.isfinite(meanv)) or meanv <= 0:
            return np.nan
        if (not np.isfinite(delta_t_over_T)) or delta_t_over_T <= 0:
            return np.nan

        return float((vmin - vmax) / ((delta_t_over_T + self.eps) * (meanv + self.eps)))

    def _r_sd(self, v: np.ndarray) -> float:
        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vmax = float(np.nanmax(v))
        if (not np.isfinite(vmax)) or vmax <= 0:
            return np.nan

        k0, k1 = self._late_window_indices(v.size)
        if k1 <= k0:
            return np.nan

        tail = np.asarray(v[k0:k1], dtype=float)
        vend = self._safe_nanmean(tail)
        if (not np.isfinite(vend)) or vend < 0:
            return np.nan

        return float(vmax / (vend + self.eps))

    def _late_cycle_mean_fraction(self, v: np.ndarray) -> float:
        """
        v_end_over_v_mean where v_end is the mean over [ratio_vend_start*T, ratio_vend_end*T].
        """
        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        meanv = self._safe_nanmean(v)
        if (not np.isfinite(meanv)) or meanv <= 0:
            return np.nan

        k0, k1 = self._late_window_indices(v.size)
        if k1 <= k0:
            return np.nan

        tail = np.asarray(v[k0:k1], dtype=float)
        vend = self._safe_nanmean(tail)
        if (not np.isfinite(vend)) or vend < 0:
            return np.nan

        return float(vend / (meanv + self.eps))

    def _delta_dti(
        self, v: np.ndarray, Tbeat: float, m0: float, t: np.ndarray
    ) -> float:
        if (
            v.size == 0
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan
        if (not np.isfinite(m0)) or m0 <= 0:
            return np.nan
        d = np.nancumsum(np.where(np.isfinite(v), v, 0.0))
        d_star = d / (m0 + self.eps)
        d0_star = t / Tbeat
        return float(np.nansum(d_star - d0_star))

    def _normalized_cumulative_displacement_samples(
        self, v: np.ndarray, Tbeat: float, m0: float
    ) -> dict:
        """
        Returns normalized cumulative displacement d_q evaluated at fixed phase q:
          d_q = D(qT) / D(T), for q in {0.10, 0.25, 0.50, 0.75, 0.90}
        """
        out = {
            "d10": np.nan,
            "d25": np.nan,
            "d50": np.nan,
            "d75": np.nan,
            "d90": np.nan,
        }

        if (
            v.size == 0
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return out
        if (not np.isfinite(m0)) or m0 <= 0:
            return out

        D = np.nancumsum(np.where(np.isfinite(v), v, 0.0)) * (Tbeat / v.size)
        d = D / (m0 + self.eps)

        def sample_at_ratio(r: float) -> float:
            idx = int(np.floor(r * (v.size - 1)))
            idx = max(0, min(v.size - 1, idx))
            return float(d[idx])

        out["d10"] = sample_at_ratio(0.10)
        out["d25"] = sample_at_ratio(0.25)
        out["d50"] = sample_at_ratio(0.50)
        out["d75"] = sample_at_ratio(0.75)
        out["d90"] = sample_at_ratio(0.90)
        return out

    def _d_quantile_shape_metrics(self, d_samples: dict) -> tuple[float, float, float]:
        """
        From d10,d25,d50,d75,d90 define:
          Q_d_width = d75 - d25
          Q_d_skew  = ((d90-d50) - (d50-d10)) / (d90-d10 + eps)
          R_Q_d     = Q_d_skew / (Q_d_width + eps)
        """
        d10 = d_samples["d10"]
        d25 = d_samples["d25"]
        d50 = d_samples["d50"]
        d75 = d_samples["d75"]
        d90 = d_samples["d90"]

        Q_d_width = np.nan
        if np.isfinite(d25) and np.isfinite(d75):
            Q_d_width = float(d75 - d25)

        Q_d_skew = np.nan
        if np.isfinite(d10) and np.isfinite(d50) and np.isfinite(d90):
            Q_d_skew = float(((d90 - d50) - (d50 - d10)) / ((d90 - d10) + self.eps))

        R_Q_d = np.nan
        if np.isfinite(Q_d_skew) and np.isfinite(Q_d_width):
            R_Q_d = float(Q_d_skew / (Q_d_width + self.eps))

        return Q_d_width, Q_d_skew, R_Q_d

    def _gamma_t(
        self,
        v: np.ndarray,
        Tbeat: float,
        mu_t: float,
        sigma_t: float,
        m0: float,
        t: np.ndarray,
    ) -> float:
        if (
            v.size == 0
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan
        if (
            (not np.isfinite(mu_t))
            or (not np.isfinite(sigma_t))
            or sigma_t <= 0
            or (not np.isfinite(m0))
            or m0 <= 0
        ):
            return np.nan
        z = (t - mu_t) / (sigma_t + self.eps)
        return float(
            np.nansum(np.where(np.isfinite(v), v, 0.0) * (z**3)) / (m0 + self.eps)
        )

    def _derivative_energies(
        self, v: np.ndarray, Tbeat: float, m0: float
    ) -> tuple[float, float]:
        """
        E_slope = T^2 / M0^2 * int (dv/dt)^2 dt
        E_curv  = T^4 / M0^2 * int (d2v/dt2)^2 dt
        """
        if (
            v.size < 3
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan, np.nan
        if (not np.isfinite(m0)) or m0 <= 0:
            return np.nan, np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        dt = Tbeat / v.size

        dvdt = np.gradient(vv, dt)
        d2vdt2 = np.gradient(dvdt, dt)

        E_slope = float((Tbeat**2) * np.sum(dvdt**2) * dt / ((m0 + self.eps) ** 2))
        E_curv = float((Tbeat**4) * np.sum(d2vdt2**2) * dt / ((m0 + self.eps) ** 2))

        return E_slope, E_curv

    def _compute_metrics_1d(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Canonical metric kernel: compute all waveform-shape metrics from a single 1D waveform v(t).
        Returns a dict of scalar metrics (floats).
        """
        v = self._rectify_keep_nan(v)
        n = int(v.size)
        if n <= 0:
            return {k[0]: np.nan for k in self._metric_keys()}

        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {k[0]: np.nan for k in self._metric_keys()}

        vv = np.where(np.isfinite(v), v, np.nan)
        m0 = float(np.nansum(vv))
        if m0 <= 0:
            return {k[0]: np.nan for k in self._metric_keys()}

        dt = Tbeat / n
        t = np.arange(n, dtype=float) * dt

        m1 = float(np.nansum(vv * t))
        mu_t = m1 / m0
        mu_t_over_T = mu_t / Tbeat

        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(vv))
        meanv = float(self._safe_nanmean(vv))

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

        k_R_VTI = int(np.ceil(n * self.ratio_R_VTI))
        k_R_VTI = max(0, min(n, k_R_VTI))
        D1_R_VTI = float(np.nansum(vv[:k_R_VTI])) if k_R_VTI > 0 else np.nan
        D2_R_VTI = float(np.nansum(vv[k_R_VTI:])) if k_R_VTI < n else np.nan
        R_VTI = D1_R_VTI / (D2_R_VTI + self.eps)

        k_sf = int(np.ceil(n * self.ratio_SF_VTI))
        k_sf = max(0, min(n, k_sf))
        D1_sf = float(np.nansum(vv[:k_sf])) if k_sf > 0 else np.nan
        D2_sf = float(np.nansum(vv[k_sf:])) if k_sf < n else np.nan
        SF_VTI = D1_sf / (D1_sf + D2_sf + self.eps)

        dtau = t - mu_t
        m2 = float(np.nansum(vv * (dtau**2)))
        sigma_t = np.sqrt(m2 / m0 + self.eps)
        sigma_t_over_T = sigma_t / Tbeat

        t10_over_T = self._quantile_time_over_T(vv, Tbeat, 0.10)
        t25_over_T = self._quantile_time_over_T(vv, Tbeat, 0.25)
        t50_over_T = self._quantile_time_over_T(vv, Tbeat, 0.50)
        t75_over_T = self._quantile_time_over_T(vv, Tbeat, 0.75)
        t90_over_T = self._quantile_time_over_T(vv, Tbeat, 0.90)

        d_samples = self._normalized_cumulative_displacement_samples(vv, Tbeat, m0)
        d10 = d_samples["d10"]
        d25 = d_samples["d25"]
        d50 = d_samples["d50"]
        d75 = d_samples["d75"]
        d90 = d_samples["d90"]

        E_low_over_E_total, E_high_over_E_total = self._spectral_ratios(vv, Tbeat)

        hp = self._harmonic_pack(vv, Tbeat)
        V = hp["V"]
        vb = hp["vb"]
        Vmax_bl = hp["Vmax"]
        omega0 = hp["omega0"]

        tauH = self._tauH_from_harmonics(V, Vmax_bl, omega0)
        crest_factor = self._crest_factor(vv)
        spectral_entropy = self._spectral_entropy_from_harmonics(V)
        ph = self._harmonic_phases(V)

        mu_h, sigma_h = self._spectral_centroid_spread(V)
        N_eff_over_T = self._n_eff_over_T(vv, Tbeat, m0)

        t_max_over_T, t_min_over_T, idx_peak, idx_min = self._peak_trough_times(vv)
        (
            slope_rise_normalized,
            slope_fall_normalized,
            t_up_over_T,
            t_down_over_T,
        ) = self._normalized_slopes_and_times(vv, Tbeat)

        Delta_t_over_T = self._peak_to_trough_interval(idx_peak, idx_min, n)
        S_decay = self._normalized_decay_slope(vmin, vmax, meanv, Delta_t_over_T)

        R_SD = self._r_sd(vv)
        Delta_DTI = self._delta_dti(vv, Tbeat, m0, t)
        gamma_t = self._gamma_t(vv, Tbeat, mu_t, sigma_t, m0, t)

        phase_locking_residual = self._phase_locking_residual(V)
        E_recon_H_MAX = self._reconstruction_error_from_vb(vv, vb)

        Q_t_skew = np.nan
        if (
            np.isfinite(t10_over_T)
            and np.isfinite(t50_over_T)
            and np.isfinite(t90_over_T)
        ):
            denom = (t90_over_T - t10_over_T) + self.eps
            Q_t_skew = float(
                ((t90_over_T - t50_over_T) - (t50_over_T - t10_over_T)) / denom
            )

        Q_t_width = np.nan
        if np.isfinite(t25_over_T) and np.isfinite(t75_over_T):
            Q_t_width = float(t75_over_T - t25_over_T)

        R_Q_t = np.nan
        if np.isfinite(Q_t_skew) and np.isfinite(Q_t_width):
            R_Q_t = float(Q_t_skew / (Q_t_width + self.eps))

        Q_d_width, Q_d_skew, R_Q_d = self._d_quantile_shape_metrics(d_samples)

        v_end_over_v_mean = self._late_cycle_mean_fraction(vv)
        E_slope, E_curv = self._derivative_energies(vv, Tbeat, m0)

        return {
            "mu_t": float(mu_t),
            "mu_t_over_T": float(mu_t_over_T),
            "RI": float(RI) if np.isfinite(RI) else np.nan,
            "PI": float(PI) if np.isfinite(PI) else np.nan,
            "R_VTI": float(R_VTI),
            "SF_VTI": float(SF_VTI),
            "sigma_t_over_T": float(sigma_t_over_T),
            "sigma_t": float(sigma_t),
            "t10_over_T": float(t10_over_T),
            "t25_over_T": float(t25_over_T),
            "t50_over_T": float(t50_over_T),
            "t75_over_T": float(t75_over_T),
            "t90_over_T": float(t90_over_T),
            "d10": float(d10) if np.isfinite(d10) else np.nan,
            "d25": float(d25) if np.isfinite(d25) else np.nan,
            "d50": float(d50) if np.isfinite(d50) else np.nan,
            "d75": float(d75) if np.isfinite(d75) else np.nan,
            "d90": float(d90) if np.isfinite(d90) else np.nan,
            "E_low_over_E_total": float(E_low_over_E_total),
            "E_high_over_E_total": float(E_high_over_E_total),
            "t_max_over_T": float(t_max_over_T)
            if np.isfinite(t_max_over_T)
            else np.nan,
            "t_min_over_T": float(t_min_over_T)
            if np.isfinite(t_min_over_T)
            else np.nan,
            "Delta_t_over_T": float(Delta_t_over_T)
            if np.isfinite(Delta_t_over_T)
            else np.nan,
            "slope_rise_normalized": float(slope_rise_normalized)
            if np.isfinite(slope_rise_normalized)
            else np.nan,
            "slope_fall_normalized": float(slope_fall_normalized)
            if np.isfinite(slope_fall_normalized)
            else np.nan,
            "t_up_over_T": float(t_up_over_T) if np.isfinite(t_up_over_T) else np.nan,
            "t_down_over_T": float(t_down_over_T)
            if np.isfinite(t_down_over_T)
            else np.nan,
            "S_decay": float(S_decay) if np.isfinite(S_decay) else np.nan,
            "R_SD": float(R_SD) if np.isfinite(R_SD) else np.nan,
            "Delta_DTI": float(Delta_DTI) if np.isfinite(Delta_DTI) else np.nan,
            "gamma_t": float(gamma_t) if np.isfinite(gamma_t) else np.nan,
            "tauH": float(tauH) if np.isfinite(tauH) else np.nan,
            "crest_factor": float(crest_factor)
            if np.isfinite(crest_factor)
            else np.nan,
            "spectral_entropy": float(spectral_entropy)
            if np.isfinite(spectral_entropy)
            else np.nan,
            "delta_phi2": float(ph["delta_phi2"])
            if np.isfinite(ph["delta_phi2"])
            else np.nan,
            "mu_h": float(mu_h) if np.isfinite(mu_h) else np.nan,
            "sigma_h": float(sigma_h) if np.isfinite(sigma_h) else np.nan,
            "N_eff_over_T": float(N_eff_over_T)
            if np.isfinite(N_eff_over_T)
            else np.nan,
            "phase_locking_residual": float(phase_locking_residual)
            if np.isfinite(phase_locking_residual)
            else np.nan,
            "E_recon_H_MAX": float(E_recon_H_MAX)
            if np.isfinite(E_recon_H_MAX)
            else np.nan,
            "Q_t_skew": float(Q_t_skew) if np.isfinite(Q_t_skew) else np.nan,
            "Q_t_width": float(Q_t_width) if np.isfinite(Q_t_width) else np.nan,
            "R_Q_t": float(R_Q_t) if np.isfinite(R_Q_t) else np.nan,
            "Q_d_skew": float(Q_d_skew) if np.isfinite(Q_d_skew) else np.nan,
            "Q_d_width": float(Q_d_width) if np.isfinite(Q_d_width) else np.nan,
            "R_Q_d": float(R_Q_d) if np.isfinite(R_Q_d) else np.nan,
            "v_end_over_v_mean": float(v_end_over_v_mean)
            if np.isfinite(v_end_over_v_mean)
            else np.nan,
            "E_slope": float(E_slope) if np.isfinite(E_slope) else np.nan,
            "E_curv": float(E_curv) if np.isfinite(E_curv) else np.nan,
        }

    @staticmethod
    def _metric_keys() -> list[list]:
        return [
            ["mu_t", "sum(w(t)*t)/sum(w(t))", "seconds"],
            ["mu_t_over_T", "mu/T", ""],
            ["RI", "(V_systole-V_diastole)/V_systole", ""],
            ["PI", "(V_systole-V_diastole)/V_mean", ""],
            ["R_VTI", "VTI_0_T2/(VTI_T2_T+eps)", ""],
            ["SF_VTI", "VTI_0_T3/VTI_0_T", ""],
            ["sigma_t_over_T", "sigma/T", ""],
            ["sigma_t", "sqrt(tau_M2-tau_M1**2)", "seconds"],
            ["t10_over_T", "t10/T", ""],
            ["t25_over_T", "t25/T", ""],
            ["t50_over_T", "t50/T", ""],
            ["t75_over_T", "t75/T", ""],
            ["t90_over_T", "t90/T", ""],
            ["d10", "D(0.1T)/D(T)", ""],
            ["d25", "D(0.25T)/D(T)", ""],
            ["d50", "D(0.5T)/D(T)", ""],
            ["d75", "D(0.75T)/D(T)", ""],
            ["d90", "D(0.9T)/D(T)", ""],
            ["E_low_over_E_total", "sum(|Vn|**2,n<=k)/sum(|Vn|**2)", ""],
            ["E_high_over_E_total", "sum(|Vn|**2,n>k)/sum(|Vn|**2)", ""],
            ["t_max_over_T", "t_max/T", ""],
            ["t_min_over_T", "t_min/T", ""],
            ["Delta_t_over_T", "(t_min-t_max)/T", ""],
            ["slope_rise_normalized", "T*max(dv/dt)/V_mean", ""],
            ["slope_fall_normalized", "T*|min(dv/dt)|/V_mean", ""],
            ["t_up_over_T", "t_up/T", ""],
            ["t_down_over_T", "t_down/T", ""],
            ["S_decay", "((V_min-V_max)T)/(Delta_t*V_mean)", ""],
            [
                "R_SD",
                "V_max/(mean(V(t in [ratio_vend_start*T,ratio_vend_end*T]))+eps)",
                "",
            ],
            ["Delta_DTI", "int_0^1(d*(tau)-d0*(tau))dtau", ""],
            ["gamma_t", "sum(w(t)*((t-mu)/sigma)^3)/sum(w(t))", ""],
            ["tauH", "sum(wn*(1/omega_n)*sqrt(1/|Xn|**2-1))/sum(wn)", "seconds"],
            ["crest_factor", "V_max/V_RMS", ""],
            ["spectral_entropy", "-sum(pn*log(pn+eps))", ""],
            ["delta_phi2", "wrap(phi2-2*phi1)", "rad"],
            ["mu_h", "sum_n n*|Vn|/sum_n |Vn|", ""],
            ["sigma_h", "sqrt(sum_n (n-mu_h)^2*|Vn|/sum_n |Vn|)", ""],
            ["N_eff_over_T", "N_eff/T", ""],
            [
                "phase_locking_residual",
                "sum_n w_n*wrap(phi_n-n*phi_1)^2/sum_n w_n",
                "rad^2",
            ],
            ["E_recon_H_MAX", "int(v-v_0:HMAX)^2 dt / int(v^2) dt", ""],
            ["Q_t_skew", "((t90-t50)-(t50-t10))/(t90-t10+eps)", ""],
            ["Q_t_width", "(t75-t25)/T", ""],
            ["R_Q_t", "Q_t_skew/(Q_t_width+eps)", ""],
            ["Q_d_skew", "((d90-d50)-(d50-d10))/(d90-d10+eps)", ""],
            ["Q_d_width", "d75-d25", ""],
            ["R_Q_d", "Q_d_skew/(Q_d_width+eps)", ""],
            [
                "v_end_over_v_mean",
                "mean(v[t in ratio_vend_start*T:ratio_vend_end*T])/mean(v)",
                "",
            ],
            ["E_slope", "T^2/M0^2 * int (dv/dt)^2 dt", ""],
            ["E_curv", "T^4/M0^2 * int (d2v/dt2)^2 dt", ""],
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

        seg = {
            k[0]: np.full((n_beats, n_segments), np.nan, dtype=float)
            for k in self._metric_keys()
        }
        br = {
            k[0]: np.full((n_beats, n_branches), np.nan, dtype=float)
            for k in self._metric_keys()
        }
        gl = {
            k[0]: np.full((n_beats,), np.nan, dtype=float) for k in self._metric_keys()
        }

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])

            gl_vals = {k[0]: [] for k in self._metric_keys()}

            for branch_idx in range(n_branches):
                br_vals = {k[0]: [] for k in self._metric_keys()}

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    m = self._compute_metrics_1d(v, Tbeat)

                    seg_idx = branch_idx * n_radii + radius_idx
                    for k in self._metric_keys():
                        seg[k[0]][beat_idx, seg_idx] = m[k[0]]
                        br_vals[k[0]].append(m[k[0]])
                        gl_vals[k[0]].append(m[k[0]])

                for k in self._metric_keys():
                    br[k[0]][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[k[0]], dtype=float)
                    )

            for k in self._metric_keys():
                gl[k[0]][beat_idx] = self._safe_nanmean(
                    np.asarray(gl_vals[k[0]], dtype=float)
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

        out = {
            k[0]: np.full((n_beats,), np.nan, dtype=float) for k in self._metric_keys()
        }

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])
            v = v_global[:, beat_idx]
            m = self._compute_metrics_1d(v, Tbeat)
            for k in self._metric_keys():
                out[k[0]][beat_idx] = m[k[0]]

        return out

    def run(self, h5file) -> ProcessResult:
        latex_formulas = {
            "mu_t": r"$\mu_t=\frac{\sum_t v(t)\,t}{\sum_t v(t)}$",
            "mu_t_over_T": r"$\frac{\mu_t}{T}$",
            "RI": r"$\frac{V_{systole}-V_{diastole}}{V_{systole}}$",
            "PI": r"$\frac{V_{systole}-V_{diastole}}{V_{mean}}$",
            "R_VTI": r"$\frac{VTI_{0\rightarrow T/2}}{VTI_{T/2\rightarrow T}}$",
            "SF_VTI": r"$\frac{VTI_{0\rightarrow T/3}}{VTI_{0\rightarrow T}}$",
            "sigma_t": r"$\sigma_t=\sqrt{\frac{\sum_t v(t)(t-\mu_t)^2}{\sum_t v(t)}}$",
            "sigma_t_over_T": r"$\frac{\sigma_t}{T}$",
            "t10_over_T": r"$\frac{t_{10}}{T}$",
            "t25_over_T": r"$\frac{t_{25}}{T}$",
            "t50_over_T": r"$\frac{t_{50}}{T}$",
            "t75_over_T": r"$\frac{t_{75}}{T}$",
            "t90_over_T": r"$\frac{t_{90}}{T}$",
            "d10": r"$\frac{D(0.1T)}{D(T)}$",
            "d25": r"$\frac{D(0.25T)}{D(T)}$",
            "d50": r"$\frac{D(0.5T)}{D(T)}$",
            "d75": r"$\frac{D(0.75T)}{D(T)}$",
            "d90": r"$\frac{D(0.9T)}{D(T)}$",
            "E_low_over_E_total": r"$\frac{\sum_{n=1}^{k}|V_n|^2}{\sum_{n\geq 0}|V_n|^2}$",
            "E_high_over_E_total": r"$\frac{\sum_{n=k+1}^{N}|V_n|^2}{\sum_{n\geq 0}|V_n|^2}$",
            "t_max_over_T": r"$\frac{t_{\max}}{T}$",
            "t_min_over_T": r"$\frac{t_{\min}}{T}$",
            "Delta_t_over_T": r"$\frac{t_{\min}-t_{\max}}{T}$",
            "slope_rise_normalized": r"$\frac{T}{\bar v}\max_t \frac{dv}{dt}$",
            "slope_fall_normalized": r"$\frac{T}{\bar v}\left|\min_t \frac{dv}{dt}\right|$",
            "t_up_over_T": r"$\frac{t_{\mathrm{up}}}{T}$",
            "t_down_over_T": r"$\frac{t_{\mathrm{down}}}{T}$",
            "S_decay": r"$\frac{(v_{\min}-v_{\max})\,T}{\Delta t\,\bar v}$",
            "R_SD": r"$\frac{v_{\max}}{\mathrm{mean}_{t\in[\alpha T,\beta T]} v(t)+\epsilon}$",
            "Delta_DTI": r"$\int_0^1 \left[d^*(\tau)-d_0^*(\tau)\right]d\tau$",
            "gamma_t": r"$\frac{1}{M_0}\sum_t v(t)\left(\frac{t-\mu_t}{\sigma_t}\right)^3$",
            "tauH": r"$\tau_H=\frac{\sum_n |V_n|\frac{1}{\omega_n}\sqrt{\frac{1}{|X_n|^2}-1}}{\sum_n |V_n|}$",
            "crest_factor": r"$\frac{v_{\max}}{v_{\mathrm{RMS}}}$",
            "spectral_entropy": r"$-\sum_{n=1}^{N} p_n\log(p_n+\epsilon)$",
            "delta_phi2": r"$\mathrm{wrap}(\phi_2-2\phi_1)$",
            "mu_h": r"$\mu_h=\sum_{n=1}^{N} n\,w_n,\quad w_n=\frac{|V_n|}{\sum_{k=1}^{N}|V_k|}$",
            "sigma_h": r"$\sigma_h=\sqrt{\sum_{n=1}^{N}(n-\mu_h)^2 w_n}$",
            "N_eff_over_T": r"$\frac{N_{\mathrm{eff}}}{T}$",
            "phase_locking_residual": r"$E_{\phi}=\frac{\sum_{n=2}^{N} w_n\,\mathrm{wrap}(\phi_n-n\phi_1)^2}{\sum_{n=2}^{N} w_n+\epsilon}$",
            "E_recon_H_MAX": r"$E_{\mathrm{recon},H_{\max}}=\frac{\int_0^T (v(t)-v_{0:H_{\max}}(t))^2\,dt}{\int_0^T v(t)^2\,dt+\epsilon}$",
            "Q_t_skew": r"$Q_{t,\mathrm{skew}}=\frac{(t_{90}-t_{50})-(t_{50}-t_{10})}{t_{90}-t_{10}+\epsilon}$",
            "Q_t_width": r"$Q_{t,\mathrm{width}}=\frac{t_{75}-t_{25}}{T}$",
            "R_Q_t": r"$R_{Q_t}=\frac{Q_{t,\mathrm{skew}}}{Q_{t,\mathrm{width}}+\epsilon}$",
            "Q_d_skew": r"$Q_{d,\mathrm{skew}}=\frac{(d_{90}-d_{50})-(d_{50}-d_{10})}{d_{90}-d_{10}+\epsilon}$",
            "Q_d_width": r"$Q_{d,\mathrm{width}}=d_{75}-d_{25}$",
            "R_Q_d": r"$R_{Q_d}=\frac{Q_{d,\mathrm{skew}}}{Q_{d,\mathrm{width}}+\epsilon}$",
            "v_end_over_v_mean": r"$\frac{\bar v_{\mathrm{end}}}{v_{\mathrm{mean}}}=\frac{\mathrm{mean}_{t\in[\alpha T,\beta T]} v(t)}{\mathrm{mean}_{t\in[0,T]} v(t)}$",
            "E_slope": r"$E_{\mathrm{slope}}=\frac{T^2}{M_0^2}\int_0^T \left(\frac{dv}{dt}\right)^2 dt$",
            "E_curv": r"$E_{\mathrm{curv}}=\frac{T^4}{M_0^2}\int_0^T \left(\frac{d^2v}{dt^2}\right)^2 dt$",
        }

        T = np.asarray(h5file[self.T_input])
        metrics = {}

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

            def pack(prefix: str, d: dict, attrs_common: dict):
                for k, arr in d.items():
                    metrics[f"{prefix}/{k}"] = with_attrs(arr, attrs_common)

            pack(
                "by_segment/bandlimited_segment",
                seg_b,
                {"segment_indexing": [seg_note]},
            )
            pack(
                "by_segment/raw_segment",
                seg_r,
                {"segment_indexing": [seg_note]},
            )

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

            metrics["by_segment/params/ratio_R_VTI"] = np.asarray(
                self.ratio_R_VTI, dtype=float
            )
            metrics["by_segment/params/ratio_SF_VTI"] = np.asarray(
                self.ratio_SF_VTI, dtype=float
            )
            metrics["by_segment/params/ratio_vend_start"] = np.asarray(
                self.ratio_vend_start, dtype=float
            )
            metrics["by_segment/params/ratio_vend_end"] = np.asarray(
                self.ratio_vend_end, dtype=float
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
            metrics["by_segment/params/H_MAX"] = np.asarray(self.H_MAX, dtype=int)
            metrics["by_segment/params/H_PHASE_RESIDUAL"] = np.asarray(
                self.H_PHASE_RESIDUAL, dtype=int
            )

        have_glob = (self.v_raw_global_input in h5file) and (
            self.v_band_global_input in h5file
        )
        if have_glob:
            v_raw_gl = np.asarray(h5file[self.v_raw_global_input])
            v_band_gl = np.asarray(h5file[self.v_band_global_input])

            out_raw = self._compute_block_global(v_raw_gl, T)
            out_band = self._compute_block_global(v_band_gl, T)

            for k in self._metric_keys():
                metrics[f"global/raw/{k[0]}"] = with_attrs(
                    out_raw[k[0]],
                    {
                        "unit": [k[2]],
                        "definition": [k[1]],
                        "latex_formula": [latex_formulas[k[0]]],
                    },
                )

                metrics[f"global/bandlimited/{k[0]}"] = with_attrs(
                    out_band[k[0]],
                    {
                        "unit": [k[2]],
                        "definition": [k[1]],
                        "latex_formula": [latex_formulas[k[0]]],
                    },
                )

            metrics["global/params/ratio_R_VTI"] = np.asarray(
                self.ratio_R_VTI, dtype=float
            )
            metrics["global/params/ratio_SF_VTI"] = np.asarray(
                self.ratio_SF_VTI, dtype=float
            )
            metrics["global/params/ratio_vend_start"] = np.asarray(
                self.ratio_vend_start, dtype=float
            )
            metrics["global/params/ratio_vend_end"] = np.asarray(
                self.ratio_vend_end, dtype=float
            )
            metrics["global/params/eps"] = np.asarray(self.eps, dtype=float)
            metrics["global/params/H_LOW_MAX"] = np.asarray(self.H_LOW_MAX, dtype=int)
            metrics["global/params/H_HIGH_MIN"] = np.asarray(self.H_HIGH_MIN, dtype=int)
            metrics["global/params/H_HIGH_MAX"] = np.asarray(self.H_HIGH_MAX, dtype=int)
            metrics["global/params/H_MAX"] = np.asarray(self.H_MAX, dtype=int)
            metrics["global/params/H_PHASE_RESIDUAL"] = np.asarray(
                self.H_PHASE_RESIDUAL, dtype=int
            )

        return ProcessResult(metrics=metrics)
