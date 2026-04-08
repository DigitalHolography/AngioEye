import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Waveform-shape metrics on per-beat, per-branch, per-radius velocity waveforms.
    Gain-invariant: all reported metrics are invariant to scaling of the waveform by
    a positive constant, up to numerical precision.

    Notes
    -----
    - This version computes the same metrics for both arterial and venous waveforms.
    - The registered pipeline name is intentionally kept unchanged for backward compatibility.
    - The harmonic-domain metrics implemented here follow the current manuscript body:
      low-frequency fraction, higher-harmonic rolloff/support, circular phase organization,
      and explained pulsatile fraction.
    """

    description = (
        "Waveform shape metrics (artery + vein; segment + aggregates + global), "
        "gain-invariant and robust."
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

    # Beat period input
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    eps = 1e-12

    ratio_R_VTI = 0.5
    ratio_SF_VTI = 1.0 / 3.0

    ratio_vend_start = 0.75
    ratio_vend_end = 0.90

    H_LOW_MAX = 1
    H_CUMSUM_INTERP_POINTS = 256
    H_MAX = 10
    H_PHASE_RESIDUAL = 10

    ratio_W50 = 0.50
    ratio_W80 = 0.80

    phase_weight_threshold = 0.02

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
        Returns t_q / Tbeat where d(t_q) >= q, with d(t)=cumsum(v)/sum(v) and q in [0,1].
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        m0 = float(np.sum(vv))
        if m0 <= 0:
            return np.nan

        q = float(np.clip(q, 0.0, 1.0))
        d_full = np.concatenate(([0.0], np.cumsum(vv) / m0))
        tau_full = np.linspace(0.0, 1.0, v.size + 1)

        return float(np.interp(q, d_full, tau_full))

    def _peak_width_over_T(self, v: np.ndarray, alpha: float) -> float:
        """
        Beat-normalized near-peak width:
          W_alpha/T = (1/T) * |{t in [0,T] : v(t) >= alpha * v_max}|

        On a uniformly sampled grid this is the fraction of valid samples above the threshold.
        """
        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan
        if (not np.isfinite(alpha)) or alpha <= 0 or alpha >= 1:
            return np.nan

        vv = np.asarray(v, dtype=float)
        vmax = float(np.nanmax(vv))
        if (not np.isfinite(vmax)) or vmax <= 0:
            return np.nan

        mask = np.isfinite(vv)
        if not np.any(mask):
            return np.nan

        above = mask & (vv >= alpha * vmax)
        return float(np.sum(above) / vv.size)

    def _n_t_over_T(self, v: np.ndarray, Tbeat: float, m0: float) -> float:
        """
        Beat-normalized entropic effective temporal support:
          p(t) = v(t)/M0
          N_t/T = exp( - integral_0^T p(t) log(T p(t)) dt )
        with the convention 0*log(0)=0.
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
        M0 = m0 * dt
        if (not np.isfinite(M0)) or M0 <= 0:
            return np.nan

        p = np.where(np.isfinite(v), v, 0.0) / M0
        Tp = Tbeat * p

        integrand = np.zeros_like(Tp, dtype=float)
        positive = Tp > 0
        integrand[positive] = p[positive] * np.log(Tp[positive])

        entropy_like = -float(np.sum(integrand) * dt)
        if not np.isfinite(entropy_like):
            return np.nan

        n_t_over_t = float(np.exp(entropy_like))
        return n_t_over_t if np.isfinite(n_t_over_t) else np.nan

    def _spectral_ratio_low(self, v: np.ndarray, Tbeat: float) -> float:
        """
        Return E_low/E_total using harmonic-index bands.
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        n = vv.size
        if n < 2:
            return np.nan

        fs = n / Tbeat
        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        h = f * Tbeat

        E_total = float(np.sum(P[1:]))
        if not np.isfinite(E_total) or E_total <= 0:
            return np.nan

        low_mask = (h >= 0.9) & (h <= float(self.H_LOW_MAX) + 0.1)
        E_low = float(np.sum(P[low_mask]))

        return float(E_low / E_total)

    def _harmonic_pack(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Compute complex harmonic coefficients Vn for n=0..H, with H=min(H_MAX, n_rfft-1),
        and synthesize band-limited waveform vb(t) using harmonics 0..H.
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {"V": None, "H": 0, "vb": None, "Vfull": None}

        if v.size == 0 or not np.any(np.isfinite(v)):
            return {"V": None, "H": 0, "vb": None, "Vfull": None}

        vv = np.where(np.isfinite(v), v, 0.0)
        n = vv.size
        if n < 2:
            return {"V": None, "H": 0, "vb": None, "Vfull": None}

        Vfull = np.fft.rfft(vv) / float(n)
        H = int(min(self.H_MAX, Vfull.size - 1))
        V = Vfull[: H + 1].copy()

        Vtrunc = np.zeros_like(Vfull)
        Vtrunc[: H + 1] = V
        vb = np.fft.irfft(Vtrunc * float(n), n=n)

        return {"V": V, "H": H, "vb": vb, "Vfull": Vfull}

    def _higher_harmonic_rolloff_metrics(self, V: np.ndarray) -> dict:
        """
        Higher-harmonic-only cumulative rolloff/support metrics:
          - rho_h = m_80 / (H-1)
          - w_h   = (m_80 - m_50) / (H-1)
        where m_q is the interpolated q-quantile index over harmonics n=2..H,
        and A^(2)(m) is the cumulative higher-harmonic energy distribution.
        """
        out = {
            "rho_h": np.nan,
            "w_h": np.nan,
            "m_50": np.nan,
            "m_80": np.nan,
            "A2_cumsum": np.full((max(self.H_MAX - 1, 0),), np.nan, dtype=float),
            "A2_m": np.full((max(self.H_MAX - 1, 0),), np.nan, dtype=float),
            "A2_cumsum_interp": np.full(
                (self.H_CUMSUM_INTERP_POINTS,), np.nan, dtype=float
            ),
            "A2_m_interp": np.full(
                (self.H_CUMSUM_INTERP_POINTS,), np.nan, dtype=float
            ),
        }

        if V is None:
            return out

        H = int(V.size - 1)
        if H < 2:
            return out

        power = np.abs(V[2 : H + 1]) ** 2
        power = np.where(np.isfinite(power), power, np.nan)
        s = float(np.nansum(power))
        if (not np.isfinite(s)) or s <= 0:
            return out

        a2 = power / s
        A2 = np.cumsum(a2)
        m = np.arange(1, H, dtype=float)  # 1..H-1

        out["A2_cumsum"][: H - 1] = A2
        out["A2_m"][: H - 1] = m

        A2_full = np.concatenate(([0.0], A2))
        m_full = np.arange(0, H, dtype=float)  # 0..H-1

        m_interp = np.linspace(0.0, float(H - 1), self.H_CUMSUM_INTERP_POINTS)
        A2_interp = np.interp(m_interp, m_full, A2_full)

        out["A2_m_interp"][:] = m_interp
        out["A2_cumsum_interp"][:] = A2_interp

        m50 = float(np.interp(0.50, A2_full, m_full))
        m80 = float(np.interp(0.80, A2_full, m_full))

        out["m_50"] = m50
        out["m_80"] = m80
        out["rho_h"] = float(m80 / (H - 1))
        out["w_h"] = float((m80 - m50) / (H - 1))
        return out

    def _higher_harmonic_effective_support(self, V: np.ndarray) -> float:
        """
        N_h/(H-1) where
          N_h = exp(-sum_{n=2}^H a_n^(2) log a_n^(2))
        """
        if V is None:
            return np.nan

        H = int(V.size - 1)
        if H < 2:
            return np.nan

        power = np.abs(V[2 : H + 1]) ** 2
        power = np.where(np.isfinite(power), power, 0.0)
        s = float(np.sum(power))
        if (not np.isfinite(s)) or s <= 0:
            return np.nan

        a2 = power / s
        positive = a2 > 0
        n_h = float(np.exp(-np.sum(a2[positive] * np.log(a2[positive]))))
        return float(n_h / (H - 1))

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

    def _phase_organization_metrics(self, V: np.ndarray, Tbeat: float) -> dict:
        """
        Returns:
          - D_phi = 1 - |sum_n w_n exp(i Delta_phi_n)|
          - t_phi_over_T = median_n [Delta_phi_n / (2*pi*n)] over selected harmonics
          - s_phi_over_T = median_n |Delta_phi_n/(2*pi*n) - t_phi_over_T|
          - delta_phi_all, t_phi_n, t_phi_n_over_T for diagnostics
        """
        out = {
            "D_phi": np.nan,
            "t_phi_over_T": np.nan,
            "s_phi_over_T": np.nan,
            "delta_phi_all": np.full(
                (max(self.H_PHASE_RESIDUAL - 1, 0),), np.nan, dtype=float
            ),
            "t_phi_n": np.full(
                (max(self.H_PHASE_RESIDUAL - 1, 0),), np.nan, dtype=float
            ),
            "t_phi_n_over_T": np.full(
                (max(self.H_PHASE_RESIDUAL - 1, 0),), np.nan, dtype=float
            ),
            "phase_harmonics_used": np.full(
                (max(self.H_PHASE_RESIDUAL - 1, 0),), 0, dtype=int
            ),
        }
        if V is None or (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        H = int(V.size - 1)
        Huse = int(min(H, self.H_PHASE_RESIDUAL))
        if Huse < 2:
            return out

        if np.abs(V[1]) <= self.eps:
            return out
        phi1 = self._wrap_pi(float(np.angle(V[1])))

        candidates = []
        total_power = 0.0

        for n in range(2, Huse + 1):
            if np.abs(V[n]) <= self.eps:
                continue
            phin = self._wrap_pi(float(np.angle(V[n])))
            dphi = self._wrap_pi(phin - n * phi1)
            pwr = float(np.abs(V[n]) ** 2)

            out["delta_phi_all"][n - 2] = dphi
            t_over_T = float(dphi / (2.0 * np.pi * n))
            out["t_phi_n_over_T"][n - 2] = t_over_T
            out["t_phi_n"][n - 2] = float(Tbeat * t_over_T)

            if np.isfinite(dphi) and np.isfinite(pwr) and pwr > 0:
                candidates.append((n, dphi, pwr))
                total_power += pwr

        if len(candidates) == 0 or (not np.isfinite(total_power)) or total_power <= 0:
            return out

        selected = []
        for n, dphi, pwr in candidates:
            rel = pwr / total_power
            if rel >= float(self.phase_weight_threshold):
                selected.append((n, dphi, pwr))

        if len(selected) == 0:
            selected = candidates

        weights = np.asarray([pwr for _, _, pwr in selected], dtype=float)
        weights = weights / np.sum(weights)

        angles = np.asarray([dphi for _, dphi, _ in selected], dtype=float)
        resultant = np.sum(weights * np.exp(1j * angles))
        R_phi = float(np.abs(resultant))
        out["D_phi"] = float(1.0 - R_phi)

        vals_over_T = np.asarray(
            [dphi / (2.0 * np.pi * n) for n, dphi, _ in selected], dtype=float
        )
        center = float(np.nanmedian(vals_over_T))
        spread = float(np.nanmedian(np.abs(vals_over_T - center)))

        out["t_phi_over_T"] = center
        out["s_phi_over_T"] = spread

        for n, _, _ in selected:
            out["phase_harmonics_used"][n - 2] = 1

        return out

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
        M0 = m0 * dt
        if (not np.isfinite(M0)) or M0 <= 0:
            return np.nan

        p = np.where(np.isfinite(v), v, 0.0) / M0

        int_p2 = float(np.sum(p * p) * dt)
        if not np.isfinite(int_p2) or int_p2 <= 0:
            return np.nan

        n_eff = 1.0 / int_p2
        return float(n_eff / Tbeat)

    def _explained_pulsatile_fraction(self, v: np.ndarray, vb: np.ndarray) -> float:
        """
        eta_h = 1 - int (v-v_H)^2 dt / int (v-v_mean)^2 dt
        dt cancels on a uniform grid, so sample sums are sufficient.
        """
        if v is None or vb is None:
            return np.nan
        if v.size < 2 or vb.size != v.size:
            return np.nan
        if not np.any(np.isfinite(v)) or not np.any(np.isfinite(vb)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        vbb = np.where(np.isfinite(vb), vb, 0.0)
        vbar = float(np.mean(vv))

        num = float(np.sum((vv - vbb) ** 2))
        den = float(np.sum((vv - vbar) ** 2))
        if (not np.isfinite(den)) or den <= 0:
            return np.nan

        return float(1.0 - num / den)

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
        Circular forward peak-to-trough interval:
          Delta_t_over_T = ((idx_min - idx_peak) mod n) / n

        This works whether the trough occurs after the peak within the sampled window
        or after wrap-around at the beat boundary.
        """
        if n <= 0 or idx_peak < 0 or idx_min < 0:
            return np.nan

        delta_idx = int((idx_min - idx_peak) % n)
        if delta_idx == 0:
            return np.nan

        return float(delta_idx / n)

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
        vv = np.where(np.isfinite(v), v, 0.0)
        d_full = np.concatenate(([0.0], np.cumsum(vv) / m0))
        tau_full = np.linspace(0.0, 1.0, v.size + 1)
        return float(np.trapezoid(d_full - tau_full, tau_full))

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

        vv = np.where(np.isfinite(v), v, 0.0)
        d_full = np.concatenate(([0.0], np.cumsum(vv) / m0))
        tau_full = np.linspace(0.0, 1.0, v.size + 1)

        def sample_at_ratio(r: float) -> float:
            return float(np.interp(r, tau_full, d_full))

        out["d10"] = sample_at_ratio(0.10)
        out["d25"] = sample_at_ratio(0.25)
        out["d50"] = sample_at_ratio(0.50)
        out["d75"] = sample_at_ratio(0.75)
        out["d90"] = sample_at_ratio(0.90)
        return out

    def _d_quantile_shape_metrics(self, d_samples: dict) -> tuple[float, float]:
        """
        From d10,d25,d50,d75,d90 define:
          w_d = d75 - d25
          s_d = ((d90-d50) - (d50-d10)) / (d90-d10 + eps)
        """
        d10 = d_samples["d10"]
        d25 = d_samples["d25"]
        d50 = d_samples["d50"]
        d75 = d_samples["d75"]
        d90 = d_samples["d90"]

        w_d = np.nan
        if np.isfinite(d25) and np.isfinite(d75):
            w_d = float(d75 - d25)

        s_d = np.nan
        if np.isfinite(d10) and np.isfinite(d50) and np.isfinite(d90):
            s_d = float(((d90 - d50) - (d50 - d10)) / ((d90 - d10) + self.eps))

        return w_d, s_d

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

    def _derivative_energy_slope(self, v: np.ndarray, Tbeat: float, m0: float) -> float:
        """
        E_slope = T^3 / M0^2 * int (dv/dt)^2 dt
        """
        if (
            v.size < 3
            or (not np.any(np.isfinite(v)))
            or (not np.isfinite(Tbeat))
            or Tbeat <= 0
        ):
            return np.nan
        if (not np.isfinite(m0)) or m0 <= 0:
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        dt = Tbeat / v.size
        M0 = m0 * dt
        if (not np.isfinite(M0)) or M0 <= 0:
            return np.nan

        dvdt = np.gradient(vv, dt)
        E_slope = float((Tbeat**3) * np.sum(dvdt**2) * dt / ((M0 + self.eps) ** 2))
        return E_slope

    def _compute_graphics_support_1d(self, v: np.ndarray, Tbeat: float) -> dict:
        v = self._rectify_keep_nan(v)
        n = int(v.size)

        if n <= 1 or (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {}

        k0, k1 = self._late_window_indices(v.size)
        if k1 <= k0:
            return {}

        tail = np.asarray(v[k0:k1], dtype=float)
        vend = self._safe_nanmean(tail)
        vv = np.where(np.isfinite(v), v, np.nan)
        m0_sum = float(np.nansum(vv))
        if m0_sum <= 0:
            return {}

        tau = np.linspace(0.0, 1.0, n, endpoint=False)
        dt = Tbeat / n
        m0 = float(m0_sum * dt)

        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(vv))
        vmean = float(np.nanmean(vv))

        d_full = np.concatenate(
            ([0.0], np.cumsum(np.where(np.isfinite(vv), vv, 0.0)) / m0_sum)
        )
        tau_full = np.linspace(0.0, 1.0, n + 1)
        cumulative = np.interp(tau, tau_full, d_full)
        d_star = np.asarray(cumulative, dtype=float)
        d0_star = np.asarray(tau, dtype=float)
        delta_dti_curve = d_star - d0_star

        dvdt = np.gradient(np.where(np.isfinite(vv), vv, 0.0), dt)
        d2vdt2 = np.gradient(dvdt, dt)
        dvdt_norm = (Tbeat**3 / ((m0 + self.eps) ** 2)) * (dvdt**2)
        d2vdt2_norm = (Tbeat**5 / ((m0 + self.eps) ** 2)) * (d2vdt2**2)

        hp = self._harmonic_pack(vv, Tbeat)
        V = hp["V"]
        vb = hp["vb"]
        H = int(hp["H"])

        harmonic_magnitudes = np.full((self.H_MAX + 1,), np.nan, dtype=float)
        harmonic_weights = np.full((self.H_MAX,), np.nan, dtype=float)
        harmonic_energy_weights = np.full((self.H_MAX,), np.nan, dtype=float)
        harmonic_phases = np.full((self.H_MAX,), np.nan, dtype=float)
        harmonic_energies = np.full((self.H_MAX + 1,), np.nan, dtype=float)

        E_total = np.nan
        E_low = np.nan

        if V is not None and H >= 0:
            mags = np.abs(V[: H + 1])
            power = mags**2
            harmonic_energies[: H + 1] = power
            harmonic_magnitudes[: H + 1] = mags

            if H >= 1:
                phases = np.angle(V[1 : H + 1])
                harmonic_phases[:H] = phases

            power_h = power[1 : H + 1]
            mags_h = mags[1 : H + 1]

            power_sum = float(np.nansum(power_h))
            mag_sum = float(np.nansum(mags_h))

            E_total = power_sum
            E_low = float(np.nansum(power[1 : self.H_LOW_MAX + 1]))

            if np.isfinite(power_sum) and power_sum > 0:
                harmonic_energy_weights[0:H] = power_h / (power_sum + self.eps)

            if np.isfinite(mag_sum) and mag_sum > 0:
                harmonic_weights[0:H] = mags_h / (mag_sum + self.eps)

        hh_rolloff = self._higher_harmonic_rolloff_metrics(V)
        ph = self._phase_organization_metrics(V, Tbeat)
        metrics = self._compute_metrics_1d(vv, Tbeat)

        vb_out = np.full((n,), np.nan, dtype=float)
        if vb is not None:
            vb_out[: min(len(vb), n)] = np.asarray(vb[:n], dtype=float)

        return {
            "A2_cumsum": hh_rolloff["A2_cumsum"],
            "A2_m": hh_rolloff["A2_m"],
            "A2_cumsum_interp": hh_rolloff["A2_cumsum_interp"],
            "A2_m_interp": hh_rolloff["A2_m_interp"],
            "m_50": np.asarray(hh_rolloff["m_50"], dtype=float),
            "m_80": np.asarray(hh_rolloff["m_80"], dtype=float),
            "rho_h": np.asarray(hh_rolloff["rho_h"], dtype=float),
            "w_h": np.asarray(hh_rolloff["w_h"], dtype=float),
            "H_MAX": np.asarray(self.H_MAX, dtype=int),
            "H_LOW_MAX": np.asarray(self.H_LOW_MAX, dtype=int),
            "E_total": np.asarray(E_total, dtype=float),
            "vend": np.asarray(vend, dtype=float),
            "E_low": np.asarray(E_low, dtype=float),
            "signal_mean": np.asarray(vv, dtype=float),
            "tau": np.asarray(tau, dtype=float),
            "cumulative": np.asarray(cumulative, dtype=float),
            "d_star": np.asarray(d_star, dtype=float),
            "d0_star": np.asarray(d0_star, dtype=float),
            "delta_dti_curve": np.asarray(delta_dti_curve, dtype=float),
            "vb": vb_out,
            "dvdt": np.asarray(dvdt, dtype=float),
            "d2vdt2": np.asarray(d2vdt2, dtype=float),
            "dvdt_norm": np.asarray(dvdt_norm, dtype=float),
            "d2vdt2_norm": np.asarray(d2vdt2_norm, dtype=float),
            "harmonic_magnitudes": harmonic_magnitudes,
            "harmonic_weights": harmonic_weights,
            "harmonic_energies": harmonic_energies,
            "harmonic_energies_weights": harmonic_energy_weights,
            "harmonic_phases": harmonic_phases,
            "delta_phi_all": ph["delta_phi_all"],
            "t_phi_n": ph["t_phi_n"],
            "t_phi_n_over_T": ph["t_phi_n_over_T"],
            "phase_harmonics_used": ph["phase_harmonics_used"],
            "vmax": np.asarray(vmax, dtype=float),
            "vmin": np.asarray(vmin, dtype=float),
            "vmean": np.asarray(vmean, dtype=float),
            "m0": np.asarray(m0, dtype=float),
            **{k: np.asarray(val, dtype=float) for k, val in metrics.items()},
        }

    def _compute_graphics_support_block(
        self, v_global: np.ndarray, T: np.ndarray
    ) -> dict:
        n_beats = int(T.shape[1])
        v_global = self._ensure_time_by_beat(v_global, n_beats)
        v_global = self._rectify_keep_nan(v_global)

        n_t = int(v_global.shape[0])
        h_mag = self.H_MAX
        h_phi = max(self.H_PHASE_RESIDUAL - 1, 0)
        h_higher = max(self.H_MAX - 1, 0)

        out = {
            "A2_cumsum": np.full((n_beats, h_higher), np.nan, dtype=float),
            "A2_m": np.full((n_beats, h_higher), np.nan, dtype=float),
            "A2_cumsum_interp": np.full(
                (n_beats, self.H_CUMSUM_INTERP_POINTS), np.nan, dtype=float
            ),
            "A2_m_interp": np.full(
                (n_beats, self.H_CUMSUM_INTERP_POINTS), np.nan, dtype=float
            ),
            "m_50": np.full((n_beats,), np.nan, dtype=float),
            "m_80": np.full((n_beats,), np.nan, dtype=float),
            "rho_h": np.full((n_beats,), np.nan, dtype=float),
            "w_h": np.full((n_beats,), np.nan, dtype=float),
            "H_MAX": np.asarray(self.H_MAX, dtype=int),
            "H_LOW_MAX": np.asarray(self.H_LOW_MAX, dtype=int),
            "signal_mean": np.full((n_t, n_beats), np.nan, dtype=float),
            "tau": np.full((n_t, n_beats), np.nan, dtype=float),
            "cumulative": np.full((n_t, n_beats), np.nan, dtype=float),
            "d_star": np.full((n_t, n_beats), np.nan, dtype=float),
            "d0_star": np.full((n_t, n_beats), np.nan, dtype=float),
            "delta_dti_curve": np.full((n_t, n_beats), np.nan, dtype=float),
            "vb": np.full((n_t, n_beats), np.nan, dtype=float),
            "m0": np.full((n_beats,), np.nan),
            "E_total": np.full((n_beats,), np.nan, dtype=float),
            "vend": np.full((n_beats,), np.nan, dtype=float),
            "E_low": np.full((n_beats,), np.nan, dtype=float),
            "dvdt": np.full((n_t, n_beats), np.nan, dtype=float),
            "dvdt_norm": np.full((n_t, n_beats), np.nan, dtype=float),
            "d2vdt2": np.full((n_t, n_beats), np.nan, dtype=float),
            "d2vdt2_norm": np.full((n_t, n_beats), np.nan, dtype=float),
            "harmonic_magnitudes": np.full((n_beats, h_mag + 1), np.nan, dtype=float),
            "harmonic_weights": np.full((n_beats, h_mag), np.nan, dtype=float),
            "harmonic_phases": np.full((n_beats, h_mag), np.nan, dtype=float),
            "harmonic_energies": np.full((n_beats, h_mag + 1), np.nan, dtype=float),
            "harmonic_energies_weights": np.full((n_beats, h_mag), np.nan, dtype=float),
            "delta_phi_all": np.full((n_beats, h_phi), np.nan, dtype=float),
            "t_phi_n": np.full((n_beats, h_phi), np.nan, dtype=float),
            "t_phi_n_over_T": np.full((n_beats, h_phi), np.nan, dtype=float),
            "phase_harmonics_used": np.full((n_beats, h_phi), 0, dtype=int),
            "vmax": np.full((n_beats,), np.nan, dtype=float),
            "vmin": np.full((n_beats,), np.nan, dtype=float),
            "vmean": np.full((n_beats,), np.nan, dtype=float),
        }

        for k in self._metric_keys():
            out[k[0]] = np.full((n_beats,), np.nan, dtype=float)

        for beat_idx in range(n_beats):
            Tbeat = float(T[0][beat_idx])
            v = v_global[:, beat_idx]
            s = self._compute_graphics_support_1d(v, Tbeat)

            out["A2_cumsum"][beat_idx, :] = s["A2_cumsum"]
            out["A2_m"][beat_idx, :] = s["A2_m"]
            out["A2_cumsum_interp"][beat_idx, :] = s["A2_cumsum_interp"]
            out["A2_m_interp"][beat_idx, :] = s["A2_m_interp"]
            out["m_50"][beat_idx] = s["m_50"]
            out["m_80"][beat_idx] = s["m_80"]
            out["rho_h"][beat_idx] = s["rho_h"]
            out["w_h"][beat_idx] = s["w_h"]

            out["E_total"][beat_idx] = s["E_total"]
            out["E_low"][beat_idx] = s["E_low"]
            out["signal_mean"][:, beat_idx] = s["signal_mean"]
            out["tau"][:, beat_idx] = s["tau"]
            out["cumulative"][:, beat_idx] = s["cumulative"]
            out["d_star"][:, beat_idx] = s["d_star"]
            out["d0_star"][:, beat_idx] = s["d0_star"]
            out["delta_dti_curve"][:, beat_idx] = s["delta_dti_curve"]
            out["vb"][:, beat_idx] = s["vb"]
            out["dvdt"][:, beat_idx] = s["dvdt"]
            out["d2vdt2"][:, beat_idx] = s["d2vdt2"]
            out["dvdt_norm"][:, beat_idx] = s["dvdt_norm"]
            out["d2vdt2_norm"][:, beat_idx] = s["d2vdt2_norm"]
            out["m0"][beat_idx] = s["m0"]
            out["harmonic_magnitudes"][beat_idx, :] = s["harmonic_magnitudes"]
            out["harmonic_weights"][beat_idx, :] = s["harmonic_weights"]
            out["harmonic_phases"][beat_idx, :] = s["harmonic_phases"]
            out["harmonic_energies"][beat_idx, :] = s["harmonic_energies"]
            out["harmonic_energies_weights"][beat_idx, :] = s[
                "harmonic_energies_weights"
            ]
            out["delta_phi_all"][beat_idx, :] = s["delta_phi_all"]
            out["t_phi_n"][beat_idx, :] = s["t_phi_n"]
            out["t_phi_n_over_T"][beat_idx, :] = s["t_phi_n_over_T"]
            out["phase_harmonics_used"][beat_idx, :] = s["phase_harmonics_used"]
            out["vmax"][beat_idx] = s["vmax"]
            out["vmin"][beat_idx] = s["vmin"]
            out["vmean"][beat_idx] = s["vmean"]
            out["vend"][beat_idx] = s["vend"]

            for k in self._metric_keys():
                out[k[0]][beat_idx] = s[k[0]]

        return out

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

        W50_over_T = self._peak_width_over_T(vv, self.ratio_W50)
        W80_over_T = self._peak_width_over_T(vv, self.ratio_W80)

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

        E_low_over_E_total = self._spectral_ratio_low(vv, Tbeat)

        hp = self._harmonic_pack(vv, Tbeat)
        V = hp["V"]
        vb = hp["vb"]

        hh_rolloff = self._higher_harmonic_rolloff_metrics(V)
        N_h_over_H_minus_1 = self._higher_harmonic_effective_support(V)
        ph = self._phase_organization_metrics(V, Tbeat)

        crest_factor = self._crest_factor(vv)
        N_eff_over_T = self._n_eff_over_T(vv, Tbeat, m0)
        N_t_over_T = self._n_t_over_T(vv, Tbeat, m0)

        t_max_over_T, t_min_over_T, idx_peak, idx_min = self._peak_trough_times(vv)
        (
            slope_rise_normalized,
            slope_fall_normalized,
            t_up_over_T,
            t_down_over_T,
        ) = self._normalized_slopes_and_times(vv, Tbeat)

        Delta_t_over_T = self._peak_to_trough_interval(idx_peak, idx_min, n)
        Delta_DTI = self._delta_dti(vv, Tbeat, m0, t)
        gamma_t = self._gamma_t(vv, Tbeat, mu_t, sigma_t, m0, t)

        eta_h = self._explained_pulsatile_fraction(vv, vb)

        s_t = np.nan
        if (
            np.isfinite(t10_over_T)
            and np.isfinite(t50_over_T)
            and np.isfinite(t90_over_T)
        ):
            denom = (t90_over_T - t10_over_T) + self.eps
            s_t = float(
                ((t90_over_T - t50_over_T) - (t50_over_T - t10_over_T)) / denom
            )

        w_t = np.nan
        if np.isfinite(t25_over_T) and np.isfinite(t75_over_T):
            w_t = float(t75_over_T - t25_over_T)

        w_d, s_d = self._d_quantile_shape_metrics(d_samples)

        v_end_over_v_mean = self._late_cycle_mean_fraction(vv)
        E_slope = self._derivative_energy_slope(vv, Tbeat, m0)

        return {
            "mu_t": float(mu_t),
            "mu_t_over_T": float(mu_t_over_T),
            "RI": float(RI) if np.isfinite(RI) else np.nan,
            "PI": float(PI) if np.isfinite(PI) else np.nan,
            "R_VTI": float(R_VTI),
            "SF_VTI": float(SF_VTI),
            "sigma_t_over_T": float(sigma_t_over_T),
            "sigma_t": float(sigma_t),
            "W50_over_T": float(W50_over_T) if np.isfinite(W50_over_T) else np.nan,
            "W80_over_T": float(W80_over_T) if np.isfinite(W80_over_T) else np.nan,
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
            "E_low_over_E_total": float(E_low_over_E_total)
            if np.isfinite(E_low_over_E_total)
            else np.nan,
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
            "Delta_DTI": float(Delta_DTI) if np.isfinite(Delta_DTI) else np.nan,
            "gamma_t": float(gamma_t) if np.isfinite(gamma_t) else np.nan,
            "crest_factor": float(crest_factor)
            if np.isfinite(crest_factor)
            else np.nan,
            "t_phi_over_T": float(ph["t_phi_over_T"])
            if np.isfinite(ph["t_phi_over_T"])
            else np.nan,
            "s_phi_over_T": float(ph["s_phi_over_T"])
            if np.isfinite(ph["s_phi_over_T"])
            else np.nan,
            "D_phi": float(ph["D_phi"]) if np.isfinite(ph["D_phi"]) else np.nan,
            "rho_h": float(hh_rolloff["rho_h"])
            if np.isfinite(hh_rolloff["rho_h"])
            else np.nan,
            "w_h": float(hh_rolloff["w_h"])
            if np.isfinite(hh_rolloff["w_h"])
            else np.nan,
            "N_h_over_H_minus_1": float(N_h_over_H_minus_1)
            if np.isfinite(N_h_over_H_minus_1)
            else np.nan,
            "N_eff_over_T": float(N_eff_over_T)
            if np.isfinite(N_eff_over_T)
            else np.nan,
            "N_t_over_T": float(N_t_over_T) if np.isfinite(N_t_over_T) else np.nan,
            "eta_h": float(eta_h) if np.isfinite(eta_h) else np.nan,
            "s_t": float(s_t) if np.isfinite(s_t) else np.nan,
            "w_t": float(w_t) if np.isfinite(w_t) else np.nan,
            "s_d": float(s_d) if np.isfinite(s_d) else np.nan,
            "w_d": float(w_d) if np.isfinite(w_d) else np.nan,
            "v_end_over_v_mean": float(v_end_over_v_mean)
            if np.isfinite(v_end_over_v_mean)
            else np.nan,
            "E_slope": float(E_slope) if np.isfinite(E_slope) else np.nan,
        }

    @staticmethod
    def _metric_keys() -> list[list]:
        return [
            ["mu_t", "sum(w(t)*t)/sum(w(t))", "seconds"],
            ["mu_t_over_T", "mu/T", ""],
            ["RI", "(V_systole-V_diastole)/V_systole", ""],
            ["PI", "(V_systole-V_diastole)/V_mean", ""],
            ["R_VTI", "D(alpha T)/(D_T-D(alpha T))", ""],
            ["SF_VTI", "D(alpha T)/D_T", ""],
            ["sigma_t_over_T", "sigma/T", ""],
            ["sigma_t", "sqrt(tau_M2-tau_M1**2)", "seconds"],
            ["W50_over_T", "W_{50}/T", ""],
            ["W80_over_T", "W_{80}/T", ""],
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
            ["t_max_over_T", "t_max/T", ""],
            ["t_min_over_T", "t_min/T", ""],
            ["Delta_t_over_T", "((t_min-t_max) mod T)/T", ""],
            ["slope_rise_normalized", "T*max(dv/dt)/V_mean", ""],
            ["slope_fall_normalized", "T*|min(dv/dt)|/V_mean", ""],
            ["t_up_over_T", "t_up/T", ""],
            ["t_down_over_T", "t_down/T", ""],
            ["Delta_DTI", "int_0^1(d*(tau)-tau)dtau", ""],
            ["gamma_t", "sum(w(t)*((t-mu)/sigma)^3)/sum(w(t))", ""],
            ["crest_factor", "V_max/V_RMS", ""],
            ["t_phi_over_T", "median_n(Delta_phi_n/(2*pi*n))", ""],
            [
                "s_phi_over_T",
                "median_n|Delta_phi_n/(2*pi*n)-t_phi/T|",
                "",
            ],
            ["D_phi", "1-|sum_n w_n exp(i Delta_phi_n)|", ""],
            ["rho_h", "m_80/(H-1)", ""],
            ["w_h", "(m_80-m_50)/(H-1)", ""],
            [
                "N_h_over_H_minus_1",
                "exp(-sum_{n>=2} a_n^(2) log a_n^(2))/(H-1)",
                "",
            ],
            ["N_eff_over_T", "N_eff/T", ""],
            ["N_t_over_T", "N_t/T", ""],
            ["eta_h", "1 - int(v-v_H)^2 dt / int((v-v_mean)^2) dt", ""],
            ["s_t", "((t90-t50)-(t50-t10))/(t90-t10+eps)", ""],
            ["w_t", "(t75-t25)/T", ""],
            ["s_d", "((d90-d50)-(d50-d10))/(d90-d10+eps)", ""],
            ["w_d", "d75-d25", ""],
            [
                "v_end_over_v_mean",
                "mean(v[t in ratio_vend_start*T:ratio_vend_end*T])/mean(v)",
                "",
            ],
            ["E_slope", "T^3/M0^2 * int (dv/dt)^2 dt", ""],
        ]

    def _compute_block_segment(self, v_block: np.ndarray, T: np.ndarray):
        """
        v_block: (n_t, n_beats, n_branches, n_radii)

        Returns:
          per-segment arrays: (n_beats, n_branches, n_radii)
          per-branch arrays:  (n_beats, n_branches)          (median over radii)
          global arrays:      (n_beats,)                     (median over all branch-radius values)
        """
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}"
            )

        _, n_beats, n_branches, n_radii = v_block.shape

        seg = {
            k[0]: np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
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

                    for k in self._metric_keys():
                        key = k[0]
                        seg[key][beat_idx, branch_idx, radius_idx] = m[key]
                        br_vals[key].append(m[key])
                        gl_vals[key].append(m[key])

                for k in self._metric_keys():
                    key = k[0]
                    br[key][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[key], dtype=float)
                    )

            for k in self._metric_keys():
                key = k[0]
                gl[key][beat_idx] = self._safe_nanmedian(
                    np.asarray(gl_vals[key], dtype=float)
                )

        seg_order_note = "segment arrays are stored as (beat, branch, radius)"
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

        def pack(prefix: str, d: dict, attrs_common: dict):
            for k, arr in d.items():
                metrics[f"{vessel_prefix}/{prefix}/{k}"] = with_attrs(arr, attrs_common)

        pack(
            "by_segment/bandlimited_segment",
            seg_b,
            {
                "definition": ["per-segment metrics stored as (beat, branch, radius)"],
                "segment_indexing": [seg_note],
            },
        )
        pack(
            "by_segment/raw_segment",
            seg_r,
            {
                "definition": ["per-segment metrics stored as (beat, branch, radius)"],
                "segment_indexing": [seg_note],
            },
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
            {"definition": ["median over all branch-radius segment values per beat"]},
        )
        pack(
            "by_segment/raw_global",
            gl_r,
            {"definition": ["median over all branch-radius segment values per beat"]},
        )

        metrics[f"{vessel_prefix}/by_segment/params/ratio_R_VTI"] = np.asarray(
            self.ratio_R_VTI, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/ratio_SF_VTI"] = np.asarray(
            self.ratio_SF_VTI, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/ratio_vend_start"] = np.asarray(
            self.ratio_vend_start, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/ratio_vend_end"] = np.asarray(
            self.ratio_vend_end, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/eps"] = np.asarray(
            self.eps, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/ratio_W50"] = np.asarray(
            self.ratio_W50, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/ratio_W80"] = np.asarray(
            self.ratio_W80, dtype=float
        )
        metrics[f"{vessel_prefix}/by_segment/params/H_LOW_MAX"] = np.asarray(
            self.H_LOW_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/by_segment/params/H_MAX"] = np.asarray(
            self.H_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/by_segment/params/H_PHASE_RESIDUAL"] = np.asarray(
            self.H_PHASE_RESIDUAL, dtype=int
        )
        metrics[f"{vessel_prefix}/by_segment/params/phase_weight_threshold"] = np.asarray(
            self.phase_weight_threshold, dtype=float
        )

    def _pack_global_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_raw_gl: np.ndarray,
        v_band_gl: np.ndarray,
        T: np.ndarray,
        latex_formulas: dict,
    ) -> None:
        out_raw = self._compute_block_global(v_raw_gl, T)
        out_band = self._compute_block_global(v_band_gl, T)

        for k in self._metric_keys():
            metrics[f"{vessel_prefix}/global/raw/{k[0]}"] = with_attrs(
                out_raw[k[0]],
                {
                    "unit": [k[2]],
                    "definition": [k[1]],
                    "latex_formula": [latex_formulas[k[0]]],
                },
            )

            metrics[f"{vessel_prefix}/global/bandlimited/{k[0]}"] = with_attrs(
                out_band[k[0]],
                {
                    "unit": [k[2]],
                    "definition": [k[1]],
                    "latex_formula": [latex_formulas[k[0]]],
                },
            )

        metrics[f"{vessel_prefix}/global/params/ratio_R_VTI"] = np.asarray(
            self.ratio_R_VTI, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/ratio_SF_VTI"] = np.asarray(
            self.ratio_SF_VTI, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/ratio_vend_start"] = np.asarray(
            self.ratio_vend_start, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/ratio_vend_end"] = np.asarray(
            self.ratio_vend_end, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/eps"] = np.asarray(
            self.eps, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/ratio_W50"] = np.asarray(
            self.ratio_W50, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/ratio_W80"] = np.asarray(
            self.ratio_W80, dtype=float
        )
        metrics[f"{vessel_prefix}/global/params/H_LOW_MAX"] = np.asarray(
            self.H_LOW_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/global/params/H_MAX"] = np.asarray(
            self.H_MAX, dtype=int
        )
        metrics[f"{vessel_prefix}/global/params/H_PHASE_RESIDUAL"] = np.asarray(
            self.H_PHASE_RESIDUAL, dtype=int
        )
        metrics[f"{vessel_prefix}/global/params/phase_weight_threshold"] = np.asarray(
            self.phase_weight_threshold, dtype=float
        )

        graphics_raw = self._compute_graphics_support_block(v_raw_gl, T)
        graphics_band = self._compute_graphics_support_block(v_band_gl, T)

        for name, arr in graphics_raw.items():
            metrics[f"{vessel_prefix}/global/raw/{name}"] = arr

        for name, arr in graphics_band.items():
            metrics[f"{vessel_prefix}/global/bandlimited/{name}"] = arr

    def run(self, h5file) -> ProcessResult:
        latex_formulas = {
            "mu_t": r"$\mu_t=\frac{\sum_t v(t)\,t}{\sum_t v(t)}$",
            "mu_t_over_T": r"$\frac{\mu_t}{T}$",
            "RI": r"$\frac{V_{systole}-V_{diastole}}{V_{systole}}$",
            "PI": r"$\frac{V_{systole}-V_{diastole}}{V_{mean}}$",
            "R_VTI": r"$\mathrm{R}_{\mathrm{VTI}}=\frac{D(\alpha T)}{D_T-D(\alpha T)}$",
            "SF_VTI": r"$\mathrm{SF}_{\mathrm{VTI}}=\frac{D(\alpha T)}{D_T}$",
            "sigma_t": r"$\sigma_t=\sqrt{\frac{\sum_t v(t)(t-\mu_t)^2}{\sum_t v(t)}}$",
            "sigma_t_over_T": r"$\frac{\sigma_t}{T}$",
            "W50_over_T": r"$\frac{W_{50}}{T}=\frac{1}{T}\left|\{t\in[0,T]:v(t)\geq 0.5\,v_{\max}\}\right|$",
            "W80_over_T": r"$\frac{W_{80}}{T}=\frac{1}{T}\left|\{t\in[0,T]:v(t)\geq 0.8\,v_{\max}\}\right|$",
            "t10_over_T": r"$\frac{t_{10}}{T}$",
            "t25_over_T": r"$\frac{t_{25}}{T}$",
            "t50_over_T": r"$\frac{t_{50}}{T}$",
            "t75_over_T": r"$\frac{t_{75}}{T}$",
            "t90_over_T": r"$\frac{t_{90}}{T}$",
            "d10": r"$d_{10}$",
            "d25": r"$d_{25}$",
            "d50": r"$d_{50}$",
            "d75": r"$d_{75}$",
            "d90": r"$d_{90}$",
            "E_low_over_E_total": r"$\frac{E_{\mathrm{low}}}{E_{\mathrm{tot}}}$",
            "t_max_over_T": r"$\frac{t_{\max}}{T}$",
            "t_min_over_T": r"$\frac{t_{\min}}{T}$",
            "Delta_t_over_T": r"$\frac{(t_{\min}-t_{\max})\bmod T}{T}$",
            "slope_rise_normalized": r"$\frac{T}{\bar v}\max_t \frac{dv}{dt}$",
            "slope_fall_normalized": r"$\frac{T}{\bar v}\left|\min_t \frac{dv}{dt}\right|$",
            "t_up_over_T": r"$\frac{t_{\mathrm{up}}}{T}$",
            "t_down_over_T": r"$\frac{t_{\mathrm{down}}}{T}$",
            "Delta_DTI": r"$\int_0^1 \left[d^*(\tau)-\tau\right]d\tau$",
            "gamma_t": r"$\frac{1}{M_0}\sum_t v(t)\left(\frac{t-\mu_t}{\sigma_t}\right)^3$",
            "crest_factor": r"$\mathrm{CF}=\frac{v_{\max}}{v_{\mathrm{RMS}}}$",
            "t_phi_over_T": r"$\frac{t_\phi}{T}=\mathrm{median}_{n\in\mathcal H_\phi}\left(\frac{\Delta\phi_n}{2\pi n}\right)$",
            "s_phi_over_T": r"$\frac{s_\phi}{T}=\mathrm{median}_{n\in\mathcal H_\phi}\left|\frac{\Delta\phi_n}{2\pi n}-\frac{t_\phi}{T}\right|$",
            "D_phi": r"$D_\phi=1-\left|\sum_{n\in\mathcal H_\phi} w_n e^{i\Delta\phi_n}\right|$",
            "rho_h": r"$\rho_h=\frac{m_{0.8}}{H-1}$",
            "w_h": r"$w_h=\frac{m_{0.8}-m_{0.5}}{H-1}$",
            "N_h_over_H_minus_1": r"$\frac{N_h}{H-1}=\frac{\exp\!\left(-\sum_{n=2}^{H} a_n^{(2)}\log a_n^{(2)}\right)}{H-1}$",
            "N_eff_over_T": r"$\frac{N_{\mathrm{eff}}}{T}=\frac{1}{T\int_0^T p(t)^2\,dt}$",
            "N_t_over_T": r"$\frac{N_t}{T}=\exp\!\left(-\int_0^T p(t)\ln(Tp(t))\,dt\right)$",
            "eta_h": r"$\eta_h=1-\frac{\int_0^T (v(t)-v_H(t))^2\,dt}{\int_0^T (v(t)-\bar v)^2\,dt}$",
            "s_t": r"$s_t=\frac{(t_{90}-t_{50})-(t_{50}-t_{10})}{t_{90}-t_{10}+\epsilon}$",
            "w_t": r"$w_t=\frac{t_{75}-t_{25}}{T}$",
            "s_d": r"$s_d=\frac{(d_{90}-d_{50})-(d_{50}-d_{10})}{d_{90}-d_{10}+\epsilon}$",
            "w_d": r"$w_d=d_{75}-d_{25}$",
            "v_end_over_v_mean": r"$\frac{\bar v_{\mathrm{end}}}{v_{\mathrm{mean}}}=\frac{\mathrm{mean}_{t\in[\alpha T,\beta T]} v(t)}{\mathrm{mean}_{t\in[0,T]} v(t)}$",
            "E_slope": r"$E_{\mathrm{slope}}=\frac{T^3}{M_0^2}\int_0^T \left(\frac{dv}{dt}\right)^2 dt$",
        }

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
                    metrics,
                    vessel_prefix,
                    v_raw_gl,
                    v_band_gl,
                    T,
                    latex_formulas,
                )

        return ProcessResult(metrics=metrics)
