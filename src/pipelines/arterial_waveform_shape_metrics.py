import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterial_waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Waveform-shape metrics on per-beat, per-branch, per-radius velocity waveforms.

    Adds harmonic-domain aggregated metrics:
      - tauH (magnitude-only damping time constant proxy)
      - crest_factor (from band-limited synthesis n=0..10)
      - spectral_entropy (harmonic magnitude distribution entropy, n=1..10)
      - phi1, phi2, phi3 (harmonic phases)
      - delta_phi2, delta_phi3 as phase-coupling:
            delta_phi2 = wrap(phi2 - 2*phi1)   (aka Δϕ2)
            delta_phi3 = wrap(phi3 - 3*phi1)   (aka Δϕ3)
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
    ratio_R_VTI = 0.5  # split for R_VTI
    ratio_SF_VTI = 1.0 / 3.0  # split for SF_VTI

    # Spectral bands (harmonic indices, inclusive)
    H_LOW_MAX = 3
    H_HIGH_MIN = 4
    H_HIGH_MAX = 8

    # Harmonic panel max for harmonic-domain metrics
    H_MAX = 10  # use n=0..10 for synthesis; n=1..10 for distributions/aggregation

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

    @staticmethod
    def _wrap_pi(x: float) -> float:
        """Wrap angle to [-pi, pi]."""
        if not np.isfinite(x):
            return np.nan
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

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

        vv = np.where(np.isfinite(v), v, np.nan)
        n = vv.size
        if n < 2:
            return np.nan, np.nan

        fs = n / Tbeat  # Hz
        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        f = np.fft.rfftfreq(n, d=1.0 / fs)  # Hz
        h = f * Tbeat  # harmonic index (cycles/beat)

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
          - V (complex array length H+1)  [Fourier series-style coefficients]
          - H (int)
          - vb (float array length n)
          - Vmax (float) = max_t vb(t)
          - omega0 (float) = 2*pi/Tbeat
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {"V": None, "H": 0, "vb": None, "Vmax": np.nan, "omega0": np.nan}

        if v.size == 0 or not np.any(np.isfinite(v)):
            return {"V": None, "H": 0, "vb": None, "Vmax": np.nan, "omega0": np.nan}

        vv = np.where(np.isfinite(v), v, np.nan)
        n = vv.size
        if n < 2:
            return {"V": None, "H": 0, "vb": None, "Vmax": np.nan, "omega0": np.nan}

        # Coeffs scaled so that irfft(V*n) reconstructs the signal (Fourier-series-like)
        Vfull = np.fft.rfft(vv) / float(n)
        H = int(min(self.H_MAX, Vfull.size - 1))
        V = Vfull[: H + 1].copy()

        # Band-limited synthesis using harmonics 0..H
        Vtrunc = np.zeros_like(Vfull)
        Vtrunc[: H + 1] = V
        vb = np.fft.irfft(Vtrunc * float(n), n=n)

        Vmax = float(np.nanmax(vb)) if vb.size else np.nan
        omega0 = float(2.0 * np.pi / Tbeat)

        return {"V": V, "H": H, "vb": vb, "Vmax": Vmax, "omega0": omega0}

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

            # mapping is only valid if |Xn| <= 1
            if Xn > 1.0 + 1e-9:
                continue

            omega_n = float(n) * omega0
            # numeric safety
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

    def _spectral_flatness_from_harmonics(self, V: np.ndarray) -> float:
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

    def _crest_factor_from_vb(self, vb: np.ndarray) -> float:
        """
        Crest factor on band-limited waveform vb:
          CF = max(vb) / rms(vb)
        """
        if vb is None or vb.size == 0:
            return np.nan
        vb = np.asarray(vb, dtype=float)
        if not np.any(np.isfinite(vb)):
            return np.nan
        x = np.where(np.isfinite(vb), vb, np.nan)
        rms = float(np.sqrt(self._safe_nanmean(x * x)))
        if rms <= 0:
            return np.nan
        return float(np.max(x) / rms)

    def _harmonic_phases(self, V: np.ndarray) -> dict:
        """
        Return phi1,phi2,phi3 and delta_phi2,delta_phi3 (phase-coupling wrt fundamental).
        delta_phi2 = wrap(phi2 - 2*phi1)
        delta_phi3 = wrap(phi3 - 3*phi1)
        """
        out = {
            "phi1": np.nan,
            "phi2": np.nan,
            "phi3": np.nan,
            "delta_phi2": np.nan,
            "delta_phi3": np.nan,
        }
        if V is None:
            return out

        H = int(V.size - 1)
        if H < 1:
            return out

        def phase_if_strong(Vn: complex) -> float:
            if not np.isfinite(Vn.real) or not np.isfinite(Vn.imag):
                return np.nan
            if np.abs(Vn) <= self.eps:
                return np.nan
            return self._wrap_pi(float(np.angle(Vn)))

        phi1 = phase_if_strong(V[1]) if H >= 1 else np.nan
        phi2 = phase_if_strong(V[2]) if H >= 2 else np.nan
        phi3 = phase_if_strong(V[3]) if H >= 3 else np.nan

        out["phi1"] = phi1
        out["phi2"] = phi2
        out["phi3"] = phi3

        if np.isfinite(phi1) and np.isfinite(phi2):
            out["delta_phi2"] = self._wrap_pi(phi2 - 2.0 * phi1)
        if np.isfinite(phi1) and np.isfinite(phi3):
            out["delta_phi3"] = self._wrap_pi(phi3 - 3.0 * phi1)

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

        # First moment
        m1 = float(np.nansum(vv * t))
        mu_t = m1 / m0
        mu_t_over_T = mu_t / Tbeat

        # RI / PI robust
        vmax = float(np.nanmax(vv))
        vmin = float(np.nanmin(v))  # preserve NaNs already
        meanv = float(self._safe_nanmean(v))

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

        # R_VTI (split 1/2)
        k_R_VTI = int(np.ceil(n * self.ratio_R_VTI))
        k_R_VTI = max(0, min(n, k_R_VTI))
        D1_R_VTI = float(np.sum(vv[:k_R_VTI])) if k_R_VTI > 0 else np.nan
        D2_R_VTI = float(np.sum(vv[k_R_VTI:])) if k_R_VTI < n else np.nan
        R_VTI = D1_R_VTI / (D2_R_VTI + self.eps)

        # SF_VTI (split 1/3 vs 2/3)
        k_sf = int(np.ceil(n * self.ratio_SF_VTI))
        k_sf = max(0, min(n, k_sf))
        D1_sf = float(np.nansum(vv[:k_sf])) if k_sf > 0 else np.nan
        D2_sf = float(np.nansum(vv[k_sf:])) if k_sf < n else np.nan
        SF_VTI = D1_sf / (D1_sf + D2_sf + self.eps)

        # Central moments around mu_t
        dtau = t - mu_t
        m2 = float(np.nansum(vv * (dtau**2)))
        sigma_t = np.sqrt(m2 / m0 + self.eps)
        sigma_t_over_T = sigma_t / Tbeat

        # Quantile timing features
        t10_over_T = self._quantile_time_over_T(vv, Tbeat, 0.10)
        t25_over_T = self._quantile_time_over_T(vv, Tbeat, 0.25)
        t50_over_T = self._quantile_time_over_T(vv, Tbeat, 0.50)
        t75_over_T = self._quantile_time_over_T(vv, Tbeat, 0.75)
        t90_over_T = self._quantile_time_over_T(vv, Tbeat, 0.90)
        m0 = float(np.nansum(vv))

        C = np.nancumsum(vv) / (m0 + self.eps)
        x_norm = np.linspace(0.0, 1.0, n, endpoint=False)
        d = C - x_norm
        AUC_cumsum = np.nansum(d)
        x5 = x_norm[:4]
        y5 = d[:4]

        Tan_cumsum, b = np.polyfit(x5, y5, 1)
        # Spectral ratios (FFT power bands)
        E_low_over_E_total, E_high_over_E_total = self._spectral_ratios(vv, Tbeat)

        # Harmonic-domain extras (n=0..10 synthesis; n=1..10 metrics)
        hp = self._harmonic_pack(np.where(np.isfinite(vv), vv, np.nan), Tbeat)
        V = hp["V"]
        vb = hp["vb"]
        Vmax_bl = hp["Vmax"]
        omega0 = hp["omega0"]

        tauH = self._tauH_from_harmonics(V, Vmax_bl, omega0)
        crest_factor = self._crest_factor_from_vb(vb)
        spectral_entropy = self._spectral_entropy_from_harmonics(V)
        ph = self._harmonic_phases(V)

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
            "E_low_over_E_total": float(E_low_over_E_total),
            "E_high_over_E_total": float(E_high_over_E_total),
            # NEW harmonic-domain requested metrics
            "tauH": float(tauH) if np.isfinite(tauH) else np.nan,
            "crest_factor": float(crest_factor)
            if np.isfinite(crest_factor)
            else np.nan,
            "spectral_entropy": float(spectral_entropy)
            if np.isfinite(spectral_entropy)
            else np.nan,
            "phi1": float(ph["phi1"]) if np.isfinite(ph["phi1"]) else np.nan,
            "phi2": float(ph["phi2"]) if np.isfinite(ph["phi2"]) else np.nan,
            "phi3": float(ph["phi3"]) if np.isfinite(ph["phi3"]) else np.nan,
            "delta_phi2": float(ph["delta_phi2"])
            if np.isfinite(ph["delta_phi2"])
            else np.nan,
            "delta_phi3": float(ph["delta_phi3"])
            if np.isfinite(ph["delta_phi3"])
            else np.nan,
            "AUC_cumsum": float(AUC_cumsum),
            "Tan_cumsum": float(Tan_cumsum),
        }

    @staticmethod
    def _metric_keys() -> list[list]:
        return [
            ["mu_t", "sum(w(t)*t)/sum(w(t))", "seconds"],
            ["mu_t_over_T", "mu/T", ""],
            ["RI", "(V_systole-V_diastole)/V_systole", ""],
            ["PI", "(V_systole-V_diastole)/V_mean", ""],
            ["R_VTI", "VTI_0_T2/(VTI_T2_T+eps)", ""],
            ["SF_VTI", "VTI_0_T2/VTI_0_T", ""],
            ["sigma_t_over_T", "sigma/T", ""],
            ["sigma_t", "sqrt(tau_M2-tau_M1**2)", "seconds"],
            ["t10_over_T", "t10/T", ""],
            ["t25_over_T", "t25/T", ""],
            ["t50_over_T", "t50/T", ""],
            ["t75_over_T", "t75/T", ""],
            ["t90_over_T", "t90/T", ""],
            ["E_low_over_E_total", "sum(|Vn|**2,n<=k)/sum(|Vn|**2)", ""],
            ["E_high_over_E_total", "sum(|Vn|**2,n>k)/sum(|Vn|**2)", ""],
            # NEW requested metrics
            ["tauH", "sum(wn*(1/omega_n)*sqrt(1/|Xn|**2-1))/sum(wn)", "seconds"],
            ["crest_factor", "max(vb(t))/rms(vb(t))", ""],
            ["spectral_entropy", "-sum(pn*log(pn+eps))", ""],
            ["phi1", "angle(V1)", "rad"],
            ["phi2", "angle(V2)", "rad"],
            ["phi3", "angle(V3)", "rad"],
            ["delta_phi2", "wrap(phi2-2*phi1)", "rad"],
            ["delta_phi3", "wrap(phi3-3*phi1)", "rad"],
            ["AUC_cumsum", "measuring the area above the y=x function", ""],
            [
                "Tan_cumsum",
                "measuring the tangent coeff director by a linear regression of the four first points",
                "",
            ],
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

                # Branch aggregates: median over radii
                for k in self._metric_keys():
                    br[k[0]][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[k[0]], dtype=float)
                    )

            # Global aggregates: mean over all branches & radii
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

    # -------------------------
    # Pipeline entrypoint
    # -------------------------
    def run(self, h5file) -> ProcessResult:
        latex_formulas = {
            # ---- Temporal centroid ----
            "mu_t": r"$\mu = \frac{\sum_t w(t)\,t}{\sum_t w(t)}$",
            "mu_t_over_T": r"$\frac{\mu}{T}$",
            # ---- Doppler indices ----
            "RI": r"$\frac{V_{systole}-V_{diastole}}{V_{systole}}$",
            "PI": r"$\frac{V_{systole}-V_{diastole}}{V_{mean}}$",
            # ---- VTI metrics ----
            "R_VTI": r"$\frac{VTI_{0\rightarrow T/2}}{VTI_{T/2\rightarrow T}}$",
            "SF_VTI": r"$\frac{VTI_{0\rightarrow T/2}}{VTI_{0\rightarrow T}}$",
            # ---- Temporal dispersion ----
            "sigma_t": r"$\sqrt{\tau_{M,2}-\tau_{M,1}^{2}}$",
            "sigma_t_over_T": r"$\frac{\sigma}{T}$",
            # ---- Percentile timings ----
            "t10_over_T": r"$\frac{t_{10}}{T}$",
            "t25_over_T": r"$\frac{t_{25}}{T}$",
            "t50_over_T": r"$\frac{t_{50}}{T}$",
            "t75_over_T": r"$\frac{t_{75}}{T}$",
            "t90_over_T": r"$\frac{t_{90}}{T}$",
            # ---- Harmonic energy partitions ----
            "E_low_over_E_total": r"$\frac{\sum_{n=1}^{k}|V_n|^2}{\sum_{n=1}^{N}|V_n|^2}$",
            "E_high_over_E_total": r"$\frac{\sum_{n=k+1}^{N}|V_n|^2}{\sum_{n=1}^{N}|V_n|^2}$",
            # ---- Harmonic damping ----
            "tauH": r"$\tau_H=\frac{\sum_{n=2}^{N} w_n \frac{1}{\omega_n}\sqrt{\frac{1}{|X_n|^2}-1}}{\sum_{n=2}^{N} w_n}$",
            # ---- Shape sharpness ----
            "crest_factor": r"$\frac{\max_t v_b(t)}{\mathrm{RMS}(v_b(t))}$",
            # ---- Spectral entropy ----
            "spectral_entropy": r"$-\sum_{n=1}^{N} p_n\log(p_n+\epsilon)$",
            # ---- Harmonic phases ----
            "phi1": r"$\arg(V_1)$",
            "phi2": r"$\arg(V_2)$",
            "phi3": r"$\arg(V_3)$",
            # ---- Phase coupling ----
            "delta_phi2": r"$\mathrm{wrap}(\phi_2-2\phi_1)$",
            "delta_phi3": r"$\mathrm{wrap}(\phi_3-3\phi_1)$",
            "AUC_cumsum": r"$AUC$",
            "Tan_cumsum": r"$Tan$",
        }
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

            def pack(prefix: str, d: dict, attrs_common: dict):
                for k, arr in d.items():
                    metrics[f"{prefix}/{k}"] = with_attrs(arr, attrs_common)

            # Per-segment outputs
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
            metrics["by_segment/params/ratio_R_VTI"] = np.asarray(
                self.ratio_R_VTI, dtype=float
            )
            metrics["by_segment/params/ratio_SF_VTI"] = np.asarray(
                self.ratio_SF_VTI, dtype=float
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

            # provenance
            metrics["global/params/ratio_R_VTI"] = (
                np.asarray(self.ratio_R_VTI, dtype=float),
            )
            metrics["global/params/ratio_SF_VTI"] = np.asarray(
                self.ratio_SF_VTI, dtype=float
            )
            metrics["global/params/eps"] = np.asarray(self.eps, dtype=float)
            metrics["global/params/H_LOW_MAX"] = np.asarray(self.H_LOW_MAX, dtype=int)
            metrics["global/params/H_HIGH_MIN"] = np.asarray(self.H_HIGH_MIN, dtype=int)
            metrics["global/params/H_HIGH_MAX"] = np.asarray(self.H_HIGH_MAX, dtype=int)
            metrics["global/params/H_MAX"] = np.asarray(self.H_MAX, dtype=int)

        return ProcessResult(metrics=metrics)
