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
      - Delta_phi1, Delta_phi2 as phase-coupling:
            Delta_phi1 = wrap(phi2 - 2*phi1)   (aka Δϕ2)
            Delta_phi2 = wrap(phi3 - 3*phi1)   (aka Δϕ3)
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

        fs = n / Tbeat  # Hz
        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        f = np.fft.rfftfreq(n, d=1.0 / fs)  # Hz
        h = f * Tbeat  # harmonic index (cycles/beat)

        E_total = float(np.sum(P))
        if not np.isfinite(E_total) or E_total <= 0:
            return np.nan, np.nan

        low_mask = (h >= 1.0) & (h <= float(self.H_LOW_MAX))
        high_mask = (h >= float(self.H_HIGH_MIN)) & (h <= float(self.H_HIGH_MAX))

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

        vv = np.where(np.isfinite(v), v, 0.0)
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
        if V is None or (not np.isfinite(Vmax)) or Vmax <= 0 or (not np.isfinite(omega0)) or omega0 <= 0:
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
        mags = np.where(np.isfinite(mags), mags, 0.0)
        s = float(np.sum(mags))
        if s <= 0:
            return np.nan

        p = mags / s
        p = np.clip(p, self.eps, 1.0)
        return float(-np.sum(p * np.log(p)))

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
        x = np.where(np.isfinite(vb), vb, 0.0)
        rms = float(np.sqrt(np.mean(x * x)))
        if rms <= 0:
            return np.nan
        return float(np.max(x) / rms)

    def _harmonic_phases(self, V: np.ndarray) -> dict:
        """
        Return phi1,phi2,phi3 and Delta_phi1,Delta_phi2 (phase-coupling wrt fundamental).
        Delta_phi1 = wrap(phi2 - 2*phi1)
        Delta_phi2 = wrap(phi3 - 3*phi1)
        """
        out = {
            "phi1": np.nan,
            "phi2": np.nan,
            "phi3": np.nan,
            "Delta_phi1": np.nan,
            "Delta_phi2": np.nan,
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
            out["Delta_phi1"] = self._wrap_pi(phi2 - 2.0 * phi1)
        if np.isfinite(phi1) and np.isfinite(phi3):
            out["Delta_phi2"] = self._wrap_pi(phi3 - 3.0 * phi1)

        return out

    def _compute_metrics_1d(self, v: np.ndarray, Tbeat: float) -> dict:
        """
        Canonical metric kernel: compute all waveform-shape metrics from a single 1D waveform v(t).
        Returns a dict of scalar metrics (floats).
        """
        v = self._rectify_keep_nan(v)
        n = int(v.size)
        if n <= 0:
            return {k: np.nan for k in self._metric_keys()}

        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return {k: np.nan for k in self._metric_keys()}

        vv = np.where(np.isfinite(v), v, np.nan)
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

        # RVTI (split 1/2)
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

        # Central moments around tau_M1
        dtau = t - tau_M1
        mu2 = float(np.nansum(vv * (dtau**2)))
        tau_M2 = np.sqrt(mu2 / m0 + self.eps)
        tau_M2_over_T = tau_M2 / Tbeat

        # Quantile timing features
        t10_over_T = self._quantile_time_over_T(vv, Tbeat, 0.10)
        t25_over_T = self._quantile_time_over_T(vv, Tbeat, 0.25)
        t50_over_T = self._quantile_time_over_T(vv, Tbeat, 0.50)
        t75_over_T = self._quantile_time_over_T(vv, Tbeat, 0.75)
        t90_over_T = self._quantile_time_over_T(vv, Tbeat, 0.90)

        # Spectral ratios (FFT power bands)
        E_low_over_E_total, E_high_over_E_total = self._spectral_ratios(vv, Tbeat)

        # Harmonic-domain extras (n=0..10 synthesis; n=1..10 metrics)
        hp = self._harmonic_pack(np.where(np.isfinite(vv), vv, 0.0), Tbeat)
        V = hp["V"]
        vb = hp["vb"]
        Vmax_bl = hp["Vmax"]
        omega0 = hp["omega0"]

        tauH = self._tauH_from_harmonics(V, Vmax_bl, omega0)
        crest_factor = self._crest_factor_from_vb(vb)
        spectral_entropy = self._spectral_entropy_from_harmonics(V)
        ph = self._harmonic_phases(V)

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

            # NEW harmonic-domain requested metrics
            "tauH": float(tauH) if np.isfinite(tauH) else np.nan,
            "crest_factor": float(crest_factor) if np.isfinite(crest_factor) else np.nan,
            "spectral_entropy": float(spectral_entropy) if np.isfinite(spectral_entropy) else np.nan,
            "phi1": float(ph["phi1"]) if np.isfinite(ph["phi1"]) else np.nan,
            "phi2": float(ph["phi2"]) if np.isfinite(ph["phi2"]) else np.nan,
            "phi3": float(ph["phi3"]) if np.isfinite(ph["phi3"]) else np.nan,
            "Delta_phi1": float(ph["Delta_phi1"]) if np.isfinite(ph["Delta_phi1"]) else np.nan,
            "Delta_phi2": float(ph["Delta_phi2"]) if np.isfinite(ph["Delta_phi2"]) else np.nan,
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

            # NEW requested metrics
            "tauH",
            "crest_factor",
            "spectral_entropy",
            "phi1",
            "phi2",
            "phi3",
            "Delta_phi1",
            "Delta_phi2",
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

            gl_vals = {k: [] for k in self._metric_keys()}

            for branch_idx in range(n_branches):
                br_vals = {k: [] for k in self._metric_keys()}

                for radius_idx in range(n_radii):
                    v = v_block[:, beat_idx, branch_idx, radius_idx]
                    m = self._compute_metrics_1d(v, Tbeat)

                    seg_idx = branch_idx * n_radii + radius_idx
                    for k in self._metric_keys():
                        seg[k][beat_idx, seg_idx] = m[k]
                        br_vals[k].append(m[k])
                        gl_vals[k].append(m[k])

                # Branch aggregates: median over radii
                for k in self._metric_keys():
                    br[k][beat_idx, branch_idx] = self._safe_nanmedian(
                        np.asarray(br_vals[k], dtype=float)
                    )

            # Global aggregates: mean over all branches & radii
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
            metrics["global/params/H_MAX"] = np.asarray(self.H_MAX, dtype=int)

        return ProcessResult(metrics=metrics)
