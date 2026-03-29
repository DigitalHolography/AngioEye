
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="Windkessel_RC")
class WindkesselRC(ProcessPipeline):
    """
    Beat-resolved artery-vein Windkessel RC analysis from global arterial and venous
    velocity waveforms.

    The pipeline computes per-beat artery-vein delay (Deltat) and RC time constant (tau)
    with several complementary estimators:

      1) frequency-domain joint identification
      2) time-domain integral (derivative-free)
      3) discrete-time ARX one-pole fit

    Notes
    -----
    - Inputs are the global arterial and venous per-beat velocity waveforms.
    - The implementation is shape-first: it estimates delay/RC from waveform dynamics.
      Because the inputs are velocities rather than volumetric flows, method-specific
      gain factors are estimated or per-beat normalization is applied where needed.
    """

    description = (
        "Beat-resolved artery-vein Windkessel RC analysis from global arterial and venous "
        "waveforms using frequency-domain, time-domain integral, and ARX estimators."
    )

    # Requested inputs
    v_raw_global_input_artery = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input_artery = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )
    v_raw_global_input_vein = "/Vein/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input_vein = (
        "/Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )

    # Beat period input used elsewhere in the repo as well
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    eps = 1e-12

    # Frequency-domain configuration
    harmonic_indices = (1, 2, 3)
    delay_min_seconds = -0.150
    delay_max_seconds = 0.150
    delay_grid_step_seconds = 0.002
    use_gain_in_frequency_fit = True

    # Time-domain integral configuration
    time_grid_step_seconds = 0.002

    # ARX configuration
    arx_delay_step_samples = 1
    arx_delay_max_fraction_of_cycle = 0.25
    arx_a_min = 1e-4
    arx_a_max = 0.9999

    # General QC / preprocessing
    min_valid_fraction = 0.80
    use_mean_normalization_time_domain = True
    use_mean_normalization_arx = True

    @staticmethod
    def _safe_nanmedian(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmedian(x))

    @staticmethod
    def _safe_nanmean(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmean(x))

    @staticmethod
    def _safe_nanstd(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanstd(x))

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        med = float(np.nanmedian(x))
        return float(np.nanmedian(np.abs(x - med)))

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
        if not np.isfinite(x):
            return np.nan
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _valid_mask_fraction(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.isfinite(x)))

    def _prepare_beat(self, x: np.ndarray) -> np.ndarray:
        """
        Replace NaNs by linear interpolation when possible. If too few valid points remain,
        returns all-NaN.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n == 0:
            return x

        mask = np.isfinite(x)
        if np.mean(mask) < self.min_valid_fraction:
            return np.full_like(x, np.nan, dtype=float)

        if np.all(mask):
            return x.astype(float, copy=True)

        idx = np.arange(n, dtype=float)
        xout = x.astype(float, copy=True)
        xout[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        return xout

    def _normalize_if_needed(self, x: np.ndarray, use_mean_normalization: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not np.any(np.isfinite(x)):
            return np.full_like(x, np.nan, dtype=float)

        y = x.copy()
        if use_mean_normalization:
            mu = float(np.nanmean(y))
            if not np.isfinite(mu) or abs(mu) <= self.eps:
                return np.full_like(y, np.nan, dtype=float)
            y = y / mu
        return y

    def _harmonic_coeff(self, x: np.ndarray, n: int) -> complex:
        """
        Discrete Fourier-series coefficient at harmonic index n on one beat sampled
        uniformly over phase.
        """
        x = np.asarray(x, dtype=float).ravel()
        m = x.size
        if m < 2:
            return np.nan + 1j * np.nan
        if not np.any(np.isfinite(x)):
            return np.nan + 1j * np.nan

        xx = np.where(np.isfinite(x), x, 0.0)
        grid = np.arange(m, dtype=float)
        coeff = np.sum(xx * np.exp(-1j * 2.0 * np.pi * n * grid / m)) / float(m)
        return complex(coeff)

    def _frequency_fit_one_beat(self, qa: np.ndarray, qv: np.ndarray, Tbeat: float) -> dict:
        """
        Joint delay + tau frequency-domain identification on one beat.

        Returns
        -------
        dict with keys:
            accepted, Deltat, tau, k, residual, harmonics_used, tau_phase_median,
            tau_amp_median
        """
        out = {
            "accepted": False,
            "Deltat": np.nan,
            "tau": np.nan,
            "k": np.nan,
            "residual": np.nan,
            "harmonics_used": 0,
            "tau_phase_median": np.nan,
            "tau_amp_median": np.nan,
        }

        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        qa = self._prepare_beat(qa)
        qv = self._prepare_beat(qv)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        # Demean for harmonic transfer identification.
        qa0 = qa - np.nanmean(qa)
        qv0 = qv - np.nanmean(qv)

        Hn = []
        omegas = []
        weights = []
        used_n = []

        for n in self.harmonic_indices:
            QAn = self._harmonic_coeff(qa0, n)
            QVn = self._harmonic_coeff(qv0, n)
            if (not np.isfinite(QAn.real)) or (not np.isfinite(QAn.imag)):
                continue
            if (not np.isfinite(QVn.real)) or (not np.isfinite(QVn.imag)):
                continue
            if abs(QAn) <= self.eps:
                continue

            Hn_val = QVn / QAn
            omega_n = 2.0 * np.pi * float(n) / float(Tbeat)
            w = float(abs(QAn) ** 2)
            if omega_n <= 0 or (not np.isfinite(w)) or w <= 0:
                continue

            Hn.append(Hn_val)
            omegas.append(omega_n)
            weights.append(w)
            used_n.append(n)

        if len(Hn) < 2:
            return out

        Hn = np.asarray(Hn, dtype=np.complex128)
        omegas = np.asarray(omegas, dtype=float)
        weights = np.asarray(weights, dtype=float)

        delays = np.arange(
            float(self.delay_min_seconds),
            float(self.delay_max_seconds) + 0.5 * float(self.delay_grid_step_seconds),
            float(self.delay_grid_step_seconds),
            dtype=float,
        )
        if delays.size == 0:
            return out

        best = None
        for delay in delays:
            Htilde = Hn * np.exp(1j * omegas * delay)

            if self.use_gain_in_frequency_fit:
                def residual_for_tau_and_k(tau_val: float) -> tuple[float, float]:
                    model_no_gain = 1.0 / (1.0 + 1j * omegas * tau_val)
                    denom = np.sum(weights * np.abs(model_no_gain) ** 2)
                    if (not np.isfinite(denom)) or denom <= self.eps:
                        return np.nan, np.nan
                    num = np.sum(weights * np.real(Htilde * np.conjugate(model_no_gain)))
                    k_hat = float(max(num / denom, 0.0))
                    res = np.sum(weights * np.abs(Htilde - k_hat * model_no_gain) ** 2)
                    return float(np.real_if_close(res)), k_hat
            else:
                def residual_for_tau_and_k(tau_val: float) -> tuple[float, float]:
                    model = 1.0 / (1.0 + 1j * omegas * tau_val)
                    res = np.sum(weights * np.abs(Htilde - model) ** 2)
                    return float(np.real_if_close(res)), 1.0

            # Closed-form tau update for k=1 case, used as candidate even when k is enabled.
            a = 1j * omegas * Htilde
            b = Htilde
            denom_tau = np.sum(weights * np.abs(a) ** 2)
            if (not np.isfinite(denom_tau)) or denom_tau <= self.eps:
                continue
            tau_closed = np.real(np.sum(weights * np.conjugate(a) * (1.0 - b))) / denom_tau
            tau_closed = float(max(tau_closed, 0.0))

            # Also probe a small local set around the closed-form candidate for robustness
            # when optional gain is enabled.
            tau_candidates = np.asarray(
                [
                    0.0,
                    tau_closed,
                    max(0.0, tau_closed * 0.5),
                    tau_closed * 1.5,
                    tau_closed + 0.005,
                    max(0.0, tau_closed - 0.005),
                ],
                dtype=float,
            )
            tau_candidates = np.unique(np.clip(tau_candidates, 0.0, None))

            local_best = None
            for tau_val in tau_candidates:
                res, k_hat = residual_for_tau_and_k(float(tau_val))
                if (not np.isfinite(res)) or (not np.isfinite(k_hat)):
                    continue
                item = (res, delay, float(tau_val), float(k_hat))
                if (local_best is None) or (item[0] < local_best[0]):
                    local_best = item

            if local_best is None:
                continue
            if (best is None) or (local_best[0] < best[0]):
                best = local_best

        if best is None:
            return out

        _, best_delay, best_tau, best_k = best

        # Per-harmonic diagnostic tau estimators after delay correction and gain removal
        Hcorr = Hn * np.exp(1j * omegas * best_delay)
        if np.isfinite(best_k) and abs(best_k) > self.eps:
            Hcorr = Hcorr / best_k

        tau_phase = []
        tau_amp = []
        for Hc, omega_n in zip(Hcorr, omegas):
            phi = self._wrap_pi(float(np.angle(Hc)))
            mag = float(abs(Hc))
            if np.isfinite(phi):
                tanphi = np.tan(-phi)
                if np.isfinite(tanphi):
                    tau_phase.append(float(max(tanphi / omega_n, 0.0)))
            if np.isfinite(mag) and 0 < mag <= 1.0:
                val = (1.0 / (mag * mag)) - 1.0
                if np.isfinite(val) and val >= 0:
                    tau_amp.append(float(np.sqrt(val) / omega_n))

        out.update(
            {
                "accepted": True,
                "Deltat": float(best_delay),
                "tau": float(best_tau),
                "k": float(best_k),
                "residual": float(best[0]),
                "harmonics_used": int(len(used_n)),
                "tau_phase_median": self._safe_nanmedian(np.asarray(tau_phase, dtype=float)),
                "tau_amp_median": self._safe_nanmedian(np.asarray(tau_amp, dtype=float)),
            }
        )
        return out

    def _cumtrapz_uniform(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return x
        out = np.zeros_like(x, dtype=float)
        if x.size == 1:
            return out
        increments = 0.5 * (x[1:] + x[:-1]) * dt
        out[1:] = np.cumsum(increments)
        return out

    def _shift_signal_periodic(self, x: np.ndarray, delay_seconds: float, Tbeat: float) -> np.ndarray:
        """
        Periodic shift of one beat by delay_seconds using Fourier phase shift.
        Positive delay means x(t - delay).
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2 or (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.full_like(x, np.nan, dtype=float)
        if not np.any(np.isfinite(x)):
            return np.full_like(x, np.nan, dtype=float)

        xx = np.where(np.isfinite(x), x, np.nanmean(x))
        X = np.fft.rfft(xx)
        freqs = np.fft.rfftfreq(n, d=Tbeat / n)
        phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_seconds)
        y = np.fft.irfft(X * phase, n=n)
        return np.asarray(y, dtype=float)

    def _time_integral_fit_one_beat(self, qa: np.ndarray, qv: np.ndarray, Tbeat: float) -> dict:
        out = {
            "accepted": False,
            "Deltat": np.nan,
            "tau": np.nan,
            "residual": np.nan,
        }
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        qa = self._prepare_beat(qa)
        qv = self._prepare_beat(qv)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        qa = self._normalize_if_needed(qa, self.use_mean_normalization_time_domain)
        qv = self._normalize_if_needed(qv, self.use_mean_normalization_time_domain)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        n = qa.size
        if n < 3:
            return out
        dt = Tbeat / n

        sv = self._cumtrapz_uniform(qv, dt)
        qv0 = float(qv[0])

        delays = np.arange(
            float(self.delay_min_seconds),
            float(self.delay_max_seconds) + 0.5 * float(self.time_grid_step_seconds),
            float(self.time_grid_step_seconds),
            dtype=float,
        )
        if delays.size == 0:
            return out

        best = None
        for delay in delays:
            qash = self._shift_signal_periodic(qa, delay, Tbeat)
            if not np.any(np.isfinite(qash)):
                continue
            sa = self._cumtrapz_uniform(qash, dt)

            x = qv - qv0
            y = sa - sv

            denom = np.sum(x * x)
            if (not np.isfinite(denom)) or denom <= self.eps:
                continue

            tau_hat = float(np.sum(x * y) / denom)
            tau_hat = max(tau_hat, 0.0)

            res = float(np.sum((y - tau_hat * x) ** 2))
            item = (res, delay, tau_hat)
            if (best is None) or (item[0] < best[0]):
                best = item

        if best is None:
            return out

        out.update(
            {
                "accepted": True,
                "Deltat": float(best[1]),
                "tau": float(best[2]),
                "residual": float(best[0]),
            }
        )
        return out

    def _arx_fit_one_beat(self, qa: np.ndarray, qv: np.ndarray, Tbeat: float) -> dict:
        out = {
            "accepted": False,
            "Deltat": np.nan,
            "tau": np.nan,
            "a": np.nan,
            "b": np.nan,
            "residual": np.nan,
        }
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        qa = self._prepare_beat(qa)
        qv = self._prepare_beat(qv)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        qa = self._normalize_if_needed(qa, self.use_mean_normalization_arx)
        qv = self._normalize_if_needed(qv, self.use_mean_normalization_arx)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        n = qa.size
        if n < 5:
            return out
        dt = Tbeat / n

        dmax = int(max(1, np.floor(self.arx_delay_max_fraction_of_cycle * n)))
        delays = range(-dmax, dmax + 1, int(self.arx_delay_step_samples))

        best = None
        for d in delays:
            rows = []
            y = []
            for idx in range(1, n):
                src = idx - d
                if src < 0 or src >= n:
                    continue
                rows.append([qv[idx - 1], qa[src]])
                y.append(qv[idx])

            if len(rows) < 3:
                continue

            Phi = np.asarray(rows, dtype=float)
            yv = np.asarray(y, dtype=float)

            try:
                theta, *_ = np.linalg.lstsq(Phi, yv, rcond=None)
            except np.linalg.LinAlgError:
                continue

            a_hat = float(np.clip(theta[0], self.arx_a_min, self.arx_a_max))
            b_hat = float(theta[1])

            res = float(np.sum((yv - (a_hat * Phi[:, 0] + b_hat * Phi[:, 1])) ** 2))
            tau_hat = -dt / np.log(a_hat)
            delay_hat = d * dt

            if (not np.isfinite(tau_hat)) or tau_hat < 0:
                continue

            item = (res, delay_hat, tau_hat, a_hat, b_hat)
            if (best is None) or (item[0] < best[0]):
                best = item

        if best is None:
            return out

        out.update(
            {
                "accepted": True,
                "Deltat": float(best[1]),
                "tau": float(best[2]),
                "a": float(best[3]),
                "b": float(best[4]),
                "residual": float(best[0]),
            }
        )
        return out

    def _analyze_representation(
        self, qa_block: np.ndarray, qv_block: np.ndarray, T: np.ndarray, representation_name: str
    ) -> dict:
        """
        Analyze one waveform representation (raw or bandlimited) beat by beat.
        """
        T = np.asarray(T, dtype=float)
        if T.ndim == 2 and T.shape[0] == 1:
            Tvec = T[0]
        else:
            Tvec = T.ravel()

        n_beats = int(Tvec.size)
        qa_block = self._ensure_time_by_beat(qa_block, n_beats)
        qv_block = self._ensure_time_by_beat(qv_block, n_beats)

        if qa_block.shape != qv_block.shape:
            raise ValueError(
                f"Artery/vein waveform shape mismatch for {representation_name}: "
                f"{qa_block.shape} vs {qv_block.shape}"
            )

        freq_delay = np.full((n_beats,), np.nan, dtype=float)
        freq_tau = np.full((n_beats,), np.nan, dtype=float)
        freq_k = np.full((n_beats,), np.nan, dtype=float)
        freq_res = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_phase = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_amp = np.full((n_beats,), np.nan, dtype=float)
        freq_ok = np.zeros((n_beats,), dtype=int)

        td_delay = np.full((n_beats,), np.nan, dtype=float)
        td_tau = np.full((n_beats,), np.nan, dtype=float)
        td_res = np.full((n_beats,), np.nan, dtype=float)
        td_ok = np.zeros((n_beats,), dtype=int)

        arx_delay = np.full((n_beats,), np.nan, dtype=float)
        arx_tau = np.full((n_beats,), np.nan, dtype=float)
        arx_a = np.full((n_beats,), np.nan, dtype=float)
        arx_b = np.full((n_beats,), np.nan, dtype=float)
        arx_res = np.full((n_beats,), np.nan, dtype=float)
        arx_ok = np.zeros((n_beats,), dtype=int)

        for beat_idx in range(n_beats):
            qa = np.asarray(qa_block[:, beat_idx], dtype=float)
            qv = np.asarray(qv_block[:, beat_idx], dtype=float)
            Tbeat = float(Tvec[beat_idx])

            fr = self._frequency_fit_one_beat(qa, qv, Tbeat)
            if fr["accepted"]:
                freq_ok[beat_idx] = 1
                freq_delay[beat_idx] = fr["Deltat"]
                freq_tau[beat_idx] = fr["tau"]
                freq_k[beat_idx] = fr["k"]
                freq_res[beat_idx] = fr["residual"]
                freq_tau_phase[beat_idx] = fr["tau_phase_median"]
                freq_tau_amp[beat_idx] = fr["tau_amp_median"]

            td = self._time_integral_fit_one_beat(qa, qv, Tbeat)
            if td["accepted"]:
                td_ok[beat_idx] = 1
                td_delay[beat_idx] = td["Deltat"]
                td_tau[beat_idx] = td["tau"]
                td_res[beat_idx] = td["residual"]

            arx = self._arx_fit_one_beat(qa, qv, Tbeat)
            if arx["accepted"]:
                arx_ok[beat_idx] = 1
                arx_delay[beat_idx] = arx["Deltat"]
                arx_tau[beat_idx] = arx["tau"]
                arx_a[beat_idx] = arx["a"]
                arx_b[beat_idx] = arx["b"]
                arx_res[beat_idx] = arx["residual"]

        return {
            "representation": representation_name,
            "freq": {
                "Deltat": freq_delay,
                "tau": freq_tau,
                "k": freq_k,
                "residual": freq_res,
                "tau_phase_median": freq_tau_phase,
                "tau_amp_median": freq_tau_amp,
                "accepted": freq_ok,
            },
            "time_integral": {
                "Deltat": td_delay,
                "tau": td_tau,
                "residual": td_res,
                "accepted": td_ok,
            },
            "arx": {
                "Deltat": arx_delay,
                "tau": arx_tau,
                "a": arx_a,
                "b": arx_b,
                "residual": arx_res,
                "accepted": arx_ok,
            },
        }

    def _summary_scalars(self, x: np.ndarray, prefix: str) -> dict:
        x = np.asarray(x, dtype=float)
        return {
            f"{prefix}/median": np.asarray(self._safe_nanmedian(x), dtype=float),
            f"{prefix}/mean": np.asarray(self._safe_nanmean(x), dtype=float),
            f"{prefix}/std": np.asarray(self._safe_nanstd(x), dtype=float),
            f"{prefix}/mad": np.asarray(self._mad(x), dtype=float),
            f"{prefix}/n_valid": np.asarray(int(np.sum(np.isfinite(x))), dtype=int),
        }

    def _pack_method_outputs(self, metrics: dict, representation: str, method_name: str, result: dict) -> None:
        base = f"{representation}/{method_name}"

        metrics[f"{base}/Deltat"] = with_attrs(
            np.asarray(result["Deltat"], dtype=float),
            {
                "unit": ["seconds"],
                "definition": [
                    "Beat-resolved artery-to-vein delay estimated from the selected method."
                ],
            },
        )
        metrics[f"{base}/tau"] = with_attrs(
            np.asarray(result["tau"], dtype=float),
            {
                "unit": ["seconds"],
                "definition": [
                    "Beat-resolved Windkessel RC time constant estimated from the selected method."
                ],
            },
        )
        metrics[f"{base}/accepted"] = with_attrs(
            np.asarray(result["accepted"], dtype=int),
            {
                "definition": ["1 if the beat passed the method-specific fit/QC, else 0."],
            },
        )

        if "residual" in result:
            metrics[f"{base}/residual"] = np.asarray(result["residual"], dtype=float)
        if "k" in result:
            metrics[f"{base}/k"] = np.asarray(result["k"], dtype=float)
        if "tau_phase_median" in result:
            metrics[f"{base}/tau_phase_median"] = np.asarray(
                result["tau_phase_median"], dtype=float
            )
        if "tau_amp_median" in result:
            metrics[f"{base}/tau_amp_median"] = np.asarray(
                result["tau_amp_median"], dtype=float
            )
        if "a" in result:
            metrics[f"{base}/a"] = np.asarray(result["a"], dtype=float)
        if "b" in result:
            metrics[f"{base}/b"] = np.asarray(result["b"], dtype=float)

        for key, value in self._summary_scalars(result["Deltat"], f"{base}/summary/Deltat").items():
            metrics[key] = value
        for key, value in self._summary_scalars(result["tau"], f"{base}/summary/tau").items():
            metrics[key] = value

    def run(self, h5file) -> ProcessResult:
        if self.T_input not in h5file:
            raise ValueError(
                f"Missing beat period input required by Windkessel_RC: {self.T_input}"
            )

        T = np.asarray(h5file[self.T_input], dtype=float)
        n_beats = int(T.shape[1]) if T.ndim == 2 else int(T.size)

        required_inputs = {
            "raw": (
                self.v_raw_global_input_artery,
                self.v_raw_global_input_vein,
            ),
            "bandlimited": (
                self.v_band_global_input_artery,
                self.v_band_global_input_vein,
            ),
        }

        metrics: dict = {}

        for rep_name, (qa_path, qv_path) in required_inputs.items():
            if qa_path not in h5file or qv_path not in h5file:
                continue

            qa = np.asarray(h5file[qa_path], dtype=float)
            qv = np.asarray(h5file[qv_path], dtype=float)

            rep_result = self._analyze_representation(qa, qv, T, rep_name)

            for method_name in ("freq", "time_integral", "arx"):
                self._pack_method_outputs(
                    metrics,
                    representation=rep_name,
                    method_name=method_name,
                    result=rep_result[method_name],
                )

        metrics["params/harmonic_indices"] = np.asarray(self.harmonic_indices, dtype=int)
        metrics["params/delay_min_seconds"] = np.asarray(
            self.delay_min_seconds, dtype=float
        )
        metrics["params/delay_max_seconds"] = np.asarray(
            self.delay_max_seconds, dtype=float
        )
        metrics["params/delay_grid_step_seconds"] = np.asarray(
            self.delay_grid_step_seconds, dtype=float
        )
        metrics["params/time_grid_step_seconds"] = np.asarray(
            self.time_grid_step_seconds, dtype=float
        )
        metrics["params/arx_delay_max_fraction_of_cycle"] = np.asarray(
            self.arx_delay_max_fraction_of_cycle, dtype=float
        )
        metrics["params/arx_a_min"] = np.asarray(self.arx_a_min, dtype=float)
        metrics["params/arx_a_max"] = np.asarray(self.arx_a_max, dtype=float)
        metrics["params/use_gain_in_frequency_fit"] = np.asarray(
            int(self.use_gain_in_frequency_fit), dtype=int
        )
        metrics["params/use_mean_normalization_time_domain"] = np.asarray(
            int(self.use_mean_normalization_time_domain), dtype=int
        )
        metrics["params/use_mean_normalization_arx"] = np.asarray(
            int(self.use_mean_normalization_arx), dtype=int
        )
        metrics["params/n_beats"] = np.asarray(n_beats, dtype=int)

        return ProcessResult(metrics=metrics)
