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

    In addition to the estimates themselves, the pipeline emits low-level QC primitives
    intended to support downstream batch QC in src/postprocess:
      - input/prepared valid fractions
      - per-beat fit residuals / normalized residuals
      - harmonic support and frequency self-consistency diagnostics
      - ARX stability and effective row-count diagnostics
      - cross-method consensus and disagreement primitives
      - compact per-representation summary fractions
    """

    description = (
        "Beat-resolved artery-vein Windkessel RC analysis from global arterial and venous "
        "waveforms using frequency-domain, time-domain integral, and ARX estimators, "
        "with intrinsic QC primitives for downstream postprocessing."
    )

    v_raw_global_input_artery = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input_artery = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )
    v_raw_global_input_vein = "/Vein/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_band_global_input_vein = (
        "/Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    eps = 1e-12

    harmonic_indices = (1, 2, 3)
    delay_min_seconds = -0.150
    delay_max_seconds = 0.150
    delay_grid_step_seconds = 0.002
    use_gain_in_frequency_fit = True

    time_grid_step_seconds = 0.002

    arx_delay_step_samples = 1
    arx_delay_max_fraction_of_cycle = 0.25
    arx_a_min = 1e-4
    arx_a_max = 0.9999

    min_valid_fraction = 0.80
    use_mean_normalization_time_domain = True
    use_mean_normalization_arx = True

    primitive_max_abs_delay_seconds = 0.300
    primitive_max_tau_seconds = 5.000
    primitive_min_methods_for_consensus = 2

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

    @staticmethod
    def _safe_rel_diff(a: float, b: float, eps: float = 1e-12) -> float:
        if (not np.isfinite(a)) or (not np.isfinite(b)):
            return np.nan
        return float(abs(a - b) / max(abs(a), abs(b), eps))

    @staticmethod
    def _pairwise_range(vals: np.ndarray) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        return float(np.nanmax(vals) - np.nanmin(vals))

    @staticmethod
    def _pairwise_rel_range(vals: np.ndarray, eps: float = 1e-12) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        med = float(np.nanmedian(vals))
        return float((np.nanmax(vals) - np.nanmin(vals)) / max(abs(med), eps))

    def _prepare_beat(self, x: np.ndarray) -> np.ndarray:
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
        out = x.astype(float, copy=True)
        out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        return out

    def _normalize_if_needed(self, x: np.ndarray, use_mean_normalization: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not np.any(np.isfinite(x)):
            return np.full_like(x, np.nan, dtype=float)
        y = x.copy()
        if use_mean_normalization:
            mu = float(np.nanmean(y))
            if (not np.isfinite(mu)) or abs(mu) <= self.eps:
                return np.full_like(y, np.nan, dtype=float)
            y = y / mu
        return y

    def _harmonic_coeff(self, x: np.ndarray, n: int) -> complex:
        x = np.asarray(x, dtype=float).ravel()
        m = x.size
        if m < 2 or not np.any(np.isfinite(x)):
            return np.nan + 1j * np.nan
        xx = np.where(np.isfinite(x), x, 0.0)
        grid = np.arange(m, dtype=float)
        coeff = np.sum(xx * np.exp(-1j * 2.0 * np.pi * n * grid / m)) / float(m)
        return complex(coeff)

    def _frequency_fit_one_beat(self, qa: np.ndarray, qv: np.ndarray, Tbeat: float) -> dict:
        out = {
            "accepted": False,
            "Deltat": np.nan,
            "tau": np.nan,
            "k": np.nan,
            "residual": np.nan,
            "residual_norm": np.nan,
            "harmonics_used": 0,
            "harmonic_weight_sum": np.nan,
            "tau_phase_median": np.nan,
            "tau_amp_median": np.nan,
            "tau_phase_rel_diff": np.nan,
            "tau_amp_rel_diff": np.nan,
        }
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return out

        qa = self._prepare_beat(qa)
        qv = self._prepare_beat(qv)
        if not np.any(np.isfinite(qa)) or not np.any(np.isfinite(qv)):
            return out

        qa0 = qa - np.nanmean(qa)
        qv0 = qv - np.nanmean(qv)

        Hn = []
        omegas = []
        weights = []
        for n in self.harmonic_indices:
            QAn = self._harmonic_coeff(qa0, n)
            QVn = self._harmonic_coeff(qv0, n)
            if not (np.isfinite(QAn.real) and np.isfinite(QAn.imag)):
                continue
            if not (np.isfinite(QVn.real) and np.isfinite(QVn.imag)):
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

            a = 1j * omegas * Htilde
            b = Htilde
            denom_tau = np.sum(weights * np.abs(a) ** 2)
            if (not np.isfinite(denom_tau)) or denom_tau <= self.eps:
                continue
            tau_closed = np.real(np.sum(weights * np.conjugate(a) * (1.0 - b))) / denom_tau
            tau_closed = float(max(tau_closed, 0.0))

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
            if (local_best is not None) and ((best is None) or (local_best[0] < best[0])):
                best = local_best

        if best is None:
            return out

        best_residual, best_delay, best_tau, best_k = best
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

        tau_phase_arr = np.asarray(tau_phase, dtype=float)
        tau_amp_arr = np.asarray(tau_amp, dtype=float)
        tau_phase_med = self._safe_nanmedian(tau_phase_arr)
        tau_amp_med = self._safe_nanmedian(tau_amp_arr)
        signal_norm = float(np.sum(weights * np.abs(Hn) ** 2))
        residual_norm = float(best_residual / max(signal_norm, self.eps))

        out.update(
            {
                "accepted": True,
                "Deltat": float(best_delay),
                "tau": float(best_tau),
                "k": float(best_k),
                "residual": float(best_residual),
                "residual_norm": float(residual_norm),
                "harmonics_used": int(Hn.size),
                "harmonic_weight_sum": float(np.sum(weights)),
                "tau_phase_median": tau_phase_med,
                "tau_amp_median": tau_amp_med,
                "tau_phase_rel_diff": self._safe_rel_diff(best_tau, tau_phase_med, self.eps),
                "tau_amp_rel_diff": self._safe_rel_diff(best_tau, tau_amp_med, self.eps),
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
        return np.asarray(np.fft.irfft(X * phase, n=n), dtype=float)

    def _time_integral_fit_one_beat(self, qa: np.ndarray, qv: np.ndarray, Tbeat: float) -> dict:
        out = {
            "accepted": False,
            "Deltat": np.nan,
            "tau": np.nan,
            "residual": np.nan,
            "residual_norm": np.nan,
            "rows_used": np.nan,
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
            tau_hat = float(max(np.sum(x * y) / denom, 0.0))
            res = float(np.sum((y - tau_hat * x) ** 2))
            item = (res, delay, tau_hat, n, denom)
            if (best is None) or (item[0] < best[0]):
                best = item

        if best is None:
            return out

        residual_norm = float(best[0] / max(best[4], self.eps))
        out.update(
            {
                "accepted": True,
                "Deltat": float(best[1]),
                "tau": float(best[2]),
                "residual": float(best[0]),
                "residual_norm": residual_norm,
                "rows_used": int(best[3]),
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
            "residual_norm": np.nan,
            "rows_used": np.nan,
            "stability_margin": np.nan,
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

            a_hat_raw = float(theta[0])
            a_hat = float(np.clip(a_hat_raw, self.arx_a_min, self.arx_a_max))
            b_hat = float(theta[1])
            fit = a_hat * Phi[:, 0] + b_hat * Phi[:, 1]
            res = float(np.sum((yv - fit) ** 2))
            tau_hat = -dt / np.log(a_hat)
            delay_hat = d * dt
            if (not np.isfinite(tau_hat)) or tau_hat < 0:
                continue
            stability_margin = float(min(a_hat - self.arx_a_min, self.arx_a_max - a_hat))
            denom = float(np.sum(yv * yv))
            residual_norm = float(res / max(denom, self.eps))
            item = (res, delay_hat, tau_hat, a_hat, b_hat, residual_norm, len(rows), stability_margin)
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
                "residual_norm": float(best[5]),
                "rows_used": int(best[6]),
                "stability_margin": float(best[7]),
            }
        )
        return out

    def _analyze_representation(self, qa_block: np.ndarray, qv_block: np.ndarray, T: np.ndarray, representation_name: str) -> dict:
        T = np.asarray(T, dtype=float)
        Tvec = T[0] if (T.ndim == 2 and T.shape[0] == 1) else T.ravel()
        n_beats = int(Tvec.size)
        qa_block = self._ensure_time_by_beat(qa_block, n_beats)
        qv_block = self._ensure_time_by_beat(qv_block, n_beats)
        if qa_block.shape != qv_block.shape:
            raise ValueError(
                f"Artery/vein waveform shape mismatch for {representation_name}: {qa_block.shape} vs {qv_block.shape}"
            )

        qa_input_valid_fraction = np.full((n_beats,), np.nan, dtype=float)
        qv_input_valid_fraction = np.full((n_beats,), np.nan, dtype=float)
        qa_prepared_valid_fraction = np.full((n_beats,), np.nan, dtype=float)
        qv_prepared_valid_fraction = np.full((n_beats,), np.nan, dtype=float)

        freq_delay = np.full((n_beats,), np.nan, dtype=float)
        freq_tau = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_over_T = np.full((n_beats,), np.nan, dtype=float)
        freq_k = np.full((n_beats,), np.nan, dtype=float)
        freq_res = np.full((n_beats,), np.nan, dtype=float)
        freq_res_norm = np.full((n_beats,), np.nan, dtype=float)
        freq_harmonics_used = np.full((n_beats,), np.nan, dtype=float)
        freq_harmonic_weight_sum = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_phase = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_amp = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_phase_rel = np.full((n_beats,), np.nan, dtype=float)
        freq_tau_amp_rel = np.full((n_beats,), np.nan, dtype=float)
        freq_ok = np.zeros((n_beats,), dtype=int)

        td_delay = np.full((n_beats,), np.nan, dtype=float)
        td_tau = np.full((n_beats,), np.nan, dtype=float)
        td_tau_over_T = np.full((n_beats,), np.nan, dtype=float)
        td_res = np.full((n_beats,), np.nan, dtype=float)
        td_res_norm = np.full((n_beats,), np.nan, dtype=float)
        td_rows_used = np.full((n_beats,), np.nan, dtype=float)
        td_ok = np.zeros((n_beats,), dtype=int)

        arx_delay = np.full((n_beats,), np.nan, dtype=float)
        arx_tau = np.full((n_beats,), np.nan, dtype=float)
        arx_tau_over_T = np.full((n_beats,), np.nan, dtype=float)
        arx_a = np.full((n_beats,), np.nan, dtype=float)
        arx_b = np.full((n_beats,), np.nan, dtype=float)
        arx_res = np.full((n_beats,), np.nan, dtype=float)
        arx_res_norm = np.full((n_beats,), np.nan, dtype=float)
        arx_rows_used = np.full((n_beats,), np.nan, dtype=float)
        arx_stability_margin = np.full((n_beats,), np.nan, dtype=float)
        arx_ok = np.zeros((n_beats,), dtype=int)

        for beat_idx in range(n_beats):
            qa = np.asarray(qa_block[:, beat_idx], dtype=float)
            qv = np.asarray(qv_block[:, beat_idx], dtype=float)
            Tbeat = float(Tvec[beat_idx])

            qa_input_valid_fraction[beat_idx] = self._valid_mask_fraction(qa)
            qv_input_valid_fraction[beat_idx] = self._valid_mask_fraction(qv)
            qa_prepared = self._prepare_beat(qa)
            qv_prepared = self._prepare_beat(qv)
            qa_prepared_valid_fraction[beat_idx] = self._valid_mask_fraction(qa_prepared)
            qv_prepared_valid_fraction[beat_idx] = self._valid_mask_fraction(qv_prepared)

            fr = self._frequency_fit_one_beat(qa, qv, Tbeat)
            if fr["accepted"]:
                freq_ok[beat_idx] = 1
                freq_delay[beat_idx] = fr["Deltat"]
                freq_tau[beat_idx] = fr["tau"]
                freq_tau_over_T[beat_idx] = (
                    fr["tau"] / Tbeat if np.isfinite(fr["tau"]) and np.isfinite(Tbeat) and Tbeat > 0 else np.nan
                )
                freq_k[beat_idx] = fr["k"]
                freq_res[beat_idx] = fr["residual"]
                freq_res_norm[beat_idx] = fr["residual_norm"]
                freq_harmonics_used[beat_idx] = fr["harmonics_used"]
                freq_harmonic_weight_sum[beat_idx] = fr["harmonic_weight_sum"]
                freq_tau_phase[beat_idx] = fr["tau_phase_median"]
                freq_tau_amp[beat_idx] = fr["tau_amp_median"]
                freq_tau_phase_rel[beat_idx] = fr["tau_phase_rel_diff"]
                freq_tau_amp_rel[beat_idx] = fr["tau_amp_rel_diff"]

            td = self._time_integral_fit_one_beat(qa, qv, Tbeat)
            if td["accepted"]:
                td_ok[beat_idx] = 1
                td_delay[beat_idx] = td["Deltat"]
                td_tau[beat_idx] = td["tau"]
                td_tau_over_T[beat_idx] = (
                    td["tau"] / Tbeat if np.isfinite(td["tau"]) and np.isfinite(Tbeat) and Tbeat > 0 else np.nan
                )
                td_res[beat_idx] = td["residual"]
                td_res_norm[beat_idx] = td["residual_norm"]
                td_rows_used[beat_idx] = td["rows_used"]

            arx = self._arx_fit_one_beat(qa, qv, Tbeat)
            if arx["accepted"]:
                arx_ok[beat_idx] = 1
                arx_delay[beat_idx] = arx["Deltat"]
                arx_tau[beat_idx] = arx["tau"]
                arx_tau_over_T[beat_idx] = (
                    arx["tau"] / Tbeat if np.isfinite(arx["tau"]) and np.isfinite(Tbeat) and Tbeat > 0 else np.nan
                )
                arx_a[beat_idx] = arx["a"]
                arx_b[beat_idx] = arx["b"]
                arx_res[beat_idx] = arx["residual"]
                arx_res_norm[beat_idx] = arx["residual_norm"]
                arx_rows_used[beat_idx] = arx["rows_used"]
                arx_stability_margin[beat_idx] = arx["stability_margin"]

        tau_consensus_median = np.full((n_beats,), np.nan, dtype=float)
        delay_consensus_median = np.full((n_beats,), np.nan, dtype=float)
        tau_intermethod_range = np.full((n_beats,), np.nan, dtype=float)
        tau_intermethod_rel_range = np.full((n_beats,), np.nan, dtype=float)
        delay_intermethod_range = np.full((n_beats,), np.nan, dtype=float)
        methods_valid_count = np.zeros((n_beats,), dtype=int)
        tau_consensus_available = np.zeros((n_beats,), dtype=int)

        for i in range(n_beats):
            tau_vals = np.asarray([freq_tau[i], td_tau[i], arx_tau[i]], dtype=float)
            delay_vals = np.asarray([freq_delay[i], td_delay[i], arx_delay[i]], dtype=float)
            methods_valid_count[i] = int(np.sum(np.isfinite(tau_vals)))
            tau_consensus_median[i] = self._safe_nanmedian(tau_vals)
            delay_consensus_median[i] = self._safe_nanmedian(delay_vals)
            tau_intermethod_range[i] = self._pairwise_range(tau_vals)
            tau_intermethod_rel_range[i] = self._pairwise_rel_range(tau_vals, self.eps)
            delay_intermethod_range[i] = self._pairwise_range(delay_vals)
            tau_consensus_available[i] = int(methods_valid_count[i] >= self.primitive_min_methods_for_consensus)

        freq_tau_reasonable = ((freq_tau > 0) & (freq_tau <= self.primitive_max_tau_seconds)).astype(int)
        td_tau_reasonable = ((td_tau > 0) & (td_tau <= self.primitive_max_tau_seconds)).astype(int)
        arx_tau_reasonable = ((arx_tau > 0) & (arx_tau <= self.primitive_max_tau_seconds)).astype(int)
        freq_delay_reasonable = (np.abs(freq_delay) <= self.primitive_max_abs_delay_seconds).astype(int)
        td_delay_reasonable = (np.abs(td_delay) <= self.primitive_max_abs_delay_seconds).astype(int)
        arx_delay_reasonable = (np.abs(arx_delay) <= self.primitive_max_abs_delay_seconds).astype(int)

        return {
            "representation": representation_name,
            "freq": {
                "Deltat": freq_delay,
                "tau": freq_tau,
                "tau_over_T": freq_tau_over_T,
                "k": freq_k,
                "residual": freq_res,
                "residual_norm": freq_res_norm,
                "harmonics_used": freq_harmonics_used,
                "harmonic_weight_sum": freq_harmonic_weight_sum,
                "tau_phase_median": freq_tau_phase,
                "tau_amp_median": freq_tau_amp,
                "tau_phase_rel_diff": freq_tau_phase_rel,
                "tau_amp_rel_diff": freq_tau_amp_rel,
                "accepted": freq_ok,
            },
            "time_integral": {
                "Deltat": td_delay,
                "tau": td_tau,
                "tau_over_T": td_tau_over_T,
                "residual": td_res,
                "residual_norm": td_res_norm,
                "rows_used": td_rows_used,
                "accepted": td_ok,
            },
            "arx": {
                "Deltat": arx_delay,
                "tau": arx_tau,
                "tau_over_T": arx_tau_over_T,
                "a": arx_a,
                "b": arx_b,
                "residual": arx_res,
                "residual_norm": arx_res_norm,
                "rows_used": arx_rows_used,
                "stability_margin": arx_stability_margin,
                "accepted": arx_ok,
            },
            "qc": {
                "qa_input_valid_fraction": qa_input_valid_fraction,
                "qv_input_valid_fraction": qv_input_valid_fraction,
                "qa_prepared_valid_fraction": qa_prepared_valid_fraction,
                "qv_prepared_valid_fraction": qv_prepared_valid_fraction,
                "cross_method": {
                    "tau_consensus_median": tau_consensus_median,
                    "delay_consensus_median": delay_consensus_median,
                    "tau_intermethod_range": tau_intermethod_range,
                    "tau_intermethod_rel_range": tau_intermethod_rel_range,
                    "delay_intermethod_range": delay_intermethod_range,
                    "methods_valid_count": methods_valid_count,
                    "tau_consensus_available": tau_consensus_available,
                },
                "plausibility": {
                    "freq_tau_reasonable": freq_tau_reasonable,
                    "time_integral_tau_reasonable": td_tau_reasonable,
                    "arx_tau_reasonable": arx_tau_reasonable,
                    "freq_delay_reasonable": freq_delay_reasonable,
                    "time_integral_delay_reasonable": td_delay_reasonable,
                    "arx_delay_reasonable": arx_delay_reasonable,
                },
                "summary": {
                    "freq_tau_reasonable_fraction": np.asarray(self._safe_nanmean(freq_tau_reasonable), dtype=float),
                    "time_integral_tau_reasonable_fraction": np.asarray(self._safe_nanmean(td_tau_reasonable), dtype=float),
                    "arx_tau_reasonable_fraction": np.asarray(self._safe_nanmean(arx_tau_reasonable), dtype=float),
                    "freq_delay_reasonable_fraction": np.asarray(self._safe_nanmean(freq_delay_reasonable), dtype=float),
                    "time_integral_delay_reasonable_fraction": np.asarray(self._safe_nanmean(td_delay_reasonable), dtype=float),
                    "arx_delay_reasonable_fraction": np.asarray(self._safe_nanmean(arx_delay_reasonable), dtype=float),
                    "consensus_available_fraction": np.asarray(self._safe_nanmean(tau_consensus_available), dtype=float),
                    "tau_intermethod_rel_range_median": np.asarray(self._safe_nanmedian(tau_intermethod_rel_range), dtype=float),
                    "delay_intermethod_range_median": np.asarray(self._safe_nanmedian(delay_intermethod_range), dtype=float),
                    "methods_valid_count_median": np.asarray(self._safe_nanmedian(methods_valid_count), dtype=float),
                },
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
        metrics[f"{base}/Deltat"] = with_attrs(np.asarray(result["Deltat"], dtype=float), {"unit": ["seconds"]})
        metrics[f"{base}/tau"] = with_attrs(np.asarray(result["tau"], dtype=float), {"unit": ["seconds"]})
        metrics[f"{base}/tau_over_T"] = with_attrs(np.asarray(result["tau_over_T"], dtype=float), {"unit": [""]})
        metrics[f"{base}/accepted"] = np.asarray(result["accepted"], dtype=int)
        for extra in (
            "residual",
            "residual_norm",
            "k",
            "harmonics_used",
            "harmonic_weight_sum",
            "tau_phase_median",
            "tau_amp_median",
            "tau_phase_rel_diff",
            "tau_amp_rel_diff",
            "rows_used",
            "a",
            "b",
            "stability_margin",
        ):
            if extra in result:
                metrics[f"{base}/{extra}"] = np.asarray(result[extra], dtype=float)
        for key, value in self._summary_scalars(result["Deltat"], f"{base}/summary/Deltat").items():
            metrics[key] = value
        for key, value in self._summary_scalars(result["tau"], f"{base}/summary/tau").items():
            metrics[key] = value
        for key, value in self._summary_scalars(result["tau_over_T"], f"{base}/summary/tau_over_T").items():
            metrics[key] = value
        if "residual_norm" in result:
            for key, value in self._summary_scalars(result["residual_norm"], f"{base}/summary/residual_norm").items():
                metrics[key] = value

    def _pack_qc_outputs(self, metrics: dict, representation: str, qc: dict) -> None:
        base = f"{representation}/qc"
        for key in (
            "qa_input_valid_fraction",
            "qv_input_valid_fraction",
            "qa_prepared_valid_fraction",
            "qv_prepared_valid_fraction",
        ):
            metrics[f"{base}/{key}"] = np.asarray(qc[key], dtype=float)
        for key, arr in qc["cross_method"].items():
            metrics[f"{base}/cross_method/{key}"] = np.asarray(arr)
        for key, arr in qc["plausibility"].items():
            metrics[f"{base}/plausibility/{key}"] = np.asarray(arr, dtype=int)
        for key, val in qc["summary"].items():
            metrics[f"{base}/summary/{key}"] = np.asarray(val)

    def run(self, h5file) -> ProcessResult:
        if self.T_input not in h5file:
            raise ValueError(f"Missing beat period input required by Windkessel_RC: {self.T_input}")

        T = np.asarray(h5file[self.T_input], dtype=float)
        n_beats = int(T.shape[1]) if T.ndim == 2 else int(T.size)
        required_inputs = {
            "raw": (self.v_raw_global_input_artery, self.v_raw_global_input_vein),
            "bandlimited": (self.v_band_global_input_artery, self.v_band_global_input_vein),
        }
        metrics: dict = {}

        for rep_name, (qa_path, qv_path) in required_inputs.items():
            if qa_path not in h5file or qv_path not in h5file:
                continue
            qa = np.asarray(h5file[qa_path], dtype=float)
            qv = np.asarray(h5file[qv_path], dtype=float)
            rep_result = self._analyze_representation(qa, qv, T, rep_name)
            for method_name in ("freq", "time_integral", "arx"):
                self._pack_method_outputs(metrics, rep_name, method_name, rep_result[method_name])
            self._pack_qc_outputs(metrics, rep_name, rep_result["qc"])

        metrics["params/harmonic_indices"] = np.asarray(self.harmonic_indices, dtype=int)
        metrics["params/delay_min_seconds"] = np.asarray(self.delay_min_seconds, dtype=float)
        metrics["params/delay_max_seconds"] = np.asarray(self.delay_max_seconds, dtype=float)
        metrics["params/delay_grid_step_seconds"] = np.asarray(self.delay_grid_step_seconds, dtype=float)
        metrics["params/time_grid_step_seconds"] = np.asarray(self.time_grid_step_seconds, dtype=float)
        metrics["params/arx_delay_max_fraction_of_cycle"] = np.asarray(self.arx_delay_max_fraction_of_cycle, dtype=float)
        metrics["params/arx_a_min"] = np.asarray(self.arx_a_min, dtype=float)
        metrics["params/arx_a_max"] = np.asarray(self.arx_a_max, dtype=float)
        metrics["params/use_gain_in_frequency_fit"] = np.asarray(int(self.use_gain_in_frequency_fit), dtype=int)
        metrics["params/use_mean_normalization_time_domain"] = np.asarray(int(self.use_mean_normalization_time_domain), dtype=int)
        metrics["params/use_mean_normalization_arx"] = np.asarray(int(self.use_mean_normalization_arx), dtype=int)
        metrics["params/min_valid_fraction"] = np.asarray(self.min_valid_fraction, dtype=float)
        metrics["params/primitive_max_abs_delay_seconds"] = np.asarray(self.primitive_max_abs_delay_seconds, dtype=float)
        metrics["params/primitive_max_tau_seconds"] = np.asarray(self.primitive_max_tau_seconds, dtype=float)
        metrics["params/primitive_min_methods_for_consensus"] = np.asarray(self.primitive_min_methods_for_consensus, dtype=int)
        metrics["params/n_beats"] = np.asarray(n_beats, dtype=int)

        return ProcessResult(metrics=metrics)
