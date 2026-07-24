import numpy as np
from scipy.sparse.csgraph import connected_components

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs
from .lowrank_pulsatility_metrics import LowRankPulsatilityMetrics


@registerPipeline(name="waveform_shape_metrics_denoised_alternatives")
class ArterialSegExample(ProcessPipeline):
    """
    Manuscript-aligned waveform-shape metrics on per-beat, per-branch,
    per-radius velocity waveforms.

    Public endpoint set
    -------------------
    The canonical outputs follow the simplified manuscript structure:
    timing and displacement distribution; near-peak crest width; excursion
    and pulsatility; cumulative-distance geometry; temporal kinetics and
    persistence; spectral ratio and harmonic-panel reconstruction fidelity;
    temporal support; and slope energy.

    All canonical metrics are gain-invariant or gain-robust under positive
    multiplicative scaling of v(t), up to numerical precision. The same formal
    definitions are applied to arterial and venous waveforms, although the
    manuscript interpretation focuses on arterial waveforms.

    Notes
    -----
    - Registered under its own name, distinct from "waveform_shape_metrics_denoised"
      (waveform_shape_metrics_denoised.py, the untouched production reference), so
      that pipeline stays the default and this one is only run when explicitly
      selected. Previously this registered under the same name as the original and
      silently shadowed it (whichever module pkgutil imported last won); that
      collision is what this rename fixes.
    - Deprecated exploratory higher-harmonic rolloff/support and phase-organization
      metrics are no longer part of the canonical public metric set.
    """

    description = (
        "Manuscript-aligned waveform-shape metrics "
        "(artery + vein; segment + aggregates + global) -- working copy with "
        "graph-Laplacian/low-rank-SVD/Kalman denoiser alternatives "
        "(.laplacian/.lowrank/.kalman) in addition to the original six-step chain."
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

    denoise_min_valid_samples = 32
    denoise_min_valid_fraction = 0.50
    denoise_harmonic_count = 8
    denoise_gaussian_sigma_samples = 1.25
    denoise_savgol_window = 9
    denoise_savgol_polyorder = 2
    denoise_hampel_window = 7
    denoise_hampel_nsigmas = 3.0
    denoise_use_robust_hampel = True

    def __init__(self) -> None:
        super().__init__()
        self.laplacian = self.GraphLaplacianDenoiser(self)
        self.lowrank = self.LowRankDenoiser(self)
        self.kalman = self.KalmanDenoiser(self)

    # ----------------------------
    # Generic Arrays/Math Helpers
    # ----------------------------

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
    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) < 3:
            return np.nan
        aa = a[mask] - float(np.mean(a[mask]))
        bb = b[mask] - float(np.mean(b[mask]))
        denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.nan
        return float(np.sum(aa * bb) / denom)
    
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
    
    # ----------------------------
    # Denoising: gating
    # ----------------------------

    def _denoise_segment_block(self, v_block: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply morphology-preserving denoising independently to each
        (time, beat, branch, radius) arterial segment pulse.
        """
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_block.shape}"
            )

        n_time, n_beats, n_branches, n_radii = v_block.shape
        pulse_shape = (n_beats, n_branches, n_radii)
        out = np.full_like(v_block, np.nan, dtype=float)
        corr = np.full(pulse_shape, np.nan, dtype=float)
        finite_fraction = np.zeros(pulse_shape, dtype=float)
        status_code = np.full(pulse_shape, 1, dtype=int)

        for beat_idx in range(n_beats):
            for branch_idx in range(n_branches):
                for radius_idx in range(n_radii):
                    index = (beat_idx, branch_idx, radius_idx)
                    pulse = np.asarray(
                        v_block[:, beat_idx, branch_idx, radius_idx], dtype=float
                    )
                    finite_mask = np.isfinite(pulse)
                    finite_count = int(np.sum(finite_mask))
                    finite_fraction[index] = finite_count / float(n_time)

                    if finite_count == 0:
                        status_code[index] = 1
                        continue
                    if not self._denoise_has_enough_valid_samples(finite_count, n_time):
                        status_code[index] = 2
                        continue
                    if float(np.nanstd(pulse)) <= self.eps:
                        status_code[index] = 3
                        out[:, beat_idx, branch_idx, radius_idx] = np.where(
                            finite_mask, pulse, np.nan
                        )
                        continue

                    denoised = self._denoise_pulse_1d(pulse)
                    out[:, beat_idx, branch_idx, radius_idx] = denoised
                    corr[index] = self._pearson_corr(pulse, denoised)
                    status_code[index] = 0

        diagnostics = {
            "original_vs_filtered_corr": corr,
            "finite_fraction": finite_fraction,
            "status_code": status_code,
        }
        return out, diagnostics

    def _denoise_has_enough_valid_samples(self, finite_count: int, n_time: int) -> bool:
        return finite_count >= int(
            self.denoise_min_valid_samples
        ) and finite_count / float(n_time) >= float(self.denoise_min_valid_fraction)

    # ----------------------------
    # Denoising: six step chain
    # ----------------------------
    
    def _hampel_filter_1d(
        self,
        x: np.ndarray,
        window: int = 7,
        n_sigmas: float = 3.0,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        n = x.size

        if n < 3:
            return x

        if window % 2 == 0:
            window += 1

        half = window // 2
        out = x.copy()

        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)

            local = x[i0:i1]
            local = local[np.isfinite(local)]

            if local.size < 3:
                continue

            med = float(np.median(local))
            mad = float(np.median(np.abs(local - med)))

            if mad <= self.eps:
                continue

            sigma = 1.4826 * mad

            if abs(x[i] - med) > n_sigmas * sigma:
                out[i] = med

        return out

    def _denoise_pulse_1d(self, pulse: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(pulse)
        x = np.arange(pulse.size, dtype=float)

        # Step 1: linearly interpolate gaps
        filled = np.interp(x, x[finite_mask], pulse[finite_mask])

        # Step 2: Hampel filter / despiking
        robust = self._hampel_filter_1d(
            filled,
            window=self.denoise_hampel_window,
            n_sigmas=self.denoise_hampel_nsigmas,
        )

        # Step 3: smoothing (FFT low-pass, Gaussian, Savitzky-Golay)
        harmonic = self._denoise_harmonic_lowpass(robust)
        gaussian = self._denoise_gaussian_smooth(robust)
        savgol = self._denoise_savgol_smooth(robust)

        # Step 4: combine (60% Savitzky-Golay + 25% FFT low-pass + 15% Gaussian)
        denoised = 0.25 * harmonic + 0.15 * gaussian + 0.60 * savgol

        # Step 5: clip to original min/max
        denoised = self._clip_to_input_range(denoised, pulse)

        # Step 6: restore original gaps as missing
        denoised[~finite_mask] = np.nan

        return denoised

    def _denoise_harmonic_lowpass(self, pulse: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(np.asarray(pulse, dtype=float))
        max_harmonic = min(int(self.denoise_harmonic_count), spectrum.size - 1)
        truncated = np.zeros_like(spectrum)
        truncated[: max_harmonic + 1] = spectrum[: max_harmonic + 1]
        return np.fft.irfft(truncated, n=pulse.size)

    def _denoise_gaussian_smooth(self, pulse: np.ndarray) -> np.ndarray:
        sigma = float(self.denoise_gaussian_sigma_samples)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= np.sum(kernel)
        return self._denoise_correlate_reflect(pulse, kernel)

    def _denoise_savgol_smooth(self, pulse: np.ndarray) -> np.ndarray:
        window = int(self.denoise_savgol_window)
        if window % 2 == 0:
            window += 1
        max_window = pulse.size if pulse.size % 2 == 1 else pulse.size - 1
        window = max(3, min(window, max_window))
        half_window = window // 2
        polyorder = min(int(self.denoise_savgol_polyorder), window - 1)
        offsets = np.arange(-half_window, half_window + 1, dtype=float)
        design = np.vander(offsets, N=polyorder + 1, increasing=True)
        coeffs = np.linalg.pinv(design)[0]
        return self._denoise_correlate_reflect(pulse, coeffs)

    @staticmethod
    def _denoise_correlate_reflect(pulse: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        radius = kernel.size // 2
        padded = np.pad(np.asarray(pulse, dtype=float), radius, mode="reflect")
        return np.correlate(padded, kernel, mode="valid")

    def _clip_to_input_range(
        self, candidate: np.ndarray, original: np.ndarray
    ) -> np.ndarray:
        finite = np.isfinite(original)
        if not np.any(finite):
            return candidate
        vmin = float(np.nanmin(original))
        vmax = float(np.nanmax(original))
        span = max(vmax - vmin, self.eps)
        margin = 0.05 * span
        out = np.asarray(candidate, dtype=float).copy()
        finite_out = np.isfinite(out)
        out[finite_out] = np.clip(out[finite_out], vmin - margin, vmax + margin)
        return out

    # ----------------------------
    # Denoising alternative: Graph-Laplacian (ported from origin/wvf_denoise)
    # ----------------------------

    class GraphLaplacianDenoiser:

        eps = 1e-12

        use_bandlimited_input = True

        gamma = 3.0
        min_corr = 0.70
        reference_min_corr = 0.80
        corr_power = 3.0

        max_lag_fraction = 0.12
        lag_sigma_fraction = 0.06

        preserve_nan_mask = True
        restore_pulse_scale = True
        clip_output = False

        branch_sigma = 2.0
        radius_sigma = 2.0

        metrics_use_filtered_signal = False

        def __init__(self, outer: "ArterialSegExample"):
            self._outer = outer

        def _interpolate_missing_samples(self, pulse: np.ndarray) -> np.ndarray:
            """Linearly interpolate NaN gaps in a 1D pulse."""
            pulse = np.asarray(pulse, dtype=float)
            finite_mask = np.isfinite(pulse)
            x = np.arange(pulse.size, dtype=float)
            return np.interp(x, x[finite_mask], pulse[finite_mask])

        @staticmethod
        def _best_circular_lag_corr(
            target_norm: np.ndarray, neighbor_norm: np.ndarray, lags: np.ndarray
        ) -> tuple[float, int]:
            """
            Best-correlating circular shift of neighbor_norm against target_norm.

            Both inputs are expected to already be zero-mean, L2-normalized (see
            _normalize_1d), so the dot product at each lag is the correlation.
            """
            best_corr = -np.inf
            best_lag = 0
            for lag in lags:
                corr = float(np.dot(target_norm, np.roll(neighbor_norm, int(lag))))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = int(lag)
            return float(np.clip(best_corr, -1.0, 1.0)), best_lag

        @staticmethod
        def _safe_mean_abs_change(original: np.ndarray, denoised: np.ndarray) -> float:
            """Mean absolute difference between original and denoised, ignoring NaNs."""
            original = np.asarray(original, dtype=float)
            denoised = np.asarray(denoised, dtype=float)
            mask = np.isfinite(original) & np.isfinite(denoised)
            if not np.any(mask):
                return np.nan
            return float(np.mean(np.abs(original[mask] - denoised[mask])))

        @staticmethod
        def _effective_count(weights: np.ndarray) -> float:
            """
            Inverse-participation-ratio effective count, same style as N_eff
            elsewhere in this file: 1 / sum(p^2) for weights normalized to sum 1.
            """
            weights = np.asarray(weights, dtype=float)
            total = float(np.sum(weights))
            if total <= 0.0:
                return 0.0
            p = weights / total
            denom = float(np.sum(p * p))
            if denom <= 0.0:
                return 0.0
            return float(1.0 / denom)

        # --- taken from reference_gate.py (ReferenceGateMixin) ---

        @staticmethod
        def _pointwise_median_reference(v_block: np.ndarray) -> np.ndarray:
            """
            Pointwise arterial reference waveform.

            For each time sample, take the median across every beat/branch/radius
            segment waveform and ignore NaNs. The result is the expected arterial
            shape used by the reference-correlation gate.
            """
            n_time = int(v_block.shape[0])
            flat = np.asarray(v_block, dtype=float).reshape(n_time, -1)
            reference = np.full((n_time,), np.nan, dtype=float)
            finite_rows = np.any(np.isfinite(flat), axis=1)
            if np.any(finite_rows):
                reference[finite_rows] = np.nanmedian(flat[finite_rows], axis=1)
            return reference

        def _prepare_reference_for_gate(
            self,
            v_block: np.ndarray,
            filled: np.ndarray,
            valid_flat_indices: np.ndarray,
            n_time: int,
        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            """
            Build the dense reference signals used by the pre-graph quality gate.

            1. reference[t] = median over all segment waveforms at time t, ignoring NaNs.
            2. If that pointwise median is too sparse, fall back to the median of the
               already-interpolated valid pulses.
            3. Interpolate any remaining missing reference samples so lag scoring uses
               dense vectors.
            4. Return two centered versions:
               - pRMS-normalized reference for the 0.80 quality gate.
               - L2-normalized reference for the phase-alignment code.
            """
            reference = self._pointwise_median_reference(v_block)
            finite_count = int(np.sum(np.isfinite(reference)))
            if not self._outer._denoise_has_enough_valid_samples(finite_count, n_time):
                reference = np.nanmedian(filled[valid_flat_indices], axis=0)

            reference = self._interpolate_missing_samples(reference)
            reference_gate_norm = self._normalize_prms_1d(reference)
            reference_norm = self._normalize_1d(reference)

            if reference_gate_norm is None or reference_norm is None:
                reference = np.nanmean(filled[valid_flat_indices], axis=0)
                reference_gate_norm = self._normalize_prms_1d(reference)
                reference_norm = self._normalize_1d(reference)

            return reference, reference_gate_norm, reference_norm

        def _apply_reference_gate(
            self,
            valid_flat_indices: np.ndarray,
            filled: np.ndarray,
            coords: list[tuple[int, int, int]],
            lags: np.ndarray,
            reference_gate_norm: np.ndarray,
            reference_corr: np.ndarray,
            reference_lag: np.ndarray,
            reference_keep_mask: np.ndarray,
            status_code: np.ndarray,
        ) -> tuple[np.ndarray, int, int]:
            """
            Keep only pulses that correlate with the median arterial reference.

            For each candidate pulse x and reference r:
            1. subtract each signal's own average waveform level;
            2. divide by pRMS = sqrt(mean(centered_signal ** 2));
            3. compute signed circular-lag correlations over the allowed lag window;
            4. retain the pulse when max_lag corr(r, x_lag) >= reference_min_corr.

            Rejected pulses are assigned status 7 and are left out of the returned
            flat-index array, so graph construction sees only retained pulses.
            """
            kept = []
            min_reference_corr = float(self.reference_min_corr)

            for flat_i in valid_flat_indices:
                pulse_gate_norm = self._normalize_prms_1d(filled[flat_i])
                if pulse_gate_norm is None:
                    status_code[coords[flat_i]] = 3
                    continue

                corr, lag = self._best_circular_lag_prms_corr(
                    reference_gate_norm, pulse_gate_norm, lags
                )
                reference_corr[coords[flat_i]] = float(corr)
                reference_lag[coords[flat_i]] = float(lag)

                if np.isfinite(corr) and corr >= min_reference_corr:
                    reference_keep_mask[coords[flat_i]] = True
                    kept.append(int(flat_i))
                else:
                    status_code[coords[flat_i]] = 7

            kept = np.asarray(kept, dtype=np.int32)
            return kept, int(kept.size), int(valid_flat_indices.size - kept.size)

        def _normalize_prms_1d(self, x: np.ndarray) -> np.ndarray | None:
            """
            Center and scale a dense signal by pointwise RMS energy.

            This keeps the reference gate shape-based: offsets and amplitudes do not
            affect the signed correlation score.
            """
            x = np.asarray(x, dtype=float)
            if x.size == 0 or not np.all(np.isfinite(x)):
                return None
            centered = x - float(np.mean(x))
            prms = float(np.sqrt(np.mean(centered * centered)))
            if prms <= self.eps:
                return None
            return centered / prms

        @staticmethod
        def _best_circular_lag_prms_corr(
            target_norm: np.ndarray, neighbor_norm: np.ndarray, lags: np.ndarray
        ) -> tuple[float, int]:
            best_corr = -np.inf
            best_lag = 0
            for lag in lags:
                corr = float(np.mean(target_norm * np.roll(neighbor_norm, int(lag))))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = int(lag)
            return float(np.clip(best_corr, -1.0, 1.0)), best_lag

        # --- ported from waveform_shape_metrics_denoised_laplace/pipeline.py ---

        def _denoise_segment_block(
            self, v_block: np.ndarray
        ) -> tuple[np.ndarray, dict]:
            """
            Denoise a ``(time, beat, branch, radius)`` arterial block by solving
            ``(I + gamma L) X = Y`` on a pulse-similarity graph.
            """
            if v_block.ndim != 4:
                raise ValueError(
                    f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                    f"got {v_block.shape}"
                )

            v_block = np.asarray(v_block, dtype=float)
            n_time, n_beats, n_branches, n_radii = v_block.shape
            pulse_shape = (n_beats, n_branches, n_radii)
            n_pulses = n_beats * n_branches * n_radii

            out = np.full_like(v_block, np.nan, dtype=float)
            corr_original = np.full(pulse_shape, np.nan, dtype=float)
            finite_fraction = np.zeros(pulse_shape, dtype=float)
            status_code = np.full(pulse_shape, 1, dtype=int)

            phase_alignment_lag = np.full(pulse_shape, np.nan, dtype=float)
            phase_alignment_corr = np.full(pulse_shape, np.nan, dtype=float)
            best_neighbor_corr = np.full(pulse_shape, np.nan, dtype=float)
            neighbor_count = np.zeros(pulse_shape, dtype=np.int32)
            effective_neighbor_count = np.zeros(pulse_shape, dtype=float)
            graph_degree = np.zeros(pulse_shape, dtype=float)
            graph_component_label = np.full(pulse_shape, -1, dtype=np.int32)
            mean_abs_change = np.full(pulse_shape, np.nan, dtype=float)
            reference_corr = np.full(pulse_shape, np.nan, dtype=float)
            reference_lag = np.full(pulse_shape, np.nan, dtype=float)
            reference_keep_mask = np.zeros(pulse_shape, dtype=bool)
            reference_kept_count = 0
            reference_rejected_count = 0

            filled = np.full((n_pulses, n_time), np.nan, dtype=float)
            normalized = np.full((n_pulses, n_time), np.nan, dtype=float)
            means = np.full((n_pulses,), np.nan, dtype=float)
            stds = np.full((n_pulses,), np.nan, dtype=float)
            finite_masks = np.zeros((n_pulses, n_time), dtype=bool)
            coords = []
            valid_flat_indices = []

            flat_index = 0
            for beat_idx in range(n_beats):
                for branch_idx in range(n_branches):
                    for radius_idx in range(n_radii):
                        index = (beat_idx, branch_idx, radius_idx)
                        pulse = v_block[:, beat_idx, branch_idx, radius_idx]
                        finite_mask = np.isfinite(pulse)
                        finite_masks[flat_index] = finite_mask
                        finite_count = int(np.sum(finite_mask))
                        finite_fraction[index] = finite_count / float(n_time)
                        coords.append(index)

                        if finite_count == 0:
                            status_code[index] = 1
                        elif not self._outer._denoise_has_enough_valid_samples(
                            finite_count, n_time
                        ):
                            status_code[index] = 2
                        elif float(np.nanstd(pulse)) <= self.eps:
                            status_code[index] = 3
                            out[:, beat_idx, branch_idx, radius_idx] = np.where(
                                finite_mask, pulse, np.nan
                            )
                        else:
                            filled_pulse = self._interpolate_missing_samples(pulse)
                            centered = filled_pulse - float(np.mean(filled_pulse))
                            norm = float(np.sqrt(np.sum(centered * centered)))
                            if norm <= self.eps:
                                status_code[index] = 3
                                out[:, beat_idx, branch_idx, radius_idx] = np.where(
                                    finite_mask, pulse, np.nan
                                )
                            else:
                                filled[flat_index] = filled_pulse
                                normalized[flat_index] = centered / norm
                                means[flat_index] = float(np.mean(filled_pulse))
                                stds[flat_index] = float(np.std(filled_pulse))
                                valid_flat_indices.append(flat_index)

                        flat_index += 1

            valid_flat_indices = np.asarray(valid_flat_indices, dtype=np.int32)
            n_valid = int(valid_flat_indices.size)
            max_lag = self._max_lag_samples(n_time)
            lags = np.arange(-max_lag, max_lag + 1, dtype=int)

            if n_valid < 2:
                self._mark_valid_without_graph_neighbors(
                    out=out,
                    v_block=v_block,
                    filled=filled,
                    finite_masks=finite_masks,
                    coords=coords,
                    valid_flat_indices=valid_flat_indices,
                    status_code=status_code,
                )
                diagnostics = self._diagnostics(
                    corr_original,
                    finite_fraction,
                    status_code,
                    phase_alignment_lag,
                    phase_alignment_corr,
                    best_neighbor_corr,
                    neighbor_count,
                    effective_neighbor_count,
                    graph_degree,
                    graph_component_label,
                    mean_abs_change,
                    reference_corr,
                    reference_lag,
                    reference_keep_mask,
                    reference_kept_count,
                    reference_rejected_count,
                    np.nan,
                    np.nan,
                    0,
                    n_valid,
                    n_valid,
                    np.nan,
                )
                self._last_denoise_diag = diagnostics
                return out, diagnostics

            # Reference-correlation prefilter.
            #
            # This is the "is this pulse an arterial-shaped waveform?" gate.
            # It happens before graph construction, so rejected pulses never become
            # graph vertices and remain NaN in the downstream metric input.
            reference, reference_gate_norm, reference_norm = (
                self._prepare_reference_for_gate(
                    v_block=v_block,
                    filled=filled,
                    valid_flat_indices=valid_flat_indices,
                    n_time=n_time,
                )
            )

            if reference_gate_norm is None or reference_norm is None:
                self._mark_valid_without_graph_neighbors(
                    out=out,
                    v_block=v_block,
                    filled=filled,
                    finite_masks=finite_masks,
                    coords=coords,
                    valid_flat_indices=valid_flat_indices,
                    status_code=status_code,
                )
                diagnostics = self._diagnostics(
                    corr_original,
                    finite_fraction,
                    status_code,
                    phase_alignment_lag,
                    phase_alignment_corr,
                    best_neighbor_corr,
                    neighbor_count,
                    effective_neighbor_count,
                    graph_degree,
                    graph_component_label,
                    mean_abs_change,
                    reference_corr,
                    reference_lag,
                    reference_keep_mask,
                    reference_kept_count,
                    reference_rejected_count,
                    np.nan,
                    np.nan,
                    0,
                    n_valid,
                    n_valid,
                    np.nan,
                )
                self._last_denoise_diag = diagnostics
                return out, diagnostics

            valid_flat_indices, reference_kept_count, reference_rejected_count = (
                self._apply_reference_gate(
                    valid_flat_indices=valid_flat_indices,
                    filled=filled,
                    coords=coords,
                    lags=lags,
                    reference_gate_norm=reference_gate_norm,
                    reference_corr=reference_corr,
                    reference_lag=reference_lag,
                    reference_keep_mask=reference_keep_mask,
                    status_code=status_code,
                )
            )
            n_valid = int(valid_flat_indices.size)

            if n_valid < 2:
                self._mark_valid_without_graph_neighbors(
                    out=out,
                    v_block=v_block,
                    filled=filled,
                    finite_masks=finite_masks,
                    coords=coords,
                    valid_flat_indices=valid_flat_indices,
                    status_code=status_code,
                )
                diagnostics = self._diagnostics(
                    corr_original,
                    finite_fraction,
                    status_code,
                    phase_alignment_lag,
                    phase_alignment_corr,
                    best_neighbor_corr,
                    neighbor_count,
                    effective_neighbor_count,
                    graph_degree,
                    graph_component_label,
                    mean_abs_change,
                    reference_corr,
                    reference_lag,
                    reference_keep_mask,
                    reference_kept_count,
                    reference_rejected_count,
                    np.nan,
                    np.nan,
                    0,
                    n_valid,
                    n_valid,
                    np.nan,
                )
                self._last_denoise_diag = diagnostics
                return out, diagnostics

            aligned = np.full((n_valid, n_time), np.nan, dtype=float)
            aligned_norm = np.full((n_valid, n_time), np.nan, dtype=float)
            aligned_shape = np.full((n_valid, n_time), np.nan, dtype=float)
            alignment_lags_by_row = np.zeros((n_valid,), dtype=int)

            for row_idx, flat_i in enumerate(valid_flat_indices):
                corr, lag = self._best_circular_lag_corr(
                    reference_norm, normalized[flat_i], lags
                )
                alignment_lags_by_row[row_idx] = int(lag)
                aligned[row_idx] = np.roll(filled[flat_i], int(lag))
                aligned_norm[row_idx] = self._normalize_1d(aligned[row_idx])
                aligned_shape[row_idx] = self._shape_signal_for_solve(
                    aligned[row_idx], means[flat_i], stds[flat_i]
                )
                phase_alignment_lag[coords[flat_i]] = float(lag)
                phase_alignment_corr[coords[flat_i]] = float(corr)

            W = self._build_weight_matrix(
                aligned_norm=aligned_norm,
                valid_flat_indices=valid_flat_indices,
                coords=coords,
                alignment_lags_by_row=alignment_lags_by_row,
                n_time=n_time,
                best_neighbor_corr=best_neighbor_corr,
                neighbor_count=neighbor_count,
                effective_neighbor_count=effective_neighbor_count,
                graph_degree=graph_degree,
            )

            n_components, component_labels = connected_components(
                W > 0.0, directed=False, return_labels=True
            )
            component_labels = np.asarray(component_labels, dtype=np.int32)
            for row_i, flat_i in enumerate(valid_flat_indices):
                graph_component_label[coords[flat_i]] = int(component_labels[row_i])

            edge_count = int(np.sum(W > 0.0) // 2)
            if edge_count == 0:
                self._mark_valid_without_graph_neighbors(
                    out=out,
                    v_block=v_block,
                    filled=filled,
                    finite_masks=finite_masks,
                    coords=coords,
                    valid_flat_indices=valid_flat_indices,
                    status_code=status_code,
                )
                diagnostics = self._diagnostics(
                    corr_original,
                    finite_fraction,
                    status_code,
                    phase_alignment_lag,
                    phase_alignment_corr,
                    best_neighbor_corr,
                    neighbor_count,
                    effective_neighbor_count,
                    graph_degree,
                    graph_component_label,
                    mean_abs_change,
                    reference_corr,
                    reference_lag,
                    reference_keep_mask,
                    reference_kept_count,
                    reference_rejected_count,
                    np.nan,
                    np.nan,
                    edge_count,
                    n_components,
                    n_valid,
                    np.nan,
                )
                self._last_denoise_diag = diagnostics
                return out, diagnostics

            gamma = float(max(self.gamma, 0.0))
            dirichlet_before = self._dirichlet_energy(W, aligned_shape)
            denoised_aligned = aligned_shape.copy()
            component_failed = np.zeros((n_valid,), dtype=bool)
            component_sizes = np.bincount(component_labels, minlength=n_components)
            condition_numbers = []

            for component_idx in range(n_components):
                rows = np.flatnonzero(component_labels == component_idx)
                if rows.size <= 1:
                    continue

                W_component = W[np.ix_(rows, rows)]
                degree = np.sum(W_component, axis=1)
                L = np.diag(degree) - W_component
                A = np.eye(rows.size, dtype=float) + gamma * L

                try:
                    condition_numbers.append(float(np.linalg.cond(A)))
                    denoised_aligned[rows] = np.linalg.solve(A, aligned_shape[rows])
                except np.linalg.LinAlgError:
                    component_failed[rows] = True

            condition_number = (
                float(np.nanmax(condition_numbers)) if condition_numbers else np.nan
            )
            dirichlet_after = self._dirichlet_energy(W, denoised_aligned)

            for row_idx, flat_i in enumerate(valid_flat_indices):
                beat_idx, branch_idx, radius_idx = coords[flat_i]
                original = v_block[:, beat_idx, branch_idx, radius_idx]
                finite_mask = finite_masks[flat_i]
                lag = int(alignment_lags_by_row[row_idx])

                denoised = np.roll(denoised_aligned[row_idx], -lag)
                denoised = self._restore_shape_signal(
                    denoised, means[flat_i], stds[flat_i]
                )
                if self.clip_output:
                    denoised = self._outer._clip_to_input_range(denoised, original)
                if self.preserve_nan_mask:
                    denoised[~finite_mask] = np.nan

                out[:, beat_idx, branch_idx, radius_idx] = denoised
                corr_original[coords[flat_i]] = self._outer._pearson_corr(
                    original, denoised
                )
                mean_abs_change[coords[flat_i]] = self._safe_mean_abs_change(
                    original, denoised
                )
                if component_sizes[int(component_labels[row_idx])] <= 1:
                    status_code[coords[flat_i]] = 6
                elif component_failed[row_idx]:
                    status_code[coords[flat_i]] = 5
                else:
                    status_code[coords[flat_i]] = 0

            diagnostics = self._diagnostics(
                corr_original,
                finite_fraction,
                status_code,
                phase_alignment_lag,
                phase_alignment_corr,
                best_neighbor_corr,
                neighbor_count,
                effective_neighbor_count,
                graph_degree,
                graph_component_label,
                mean_abs_change,
                reference_corr,
                reference_lag,
                reference_keep_mask,
                reference_kept_count,
                reference_rejected_count,
                dirichlet_before,
                dirichlet_after,
                edge_count,
                n_components,
                n_valid,
                condition_number,
            )
            self._last_denoise_diag = diagnostics
            return out, diagnostics

        def _build_weight_matrix(
            self,
            aligned_norm: np.ndarray,
            valid_flat_indices: np.ndarray,
            coords: list[tuple[int, int, int]],
            alignment_lags_by_row: np.ndarray,
            n_time: int,
            best_neighbor_corr: np.ndarray,
            neighbor_count: np.ndarray,
            effective_neighbor_count: np.ndarray,
            graph_degree: np.ndarray,
        ) -> np.ndarray:
            n_valid = int(valid_flat_indices.size)
            W = np.zeros((n_valid, n_valid), dtype=float)
            corr_matrix = np.asarray(aligned_norm, dtype=float) @ np.asarray(
                aligned_norm, dtype=float
            ).T
            np.fill_diagonal(corr_matrix, 0.0)

            for row_i, flat_i in enumerate(valid_flat_indices):
                for row_j in range(row_i + 1, n_valid):
                    flat_j = int(valid_flat_indices[row_j])
                    corr = float(corr_matrix[row_i, row_j])
                    if (not np.isfinite(corr)) or corr < float(self.min_corr):
                        continue

                    weight = self._neighbor_weight(
                        corr=corr,
                        lag_delta=int(
                            alignment_lags_by_row[row_i] - alignment_lags_by_row[row_j]
                        ),
                        n_time=n_time,
                        coord_i=coords[flat_i],
                        coord_j=coords[flat_j],
                    )
                    if weight <= 0.0 or not np.isfinite(weight):
                        continue
                    W[row_i, row_j] = weight
                    W[row_j, row_i] = weight

            for row_i, flat_i in enumerate(valid_flat_indices):
                neighbor_rows = np.flatnonzero(W[row_i] > 0.0)
                neighbor_count[coords[flat_i]] = int(neighbor_rows.size)
                if neighbor_rows.size:
                    weights = W[row_i, neighbor_rows]
                    corrs = corr_matrix[row_i, neighbor_rows]
                    best_neighbor_corr[coords[flat_i]] = float(np.max(corrs))
                    effective_neighbor_count[coords[flat_i]] = self._effective_count(
                        weights
                    )
                graph_degree[coords[flat_i]] = float(np.sum(W[row_i]))

            return W

        def _neighbor_weight(
            self,
            corr: float,
            lag_delta: int,
            n_time: int,
            coord_i: tuple[int, int, int],
            coord_j: tuple[int, int, int],
        ) -> float:
            min_corr = float(self.min_corr)
            corr_score = (float(corr) - min_corr) / max(1.0 - min_corr, self.eps)
            corr_score = float(np.clip(corr_score, 0.0, 1.0))
            corr_weight = corr_score ** float(self.corr_power)

            lag_sigma = float(self.lag_sigma_fraction) * float(n_time)
            if lag_sigma > self.eps:
                lag_weight = float(np.exp(-0.5 * (float(lag_delta) / lag_sigma) ** 2))
            else:
                lag_weight = 1.0

            spatial_weight = 1.0
            branch_sigma = float(self.branch_sigma)
            radius_sigma = float(self.radius_sigma)

            if branch_sigma > 0.0:
                dk = float(coord_i[1] - coord_j[1])
                spatial_weight *= float(np.exp(-0.5 * (dk / branch_sigma) ** 2))

            if radius_sigma > 0.0:
                dr = float(coord_i[2] - coord_j[2])
                spatial_weight *= float(np.exp(-0.5 * (dr / radius_sigma) ** 2))

            return corr_weight * lag_weight * spatial_weight

        def _max_lag_samples(self, n_time: int) -> int:
            max_lag = int(round(float(self.max_lag_fraction) * n_time))
            return max(0, min(max_lag, max(0, n_time // 2 - 1)))

        def _normalize_1d(self, x: np.ndarray) -> np.ndarray | None:
            x = np.asarray(x, dtype=float)
            if x.size == 0 or not np.all(np.isfinite(x)):
                return None
            centered = x - float(np.mean(x))
            norm = float(np.sqrt(np.sum(centered * centered)))
            if norm <= self.eps:
                return None
            return centered / norm

        def _shape_signal_for_solve(
            self, pulse: np.ndarray, pulse_mean: float, pulse_std: float
        ) -> np.ndarray:
            if (
                self.restore_pulse_scale
                and np.isfinite(pulse_mean)
                and np.isfinite(pulse_std)
                and pulse_std > self.eps
            ):
                return (np.asarray(pulse, dtype=float) - pulse_mean) / pulse_std
            return np.asarray(pulse, dtype=float)

        def _restore_shape_signal(
            self, solved: np.ndarray, pulse_mean: float, pulse_std: float
        ) -> np.ndarray:
            if (
                self.restore_pulse_scale
                and np.isfinite(pulse_mean)
                and np.isfinite(pulse_std)
                and pulse_std > self.eps
            ):
                return pulse_mean + pulse_std * np.asarray(solved, dtype=float)
            return np.asarray(solved, dtype=float)

        @staticmethod
        def _dirichlet_energy(W: np.ndarray, X: np.ndarray) -> float:
            if W.size == 0 or X.size == 0:
                return np.nan
            diff = X[:, None, :] - X[None, :, :]
            return float(0.5 * np.sum(W[:, :, None] * diff * diff))

        def _mark_valid_without_graph_neighbors(
            self,
            out: np.ndarray,
            v_block: np.ndarray,
            filled: np.ndarray,
            finite_masks: np.ndarray,
            coords: list[tuple[int, int, int]],
            valid_flat_indices: np.ndarray,
            status_code: np.ndarray,
        ) -> None:
            for flat_i in valid_flat_indices:
                beat_idx, branch_idx, radius_idx = coords[flat_i]
                restored = filled[flat_i].copy()
                restored[~finite_masks[flat_i]] = np.nan
                if not np.any(np.isfinite(restored)):
                    restored = v_block[:, beat_idx, branch_idx, radius_idx]
                out[:, beat_idx, branch_idx, radius_idx] = restored
                status_code[coords[flat_i]] = 4

        @staticmethod
        def _diagnostics(
            corr_original: np.ndarray,
            finite_fraction: np.ndarray,
            status_code: np.ndarray,
            phase_alignment_lag: np.ndarray,
            phase_alignment_corr: np.ndarray,
            best_neighbor_corr: np.ndarray,
            neighbor_count: np.ndarray,
            effective_neighbor_count: np.ndarray,
            graph_degree: np.ndarray,
            graph_component_label: np.ndarray,
            mean_abs_change: np.ndarray,
            reference_corr: np.ndarray,
            reference_lag: np.ndarray,
            reference_keep_mask: np.ndarray,
            reference_kept_count: int,
            reference_rejected_count: int,
            dirichlet_before: float,
            dirichlet_after: float,
            edge_count: int,
            component_count: int,
            valid_pulse_count: int,
            condition_number: float,
        ) -> dict:
            return {
                "original_vs_filtered_corr": corr_original,
                "finite_fraction": finite_fraction,
                "status_code": status_code,
                "phase_alignment_lag_samples": phase_alignment_lag,
                "phase_alignment_corr": phase_alignment_corr,
                "best_neighbor_corr": best_neighbor_corr,
                "neighbor_count": neighbor_count,
                "effective_neighbor_count": effective_neighbor_count,
                "graph_degree": graph_degree,
                "graph_component_label": graph_component_label,
                "mean_abs_change": mean_abs_change,
                "reference_corr": reference_corr,
                "reference_lag_samples": reference_lag,
                "reference_keep_mask": reference_keep_mask,
                "reference_kept_count": int(reference_kept_count),
                "reference_rejected_count": int(reference_rejected_count),
                "dirichlet_energy_before": float(dirichlet_before),
                "dirichlet_energy_after": float(dirichlet_after),
                "graph_edge_count": int(edge_count),
                "graph_component_count": int(component_count),
                "valid_pulse_count": int(valid_pulse_count),
                "system_condition_number": float(condition_number),
            }

        def _replace_denoising_params(self, metrics: dict) -> None:
            for key in list(metrics):
                if key.startswith("artery/by_segment/denoising/params/"):
                    del metrics[key]

            status_key = "artery/by_segment/denoising/status_code"
            status_value = metrics.get(status_key)
            if hasattr(status_value, "attrs") and status_value.attrs is not None:
                status_value.attrs["definition"] = (
                    "0 denoised, 1 all_nan, 2 too_sparse, 3 low_variance, "
                    "4 no_eligible_graph_neighbors, 5 graph_solve_failed, "
                    "6 singleton_graph_component, 7 low_reference_corr"
                )

        def _pack_denoising_outputs(self, metrics: dict) -> None:
            diag = getattr(self, "_last_denoise_diag", None)
            if not diag:
                return

            base = "artery/by_segment/denoising"
            attrs = {"axis_order": "beat, branch, radius"}
            per_segment_defs = {
                "phase_alignment_lag_samples": (
                    "Circular lag, in samples, applied to align each pulse to the "
                    "ensemble reference before graph construction."
                ),
                "phase_alignment_corr": (
                    "Cross-correlation between each pulse and the ensemble reference "
                    "after the selected alignment lag."
                ),
                "best_neighbor_corr": (
                    "Highest zero-lag correlation among retained graph neighbors "
                    "after ensemble phase alignment."
                ),
                "neighbor_count": (
                    "Number of positive-weight neighbors in the final undirected "
                    "pulse graph."
                ),
                "effective_neighbor_count": (
                    "Inverse-concentration effective count of final undirected "
                    "neighbor weights."
                ),
                "graph_degree": (
                    "Weighted degree in the final undirected pulse graph."
                ),
                "graph_component_label": (
                    "Connected-component label in the final undirected pulse graph; "
                    "-1 for pulses excluded from graph construction."
                ),
                "mean_abs_change": (
                    "Mean absolute pointwise change between the original pulse and "
                    "the graph-Laplacian denoised pulse."
                ),
                "reference_corr": (
                    "Best signed circular-lag cross-correlation between each pulse "
                    "and the pointwise median arterial reference before graph "
                    "construction."
                ),
                "reference_lag_samples": (
                    "Circular lag, in samples, giving reference_corr for the "
                    "preprocessing reference-correlation gate."
                ),
                "reference_keep_mask": (
                    "Boolean mask indicating pulses retained by the preprocessing "
                    "reference-correlation gate."
                ),
            }

            for name, definition in per_segment_defs.items():
                if name in {"neighbor_count", "graph_component_label"}:
                    dtype = np.int32
                elif name == "reference_keep_mask":
                    dtype = bool
                else:
                    dtype = float
                metrics[f"{base}/{name}"] = with_attrs(
                    np.asarray(diag[name], dtype=dtype),
                    {**attrs, "definition": definition},
                )

            scalar_defs = {
                "dirichlet_energy_before": (
                    "Graph Dirichlet energy Tr(Y^T L Y) before Laplacian smoothing."
                ),
                "dirichlet_energy_after": (
                    "Graph Dirichlet energy Tr(X^T L X) after Laplacian smoothing."
                ),
                "graph_edge_count": (
                    "Number of undirected positive-weight edges in the pulse graph."
                ),
                "graph_component_count": (
                    "Number of connected components among valid pulse-graph vertices."
                ),
                "valid_pulse_count": (
                    "Number of valid pulse realizations included in the graph solve."
                ),
                "reference_kept_count": (
                    "Number of valid pulse realizations retained by the preprocessing "
                    "reference-correlation gate."
                ),
                "reference_rejected_count": (
                    "Number of otherwise valid pulse realizations rejected by the "
                    "preprocessing reference-correlation gate."
                ),
                "system_condition_number": (
                    "Dense condition number of I + gamma L used for the graph solve."
                ),
            }
            for name, definition in scalar_defs.items():
                dtype = np.int32 if name.endswith("_count") else float
                metrics[f"{base}/{name}"] = with_attrs(
                    np.asarray(diag[name], dtype=dtype), {"definition": definition}
                )

            params = {
                "method": "graph_laplacian_tikhonov",
                "use_bandlimited_input": int(bool(self.use_bandlimited_input)),
                "gamma": float(self.gamma),
                "min_corr": float(self.min_corr),
                "reference_min_corr": float(self.reference_min_corr),
                "corr_power": float(self.corr_power),
                "max_lag_fraction": float(self.max_lag_fraction),
                "lag_sigma_fraction": float(self.lag_sigma_fraction),
                "preserve_nan_mask": int(bool(self.preserve_nan_mask)),
                "restore_pulse_scale": int(bool(self.restore_pulse_scale)),
                "clip_output": int(bool(self.clip_output)),
                "metrics_use_filtered_signal": int(
                    bool(self.metrics_use_filtered_signal)
                ),
                "bandlimited_segment_metrics_source": (
                    "filtered_for_metrics"
                    if bool(self.metrics_use_filtered_signal)
                    else "bandlimited"
                ),
                "branch_sigma": float(self.branch_sigma),
                "radius_sigma": float(self.radius_sigma),
            }
            for name, value in params.items():
                metrics[f"{base}/params/{name}"] = np.asarray(value)

    # ----------------------------
    # Denoising alternative: SVD/low-rank across beats
    # ----------------------------

    class LowRankDenoiser:
        """
        Instead of denoising one pulse at a time, this stacks every valid pulse
        in the acquisition (beat x branch x radius) into a matrix and replaces
        each pulse with its rank-K reconstruction from an SVD fit across the
        whole ensemble -- pulses "borrow" shape from each other instead of being
        smoothed in isolation.

        Reuses LowRankPulsatilityMetrics._compute_representation (the SVD
        construction already used to compute the A1/rho1 diagnostics in
        lowrank_pulsatility_metrics.py) rather than reimplementing it. That
        pipeline already forms the rank-K reconstruction internally as a step
        toward computing residual-fraction (rho_m) statistics; this candidate
        just maps that reconstruction back into a full (time, beat, branch,
        radius) block and exposes it as an actual denoised signal instead of
        only using it to derive summary scalars.
        """

        max_modes = 12
        min_valid_samples_fraction = 0.95
        min_valid_columns = 3
        preserve_nan_mask = True

        def __init__(self, outer: "ArterialSegExample"):
            self._outer = outer

        def _denoise_segment_block(
            self, v_block: np.ndarray, T: np.ndarray | None = None
        ) -> tuple[np.ndarray, dict]:
            """
            Denoise a ``(time, beat, branch, radius)`` arterial block by replacing
            each valid pulse with its rank-K low-rank reconstruction
            ``mu(b,k,r) + sum_{m=1..K} a_m(b,k,r) * u_m(t)``.

            ``T`` (beat period, seconds) is optional here: the reused pipeline
            uses it only as an extra per-beat validity gate, not in the
            reconstruction math itself. If omitted, a trivially-valid placeholder
            is used so this candidate has the same ``(v_block) -> (denoised,
            diagnostics)`` calling convention as the six-step chain and the
            graph-Laplacian candidate, neither of which take T either.

            Status codes: 0 denoised, 1 all_nan, 2 excluded_by_lowrank_gate
            (didn't meet the reused pipeline's min_valid_samples_fraction or beat
            period validity check), 3 svd_unavailable (whole acquisition had
            fewer than min_valid_columns valid pulses).
            """
            v_block = np.asarray(v_block, dtype=float)
            if v_block.ndim != 4:
                raise ValueError(
                    f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                    f"got {v_block.shape}"
                )
            n_time, n_beats, n_branches, n_radii = v_block.shape
            pulse_shape = (n_beats, n_branches, n_radii)

            if T is None:
                T = np.ones((1, n_beats), dtype=float)

            helper = LowRankPulsatilityMetrics()
            helper.min_valid_samples_fraction = self.min_valid_samples_fraction
            helper.min_valid_columns = self.min_valid_columns
            helper.max_modes_panel = self.max_modes
            rep = helper._compute_representation(v_block=v_block, T=T)

            out = np.full_like(v_block, np.nan, dtype=float)
            corr = np.full(pulse_shape, np.nan, dtype=float)
            finite_mask_full = np.isfinite(v_block)
            finite_fraction = np.mean(finite_mask_full, axis=0)
            status_code = np.full(pulse_shape, 1, dtype=int)
            all_nan = finite_fraction <= 0.0
            status_code[~all_nan] = 2

            if not rep.get("svd_available", False):
                status_code[~all_nan] = 3
                diagnostics = {
                    "original_vs_filtered_corr": corr,
                    "finite_fraction": finite_fraction,
                    "status_code": status_code,
                    "n_modes_used": 0,
                    "svd_reason": rep.get("svd_reason", "unknown"),
                    "singular_energy_fraction": np.full((0,), np.nan),
                    "effective_rank": np.nan,
                    "eta1": np.nan,
                    "eta23": np.nan,
                }
                return out, diagnostics

            valid_column_mask = rep["valid_column_mask"]
            n_modes = int(rep["n_modes_panel"])
            U_panel = rep["U_panel"][:, :n_modes]
            score_flat = rep["score_panel_flat"][:n_modes, :]
            X_recon = U_panel @ score_flat

            mu = rep["mu"]
            out[:, valid_column_mask] = X_recon + mu[valid_column_mask][None, :]
            if self.preserve_nan_mask:
                out[~finite_mask_full] = np.nan

            status_code[valid_column_mask] = 0

            for beat_idx, branch_idx, radius_idx in np.argwhere(valid_column_mask):
                corr[beat_idx, branch_idx, radius_idx] = self._outer._pearson_corr(
                    v_block[:, beat_idx, branch_idx, radius_idx],
                    out[:, beat_idx, branch_idx, radius_idx],
                )

            acq = rep["acq"]
            diagnostics = {
                "original_vs_filtered_corr": corr,
                "finite_fraction": finite_fraction,
                "status_code": status_code,
                "n_modes_used": n_modes,
                "svd_reason": rep.get("svd_reason", "ok"),
                "singular_energy_fraction": rep["energy_fraction"],
                "captured_energy_fraction": float(
                    np.sum(rep["energy_fraction"][:n_modes])
                ),
                "effective_rank": acq.get("effective_rank", np.nan),
                "eta1": acq.get("eta1", np.nan),
                "eta23": acq.get("eta23", np.nan),
            }
            return out, diagnostics

    # ----------------------------
    # Denoising alternative: Cross-realization Kalman filtering
    # ----------------------------

    class KalmanDenoiser:
        """
        Two directional passes, each denoising along one axis so that comparing
        the result along the OTHER axis isolates genuine variation instead of
        noise -- the same spatial-vs-temporal split as the variability paper's
        Vs/Vt scores, but produced by actual smoothing rather than descriptive
        dispersion statistics:

          - beat-axis pass: for a fixed (branch, radius), treat each pulse's
            value at time-sample t as a noisy repeated measurement of a
            slowly-evolving true value across beats b, and Kalman-smooth across
            b. Comparing the smoothed result across different (branch, radius)
            then isolates genuine spatial variation.
          - radius-axis pass: for a fixed (beat, branch), Kalman-smooth across
            radius r instead (the physically continuous, ordered axis across
            the vessel cross-section -- branch is categorical/different vessels
            and is deliberately excluded from this axis). Comparing across
            beats then isolates genuine temporal variation.

        Both use a local-level (random walk) Kalman filter + RTS smoother,
        implemented directly here since the scalar recursions are simple and
        this keeps the method fully auditable without a new dependency.
        """

        eps = 1e-12

        process_to_observation_ratio = 0.1
        min_valid_along_axis = 3

        def __init__(self, outer: "ArterialSegExample"):
            self._outer = outer

        def _robust_noise_variance(self, values: np.ndarray) -> float:
            """
            Robust observation-noise variance estimate from first differences,
            using the same MAD-based approach as this file's Hampel filter
            (median/MAD instead of mean/std so a handful of outliers don't
            dominate the estimate).

            For a slowly-varying true signal plus i.i.d. noise, the variance of
            first differences is twice the noise variance, hence the /2 below.
            """
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 3:
                return float(np.var(finite)) if finite.size > 0 else self.eps
            diffs = np.diff(finite)
            mad = float(np.median(np.abs(diffs - np.median(diffs))))
            sigma = 1.4826 * mad
            return float(max((sigma**2) / 2.0, self.eps))

        def _kalman_rts_smooth_1d(self, y: np.ndarray, Q: float, R: float) -> np.ndarray:
            """
            Local-level (random walk) Kalman filter + Rauch-Tung-Striebel smoother
            along one 1D axis. NaN entries are treated as missing observations
            (prediction-only update, no measurement correction that step), so
            gaps don't need separate interpolation.
            """
            y = np.asarray(y, dtype=float)
            n = y.size
            finite = np.isfinite(y)
            if np.sum(finite) < 2:
                return y.copy()

            finite_vals = y[finite]
            x_prev = float(np.mean(finite_vals))
            P_prev = float(np.var(finite_vals)) + R

            x_filt = np.empty(n, dtype=float)
            P_filt = np.empty(n, dtype=float)
            x_pred = np.empty(n, dtype=float)
            P_pred = np.empty(n, dtype=float)

            for k in range(n):
                xp = x_prev
                Pp = P_prev + Q
                x_pred[k] = xp
                P_pred[k] = Pp
                if finite[k]:
                    gain = Pp / (Pp + R)
                    xf = xp + gain * (y[k] - xp)
                    pf = (1.0 - gain) * Pp
                else:
                    xf = xp
                    pf = Pp
                x_filt[k] = xf
                P_filt[k] = pf
                x_prev, P_prev = xf, pf

            x_smooth = x_filt.copy()
            for k in range(n - 2, -1, -1):
                denom = P_pred[k + 1]
                if denom <= self.eps:
                    continue
                gain_smooth = P_filt[k] / denom
                x_smooth[k] = x_filt[k] + gain_smooth * (x_smooth[k + 1] - x_pred[k + 1])

            return x_smooth

        def _pulse_validity_mask(self, v_block: np.ndarray) -> np.ndarray:
            """Same per-pulse validity notion as the six-step chain's gating."""
            n_time = v_block.shape[0]
            finite_count = np.sum(np.isfinite(v_block), axis=0)
            return (
                finite_count >= int(self._outer.denoise_min_valid_samples)
            ) & (
                finite_count / float(n_time)
                >= float(self._outer.denoise_min_valid_fraction)
            )

        def _denoise_segment_block_beat_axis(
            self, v_block: np.ndarray
        ) -> tuple[np.ndarray, dict]:
            """
            Denoise by Kalman-smoothing each (branch, radius) location's pulses
            across the beat axis. Status codes: 0 denoised, 1 all_nan,
            2 too_sparse (this pulse itself), 3 axis_too_sparse (not enough other
            valid beats at this branch/radius to smooth across).
            """
            v_block = np.asarray(v_block, dtype=float)
            if v_block.ndim != 4:
                raise ValueError(
                    f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                    f"got {v_block.shape}"
                )
            n_time, n_beats, n_branches, n_radii = v_block.shape
            pulse_shape = (n_beats, n_branches, n_radii)

            out = np.full_like(v_block, np.nan, dtype=float)
            corr = np.full(pulse_shape, np.nan, dtype=float)
            finite_count = np.sum(np.isfinite(v_block), axis=0)
            finite_fraction = finite_count / float(n_time)
            status_code = np.where(finite_count == 0, 1, 2).astype(int)

            pulse_ok = self._pulse_validity_mask(v_block)

            for branch_idx in range(n_branches):
                for radius_idx in range(n_radii):
                    valid_beats = pulse_ok[:, branch_idx, radius_idx]
                    n_valid = int(np.sum(valid_beats))
                    if n_valid < max(2, int(self.min_valid_along_axis)):
                        continue

                    y_col = np.where(
                        valid_beats, v_block[:, :, branch_idx, radius_idx], np.nan
                    )
                    # Robust R estimate pooled across all time samples in this
                    # column (few beats means a single time-sample's sequence is
                    # too short to estimate noise from reliably on its own).
                    pooled = y_col[:, valid_beats].ravel()
                    r_var = self._robust_noise_variance(pooled)
                    q_var = r_var * float(self.process_to_observation_ratio)

                    for t in range(n_time):
                        out[t, :, branch_idx, radius_idx] = self._kalman_rts_smooth_1d(
                            y_col[t, :], Q=q_var, R=r_var
                        )

                    out[:, ~valid_beats, branch_idx, radius_idx] = np.nan
                    status_code[valid_beats, branch_idx, radius_idx] = 0

                    for beat_idx in np.flatnonzero(valid_beats):
                        corr[beat_idx, branch_idx, radius_idx] = (
                            self._outer._pearson_corr(
                                v_block[:, beat_idx, branch_idx, radius_idx],
                                out[:, beat_idx, branch_idx, radius_idx],
                            )
                        )

            diagnostics = {
                "original_vs_filtered_corr": corr,
                "finite_fraction": finite_fraction,
                "status_code": status_code,
                "axis": "beat",
            }
            return out, diagnostics

        def _denoise_segment_block_radius_axis(
            self, v_block: np.ndarray
        ) -> tuple[np.ndarray, dict]:
            """
            Denoise by Kalman-smoothing each (beat, branch) location's pulses
            across the radius axis (the physically ordered axis across the
            vessel cross-section). Same status-code convention as the beat-axis
            pass, with the "axis" (beats vs radii) swapped.
            """
            v_block = np.asarray(v_block, dtype=float)
            if v_block.ndim != 4:
                raise ValueError(
                    f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                    f"got {v_block.shape}"
                )
            n_time, n_beats, n_branches, n_radii = v_block.shape
            pulse_shape = (n_beats, n_branches, n_radii)

            out = np.full_like(v_block, np.nan, dtype=float)
            corr = np.full(pulse_shape, np.nan, dtype=float)
            finite_count = np.sum(np.isfinite(v_block), axis=0)
            finite_fraction = finite_count / float(n_time)
            status_code = np.where(finite_count == 0, 1, 2).astype(int)

            pulse_ok = self._pulse_validity_mask(v_block)

            for beat_idx in range(n_beats):
                for branch_idx in range(n_branches):
                    valid_radii = pulse_ok[beat_idx, branch_idx, :]
                    n_valid = int(np.sum(valid_radii))
                    if n_valid < max(2, int(self.min_valid_along_axis)):
                        continue

                    y_col = np.where(
                        valid_radii, v_block[:, beat_idx, branch_idx, :], np.nan
                    )
                    pooled = y_col[:, valid_radii].ravel()
                    r_var = self._robust_noise_variance(pooled)
                    q_var = r_var * float(self.process_to_observation_ratio)

                    for t in range(n_time):
                        out[t, beat_idx, branch_idx, :] = self._kalman_rts_smooth_1d(
                            y_col[t, :], Q=q_var, R=r_var
                        )

                    out[:, beat_idx, branch_idx, ~valid_radii] = np.nan
                    status_code[beat_idx, branch_idx, valid_radii] = 0

                    for radius_idx in np.flatnonzero(valid_radii):
                        corr[beat_idx, branch_idx, radius_idx] = (
                            self._outer._pearson_corr(
                                v_block[:, beat_idx, branch_idx, radius_idx],
                                out[:, beat_idx, branch_idx, radius_idx],
                            )
                        )

            diagnostics = {
                "original_vs_filtered_corr": corr,
                "finite_fraction": finite_fraction,
                "status_code": status_code,
                "axis": "radius",
            }
            return out, diagnostics

    # ----------------------------
    # Single-Pulse Metric Building Blocks
    # ----------------------------

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
          S_rise, S_fall, t_rise_over_T, t_fall_over_T
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

    def _late_cycle_mean_fraction(self, v: np.ndarray) -> float:
        """
        v_end_over_vbar where v_end is the mean over [ratio_vend_start*T, ratio_vend_end*T].
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

    def _normalized_cumulative_distance_samples(
        self, v: np.ndarray, Tbeat: float, m0: float
    ) -> dict:
        """
        Returns normalized cumulative-distance landmarks d_q/D evaluated at fixed phase q:
          d_q = D(qT) / D(T), for q in {0.10, 0.25, 0.50, 0.75, 0.90}
        """
        out = {
            "d10_over_D": np.nan,
            "d25_over_D": np.nan,
            "d50_over_D": np.nan,
            "d75_over_D": np.nan,
            "d90_over_D": np.nan,
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

        out["d10_over_D"] = sample_at_ratio(0.10)
        out["d25_over_D"] = sample_at_ratio(0.25)
        out["d50_over_D"] = sample_at_ratio(0.50)
        out["d75_over_D"] = sample_at_ratio(0.75)
        out["d90_over_D"] = sample_at_ratio(0.90)
        return out

    def _d_quantile_shape_metrics(self, d_samples: dict) -> tuple[float, float]:
        """
        From d10,d25,d50,d75,d90 define:
          Q_d_width = d75 - d25
          Q_d_skew = ((d90-d50) - (d50-d10)) / (d90-d10 + eps)
        """
        d10 = d_samples["d10_over_D"]
        d25 = d_samples["d25_over_D"]
        d50 = d_samples["d50_over_D"]
        d75 = d_samples["d75_over_D"]
        d90 = d_samples["d90_over_D"]

        Q_d_width = np.nan
        if np.isfinite(d25) and np.isfinite(d75):
            Q_d_width = float(d75 - d25)

        Q_d_skew = np.nan
        if np.isfinite(d10) and np.isfinite(d50) and np.isfinite(d90):
            Q_d_skew = float(((d90 - d50) - (d50 - d10)) / ((d90 - d10) + self.eps))

        return Q_d_width, Q_d_skew

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
    
    # ----------------------------
    # Harmonic/Spectral Analysis (Metric-Side not Denoising)
    # ----------------------------
    
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

    def _spectral_ratio_LF_over_HF(self, v: np.ndarray, Tbeat: float) -> float:
        """
        Low-frequency spectral ratio from the full one-beat positive-frequency spectrum:
          E_LF / E_HF = a_1 / sum_{n>=2} a_n,
        where a_n = |V_n|^2.
        """
        if (not np.isfinite(Tbeat)) or Tbeat <= 0:
            return np.nan

        if v.size == 0 or not np.any(np.isfinite(v)):
            return np.nan

        vv = np.where(np.isfinite(v), v, 0.0)
        n = vv.size
        if n < 3:
            return np.nan

        X = np.fft.rfft(vv)
        P = np.abs(X) ** 2
        if P.size < 3:
            return np.nan

        E_LF = float(P[1])
        E_HF = float(np.sum(P[2:]))
        if (not np.isfinite(E_LF)) or (not np.isfinite(E_HF)) or E_HF <= 0:
            return np.nan

        return float(E_LF / E_HF)

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
            "A2_m_interp": np.full((self.H_CUMSUM_INTERP_POINTS,), np.nan, dtype=float),
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

    # ----------------------------
    # Graphics / Diagnostics Support
    # ----------------------------

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
    
    # ----------------------------
    # Core metric kernel + registry
    # ----------------------------

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

        d_samples = self._normalized_cumulative_distance_samples(vv, Tbeat, m0)
        d10 = d_samples["d10_over_D"]
        d25 = d_samples["d25_over_D"]
        d50 = d_samples["d50_over_D"]
        d75 = d_samples["d75_over_D"]
        d90 = d_samples["d90_over_D"]

        E_LF_over_E_HF = self._spectral_ratio_LF_over_HF(vv, Tbeat)

        hp = self._harmonic_pack(vv, Tbeat)
        vb = hp["vb"]

        CF = self._crest_factor(vv)
        N_eff_over_T = self._n_eff_over_T(vv, Tbeat, m0)
        N_t_over_T = self._n_t_over_T(vv, Tbeat, m0)

        t_max_over_T, t_min_over_T, _, _ = self._peak_trough_times(vv)
        (
            S_rise,
            S_fall,
            t_rise_over_T,
            t_fall_over_T,
        ) = self._normalized_slopes_and_times(vv, Tbeat)

        Delta_DTI = self._delta_dti(vv, Tbeat, m0, t)
        gamma_t = self._gamma_t(vv, Tbeat, mu_t, sigma_t, m0, t)

        eta_h = self._explained_pulsatile_fraction(vv, vb)

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

        Q_d_width, Q_d_skew = self._d_quantile_shape_metrics(d_samples)

        v_end_over_vbar = self._late_cycle_mean_fraction(vv)
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
            "d10_over_D": float(d10) if np.isfinite(d10) else np.nan,
            "d25_over_D": float(d25) if np.isfinite(d25) else np.nan,
            "d50_over_D": float(d50) if np.isfinite(d50) else np.nan,
            "d75_over_D": float(d75) if np.isfinite(d75) else np.nan,
            "d90_over_D": float(d90) if np.isfinite(d90) else np.nan,
            "E_LF_over_E_HF": float(E_LF_over_E_HF)
            if np.isfinite(E_LF_over_E_HF)
            else np.nan,
            "t_max_over_T": float(t_max_over_T)
            if np.isfinite(t_max_over_T)
            else np.nan,
            "t_min_over_T": float(t_min_over_T)
            if np.isfinite(t_min_over_T)
            else np.nan,
            "S_rise": float(S_rise) if np.isfinite(S_rise) else np.nan,
            "S_fall": float(S_fall) if np.isfinite(S_fall) else np.nan,
            "t_rise_over_T": float(t_rise_over_T)
            if np.isfinite(t_rise_over_T)
            else np.nan,
            "t_fall_over_T": float(t_fall_over_T)
            if np.isfinite(t_fall_over_T)
            else np.nan,
            "Delta_DTI": float(Delta_DTI) if np.isfinite(Delta_DTI) else np.nan,
            "gamma_t": float(gamma_t) if np.isfinite(gamma_t) else np.nan,
            "CF": float(CF) if np.isfinite(CF) else np.nan,
            "N_eff_over_T": float(N_eff_over_T)
            if np.isfinite(N_eff_over_T)
            else np.nan,
            "N_t_over_T": float(N_t_over_T) if np.isfinite(N_t_over_T) else np.nan,
            "eta_h": float(eta_h) if np.isfinite(eta_h) else np.nan,
            "Q_t_skew": float(Q_t_skew) if np.isfinite(Q_t_skew) else np.nan,
            "Q_t_width": float(Q_t_width) if np.isfinite(Q_t_width) else np.nan,
            "Q_d_skew": float(Q_d_skew) if np.isfinite(Q_d_skew) else np.nan,
            "Q_d_width": float(Q_d_width) if np.isfinite(Q_d_width) else np.nan,
            "v_end_over_vbar": float(v_end_over_vbar)
            if np.isfinite(v_end_over_vbar)
            else np.nan,
            "E_slope": float(E_slope) if np.isfinite(E_slope) else np.nan,
        }

    @staticmethod
    def _metric_keys() -> list[list]:
        """
        Canonical manuscript-aligned scalar metrics.

        Each row is [key, compact_definition, unit, metric_family].
        Deprecated exploratory harmonic rolloff/support and phase-organization
        metrics are intentionally excluded from this public endpoint list.
        """
        return [
            [
                "mu_t",
                "VTI-weighted centroid time",
                "seconds",
                "timing_and_distribution",
            ],
            ["mu_t_over_T", "mu_t/T", "", "timing_and_distribution"],
            [
                "sigma_t",
                "VTI-weighted time spread",
                "seconds",
                "timing_and_distribution",
            ],
            ["sigma_t_over_T", "sigma_t/T", "", "timing_and_distribution"],
            [
                "gamma_t",
                "VTI-weighted temporal skewness",
                "",
                "timing_and_distribution",
            ],
            ["t10_over_T", "t_10/T", "", "timing_quantiles"],
            ["t25_over_T", "t_25/T", "", "timing_quantiles"],
            ["t50_over_T", "t_50/T", "", "timing_quantiles"],
            ["t75_over_T", "t_75/T", "", "timing_quantiles"],
            ["t90_over_T", "t_90/T", "", "timing_quantiles"],
            ["Q_t_width", "(t_75-t_25)/T", "", "timing_quantiles"],
            [
                "Q_t_skew",
                "((t_90-t_50)-(t_50-t_10))/(t_90-t_10)",
                "",
                "timing_quantiles",
            ],
            ["W50_over_T", "W_50/T", "", "crest_width"],
            ["W80_over_T", "W_80/T", "", "crest_width"],
            ["RI", "1-v_min/v_max", "", "pulsatility"],
            ["PI", "(v_max-v_min)/v_mean", "", "pulsatility"],
            ["R_VTI", "d(alpha*T)/(D-d(alpha*T))", "", "cumulative_distance_geometry"],
            ["SF_VTI", "d(alpha*T)/D", "", "cumulative_distance_geometry"],
            [
                "Delta_DTI",
                "integral_0^1(d(tau*T)/D - tau) d tau",
                "",
                "cumulative_distance_geometry",
            ],
            ["d10_over_D", "d(0.10*T)/D", "", "cumulative_distance_geometry"],
            ["d25_over_D", "d(0.25*T)/D", "", "cumulative_distance_geometry"],
            ["d50_over_D", "d(0.50*T)/D", "", "cumulative_distance_geometry"],
            ["d75_over_D", "d(0.75*T)/D", "", "cumulative_distance_geometry"],
            ["d90_over_D", "d(0.90*T)/D", "", "cumulative_distance_geometry"],
            ["Q_d_width", "(d_75-d_25)/D", "", "cumulative_distance_geometry"],
            [
                "Q_d_skew",
                "((d_90-d_50)-(d_50-d_10))/(d_90-d_10)",
                "",
                "cumulative_distance_geometry",
            ],
            ["t_max_over_T", "t_max/T", "", "kinetics_and_persistence"],
            ["t_min_over_T", "t_min/T", "", "kinetics_and_persistence"],
            ["S_rise", "T*max(dv/dt)/v_mean", "", "kinetics_and_persistence"],
            ["S_fall", "T*|min(dv/dt)|/v_mean", "", "kinetics_and_persistence"],
            ["t_rise_over_T", "t_rise/T", "", "kinetics_and_persistence"],
            ["t_fall_over_T", "t_fall/T", "", "kinetics_and_persistence"],
            ["v_end_over_vbar", "v_end/v_mean", "", "kinetics_and_persistence"],
            ["CF", "v_max/v_RMS", "", "kinetics_and_persistence"],
            ["E_LF_over_E_HF", "E_LF/E_HF", "", "spectral_and_reconstruction"],
            [
                "eta_h",
                "explained pulsatile fraction",
                "",
                "spectral_and_reconstruction",
            ],
            ["N_eff_over_T", "N_eff/T", "", "temporal_support"],
            ["N_t_over_T", "N_t/T", "", "temporal_support"],
            ["E_slope", "T^3/M0^2 * integral_0^T (dv/dt)^2 dt", "", "slope_energy"],
        ]

    # ----------------------------
    # Output packing
    # ----------------------------
    
    def _pack_segment_signal_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        raw_original_seg: np.ndarray,
        metric_input_seg: np.ndarray,
        bandlimited_seg: np.ndarray,
    ) -> None:
        """
        Store segment waveforms used for by_segment metrics.

        Signal arrays are stored as:
        (time, beat, branch, radius)

        Metric arrays are stored as:
        (beat, branch, radius)

        Therefore:
        signal[:, beat_idx, branch_idx, radius_idx]
        is the waveform used for:
        metric[beat_idx, branch_idx, radius_idx]
        """

        common_attrs = {
            "axis_order": ["time, beat, branch, radius"],
            "metric_alignment": [
                "signal[:, beat, branch, radius] corresponds to "
                "by_segment/*_segment metrics[beat, branch, radius]"
            ],
        }

        metrics[f"{vessel_prefix}/by_segment/signals/raw_original/value"] = with_attrs(
            np.asarray(raw_original_seg, dtype=float),
            {
                **common_attrs,
                "definition": [
                    "Original per-segment velocity waveform before denoising/filtering"
                ],
            },
        )

        metrics[f"{vessel_prefix}/by_segment/signals/metric_input/value"] = with_attrs(
            np.asarray(metric_input_seg, dtype=float),
            {
                **common_attrs,
                "definition": [
                    "Per-segment waveform actually passed to _compute_block_segment "
                    "for raw_segment metrics"
                ],
            },
        )

        metrics[f"{vessel_prefix}/by_segment/signals/metric_input_rectified/value"] = (
            with_attrs(
                self._rectify_keep_nan(metric_input_seg),
                {
                    **common_attrs,
                    "definition": [
                        "Rectified version of metric_input; this is the effective waveform "
                        "seen by _compute_metrics_1d after _rectify_keep_nan"
                    ],
                },
            )
        )

        metrics[f"{vessel_prefix}/by_segment/signals/bandlimited/value"] = with_attrs(
            np.asarray(bandlimited_seg, dtype=float),
            {
                **common_attrs,
                "definition": [
                    "Bandlimited per-segment waveform passed to _compute_block_segment "
                    "for bandlimited_segment metrics"
                ],
            },
        )

        metrics[f"{vessel_prefix}/by_segment/signals/bandlimited_rectified/value"] = (
            with_attrs(
                self._rectify_keep_nan(bandlimited_seg),
                {
                    **common_attrs,
                    "definition": [
                        "Rectified version of bandlimited signal; this is the effective "
                        "waveform seen by _compute_metrics_1d"
                    ],
                },
            )
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
                    "metric_family": [k[3]],
                    "latex_formula": [latex_formulas[k[0]]],
                },
            )

            metrics[f"{vessel_prefix}/global/bandlimited/{k[0]}"] = with_attrs(
                out_band[k[0]],
                {
                    "unit": [k[2]],
                    "definition": [k[1]],
                    "metric_family": [k[3]],
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

        graphics_raw = self._compute_graphics_support_block(v_raw_gl, T)
        graphics_band = self._compute_graphics_support_block(v_band_gl, T)

        diagnostic_graphics = {
            "A2_cumsum",
            "A2_m",
            "A2_cumsum_interp",
            "A2_m_interp",
            "m_50",
            "m_80",
            "rho_h",
            "w_h",
            "delta_phi_all",
            "t_phi_n",
            "t_phi_n_over_T",
            "phase_harmonics_used",
            "harmonic_phases",
            "harmonic_weights",
            "harmonic_energies_weights",
        }

        for name, arr in graphics_raw.items():
            if name in diagnostic_graphics:
                metrics[f"{vessel_prefix}/global/raw/diagnostics/{name}"] = arr
            else:
                metrics[f"{vessel_prefix}/global/raw/{name}"] = arr

        for name, arr in graphics_band.items():
            if name in diagnostic_graphics:
                metrics[f"{vessel_prefix}/global/bandlimited/diagnostics/{name}"] = arr
            else:
                metrics[f"{vessel_prefix}/global/bandlimited/{name}"] = arr

    # ----------------------------
    # Block-level aggregation
    # ----------------------------
    
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

    # ----------------------------
    # Main entry point
    # ----------------------------

    def run(self, h5file) -> ProcessResult:
        latex_formulas = {
            "mu_t": r"$\mu_t=\frac{\int_0^T v(t)t\,dt}{\int_0^T v(t)\,dt}$",
            "mu_t_over_T": r"$\mu_t/T$",
            "sigma_t": r"$\sigma_t=\sqrt{\frac{\int_0^T v(t)(t-\mu_t)^2\,dt}{\int_0^T v(t)\,dt}}$",
            "sigma_t_over_T": r"$\sigma_t/T$",
            "gamma_t": r"$\gamma_t=\frac{\int_0^T v(t)(t-\mu_t)^3\,dt}{M_0\sigma_t^3}$",
            "t10_over_T": r"$t_{10}/T$",
            "t25_over_T": r"$t_{25}/T$",
            "t50_over_T": r"$t_{50}/T$",
            "t75_over_T": r"$t_{75}/T$",
            "t90_over_T": r"$t_{90}/T$",
            "Q_t_width": r"$Q_{t,\mathrm{width}}=(t_{75}-t_{25})/T$",
            "Q_t_skew": r"$Q_{t,\mathrm{skew}}=\frac{(t_{90}-t_{50})-(t_{50}-t_{10})}{t_{90}-t_{10}}$",
            "W50_over_T": r"$W_{50}/T$",
            "W80_over_T": r"$W_{80}/T$",
            "RI": r"$\mathrm{RI}=1-v_{\min}/v_{\max}$",
            "PI": r"$\mathrm{PI}=(v_{\max}-v_{\min})/\bar v$",
            "R_VTI": r"$\mathrm{R}_{\mathrm{VTI}}=\frac{d(\alpha T)}{D-d(\alpha T)}$",
            "SF_VTI": r"$\mathrm{SF}_{\mathrm{VTI}}=\frac{d(\alpha T)}{D}$",
            "Delta_DTI": r"$\Delta_{\mathrm{DTI}}=\int_0^1\left[\frac{d(\tau T)}{D}-\tau\right]d\tau$",
            "d10_over_D": r"$d_{10}/D$",
            "d25_over_D": r"$d_{25}/D$",
            "d50_over_D": r"$d_{50}/D$",
            "d75_over_D": r"$d_{75}/D$",
            "d90_over_D": r"$d_{90}/D$",
            "Q_d_width": r"$Q_{d,\mathrm{width}}=(d_{75}-d_{25})/D$",
            "Q_d_skew": r"$Q_{d,\mathrm{skew}}=\frac{(d_{90}-d_{50})-(d_{50}-d_{10})}{d_{90}-d_{10}}$",
            "t_max_over_T": r"$t_{\max}/T$",
            "t_min_over_T": r"$t_{\min}/T$",
            "S_rise": r"$S_{\mathrm{rise}}=\frac{T}{\bar v}\max_t\dot v(t)$",
            "S_fall": r"$S_{\mathrm{fall}}=\frac{T}{\bar v}|\min_t\dot v(t)|$",
            "t_rise_over_T": r"$t_{\mathrm{rise}}/T$",
            "t_fall_over_T": r"$t_{\mathrm{fall}}/T$",
            "v_end_over_vbar": r"$\bar v_{\mathrm{end}}/\bar v$",
            "CF": r"$\mathrm{CF}=v_{\max}/v_{\mathrm{RMS}}$",
            "E_LF_over_E_HF": r"$E_{\mathrm{LF}}/E_{\mathrm{HF}}$",
            "eta_h": r"$\eta_h=1-\frac{\int_0^T (v(t)-v_H(t))^2\,dt}{\int_0^T (v(t)-\bar v)^2\,dt}$",
            "N_eff_over_T": r"$N_{\mathrm{eff}}/T=(T\int_0^T p(t)^2\,dt)^{-1}$",
            "N_t_over_T": r"$N_t/T=\exp[-\int_0^T p(t)\ln(Tp(t))\,dt]$",
            "E_slope": r"$E_{\mathrm{slope}}=\frac{T^3}{M_0^2}\int_0^T \dot v(t)^2\,dt$",
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
                v_raw_seg_metric_input = v_raw_seg
                if vessel_prefix == "artery":
                    v_raw_seg_metric_input, denoise_diag = self._denoise_segment_block(
                        v_raw_seg
                    )
                    metrics["artery/by_segment/denoising/original_vs_filtered_corr"] = (
                        with_attrs(
                            denoise_diag["original_vs_filtered_corr"],
                            {
                                "definition": (
                                    "Pearson correlation between each original "
                                    "arterial segment pulse and the denoised pulse"
                                ),
                                "axis_order": "beat, branch, radius",
                            },
                        )
                    )
                    metrics["artery/by_segment/denoising/finite_fraction"] = with_attrs(
                        denoise_diag["finite_fraction"],
                        {
                            "definition": (
                                "Fraction of finite samples in each original "
                                "arterial segment pulse"
                            ),
                            "axis_order": "beat, branch, radius",
                        },
                    )
                    metrics["artery/by_segment/denoising/status_code"] = with_attrs(
                        denoise_diag["status_code"],
                        {
                            "definition": (
                                "0 denoised, 1 all_nan, 2 too_sparse, 3 low_variance"
                            ),
                            "axis_order": "beat, branch, radius",
                        },
                    )
                    metrics["artery/by_segment/denoising/params/method"] = (
                        "mean consensus of harmonic_lowpass, gaussian, savitzky_golay"
                    )
                    metrics["artery/by_segment/denoising/params/min_valid_samples"] = (
                        np.asarray(self.denoise_min_valid_samples, dtype=int)
                    )
                    metrics["artery/by_segment/denoising/params/min_valid_fraction"] = (
                        np.asarray(self.denoise_min_valid_fraction, dtype=float)
                    )
                    metrics["artery/by_segment/denoising/params/harmonic_count"] = (
                        np.asarray(self.denoise_harmonic_count, dtype=int)
                    )
                    metrics[
                        "artery/by_segment/denoising/params/gaussian_sigma_samples"
                    ] = np.asarray(self.denoise_gaussian_sigma_samples, dtype=float)
                    metrics["artery/by_segment/denoising/params/savgol_window"] = (
                        np.asarray(self.denoise_savgol_window, dtype=int)
                    )
                    metrics["artery/by_segment/denoising/params/savgol_polyorder"] = (
                        np.asarray(self.denoise_savgol_polyorder, dtype=int)
                    )
                self._pack_segment_signal_outputs(
                    metrics=metrics,
                    vessel_prefix=vessel_prefix,
                    raw_original_seg=v_raw_seg,
                    metric_input_seg=v_raw_seg_metric_input,
                    bandlimited_seg=v_band_seg,
                )

                self._pack_segment_outputs(
                    metrics=metrics,
                    vessel_prefix=vessel_prefix,
                    v_raw_seg=v_raw_seg_metric_input,
                    v_band_seg=v_band_seg,
                    T=T,
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
