import numpy as np


class ReferenceGateMixin:
    """Pre-graph median-reference quality gate for arterial pulse waveforms."""

    @staticmethod
    def _laplacian_pointwise_median_reference(v_block: np.ndarray) -> np.ndarray:
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

    def _prepare_laplacian_reference_for_gate(
        self,
        v_block: np.ndarray,
        filled: np.ndarray,
        valid_flat_indices: np.ndarray,
        n_time: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Build the dense reference signals used by the pre-graph quality gate.

        Script version of the math:
        1. reference[t] = median over all segment waveforms at time t, ignoring NaNs.
        2. If that pointwise median is too sparse, fall back to the median of the
           already-interpolated valid pulses.
        3. Interpolate any remaining missing reference samples so lag scoring uses
           dense vectors.
        4. Return two centered versions:
           - pRMS-normalized reference for the 0.80 quality gate.
           - L2-normalized reference for the existing phase-alignment code.
        """
        reference = self._laplacian_pointwise_median_reference(v_block)
        finite_count = int(np.sum(np.isfinite(reference)))
        if not self._denoise_has_enough_valid_samples(finite_count, n_time):
            reference = np.nanmedian(filled[valid_flat_indices], axis=0)

        reference = self._interpolate_missing_samples(reference)
        reference_gate_norm = self._normalize_prms_1d(reference)
        reference_norm = self._normalize_1d(reference)

        if reference_gate_norm is None or reference_norm is None:
            reference = np.nanmean(filled[valid_flat_indices], axis=0)
            reference_gate_norm = self._normalize_prms_1d(reference)
            reference_norm = self._normalize_1d(reference)

        return reference, reference_gate_norm, reference_norm

    def _apply_laplacian_reference_gate(
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
        min_reference_corr = float(self.laplacian_reference_min_corr)

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
