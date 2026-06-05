import numpy as np

from .core.base import ProcessResult, registerPipeline, with_attrs
from .waveform_shape_metrics_denoised import ArterialSegExample as BaseDenoisedMetrics


@registerPipeline(name="waveform_shape_metrics_denoised_correl")
class WaveformShapeMetricsDenoisedCorrel(BaseDenoisedMetrics):
    """
    Waveform-shape metrics after cross-correlation denoising of arterial pulses.

    The denoiser treats each arterial segment pulse
    ``v[:, beat, branch, radius]`` as one realization of the same cardiac
    pulse family. For each pulse, it finds highly correlated realizations,
    circularly phase-aligns them, rescales their amplitude to the target pulse,
    and blends a weighted local template back into the original pulse.
    """

    description = (
        "Waveform-shape metrics with cross-correlation non-local arterial pulse "
        "denoising across beats, branches, and radii."
    )

    # The requested source for the correlation ensemble. The inherited pipeline
    # still exposes both raw and bandlimited segment signals in the output.
    correlation_use_bandlimited_input = True

    correlation_top_k_neighbors = 12
    correlation_min_corr = 0.65
    correlation_corr_power = 3.0
    correlation_blend_alpha = 0.75
    correlation_self_weight = 0.25

    correlation_max_lag_fraction = 0.12
    correlation_lag_sigma_fraction = 0.06

    correlation_match_neighbor_amplitude = True
    correlation_preserve_nan_mask = True
    arterial_metrics_use_filtered_signal = True

    # Set either sigma to a positive value to add index-space spatial locality.
    # A value <= 0 disables that factor.
    correlation_branch_sigma = 0.0
    correlation_radius_sigma = 0.0

    def run(self, h5file) -> ProcessResult:
        self._last_correlation_denoise_diag = None

        original_raw_segment_input = self.v_raw_segment_input
        if self.correlation_use_bandlimited_input:
            self.v_raw_segment_input = self.v_band_segment_input

        try:
            result = super().run(h5file)
        finally:
            self.v_raw_segment_input = original_raw_segment_input

        self._replace_legacy_denoising_params(result.metrics)
        self._pack_correlation_denoising_outputs(result.metrics)

        attrs = dict(result.attrs or {})
        attrs.update(
            {
                "correlation_denoising": (
                    "Arterial metric_input pulses are denoised by weighted "
                    "cross-correlation averaging over beat, branch, and radius "
                    "realizations."
                ),
                "correlation_denoising_input_path": (
                    self.v_band_segment_input
                    if self.correlation_use_bandlimited_input
                    else original_raw_segment_input
                ),
            }
        )
        result.attrs = attrs
        return result

    def _pack_segment_signal_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        raw_original_seg: np.ndarray,
        metric_input_seg: np.ndarray,
        bandlimited_seg: np.ndarray,
    ) -> None:
        super()._pack_segment_signal_outputs(
            metrics=metrics,
            vessel_prefix=vessel_prefix,
            raw_original_seg=raw_original_seg,
            metric_input_seg=metric_input_seg,
            bandlimited_seg=bandlimited_seg,
        )

        if (
            vessel_prefix != "artery"
            or not bool(self.arterial_metrics_use_filtered_signal)
        ):
            return

        common_attrs = {
            "axis_order": ["time, beat, branch, radius"],
            "metric_alignment": [
                "signal[:, beat, branch, radius] corresponds to "
                "by_segment/*_segment metrics[beat, branch, radius]"
            ],
        }
        metrics["artery/by_segment/signals/filtered_for_metrics/value"] = with_attrs(
            np.asarray(metric_input_seg, dtype=float),
            {
                **common_attrs,
                "definition": [
                    "Filtered arterial waveform used for both raw_segment and "
                    "bandlimited_segment metrics in this pipeline."
                ],
            },
        )
        metrics[
            "artery/by_segment/signals/filtered_for_metrics_rectified/value"
        ] = with_attrs(
            self._rectify_keep_nan(metric_input_seg),
            {
                **common_attrs,
                "definition": [
                    "Rectified filtered arterial waveform seen by "
                    "_compute_metrics_1d for both raw_segment and "
                    "bandlimited_segment metrics."
                ],
            },
        )

        band_key = "artery/by_segment/signals/bandlimited/value"
        band_value = metrics.get(band_key)
        if hasattr(band_value, "attrs") and band_value.attrs is not None:
            band_value.attrs["definition"] = [
                "Original bandlimited arterial per-segment waveform. In this "
                "pipeline, bandlimited_segment metrics are computed from "
                "signals/filtered_for_metrics/value."
            ]

    def _pack_segment_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_raw_seg: np.ndarray,
        v_band_seg: np.ndarray,
        T: np.ndarray,
    ) -> None:
        metric_band_seg = v_band_seg
        if vessel_prefix == "artery" and bool(self.arterial_metrics_use_filtered_signal):
            metric_band_seg = v_raw_seg

        super()._pack_segment_outputs(
            metrics=metrics,
            vessel_prefix=vessel_prefix,
            v_raw_seg=v_raw_seg,
            v_band_seg=metric_band_seg,
            T=T,
        )

        if vessel_prefix == "artery":
            metrics[
                "artery/by_segment/denoising/params/metrics_use_filtered_signal"
            ] = np.asarray(int(bool(self.arterial_metrics_use_filtered_signal)))
            metrics[
                "artery/by_segment/denoising/params/bandlimited_segment_metrics_source"
            ] = (
                "filtered_for_metrics"
                if bool(self.arterial_metrics_use_filtered_signal)
                else "bandlimited"
            )

    def _denoise_segment_block(self, v_block: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Denoise a ``(time, beat, branch, radius)`` arterial block by
        cross-correlation weighted averaging of similar pulse realizations.
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

        best_neighbor_corr = np.full(pulse_shape, np.nan, dtype=float)
        best_neighbor_lag = np.full(pulse_shape, np.nan, dtype=float)
        neighbor_count = np.zeros(pulse_shape, dtype=np.int32)
        effective_neighbor_count = np.zeros(pulse_shape, dtype=float)
        template_corr = np.full(pulse_shape, np.nan, dtype=float)
        mean_abs_change = np.full(pulse_shape, np.nan, dtype=float)

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
                    elif not self._denoise_has_enough_valid_samples(
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
        max_lag = self._correlation_max_lag_samples(n_time)
        lags = np.arange(-max_lag, max_lag + 1, dtype=int)

        for flat_i in valid_flat_indices:
            index_i = coords[flat_i]
            candidates = []
            target_norm = normalized[flat_i]

            for flat_j in valid_flat_indices:
                if flat_i == flat_j:
                    continue

                corr, lag = self._best_circular_lag_corr(
                    target_norm, normalized[flat_j], lags
                )
                if (not np.isfinite(corr)) or corr < float(self.correlation_min_corr):
                    continue

                weight = self._correlation_neighbor_weight(
                    corr=corr,
                    lag=lag,
                    n_time=n_time,
                    coord_i=index_i,
                    coord_j=coords[flat_j],
                )
                if weight <= 0.0 or not np.isfinite(weight):
                    continue

                candidates.append((weight, corr, lag, flat_j))

            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[: int(self.correlation_top_k_neighbors)]

            beat_idx, branch_idx, radius_idx = index_i
            original = v_block[:, beat_idx, branch_idx, radius_idx]
            finite_mask = finite_masks[flat_i]

            if len(candidates) == 0:
                restored = filled[flat_i].copy()
                restored[~finite_mask] = np.nan
                out[:, beat_idx, branch_idx, radius_idx] = restored
                status_code[index_i] = 4
                continue

            weights = [float(self.correlation_self_weight)]
            aligned = [filled[flat_i]]

            for weight, _, lag, flat_j in candidates:
                neighbor = np.roll(filled[flat_j], int(lag))
                if self.correlation_match_neighbor_amplitude:
                    neighbor = self._match_amplitude(
                        neighbor=neighbor,
                        neighbor_mean=means[flat_j],
                        neighbor_std=stds[flat_j],
                        target_mean=means[flat_i],
                        target_std=stds[flat_i],
                    )
                weights.append(float(weight))
                aligned.append(neighbor)

            weight_arr = np.asarray(weights, dtype=float)
            aligned_arr = np.asarray(aligned, dtype=float)
            template = np.average(aligned_arr, axis=0, weights=weight_arr)

            alpha = float(np.clip(self.correlation_blend_alpha, 0.0, 1.0))
            denoised = (1.0 - alpha) * filled[flat_i] + alpha * template
            denoised = self._clip_to_input_range(denoised, original)
            if self.correlation_preserve_nan_mask:
                denoised[~finite_mask] = np.nan

            out[:, beat_idx, branch_idx, radius_idx] = denoised

            candidate_corrs = np.asarray([c[1] for c in candidates], dtype=float)
            candidate_lags = np.asarray([c[2] for c in candidates], dtype=float)
            candidate_weights = np.asarray([c[0] for c in candidates], dtype=float)

            corr_original[index_i] = self._pearson_corr(original, denoised)
            best_idx = int(np.nanargmax(candidate_corrs))
            best_neighbor_corr[index_i] = float(candidate_corrs[best_idx])
            best_neighbor_lag[index_i] = float(candidate_lags[best_idx])
            neighbor_count[index_i] = len(candidates)
            effective_neighbor_count[index_i] = self._effective_count(
                candidate_weights
            )
            template_corr[index_i] = self._pearson_corr(filled[flat_i], template)
            mean_abs_change[index_i] = self._safe_mean_abs_change(original, denoised)
            status_code[index_i] = 0

        diagnostics = {
            "original_vs_filtered_corr": corr_original,
            "finite_fraction": finite_fraction,
            "status_code": status_code,
            "best_neighbor_corr": best_neighbor_corr,
            "best_neighbor_lag_samples": best_neighbor_lag,
            "neighbor_count": neighbor_count,
            "effective_neighbor_count": effective_neighbor_count,
            "template_corr": template_corr,
            "mean_abs_change": mean_abs_change,
        }
        self._last_correlation_denoise_diag = diagnostics
        return out, diagnostics

    def _interpolate_missing_samples(self, pulse: np.ndarray) -> np.ndarray:
        pulse = np.asarray(pulse, dtype=float)
        finite_mask = np.isfinite(pulse)
        x = np.arange(pulse.size, dtype=float)
        return np.interp(x, x[finite_mask], pulse[finite_mask])

    def _correlation_max_lag_samples(self, n_time: int) -> int:
        max_lag = int(round(float(self.correlation_max_lag_fraction) * n_time))
        return max(0, min(max_lag, max(0, n_time // 2 - 1)))

    @staticmethod
    def _best_circular_lag_corr(
        target_norm: np.ndarray, neighbor_norm: np.ndarray, lags: np.ndarray
    ) -> tuple[float, int]:
        best_corr = -np.inf
        best_lag = 0
        for lag in lags:
            corr = float(np.dot(target_norm, np.roll(neighbor_norm, int(lag))))
            if corr > best_corr:
                best_corr = corr
                best_lag = int(lag)
        return best_corr, best_lag

    def _correlation_neighbor_weight(
        self,
        corr: float,
        lag: int,
        n_time: int,
        coord_i: tuple[int, int, int],
        coord_j: tuple[int, int, int],
    ) -> float:
        corr_weight = max(float(corr), 0.0) ** float(self.correlation_corr_power)

        lag_sigma = float(self.correlation_lag_sigma_fraction) * float(n_time)
        if lag_sigma > self.eps:
            lag_weight = float(np.exp(-0.5 * (float(lag) / lag_sigma) ** 2))
        else:
            lag_weight = 1.0

        spatial_weight = 1.0
        branch_sigma = float(self.correlation_branch_sigma)
        radius_sigma = float(self.correlation_radius_sigma)

        if branch_sigma > 0.0:
            dk = float(coord_i[1] - coord_j[1])
            spatial_weight *= float(np.exp(-0.5 * (dk / branch_sigma) ** 2))

        if radius_sigma > 0.0:
            dr = float(coord_i[2] - coord_j[2])
            spatial_weight *= float(np.exp(-0.5 * (dr / radius_sigma) ** 2))

        return corr_weight * lag_weight * spatial_weight

    def _match_amplitude(
        self,
        neighbor: np.ndarray,
        neighbor_mean: float,
        neighbor_std: float,
        target_mean: float,
        target_std: float,
    ) -> np.ndarray:
        if (
            not np.isfinite(neighbor_mean)
            or not np.isfinite(neighbor_std)
            or neighbor_std <= self.eps
            or not np.isfinite(target_mean)
            or not np.isfinite(target_std)
        ):
            return neighbor
        return target_mean + (neighbor - neighbor_mean) * (target_std / neighbor_std)

    @staticmethod
    def _effective_count(weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = weights[np.isfinite(weights) & (weights > 0.0)]
        if weights.size == 0:
            return 0.0
        return float((np.sum(weights) ** 2) / np.sum(weights * weights))

    @staticmethod
    def _safe_mean_abs_change(original: np.ndarray, denoised: np.ndarray) -> float:
        mask = np.isfinite(original) & np.isfinite(denoised)
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs(denoised[mask] - original[mask])))

    def _replace_legacy_denoising_params(self, metrics: dict) -> None:
        for key in list(metrics):
            if key.startswith("artery/by_segment/denoising/params/"):
                del metrics[key]

        status_key = "artery/by_segment/denoising/status_code"
        status_value = metrics.get(status_key)
        if hasattr(status_value, "attrs") and status_value.attrs is not None:
            status_value.attrs["definition"] = (
                "0 denoised, 1 all_nan, 2 too_sparse, 3 low_variance, "
                "4 no_eligible_correlation_neighbors"
            )

    def _pack_correlation_denoising_outputs(self, metrics: dict) -> None:
        diag = getattr(self, "_last_correlation_denoise_diag", None)
        if not diag:
            return

        base = "artery/by_segment/denoising"
        attrs = {"axis_order": "beat, branch, radius"}

        metrics[f"{base}/best_neighbor_corr"] = with_attrs(
            np.asarray(diag["best_neighbor_corr"], dtype=float),
            {
                **attrs,
                "definition": (
                    "Highest cross-correlation among retained neighbor pulses "
                    "after lag search."
                ),
            },
        )
        metrics[f"{base}/best_neighbor_lag_samples"] = with_attrs(
            np.asarray(diag["best_neighbor_lag_samples"], dtype=float),
            {
                **attrs,
                "definition": (
                    "Circular lag, in samples, applied to the most correlated "
                    "neighbor pulse to align it with the target pulse."
                ),
            },
        )
        metrics[f"{base}/neighbor_count"] = with_attrs(
            np.asarray(diag["neighbor_count"], dtype=np.int32),
            {
                **attrs,
                "definition": (
                    "Number of retained neighbor pulses used in the weighted "
                    "correlation template."
                ),
            },
        )
        metrics[f"{base}/effective_neighbor_count"] = with_attrs(
            np.asarray(diag["effective_neighbor_count"], dtype=float),
            {
                **attrs,
                "definition": (
                    "Inverse-concentration effective count of retained neighbor "
                    "weights."
                ),
            },
        )
        metrics[f"{base}/template_corr"] = with_attrs(
            np.asarray(diag["template_corr"], dtype=float),
            {
                **attrs,
                "definition": (
                    "Pearson correlation between the target filled pulse and "
                    "its weighted aligned neighbor template."
                ),
            },
        )
        metrics[f"{base}/mean_abs_change"] = with_attrs(
            np.asarray(diag["mean_abs_change"], dtype=float),
            {
                **attrs,
                "definition": (
                    "Mean absolute pointwise change between the original pulse "
                    "and the correlation-denoised pulse."
                ),
            },
        )

        params = {
            "use_bandlimited_input": int(bool(self.correlation_use_bandlimited_input)),
            "top_k_neighbors": int(self.correlation_top_k_neighbors),
            "min_corr": float(self.correlation_min_corr),
            "corr_power": float(self.correlation_corr_power),
            "blend_alpha": float(self.correlation_blend_alpha),
            "self_weight": float(self.correlation_self_weight),
            "max_lag_fraction": float(self.correlation_max_lag_fraction),
            "lag_sigma_fraction": float(self.correlation_lag_sigma_fraction),
            "match_neighbor_amplitude": int(
                bool(self.correlation_match_neighbor_amplitude)
            ),
            "preserve_nan_mask": int(bool(self.correlation_preserve_nan_mask)),
            "metrics_use_filtered_signal": int(
                bool(self.arterial_metrics_use_filtered_signal)
            ),
            "bandlimited_segment_metrics_source": (
                "filtered_for_metrics"
                if bool(self.arterial_metrics_use_filtered_signal)
                else "bandlimited"
            ),
            "branch_sigma": float(self.correlation_branch_sigma),
            "radius_sigma": float(self.correlation_radius_sigma),
        }
        for name, value in params.items():
            metrics[f"{base}/params/{name}"] = np.asarray(value)
