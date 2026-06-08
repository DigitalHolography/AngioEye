import numpy as np
from scipy.sparse.csgraph import connected_components

from ..core.base import ProcessResult, with_attrs
from ..waveform_shape_metrics_denoised import ArterialSegExample as BaseDenoisedMetrics
from ..waveform_shape_metrics_denoised_correl import WaveformShapeMetricsDenoisedCorrel
from .reference_gate import ReferenceGateMixin


class WaveformShapeMetricsDenoisedLaplaceBase(
    ReferenceGateMixin, WaveformShapeMetricsDenoisedCorrel
):
    """
    Waveform-shape metrics after graph-Laplacian denoising of arterial pulses.

    Each arterial segment pulse ``v[:, beat, branch, radius]`` is treated as a
    graph vertex. Edges connect pulse realizations with similar phase-aligned
    morphology. The denoised ensemble solves

        min_X ||X - Y||_F^2 + gamma * Tr(X^T L X)

    where rows are pulse realizations, columns are time samples, and ``L`` is
    the weighted graph Laplacian.
    """

    description = (
        "Waveform-shape metrics with graph-Laplacian arterial pulse denoising "
        "across beats, branches, and radii."
    )

    laplacian_use_bandlimited_input = True

    laplacian_gamma = 0.75
    laplacian_min_corr = 0.70
    laplacian_reference_min_corr = 0.80
    laplacian_corr_power = 3.0

    laplacian_max_lag_fraction = 0.12
    laplacian_lag_sigma_fraction = 0.06

    laplacian_preserve_nan_mask = True
    laplacian_restore_pulse_scale = True
    laplacian_clip_output = False

    # Optional index-space locality. Values <= 0 disable the factor.
    laplacian_branch_sigma = 2.0
    laplacian_radius_sigma = 2.0

    def run(self, h5file) -> ProcessResult:
        self._last_laplacian_denoise_diag = None

        original_raw_segment_input = self.v_raw_segment_input
        if self.laplacian_use_bandlimited_input:
            self.v_raw_segment_input = self.v_band_segment_input

        try:
            result = BaseDenoisedMetrics.run(self, h5file)
        finally:
            self.v_raw_segment_input = original_raw_segment_input

        self._replace_laplacian_denoising_params(result.metrics)
        self._pack_laplacian_denoising_outputs(result.metrics)

        attrs = dict(result.attrs or {})
        attrs.update(
            {
                "laplacian_denoising": (
                    "Arterial metric_input pulses are denoised by graph-Laplacian "
                    "Tikhonov smoothing over phase-aligned beat, branch, and "
                    "radius realizations."
                ),
                "laplacian_denoising_input_path": (
                    self.v_band_segment_input
                    if self.laplacian_use_bandlimited_input
                    else original_raw_segment_input
                ),
            }
        )
        result.attrs = attrs
        return result

    def _denoise_segment_block(self, v_block: np.ndarray) -> tuple[np.ndarray, dict]:
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
        n_valid = int(valid_flat_indices.size)
        max_lag = self._laplacian_max_lag_samples(n_time)
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
            diagnostics = self._laplacian_diagnostics(
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
            self._last_laplacian_denoise_diag = diagnostics
            return out, diagnostics

        # Reference-correlation prefilter.
        #
        # This is the "is this pulse an arterial-shaped waveform?" gate.
        # It happens before graph construction, so rejected pulses never become
        # graph vertices and remain NaN in the downstream metric input.
        reference, reference_gate_norm, reference_norm = (
            self._prepare_laplacian_reference_for_gate(
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
            diagnostics = self._laplacian_diagnostics(
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
            self._last_laplacian_denoise_diag = diagnostics
            return out, diagnostics

        valid_flat_indices, reference_kept_count, reference_rejected_count = (
            self._apply_laplacian_reference_gate(
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
            diagnostics = self._laplacian_diagnostics(
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
            self._last_laplacian_denoise_diag = diagnostics
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

        W = self._build_laplacian_weight_matrix(
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
            diagnostics = self._laplacian_diagnostics(
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
            self._last_laplacian_denoise_diag = diagnostics
            return out, diagnostics

        gamma = float(max(self.laplacian_gamma, 0.0))
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
            if self.laplacian_clip_output:
                denoised = self._clip_to_input_range(denoised, original)
            if self.laplacian_preserve_nan_mask:
                denoised[~finite_mask] = np.nan

            out[:, beat_idx, branch_idx, radius_idx] = denoised
            corr_original[coords[flat_i]] = self._pearson_corr(original, denoised)
            mean_abs_change[coords[flat_i]] = self._safe_mean_abs_change(
                original, denoised
            )
            if component_sizes[int(component_labels[row_idx])] <= 1:
                status_code[coords[flat_i]] = 6
            elif component_failed[row_idx]:
                status_code[coords[flat_i]] = 5
            else:
                status_code[coords[flat_i]] = 0

        diagnostics = self._laplacian_diagnostics(
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
        self._last_laplacian_denoise_diag = diagnostics
        return out, diagnostics

    def _build_laplacian_weight_matrix(
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
                if (not np.isfinite(corr)) or corr < float(self.laplacian_min_corr):
                    continue

                weight = self._laplacian_neighbor_weight(
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

    def _laplacian_neighbor_weight(
        self,
        corr: float,
        lag_delta: int,
        n_time: int,
        coord_i: tuple[int, int, int],
        coord_j: tuple[int, int, int],
    ) -> float:
        min_corr = float(self.laplacian_min_corr)
        corr_score = (float(corr) - min_corr) / max(1.0 - min_corr, self.eps)
        corr_score = float(np.clip(corr_score, 0.0, 1.0))
        corr_weight = corr_score ** float(self.laplacian_corr_power)

        lag_sigma = float(self.laplacian_lag_sigma_fraction) * float(n_time)
        if lag_sigma > self.eps:
            lag_weight = float(np.exp(-0.5 * (float(lag_delta) / lag_sigma) ** 2))
        else:
            lag_weight = 1.0

        spatial_weight = 1.0
        branch_sigma = float(self.laplacian_branch_sigma)
        radius_sigma = float(self.laplacian_radius_sigma)

        if branch_sigma > 0.0:
            dk = float(coord_i[1] - coord_j[1])
            spatial_weight *= float(np.exp(-0.5 * (dk / branch_sigma) ** 2))

        if radius_sigma > 0.0:
            dr = float(coord_i[2] - coord_j[2])
            spatial_weight *= float(np.exp(-0.5 * (dr / radius_sigma) ** 2))

        return corr_weight * lag_weight * spatial_weight

    def _laplacian_max_lag_samples(self, n_time: int) -> int:
        max_lag = int(round(float(self.laplacian_max_lag_fraction) * n_time))
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
            self.laplacian_restore_pulse_scale
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
            self.laplacian_restore_pulse_scale
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
    def _laplacian_diagnostics(
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

    def _replace_laplacian_denoising_params(self, metrics: dict) -> None:
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

    def _pack_laplacian_denoising_outputs(self, metrics: dict) -> None:
        diag = getattr(self, "_last_laplacian_denoise_diag", None)
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
            "use_bandlimited_input": int(bool(self.laplacian_use_bandlimited_input)),
            "gamma": float(self.laplacian_gamma),
            "min_corr": float(self.laplacian_min_corr),
            "reference_min_corr": float(self.laplacian_reference_min_corr),
            "corr_power": float(self.laplacian_corr_power),
            "max_lag_fraction": float(self.laplacian_max_lag_fraction),
            "lag_sigma_fraction": float(self.laplacian_lag_sigma_fraction),
            "preserve_nan_mask": int(bool(self.laplacian_preserve_nan_mask)),
            "restore_pulse_scale": int(bool(self.laplacian_restore_pulse_scale)),
            "clip_output": int(bool(self.laplacian_clip_output)),
            "metrics_use_filtered_signal": int(
                bool(self.arterial_metrics_use_filtered_signal)
            ),
            "bandlimited_segment_metrics_source": (
                "filtered_for_metrics"
                if bool(self.arterial_metrics_use_filtered_signal)
                else "bandlimited"
            ),
            "branch_sigma": float(self.laplacian_branch_sigma),
            "radius_sigma": float(self.laplacian_radius_sigma),
        }
        for name, value in params.items():
            metrics[f"{base}/params/{name}"] = np.asarray(value)
