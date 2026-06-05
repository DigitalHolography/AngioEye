import numpy as np

from .core.base import ProcessResult, registerPipeline, with_attrs
from .waveform_shape_metrics_denoised import ArterialSegExample as BaseDenoisedMetrics
from .waveform_shape_metrics_denoised_laplace import WaveformShapeMetricsDenoisedLaplace


@registerPipeline(name="waveform_shape_metrics_denoised_joint")
class WaveformShapeMetricsDenoisedJoint(WaveformShapeMetricsDenoisedLaplace):
    """
    Waveform-shape metrics after joint pulse-graph and temporal-graph denoising.

    Rows are arterial pulse realizations ``(beat, branch, radius)`` and columns
    are within-beat time samples. The denoised aligned ensemble solves

        min_X ||X - Y||_F^2
            + gamma_p * Tr(X^T L_p X)
            + gamma_t * Tr(X L_t X^T)

    where ``L_p`` is the pulse-similarity graph Laplacian and ``L_t`` is the
    within-pulse temporal graph Laplacian.
    """

    description = (
        "Waveform-shape metrics with joint pulse-graph and temporal-graph "
        "arterial pulse denoising."
    )

    joint_use_bandlimited_input = True

    joint_pulse_gamma = 0.75
    joint_temporal_gamma = 0.03
    joint_top_k_neighbors = 6
    joint_min_corr = 0.70
    joint_corr_power = 3.0
    joint_self_jitter = 1.0e-8

    joint_max_lag_fraction = 0.12
    joint_lag_sigma_fraction = 0.06

    joint_blend_alpha = 1.0
    joint_temporal_cycle = True
    joint_preserve_nan_mask = True
    joint_restore_pulse_scale = True
    joint_clip_output = False

    # Optional index-space locality. Values <= 0 disable the factor.
    joint_branch_sigma = 2.0
    joint_radius_sigma = 2.0

    # Aliases used by inherited graph-construction helpers.
    laplacian_top_k_neighbors = joint_top_k_neighbors
    laplacian_min_corr = joint_min_corr
    laplacian_corr_power = joint_corr_power
    laplacian_max_lag_fraction = joint_max_lag_fraction
    laplacian_lag_sigma_fraction = joint_lag_sigma_fraction
    laplacian_branch_sigma = joint_branch_sigma
    laplacian_radius_sigma = joint_radius_sigma
    laplacian_restore_pulse_scale = joint_restore_pulse_scale

    def run(self, h5file) -> ProcessResult:
        self._sync_laplacian_aliases()
        self._last_joint_denoise_diag = None

        original_raw_segment_input = self.v_raw_segment_input
        if self.joint_use_bandlimited_input:
            self.v_raw_segment_input = self.v_band_segment_input

        try:
            result = BaseDenoisedMetrics.run(self, h5file)
        finally:
            self.v_raw_segment_input = original_raw_segment_input

        self._replace_joint_denoising_params(result.metrics)
        self._pack_joint_denoising_outputs(result.metrics)

        attrs = dict(result.attrs or {})
        attrs.update(
            {
                "joint_denoising": (
                    "Arterial metric_input pulses are denoised by joint "
                    "pulse-graph and temporal-graph Tikhonov smoothing over "
                    "phase-aligned beat, branch, and radius realizations."
                ),
                "joint_denoising_input_path": (
                    self.v_band_segment_input
                    if self.joint_use_bandlimited_input
                    else original_raw_segment_input
                ),
            }
        )
        result.attrs = attrs
        return result

    def _denoise_segment_block(self, v_block: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Denoise a ``(time, beat, branch, radius)`` arterial block using a joint
        pulse/temporal graph solve.
        """
        self._sync_laplacian_aliases()

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
            diagnostics = self._joint_diagnostics(
                corr_original,
                finite_fraction,
                status_code,
                phase_alignment_lag,
                phase_alignment_corr,
                best_neighbor_corr,
                neighbor_count,
                effective_neighbor_count,
                graph_degree,
                mean_abs_change,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                n_valid,
                n_time,
                np.nan,
            )
            self._last_joint_denoise_diag = diagnostics
            return out, diagnostics

        reference = np.nanmedian(filled[valid_flat_indices], axis=0)
        reference_norm = self._normalize_1d(reference)
        if reference_norm is None:
            reference = np.nanmean(filled[valid_flat_indices], axis=0)
            reference_norm = self._normalize_1d(reference)

        if reference_norm is None:
            self._mark_valid_without_graph_neighbors(
                out=out,
                v_block=v_block,
                filled=filled,
                finite_masks=finite_masks,
                coords=coords,
                valid_flat_indices=valid_flat_indices,
                status_code=status_code,
            )
            diagnostics = self._joint_diagnostics(
                corr_original,
                finite_fraction,
                status_code,
                phase_alignment_lag,
                phase_alignment_corr,
                best_neighbor_corr,
                neighbor_count,
                effective_neighbor_count,
                graph_degree,
                mean_abs_change,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                n_valid,
                n_time,
                np.nan,
            )
            self._last_joint_denoise_diag = diagnostics
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
            diagnostics = self._joint_diagnostics(
                corr_original,
                finite_fraction,
                status_code,
                phase_alignment_lag,
                phase_alignment_corr,
                best_neighbor_corr,
                neighbor_count,
                effective_neighbor_count,
                graph_degree,
                mean_abs_change,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                edge_count,
                n_valid,
                n_time,
                np.nan,
            )
            self._last_joint_denoise_diag = diagnostics
            return out, diagnostics

        degree = np.sum(W, axis=1)
        L_pulse = np.diag(degree) - W
        L_time = self._temporal_laplacian(n_time)

        graph_before = self._dirichlet_energy(W, aligned_shape)
        temporal_before = self._temporal_dirichlet_energy(L_time, aligned_shape)
        joint_before = (
            float(self.joint_pulse_gamma) * graph_before
            + float(self.joint_temporal_gamma) * temporal_before
        )

        solve_failed = False
        condition_number = np.nan
        try:
            denoised_aligned, condition_number = self._solve_joint_system(
                aligned=aligned_shape,
                L_pulse=L_pulse,
                L_time=L_time,
            )
        except np.linalg.LinAlgError:
            solve_failed = True
            denoised_aligned = aligned_shape.copy()

        alpha = float(np.clip(self.joint_blend_alpha, 0.0, 1.0))
        denoised_aligned = (1.0 - alpha) * aligned_shape + alpha * denoised_aligned

        graph_after = self._dirichlet_energy(W, denoised_aligned)
        temporal_after = self._temporal_dirichlet_energy(L_time, denoised_aligned)
        residual = denoised_aligned - aligned_shape
        joint_after = (
            float(np.sum(residual * residual))
            + float(self.joint_pulse_gamma) * graph_after
            + float(self.joint_temporal_gamma) * temporal_after
        )

        for row_idx, flat_i in enumerate(valid_flat_indices):
            beat_idx, branch_idx, radius_idx = coords[flat_i]
            original = v_block[:, beat_idx, branch_idx, radius_idx]
            finite_mask = finite_masks[flat_i]
            lag = int(alignment_lags_by_row[row_idx])

            denoised = np.roll(denoised_aligned[row_idx], -lag)
            denoised = self._restore_shape_signal(
                denoised, means[flat_i], stds[flat_i]
            )
            if self.joint_clip_output:
                denoised = self._clip_to_input_range(denoised, original)
            if self.joint_preserve_nan_mask:
                denoised[~finite_mask] = np.nan

            out[:, beat_idx, branch_idx, radius_idx] = denoised
            corr_original[coords[flat_i]] = self._pearson_corr(original, denoised)
            mean_abs_change[coords[flat_i]] = self._safe_mean_abs_change(
                original, denoised
            )
            status_code[coords[flat_i]] = 5 if solve_failed else 0

        diagnostics = self._joint_diagnostics(
            corr_original,
            finite_fraction,
            status_code,
            phase_alignment_lag,
            phase_alignment_corr,
            best_neighbor_corr,
            neighbor_count,
            effective_neighbor_count,
            graph_degree,
            mean_abs_change,
            graph_before,
            graph_after,
            temporal_before,
            temporal_after,
            joint_before,
            joint_after,
            edge_count,
            n_valid,
            n_time,
            condition_number,
        )
        self._last_joint_denoise_diag = diagnostics
        return out, diagnostics

    def _sync_laplacian_aliases(self) -> None:
        self.laplacian_top_k_neighbors = self.joint_top_k_neighbors
        self.laplacian_min_corr = self.joint_min_corr
        self.laplacian_corr_power = self.joint_corr_power
        self.laplacian_max_lag_fraction = self.joint_max_lag_fraction
        self.laplacian_lag_sigma_fraction = self.joint_lag_sigma_fraction
        self.laplacian_branch_sigma = self.joint_branch_sigma
        self.laplacian_radius_sigma = self.joint_radius_sigma
        self.laplacian_restore_pulse_scale = self.joint_restore_pulse_scale

    def _temporal_laplacian(self, n_time: int) -> np.ndarray:
        L = np.zeros((n_time, n_time), dtype=float)
        if n_time <= 1:
            return L

        for idx in range(n_time - 1):
            L[idx, idx] += 1.0
            L[idx + 1, idx + 1] += 1.0
            L[idx, idx + 1] -= 1.0
            L[idx + 1, idx] -= 1.0

        if self.joint_temporal_cycle and n_time > 2:
            L[0, 0] += 1.0
            L[-1, -1] += 1.0
            L[0, -1] -= 1.0
            L[-1, 0] -= 1.0

        return L

    def _solve_joint_system(
        self,
        aligned: np.ndarray,
        L_pulse: np.ndarray,
        L_time: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n_valid = int(aligned.shape[0])
        gamma_p = float(max(self.joint_pulse_gamma, 0.0))
        gamma_t = float(max(self.joint_temporal_gamma, 0.0))

        A_pulse = np.eye(n_valid, dtype=float) + gamma_p * L_pulse
        A_pulse += float(max(self.joint_self_jitter, 0.0)) * np.eye(
            n_valid, dtype=float
        )

        eig_p, vec_p = np.linalg.eigh(A_pulse)
        eig_t, vec_t = np.linalg.eigh(L_time)
        denom = eig_p[:, None] + gamma_t * eig_t[None, :]
        if np.any(np.abs(denom) <= self.eps):
            raise np.linalg.LinAlgError("Joint graph system is singular.")

        spectral = vec_p.T @ aligned @ vec_t
        solved = spectral / denom
        denoised = vec_p @ solved @ vec_t.T
        condition_number = float(np.max(np.abs(denom)) / np.min(np.abs(denom)))
        return denoised, condition_number

    @staticmethod
    def _temporal_dirichlet_energy(L_time: np.ndarray, X: np.ndarray) -> float:
        if L_time.size == 0 or X.size == 0:
            return np.nan
        return float(np.trace(X @ L_time @ X.T))

    @staticmethod
    def _joint_diagnostics(
        corr_original: np.ndarray,
        finite_fraction: np.ndarray,
        status_code: np.ndarray,
        phase_alignment_lag: np.ndarray,
        phase_alignment_corr: np.ndarray,
        best_neighbor_corr: np.ndarray,
        neighbor_count: np.ndarray,
        effective_neighbor_count: np.ndarray,
        graph_degree: np.ndarray,
        mean_abs_change: np.ndarray,
        graph_energy_before: float,
        graph_energy_after: float,
        temporal_energy_before: float,
        temporal_energy_after: float,
        joint_objective_before: float,
        joint_objective_after: float,
        edge_count: int,
        valid_pulse_count: int,
        temporal_node_count: int,
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
            "mean_abs_change": mean_abs_change,
            "graph_dirichlet_energy_before": float(graph_energy_before),
            "graph_dirichlet_energy_after": float(graph_energy_after),
            "temporal_dirichlet_energy_before": float(temporal_energy_before),
            "temporal_dirichlet_energy_after": float(temporal_energy_after),
            "joint_objective_before": float(joint_objective_before),
            "joint_objective_after": float(joint_objective_after),
            "graph_edge_count": int(edge_count),
            "valid_pulse_count": int(valid_pulse_count),
            "temporal_node_count": int(temporal_node_count),
            "system_condition_number": float(condition_number),
        }

    def _replace_joint_denoising_params(self, metrics: dict) -> None:
        for key in list(metrics):
            if key.startswith("artery/by_segment/denoising/params/"):
                del metrics[key]

        status_key = "artery/by_segment/denoising/status_code"
        status_value = metrics.get(status_key)
        if hasattr(status_value, "attrs") and status_value.attrs is not None:
            status_value.attrs["definition"] = (
                "0 denoised, 1 all_nan, 2 too_sparse, 3 low_variance, "
                "4 no_eligible_graph_neighbors, 5 joint_graph_solve_failed"
            )

    def _pack_joint_denoising_outputs(self, metrics: dict) -> None:
        diag = getattr(self, "_last_joint_denoise_diag", None)
        if not diag:
            return

        base = "artery/by_segment/denoising"
        attrs = {"axis_order": "beat, branch, radius"}
        per_segment_defs = {
            "phase_alignment_lag_samples": (
                "Circular lag, in samples, applied to align each pulse to the "
                "ensemble reference before joint graph construction."
            ),
            "phase_alignment_corr": (
                "Cross-correlation between each pulse and the ensemble reference "
                "after the selected alignment lag."
            ),
            "best_neighbor_corr": (
                "Highest zero-lag correlation among retained pulse-graph "
                "neighbors after ensemble phase alignment."
            ),
            "neighbor_count": (
                "Number of retained pulse-graph neighbors before symmetrizing "
                "the adjacency matrix."
            ),
            "effective_neighbor_count": (
                "Inverse-concentration effective count of directed retained "
                "neighbor weights."
            ),
            "graph_degree": (
                "Weighted pulse-graph degree after symmetrizing the adjacency "
                "matrix."
            ),
            "mean_abs_change": (
                "Mean absolute pointwise change between the original pulse and "
                "the joint-denoised pulse."
            ),
        }

        for name, definition in per_segment_defs.items():
            dtype = np.int32 if name == "neighbor_count" else float
            metrics[f"{base}/{name}"] = with_attrs(
                np.asarray(diag[name], dtype=dtype),
                {**attrs, "definition": definition},
            )

        scalar_defs = {
            "graph_dirichlet_energy_before": (
                "Pulse-graph Dirichlet energy Tr(Y^T L_p Y) before joint smoothing."
            ),
            "graph_dirichlet_energy_after": (
                "Pulse-graph Dirichlet energy Tr(X^T L_p X) after joint smoothing."
            ),
            "temporal_dirichlet_energy_before": (
                "Temporal graph energy Tr(Y L_t Y^T) before joint smoothing."
            ),
            "temporal_dirichlet_energy_after": (
                "Temporal graph energy Tr(X L_t X^T) after joint smoothing."
            ),
            "joint_objective_before": (
                "Joint objective value at the aligned input Y, excluding zero "
                "fidelity residual."
            ),
            "joint_objective_after": (
                "Joint objective value after smoothing, including fidelity residual."
            ),
            "graph_edge_count": (
                "Number of undirected positive-weight edges in the pulse graph."
            ),
            "valid_pulse_count": (
                "Number of valid pulse realizations included in the joint solve."
            ),
            "temporal_node_count": (
                "Number of within-beat time samples in the temporal graph."
            ),
            "system_condition_number": (
                "Spectral condition number of the joint product-graph solve."
            ),
        }
        for name, definition in scalar_defs.items():
            dtype = np.int32 if name.endswith("_count") else float
            metrics[f"{base}/{name}"] = with_attrs(
                np.asarray(diag[name], dtype=dtype), {"definition": definition}
            )

        params = {
            "method": "joint_pulse_temporal_graph_tikhonov",
            "use_bandlimited_input": int(bool(self.joint_use_bandlimited_input)),
            "pulse_gamma": float(self.joint_pulse_gamma),
            "temporal_gamma": float(self.joint_temporal_gamma),
            "top_k_neighbors": int(self.joint_top_k_neighbors),
            "min_corr": float(self.joint_min_corr),
            "corr_power": float(self.joint_corr_power),
            "self_jitter": float(self.joint_self_jitter),
            "max_lag_fraction": float(self.joint_max_lag_fraction),
            "lag_sigma_fraction": float(self.joint_lag_sigma_fraction),
            "blend_alpha": float(self.joint_blend_alpha),
            "temporal_cycle": int(bool(self.joint_temporal_cycle)),
            "preserve_nan_mask": int(bool(self.joint_preserve_nan_mask)),
            "restore_pulse_scale": int(bool(self.joint_restore_pulse_scale)),
            "clip_output": int(bool(self.joint_clip_output)),
            "metrics_use_filtered_signal": int(
                bool(self.arterial_metrics_use_filtered_signal)
            ),
            "bandlimited_segment_metrics_source": (
                "filtered_for_metrics"
                if bool(self.arterial_metrics_use_filtered_signal)
                else "bandlimited"
            ),
            "branch_sigma": float(self.joint_branch_sigma),
            "radius_sigma": float(self.joint_radius_sigma),
        }
        for name, value in params.items():
            metrics[f"{base}/params/{name}"] = np.asarray(value)
