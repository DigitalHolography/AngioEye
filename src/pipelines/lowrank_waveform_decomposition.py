import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformDecomposition(ProcessPipeline):
    """
    Low-rank SVD decomposition for beat-aligned arterial segment waveforms.

    For each acquisition and each configured waveform source, the pipeline removes
    the local temporal mean, performs one joint SVD over all valid beat-location
    waveforms, and reports the four primary endpoints: A1, rho1, A2, and rho2.

    Native estimator (the meaning of the existing ``<source>/endpoints/...`` paths,
    unchanged by this module's variant/QC additions): for a per-(beat,branch,radius)
    array ``z``, the acquisition-level value is the *nested median*,
    ``median_b(median_kr(z))`` -- see ``_nested_median``. This is also exported
    explicitly at ``<source>/variants/joint_svd/nested_median/endpoints/...``, of
    which ``<source>/endpoints/...`` is a documented alias (same code path, same
    numbers). Two additional acquisition-level estimators are exported as
    robustness/sensitivity variants, never as replacements for the native one:
    ``nested_mean`` (``mean_b(median_kr(z))``, the article's mean-over-beats
    robustness check) and ``pooled_median`` (a flat ``median`` over every valid
    ``(b,k,r)`` entry, a diagnostic estimator only). ``rho_m`` is always computed
    as a ratio of aggregates (``aggregate(R_m) / aggregate(TPR)``), never as an
    aggregate of per-beat ratios ``rho_m(b)`` -- the per-beat ratios remain
    available separately as the ``beatwise/rho{m}_b`` diagnostic.

    Beat-period residualization (regressing an endpoint on beat period to control
    for a heart-rate confound) is NOT performed by this pipeline: it requires a
    regression across multiple acquisitions of one dataset/epoch, which a
    single-acquisition pipeline cannot do correctly. That step is a downstream,
    cross-acquisition analysis -- see
    ``residualize_against_beat_period`` in
    ``/Users/admin/Desktop/langevin-internship/flicker-detection/FULL_PIPELINE.py``
    (outside this repository).

    Raw vs. bandlimited: this pipeline computes both representations
    symmetrically and does not itself designate either as primary -- see
    ``attrs["primary_representation"]`` on the returned ``ProcessResult``.
    """

    description = (
        "Joint low-rank waveform decomposition from beat-aligned arterial segment "
        "waveforms, reporting A1, rho1, A2, rho2, and TPR for raw and bandlimited "
        "signals, plus nested-mean and pooled-median aggregation variants."
    )

    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    v_band_segment_input = (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )

    eps = 1e-12
    min_valid_samples_fraction = 0.95
    min_valid_columns = 3
    max_modes_panel = 3
    exported_modes = 2
    strict_complete_case = False

    @staticmethod
    def _safe_nanmean(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmean(x))

    @staticmethod
    def _safe_nanmedian(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanmedian(x))

    @staticmethod
    def _safe_nanstd(x: np.ndarray) -> float:
        """Population standard deviation (ddof=0), matching this repository's
        convention for every other pipeline's beatwise/within-acquisition
        variability metric (Windkessel_RC.py, absolute_waveform_metrics.py,
        waveform_harmonic_organization*.py, waveform_shape_metrics*.py all use
        ddof=0). This is intentionally NOT the sample SD (ddof=1) used by the
        downstream manuscript scripts for cross-acquisition, epoch-level SD in
        the reported Table I/II "median +/- SD" columns -- that is a different
        quantity (dispersion across acquisitions, not across beats within one
        acquisition) computed outside this pipeline.
        """
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        return float(np.nanstd(x))

    @staticmethod
    def _safe_nanmad(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return np.nan
        med = np.nanmedian(x)
        return float(np.nanmedian(np.abs(x - med)))

    def _safe_nancv(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        mu = self._safe_nanmean(x)
        sd = self._safe_nanstd(x)
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or abs(mu) <= self.eps:
            return np.nan
        return float(sd / abs(mu))

    def _ensure_segment_shape(
        self, v_block: np.ndarray, T: np.ndarray | None = None
    ) -> tuple[np.ndarray, bool]:
        v_block = np.asarray(v_block, dtype=float)
        if v_block.ndim != 4:
            raise ValueError(
                "Expected segment waveform block with shape "
                f"(n_t, n_beats, n_branches, n_radii), got {v_block.shape}"
            )
        if T is None:
            return v_block, False

        n_beats = int(self._normalize_T(T).shape[1])
        ambiguous = v_block.shape[0] == n_beats and v_block.shape[1] == n_beats
        if v_block.shape[1] == n_beats:
            return v_block, ambiguous
        if v_block.shape[0] == n_beats and v_block.shape[1] != n_beats:
            return np.transpose(v_block, (1, 0, 2, 3)), False
        raise ValueError(
            "Expected segment waveform block with one axis matching the beat-period "
            f"count ({n_beats}) in shape (n_t,n_beats,n_branches,n_radii) or "
            f"(n_beats,n_t,n_branches,n_radii), got {v_block.shape}"
        )

    def _normalize_T(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.ndim == 1:
            return T.reshape(1, -1)
        if T.ndim == 2 and T.shape[0] == 1:
            return T
        if T.ndim == 2 and T.shape[1] == 1:
            return T.T
        raise ValueError(
            "Beat period input must be shape (n_beats,), (1, n_beats), or "
            f"(n_beats, 1); got {T.shape}"
        )

    @staticmethod
    def _mode_label(m: int) -> str:
        return f"mode{m}"

    def _median_kr_per_beat(
        self, arr_bkr: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        n_beats = int(arr_bkr.shape[0])
        out = np.full((n_beats,), np.nan, dtype=float)
        for b in range(n_beats):
            vals = np.asarray(arr_bkr[b], dtype=float)
            mask = np.asarray(valid_mask[b], dtype=bool)
            if not np.any(mask):
                continue
            x = vals[mask]
            if x.size == 0 or not np.any(np.isfinite(x)):
                continue
            out[b] = float(np.nanmedian(x))
        return out

    def _spatial_mad_per_beat(
        self, arr_bkr: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        n_beats = int(arr_bkr.shape[0])
        out = np.full((n_beats,), np.nan, dtype=float)
        for b in range(n_beats):
            vals = np.asarray(arr_bkr[b], dtype=float)
            mask = np.asarray(valid_mask[b], dtype=bool)
            if not np.any(mask):
                continue
            x = vals[mask]
            if x.size == 0 or not np.any(np.isfinite(x)):
                continue
            med = np.nanmedian(x)
            out[b] = float(np.nanmedian(np.abs(x - med)))
        return out

    def _nested_median(self, arr_bkr: np.ndarray, valid_mask: np.ndarray) -> float:
        """Native acquisition-level estimator: median_b(median_kr(z))."""
        return self._safe_nanmedian(self._median_kr_per_beat(arr_bkr, valid_mask))

    def _nested_mean(self, arr_bkr: np.ndarray, valid_mask: np.ndarray) -> float:
        """Mean-over-beats robustness variant: mean_b(median_kr(z))."""
        return self._safe_nanmean(self._median_kr_per_beat(arr_bkr, valid_mask))

    def _pooled_median(self, arr_bkr: np.ndarray, valid_mask: np.ndarray) -> float:
        """Diagnostic-only estimator: a flat median over every valid (b,k,r)
        entry, with no per-beat grouping step at all. Exported strictly as a
        sensitivity check (`variants/joint_svd/pooled_median/...`), never as
        the native acquisition endpoint."""
        arr = np.asarray(arr_bkr, dtype=float)
        mask = np.asarray(valid_mask, dtype=bool)
        return self._safe_nanmedian(arr[mask])

    def _aggregate(
        self, arr_bkr: np.ndarray, valid_mask: np.ndarray, estimator: str
    ) -> float:
        if estimator == "nested_median":
            return self._nested_median(arr_bkr, valid_mask)
        if estimator == "nested_mean":
            return self._nested_mean(arr_bkr, valid_mask)
        if estimator == "pooled_median":
            return self._pooled_median(arr_bkr, valid_mask)
        raise ValueError(f"Unknown aggregation estimator: {estimator!r}")

    def _aggregate_ratio(
        self,
        numerator_bkr: np.ndarray,
        denominator_bkr: np.ndarray,
        valid_mask: np.ndarray,
        estimator: str,
    ) -> float:
        """Ratio of aggregates: aggregate(numerator) / aggregate(denominator).
        Never an aggregate of pointwise ratios -- see module docstring and
        Sec. requirement 1."""
        numerator = self._aggregate(numerator_bkr, valid_mask, estimator)
        denominator = self._aggregate(denominator_bkr, valid_mask, estimator)
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return np.nan
        if denominator <= 0.0:
            return np.nan
        return float(numerator / denominator)

    @staticmethod
    def _effective_rank(energy_fraction: np.ndarray) -> float:
        p = np.asarray(energy_fraction, dtype=float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size == 0:
            return np.nan
        p = p / np.sum(p)
        return float(np.exp(-np.sum(p * np.log(p))))

    @staticmethod
    def _participation_ratio(energy_fraction: np.ndarray) -> float:
        p = np.asarray(energy_fraction, dtype=float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size == 0:
            return np.nan
        p = p / np.sum(p)
        denom = float(np.sum(p**2))
        if denom <= 0:
            return np.nan
        return float(1.0 / denom)

    def _deterministic_sign(self, scores: np.ndarray, u: np.ndarray) -> int:
        scores = np.asarray(scores, dtype=float)
        med_score = self._safe_nanmedian(scores)
        if np.isfinite(med_score) and med_score != 0:
            return -1 if med_score < 0 else 1

        if scores.size:
            idx = int(np.nanargmax(np.abs(scores)))
            largest_score = scores[idx]
            if np.isfinite(largest_score) and largest_score != 0:
                return -1 if largest_score < 0 else 1

        u = np.asarray(u, dtype=float)
        if u.size:
            idx = int(np.nanargmax(np.abs(u)))
            largest_u = u[idx]
            if np.isfinite(largest_u) and largest_u != 0:
                return -1 if largest_u < 0 else 1

        return 1

    def _mode_component_rms(
        self, u: np.ndarray, scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        """RMS_t(a_m u_m) at each valid (b,k,r). For unit-Euclidean-norm u_m
        (guaranteed by np.linalg.svd's convention), rms_u == 1/sqrt(n_t)
        exactly, so this equals |a_m|/sqrt(n_t) -- the same sampled-signal RMS
        scale as TPR (see test_mode_amplitude_matches_tpr_rms_scale)."""
        rms_u = float(np.sqrt(np.mean(np.asarray(u, dtype=float) ** 2)))
        comp = np.full(valid_mask.shape, np.nan, dtype=float)
        comp[valid_mask] = np.abs(np.asarray(scores, dtype=float)) * rms_u
        return comp

    @staticmethod
    def _reconstruct_mode_sum(U_r: np.ndarray, scores_r: np.ndarray) -> np.ndarray:
        if U_r.size == 0 or scores_r.size == 0:
            return np.zeros((U_r.shape[0], scores_r.shape[1]), dtype=float)
        return U_r @ scores_r

    def _residual_t_bkr(
        self,
        x_full: np.ndarray,
        valid_column_mask: np.ndarray,
        residual_valid: np.ndarray,
    ) -> np.ndarray:
        residual = np.full_like(x_full, np.nan, dtype=float)
        residual[:, valid_column_mask] = residual_valid
        return residual

    def _read_beat_period(self, h5file) -> np.ndarray:
        if self.T_input not in h5file:
            raise ValueError(f"Missing required beat-period input: {self.T_input}")
        return self._normalize_T(np.asarray(h5file[self.T_input]))

    def _compute_representation(
        self, v_block: np.ndarray, T: np.ndarray, *, min_valid_samples_fraction=None
    ) -> dict:
        """Compute the joint (t,bkr) SVD representation for one acquisition.

        min_valid_samples_fraction: override for the column-validity threshold.
        Defaults to self.min_valid_samples_fraction; pass 1.0 for the strict
        complete-case sensitivity mode (Sec. requirement 3) without mutating
        the instance.
        """
        threshold = (
            self.min_valid_samples_fraction
            if min_valid_samples_fraction is None
            else float(min_valid_samples_fraction)
        )

        T = self._normalize_T(T)
        v_block, axis_order_ambiguous = self._ensure_segment_shape(v_block, T)

        n_t, n_beats, n_branches, n_radii = v_block.shape
        if T.shape[1] != n_beats:
            raise ValueError(
                "Beat-period length mismatch: "
                f"T has {T.shape[1]} beats, waveform block has {n_beats} beats."
            )

        finite = np.isfinite(v_block)
        finite_fraction = np.mean(finite, axis=0)
        sufficiently_finite_mask = finite_fraction >= threshold
        beat_period_valid = np.isfinite(T[0]) & (T[0] > 0)
        valid_column_mask = sufficiently_finite_mask & beat_period_valid[:, None, None]

        fully_finite_mask = finite_fraction >= 1.0
        n_fully_finite_columns = int(np.sum(fully_finite_mask & valid_column_mask))
        partially_imputed_mask = valid_column_mask & ~fully_finite_mask
        n_partially_imputed_columns = int(np.sum(partially_imputed_mask))
        n_excluded_insufficient_finite_columns = int(np.sum(~sufficiently_finite_mask))
        n_excluded_beat_period_only_columns = int(
            np.sum(sufficiently_finite_mask & ~beat_period_valid[:, None, None])
        )
        n_imputed_samples_total = int(
            np.sum((~finite) & valid_column_mask[None, :, :, :])
        )

        n_total_columns = int(n_beats * n_branches * n_radii)
        n_valid_columns = int(np.sum(valid_column_mask))

        out = {
            "shape": {
                "n_t": n_t,
                "n_beats": n_beats,
                "n_branches": n_branches,
                "n_radii": n_radii,
                "n_total_columns": n_total_columns,
                "n_valid_columns": n_valid_columns,
            },
            "valid_column_mask": valid_column_mask,
            "finite_fraction_per_column": finite_fraction,
            "beat_period_valid": beat_period_valid,
            "axis_order_ambiguous": axis_order_ambiguous,
            "missing_data_qc": {
                "n_fully_finite_columns": n_fully_finite_columns,
                "fraction_fully_finite_columns": (
                    n_fully_finite_columns / n_total_columns
                    if n_total_columns
                    else np.nan
                ),
                "n_partially_imputed_columns": n_partially_imputed_columns,
                "fraction_partially_imputed_columns": (
                    n_partially_imputed_columns / n_total_columns
                    if n_total_columns
                    else np.nan
                ),
                "n_imputed_samples_total": n_imputed_samples_total,
                "n_excluded_insufficient_finite_columns": (
                    n_excluded_insufficient_finite_columns
                ),
                "n_excluded_beat_period_only_columns": (
                    n_excluded_beat_period_only_columns
                ),
                "imputation_strategy": "column-local temporal mean",
                "waveform_validity_threshold": threshold,
            },
        }

        has_any_finite = np.any(finite, axis=0)
        mu = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        mu[has_any_finite] = np.nanmean(v_block[:, has_any_finite], axis=0)
        out["mu"] = mu

        v_filled = np.where(finite, v_block, mu[None, :, :, :])
        x_full = v_filled - mu[None, :, :, :]
        x_full = np.where(np.isfinite(x_full), x_full, 0.0)
        out["x_full"] = x_full

        rms_x = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        rms_x[valid_column_mask] = np.sqrt(
            np.mean(x_full[:, valid_column_mask] ** 2, axis=0)
        )
        out["rms_x"] = rms_x
        out["total_rms_bkr"] = rms_x

        valid_counts_per_beat = np.sum(valid_column_mask, axis=(1, 2))
        valid_fraction_per_beat = valid_counts_per_beat / float(
            max(1, n_branches * n_radii)
        )
        out["valid_counts_per_beat"] = valid_counts_per_beat
        out["valid_fraction_per_beat"] = valid_fraction_per_beat

        tpr_b = self._median_kr_per_beat(rms_x, valid_column_mask)
        tpr = self._safe_nanmedian(tpr_b)
        beatwise = {
            "mu_b": self._median_kr_per_beat(mu, valid_column_mask),
            "TPR_b": tpr_b,
        }
        acq = {
            "mu_acq": self._safe_nanmedian(beatwise["mu_b"]),
            "beat_period_mean": self._safe_nanmean(T[0][beat_period_valid]),
            "beat_period_median": self._safe_nanmedian(T[0][beat_period_valid]),
            "beat_period_std": self._safe_nanstd(T[0][beat_period_valid]),
            "sigma_mu_beat": self._safe_nanstd(beatwise["mu_b"]),
            "mad_mu_beat": self._safe_nanmad(beatwise["mu_b"]),
            "TPR": tpr,
            "sigma_TPR_beat": self._safe_nanstd(tpr_b),
            "mad_TPR_beat": self._safe_nanmad(tpr_b),
        }
        tpr_variants = {
            estimator: self._aggregate(rms_x, valid_column_mask, estimator)
            for estimator in ("nested_median", "nested_mean", "pooled_median")
        }
        acq["TPR_variants"] = tpr_variants

        if n_valid_columns < int(self.min_valid_columns):
            out["beatwise"] = beatwise
            out["acq"] = acq
            out["svd_available"] = False
            out["svd_reason"] = "too_few_valid_columns"
            return out

        X = x_full[:, valid_column_mask]
        if X.size == 0:
            out["beatwise"] = beatwise
            out["acq"] = acq
            out["svd_available"] = False
            out["svd_reason"] = "empty_valid_matrix"
            return out

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        energy = s**2
        total_energy = float(np.sum(energy))
        if total_energy > 0.0:
            energy_fraction = energy / total_energy
        else:
            energy_fraction = np.full_like(energy, np.nan)

        out["svd_available"] = True
        out["svd_reason"] = "ok"
        out["X"] = X
        out["U"] = U
        out["s"] = s
        out["Vt"] = Vt
        out["energy"] = energy
        out["energy_fraction"] = energy_fraction

        n_modes = int(min(self.max_modes_panel, len(s)))
        out["n_modes_panel"] = n_modes

        score_list = []
        sign_flips = np.zeros((n_modes,), dtype=int)
        u_panel = np.full((n_t, self.max_modes_panel), np.nan, dtype=float)
        score_panel_flat = np.full(
            (self.max_modes_panel, n_valid_columns), np.nan, dtype=float
        )

        for m in range(n_modes):
            scores = s[m] * Vt[m, :]
            sign = self._deterministic_sign(scores, U[:, m])
            if sign < 0:
                U[:, m] *= -1.0
                Vt[m, :] *= -1.0
                scores = scores * -1.0
                sign_flips[m] = 1

            u_panel[:, m] = U[:, m]
            score_panel_flat[m, :] = scores
            score_list.append(scores)

        out["U_panel"] = u_panel
        out["score_panel_flat"] = score_panel_flat
        out["sign_flips"] = sign_flips

        score_panel_bkr = np.full(
            (self.max_modes_panel, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        for m in range(n_modes):
            score_panel_bkr[m, valid_column_mask] = score_list[m]
        out["score_panel_bkr"] = score_panel_bkr

        rms_mode_panel = np.full_like(score_panel_bkr, np.nan, dtype=float)
        residual_rms_panel = np.full_like(score_panel_bkr, np.nan, dtype=float)
        residual_t_bkr_panel = np.full(
            (self.exported_modes, n_t, n_beats, n_branches, n_radii),
            np.nan,
            dtype=float,
        )
        rho_panel = np.full((self.max_modes_panel,), np.nan, dtype=float)

        for m in range(1, n_modes + 1):
            u_m = U[:, m - 1]
            scores_m = score_list[m - 1]
            rms_mode_panel[m - 1] = self._mode_component_rms(
                u=u_m,
                scores=scores_m,
                valid_mask=valid_column_mask,
            )

            X_recon_m = self._reconstruct_mode_sum(U[:, :m], np.vstack(score_list[:m]))
            X_res_m = X - X_recon_m
            if m <= self.exported_modes:
                residual_t_bkr_panel[m - 1] = self._residual_t_bkr(
                    x_full=x_full,
                    valid_column_mask=valid_column_mask,
                    residual_valid=X_res_m,
                )

            residual_rms_bkr = np.full(
                (n_beats, n_branches, n_radii), np.nan, dtype=float
            )
            residual_rms_bkr[valid_column_mask] = np.sqrt(np.mean(X_res_m**2, axis=0))
            residual_rms_panel[m - 1] = residual_rms_bkr

            r_b = self._median_kr_per_beat(residual_rms_bkr, valid_column_mask)
            a_b = self._median_kr_per_beat(rms_mode_panel[m - 1], valid_column_mask)
            rho_b = np.where(
                np.isfinite(r_b) & np.isfinite(tpr_b) & (tpr_b > 0.0),
                r_b / tpr_b,
                np.nan,
            )
            R_m = self._safe_nanmedian(r_b)

            beatwise[f"A{m}_b"] = a_b
            beatwise[f"R{m}_b"] = r_b
            beatwise[f"rho{m}_b"] = rho_b
            beatwise[f"median_abs_a{m}_b"] = self._median_kr_per_beat(
                np.abs(score_panel_bkr[m - 1]), valid_column_mask
            )

            acq[f"A{m}"] = self._safe_nanmedian(a_b)
            acq[f"R{m}"] = R_m
            acq[f"rho{m}"] = (
                float(R_m / tpr)
                if np.isfinite(R_m) and np.isfinite(tpr) and tpr > 0.0
                else np.nan
            )
            rho_panel[m - 1] = acq[f"rho{m}"]
            acq[f"sigma_A{m}_beat"] = self._safe_nanstd(a_b)
            acq[f"mad_A{m}_beat"] = self._safe_nanmad(a_b)
            acq[f"cv_A{m}_beat"] = self._safe_nancv(a_b)
            acq[f"sigma_R{m}_beat"] = self._safe_nanstd(r_b)
            acq[f"mad_R{m}_beat"] = self._safe_nanmad(r_b)
            acq[f"cv_R{m}_beat"] = self._safe_nancv(r_b)
            acq[f"sigma_rho{m}_beat"] = self._safe_nanstd(rho_b)
            acq[f"mad_rho{m}_beat"] = self._safe_nanmad(rho_b)
            acq[f"cv_rho{m}_beat"] = self._safe_nancv(rho_b)
            acq[f"median_abs_a{m}"] = self._safe_nanmedian(
                beatwise[f"median_abs_a{m}_b"]
            )
            acq[f"spatial_mad_A{m}_median_over_beats"] = self._safe_nanmedian(
                self._spatial_mad_per_beat(rms_mode_panel[m - 1], valid_column_mask)
            )
            acq[f"spatial_mad_R{m}_median_over_beats"] = self._safe_nanmedian(
                self._spatial_mad_per_beat(residual_rms_bkr, valid_column_mask)
            )

            if m <= self.exported_modes:
                acq[f"A{m}_variants"] = {
                    estimator: self._aggregate(
                        rms_mode_panel[m - 1], valid_column_mask, estimator
                    )
                    for estimator in ("nested_median", "nested_mean", "pooled_median")
                }
                acq[f"rho{m}_variants"] = {
                    estimator: self._aggregate_ratio(
                        residual_rms_bkr, rms_x, valid_column_mask, estimator
                    )
                    for estimator in ("nested_median", "nested_mean", "pooled_median")
                }

        acq["eta1"] = float(energy_fraction[0]) if len(energy_fraction) >= 1 else np.nan
        acq["eta2"] = float(energy_fraction[1]) if len(energy_fraction) >= 2 else np.nan
        acq["eta12"] = (
            float(np.sum(energy_fraction[:2])) if len(energy_fraction) >= 2 else np.nan
        )
        acq["effective_rank"] = self._effective_rank(energy_fraction)
        acq["participation_ratio"] = self._participation_ratio(energy_fraction)

        out["higher_mode_numbers"] = list(range(self.exported_modes + 1, n_modes + 1))

        # Q2(t): phase-resolved residual energy after removing modes 1-2, using
        # the pooled-median estimator (a flat median over valid (b,k,r) at each
        # timepoint) as specified by the article -- NOT an acquisition endpoint,
        # and NOT combined across acquisitions (no Delta Q2(t) baseline-vs-
        # flicker comparison here; that is a downstream, cross-acquisition
        # analysis -- see measurements/ARTICLE__Sienna/
        # explore_rho2_residual_waveforms.py in the flicker-detection repo).
        if self.exported_modes <= n_modes:
            r2_t_bkr = residual_t_bkr_panel[self.exported_modes - 1]
            r2_valid = r2_t_bkr[:, valid_column_mask]
            if r2_valid.size:
                out["Q2_t"] = np.nanmedian(r2_valid**2, axis=1)
            else:
                out["Q2_t"] = np.full((n_t,), np.nan, dtype=float)
        else:
            out["Q2_t"] = np.full((n_t,), np.nan, dtype=float)

        out["rms_mode_panel"] = rms_mode_panel
        out["residual_rms_panel"] = residual_rms_panel
        out["residual_t_bkr_panel"] = residual_t_bkr_panel
        out["rho_panel"] = rho_panel
        out["beatwise"] = beatwise
        out["acq"] = acq
        return out

    # ------------------------------------------------------------------
    # Metrics export
    # ------------------------------------------------------------------

    _VARIANT_ESTIMATORS = ("nested_median", "nested_mean", "pooled_median")

    def _append_nan_mode_metrics(
        self,
        metrics: dict,
        prefix: str,
        mode_number: int,
        rep: dict,
    ) -> None:
        sh = rep["shape"]
        n_t = int(sh["n_t"])
        n_beats = int(sh["n_beats"])
        n_branches = int(sh["n_branches"])
        n_radii = int(sh["n_radii"])
        mode_key = self._mode_label(mode_number)

        metrics[f"{prefix}/decomposition/u_{mode_key}"] = np.full(
            (n_t,), np.nan, dtype=float
        )
        metrics[f"{prefix}/decomposition/scores_{mode_key}_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/rms/{mode_key}_amplitude_rms_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/residuals/r{mode_number}_t_bkr"] = np.full(
            (n_t, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/residuals/rms_r{mode_number}_bkr"] = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/A{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/R{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/rho{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )
        metrics[f"{prefix}/beatwise/median_abs_a{mode_number}_b"] = np.full(
            (n_beats,), np.nan, dtype=float
        )

        for endpoint in ("A", "R", "rho", "median_abs_a"):
            metrics[f"{prefix}/endpoints/{endpoint}{mode_number}"] = np.asarray(
                np.nan, dtype=float
            )
        for stem in (
            "sigma_A",
            "mad_A",
            "cv_A",
            "sigma_R",
            "mad_R",
            "cv_R",
            "sigma_rho",
            "mad_rho",
            "cv_rho",
        ):
            metrics[f"{prefix}/variability/{stem}{mode_number}_beat"] = np.asarray(
                np.nan, dtype=float
            )
        metrics[
            f"{prefix}/variability/spatial_mad_A{mode_number}_median_over_beats"
        ] = np.asarray(np.nan, dtype=float)
        metrics[
            f"{prefix}/variability/spatial_mad_R{mode_number}_median_over_beats"
        ] = np.asarray(np.nan, dtype=float)

        for estimator in self._VARIANT_ESTIMATORS:
            for endpoint in ("A", "rho"):
                metrics[
                    f"{prefix}/variants/joint_svd/{estimator}/endpoints/"
                    f"{endpoint}{mode_number}"
                ] = np.asarray(np.nan, dtype=float)

    def _append_nan_higher_mode_metrics(
        self, metrics: dict, prefix: str, mode_number: int
    ) -> None:
        for stat in ("A", "R", "rho"):
            metrics[f"{prefix}/higher_modes/{stat}{mode_number}"] = np.asarray(
                np.nan, dtype=float
            )
        metrics[f"{prefix}/higher_modes/rho{mode_number}_b"] = np.full(
            (0,), np.nan, dtype=float
        )

    def _append_missing_data_qc(self, metrics: dict, prefix: str, rep: dict) -> None:
        qc = rep["missing_data_qc"]
        metrics[f"{prefix}/qc/n_fully_finite_columns"] = np.asarray(
            qc["n_fully_finite_columns"], dtype=int
        )
        metrics[f"{prefix}/qc/fraction_fully_finite_columns"] = np.asarray(
            qc["fraction_fully_finite_columns"], dtype=float
        )
        metrics[f"{prefix}/qc/n_partially_imputed_columns"] = np.asarray(
            qc["n_partially_imputed_columns"], dtype=int
        )
        metrics[f"{prefix}/qc/fraction_partially_imputed_columns"] = np.asarray(
            qc["fraction_partially_imputed_columns"], dtype=float
        )
        metrics[f"{prefix}/qc/n_imputed_samples_total"] = np.asarray(
            qc["n_imputed_samples_total"], dtype=int
        )
        metrics[f"{prefix}/qc/n_excluded_insufficient_finite_columns"] = np.asarray(
            qc["n_excluded_insufficient_finite_columns"], dtype=int
        )
        metrics[f"{prefix}/qc/n_excluded_beat_period_only_columns"] = np.asarray(
            qc["n_excluded_beat_period_only_columns"], dtype=int
        )
        metrics[f"{prefix}/qc/imputation_strategy"] = str(qc["imputation_strategy"])
        metrics[f"{prefix}/qc/waveform_validity_threshold"] = np.asarray(
            qc["waveform_validity_threshold"], dtype=float
        )
        metrics[f"{prefix}/qc/axis_order_ambiguous"] = np.asarray(
            int(bool(rep["axis_order_ambiguous"])), dtype=np.uint8
        )

    def _append_strict_complete_case_panel(
        self, metrics: dict, prefix: str, rep_strict: dict
    ) -> None:
        base = f"{prefix}/qc/strict_complete_case"
        acq = rep_strict["acq"]
        metrics[f"{base}/n_valid_columns"] = np.asarray(
            rep_strict["shape"]["n_valid_columns"], dtype=int
        )
        metrics[f"{base}/endpoints/TPR"] = np.asarray(
            acq.get("TPR", np.nan), dtype=float
        )
        for m in range(1, self.exported_modes + 1):
            metrics[f"{base}/endpoints/A{m}"] = np.asarray(
                acq.get(f"A{m}", np.nan), dtype=float
            )
            metrics[f"{base}/endpoints/rho{m}"] = np.asarray(
                acq.get(f"rho{m}", np.nan), dtype=float
            )

    def _append_representation_metrics(
        self,
        metrics: dict,
        source_name: str,
        dataset_path: str,
        rep: dict,
        rep_strict: dict,
    ) -> None:
        prefix = source_name
        sh = rep["shape"]
        acq = rep["acq"]

        metrics[f"{prefix}/config/signal_source"] = source_name
        metrics[f"{prefix}/config/input_dataset_path"] = dataset_path
        metrics[f"{prefix}/config/svd_method"] = "joint (t,bkr) SVD"
        metrics[f"{prefix}/config/aggregation"] = (
            "native estimator: nested_median = median over (k,r), then median over b"
        )
        metrics[f"{prefix}/config/max_exported_modes"] = np.asarray(
            self.exported_modes, dtype=int
        )
        metrics[f"{prefix}/config/max_modes_panel"] = np.asarray(
            self.max_modes_panel, dtype=int
        )
        metrics[f"{prefix}/config/min_valid_samples_fraction"] = np.asarray(
            self.min_valid_samples_fraction, dtype=float
        )
        metrics[f"{prefix}/config/min_valid_columns"] = np.asarray(
            self.min_valid_columns, dtype=int
        )

        metrics[f"{prefix}/inputs/n_t"] = np.asarray(sh["n_t"], dtype=int)
        metrics[f"{prefix}/inputs/n_beats"] = np.asarray(sh["n_beats"], dtype=int)
        metrics[f"{prefix}/inputs/n_branches"] = np.asarray(sh["n_branches"], dtype=int)
        metrics[f"{prefix}/inputs/n_radii"] = np.asarray(sh["n_radii"], dtype=int)
        metrics[f"{prefix}/inputs/n_total_columns"] = np.asarray(
            sh["n_total_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs/n_valid_columns"] = np.asarray(
            sh["n_valid_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs/valid_fraction_columns"] = np.asarray(
            sh["n_valid_columns"] / float(max(1, sh["n_total_columns"])), dtype=float
        )
        metrics[f"{prefix}/inputs/finite_fraction_per_column_bkr"] = rep[
            "finite_fraction_per_column"
        ]
        metrics[f"{prefix}/inputs/valid_column_mask_bkr"] = rep[
            "valid_column_mask"
        ].astype(np.uint8)
        metrics[f"{prefix}/inputs/valid_columns_per_beat"] = rep[
            "valid_counts_per_beat"
        ]
        metrics[f"{prefix}/inputs/valid_fraction_columns_per_beat"] = rep[
            "valid_fraction_per_beat"
        ]
        metrics[f"{prefix}/inputs/beat_period_valid_b"] = rep[
            "beat_period_valid"
        ].astype(np.uint8)

        metrics[f"{prefix}/baseline/mu_bkr"] = rep["mu"]
        metrics[f"{prefix}/baseline/mu_b"] = rep["beatwise"]["mu_b"]
        metrics[f"{prefix}/baseline/mu_acq"] = np.asarray(acq["mu_acq"], dtype=float)
        metrics[f"{prefix}/baseline/sigma_mu_beat"] = np.asarray(
            acq["sigma_mu_beat"], dtype=float
        )
        metrics[f"{prefix}/baseline/mad_mu_beat"] = np.asarray(
            acq["mad_mu_beat"], dtype=float
        )

        metrics[f"{prefix}/beat_period/mean"] = np.asarray(
            acq["beat_period_mean"], dtype=float
        )
        metrics[f"{prefix}/beat_period/median"] = np.asarray(
            acq["beat_period_median"], dtype=float
        )
        metrics[f"{prefix}/beat_period/std"] = np.asarray(
            acq["beat_period_std"], dtype=float
        )

        metrics[f"{prefix}/rms/total_pulsatile_rms_bkr"] = rep["total_rms_bkr"]
        metrics[f"{prefix}/beatwise/TPR_b"] = rep["beatwise"]["TPR_b"]
        metrics[f"{prefix}/endpoints/TPR"] = np.asarray(acq["TPR"], dtype=float)
        metrics[f"{prefix}/variability/sigma_TPR_beat"] = np.asarray(
            acq["sigma_TPR_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/mad_TPR_beat"] = np.asarray(
            acq["mad_TPR_beat"], dtype=float
        )
        for estimator in self._VARIANT_ESTIMATORS:
            metrics[f"{prefix}/variants/joint_svd/{estimator}/endpoints/TPR"] = (
                np.asarray(acq["TPR_variants"][estimator], dtype=float)
            )

        self._append_missing_data_qc(metrics, prefix, rep)
        self._append_strict_complete_case_panel(metrics, prefix, rep_strict)

        if not rep.get("svd_available", False):
            metrics[f"{prefix}/qc/svd_available"] = np.asarray(0, dtype=np.uint8)
            metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "unknown"))
            metrics[f"{prefix}/qc/n_modes"] = np.asarray(0, dtype=int)
            metrics[f"{prefix}/qc/sign_flips_mode1to2"] = np.zeros(
                (self.exported_modes,), dtype=int
            )
            metrics[f"{prefix}/residuals/Q2_t"] = np.full(
                (int(sh["n_t"]),), np.nan, dtype=float
            )
            for m in range(1, self.exported_modes + 1):
                metrics[f"{prefix}/qc/rho{m}_nonfinite"] = np.asarray(1, dtype=np.uint8)
                # Deprecated alias: kept for backward compatibility. The name
                # historically meant "rho is nonfinite", NOT "a numerical eps
                # floor was used in the denominator" -- use qc/rho{m}_nonfinite.
                metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = np.asarray(
                    1, dtype=np.uint8
                )
                self._append_nan_mode_metrics(metrics, prefix, m, rep)
            for m in range(self.exported_modes + 1, self.max_modes_panel + 1):
                self._append_nan_higher_mode_metrics(metrics, prefix, m)
            return

        metrics[f"{prefix}/qc/svd_available"] = np.asarray(1, dtype=np.uint8)
        metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "ok"))
        metrics[f"{prefix}/qc/n_modes"] = np.asarray(rep["n_modes_panel"], dtype=int)
        sign_flips = np.zeros((self.exported_modes,), dtype=int)
        available_sign_flips = rep["sign_flips"][: self.exported_modes]
        sign_flips[: available_sign_flips.size] = available_sign_flips
        metrics[f"{prefix}/qc/sign_flips_mode1to2"] = sign_flips
        for m in range(1, self.exported_modes + 1):
            nonfinite = np.asarray(
                int(not np.isfinite(acq.get(f"rho{m}", np.nan))), dtype=np.uint8
            )
            metrics[f"{prefix}/qc/rho{m}_nonfinite"] = nonfinite
            metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = nonfinite

        metrics[f"{prefix}/decomposition/singular_values"] = rep["s"]
        metrics[f"{prefix}/decomposition/singular_energy"] = rep["energy"]
        metrics[f"{prefix}/decomposition/singular_energy_fraction"] = rep[
            "energy_fraction"
        ]
        metrics[f"{prefix}/decomposition/effective_rank"] = np.asarray(
            acq["effective_rank"], dtype=float
        )
        metrics[f"{prefix}/decomposition/participation_ratio"] = np.asarray(
            acq["participation_ratio"], dtype=float
        )
        metrics[f"{prefix}/decomposition/eta1"] = np.asarray(acq["eta1"], dtype=float)
        metrics[f"{prefix}/decomposition/eta2"] = np.asarray(acq["eta2"], dtype=float)
        metrics[f"{prefix}/decomposition/eta12"] = np.asarray(acq["eta12"], dtype=float)

        metrics[f"{prefix}/residuals/Q2_t"] = rep["Q2_t"]

        for m in range(1, self.exported_modes + 1):
            idx = m - 1
            mode_key = self._mode_label(m)
            if rep["n_modes_panel"] < m:
                self._append_nan_mode_metrics(metrics, prefix, m, rep)
                continue

            metrics[f"{prefix}/decomposition/u_{mode_key}"] = rep["U_panel"][:, idx]
            metrics[f"{prefix}/decomposition/scores_{mode_key}_bkr"] = rep[
                "score_panel_bkr"
            ][idx]
            metrics[f"{prefix}/rms/{mode_key}_amplitude_rms_bkr"] = rep[
                "rms_mode_panel"
            ][idx]
            metrics[f"{prefix}/residuals/r{m}_t_bkr"] = rep["residual_t_bkr_panel"][idx]
            metrics[f"{prefix}/residuals/rms_r{m}_bkr"] = rep["residual_rms_panel"][idx]

            metrics[f"{prefix}/beatwise/A{m}_b"] = rep["beatwise"][f"A{m}_b"]
            metrics[f"{prefix}/beatwise/R{m}_b"] = rep["beatwise"][f"R{m}_b"]
            metrics[f"{prefix}/beatwise/rho{m}_b"] = rep["beatwise"][f"rho{m}_b"]
            metrics[f"{prefix}/beatwise/median_abs_a{m}_b"] = rep["beatwise"][
                f"median_abs_a{m}_b"
            ]

            metrics[f"{prefix}/endpoints/A{m}"] = np.asarray(acq[f"A{m}"], dtype=float)
            metrics[f"{prefix}/endpoints/R{m}"] = np.asarray(acq[f"R{m}"], dtype=float)
            metrics[f"{prefix}/endpoints/rho{m}"] = np.asarray(
                acq[f"rho{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/median_abs_a{m}"] = np.asarray(
                acq[f"median_abs_a{m}"], dtype=float
            )

            # <source>/endpoints/{A,rho}{m} above is the documented alias of the
            # native joint_svd/nested_median variant below (identical values,
            # same code path -- see class docstring).
            for estimator in self._VARIANT_ESTIMATORS:
                metrics[f"{prefix}/variants/joint_svd/{estimator}/endpoints/A{m}"] = (
                    np.asarray(acq[f"A{m}_variants"][estimator], dtype=float)
                )
                metrics[f"{prefix}/variants/joint_svd/{estimator}/endpoints/rho{m}"] = (
                    np.asarray(acq[f"rho{m}_variants"][estimator], dtype=float)
                )

            metrics[f"{prefix}/variability/sigma_A{m}_beat"] = np.asarray(
                acq[f"sigma_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_A{m}_beat"] = np.asarray(
                acq[f"mad_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_A{m}_beat"] = np.asarray(
                acq[f"cv_A{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/sigma_R{m}_beat"] = np.asarray(
                acq[f"sigma_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_R{m}_beat"] = np.asarray(
                acq[f"mad_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_R{m}_beat"] = np.asarray(
                acq[f"cv_R{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/sigma_rho{m}_beat"] = np.asarray(
                acq[f"sigma_rho{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/mad_rho{m}_beat"] = np.asarray(
                acq[f"mad_rho{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/cv_rho{m}_beat"] = np.asarray(
                acq[f"cv_rho{m}_beat"], dtype=float
            )
            metrics[f"{prefix}/variability/spatial_mad_A{m}_median_over_beats"] = (
                np.asarray(acq[f"spatial_mad_A{m}_median_over_beats"], dtype=float)
            )
            metrics[f"{prefix}/variability/spatial_mad_R{m}_median_over_beats"] = (
                np.asarray(acq[f"spatial_mad_R{m}_median_over_beats"], dtype=float)
            )

        # Higher-mode diagnostics (e.g. mode 3): never exported under
        # endpoints/, always under higher_modes/, so they cannot be mistaken
        # for primary endpoints.
        for m in range(self.exported_modes + 1, self.max_modes_panel + 1):
            if m > rep["n_modes_panel"]:
                self._append_nan_higher_mode_metrics(metrics, prefix, m)
                continue
            metrics[f"{prefix}/higher_modes/A{m}"] = np.asarray(
                acq.get(f"A{m}", np.nan), dtype=float
            )
            metrics[f"{prefix}/higher_modes/R{m}"] = np.asarray(
                acq.get(f"R{m}", np.nan), dtype=float
            )
            metrics[f"{prefix}/higher_modes/rho{m}"] = np.asarray(
                acq.get(f"rho{m}", np.nan), dtype=float
            )
            metrics[f"{prefix}/higher_modes/rho{m}_b"] = rep["beatwise"].get(
                f"rho{m}_b", np.full((int(sh["n_beats"]),), np.nan, dtype=float)
            )

    def run(self, h5file) -> ProcessResult:
        metrics = {}
        T = self._read_beat_period(h5file)

        source_map = {
            "raw": self.v_raw_segment_input,
            "bandlimited": self.v_band_segment_input,
        }

        for source_name, dataset_path in source_map.items():
            if dataset_path not in h5file:
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
                continue

            metrics[f"{source_name}/qc/input_available"] = np.asarray(1, dtype=np.uint8)
            v_block = np.asarray(h5file[dataset_path], dtype=float)
            rep = self._compute_representation(v_block=v_block, T=T)
            rep_strict = self._compute_representation(
                v_block=v_block, T=T, min_valid_samples_fraction=1.0
            )
            self._append_representation_metrics(
                metrics=metrics,
                source_name=source_name,
                dataset_path=dataset_path,
                rep=rep,
                rep_strict=rep_strict,
            )

        attrs = {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD",
            "aggregation": (
                "native estimator: nested_median = median over (k,r), then "
                "median over b"
            ),
            "aggregation_variants": list(self._VARIANT_ESTIMATORS),
            "mode_panel_max": int(self.exported_modes),
            "max_modes_panel": int(self.max_modes_panel),
            "representations": ["raw", "bandlimited"],
            "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
            "context_endpoint": "TPR",
            "higher_mode_diagnostics": (
                "R3/A3/rho3 (when acquisition rank permits) exported under "
                "'higher_modes/', not 'endpoints/' -- never primary endpoints."
            ),
            "input_beat_period_path": self.T_input,
            "input_raw_segment_path": self.v_raw_segment_input,
            "input_bandlimited_segment_path": self.v_band_segment_input,
            "beat_period_residualization": (
                "Not performed by this pipeline: it requires a regression "
                "across multiple acquisitions of one dataset/epoch, which a "
                "single-acquisition pipeline cannot do correctly. Downstream, "
                "cross-acquisition implementation: "
                "residualize_against_beat_period() in FULL_PIPELINE.py "
                "(/Users/admin/Desktop/langevin-internship/flicker-detection/, "
                "outside this repository)."
            ),
            "delta_q2_t": (
                "Not computed by this pipeline: Delta Q2(t) (flicker-minus-"
                "baseline) requires comparing multiple acquisitions across "
                "epochs. Downstream implementation: measurements/ARTICLE__"
                "Sienna/explore_rho2_residual_waveforms.py (flicker-detection "
                "repo, outside this repository), which computes the "
                "analogous energy_G2(t) baseline-vs-flicker comparison."
            ),
            "primary_representation": "unspecified_by_repository",
            "primary_representation_evidence": (
                "Neither this pipeline nor the manuscript (main_final.tex) "
                "ever declares raw or bandlimited as primary. In practice, "
                "every downstream manuscript-generation script "
                "(generate_main_final_artifacts.py, FULL_PIPELINE.py, and "
                "~60 other analysis scripts in the flicker-detection repo) "
                "reads exclusively the raw representation; bandlimited is "
                "exercised only by unit tests and a couple of generic "
                "exploratory grid scripts. This is de facto usage evidence, "
                "not a documented repository decision -- treat as an open "
                "item rather than an authoritative choice."
            ),
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
