import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformDecomposition(ProcessPipeline):
    """
    Low-rank SVD decomposition for beat-aligned arterial segment
    waveforms.

    For each acquisition and each configured waveform source, the pipeline removes
    the local temporal mean, performs one joint SVD over all valid beat-location
    waveforms, and reports the four primary endpoints A1, rho1, A2, and rho2.
    """

    description = (
        "Joint low-rank waveform decomposition from beat-aligned arterial segment "
        "waveforms, reporting A1, rho1, A2, rho2, and TPR for raw and bandlimited "
        "signals."
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
    max_modes_panel = 2
    exported_modes = 2

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
        return float(sd / (abs(mu) + self.eps))

    def _ensure_segment_shape(
        self, v_block: np.ndarray, T: np.ndarray | None = None
    ) -> np.ndarray:
        v_block = np.asarray(v_block, dtype=float)
        if v_block.ndim != 4:
            raise ValueError(
                "Expected segment waveform block with shape "
                f"(n_t, n_beats, n_branches, n_radii), got {v_block.shape}"
            )
        if T is None:
            return v_block

        n_beats = int(self._normalize_T(T).shape[1])
        if v_block.shape[1] == n_beats:
            return v_block
        if v_block.shape[0] == n_beats and v_block.shape[1] != n_beats:
            return np.transpose(v_block, (1, 0, 2, 3))
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

    def _mode_component_rms(
        self, u: np.ndarray, scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        rms_u = float(np.sqrt(np.mean(np.asarray(u, dtype=float) ** 2)))
        comp = np.full(valid_mask.shape, np.nan, dtype=float)
        comp[valid_mask] = np.abs(np.asarray(scores, dtype=float)) * rms_u
        return comp

    def _effective_rank(self, energy_fraction: np.ndarray) -> float:
        p = np.asarray(energy_fraction, dtype=float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size == 0:
            return np.nan
        return float(np.exp(-np.sum(p * np.log(p + self.eps))))

    def _participation_ratio(self, energy_fraction: np.ndarray) -> float:
        p = np.asarray(energy_fraction, dtype=float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size == 0:
            return np.nan
        denom = float(np.sum(p**2))
        if denom <= 0:
            return np.nan
        return float(1.0 / denom)

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

    def _median_kr_then_median_b(
        self, arr_bkr: np.ndarray, valid_mask: np.ndarray
    ) -> float:
        return self._safe_nanmedian(self._median_kr_per_beat(arr_bkr, valid_mask))

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

    def _compute_representation(self, v_block: np.ndarray, T: np.ndarray) -> dict:
        T = self._normalize_T(T)
        v_block = self._ensure_segment_shape(v_block, T)

        n_t, n_beats, n_branches, n_radii = v_block.shape
        if T.shape[1] != n_beats:
            raise ValueError(
                "Beat-period length mismatch: "
                f"T has {T.shape[1]} beats, waveform block has {n_beats} beats."
            )

        finite_fraction = np.mean(np.isfinite(v_block), axis=0)
        valid_column_mask = finite_fraction >= float(self.min_valid_samples_fraction)
        beat_period_valid = np.isfinite(T[0]) & (T[0] > 0)
        if np.any(~beat_period_valid):
            valid_column_mask &= beat_period_valid[:, None, None]

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
        }

        mu = np.nanmean(v_block, axis=0)
        out["mu"] = mu

        v_filled = np.where(np.isfinite(v_block), v_block, mu[None, :, :, :])
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
        energy_fraction = energy / (np.sum(energy) + self.eps)

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
            med_score = self._safe_nanmedian(scores)
            if np.isfinite(med_score) and med_score < 0:
                U[:, m] *= -1.0
                Vt[m, :] *= -1.0
                scores *= -1.0
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
            residual_rms_bkr[valid_column_mask] = np.sqrt(
                np.mean(X_res_m**2, axis=0)
            )
            residual_rms_panel[m - 1] = residual_rms_bkr

            r_b = self._median_kr_per_beat(residual_rms_bkr, valid_column_mask)
            a_b = self._median_kr_per_beat(rms_mode_panel[m - 1], valid_column_mask)
            rho_b = np.where(
                np.isfinite(r_b) & np.isfinite(tpr_b) & (tpr_b > self.eps),
                r_b / (tpr_b + self.eps),
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
                float(R_m / (tpr + self.eps))
                if np.isfinite(R_m) and np.isfinite(tpr) and tpr > self.eps
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

        acq["eta1"] = float(energy_fraction[0]) if len(energy_fraction) >= 1 else np.nan
        acq["eta2"] = float(energy_fraction[1]) if len(energy_fraction) >= 2 else np.nan
        acq["eta12"] = (
            float(np.sum(energy_fraction[:2])) if len(energy_fraction) >= 1 else np.nan
        )
        acq["effective_rank"] = self._effective_rank(energy_fraction)
        acq["participation_ratio"] = self._participation_ratio(energy_fraction)

        out["rms_mode_panel"] = rms_mode_panel
        out["residual_rms_panel"] = residual_rms_panel
        out["residual_t_bkr_panel"] = residual_t_bkr_panel
        out["rho_panel"] = rho_panel
        out["beatwise"] = beatwise
        out["acq"] = acq
        return out

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

    def _append_representation_metrics(
        self,
        metrics: dict,
        source_name: str,
        dataset_path: str,
        rep: dict,
    ) -> None:
        prefix = source_name
        sh = rep["shape"]
        acq = rep["acq"]

        metrics[f"{prefix}/config/signal_source"] = source_name
        metrics[f"{prefix}/config/input_dataset_path"] = dataset_path
        metrics[f"{prefix}/config/svd_method"] = "joint (t,bkr) SVD"
        metrics[f"{prefix}/config/aggregation"] = (
            "median over (k,r), then median over b"
        )
        metrics[f"{prefix}/config/max_exported_modes"] = np.asarray(
            self.exported_modes, dtype=int
        )
        metrics[f"{prefix}/config/min_valid_samples_fraction"] = np.asarray(
            self.min_valid_samples_fraction, dtype=float
        )
        metrics[f"{prefix}/config/min_valid_columns"] = np.asarray(
            self.min_valid_columns, dtype=int
        )

        metrics[f"{prefix}/inputs/n_t"] = np.asarray(sh["n_t"], dtype=int)
        metrics[f"{prefix}/inputs/n_beats"] = np.asarray(sh["n_beats"], dtype=int)
        metrics[f"{prefix}/inputs/n_branches"] = np.asarray(
            sh["n_branches"], dtype=int
        )
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
        metrics[f"{prefix}/baseline/mu_acq"] = np.asarray(
            acq["mu_acq"], dtype=float
        )
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

        if not rep.get("svd_available", False):
            metrics[f"{prefix}/qc/svd_available"] = np.asarray(0, dtype=np.uint8)
            metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "unknown"))
            metrics[f"{prefix}/qc/n_modes"] = np.asarray(0, dtype=int)
            metrics[f"{prefix}/qc/sign_flips_mode1to2"] = np.zeros(
                (self.exported_modes,), dtype=int
            )
            for m in range(1, self.exported_modes + 1):
                metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = np.asarray(
                    1, dtype=np.uint8
                )
                self._append_nan_mode_metrics(metrics, prefix, m, rep)
            return

        metrics[f"{prefix}/qc/svd_available"] = np.asarray(1, dtype=np.uint8)
        metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "ok"))
        metrics[f"{prefix}/qc/n_modes"] = np.asarray(rep["n_modes_panel"], dtype=int)
        sign_flips = np.zeros((self.exported_modes,), dtype=int)
        available_sign_flips = rep["sign_flips"][: self.exported_modes]
        sign_flips[: available_sign_flips.size] = available_sign_flips
        metrics[f"{prefix}/qc/sign_flips_mode1to2"] = sign_flips
        for m in range(1, self.exported_modes + 1):
            metrics[f"{prefix}/qc/denominator_floor_rho{m}"] = np.asarray(
                int(not np.isfinite(acq.get(f"rho{m}", np.nan))), dtype=np.uint8
            )

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
        metrics[f"{prefix}/decomposition/eta12"] = np.asarray(
            acq["eta12"], dtype=float
        )

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
            metrics[f"{prefix}/residuals/r{m}_t_bkr"] = rep["residual_t_bkr_panel"][
                idx
            ]
            metrics[f"{prefix}/residuals/rms_r{m}_bkr"] = rep["residual_rms_panel"][
                idx
            ]

            metrics[f"{prefix}/beatwise/A{m}_b"] = rep["beatwise"][f"A{m}_b"]
            metrics[f"{prefix}/beatwise/R{m}_b"] = rep["beatwise"][f"R{m}_b"]
            metrics[f"{prefix}/beatwise/rho{m}_b"] = rep["beatwise"][f"rho{m}_b"]
            metrics[f"{prefix}/beatwise/median_abs_a{m}_b"] = rep["beatwise"][
                f"median_abs_a{m}_b"
            ]

            metrics[f"{prefix}/endpoints/A{m}"] = np.asarray(
                acq[f"A{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/R{m}"] = np.asarray(
                acq[f"R{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/rho{m}"] = np.asarray(
                acq[f"rho{m}"], dtype=float
            )
            metrics[f"{prefix}/endpoints/median_abs_a{m}"] = np.asarray(
                acq[f"median_abs_a{m}"], dtype=float
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
            metrics[
                f"{prefix}/variability/spatial_mad_A{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_A{m}_median_over_beats"], dtype=float)
            metrics[
                f"{prefix}/variability/spatial_mad_R{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_R{m}_median_over_beats"], dtype=float)

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

            metrics[f"{source_name}/qc/input_available"] = np.asarray(
                1, dtype=np.uint8
            )
            v_block = np.asarray(h5file[dataset_path], dtype=float)
            rep = self._compute_representation(v_block=v_block, T=T)
            self._append_representation_metrics(
                metrics=metrics,
                source_name=source_name,
                dataset_path=dataset_path,
                rep=rep,
            )

        attrs = {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD",
            "aggregation": "median over (k,r), then median over b",
            "mode_panel_max": int(self.exported_modes),
            "representations": ["raw", "bandlimited"],
            "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
            "context_endpoint": "TPR",
            "input_beat_period_path": self.T_input,
            "input_raw_segment_path": self.v_raw_segment_input,
            "input_bandlimited_segment_path": self.v_band_segment_input,
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
