import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="lowrank_pulsatility_metrics")
class LowRankPulsatilityMetrics(ProcessPipeline):
    """
    Compute a low-rank decomposition of beat-aligned arterial segment waveforms within one
    acquisition and report robust acquisition-level summaries of baseline level, canonical
    pulsatile amplitude, residual pulsatile fraction, singular-spectrum concentration, and
    beat/spatial heterogeneity.

    The pipeline recomputes the SVD independently for raw and band-limited arterial segment
    waveforms and exposes a compact Mode-1 / Mode-2 / Mode-3 panel together with contextual
    and QC quantities.
    """

    description = (
        "Low-rank pulsatility metrics from beat-aligned arterial segment waveforms "
        "(raw + bandlimited), including a Mode-1/2/3 panel and robust within-acquisition "
        "summaries relevant to flicker-provocation studies."
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

    def _ensure_segment_shape(self, v_block: np.ndarray) -> np.ndarray:
        v_block = np.asarray(v_block, dtype=float)
        if v_block.ndim != 4:
            raise ValueError(
                "Expected segment waveform block with shape "
                f"(n_t, n_beats, n_branches, n_radii), got {v_block.shape}"
            )
        return v_block

    def _normalize_T(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.ndim == 1:
            return T.reshape(1, -1)
        if T.ndim == 2 and T.shape[0] == 1:
            return T
        if T.ndim == 2 and T.shape[1] == 1:
            return T.T
        raise ValueError(
            "Beat period input must be shape (n_beats,), (1, n_beats), or (n_beats, 1); "
            f"got {T.shape}"
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

    def _median_per_beat(self, arr_bkr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _reconstruct_mode_sum(U_r: np.ndarray, scores_r: np.ndarray) -> np.ndarray:
        if U_r.size == 0 or scores_r.size == 0:
            return np.zeros((U_r.shape[0], scores_r.shape[1]), dtype=float)
        return U_r @ scores_r

    def _compute_representation(self, v_block: np.ndarray, T: np.ndarray) -> dict:
        v_block = self._ensure_segment_shape(v_block)
        T = self._normalize_T(T)

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

        rms_x = np.sqrt(np.mean(x_full**2, axis=0))
        out["rms_x"] = rms_x

        valid_counts_per_beat = np.sum(valid_column_mask, axis=(1, 2))
        valid_fraction_per_beat = valid_counts_per_beat / float(max(1, n_branches * n_radii))
        out["valid_counts_per_beat"] = valid_counts_per_beat
        out["valid_fraction_per_beat"] = valid_fraction_per_beat

        if n_valid_columns < int(self.min_valid_columns):
            out["svd_available"] = False
            out["svd_reason"] = "too_few_valid_columns"
            return out

        X = x_full[:, valid_column_mask]
        if X.size == 0:
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
        score_panel_flat = np.full((self.max_modes_panel, n_valid_columns), np.nan, dtype=float)

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
        rho_panel = np.full((self.max_modes_panel,), np.nan, dtype=float)

        X_total_rms_valid = np.sqrt(np.mean(X**2, axis=0))
        total_rms_bkr = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        total_rms_bkr[valid_column_mask] = X_total_rms_valid
        out["total_rms_bkr"] = total_rms_bkr

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
            res_rms_valid = np.sqrt(np.mean(X_res_m**2, axis=0))
            residual_rms_bkr = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            residual_rms_bkr[valid_column_mask] = res_rms_valid
            residual_rms_panel[m - 1] = residual_rms_bkr

            num = self._safe_nanmedian(res_rms_valid)
            den = self._safe_nanmedian(X_total_rms_valid)
            rho_panel[m - 1] = (
                float(num / (den + self.eps))
                if np.isfinite(num) and np.isfinite(den) and den > self.eps
                else np.nan
            )

        out["rms_mode_panel"] = rms_mode_panel
        out["residual_rms_panel"] = residual_rms_panel
        out["rho_panel"] = rho_panel

        beatwise = {
            "median_kr_mu": self._median_per_beat(mu, valid_column_mask),
            "spatial_mad_mu": self._spatial_mad_per_beat(mu, valid_column_mask),
            "median_kr_total_rms": self._median_per_beat(total_rms_bkr, valid_column_mask),
        }

        for m in range(1, n_modes + 1):
            mode_rms = rms_mode_panel[m - 1]
            res_rms = residual_rms_panel[m - 1]
            beatwise[f"median_kr_A{m}"] = self._median_per_beat(mode_rms, valid_column_mask)
            beatwise[f"spatial_mad_A{m}"] = self._spatial_mad_per_beat(mode_rms, valid_column_mask)
            beatwise[f"median_kr_R{m}"] = self._median_per_beat(res_rms, valid_column_mask)
            beatwise[f"spatial_mad_R{m}"] = self._spatial_mad_per_beat(res_rms, valid_column_mask)
            beatwise[f"median_kr_abs_a{m}"] = self._median_per_beat(
                np.abs(score_panel_bkr[m - 1]), valid_column_mask
            )
            beatwise[f"spatial_mad_abs_a{m}"] = self._spatial_mad_per_beat(
                np.abs(score_panel_bkr[m - 1]), valid_column_mask
            )
            num_b = beatwise[f"median_kr_R{m}"]
            den_b = beatwise["median_kr_total_rms"]
            beatwise[f"rho{m}_beatwise"] = np.where(
                np.isfinite(num_b) & np.isfinite(den_b) & (den_b > self.eps),
                num_b / (den_b + self.eps),
                np.nan,
            )

        out["beatwise"] = beatwise

        acq = {
            "mu_acq": self._safe_nanmedian(mu[valid_column_mask]),
            "beat_period_mean": self._safe_nanmean(T[0][beat_period_valid]),
            "beat_period_median": self._safe_nanmedian(T[0][beat_period_valid]),
            "beat_period_std": self._safe_nanstd(T[0][beat_period_valid]),
            "sigma_mu_beat": self._safe_nanstd(beatwise["median_kr_mu"]),
            "mad_mu_beat": self._safe_nanmad(beatwise["median_kr_mu"]),
            "spatial_mad_mu_median_over_beats": self._safe_nanmedian(
                beatwise["spatial_mad_mu"]
            ),
        }

        for m in range(1, n_modes + 1):
            mode_rms = rms_mode_panel[m - 1]
            res_rms = residual_rms_panel[m - 1]
            acq[f"A{m}"] = self._safe_nanmedian(mode_rms[valid_column_mask])
            acq[f"sigma_A{m}_beat"] = self._safe_nanstd(beatwise[f"median_kr_A{m}"])
            acq[f"mad_A{m}_beat"] = self._safe_nanmad(beatwise[f"median_kr_A{m}"])
            acq[f"cv_A{m}_beat"] = self._safe_nancv(beatwise[f"median_kr_A{m}"])
            acq[f"spatial_mad_A{m}_median_over_beats"] = self._safe_nanmedian(
                beatwise[f"spatial_mad_A{m}"]
            )
            acq[f"R{m}"] = self._safe_nanmedian(res_rms[valid_column_mask])
            acq[f"rho{m}"] = rho_panel[m - 1]
            acq[f"sigma_R{m}_beat"] = self._safe_nanstd(beatwise[f"median_kr_R{m}"])
            acq[f"mad_R{m}_beat"] = self._safe_nanmad(beatwise[f"median_kr_R{m}"])
            acq[f"cv_R{m}_beat"] = self._safe_nancv(beatwise[f"median_kr_R{m}"])
            acq[f"sigma_rho{m}_beat"] = self._safe_nanstd(beatwise[f"rho{m}_beatwise"])
            acq[f"mad_rho{m}_beat"] = self._safe_nanmad(beatwise[f"rho{m}_beatwise"])
            acq[f"cv_rho{m}_beat"] = self._safe_nancv(beatwise[f"rho{m}_beatwise"])
            acq[f"spatial_mad_R{m}_median_over_beats"] = self._safe_nanmedian(
                beatwise[f"spatial_mad_R{m}"]
            )
            acq[f"median_abs_a{m}"] = self._safe_nanmedian(
                np.abs(score_panel_bkr[m - 1])[valid_column_mask]
            )
            acq[f"spatial_mad_abs_a{m}_median_over_beats"] = self._safe_nanmedian(
                beatwise[f"spatial_mad_abs_a{m}"]
            )

        acq["eta1"] = float(energy_fraction[0]) if len(energy_fraction) >= 1 else np.nan
        acq["eta2"] = float(energy_fraction[1]) if len(energy_fraction) >= 2 else np.nan
        acq["eta3"] = float(energy_fraction[2]) if len(energy_fraction) >= 3 else np.nan
        acq["eta23"] = (
            float(np.sum(energy_fraction[1:3])) if len(energy_fraction) >= 2 else np.nan
        )
        acq["effective_rank"] = self._effective_rank(energy_fraction)
        acq["participation_ratio"] = self._participation_ratio(energy_fraction)

        out["acq"] = acq
        return out

    def _append_representation_metrics(self, metrics: dict, rep_name: str, rep: dict) -> None:
        prefix = rep_name
        sh = rep["shape"]

        metrics[f"{prefix}/inputs_summary/n_t"] = np.asarray(sh["n_t"], dtype=int)
        metrics[f"{prefix}/inputs_summary/n_beats"] = np.asarray(sh["n_beats"], dtype=int)
        metrics[f"{prefix}/inputs_summary/n_branches"] = np.asarray(
            sh["n_branches"], dtype=int
        )
        metrics[f"{prefix}/inputs_summary/n_radii"] = np.asarray(sh["n_radii"], dtype=int)
        metrics[f"{prefix}/inputs_summary/n_total_columns"] = np.asarray(
            sh["n_total_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs_summary/n_valid_columns"] = np.asarray(
            sh["n_valid_columns"], dtype=int
        )
        metrics[f"{prefix}/inputs_summary/valid_fraction_columns"] = np.asarray(
            sh["n_valid_columns"] / float(max(1, sh["n_total_columns"])), dtype=float
        )
        metrics[f"{prefix}/inputs_summary/min_valid_samples_fraction"] = np.asarray(
            self.min_valid_samples_fraction, dtype=float
        )
        metrics[f"{prefix}/inputs_summary/finite_fraction_per_column"] = rep[
            "finite_fraction_per_column"
        ]
        metrics[f"{prefix}/inputs_summary/valid_column_mask"] = rep[
            "valid_column_mask"
        ].astype(np.uint8)
        metrics[f"{prefix}/inputs_summary/valid_columns_per_beat"] = rep[
            "valid_counts_per_beat"
        ]
        metrics[f"{prefix}/inputs_summary/valid_fraction_columns_per_beat"] = rep[
            "valid_fraction_per_beat"
        ]
        metrics[f"{prefix}/inputs_summary/beat_period_valid"] = rep[
            "beat_period_valid"
        ].astype(np.uint8)

        metrics[f"{prefix}/decomposition/mu"] = rep["mu"]
        metrics[f"{prefix}/decomposition/x_rms"] = rep["rms_x"]

        if not rep.get("svd_available", False):
            metrics[f"{prefix}/qc/svd_available"] = np.asarray(0, dtype=np.uint8)
            metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "unknown"))
            return

        metrics[f"{prefix}/qc/svd_available"] = np.asarray(1, dtype=np.uint8)
        metrics[f"{prefix}/qc/svd_reason"] = str(rep.get("svd_reason", "ok"))
        metrics[f"{prefix}/qc/sign_flips_mode1to3"] = rep["sign_flips"]
        metrics[f"{prefix}/qc/n_modes_panel"] = np.asarray(rep["n_modes_panel"], dtype=int)
        metrics[f"{prefix}/qc/denominator_floor_rho1"] = np.asarray(
            int(not np.isfinite(rep["acq"].get("rho1", np.nan))), dtype=np.uint8
        )
        metrics[f"{prefix}/qc/denominator_floor_rho2"] = np.asarray(
            int(not np.isfinite(rep["acq"].get("rho2", np.nan))), dtype=np.uint8
        )
        metrics[f"{prefix}/qc/denominator_floor_rho3"] = np.asarray(
            int(not np.isfinite(rep["acq"].get("rho3", np.nan))), dtype=np.uint8
        )

        metrics[f"{prefix}/decomposition/singular_values"] = rep["s"]
        metrics[f"{prefix}/decomposition/singular_energy"] = rep["energy"]
        metrics[f"{prefix}/decomposition/singular_energy_fraction"] = rep["energy_fraction"]
        metrics[f"{prefix}/decomposition/effective_rank"] = np.asarray(
            rep["acq"]["effective_rank"], dtype=float
        )
        metrics[f"{prefix}/decomposition/participation_ratio"] = np.asarray(
            rep["acq"]["participation_ratio"], dtype=float
        )
        metrics[f"{prefix}/decomposition/u_panel_mode1to3"] = rep["U_panel"]
        metrics[f"{prefix}/decomposition/score_panel_flat_mode1to3"] = rep[
            "score_panel_flat"
        ]
        metrics[f"{prefix}/decomposition/score_panel_bkr_mode1to3"] = rep[
            "score_panel_bkr"
        ]
        metrics[f"{prefix}/decomposition/rms_mode_panel_bkr_mode1to3"] = rep[
            "rms_mode_panel"
        ]
        metrics[f"{prefix}/decomposition/residual_rms_panel_bkr_after_mode1to3"] = rep[
            "residual_rms_panel"
        ]
        metrics[f"{prefix}/decomposition/rho_panel_after_mode1to3"] = rep["rho_panel"]
        metrics[f"{prefix}/decomposition/total_rms_bkr"] = rep["total_rms_bkr"]

        acq = rep["acq"]
        metrics[f"{prefix}/acquisition_dots/baseline/mu_acq"] = np.asarray(
            acq["mu_acq"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/baseline/sigma_mu_beat"] = np.asarray(
            acq["sigma_mu_beat"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/baseline/mad_mu_beat"] = np.asarray(
            acq["mad_mu_beat"], dtype=float
        )
        metrics[
            f"{prefix}/acquisition_dots/heterogeneity/spatial_mad_mu_median_over_beats"
        ] = np.asarray(acq["spatial_mad_mu_median_over_beats"], dtype=float)

        metrics[f"{prefix}/acquisition_dots/beat_period/mean"] = np.asarray(
            acq["beat_period_mean"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/beat_period/median"] = np.asarray(
            acq["beat_period_median"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/beat_period/std"] = np.asarray(
            acq["beat_period_std"], dtype=float
        )

        metrics[f"{prefix}/acquisition_dots/singular_spectrum/eta1"] = np.asarray(
            acq["eta1"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/singular_spectrum/eta2"] = np.asarray(
            acq["eta2"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/singular_spectrum/eta3"] = np.asarray(
            acq["eta3"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/singular_spectrum/eta23"] = np.asarray(
            acq["eta23"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/singular_spectrum/effective_rank"] = np.asarray(
            acq["effective_rank"], dtype=float
        )
        metrics[f"{prefix}/acquisition_dots/singular_spectrum/participation_ratio"] = np.asarray(
            acq["participation_ratio"], dtype=float
        )

        for m in range(1, rep["n_modes_panel"] + 1):
            mode_key = self._mode_label(m)
            metrics[f"{prefix}/acquisition_dots/mode_panel/{mode_key}/A{m}"] = np.asarray(
                acq[f"A{m}"], dtype=float
            )
            metrics[
                f"{prefix}/acquisition_dots/mode_panel/{mode_key}/median_abs_a{m}"
            ] = np.asarray(acq[f"median_abs_a{m}"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/mode_panel/{mode_key}/sigma_A{m}_beat"
            ] = np.asarray(acq[f"sigma_A{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/mode_panel/{mode_key}/mad_A{m}_beat"
            ] = np.asarray(acq[f"mad_A{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/mode_panel/{mode_key}/cv_A{m}_beat"
            ] = np.asarray(acq[f"cv_A{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/heterogeneity/{mode_key}/spatial_mad_A{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_A{m}_median_over_beats"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/heterogeneity/{mode_key}/spatial_mad_abs_a{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_abs_a{m}_median_over_beats"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/R{m}"
            ] = np.asarray(acq[f"R{m}"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/rho{m}"
            ] = np.asarray(acq[f"rho{m}"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/sigma_R{m}_beat"
            ] = np.asarray(acq[f"sigma_R{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/mad_R{m}_beat"
            ] = np.asarray(acq[f"mad_R{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/cv_R{m}_beat"
            ] = np.asarray(acq[f"cv_R{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/sigma_rho{m}_beat"
            ] = np.asarray(acq[f"sigma_rho{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/mad_rho{m}_beat"
            ] = np.asarray(acq[f"mad_rho{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/residuals/after_{mode_key}/cv_rho{m}_beat"
            ] = np.asarray(acq[f"cv_rho{m}_beat"], dtype=float)
            metrics[
                f"{prefix}/acquisition_dots/heterogeneity/after_{mode_key}/spatial_mad_R{m}_median_over_beats"
            ] = np.asarray(acq[f"spatial_mad_R{m}_median_over_beats"], dtype=float)

        beatwise = rep["beatwise"]
        for key, arr in beatwise.items():
            metrics[f"{prefix}/beatwise/{key}"] = np.asarray(arr, dtype=float)

    def run(self, h5file) -> ProcessResult:
        metrics = {}
        T = self._normalize_T(np.asarray(h5file[self.T_input]))

        rep_map = {
            "raw": self.v_raw_segment_input,
            "bandlimited": self.v_band_segment_input,
        }

        for rep_name, dataset_path in rep_map.items():
            if dataset_path not in h5file:
                metrics[f"{rep_name}/qc/input_available"] = np.asarray(0, dtype=np.uint8)
                continue

            metrics[f"{rep_name}/qc/input_available"] = np.asarray(1, dtype=np.uint8)
            v_block = np.asarray(h5file[dataset_path], dtype=float)
            rep = self._compute_representation(v_block=v_block, T=T)
            self._append_representation_metrics(metrics=metrics, rep_name=rep_name, rep=rep)

        attrs = {
            "pipeline_family": "low_rank_pulsatility",
            "recomputes_svd": True,
            "mode_panel_max": int(self.max_modes_panel),
            "representations": ["raw", "bandlimited"],
            "input_beat_period_path": self.T_input,
            "input_raw_segment_path": self.v_raw_segment_input,
            "input_bandlimited_segment_path": self.v_band_segment_input,
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
