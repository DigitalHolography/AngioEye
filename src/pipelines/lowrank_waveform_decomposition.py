from pathlib import Path

import h5py
import numpy as np

from input_output.hdf5_io import create_h5_file, write_metrics_trees_to_h5
from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT

from .lowrank_svd_math import LowRankSVDMath
from .core.base import (
    ProcessPipeline,
    ProcessResult,
    process_result_to_metrics_tree,
    registerPipeline,
)

@registerPipeline(name="lowrank_waveform_decomposition")

class LowRankSVDMath:
    """Pure-numpy low-rank SVD decomposition and endpoint math for
    beat-aligned waveform blocks.

    Everything here operates on plain numpy arrays only -- no h5py, no
    AngioEye pipeline/schema machinery -- so it can be imported and unit
    tested on its own (e.g. `LowRankSVDMath().compute_representation(...)`
    against a synthetic (n_t, n_beats, n_branches, n_radii) array) without
    an HDF5 fixture. `LowRankWaveformDecomposition` (lowrank_waveform_decomposition.py)
    mixes this class in to get the same methods/attributes on `self`, then
    adds h5py-backed schema resolution and metrics-tree export on top.
    """

    eps = 1e-12
    min_valid_samples_fraction = 0.95
    min_valid_columns = 3
    # Single source of truth for "how many modes does every mode-indexed
    # panel/endpoint/export keep" -- the joint SVD's mode panel, the
    # per-beat SVD's mode panel, and the metrics-export loop all share it.
    exported_modes = 2

    # =====================================================================
    # Safe scalar statistics
    #
    # NaN-aware reductions shared by both the SVD core and the endpoint
    # calculations below; none of these touch the decomposition itself.
    # =====================================================================

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

    # =====================================================================
    # Aggregation reducers
    #
    # The pipeline's canonical "median over (k,r), then median over b"
    # reduction, plus the beat-aggregation helpers used by the endpoint
    # calculations below (and by external reproduction scripts).
    # =====================================================================

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

    def aggregate_beatwise(self, values_per_beat: np.ndarray, stat: str) -> float:
        """Collapses a per-beat array to one acquisition-level scalar via
        the chosen beat-aggregation rule: 'median' (the default paper
        convention) or 'mean' (the robustness variant)."""
        x = np.asarray(values_per_beat, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan")
        return float(np.mean(x)) if stat == "mean" else float(np.median(x))

    def aggregate_rho(self, R_b: np.ndarray, TPR_b: np.ndarray, stat: str) -> float:
        """Ratio of aggregates, not aggregate of ratios: matches both
        acq['rho1'] = R1/(tpr+eps) and Eq. (12)/(14)."""
        R = self.aggregate_beatwise(R_b, stat)
        T = self.aggregate_beatwise(TPR_b, stat)
        if not np.isfinite(R) or not np.isfinite(T) or T <= self.eps:
            return float("nan")
        return float(R / (T + self.eps))

    # =====================================================================
    # Singular-spectrum diagnostics
    #
    # Scalars summarizing how variance is spread across modes, applied both
    # to the acquisition-wide spectrum (joint SVD) and to each beat's own
    # spectrum (per-beat SVD).
    # =====================================================================

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

    def _alpha(self, s: np.ndarray, n_modes: int) -> float:
        s = np.asarray(s, dtype=float)
        if n_modes < 2 or s.size < 2:
            return np.nan
        energy_panel = s[:n_modes] ** 2
        p1 = float(energy_panel[0] / (np.sum(energy_panel) + self.eps))
        if not np.isfinite(p1) or p1 <= 0:
            return np.nan
        return float((1.0 - p1) / p1)

    def _g1(self, s: np.ndarray) -> float:
        """Relative dominant-mode gap G1 = (lambda1 - lambda2) / lambda1
        = 1 - lambda2/lambda1. Requires at least two singular values and
        lambda1 > 0; returns NaN otherwise. Invariant to a common
        multiplicative velocity gain."""
        s = np.asarray(s, dtype=float)
        if s.size < 2:
            return np.nan
        lam1 = float(s[0])
        lam2 = float(s[1])
        if not (np.isfinite(lam1) and np.isfinite(lam2) and lam1 > 0):
            return np.nan
        return float(1.0 - lam2 / lam1)

    # =====================================================================
    # Input shaping
    # =====================================================================

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
    def _mean_subtract(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Removes each column's temporal mean (over axis 0) to produce the
        zero-mean waveform block the SVD operates on. NaN samples are filled
        with their own column mean so gaps become zero after subtraction, and
        any value still non-finite (e.g. an all-NaN column, whose mean is
        itself NaN) is forced to 0. Returns (mu, x_full): the per-column mean
        and the resulting mean-subtracted block. Shared by the joint
        (compute_representation) and per-beat (_svd_beat_panel) paths, which
        differ only in the block's dimensionality -- mu[None] broadcasts over
        the leading time axis in either case."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            mu = np.nanmean(v, axis=0)
        v_filled = np.where(np.isfinite(v), v, mu[None])
        x_full = v_filled - mu[None]
        x_full = np.where(np.isfinite(x_full), x_full, 0.0)
        return mu, x_full

    # =====================================================================
    # SVD METHOD -- core decomposition
    #
    # The joint low-rank model. Every valid (beat, branch, radius) waveform,
    # after its own temporal mean has been removed, becomes one column of a
    # single matrix X of shape (n_t, n_valid_columns); a truncated SVD
    #
    #     X = U S Vt
    #
    # then factors all of those beat-location waveforms at once into a shared
    # set of temporal shape modes. The columns of U are the mode shapes over
    # the beat's time axis; each mode's per-column weights (its "scores") are
    # s[m] * Vt[m, :], i.e. how strongly that shared shape appears in each
    # individual (beat, branch, radius) waveform. Mode signs are arbitrary in
    # an SVD, so each mode is canonicalized to a positive median score for a
    # consistent orientation across acquisitions, and singular values are kept
    # as an energy spectrum (s**2, and its normalized fraction) for the modal
    # diagnostics computed downstream.
    #
    # Everything in this section either performs that decomposition
    # (_run_joint_svd) or is a building block for turning its U/s/Vt output
    # into per-mode panels: the sign convention, mode-sum reconstruction, and
    # the per-mode RMS of a single mode's contribution.
    # =====================================================================

    def _mode_component_rms(
        self, u: np.ndarray, scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        rms_u = float(np.sqrt(np.mean(np.asarray(u, dtype=float) ** 2)))
        comp = np.full(valid_mask.shape, np.nan, dtype=float)
        comp[valid_mask] = np.abs(np.asarray(scores, dtype=float)) * rms_u
        return comp

    @staticmethod
    def _mode_sign_should_flip(scores: np.ndarray) -> bool:
        """The paper's mode sign convention: a mode is oriented so its
        median score is positive. Shared by the joint SVD (_run_joint_svd,
        which also flips its own V^T row) and the per-beat SVD
        (_svd_beat_panel, which has no V^T to carry) -- both apply this
        same test to decide whether to flip a mode's u/scores."""
        med_score = LowRankSVDMath._safe_nanmedian(scores)
        return bool(np.isfinite(med_score) and med_score < 0)

    @staticmethod
    def _reconstruct_mode_sum(U_r: np.ndarray, scores_r: np.ndarray) -> np.ndarray:
        # instead of carrying \Sigma\V^T, we carry a_m(j) = \sigma_m * V_{jm}
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

    def _run_joint_svd(
        self,
        X: np.ndarray,
        n_t: int,
        n_beats: int,
        n_branches: int,
        n_radii: int,
        n_valid_columns: int,
        valid_column_mask: np.ndarray,
    ) -> dict:
        """joint (t,bkr) SVD: one decomposition shared across every
        valid (beat, branch, radius) column at once. Runs np.linalg.svd,
        canonicalizes each mode's sign (median score >= 0), and reshapes
        the per-mode scores into both a flat (mode, valid_column) panel and
        a (mode, beat, branch, radius) panel. Returns the raw decomposition
        products only"""
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        energy = s**2
        energy_fraction = energy / (np.sum(energy) + self.eps)

        n_modes = int(min(self.exported_modes, len(s)))

        score_list: list[np.ndarray] = []
        sign_flips = np.zeros((n_modes,), dtype=int)
        u_panel = np.full((n_t, self.exported_modes), np.nan, dtype=float)
        score_panel_flat = np.full(
            (self.exported_modes, n_valid_columns), np.nan, dtype=float
        )

        for m in range(n_modes):
            scores = s[m] * Vt[m, :]
            if self._mode_sign_should_flip(scores):
                U[:, m] *= -1.0
                Vt[m, :] *= -1.0
                scores *= -1.0
                sign_flips[m] = 1

            u_panel[:, m] = U[:, m]
            score_panel_flat[m, :] = scores
            score_list.append(scores)

        score_panel_bkr = np.full(
            (self.exported_modes, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        for m in range(n_modes):
            score_panel_bkr[m, valid_column_mask] = score_list[m]

        return {
            "U": U,
            "s": s,
            "Vt": Vt,
            "energy": energy,
            "energy_fraction": energy_fraction,
            "n_modes_panel": n_modes,
            "score_list": score_list,
            "sign_flips": sign_flips,
            "U_panel": u_panel,
            "score_panel_flat": score_panel_flat,
            "score_panel_bkr": score_panel_bkr,
        }

    # =====================================================================
    # SVD METHOD -- beat-by-beat with endpoints
    #
    # Rather than one
    # decomposition shared across the whole acquisition, this fixes a
    # single beat, runs an independent SVD over just that beat's own
    # (t, branch, radius) block, and immediately computes that beat's own
    # endpoints from it -- decomposition and endpoint calculation happen
    # together per beat, unlike the joint method above where they're split
    # into separate methods/sections. Repeating this for beat 1, beat 2,
    # ..., beat n_beats and stacking the results gives the beat-by-beat
    # endpoint sequence: A1/A2 (mode amplitude), R1/R2 (residual), TPR
    # (total pulsatile RMS) and MPR (mean pulsatile ratio) all resolved
    # per (branch, radius) at this stage, plus effective_rank/
    # participation_ratio/alpha, which come
    # out as one plain scalar per beat since each beat has only one
    # singular spectrum to begin with. See compute_per_beat_endpoints in
    # "ENDPOINT CALCULATIONS" below for where TPR/MPR/A/R get collapsed
    # over (branch, radius) into the final per-beat scalars, and rho1/rho2
    # get formed from R1/R2 divided by that beat's TPR.
    # =====================================================================

    def _svd_beat_panel(self, v_beat: np.ndarray, beat_period: float) -> dict:
        """Runs the SVD for one fixed beat's own (t, branch, radius) block
        and computes that beat's endpoints from it. Called once per beat by
        per_beat_svd_panels below to build the beat-by-beat endpoint sequence."""
        max_modes = self.exported_modes
        _, n_branches, n_radii = v_beat.shape
        mode_rms = np.full((max_modes, n_branches, n_radii), np.nan, dtype=float)
        residual_rms = np.full((max_modes, n_branches, n_radii), np.nan, dtype=float)
        total_rms = np.full((n_branches, n_radii), np.nan, dtype=float)
        mean_pulsatile_ratio = np.full((n_branches, n_radii), np.nan, dtype=float)
        effective_rank = np.nan
        participation_ratio = np.nan
        alpha = np.nan
        g1 = np.nan

        def _pack(valid_mask: np.ndarray) -> dict:
            return {
                "mode_rms": mode_rms,
                "residual_rms": residual_rms,
                "total_rms": total_rms,
                "mean_pulsatile_ratio": mean_pulsatile_ratio,
                "valid_mask": valid_mask,
                "effective_rank": effective_rank,
                "participation_ratio": participation_ratio,
                "alpha": alpha,
                "g1": g1,
            }

        finite_fraction = np.mean(np.isfinite(v_beat), axis=0)
        valid_mask = finite_fraction >= float(self.min_valid_samples_fraction)
        beat_period_valid = np.isfinite(beat_period) and beat_period > 0
        if not beat_period_valid:
            valid_mask[:] = False
            return _pack(valid_mask)
        if int(np.sum(valid_mask)) < int(self.min_valid_columns):
            return _pack(valid_mask)

        mu, x_full = self._mean_subtract(v_beat)

        X = x_full[:, valid_mask]
        if X.size == 0:
            return _pack(valid_mask)

        total_rms[valid_mask] = np.sqrt(np.mean(X**2, axis=0))
        mean_pulsatile_ratio[valid_mask] = np.abs(mu[valid_mask]) / (
            total_rms[valid_mask] + self.eps
        )

        # -- core SVD call, for this beat only --
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # -- this beat's modal diagnostics, from its own singular spectrum --
        energy = s**2
        energy_fraction = energy / (np.sum(energy) + self.eps)
        effective_rank = self._effective_rank(energy_fraction)
        participation_ratio = self._participation_ratio(energy_fraction)
        n_modes_alpha = int(min(self.exported_modes, len(s)))
        alpha = self._alpha(s, n_modes_alpha)
        g1 = self._g1(s)

        n_modes = int(min(max_modes, len(s)))
        scores_list: list[np.ndarray] = []

        # -- this beat's A/R endpoints: sign convention, mode RMS, residual RMS --
        for m in range(n_modes):
            scores = s[m] * Vt[m, :]
            if self._mode_sign_should_flip(scores):
                U[:, m] *= -1.0
                scores = -scores
            scores_list.append(scores)

            rms_u = float(np.sqrt(np.mean(U[:, m] ** 2)))
            mode_rms[m][valid_mask] = np.abs(scores) * rms_u

            X_recon = self._reconstruct_mode_sum(
                U[:, : m + 1], np.vstack(scores_list[: m + 1])
            )
            residual = X - X_recon
            residual_rms[m][valid_mask] = np.sqrt(np.mean(residual**2, axis=0))

        return _pack(valid_mask)

    def per_beat_svd_panels(self, v_block: np.ndarray, T: np.ndarray) -> dict:
        """Builds the beat-by-beat endpoint sequence: calls
        _svd_beat_panel once per beat (beat 1, beat 2, ..., beat n_beats)
        and stacks each beat's independent SVD + endpoints along the beat
        axis. Returns a dict of stacked panels: mode_rms/residual_rms
        shaped (mode, beat, branch, radius). Each beat's slice comes from
        its own separate decomposition rather than one shared across all beats."""
        T = self._normalize_T(T)
        _, n_beats, n_branches, n_radii = v_block.shape
        max_modes = self.exported_modes
        mode_rms = np.full(
            (max_modes, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        residual_rms = np.full(
            (max_modes, n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        total_rms = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        mean_pulsatile_ratio = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        valid_mask = np.zeros((n_beats, n_branches, n_radii), dtype=bool)
        effective_rank_b = np.full((n_beats,), np.nan, dtype=float)
        participation_ratio_b = np.full((n_beats,), np.nan, dtype=float)
        alpha_b = np.full((n_beats,), np.nan, dtype=float)
        g1_b = np.full((n_beats,), np.nan, dtype=float)

        for b in range(n_beats):
            beat = self._svd_beat_panel(v_block[:, b, :, :], beat_period=float(T[0, b]))
            mode_rms[:, b, :, :] = beat["mode_rms"]
            residual_rms[:, b, :, :] = beat["residual_rms"]
            total_rms[b, :, :] = beat["total_rms"]
            mean_pulsatile_ratio[b, :, :] = beat["mean_pulsatile_ratio"]
            valid_mask[b, :, :] = beat["valid_mask"]
            effective_rank_b[b] = beat["effective_rank"]
            participation_ratio_b[b] = beat["participation_ratio"]
            alpha_b[b] = beat["alpha"]
            g1_b[b] = beat["g1"]

        return {
            "mode_rms": mode_rms,
            "residual_rms": residual_rms,
            "total_rms": total_rms,
            "mean_pulsatile_ratio": mean_pulsatile_ratio,
            "valid_mask": valid_mask,
            "effective_rank_b": effective_rank_b,
            "participation_ratio_b": participation_ratio_b,
            "alpha_b": alpha_b,
            "g1_b": g1_b,
        }

    # =====================================================================
    # ENDPOINT CALCULATIONS -- derived from the SVD above
    # =====================================================================

    def _compute_baseline_endpoints(
        self,
        mu: np.ndarray,
        x_full: np.ndarray,
        valid_column_mask: np.ndarray,
        T: np.ndarray,
        beat_period_valid: np.ndarray,
    ) -> dict:
        """Baseline / total-pulsatility endpoints (TPR, MPR, mpr_prime)
        computed directly from the mean-subtracted waveform block. These
        characterize the raw signal the SVD is subsequently applied to and
        do not depend on _run_joint_svd or _compute_modal_endpoints below
        -- TPR in particular is the denominator every rho{m} endpoint is
        normalized by.

        MPR (paper Eq. eq:mpr_def) is unrelated to rho1/rho2/...: those are
        Rm/R0, whereas MPR is the median of the paired local ratio
        |mu|/RMS_t[w] -- an internal "mpr" name (not "rho0") is used
        throughout to avoid colliding with the paper's own rho0 (trivially
        1, the m=0 base case of the Rm/R0 sequence)."""
        n_beats, n_branches, n_radii = mu.shape
        rms_x = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        rms_x[valid_column_mask] = np.sqrt(
            np.mean(x_full[:, valid_column_mask] ** 2, axis=0)
        )

        mean_pulsatile_ratio_bkr = np.full(
            (n_beats, n_branches, n_radii), np.nan, dtype=float
        )
        mean_pulsatile_ratio_bkr[valid_column_mask] = np.abs(
            mu[valid_column_mask]
        ) / (rms_x[valid_column_mask] + self.eps)

        tpr_b = self._median_kr_per_beat(rms_x, valid_column_mask)
        tpr = self._safe_nanmedian(tpr_b)
        mpr_b = self._median_kr_per_beat(mean_pulsatile_ratio_bkr, valid_column_mask)
        mpr = self._safe_nanmedian(mpr_b)
        abs_mu_b = self._median_kr_per_beat(np.abs(mu), valid_column_mask)
        abs_mu_acq = self._safe_nanmedian(abs_mu_b)
        mpr_prime = (
            float(abs_mu_acq / (tpr + self.eps))
            if np.isfinite(abs_mu_acq) and np.isfinite(tpr) and tpr > self.eps
            else np.nan
        )
        beatwise = {
            "mu_b": self._median_kr_per_beat(mu, valid_column_mask),
            "TPR_b": tpr_b,
            "mpr_b": mpr_b,
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
            "mpr": mpr,
            "sigma_mpr_beat": self._safe_nanstd(mpr_b),
            "mad_mpr_beat": self._safe_nanmad(mpr_b),
            "cv_mpr_beat": self._safe_nancv(mpr_b),
            "abs_mu_acq": abs_mu_acq,
            "mpr_prime": mpr_prime,
            # Display-name alias: the paper (Eq. eq:mpr_def) and every
            # downstream table/figure refer to this endpoint as "MPR".
            "MPR": mpr,
        }
        return {
            "rms_x": rms_x,
            "mean_pulsatile_ratio_bkr": mean_pulsatile_ratio_bkr,
            "tpr_b": tpr_b,
            "tpr": tpr,
            "beatwise": beatwise,
            "acq": acq,
        }

    def _compute_modal_endpoints(
        self,
        X: np.ndarray,
        x_full: np.ndarray,
        svd: dict,
        valid_column_mask: np.ndarray,
        tpr_b: np.ndarray,
        tpr: float,
        n_t: int,
        n_beats: int,
        n_branches: int,
        n_radii: int,
        beatwise: dict,
        acq: dict,
    ) -> dict:
        """Turns _run_joint_svd's mode panels into the acquisition-level
        modal endpoints: per-mode amplitude/residual/ratio values and the
        diagnostics read off the singular spectrum. Mutates `beatwise` and
        `acq` in place, adding these to the baseline endpoints
        _compute_baseline_endpoints already put there, and returns the
        RMS/residual panels for the caller to stash separately."""
        U = svd["U"]
        s = svd["s"]
        score_list = svd["score_list"]
        score_panel_bkr = svd["score_panel_bkr"]
        energy_fraction = svd["energy_fraction"]
        n_modes = svd["n_modes_panel"]

        rms_mode_panel = np.full_like(score_panel_bkr, np.nan, dtype=float)
        residual_rms_panel = np.full_like(score_panel_bkr, np.nan, dtype=float)
        residual_t_bkr_panel = np.full(
            (self.exported_modes, n_t, n_beats, n_branches, n_radii),
            np.nan,
            dtype=float,
        )

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
        acq["alpha"] = self._alpha(s, n_modes)
        acq["G1"] = self._g1(s)

        return {
            "rms_mode_panel": rms_mode_panel,
            "residual_rms_panel": residual_rms_panel,
            "residual_t_bkr_panel": residual_t_bkr_panel,
        }

    def _compute_per_beat_endpoints(self, panels: dict) -> dict:
        """Collapses per_beat_svd_panels' (branch,radius)-resolved panels
        into the final per-beat scalar endpoint sequences: the beat-by-beat
        analogs of the acquisition-level endpoints, one value per beat.

        The amplitude/residual/ratio panels are reduced over (branch, radius)
        per beat, with each rho{m}_b_pb normalized by that beat's own
        TPR_b_pb. The spectrum diagnostics (effective_rank/participation_
        ratio/alpha/G1) are already per-beat scalars and pass through unchanged.
        """
        valid_mask = panels["valid_mask"]
        tpr_b_pb = self._median_kr_per_beat(panels["total_rms"], valid_mask)
        mpr_b_pb = self._median_kr_per_beat(
            panels["mean_pulsatile_ratio"], valid_mask
        )
        a1_b_pb = self._median_kr_per_beat(panels["mode_rms"][0], valid_mask)
        a2_b_pb = self._median_kr_per_beat(panels["mode_rms"][1], valid_mask)
        r1_b_pb = self._median_kr_per_beat(panels["residual_rms"][0], valid_mask)
        r2_b_pb = self._median_kr_per_beat(panels["residual_rms"][1], valid_mask)
        rho1_b_pb = np.where(
            np.isfinite(r1_b_pb) & np.isfinite(tpr_b_pb) & (tpr_b_pb > self.eps),
            r1_b_pb / (tpr_b_pb + self.eps),
            np.nan,
        )
        rho2_b_pb = np.where(
            np.isfinite(r2_b_pb) & np.isfinite(tpr_b_pb) & (tpr_b_pb > self.eps),
            r2_b_pb / (tpr_b_pb + self.eps),
            np.nan,
        )
        return {
            "A1_b_pb": a1_b_pb,
            "A2_b_pb": a2_b_pb,
            "R1_b_pb": r1_b_pb,
            "R2_b_pb": r2_b_pb,
            "rho1_b_pb": rho1_b_pb,
            "rho2_b_pb": rho2_b_pb,
            "TPR_b_pb": tpr_b_pb,
            "MPR_b_pb": mpr_b_pb,
            "effective_rank_b_pb": panels["effective_rank_b"],
            "participation_ratio_b_pb": panels["participation_ratio_b"],
            "alpha_b_pb": panels["alpha_b"],
            "G1_b_pb": panels["g1_b"],
        }

    # =====================================================================
    # Representation orchestrator
    #
    # Ties the sections above together for a single (vessel, representation)
    # waveform block: shape/mask -> mean removal -> baseline endpoints ->
    # joint SVD -> modal endpoints. Returns the `rep` dict the metrics-export
    # and entry-point sections of LowRankWaveformDecomposition consume.
    # =====================================================================

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

        mu, x_full = self._mean_subtract(v_block)
        out["mu"] = mu
        out["x_full"] = x_full

        valid_counts_per_beat = np.sum(valid_column_mask, axis=(1, 2))
        valid_fraction_per_beat = valid_counts_per_beat / float(
            max(1, n_branches * n_radii)
        )
        out["valid_counts_per_beat"] = valid_counts_per_beat
        out["valid_fraction_per_beat"] = valid_fraction_per_beat

        # ---- Baseline / non-modal endpoints (TPR, mpr, mpr_prime) ----
        baseline = self._compute_baseline_endpoints(
            mu=mu,
            x_full=x_full,
            valid_column_mask=valid_column_mask,
            T=T,
            beat_period_valid=beat_period_valid,
        )
        out["rms_x"] = baseline["rms_x"]
        out["total_rms_bkr"] = baseline["rms_x"]
        out["mean_pulsatile_ratio_bkr"] = baseline["mean_pulsatile_ratio_bkr"]
        beatwise = baseline["beatwise"]
        acq = baseline["acq"]
        tpr_b = baseline["tpr_b"]
        tpr = baseline["tpr"]

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

        # ---- SVD METHOD: joint (t,bkr) decomposition ----
        svd = self._run_joint_svd(
            X=X,
            n_t=n_t,
            n_beats=n_beats,
            n_branches=n_branches,
            n_radii=n_radii,
            n_valid_columns=n_valid_columns,
            valid_column_mask=valid_column_mask,
        )

        out["svd_available"] = True
        out["svd_reason"] = "ok"
        out["X"] = X
        out["U"] = svd["U"]
        out["s"] = svd["s"]
        out["Vt"] = svd["Vt"]
        out["energy"] = svd["energy"]
        out["energy_fraction"] = svd["energy_fraction"]
        out["n_modes_panel"] = svd["n_modes_panel"]
        out["U_panel"] = svd["U_panel"]
        out["score_panel_flat"] = svd["score_panel_flat"]
        out["sign_flips"] = svd["sign_flips"]
        out["score_panel_bkr"] = svd["score_panel_bkr"]

        # ---- ENDPOINT CALCULATIONS: derived from the SVD above ----
        modal = self._compute_modal_endpoints(
            X=X,
            x_full=x_full,
            svd=svd,
            valid_column_mask=valid_column_mask,
            tpr_b=tpr_b,
            tpr=tpr,
            n_t=n_t,
            n_beats=n_beats,
            n_branches=n_branches,
            n_radii=n_radii,
            beatwise=beatwise,
            acq=acq,
        )
        out["rms_mode_panel"] = modal["rms_mode_panel"]
        out["residual_rms_panel"] = modal["residual_rms_panel"]
        out["residual_t_bkr_panel"] = modal["residual_t_bkr_panel"]
        out["beatwise"] = beatwise
        out["acq"] = acq
        return out
    
class LowRankWaveformDecomposition(LowRankSVDMath, ProcessPipeline):
    """
    Low-rank SVD decomposition for beat-aligned arterial (and optionally
    venous) segment waveforms.

    For each acquisition, each enabled vessel, and each configured
    waveform source, the pipeline removes the local temporal mean,
    performs one joint SVD over all valid beat-location waveforms, and
    reports the four primary endpoints A1, rho1, A2, and rho2 -- plus,
    independently, a per-beat SVD robustness variant of the same
    endpoints (see "SVD METHOD -- beat-by-beat with endpoints" below).
    Beat period is shared across vessels (a single per-acquisition value
    used for both artery and vein), matching the convention in
    waveform_harmonic_organization.py.

    Vein processing is opt-in via ``process_veins`` (default False): set
    ``helper.process_veins = True`` (or subclass / construct with that
    attribute) before calling ``run`` / ``compute_acquisition_endpoints`` /
    ``write_acquisition_h5`` to also process venous segments.
    """

    description = (
        "Joint low-rank waveform decomposition from beat-aligned arterial "
        "(and optionally venous) segment waveforms, reporting A1, rho1, A2, "
        "rho2, and TPR per acquisition and per beat, for raw and bandlimited "
        "signals."
    )

    # When False (default), only artery segment sources are resolved and
    # processed. Set True to also include vein/raw and vein/bandlimited.
    process_veins = False

    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    v_band_segment_input = (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )
    v_raw_segment_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_raw_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_band_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value"
    )

    T_input_eyeflow = "EyeFlow/Processing/VelocityPerBeat/BeatPeriodSeconds/value"
    v_raw_segment_input_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/Raw/value"
    )
    v_band_segment_input_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
    )
    v_raw_segment_input_vein_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/Raw/value"
    )
    v_band_segment_input_vein_eyeflow = (
        "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
    )

    # =====================================================================
    # The pure-numpy SVD/endpoint math (safe-stat helpers, aggregation
    # reducers, singular-spectrum diagnostics, input shaping, the joint and
    # per-beat SVD, and the endpoint calculations built on them) lives in
    # LowRankSVDMath (lowrank_svd_math.py), mixed into this class below so
    # every `self.xxx(...)` call site here is unchanged -- only schema
    # resolution and metrics-tree export, which need h5py, stay here.
    # =====================================================================

    def _enabled_vessels(self) -> tuple[str, ...]:
        """Vessel compartments this instance will process (always includes
        artery; vein only when ``process_veins`` is True)."""
        return ("artery", "vein") if self.process_veins else ("artery",)

    def _filter_vessel_candidates(self, candidates: dict[str, str]) -> dict[str, str]:
        """Drop vein/* candidates when ``process_veins`` is False."""
        if self.process_veins:
            return candidates
        return {k: v for k, v in candidates.items() if not k.startswith("vein/")}

    def _resolve_vessel_sources(self, h5file) -> tuple[dict[str, str], str] | None:
        """Resolves the (vessel, representation) -> dataset_path candidate
        map for whichever schema family this file uses (current
        EyeFlow/... export schema first, falling back to the older
        Artery/Vein VelocityPerBeat schema), plus the shared beat-period
        path -- beat period is not vessel-specific. Vein candidates are
        omitted unless ``process_veins`` is True. Remaining candidate paths
        are returned whether or not they're actually present in h5file
        (e.g. artery/bandlimited might be missing); callers check
        membership themselves the same way run() always has, so a
        per-combo QC/missing flag can still be reported. Returns None if
        neither schema's beat-period path is present at all."""
        if self.T_input_eyeflow in h5file:
            candidates = {
                "artery/raw": self.v_raw_segment_input_eyeflow,
                "artery/bandlimited": self.v_band_segment_input_eyeflow,
                "vein/raw": self.v_raw_segment_input_vein_eyeflow,
                "vein/bandlimited": self.v_band_segment_input_vein_eyeflow,
            }
            return self._filter_vessel_candidates(candidates), self.T_input_eyeflow
        if self.T_input in h5file:
            candidates = {
                "artery/raw": self.v_raw_segment_input,
                "artery/bandlimited": self.v_band_segment_input,
                "vein/raw": self.v_raw_segment_input_vein,
                "vein/bandlimited": self.v_band_segment_input_vein,
            }
            return self._filter_vessel_candidates(candidates), self.T_input
        return None

    # =====================================================================
    # Metrics export
    # =====================================================================

    @staticmethod
    def _mode_label(m: int) -> str:
        return f"mode{m}"

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

    def _append_config_metrics(
        self, metrics: dict, prefix: str, source_name: str, dataset_path: str
    ) -> None:
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

    def _append_input_metrics(self, metrics: dict, prefix: str, rep: dict) -> None:
        sh = rep["shape"]
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

    def _append_baseline_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
        """Non-modal endpoints (mu, beat period, TPR, MPR, mpr_prime) --
        these characterize the raw signal and don't depend on whether the
        SVD below is available."""
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

        metrics[f"{prefix}/rms/mean_pulsatile_ratio_bkr"] = rep[
            "mean_pulsatile_ratio_bkr"
        ]
        metrics[f"{prefix}/beatwise/mpr_b"] = rep["beatwise"]["mpr_b"]
        metrics[f"{prefix}/endpoints/mpr"] = np.asarray(acq["mpr"], dtype=float)
        metrics[f"{prefix}/variability/sigma_mpr_beat"] = np.asarray(
            acq["sigma_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/mad_mpr_beat"] = np.asarray(
            acq["mad_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/variability/cv_mpr_beat"] = np.asarray(
            acq["cv_mpr_beat"], dtype=float
        )
        metrics[f"{prefix}/baseline/abs_mu_acq"] = np.asarray(
            acq["abs_mu_acq"], dtype=float
        )
        metrics[f"{prefix}/endpoints/mpr_prime"] = np.asarray(
            acq["mpr_prime"], dtype=float
        )

    def _append_qc_unavailable_metrics(
        self, metrics: dict, prefix: str, rep: dict
    ) -> None:
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

    def _append_qc_available_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
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

    def _append_decomposition_metrics(
        self, metrics: dict, prefix: str, rep: dict, acq: dict
    ) -> None:
        """Singular-spectrum diagnostics plus the per-mode panels (u,
        scores, amplitude/residual RMS, beatwise/endpoint/variability
        values) for every exported mode. Only called once QC has confirmed
        the SVD is available."""
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
        metrics[f"{prefix}/decomposition/alpha"] = np.asarray(
            acq["alpha"], dtype=float
        )
        metrics[f"{prefix}/decomposition/G1"] = np.asarray(
            acq["G1"], dtype=float
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

    def _append_representation_metrics(
        self,
        metrics: dict,
        source_name: str,
        dataset_path: str,
        rep: dict,
    ) -> None:
        """Orchestrates the config/inputs/baseline/QC/decomposition
        sub-methods above into the full metrics tree for one (vessel,
        representation) source: config -> inputs -> baseline -> QC (+
        decomposition, only if the SVD was available)."""
        prefix = source_name
        acq = rep["acq"]

        self._append_config_metrics(metrics, prefix, source_name, dataset_path)
        self._append_input_metrics(metrics, prefix, rep)
        self._append_baseline_metrics(metrics, prefix, rep, acq)

        if not rep.get("svd_available", False):
            self._append_qc_unavailable_metrics(metrics, prefix, rep)
            return

        self._append_qc_available_metrics(metrics, prefix, rep, acq)
        self._append_decomposition_metrics(metrics, prefix, rep, acq)

    def _append_per_beat_metrics(
        self,
        metrics: dict,
        source_name: str,
        per_beat_endpoints: dict,
    ) -> None:
        """Writes per_beat_svd_panels + _compute_per_beat_endpoints' per-beat
        endpoint sequences (A1_b_pb, A2_b_pb, R1_b_pb, R2_b_pb, rho1_b_pb,
        rho2_b_pb, TPR_b_pb, MPR_b_pb, effective_rank_b_pb,
        participation_ratio_b_pb, alpha_b_pb -- each a (n_beats,) array)
        into the metrics tree under
        {source_name}/per_beat/{key}, alongside the acquisition-level
        metrics _append_representation_metrics writes for the same
        source_name."""
        prefix = source_name
        for key, arr in per_beat_endpoints.items():
            metrics[f"{prefix}/per_beat/{key}"] = arr

    # =====================================================================
    # Entry points
    #
    # run() is the framework entry point (engine hands us an open h5file);
    # compute_acquisition_endpoints and write_acquisition_h5 are the
    # standalone, path-opening callers for use outside the pipeline_engine.
    # _build_attrs and _compute_source_metrics are shared by run() and
    # write_acquisition_h5(), which otherwise differ only in how they open
    # the file and what they do with the finished metrics dict.
    # =====================================================================

    def _build_attrs(self, representations: list[str], input_beat_period_path: str) -> dict:
        return {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD + per-beat (t,kr) SVD robustness variant",
            "aggregation": "median over (k,r), then median over b",
            "mode_panel_max": int(self.exported_modes),
            "vessels": list(self._enabled_vessels()),
            "process_veins": bool(self.process_veins),
            "representations": representations,
            "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
            "context_endpoint": "TPR",
            "input_beat_period_path": input_beat_period_path,
        }

    def _compute_source_metrics(
        self, h5file, candidates: dict[str, str], T: np.ndarray
    ) -> tuple[dict, list[str]]:
        """For every (vessel, representation) candidate present in h5file,
        computes the joint-SVD and per-beat-SVD metrics and appends them to
        one metrics dict. Returns (metrics, resolved), where resolved is the
        source names that were actually present and processed."""
        metrics: dict = {}
        resolved: list[str] = []
        for source_name, dataset_path in candidates.items():
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

            # _mean_subtract already suppresses the "Mean of empty slice"
            # warning internally, so no outer catch_warnings is needed here.
            rep = self._compute_representation(v_block=v_block, T=T)
            self._append_representation_metrics(
                metrics=metrics,
                source_name=source_name,
                dataset_path=dataset_path,
                rep=rep,
            )

            panels = self.per_beat_svd_panels(v_block, T)
            per_beat_endpoints = self._compute_per_beat_endpoints(panels)
            self._append_per_beat_metrics(
                metrics=metrics,
                source_name=source_name,
                per_beat_endpoints=per_beat_endpoints,
            )

            resolved.append(source_name)

        return metrics, resolved

    def run(self, h5file) -> ProcessResult:
        """Framework entry point: the engine hands us an already-open
        h5file (never a path -- see compute_acquisition_endpoints/
        write_acquisition_h5 for the standalone, path-opening callers).
        Resolves the vessel/representation schema (current EyeFlow/...
        first, legacy Artery/Vein/... fallback) and, for every combination
        the file has, computes both the joint-SVD and per-beat-SVD
        endpoints, returning them as the pipeline's metrics tree."""
        legacy_candidates = self._filter_vessel_candidates(
            {
                "artery/raw": self.v_raw_segment_input,
                "artery/bandlimited": self.v_band_segment_input,
                "vein/raw": self.v_raw_segment_input_vein,
                "vein/bandlimited": self.v_band_segment_input_vein,
            }
        )
        schema = self._resolve_vessel_sources(h5file)
        if schema is None:
            metrics = {}
            for source_name, dataset_path in legacy_candidates.items():
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
            attrs = self._build_attrs([], self.T_input)
            return ProcessResult(metrics=metrics, attrs=attrs)

        candidates, t_path = schema
        T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))
        metrics, resolved = self._compute_source_metrics(h5file, candidates, T)
        attrs = self._build_attrs(resolved, t_path)
        return ProcessResult(metrics=metrics, attrs=attrs)

    def compute_acquisition_endpoints(self, h5_path) -> dict | None:
        """Standalone entry point for callers outside the AngioEye
        pipeline_engine framework (e.g. an external reproduction script):
        given a path to one acquisition's raw .h5 file, resolves whichever
        known schema it uses and, for each enabled vessel (artery always;
        vein only when ``process_veins`` is True), runs both the joint and
        per-beat SVDs on that vessel's raw segment.

        Returns None if the file has no known beat-period path or no
        enabled vessel's raw segment at all. Otherwise returns a dict keyed
        by enabled vessel name (``{"artery": {...} | None}``, and also
        ``"vein"`` when ``process_veins``), where a vessel is None when its
        raw segment is absent or its SVD was unavailable (too few valid
        columns). Each vessel dict carries the endpoints plus the raw "mu"
        and "energy_fraction" arrays some callers need directly."""
        with h5py.File(h5_path, "r") as h5file:
            schema = self._resolve_vessel_sources(h5file)
            if schema is None:
                return None
            candidates, t_path = schema
            T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))

            vessel_blocks: dict[str, np.ndarray | None] = {}
            for vessel in self._enabled_vessels():
                raw_path = candidates.get(f"{vessel}/raw")
                if raw_path is not None and raw_path in h5file:
                    vessel_blocks[vessel] = np.asarray(h5file[raw_path], dtype=float)
                else:
                    vessel_blocks[vessel] = None

        if all(v_block is None for v_block in vessel_blocks.values()):
            return None

        result: dict[str, dict | None] = {}
        for vessel, v_block in vessel_blocks.items():
            if v_block is None:
                result[vessel] = None
                continue

            v_block = self._ensure_segment_shape(v_block, T)

            # _mean_subtract already suppresses the "Mean of empty slice"
            # warning internally, so no outer catch_warnings is needed here.
            rep = self._compute_representation(v_block=v_block, T=T)
            if not rep.get("svd_available", False):
                result[vessel] = None
                continue

            per_beat_panels = self.per_beat_svd_panels(v_block, T)
            per_beat_svd = self._compute_per_beat_endpoints(per_beat_panels)
            valid_fraction_per_beat = np.asarray(
                rep.get("valid_fraction_per_beat", []), dtype=float
            )
            beat_period_arr = np.asarray(T[0], dtype=float)

            result[vessel] = {
                "acq": rep["acq"],
                "beatwise": rep["beatwise"],
                "per_beat_svd": per_beat_svd,
                "mu": rep["mu"],
                "energy_fraction": np.asarray(rep.get("energy_fraction", []), dtype=float),
                "beat_period_mean": (
                    float(np.nanmean(beat_period_arr))
                    if beat_period_arr.size
                    else float("nan")
                ),
                "beat_period_sd": (
                    float(np.nanstd(beat_period_arr, ddof=1))
                    if beat_period_arr.size > 1
                    else float("nan")
                ),
                "beat_period_b": beat_period_arr,
                "valid_fraction_per_beat": valid_fraction_per_beat,
                "n_valid_columns": int(rep["shape"]["n_valid_columns"]),
                "n_total_columns": int(rep["shape"]["n_total_columns"]),
            }

        return result

    def write_acquisition_h5(self, h5_path, out_path: Path | str) -> bool:
        """Standalone counterpart to run() for callers outside the
        pipeline_engine framework: writes the same metrics run() would
        produce -- acquisition-level joint-SVD plus per-beat SVD, for every
        enabled (vessel, representation) the file has (vein only when
        ``process_veins`` is True) -- to a new .h5 at out_path.

        Uses the same MetricsTree/ProcessResult conversion and
        ANGIOEYE_PROCESSING_ROOT group as the normal batch workflow, so the
        output is found where a downstream reader expects. Only the computed
        metrics are written, not the (often large) source waveform arrays.

        Returns False (and writes nothing) if the file has no known schema.
        """
        with h5py.File(h5_path, "r") as h5file:
            schema = self._resolve_vessel_sources(h5file)
            if schema is None:
                return False
            candidates, t_path = schema
            T = self._normalize_T(np.asarray(h5file[t_path], dtype=float))
            metrics, resolved = self._compute_source_metrics(h5file, candidates, T)

        if not resolved:
            return False

        attrs = self._build_attrs(resolved, t_path)

        create_h5_file(out_path, source_file=str(h5_path), trim_source=True)
        metric_tree = process_result_to_metrics_tree(
            self.name, ProcessResult(metrics=metrics, attrs=attrs)
        )
        write_metrics_trees_to_h5(
            out_path, ANGIOEYE_PROCESSING_ROOT, [metric_tree], overwrite=False
        )
        return True

"""Low-rank flicker confound-controlled statistics.

Takes a batch of raw acquisition ``.h5`` files, classifies each into a dataset
and experimental epoch (baseline1 / flicker / baseline2), runs the low-rank
waveform engine on every acquisition, and writes the paper's Sec. V.A/V.B/V.C
outputs as CSVs under ``<output>/lowrank_confound_statistics/``:

- ``points/``  one acquisition-level dot per endpoint (plus beat-to-beat SDs)
- ``beats/``  the underlying per-beat endpoint arrays (long format)
- ``tables/``  Table I/II endpoint summaries with per-epoch tests
- ``confound_control/``  the Sec. V.B robustness grid and per-metric verdicts

Organisation: ``Statistics`` (Sec. V.A/V.C tests + endpoint tables) and
``Confounds`` (Sec. V.B robustness grid) hold the analysis logic;
dataset/epoch classification and per-acquisition collection feed them;
``run_confound_statistics`` orchestrates; ``LowRankConfoundStatisticsPostprocess``
is the registered postprocess wrapper.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from pipelines.lowrank_waveform_decomposition import LowRankWaveformDecomposition

from ..core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)

EPOCH_ORDER = ["baseline1", "flicker", "baseline2"]

_LR = LowRankWaveformDecomposition()
# Confound statistics report both compartments; opt into the pipeline's
# optional vein processing (artery-only is the pipeline default).
_LR.process_veins = True


# =====================================================================
# STATISTICS (Sec. V.A / V.C)
# Nonparametric primitives and Table I/II endpoint summaries.
# =====================================================================


class Statistics:
    """Sec. V.A nonparametric test bundle and Sec. V.C endpoint tables."""

    TABLE_METRICS = ["rho2", "A2", "rho1", "A1", "TPR", "mpr", "mpr_prime", "alpha", "G1"]

    @staticmethod
    def clean(x) -> np.ndarray:
        """Return ``x`` as a 1-D float array with non-finite (NaN/inf) entries
        dropped, so downstream tests never see missing values."""
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Cliff's delta effect size: Pr(X>Y) - Pr(X<Y), in [-1, 1]. A positive
        delta means values in ``x`` tend to exceed those in ``y``. Returns NaN if
        either sample is empty after cleaning."""
        x = Statistics.clean(x)
        y = Statistics.clean(y)
        if x.size == 0 or y.size == 0:
            return float("nan")
        diff = x[:, None] - y[None, :]
        return float(np.mean(diff > 0) - np.mean(diff < 0))

    @staticmethod
    def holm_adjust(p_values: list[float]) -> list[float]:
        """Holm-Bonferroni step-down correction, returning adjusted p-values in the
        same order as the input. NaN entries stay NaN and are excluded from the
        family size, so missing tests don't inflate the correction applied to the
        rest; adjusted values are monotone non-decreasing and capped at 1.0."""
        p = np.asarray(p_values, dtype=float)
        n_valid = int(np.sum(np.isfinite(p)))
        order = np.argsort(np.where(np.isfinite(p), p, np.inf))
        adjusted = np.full(p.size, np.nan)
        running_max = 0.0
        rank = 0
        for idx in order:
            if not np.isfinite(p[idx]):
                continue
            running_max = max(running_max, (n_valid - rank) * p[idx])
            adjusted[idx] = min(running_max, 1.0)
            rank += 1
        return adjusted.tolist()

    @staticmethod
    def epoch_group_tests(epoch_values: dict[str, np.ndarray]) -> dict:
        """Sec. V.A nonparametric test bundle for one metric across the three
        epochs (baseline1/flicker/baseline2 arrays of acquisition-level dots).

        Returns a dict with: the Kruskal-Wallis p-value across all three epochs;
        a ``pairwise`` list of the three two-sided Mann-Whitney U tests (raw p,
        Holm-adjusted p, Cliff's delta, in baseline-first order); the pooled
        baseline (B1+B2) vs flicker p-value and delta; and per-epoch sample sizes.
        Tests that lack enough data are reported as NaN rather than raising."""
        b1 = Statistics.clean(epoch_values["baseline1"])
        fl = Statistics.clean(epoch_values["flicker"])
        b2 = Statistics.clean(epoch_values["baseline2"])
        pooled_baseline = np.concatenate([b1, b2])

        kw_p = np.nan
        if b1.size >= 1 and fl.size >= 1 and b2.size >= 1:
            try:
                kw_p = float(kruskal(b1, fl, b2).pvalue)
            except ValueError:
                kw_p = np.nan

        pairs = [
            ("baseline1", "flicker", b1, fl),
            ("flicker", "baseline2", fl, b2),
            ("baseline1", "baseline2", b1, b2),
        ]
        raw_p = [
            float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            if x.size >= 2 and y.size >= 2
            else np.nan
            for _, _, x, y in pairs
        ]
        holm_p = Statistics.holm_adjust(raw_p)
        deltas = [Statistics.cliffs_delta(x, y) for _, _, x, y in pairs]

        pooled_p = np.nan
        pooled_delta = np.nan
        if pooled_baseline.size >= 2 and fl.size >= 2:
            pooled_p = float(
                mannwhitneyu(fl, pooled_baseline, alternative="two-sided").pvalue
            )
            pooled_delta = Statistics.cliffs_delta(fl, pooled_baseline)

        return {
            "kruskal_wallis_p": kw_p,
            "pairwise": [
                {"pair": f"{a} vs {b}", "p": p, "p_holm": ph, "cliffs_delta": d}
                for (a, b, _, _), p, ph, d in zip(pairs, raw_p, holm_p, deltas)
            ],
            "pooled_baseline_vs_flicker_p": pooled_p,
            "pooled_baseline_vs_flicker_delta": pooled_delta,
            "n": {"baseline1": b1.size, "flicker": fl.size, "baseline2": b2.size},
        }

    @classmethod
    def build_endpoint_table(
        cls, dataset_name: str, vessel: str, points_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the Table I/II summary (one row per metric in TABLE_METRICS) from
        a vessel's acquisition-level points frame: per-epoch median/SD/n, the
        Kruskal-Wallis and Holm-adjusted pairwise p-values, the pooled
        baseline-vs-flicker p-value, and flicker-vs-baseline Cliff's deltas."""
        rows = []
        for metric in cls.TABLE_METRICS:
            b1 = points_df.loc[points_df["epoch"] == "B1", metric].to_numpy(dtype=float)
            fl = points_df.loc[points_df["epoch"] == "Flicker", metric].to_numpy(
                dtype=float
            )
            b2 = points_df.loc[points_df["epoch"] == "B2", metric].to_numpy(dtype=float)
            tests = cls.epoch_group_tests(
                {"baseline1": b1, "flicker": fl, "baseline2": b2}
            )
            pair_lookup = {p["pair"]: p for p in tests["pairwise"]}

            b1c, flc, b2c = cls.clean(b1), cls.clean(fl), cls.clean(b2)
            pooled_baseline = np.concatenate([b1c, b2c])

            rows.append(
                {
                    "dataset": dataset_name,
                    "vessel": vessel,
                    "metric": metric,
                    "B1_median": np.nanmedian(b1) if b1.size else np.nan,
                    "B1_sd": np.nanstd(b1, ddof=1) if b1c.size > 1 else np.nan,
                    "n_B1": int(b1c.size),
                    "Flicker_median": np.nanmedian(fl) if fl.size else np.nan,
                    "Flicker_sd": np.nanstd(fl, ddof=1) if flc.size > 1 else np.nan,
                    "n_Flicker": int(flc.size),
                    "B2_median": np.nanmedian(b2) if b2.size else np.nan,
                    "B2_sd": np.nanstd(b2, ddof=1) if b2c.size > 1 else np.nan,
                    "n_B2": int(b2c.size),
                    "kruskal_wallis_p": tests["kruskal_wallis_p"],
                    "B1_vs_F_p_holm": pair_lookup["baseline1 vs flicker"]["p_holm"],
                    "F_vs_B2_p_holm": pair_lookup["flicker vs baseline2"]["p_holm"],
                    "B1_vs_B2_p_holm": pair_lookup["baseline1 vs baseline2"]["p_holm"],
                    "pooled_baseline_vs_flicker_p": tests["pooled_baseline_vs_flicker_p"],
                    # Flicker-first convention (matches paper's "Cliff's delta (F vs ...)"
                    # columns), NOT the epoch_group_tests pairwise list's (baseline,
                    # flicker) order, whose first-pair delta sign is flipped relative
                    # to this.
                    "cliffs_delta_F_vs_B1": cls.cliffs_delta(flc, b1c),
                    "cliffs_delta_F_vs_B2": cls.cliffs_delta(flc, b2c),
                    "cliffs_delta_F_vs_pooled_baseline": cls.cliffs_delta(
                        flc, pooled_baseline
                    ),
                }
            )
        return pd.DataFrame(rows)


# Module-level aliases for callers that import the free functions by name.
clean = Statistics.clean
cliffs_delta = Statistics.cliffs_delta
holm_adjust = Statistics.holm_adjust
epoch_group_tests = Statistics.epoch_group_tests
build_endpoint_table = Statistics.build_endpoint_table


EPOCH_ALIASES = {
    "bl1": "baseline1",
    "b1": "baseline1",
    "baseline1": "baseline1",
    "f": "flicker",
    "flicker": "flicker",
    "bl2": "baseline2",
    "b2": "baseline2",
    "baseline2": "baseline2",
}
EPOCH_SHORT = {"baseline1": "B1", "flicker": "Flicker", "baseline2": "B2"}
EPOCH_SHORT_TO_KEY = {v: k for k, v in EPOCH_SHORT.items()}


def _epoch_rank(epoch_short: str) -> int:
    """Position of a short epoch label (B1/Flicker/B2) in EPOCH_ORDER, for
    sorting rows into the canonical baseline1 -> flicker -> baseline2 order."""
    return EPOCH_ORDER.index(EPOCH_SHORT_TO_KEY[epoch_short])


# Acquisition-number patterns tried in order against a file name/path:
# BOM ("<n>_HD"), OSS ("OSS_<n>" / "OSS_R_<n>"), then a generic "_<n>_"/"_<n>.h5".
_BOM_ACQ_RE = re.compile(r"(\d+)_HD")
_OSS_ACQ_RE = re.compile(r"OSS(?:_R)?_(\d+)")
_GENERIC_ACQ_RE = re.compile(r"_(\d+)(?:_|\.h5$)")


def acquisition_index(path: Path) -> int:
    """Extract the integer acquisition number from a file's name (falling back
    to its full path) using the naming patterns above, tried in order. Returns
    a large sentinel (1e9) when none match, so unrecognised files sort last."""
    for pattern in (_BOM_ACQ_RE, _OSS_ACQ_RE, _GENERIC_ACQ_RE):
        for candidate in (path.name, str(path)):
            match = pattern.search(candidate)
            if match:
                return int(match.group(1))
    return 10**9


def _classify_epoch(relative_parts: tuple[str, ...]) -> str | None:
    """Map the first path component that matches a known epoch alias (see
    EPOCH_ALIASES) to its canonical epoch name, for the folder-named layout
    (e.g. a ``.../flicker/...`` subfolder). Returns None if none match."""
    for part in relative_parts:
        epoch = EPOCH_ALIASES.get(part.lower())
        if epoch is not None:
            return epoch
    return None


MANIP_EPOCH_CODES = {"B1": "baseline1", "B": "baseline1", "F": "flicker", "B2": "baseline2"}


def parse_manip_txt(path: Path) -> dict[int, str]:
    """Parse a dataset's ``manip.txt`` into an acquisition-index -> epoch map.

    Each line is ``<ranges> <code>`` where ``<code>`` is a MANIP_EPOCH_CODES
    key (B1/B/F/B2) and ``<ranges>`` is a comma-separated list of acquisition
    indices or inclusive ``lo-hi`` spans, e.g. ``1-10,13 B1``. Lines with an
    unknown code or no range are ignored; later lines win on overlap."""
    epoch_for_index: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        range_spec, _, code = line.rpartition(" ")
        if not range_spec:
            continue
        epoch = MANIP_EPOCH_CODES.get(code.strip().upper())
        if epoch is None:
            continue
        for chunk in range_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                lo_s, hi_s = chunk.split("-", 1)
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            else:
                lo = hi = int(chunk)
            for idx in range(lo, hi + 1):
                epoch_for_index[idx] = epoch
    return epoch_for_index


def _epoch_for_acquisition(
    h5_path: Path,
    dataset_root: Path,
    subfolder_parts: tuple[str, ...],
    manip_cache: dict[Path, dict[int, str] | None],
) -> str | None:
    """Epoch classification for one candidate dataset root: that dataset's
    own manip.txt (acquisition-index ranges) if present, else
    EPOCH_ALIASES-based matching against the subfolder path between
    dataset_root and the file (the older, nested-epoch-subfolder
    "*_Unorganised" layout). manip_cache memoizes one parse_manip_txt call
    per dataset root across the whole batch."""
    manip_path = dataset_root / "manip.txt"
    if manip_path not in manip_cache:
        manip_cache[manip_path] = (
            parse_manip_txt(manip_path) if manip_path.exists() else None
        )
    epoch_for_index = manip_cache[manip_path]
    if epoch_for_index is not None:
        return epoch_for_index.get(acquisition_index(h5_path))
    return _classify_epoch(subfolder_parts)


def classify_dataset_and_epoch(
    h5_path: Path,
    input_root: Path,
    manip_cache: dict[Path, dict[int, str] | None],
) -> tuple[str, str] | None:
    """Classifies one raw acquisition path (e.g. from
    context.input_h5_paths) into (dataset_name, epoch), given the --data
    root it was discovered under (context.input_path). Supports both
    invocation styles: --data pointed at the shared parent of several
    dataset roots (the normal multi-dataset case -- dataset_name is the
    first path component under input_root), and --data pointed directly
    at a single dataset root (dataset_name is input_root's own folder
    name). Returns None if the file isn't under input_root, or no epoch
    could be determined either way."""
    h5_path = Path(h5_path)
    try:
        rel = h5_path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None

    if len(parts) >= 2:
        dataset_name = parts[0]
        epoch = _epoch_for_acquisition(
            h5_path, input_root / dataset_name, parts[1:-1], manip_cache
        )
        if epoch is not None:
            return dataset_name, epoch

    epoch = _epoch_for_acquisition(h5_path, input_root, parts[:-1], manip_cache)
    if epoch is not None:
        return input_root.name, epoch

    return None


def group_by_dataset(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[dict[str, list[tuple[str, Path]]], list[Path]]:
    """Classify every raw acquisition path into (dataset, epoch) via
    classify_dataset_and_epoch, then group per dataset and sort each group by
    (epoch order, acquisition index, path). Returns (records_by_dataset,
    skipped), where ``skipped`` collects every path that couldn't be
    classified (e.g. not covered by that dataset's manip.txt)."""
    manip_cache: dict[Path, dict[int, str] | None] = {}
    records_by_dataset: dict[str, list[tuple[str, Path]]] = {}
    skipped: list[Path] = []
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        result = classify_dataset_and_epoch(h5_path, input_root, manip_cache)
        if result is None:
            skipped.append(h5_path)
            continue
        dataset_name, epoch = result
        records_by_dataset.setdefault(dataset_name, []).append((epoch, h5_path))

    for records in records_by_dataset.values():
        records.sort(
            key=lambda r: (EPOCH_ORDER.index(r[0]), acquisition_index(r[1]), str(r[1]))
        )
    return records_by_dataset, skipped


# =====================================================================
# PER-ACQUISITION COLLECTION
# The heavy lifting (h5 opening with retry, schema resolution, the joint SVD,
# and the per-beat SVD variant, for both artery and vein) lives in
# LowRankWaveformDecomposition.compute_acquisition_endpoints. The functions
# below just call it and fan the per-acquisition result out into the per-vessel
# points and beats tables.
# =====================================================================

VESSEL_TYPES = ("artery", "vein")


def build_beat_rows(
    dataset_name: str,
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> list[dict]:
    """Expand one acquisition into one long-format row per beat, keeping the
    per-beat endpoint arrays the engine computes in both representations: the
    joint-SVD ``beatwise`` values and the per-beat-SVD ``*_pb`` variants.

    ``alpha`` and ``mpr_prime`` are acquisition-level-only by construction
    (whole singular spectrum / ratio of aggregates) and have no per-beat value,
    so they appear only in the points table, not here. Missing/short entries
    are filled with NaN."""
    beatwise = vessel_data["beatwise"]
    per_beat_svd = vessel_data["per_beat_svd"]
    n_beats = len(beatwise["TPR_b"])
    vfb = vessel_data["valid_fraction_per_beat"]
    period_b = vessel_data["beat_period_b"]

    def at(arr, b):
        arr = np.asarray(arr, dtype=float)
        return float(arr[b]) if b < arr.size and np.isfinite(arr[b]) else float("nan")

    rows = []
    for b in range(n_beats):
        rows.append(
            {
                "dataset": dataset_name,
                "vessel": vessel,
                "acquisition": acquisition_index(h5_path),
                "sequence": sequence,
                "file": h5_path.name,
                "epoch": epoch_short,
                "beat_index": b,
                "beat_period": at(period_b, b),
                "valid_fraction": at(vfb, b),
                "mu": at(beatwise["mu_b"], b),
                "TPR": at(beatwise["TPR_b"], b),
                "mpr": at(beatwise["mpr_b"], b),
                "A1": at(beatwise["A1_b"], b),
                "R1": at(beatwise["R1_b"], b),
                "rho1": at(beatwise["rho1_b"], b),
                "A2": at(beatwise["A2_b"], b),
                "R2": at(beatwise["R2_b"], b),
                "rho2": at(beatwise["rho2_b"], b),
                "A1_pb": at(per_beat_svd["A1_b_pb"], b),
                "R1_pb": at(per_beat_svd["R1_b_pb"], b),
                "A2_pb": at(per_beat_svd["A2_b_pb"], b),
                "R2_pb": at(per_beat_svd["R2_b_pb"], b),
            }
        )
    return rows


def build_points_row(
    dataset_name: str,
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> dict:
    """One acquisition-level row (a single dot per endpoint) for the points
    table: the acquisition-aggregated endpoints plus their beat-to-beat SDs,
    mean beat period, and column-validity bookkeeping."""
    acq = vessel_data["acq"]
    vfb = vessel_data["valid_fraction_per_beat"]

    def scalar(key: str) -> float:
        return float(acq.get(key, np.nan))

    return {
        "dataset": dataset_name,
        "vessel": vessel,
        "acquisition": acquisition_index(h5_path),
        "sequence": sequence,
        "file": h5_path.name,
        "epoch": epoch_short,
        "A1": scalar("A1"),
        "A1_sd": scalar("sigma_A1_beat"),
        "rho1": scalar("rho1"),
        "rho1_sd": scalar("sigma_rho1_beat"),
        "A2": scalar("A2"),
        "A2_sd": scalar("sigma_A2_beat"),
        "rho2": scalar("rho2"),
        "rho2_sd": scalar("sigma_rho2_beat"),
        "TPR": scalar("TPR"),
        "TPR_sd": scalar("sigma_TPR_beat"),
        "mpr": scalar("mpr"),
        "mpr_sd": scalar("sigma_mpr_beat"),
        "mpr_prime": scalar("mpr_prime"),
        "alpha": scalar("alpha"),
        "G1": scalar("G1"),
        "beat_period": vessel_data["beat_period_mean"],
        "beat_period_sd": vessel_data["beat_period_sd"],
        "mu": scalar("mu_acq"),
        "mu_sd": scalar("sigma_mu_beat"),
        "valid_fraction": float(np.nanmean(vfb)) if vfb.size else float("nan"),
        "valid_fraction_sd": (
            float(np.nanstd(vfb, ddof=1)) if vfb.size > 1 else float("nan")
        ),
        "n_valid_columns": vessel_data["n_valid_columns"],
        "n_total_columns": vessel_data["n_total_columns"],
    }


def collect_dataset(
    name: str,
    records: list[tuple[str, Path]],
    h5_out_dir: Path | None = None,
) -> dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]]:
    """Runs the low-rank engine over one dataset's already-classified,
    already-sorted (epoch, h5_path) records (see group_by_dataset), and
    builds each vessel's acqs_by_epoch/points_rows/beat_rows. Returns
    {"artery": (acqs_by_epoch, points_rows, beat_rows), "vein": (...)} --
    an acquisition contributes to a vessel's tables only if that vessel's
    raw segment was present and its joint SVD was available for it (see
    compute_acquisition_endpoints)."""
    per_vessel: dict[str, dict] = {
        vessel: {
            "acqs_by_epoch": {e: [] for e in EPOCH_ORDER},
            "points_rows": [],
            "beat_rows": [],
            "sequence_counters": {e: 0 for e in EPOCH_ORDER},
        }
        for vessel in VESSEL_TYPES
    }

    for epoch, h5_path in records:
        data = _LR.compute_acquisition_endpoints(h5_path)
        if data is None:
            continue
        if h5_out_dir is not None:
            # Also persist this acquisition's joint-SVD + per-beat endpoints
            # (both vessels, both representations) to their own .h5 (the same
            # MetricsTree/ANGIOEYE_PROCESSING_ROOT structure a normal batch run
            # produces), so they aren't confined to this run's in-memory
            # points/beats tables.
            _LR.write_acquisition_h5(
                h5_path, h5_out_dir / f"{h5_path.stem}_pipelines_result.h5"
            )

        for vessel in VESSEL_TYPES:
            vessel_data = data.get(vessel)
            if vessel_data is None:
                continue
            state = per_vessel[vessel]
            state["acqs_by_epoch"][epoch].append(vessel_data)
            seq = state["sequence_counters"][epoch]
            state["sequence_counters"][epoch] += 1
            epoch_short = EPOCH_SHORT[epoch]

            state["points_rows"].append(
                build_points_row(name, vessel, h5_path, seq, epoch_short, vessel_data)
            )
            state["beat_rows"].extend(
                build_beat_rows(name, vessel, h5_path, seq, epoch_short, vessel_data)
            )

    result: dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]] = {}
    for vessel, state in per_vessel.items():
        points_rows = state["points_rows"]
        beat_rows = state["beat_rows"]
        points_rows.sort(
            key=lambda r: (_epoch_rank(r["epoch"]), r["acquisition"])
        )
        beat_rows.sort(
            key=lambda r: (_epoch_rank(r["epoch"]), r["acquisition"], r["beat_index"])
        )
        result[vessel] = (state["acqs_by_epoch"], points_rows, beat_rows)
    return result


# =====================================================================
# CONFOUNDS (Sec. V.B)
# Robustness grid over SVD method / beat aggregation / beat-period control.
# =====================================================================


class Confounds:
    """Sec. V.B confound-control grid: analysis-choice sweep + verdicts."""

    METRICS_SVD = ["A1", "A2", "rho1", "rho2"]

    def __init__(self, engine: LowRankWaveformDecomposition | None = None):
        self._lr = engine if engine is not None else _LR

    @staticmethod
    def residualize_against_beat_period(
        epoch_values: dict[str, np.ndarray],
        epoch_beat_periods: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Sec. V.B.4 beat-period control. Pools the metric and the mean beat
        period across all acquisitions/epochs, fits ``d = alpha + beta*T`` by OLS
        (one global line), and returns each value replaced by its residual from
        that line. If fewer than 3 finite (value, period) pairs exist the fit is
        skipped and the inputs are returned unchanged (as float arrays)."""
        all_values = np.concatenate(
            [np.asarray(v, dtype=float) for v in epoch_values.values()]
        )
        all_periods = np.concatenate(
            [np.asarray(v, dtype=float) for v in epoch_beat_periods.values()]
        )
        mask = np.isfinite(all_values) & np.isfinite(all_periods)
        if np.sum(mask) < 3:
            return {k: np.asarray(v, dtype=float) for k, v in epoch_values.items()}

        slope, intercept = np.polyfit(all_periods[mask], all_values[mask], 1)

        out = {}
        for epoch, values in epoch_values.items():
            values = np.asarray(values, dtype=float)
            periods = np.asarray(epoch_beat_periods[epoch], dtype=float)
            out[epoch] = values - (slope * periods + intercept)
        return out

    def acq_dots(
        self,
        acqs: list[dict],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> np.ndarray:
        """One acquisition-level value ("dot") per acquisition for the given metric,
        under a specific analysis choice. ``stat`` is the beat-aggregation
        ("median"/"mean") and ``svd_method`` selects the joint vs per-beat SVD
        representation; both are ignored for metrics that have no such axis (alpha,
        mpr_prime are acquisition scalars; TPR, mpr have no SVD axis). See the
        per-branch comments for how each metric maps to the engine output."""
        out = []
        for a in acqs:
            if metric == "TPR":
                out.append(self._lr.aggregate_beatwise(a["beatwise"]["TPR_b"], stat))
            elif metric == "mpr":
                # Not SVD-derived (ratio of mu to pulsatile RMS, Eq. 18) -- same
                # beat-aggregation treatment as TPR, no joint/per-beat SVD axis.
                out.append(self._lr.aggregate_beatwise(a["beatwise"]["mpr_b"], stat))
            elif metric == "alpha":
                # Single acquisition-level scalar from the joint-SVD singular
                # spectrum (Eq. 19), like effective_rank/participation_ratio --
                # no per-beat array, so svd_method/stat are not applicable axes.
                out.append(float(a["acq"].get("alpha", np.nan)))
            elif metric == "G1":
                # Exploratory dominant-mode gap (Eq. 23): 1 - lambda2/lambda1
                # from the joint-SVD singular values -- same "no beat-aggregation
                # axis" treatment as alpha.
                out.append(float(a["acq"].get("G1", np.nan)))
            elif metric == "mpr_prime":
                # MPR' (Eq. 17): ratio of two separately-aggregated acquisition
                # scalars (median|mu| / R0), not an average of per-beat ratios --
                # same "no beat-aggregation axis" treatment as alpha.
                out.append(float(a["acq"].get("mpr_prime", np.nan)))
            elif metric in ("A1", "A2"):
                if svd_method == "joint":
                    arr = a["beatwise"][f"{metric}_b"]
                else:
                    arr = a["per_beat_svd"][f"{metric}_b_pb"]
                out.append(self._lr.aggregate_beatwise(arr, stat))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "joint":
                    R_b = a["beatwise"][f"R{m}_b"]
                else:
                    R_b = a["per_beat_svd"][f"R{m}_b_pb"]
                out.append(self._lr.aggregate_rho(R_b, a["beatwise"]["TPR_b"], stat))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return np.asarray(out, dtype=float)

    def epoch_dots(
        self,
        acqs_by_epoch: dict[str, list[dict]],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> dict[str, np.ndarray]:
        """``acq_dots`` applied per epoch, returning {epoch: dots} keyed by
        EPOCH_ORDER (the shape Statistics.epoch_group_tests /
        residualize_against_beat_period expect)."""
        return {
            epoch: self.acq_dots(acqs_by_epoch[epoch], metric, svd_method, stat)
            for epoch in EPOCH_ORDER
        }

    @staticmethod
    def epoch_beat_periods(
        acqs_by_epoch: dict[str, list[dict]],
    ) -> dict[str, np.ndarray]:
        """{epoch: mean beat period per acquisition}, the period covariate used to
        residualize each metric in the beat-period-controlled grid rows."""
        return {
            epoch: np.array(
                [a["beat_period_mean"] for a in acqs_by_epoch[epoch]], dtype=float
            )
            for epoch in EPOCH_ORDER
        }

    def build_grid(
        self,
        dataset_name: str,
        vessel: str,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """Sec. V.B robustness sweep for one vessel. Runs Statistics.epoch_group_tests
        over every applicable combination of analysis choices -- metric x SVD method
        (joint/per_beat, where relevant) x beat aggregation (median/mean) x beat
        period control (native/residualized) -- producing one ``grid_row`` each.

        Returns (grid_rows, verdict_rows): the per-metric ``verdict_rows`` summarise
        whether the flicker effect survived every combination (all pooled p < 0.05)
        with a consistent Cliff's-delta sign across combinations."""
        periods = self.epoch_beat_periods(acqs_by_epoch)
        grid_rows: list[dict] = []
        verdict_source: dict[str, list[dict]] = {}

        def run_combo(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> dict:
            """Run epoch_group_tests for one grid cell, residualizing against beat
            period first when ``regressed`` is set."""
            values = self.epoch_dots(acqs_by_epoch, metric, svd_method, stat)
            if regressed:
                values = self.residualize_against_beat_period(values, periods)
            return Statistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Compute one grid cell and append its flattened row to grid_rows (and
            to verdict_source, grouped by metric for the verdict summary)."""
            tests = run_combo(metric, svd_method, stat, regressed)
            row = {
                "dataset": dataset_name,
                "vessel": vessel,
                "metric": metric,
                "svd_method": svd_method if svd_method is not None else "n/a",
                "beat_aggregation": stat,
                "beat_period_control": "residualized" if regressed else "native",
                "kruskal_wallis_p": tests["kruskal_wallis_p"],
                "pooled_baseline_vs_flicker_p": tests["pooled_baseline_vs_flicker_p"],
                "cliffs_delta_flicker_vs_pooled_baseline": tests[
                    "pooled_baseline_vs_flicker_delta"
                ],
                "n_B1": tests["n"]["baseline1"],
                "n_Flicker": tests["n"]["flicker"],
                "n_B2": tests["n"]["baseline2"],
            }
            grid_rows.append(row)
            verdict_source.setdefault(metric, []).append(row)

        for metric in self.METRICS_SVD:
            for svd_method in ("joint", "per_beat"):
                for stat in ("median", "mean"):
                    for regressed in (False, True):
                        add_row(metric, svd_method, stat, regressed)

        for stat in ("median", "mean"):
            for regressed in (False, True):
                add_row("TPR", None, stat, regressed)
                add_row("mpr", None, stat, regressed)

        for regressed in (False, True):
            add_row("alpha", None, "n/a", regressed)
            add_row("G1", None, "n/a", regressed)
            add_row("mpr_prime", None, "n/a", regressed)

        verdict_rows = []
        for metric, rows in verdict_source.items():
            ps = [r["pooled_baseline_vs_flicker_p"] for r in rows]
            deltas = [r["cliffs_delta_flicker_vs_pooled_baseline"] for r in rows]
            n_significant = int(sum(1 for p in ps if np.isfinite(p) and p < 0.05))
            signed = [d for d in deltas if np.isfinite(d)]
            consistent_direction = bool(signed) and (
                all(d > 0 for d in signed) or all(d < 0 for d in signed)
            )
            all_significant = n_significant == len(rows) and len(rows) > 0
            finite_ps = [p for p in ps if np.isfinite(p)]
            verdict_rows.append(
                {
                    "dataset": dataset_name,
                    "vessel": vessel,
                    "metric": metric,
                    "n_combinations": len(rows),
                    "n_significant_p_lt_0.05": n_significant,
                    "consistent_direction": consistent_direction,
                    "retained_all_combinations_significant_and_consistent": bool(
                        all_significant and consistent_direction
                    ),
                    "max_pooled_p": max(finite_ps) if finite_ps else np.nan,
                    "min_pooled_p": min(finite_ps) if finite_ps else np.nan,
                }
            )

        return grid_rows, verdict_rows


# Shared Confounds instance used by the orchestrator (and free-function aliases).
_CONFOUNDS = Confounds(_LR)

# Module-level aliases for callers that import the free functions by name.
residualize_against_beat_period = Confounds.residualize_against_beat_period
build_confound_grid = _CONFOUNDS.build_grid
CONFOUND_METRICS_SVD = Confounds.METRICS_SVD
TABLE_METRICS = Statistics.TABLE_METRICS


# =====================================================================
# ORCHESTRATION -- called by LowRankConfoundStatisticsPostprocess.run
# (below), which just hands us
# context.input_h5_paths/context.input_path/context.output_dir. Point
# --data at the shared parent of every dataset root (e.g. the folder
# containing 251219_Flicker_BOM_Final/, 260803_Flicker_AE_Final/, ...) to
# get every dataset's acquisitions classified and combined in one run; a
# single dataset root also works (see classify_dataset_and_epoch).
# =====================================================================


def run_confound_statistics(
    input_h5_paths: Iterable[Path],
    input_root: Path,
    output_dir: Path,
) -> tuple[str, list[Path]]:
    """Groups input_h5_paths into datasets/epochs (group_by_dataset),
    computes every acquisition's endpoints, writes per-acquisition
    metrics-only .h5 files plus the points/beats/tables/confound_control
    CSVs (per-dataset and combined) under
    output_dir/lowrank_confound_statistics/, and returns
    (summary, generated_paths)."""
    input_root = Path(input_root)
    root_dir = Path(output_dir) / "lowrank_confound_statistics"
    for sub in ("points", "beats", "tables", "confound_control", "endpoint_h5"):
        (root_dir / sub).mkdir(parents=True, exist_ok=True)

    records_by_dataset, skipped = group_by_dataset(input_h5_paths, input_root)
    if not records_by_dataset:
        raise ValueError(
            "No acquisitions could be classified into a dataset/epoch under "
            f"{input_root} (checked for each subfolder's manip.txt and for "
            "baseline1/flicker/baseline2-named subfolders)."
        )

    generated_paths: list[Path] = []
    all_points: list[pd.DataFrame] = []
    all_beats: list[pd.DataFrame] = []
    all_tables: list[pd.DataFrame] = []
    all_grid: list[pd.DataFrame] = []
    all_verdicts: list[pd.DataFrame] = []
    dataset_summaries: list[str] = []

    def _write_csv(df: pd.DataFrame, subdir: str, filename: str) -> Path:
        path = root_dir / subdir / filename
        df.to_csv(path, index=False)
        generated_paths.append(path)
        return path

    for name, records in records_by_dataset.items():
        per_vessel = collect_dataset(
            name, records, h5_out_dir=root_dir / "endpoint_h5" / name
        )

        dataset_points: list[pd.DataFrame] = []
        dataset_beats: list[pd.DataFrame] = []
        dataset_tables: list[pd.DataFrame] = []
        dataset_grid: list[pd.DataFrame] = []
        dataset_verdicts: list[pd.DataFrame] = []
        vessel_summaries: list[str] = []

        for vessel, (acqs_by_epoch, points_rows, beat_rows) in per_vessel.items():
            n_acq = sum(len(acqs_by_epoch[e]) for e in EPOCH_ORDER)
            if n_acq == 0:
                continue
            vessel_summaries.append(
                f"{vessel}={n_acq} (B1={len(acqs_by_epoch['baseline1'])}, "
                f"F={len(acqs_by_epoch['flicker'])}, B2={len(acqs_by_epoch['baseline2'])})"
            )

            points_df = pd.DataFrame(points_rows)
            dataset_points.append(points_df)

            dataset_beats.append(pd.DataFrame(beat_rows))

            dataset_tables.append(
                Statistics.build_endpoint_table(name, vessel, points_df)
            )

            grid_rows, verdict_rows = _CONFOUNDS.build_grid(
                name, vessel, acqs_by_epoch
            )
            dataset_grid.append(pd.DataFrame(grid_rows))
            dataset_verdicts.append(pd.DataFrame(verdict_rows))

        if not dataset_points:
            continue
        dataset_summaries.append(f"{name}: {', '.join(vessel_summaries)}")

        points_df = pd.concat(dataset_points, ignore_index=True)
        _write_csv(points_df, "points", f"{name}_points.csv")
        all_points.append(points_df)

        beats_df = pd.concat(dataset_beats, ignore_index=True)
        _write_csv(beats_df, "beats", f"{name}_beats.csv")
        all_beats.append(beats_df)

        table_df = pd.concat(dataset_tables, ignore_index=True)
        _write_csv(table_df, "tables", f"{name}_endpoint_table.csv")
        all_tables.append(table_df)

        grid_df = pd.concat(dataset_grid, ignore_index=True)
        _write_csv(grid_df, "confound_control", f"{name}_confound_grid.csv")
        all_grid.append(grid_df)

        verdict_df = pd.concat(dataset_verdicts, ignore_index=True)
        _write_csv(verdict_df, "confound_control", f"{name}_confound_verdict.csv")
        all_verdicts.append(verdict_df)

    if not all_points:
        raise ValueError(
            "Every classified dataset had zero valid acquisitions; nothing to write."
        )

    combined_points = pd.concat(all_points, ignore_index=True)
    _write_csv(combined_points, "points", "combined_points.csv")
    _write_csv(pd.concat(all_beats, ignore_index=True), "beats", "combined_beats.csv")
    _write_csv(
        pd.concat(all_tables, ignore_index=True), "tables", "combined_endpoint_table.csv"
    )
    _write_csv(
        pd.concat(all_grid, ignore_index=True),
        "confound_control",
        "combined_confound_grid.csv",
    )
    _write_csv(
        pd.concat(all_verdicts, ignore_index=True),
        "confound_control",
        "combined_confound_verdict.csv",
    )

    skipped_note = (
        f"; {len(skipped)} file(s) not classified into any dataset/epoch"
        if skipped
        else ""
    )
    summary = (
        f"Low-rank confound-controlled statistics: {len(dataset_summaries)} dataset(s), "
        f"{len(combined_points)} acquisition-vessel row(s) total (artery+vein combined) "
        f"({'; '.join(dataset_summaries)}){skipped_note}."
    )
    return summary, generated_paths


@registerPostprocess(
    name="lowrank confound-controlled statistics",
    description=(
        "Sec. V.A/V.B statistics tables (Kruskal-Wallis, Holm-adjusted "
        "pairwise Mann-Whitney, Cliff's delta) and the confound-control grid "
        "for the low-rank flicker endpoints, computed directly from raw "
        "acquisitions."
    ),
    required_deps=["scipy>=1.10", "pandas>=2.1"],
    visibility="hidden",
)
class LowRankConfoundStatisticsPostprocess(BatchPostprocess):
    """Registered postprocess wrapper: adapts a PostprocessContext to
    run_confound_statistics and returns its summary/outputs as a
    PostprocessResult. Registered as hidden (opt-in) since it recomputes the
    low-rank endpoints from raw acquisitions rather than reusing pipeline
    outputs."""

    def run(self, context: PostprocessContext) -> PostprocessResult:
        """Validate that raw acquisition inputs are present, run the statistics
        pipeline, and wrap the generated CSV paths in a PostprocessResult."""
        if not context.input_h5_paths:
            raise ValueError(
                "No input acquisition files are available for postprocessing."
            )

        summary, generated_paths = run_confound_statistics(
            input_h5_paths=context.input_h5_paths,
            input_root=context.input_path,
            output_dir=context.output_dir,
        )
        return PostprocessResult(
            summary=summary,
            generated_paths=[str(path) for path in generated_paths],
        )

#!/usr/bin/env python
"""
Generate all artifacts used by main_final.tex and main (30).tex, the
ARTICLE__Sienna flicker-provocation manuscripts (not part of the AngioEye
package itself -- this script lives under src/scripts/ alongside AngioEye's
other maintenance/reproduction scripts because it depends directly on
AngioEye's LowRankWaveformDecomposition pipeline).

This is the single entry point for the current SVD-only manuscript. It replaces
the previous multi-script workflow:

  1. plot_mode2_extension_figures.py
  2. plot_part1_lowrank_endpoint_figures.py
  3. plot_section_iv_vi_all_datasets.py (Fig. 3 waveform decomposition, Fig. 5
     dimensionality/spectrum panels)
  4. explore_rho2_residual_waveforms.py (Fig. 6 residual-energy panel)
1
The core native pipeline (native SVD endpoints A1, rho1, A2, rho2, TPR, MPR,
rho1, rho2, effective rank, participation ratio, alpha, mode energy
fractions; beat-period and mu context; Fig. 3) recomputes directly from raw
HDF5 acquisitions via AngioEye for OSS L, GOA, and OSS R -- including the
Fig. 5 dimensionality/spectrum panels and Fig. 3's TPR column, which
previously depended on a cache written by plot_section_iv_vi_all_datasets.py
(now retired; see prepare_all_points_df). Only the Fig. 6 residual-energy
curves (not part of main_final.tex) remain a richer per-acquisition quantity
computed by its own specialized script; this script only loads and plots the
CSV cache explore_rho2_residual_waveforms.py writes, and rerunning that
script first is only needed if that figure is regenerated.
Fig. 6 is the only figure that also pulls in OSS R, as a side condition (see
Sec. II of the article).

Outputs:
  figs/main_final_endpoint_points.csv
  figs/main_final_endpoint_stats.csv
  figs/section2_beat_period_primary_by_dataset.{eps,png}
  figs/section2_mu_primary_by_dataset.{eps,png}
  figs/section2_waveform_decomposition_by_dataset.{eps,png}   (Fig. 3)
  figs/section3_nonsvd_endpoints_primary_by_dataset.{eps,png}  (Part I: TPR, mu,
      MPR numerator/denominator, beat period)
  figs/section4_lowrank_endpoints_primary_by_dataset.{eps,png}  (Fig. 4)
  figs/section6_C_dimensionality_by_dataset.{eps,png}   (Fig. 5, top)
  figs/section6_C_spectrum_by_dataset.{eps,png}          (Fig. 5, bottom)
  figs/rho2_residual_energy_by_dataset.{eps,png}         (Fig. 6)

Usage:
  python3 generate_lowrank_endpoint_figures.py
  python3 generate_lowrank_endpoint_figures.py --recompute-from-hdf5
  python3 generate_lowrank_endpoint_figures.py --compile-tex
  python3 generate_lowrank_endpoint_figures.py --acquisition <h5_path>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
(HERE / ".mplconfig").mkdir(exist_ok=True)
(HERE / ".cache").mkdir(exist_ok=True)

# This script lives under AngioEye/src/scripts, but its outputs (figs/,
# main_final.tex, main (30).tex) belong to the article, not the AngioEye
# package -- so the article's own directory is tracked separately from HERE
# (which is only used above/below for this script's own scratch caches and
# its own log/output-listing paths).
ARTICLE_DIR = Path("/Users/admin/Desktop/langevin-internship/flicker-detection/measurements/ARTICLE__Sienna")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FIGS_DIR = ARTICLE_DIR / "figs"
SOURCE_POINTS_CSV = FIGS_DIR / "mode2_extension_points.csv"
PRIMARY_POINTS_CSV = FIGS_DIR / "main_final_endpoint_points.csv"
PRIMARY_POINTS_CSV_VEIN = FIGS_DIR / "main_final_endpoint_points_vein.csv"
STATS_CSV = FIGS_DIR / "main_final_endpoint_stats.csv"
FULL_TABLE_STATS_CSV = FIGS_DIR / "main_final_full_table_stats.csv"
TABLE1_HOLM_STATS_CSV = FIGS_DIR / "main_final_table1_holm_stats.csv"
RHO2_CURVES_CSV = FIGS_DIR / "rho2_residual_waveform_curves.csv"
RHO2_SUMMARY_CSV = FIGS_DIR / "rho2_residual_waveform_summary.csv"
MAIN_TEX = ARTICLE_DIR / "main_final.tex"

ANGIOEYE_SRC = Path("/Users/admin/Developer/AngioEye/src")
if str(ANGIOEYE_SRC) not in sys.path:
    sys.path.insert(0, str(ANGIOEYE_SRC))

# AngioEye-native ports of flicker_stats.epoch_group_tests/holm_adjust
# (identical formulas -- Holm-Bonferroni step-down, Kruskal-Wallis +
# pairwise Mann-Whitney/Cliff's delta), so this script depends on
# calculations living inside the AngioEye repo rather than reaching out to
# the external flicker-detection/flicker_stats.py module.
from postprocess.utils.lowrank_confound_statistics import (  # noqa: E402
    epoch_group_tests,
    holm_adjust,
)


# =====================================================================
# DATASET / SCHEMA CONFIGURATION -- all three datasets (OSS L, GOA, OSS R)
# now share the newer EyeFlow-nested HDF5 layout. Segment/beat-period
# paths are resolved per dataset via DATASET_SCHEMA + SEGMENT_PATHS_BY_
# SCHEMA, rather than as single module-level constants (matching
# LowRankWaveformDecomposition's own EyeFlow-first, Artery/Vein-fallback
# schema resolution).
# =====================================================================

SEGMENT_PATHS_BY_SCHEMA = {
    "old": {
        "raw": "Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value",
        "bandlimited": "Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value",
        "beat_period": "Artery/VelocityPerBeat/beatPeriodSeconds/value",
    },
    "new": {
        "raw": "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/Raw/value",
        "bandlimited": "EyeFlow/Processing/VelocityPerBeat/Artery/Segments/BandLimited/value",
        "beat_period": "EyeFlow/Processing/VelocityPerBeat/BeatPeriodSeconds/value",
        # Vein segment paths mirror the Artery ones one-for-one (same schema,
        # same (n_t, n_beats, n_branches, n_radii) shape convention, just a
        # different vessel subtree); beat_period is shared across vessels.
        "vein_raw": "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/Raw/value",
        "vein_bandlimited": "EyeFlow/Processing/VelocityPerBeat/Vein/Segments/BandLimited/value",
    },
}
VESSELS = ("artery", "vein")
DATASET_SCHEMA = {"OSS_L": "new", "OSS_R": "new", "GOA": "new"}

# 2026-08-06 reprocessed/finalized acquisitions (all four datasets unified onto
# the "new" EyeFlow/Processing HDF5 schema); supersedes the older mixed-schema
# "*_AE_Unorganised" directories under DATA_ROOT.
DATA_ROOT = Path("/Users/admin/Desktop/langevin-internship/flicker-detection/data_august_6")

OSS_L_RE = re.compile(r"OSS_(\d+)_AE")
OSS_R_RE = re.compile(r"OSS_R_(\d+)_AE")
GOA_RE = re.compile(r"GOA_(\d+)_AE")

# (directory, acquisition-index regex); manip.txt in each directory gives the
# per-acquisition B1/F/B2 split (see parse_manip_epochs). OSS R now ships its
# own non-blank manip.txt (all 28 acquisitions), unlike the older Unorganised
# tree where it was blank and had to borrow OSS L's boundaries.
DATASET_SOURCES = {
    "OSS_L": (DATA_ROOT / "260720_Flicker_OSS_L_Final", OSS_L_RE),
    "OSS_R": (DATA_ROOT / "260720_Flicker_OSS_R_Final", OSS_R_RE),
    "GOA": (DATA_ROOT / "260803_Flicker_AE_Final", GOA_RE),
}

# =====================================================================
# PLOT STYLE, LABELS, AND ENDPOINT COLUMN DEFINITIONS -- colors, panel
# sizing, epoch/dataset display labels, and the (column, title) tuples
# every figure-grid function below iterates over. plt.rcParams.update at
# the end of this block is global matplotlib style, applied once at
# import time.
# =====================================================================

FLICKER_BLUE = "#add8e6"
BLACK = "black"
DARK_GRAY = "#555555"
RNG = np.random.default_rng(0)

# Shared square panel size (inches) for the epoch dot-whisker grids: Fig. 3
# (plot_nonsvd_endpoint_grid), Fig. 4 (plot_lowrank_endpoint_grid), and Fig. 5
# (plot_dimensionality via plot_grid). Keeping one constant ensures every
# individual panel across all three figures is the same square size.
ENDPOINT_PANEL_SIZE = 2.5
BOOTSTRAP_RNG = np.random.default_rng(123)

EPOCHS = (("baseline1", "B1"), ("flicker", "Flicker"), ("baseline2", "B2"))
EPOCH_ORDER = ("B1", "Flicker", "B2")
EPOCH_LABELS = {"B1": "B1", "Flicker": "F", "B2": "B2"}

DATASET_ORDER = ("OSS_L", "GOA", "OSS_R")
DATASET_LABELS = {"OSS_L": "OSS L", "GOA": "GOA", "OSS_R": "OSS R"}

LOWRANK_ENDPOINTS = (
    ("A1", r"$A_1$", r"Mode-1 amplitude"),
    ("A2", r"$A_2$", r"Mode-2 amplitude"),
    ("R1", r"$R_1$", r"Residual level after mode 1"),
    ("R2", r"$R_2$", r"Residual level after modes 1--2"),
)

CONTEXT_ENDPOINTS = (
    ("beat_period", "Beat period", "section2_beat_period_primary_by_dataset"),
    ("mu", r"Baseline level $\mu$", "section2_mu_primary_by_dataset"),
)

# Fig. 1 replacement: one representative acquisition per dataset, showing the
# fRMS frequency map alongside the global raw artery velocity waveform.
FRMS_MAP_PATH = "EyeFlow/Processing/FrequencyMaps/fRMS_avg/value"
GLOBAL_ARTERY_RAW_PATH = "EyeFlow/Processing/Velocity/global/Artery/Raw/value"
GLOBAL_VEIN_RAW_PATH = "EyeFlow/Processing/Velocity/global/Vein/Raw/value"
FREQ_VELOCITY_ACQUISITIONS = (
    ("OSS_L", 10),
    ("OSS_R", 13),
    ("GOA", 6),
)

# Non-SVD endpoints (Part I): mu is the same quantity defined in Sec.
# "Baseline level and total pulsatile RMS"; MPR is the paired-local
# mean-to-pulsatile ratio (Eq. eq:mpr_def, AngioEye's acq["mpr"]); $R_0$
# reuses the TPR column data under its R_0 notation (TPR and R_0 are the
# same quantity, so the separate raw-TPR column has been dropped).
NONSVD_ENDPOINTS = (
    ("beat_period", "Beat period"),
    ("mu", r"Baseline level $\mu$"),
    ("TPR", r"$R_0$"),
    ("MPR", "MPR"),
)

DIMENSIONALITY_ENDPOINTS = (
    ("rho1", r"$\rho_1$"),
    ("rho2", r"$\rho_2$"),
    ("effective_rank", r"$R_\mathrm{eff}$"),
    ("participation_ratio", "PR"),
    ("alpha", r"$\alpha$"),
    ("G1", r"$G_1$"),
)

# Branch-labeled panels (Fig. 5/6) use "B"/"F"/"B" ticks, matching the
# original plot_section_iv_vi_all_datasets.py / explore_rho2_residual_waveforms.py
# style, which is distinct from the "B1"/"F"/"B2" ticks used by draw_epoch_plot.
BRANCH_ORDER = ("B1", "F", "B2")
BRANCH_LABELS = {"B1": "B", "F": "F", "B2": "B"}

DATASETS_WITH_OSS_R = (
    ("OSS_L", "OSS L"),
    ("OSS_R", "OSS R"),
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": BLACK,
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }
)


# =====================================================================
# GENERIC HELPERS -- shared by every collection and plotting function
# below; none of these are specific to one figure or table.
# =====================================================================


def acquisition_index(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.search(path.name)
    if match:
        return int(match.group(1))
    match = pattern.search(str(path))
    return int(match.group(1)) if match else 10**9


def apply_axes(ax, tick_size: int = 10, label_size: int = 11) -> None:
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=tick_size, width=1.2, length=5, colors=BLACK)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)


def save_all(fig, basename: Path) -> None:
    basename.parent.mkdir(exist_ok=True)
    # EPS has no native alpha/transparency support; artists using alpha< 1
    # (shaded fill_between/axhspan bands) are marked rasterized=True at their
    # call sites so they still alpha-blend correctly, embedded as a raster
    # sub-image inside the otherwise-vector EPS. dpi=220 keeps that raster
    # (and any imshow content) sharp, matching the PNG's resolution.
    fig.savefig(basename.with_suffix(".eps"), dpi=220, bbox_inches="tight")
    fig.savefig(basename.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# DATASET DISCOVERY -- manip.txt in each dataset directory gives the
# per-acquisition B1/F/B2 split (see parse_manip_epochs); DATASETS below
# is the (key, label, path_fn) triple every downstream collection/plot
# function iterates over.
# =====================================================================


def parse_manip_epochs(manip_path: Path) -> dict[int, str]:
    """Parse a manip.txt like '0-6 B1\\n7-18 F\\n19-22, 24-27 B2' into {acquisition_index: epoch}."""
    label_to_epoch = {"B1": "B1", "B": "B1", "F": "Flicker", "B2": "B2"}
    mapping: dict[int, str] = {}
    for line in manip_path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        range_part, label = line.rsplit(" ", 1)
        epoch = label_to_epoch[label.strip()]
        for chunk in range_part.split(","):
            chunk = chunk.strip()
            if "-" in chunk:
                lo, hi = chunk.split("-")
                mapping.update({idx: epoch for idx in range(int(lo), int(hi) + 1)})
            else:
                mapping[int(chunk)] = epoch
    return mapping


def dataset_paths(dataset_key: str) -> dict[str, list[Path]]:
    """Flat acquisition directory -> {baseline1/flicker/baseline2: [h5 paths]}, per manip.txt.

    Falls back to OSS L's B1/F/B2 index boundaries only if a dataset's own
    manip.txt is missing or blank (historically true for OSS R under the
    older Unorganised tree; the 2026-08-06 Final tree ships its own).
    """
    root, pattern = DATASET_SOURCES[dataset_key]
    manip_path = root / "manip.txt"
    if dataset_key == "OSS_R" and not (manip_path.exists() and manip_path.read_text().strip()):
        manip_path = DATASET_SOURCES["OSS_L"][0] / "manip.txt"
    epoch_by_index = parse_manip_epochs(manip_path)

    epoch_to_folder = {"B1": "baseline1", "Flicker": "flicker", "B2": "baseline2"}
    grouped: dict[str, list[Path]] = {"baseline1": [], "flicker": [], "baseline2": []}
    for h5_path in sorted(root.glob("*.h5"), key=lambda p: acquisition_index(p, pattern)):
        epoch = epoch_by_index.get(acquisition_index(h5_path, pattern))
        if epoch is None:
            continue
        grouped[epoch_to_folder[epoch]].append(h5_path)
    return grouped


def oss_left_paths() -> dict[str, list[Path]]:
    return dataset_paths("OSS_L")


def oss_right_paths() -> dict[str, list[Path]]:
    return dataset_paths("OSS_R")


def goa_paths() -> dict[str, list[Path]]:
    return dataset_paths("GOA")


DATASETS = (
    ("OSS_L", "OSS L", oss_left_paths),
    ("GOA", "GOA", goa_paths),
    ("OSS_R", "OSS R", oss_right_paths),
)


def make_lowrank_helper():
    try:
        from pipelines.lowrank_waveform_decomposition import LowRankWaveformDecomposition
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import AngioEye's LowRankWaveformDecomposition. "
            f"Expected source tree at {ANGIOEYE_SRC}."
        ) from exc
    helper = LowRankWaveformDecomposition()
    # Paired arterial/venous figures and tables need both compartments;
    # the pipeline defaults to artery-only.
    helper.process_veins = True
    return helper


# =====================================================================
# PER-ACQUISITION COLLECTION -- the native SVD pipeline (A1, rho1, A2,
# rho2, TPR, MPR, dimensionality/spectrum quantities) recomputes directly
# from raw HDF5 acquisitions via AngioEye's LowRankWaveformDecomposition,
# for OSS L, GOA, and OSS R, for either vessel.
# =====================================================================


def collect_dataset(
    dataset_key: str,
    dataset_label: str,
    paths_by_epoch: dict[str, list[Path]],
    pattern: re.Pattern[str],
    vessel: str = "artery",
) -> pd.DataFrame:
    """Matches lowrank_confound_statistics.py's collect_dataset in spirit:
    schema resolution, the joint SVD, and per-vessel endpoint retrieval
    all go through LowRankWaveformDecomposition's public
    compute_acquisition_endpoints(h5_path) rather than reaching into its
    protected _normalize_T/_ensure_segment_shape/_compute_representation
    (contrast with waveform_summary below, which genuinely needs the raw
    per-mode reconstruction matrices compute_acquisition_endpoints does
    not -- and structurally should not -- expose)."""
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    helper = make_lowrank_helper()
    rows: list[dict[str, float | int | str]] = []

    for epoch_folder, epoch_label in EPOCHS:
        for sequence, h5_path in enumerate(paths_by_epoch[epoch_folder]):
            data = helper.compute_acquisition_endpoints(h5_path)
            if data is None:
                continue
            vessel_data = data.get(vessel)
            if vessel_data is None:
                continue

            acq = vessel_data["acq"]
            energy_fraction = np.asarray(vessel_data.get("energy_fraction", []), dtype=float)
            valid_fraction_per_beat = np.asarray(vessel_data["valid_fraction_per_beat"], dtype=float)
            mu_bkr = np.asarray(vessel_data["mu"], dtype=float)
            mu_per_beat = np.nanmedian(mu_bkr, axis=(1, 2))

            row = {
                "dataset": dataset_key,
                "dataset_label": dataset_label,
                "acquisition": acquisition_index(h5_path, pattern),
                "sequence": sequence,
                "epoch": epoch_label,
                "A1": float(acq.get("A1", np.nan)),
                "A1_sd": float(acq.get("sigma_A1_beat", np.nan)),
                "rho1": float(acq.get("rho1", np.nan)),
                "rho1_sd": float(acq.get("sigma_rho1_beat", np.nan)),
                "R1": float(acq.get("R1", np.nan)),
                "R1_sd": float(acq.get("sigma_R1_beat", np.nan)),
                "A2": float(acq.get("A2", np.nan)),
                "A2_sd": float(acq.get("sigma_A2_beat", np.nan)),
                "rho2": float(acq.get("rho2", np.nan)),
                "rho2_sd": float(acq.get("sigma_rho2_beat", np.nan)),
                "R2": float(acq.get("R2", np.nan)),
                "R2_sd": float(acq.get("sigma_R2_beat", np.nan)),
                "beat_period": vessel_data["beat_period_mean"],
                "beat_period_sd": vessel_data["beat_period_sd"],
                "mu": float(np.nanmedian(mu_bkr)),
                "mu_sd": float(np.nanstd(mu_per_beat, ddof=1)),
                "mu_abs": float(np.nanmedian(np.abs(mu_bkr))),
                "TPR": float(acq.get("TPR", np.nan)),
                "TPR_sd": float(acq.get("sigma_TPR_beat", np.nan)),
                "MPR": float(acq.get("MPR", np.nan)),
                "MPR_sd": float(acq.get("sigma_mpr_beat", np.nan)),
                "Reff": float(acq.get("effective_rank", np.nan)),
                "PR": float(acq.get("participation_ratio", np.nan)),
                "alpha": float(acq.get("alpha", np.nan)),
                "G1": float(acq.get("G1", np.nan)),
                "valid_fraction": float(np.nanmean(valid_fraction_per_beat)),
                "valid_fraction_sd": float(np.nanstd(valid_fraction_per_beat, ddof=1)),
                "n_valid_columns": int(vessel_data["n_valid_columns"]),
                "n_total_columns": int(vessel_data["n_total_columns"]),
            }
            for m in range(1, 13):
                row[f"mode{m}"] = float(energy_fraction[m - 1]) if energy_fraction.size >= m else np.nan
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid acquisitions collected for {dataset_label} ({vessel})")
    return pd.DataFrame(rows).sort_values(["epoch", "acquisition"]).reset_index(drop=True)


def collect_all_points(vessel: str = "artery") -> pd.DataFrame:
    all_rows = [
        collect_dataset(dataset_key, dataset_label, path_fn(), DATASET_SOURCES[dataset_key][1], vessel=vessel)
        for dataset_key, dataset_label, path_fn in DATASETS
    ]
    df = pd.concat(all_rows, ignore_index=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["epoch"] = pd.Categorical(df["epoch"], categories=EPOCH_ORDER, ordered=True)
    df = df.sort_values(["dataset", "epoch", "acquisition"]).reset_index(drop=True)
    out_csv = PRIMARY_POINTS_CSV if vessel == "artery" else PRIMARY_POINTS_CSV_VEIN
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    if vessel == "artery":
        df.to_csv(SOURCE_POINTS_CSV, index=False)
        print(f"wrote {SOURCE_POINTS_CSV}")
    return df


def load_points_from_csv(vessel: str = "artery") -> pd.DataFrame:
    if vessel == "artery":
        source = PRIMARY_POINTS_CSV if PRIMARY_POINTS_CSV.exists() else SOURCE_POINTS_CSV
    else:
        source = PRIMARY_POINTS_CSV_VEIN
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; rerun with --recompute-from-hdf5 first.")
    df = pd.read_csv(source)
    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["epoch"] = pd.Categorical(df["epoch"], categories=EPOCH_ORDER, ordered=True)
    df = df.sort_values(["dataset", "epoch", "acquisition"]).reset_index(drop=True)
    if vessel == "artery":
        df.to_csv(PRIMARY_POINTS_CSV, index=False)
        print(f"wrote {PRIMARY_POINTS_CSV}")
    return df


# =====================================================================
# SIBLING-SCRIPT CSV CACHES -- the rho2 residual-energy curves (not part
# of main_final.tex) are still computed by explore_rho2_residual_waveforms.py;
# this script only loads and plots the CSV cache that script writes.
# Rerun it first if the underlying analysis changes. Dimensionality/
# spectrum (Reff, PR, alpha, mode-wise energy fractions) and TPR used to
# depend on a similar cache from plot_section_iv_vi_all_datasets.py, but
# that script no longer exists and its cache had drifted from AngioEye's
# current endpoint set (no "alpha" column, a stale "rho3" column) -- see
# prepare_all_points_df, which now derives the same quantities natively
# from collect_dataset's own per-acquisition rows instead.
# =====================================================================


def prepare_all_points_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adapts collect_all_points(...)'s output (native "epoch" B1/Flicker/B2
    labels, AngioEye short names Reff/PR) to the "branch" B1/F/B2 +
    effective_rank/participation_ratio schema that plot_grid / plot_spectrum
    / plot_spectrum_cumulative expect. Used for both artery and vein so
    every quantity in those figures (rho1, rho2, effective_rank,
    participation_ratio, alpha, TPR, mode1..mode12) comes straight out of
    LowRankWaveformDecomposition's own endpoint dict, with no separate
    cache or hand-reconstructed formula involved."""
    out = df.copy()
    out["branch"] = out["epoch"].map({"B1": "B1", "Flicker": "F", "B2": "B2"})
    out = out.rename(columns={"Reff": "effective_rank", "PR": "participation_ratio"})
    return out


def load_rho2_residual_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RHO2_CURVES_CSV.exists() or not RHO2_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RHO2_CURVES_CSV} or {RHO2_SUMMARY_CSV}. "
            "Run explore_rho2_residual_waveforms.py first."
        )
    return pd.read_csv(RHO2_CURVES_CSV), pd.read_csv(RHO2_SUMMARY_CSV)


# =====================================================================
# STATISTICS PRIMITIVES -- shared by the table writers and every
# annotated figure panel below (draw_epoch_plot, epoch_dot_whisker).
# =====================================================================


def cliffs_delta(flicker: np.ndarray, baseline: np.ndarray) -> float:
    flicker = flicker[np.isfinite(flicker)]
    baseline = baseline[np.isfinite(baseline)]
    if flicker.size == 0 or baseline.size == 0:
        return float("nan")
    gt = np.sum(flicker[:, None] > baseline[None, :])
    lt = np.sum(flicker[:, None] < baseline[None, :])
    return float((gt - lt) / (flicker.size * baseline.size))


def pooled_test(
    df: pd.DataFrame,
    metric: str,
    branch_col: str = "epoch",
    flicker_val: str = "Flicker",
    baseline_vals: tuple[str, str] = ("B1", "B2"),
) -> tuple[float, float]:
    baseline = df.loc[df[branch_col].isin(baseline_vals), metric].to_numpy(dtype=float)
    flicker = df.loc[df[branch_col] == flicker_val, metric].to_numpy(dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    flicker = flicker[np.isfinite(flicker)]
    if baseline.size < 2 or flicker.size < 2:
        return float("nan"), float("nan")
    p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
    return p, cliffs_delta(flicker, baseline)


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def format_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return r"$\delta$=n/a"
    return rf"$\delta$={delta:+.2f}"


# =====================================================================
# TABLE / STATS CSV WRITERS -- Sec. V.A/Table I style breakdowns, all
# computed from collect_dataset's own per-acquisition rows via the
# project's canonical epoch_group_tests/holm_adjust.
# =====================================================================


def write_endpoint_stats(df: pd.DataFrame) -> None:
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset]
        for metric, label, _ in LOWRANK_ENDPOINTS:
            p, delta = pooled_test(sub, metric)
            baseline = sub.loc[sub["epoch"].isin(["B1", "B2"]), metric]
            flicker = sub.loc[sub["epoch"] == "Flicker", metric]
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "metric": metric,
                    "metric_label": label,
                    "baseline_median": float(np.nanmedian(baseline)),
                    "flicker_median": float(np.nanmedian(flicker)),
                    "pooled_baseline_vs_flicker_p": p,
                    "cliffs_delta_flicker_vs_pooled_baseline": delta,
                    "n_B1": int(np.sum(sub["epoch"] == "B1")),
                    "n_Flicker": int(np.sum(sub["epoch"] == "Flicker")),
                    "n_B2": int(np.sum(sub["epoch"] == "B2")),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(STATS_CSV, index=False)
    print(f"wrote {STATS_CSV}")


TABLE_METRICS = (
    ("A1", r"$A_1$"),
    ("A2", r"$A_2$"),
    ("R0", r"$R_0"), 
    ("R1", r"$R_1$"),
    ("R2", r"$R_2$"),
    ("rho1", r"$\rho_1$"),
    ("rho2", r"$\rho_2$"),
    ("TPR", "TPR"),
    ("MPR", "MPR"),
    ("Reff", r"$R_{\mathrm{eff}}$"),
    ("PR", "PR"),
    ("alpha", r"$\alpha$"),
)


def write_full_endpoint_table_stats(df: pd.DataFrame) -> None:
    """Table I-style full breakdown (KW p, 3 Holm-adjusted pairwise p's,
    3 Cliff's deltas, per-epoch median+-SD+N) for every dataset and all 11
    endpoints (A1, A2, R1, R2, rho1, rho2, TPR, MPR, Reff, PR, alpha), all
    computed natively by AngioEye's LowRankWaveformDecomposition in
    collect_dataset (one acq dict per acquisition), via the project's
    canonical epoch_group_tests."""
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASET_ORDER:
        for metric, label in TABLE_METRICS:
            sub = df[df["dataset"] == dataset]
            epoch_values = {
                "baseline1": sub.loc[sub["epoch"] == "B1", metric].dropna().to_numpy(dtype=float),
                "flicker": sub.loc[sub["epoch"] == "Flicker", metric].dropna().to_numpy(dtype=float),
                "baseline2": sub.loc[sub["epoch"] == "B2", metric].dropna().to_numpy(dtype=float),
            }
            report = epoch_group_tests(epoch_values)
            pairwise_by_pair = {p["pair"]: p for p in report["pairwise"]}
            b1_f = pairwise_by_pair["baseline1 vs flicker"]
            f_b2 = pairwise_by_pair["flicker vs baseline2"]
            b1_b2 = pairwise_by_pair["baseline1 vs baseline2"]

            def med_sd_n(vals: np.ndarray) -> tuple[float, float, int]:
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return np.nan, np.nan, 0
                med = float(np.median(vals))
                sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                return med, sd, int(vals.size)

            b1_med, b1_sd, n_b1 = med_sd_n(epoch_values["baseline1"])
            f_med, f_sd, n_f = med_sd_n(epoch_values["flicker"])
            b2_med, b2_sd, n_b2 = med_sd_n(epoch_values["baseline2"])

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "metric": metric,
                    "metric_label": label,
                    "B1_median": b1_med,
                    "B1_sd": b1_sd,
                    "n_B1": n_b1,
                    "Flicker_median": f_med,
                    "Flicker_sd": f_sd,
                    "n_Flicker": n_f,
                    "B2_median": b2_med,
                    "B2_sd": b2_sd,
                    "n_B2": n_b2,
                    "kruskal_wallis_p": report["kruskal_wallis_p"],
                    "B1_vs_F_p_holm": b1_f["p_holm"],
                    "F_vs_B2_p_holm": f_b2["p_holm"],
                    "B1_vs_B2_p_holm": b1_b2["p_holm"],
                    "pooled_baseline_vs_flicker_p": report["pooled_baseline_vs_flicker_p"],
                    "cliffs_delta_F_vs_B1": -b1_f["cliffs_delta"],
                    "cliffs_delta_F_vs_B2": f_b2["cliffs_delta"],
                    "cliffs_delta_F_vs_pooled_baseline": report["pooled_baseline_vs_flicker_delta"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(FULL_TABLE_STATS_CSV, index=False)
    print(f"wrote {FULL_TABLE_STATS_CSV}")


def write_table1_holm_pooled_stats() -> None:
    """Table I: per-dataset, Holm-adjusted pooled baseline-vs-Flicker
    significance across all 11 endpoints (A1, A2, R1, R2, rho1, rho2, TPR,
    MPR, Reff, PR, alpha). Holm adjustment is applied across that dataset's
    full 11-endpoint family, controlling the family-wise error rate across
    the whole endpoint panel rather than within a single metric's three
    pairwise contrasts. Reuses the raw pooled p-values already written by
    write_full_endpoint_table_stats, so it must run after that."""
    full = pd.read_csv(FULL_TABLE_STATS_CSV)
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASET_ORDER:
        sub = full[full["dataset"] == dataset].set_index("metric")
        raw_p = [float(sub.loc[metric, "pooled_baseline_vs_flicker_p"]) for metric, _label in TABLE_METRICS]
        holm_p = holm_adjust(raw_p)
        row: dict[str, float | int | str] = {"dataset": dataset, "dataset_label": DATASET_LABELS[dataset]}
        for (metric, _label), p in zip(TABLE_METRICS, holm_p):
            row[metric] = p
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE1_HOLM_STATS_CSV, index=False)
    print(f"wrote {TABLE1_HOLM_STATS_CSV}")


# Endpoint order / labels for the pooled "one-eye" endpoint table
# (tab:arterial_one_eye_endpoints layout): R_0 is TPR under its R_0 notation.
# The eleven rows below form the Holm family; the exploratory G_1 dominant-mode
# gap is appended afterward and excluded from the Holm adjustment.
ONE_EYE_TABLE_METRICS = (
    ("A1", r"$A_1$"),
    ("A2", r"$A_2$"),
    ("TPR", r"$R_0$"),
    ("R1", r"$R_1$"),
    ("R2", r"$R_2$"),
    ("rho1", r"$\rho_1$"),
    ("rho2", r"$\rho_2$"),
    ("MPR", "MPR"),
    ("Reff", r"$R_{\mathrm{eff}}$"),
    ("PR", "PR"),
    ("alpha", r"$\alpha$"),
)
ONE_EYE_EXPLORATORY_METRICS = (("G1", r"$G_1$"),)


def _fmt_median_iqr(vals: np.ndarray) -> str:
    """`$median\\,[IQR]$` (3 significant figures), IQR being the Q3-Q1 width."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "--"
    med = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    return f"${med:.3g}\\,[{iqr:.3g}]$"


def _fmt_p_tex(p: float) -> str:
    """`$m.mm\\times10^{-k}$` for p < 0.01, else `$0.ddd$`; `--` for NaN."""
    if not np.isfinite(p):
        return "--"
    if p >= 1e-2:
        return f"${p:.3f}$"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10 ** exp
    return f"${mant:.2f}\\times10^{{{exp}}}$"


def _fmt_delta_tex(d: float) -> str:
    return "--" if not np.isfinite(d) else f"${d:+.2f}$"


def build_pooled_endpoint_latex_table(
    df: pd.DataFrame, dataset_key: str, dataset_label: str, vessel: str = "artery"
) -> str:
    """Build the pooled baseline-vs-Flicker "one-eye" endpoint LaTeX table
    (tab:arterial_one_eye_endpoints layout) for one dataset and vessel: median
    [IQR] per epoch group (baseline = B1+B2), Cliff's delta (Flicker vs pooled
    baseline), the raw two-sided Mann-Whitney p, and the Holm-adjusted p_H over
    the eleven-endpoint family. The exploratory $G_1$ row is appended with the
    same descriptive statistics and raw p, but $p_{\rm H}$ is left as ``--``
    because $G_1$ is excluded from the eleven-endpoint Holm family."""
    sub = df[df["dataset"] == dataset_key]
    raw_ps: list[float] = []
    cells: list[tuple[str, str, str, str, str]] = []
    for col, label in ONE_EYE_TABLE_METRICS:
        base = sub.loc[sub["epoch"].isin(["B1", "B2"]), col].to_numpy(dtype=float)
        fl = sub.loc[sub["epoch"] == "Flicker", col].to_numpy(dtype=float)
        p, delta = pooled_test(sub, col)
        raw_ps.append(p)
        cells.append(
            (label, _fmt_median_iqr(base), _fmt_median_iqr(fl), _fmt_delta_tex(delta), _fmt_p_tex(p))
        )
    holm_ps = holm_adjust(raw_ps)

    exploratory_cells: list[tuple[str, str, str, str, str]] = []
    for col, label in ONE_EYE_EXPLORATORY_METRICS:
        base = sub.loc[sub["epoch"].isin(["B1", "B2"]), col].to_numpy(dtype=float)
        fl = sub.loc[sub["epoch"] == "Flicker", col].to_numpy(dtype=float)
        p, delta = pooled_test(sub, col)
        exploratory_cells.append(
            (label, _fmt_median_iqr(base), _fmt_median_iqr(fl), _fmt_delta_tex(delta), _fmt_p_tex(p))
        )

    vessel_word = "arterial" if vessel == "artery" else "venous"
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Pooled baseline-versus-flicker comparison of the eleven predefined "
        + vessel_word
        + r" endpoints and the exploratory dominant-mode gap $G_1$ for "
        + dataset_label
        + r". Baseline combines Baseline~1 and Baseline~2. Values report the median [IQR], "
        r"Cliff's $\delta$, raw Mann--Whitney $p$-value, and Holm-adjusted $p_{\rm H}$. The "
        r"reported adjusted values correspond to the eleven-endpoint family; $G_1$ is "
        r"reported with its raw $p$ but excluded from that Holm adjustment ($p_{\rm H}$ "
        r"marked --). These are within-eye acquisition-level results.}",
        r"\label{tab:%s_one_eye_endpoints_%s}" % (vessel_word, dataset_key.lower()),
        r"\centering",
        r"\begingroup",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccccc}",
        r"Endpoint",
        r"& Baseline median [IQR]",
        r"& Flicker median [IQR]",
        r"& Cliff's $\delta$",
        r"& Raw $p$",
        r"& $p_{\rm H}$ \\",
        r"\hline",
    ]
    for (label, base_c, fl_c, d_c, p_c), ph in zip(cells, holm_ps):
        lines.append(f"{label} & {base_c} & {fl_c} & {d_c} & {p_c} & {_fmt_p_tex(ph)} \\\\")
    for label, base_c, fl_c, d_c, p_c in exploratory_cells:
        lines.append(f"{label} & {base_c} & {fl_c} & {d_c} & {p_c} & -- \\\\")
    lines += [
        r"\end{tabular}",
        r"\end{ruledtabular}",
        r"\endgroup",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def write_pooled_endpoint_latex_table(
    df: pd.DataFrame, dataset_key: str, dataset_label: str, vessel: str = "artery"
) -> str:
    """Write build_pooled_endpoint_latex_table's output to a .tex file under
    FIGS_DIR and return the LaTeX string."""
    tex = build_pooled_endpoint_latex_table(df, dataset_key, dataset_label, vessel=vessel)
    out = FIGS_DIR / f"pooled_endpoint_table_{vessel}_{dataset_key}.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex + "\n")
    print(f"wrote {out}")
    return tex


def write_paired_table1_holm_stats(
    dataset_key: str, dataset_label: str, df_artery: pd.DataFrame, df_vein: pd.DataFrame
) -> pd.DataFrame:
    """Table I of the paired arterial/venous draft: Holm-adjusted pooled
    baseline-vs-Flicker significance across all 11 endpoints, computed
    separately (not pooled) for the artery and vein panels of one dataset."""

    def raw_pooled_p(df: pd.DataFrame) -> list[float]:
        raw_p = []
        for metric, _label in TABLE_METRICS:
            epoch_values = {
                "baseline1": df.loc[df["epoch"] == "B1", metric].dropna().to_numpy(dtype=float),
                "flicker": df.loc[df["epoch"] == "Flicker", metric].dropna().to_numpy(dtype=float),
                "baseline2": df.loc[df["epoch"] == "B2", metric].dropna().to_numpy(dtype=float),
            }
            report = epoch_group_tests(epoch_values)
            raw_p.append(report["pooled_baseline_vs_flicker_p"])
        return raw_p

    holm_p_artery = holm_adjust(raw_pooled_p(df_artery))
    holm_p_vein = holm_adjust(raw_pooled_p(df_vein))

    rows = [
        {"metric": metric, "metric_label": label, "p_holm_artery": pa, "p_holm_vein": pv}
        for (metric, label), pa, pv in zip(TABLE_METRICS, holm_p_artery, holm_p_vein)
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "dataset_label", dataset_label)
    out_csv = FIGS_DIR / f"paired_table1_holm_stats_{dataset_key}.csv"
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    return out


# =====================================================================
# EPOCH DOT-WHISKER FIGURES -- Fig. 2 context panels, Fig. 3 non-SVD
# endpoints, Fig. 4 low-rank endpoints, and their paired-draft
# equivalents. draw_epoch_plot is the shared per-panel primitive (B1/F/B2
# ticks); everything below builds a grid of panels around it.
# =====================================================================


def draw_epoch_plot(
    ax,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    annotate: bool,
) -> None:
    positions = {epoch: idx for idx, epoch in enumerate(EPOCH_ORDER)}
    ax.axvspan(0.5, 1.5, color=FLICKER_BLUE, zorder=0)
    ax.axvline(0.5, color=BLACK, linestyle=":", linewidth=1.5, zorder=1)
    ax.axvline(1.5, color=BLACK, linestyle=":", linewidth=1.5, zorder=1)

    for epoch in EPOCH_ORDER:
        vals = df.loc[df["epoch"] == epoch, metric].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        x0 = positions[epoch]
        jitter = (RNG.random(vals.size) - 0.5) * 0.16
        med = float(np.nanmedian(vals))
        sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.errorbar(
            [x0],
            [med],
            yerr=[sd],
            fmt="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markeredgewidth=1.6,
            ecolor=BLACK,
            elinewidth=1.7,
            capsize=4,
            capthick=1.5,
            zorder=4,
        )
        # Drawn after (on top of) the hollow median marker so individual dots
        # are never hidden behind it when N is small and jitter lands them
        # close to the marker's center (as happens for OSS R's n=3 groups).
        ax.scatter(x0 + jitter, vals, s=24, color=BLACK, edgecolors="none", zorder=5)

    ax.set_xticks([positions[e] for e in EPOCH_ORDER])
    ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER], fontsize=10)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(ylabel, fontsize=11, color=BLACK)
    apply_axes(ax, tick_size=10, label_size=10)

    all_vals = df[metric].to_numpy(dtype=float)
    if annotate and np.isfinite(all_vals).any():
        p, delta = pooled_test(df, metric)
        y_min = float(np.nanmin(all_vals))
        y_max = float(np.nanmax(all_vals))
        pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        ax.text(
            0.03,
            0.97,
            f"{format_p(p)}\n{format_delta(delta)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=BLACK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
        ax.set_ylim(y_min - pad, y_max + 2.2 * pad)


def plot_lowrank_endpoint_grid(df: pd.DataFrame, vessel: str = "artery") -> None:
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    columns = [(metric, short_title) for metric, short_title, _ in LOWRANK_ENDPOINTS]
    n_rows = len(DATASET_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        len(columns),
        figsize=(ENDPOINT_PANEL_SIZE * len(columns), ENDPOINT_PANEL_SIZE * n_rows),
        squeeze=False,
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, (metric, long_title) in enumerate(columns):
            sub = df[df["dataset"] == dataset]
            ax = axes[row_idx, col_idx]
            draw_epoch_plot(ax, sub, metric, "", annotate=True)
            ax.set_box_aspect(1)
            if row_idx == 0:
                ax.set_title(long_title, fontsize=11, color=BLACK)
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    DATASET_LABELS[dataset],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=BLACK,
                )
    fig.tight_layout(w_pad=1.1, h_pad=1.1)
    suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = FIGS_DIR / f"section4_lowrank_endpoints_primary{suffix}_by_dataset"
    save_all(fig, basename)
    print(f"wrote {basename}.eps/.png")


def plot_nonsvd_endpoint_grid(df: pd.DataFrame, vessel: str = "artery") -> None:
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    columns = list(NONSVD_ENDPOINTS)
    n_rows = len(DATASET_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        len(columns),
        figsize=(ENDPOINT_PANEL_SIZE * len(columns), ENDPOINT_PANEL_SIZE * n_rows),
        squeeze=False,
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, (metric, long_title) in enumerate(columns):
            sub = df[df["dataset"] == dataset]
            ax = axes[row_idx, col_idx]
            draw_epoch_plot(ax, sub, metric, "", annotate=True)
            ax.set_box_aspect(1)
            if row_idx == 0:
                ax.set_title(long_title, fontsize=11, color=BLACK)
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    DATASET_LABELS[dataset],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=BLACK,
                )
    fig.tight_layout(w_pad=1.1, h_pad=1.1)
    suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = FIGS_DIR / f"section3_nonsvd_endpoints_primary{suffix}_by_dataset"
    save_all(fig, basename)
    print(f"wrote {basename}.eps/.png")


# Column sets for the paired arterial/venous draft's Figs. 4-6. All keys
# already exist verbatim on collect_dataset()'s own per-acquisition rows, so
# no renaming/adapter step is needed the way the old sibling-script-derived
# all_points_df required for Figs. 5/6 of the main (30).tex pipeline.
# Non-SVD endpoints reuse NONSVD_ENDPOINTS's 4-column layout (beat period,
# mu, R0, MPR) directly -- TPR and R0 are the same quantity, so there is no
# separate MPR-numerator/denominator split here.
PAIRED_NONSVD_ENDPOINTS = NONSVD_ENDPOINTS
PAIRED_LOWRANK_ENDPOINTS = tuple((metric, title) for metric, title, _long in LOWRANK_ENDPOINTS)
PAIRED_DIMENSIONALITY_ENDPOINTS = (
    ("rho1", r"$\rho_1$"),
    ("rho2", r"$\rho_2$"),
    ("Reff", r"$R_{\mathrm{eff}}$"),
    ("PR", "PR"),
    ("alpha", r"$\alpha$"),
    ("G1", r"$G_1$"),
)


def plot_paired_epoch_grid(
    df_artery: pd.DataFrame, df_vein: pd.DataFrame, columns: tuple[tuple[str, str], ...], basename: str
) -> None:
    """Shared grid core for Figs. 4-6 of the paired arterial/venous draft:
    two rows (artery, vein), one column per endpoint, epoch dot-whisker
    panels for one acquisition's parent dataset."""
    n_cols = len(columns)
    fig, axes = plt.subplots(
        len(VESSELS), n_cols, figsize=(ENDPOINT_PANEL_SIZE * n_cols, ENDPOINT_PANEL_SIZE * len(VESSELS)), squeeze=False
    )
    for row_idx, (row_label, df) in enumerate(zip(("Artery", "Vein"), (df_artery, df_vein))):
        for col_idx, (metric, title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            draw_epoch_plot(ax, df, metric, "", annotate=True)
            ax.set_box_aspect(1)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, color=BLACK)
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=BLACK,
                )
    fig.tight_layout(w_pad=1.1, h_pad=1.1)
    save_all(fig, FIGS_DIR / basename)
    print(f"wrote {FIGS_DIR / basename}.eps/.png")


def plot_context_figures(df: pd.DataFrame) -> None:
    for metric, title, basename in CONTEXT_ENDPOINTS:
        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(3.4 * len(DATASET_ORDER), 3.0), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, dataset in zip(axes, DATASET_ORDER):
            sub = df[df["dataset"] == dataset]
            draw_epoch_plot(ax, sub, metric, title, annotate=False)
            ax.set_title(DATASET_LABELS[dataset], fontsize=12, color=BLACK)
        fig.suptitle(title, fontsize=13, color=BLACK, y=1.02)
        fig.tight_layout()
        out = FIGS_DIR / basename
        save_all(fig, out)
        print(f"wrote {out}.eps/.png")


def find_acquisition_path(dataset_key: str, acquisition: int) -> Path:
    root, pattern = DATASET_SOURCES[dataset_key]
    for h5_path in sorted(root.glob("*.h5")):
        if acquisition_index(h5_path, pattern) == acquisition:
            return h5_path
    raise FileNotFoundError(f"No acquisition {acquisition} found for {dataset_key} under {root}")


def resolve_acquisition(h5_path: Path | str) -> tuple[str, str, int]:
    """AngioEye-style single-h5-path entry point (mirrors
    LowRankWaveformDecomposition.compute_acquisition_endpoints(h5_path)):
    given a path to one acquisition's raw .h5 file, resolves which known
    dataset it belongs to and returns (dataset_key, dataset_label,
    acquisition_index)."""
    h5_path = Path(h5_path).resolve()
    for dataset_key, (root, pattern) in DATASET_SOURCES.items():
        if h5_path.parent == root.resolve():
            return dataset_key, DATASET_LABELS[dataset_key], acquisition_index(h5_path, pattern)
    raise ValueError(f"{h5_path} does not belong to any known dataset root: {sorted(DATASET_SOURCES)}")


def plot_frequency_velocity_overview(vessel: str = "artery") -> None:
    """Fig. 1: fRMS_avg frequency map (left) and global raw velocity
    waveform (right) for one representative acquisition per dataset. The
    fRMS map is not vessel-specific (one whole-retina frequency map per
    acquisition), so only the waveform panel differs between vessels."""
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    global_path = GLOBAL_ARTERY_RAW_PATH if vessel == "artery" else GLOBAL_VEIN_RAW_PATH
    n_rows = len(FREQ_VELOCITY_ACQUISITIONS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8.0, 3.4 * n_rows), squeeze=False)
    for row_idx, (dataset_key, acquisition) in enumerate(FREQ_VELOCITY_ACQUISITIONS):
        h5_path = find_acquisition_path(dataset_key, acquisition)
        with h5py.File(h5_path, "r") as h5:
            frms = np.asarray(h5[FRMS_MAP_PATH], dtype=float)
            velocity = np.asarray(h5[global_path], dtype=float).reshape(-1)

        ax_map = axes[row_idx, 0]
        ax_map.imshow(frms, cmap="gray", aspect="equal")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_ylabel(
            f"{DATASET_LABELS[dataset_key]}\n(acq. {acquisition})", fontsize=11, color=BLACK
        )
        if row_idx == 0:
            ax_map.set_title(r"$f_{\mathrm{RMS}}$ map", fontsize=12, color=BLACK)

        ax_wave = axes[row_idx, 1]
        ax_wave.plot(np.arange(velocity.size), velocity, color=BLACK, linewidth=1.0)
        ax_wave.axhline(0, color=DARK_GRAY, linewidth=0.6, linestyle=":")
        ax_wave.set_xlabel("Sample index", fontsize=10, color=BLACK)
        ax_wave.set_ylabel("Velocity (mm/s)", fontsize=10, color=BLACK)
        apply_axes(ax_wave, tick_size=9, label_size=10)
        if row_idx == 0:
            ax_wave.set_title(f"Global raw {vessel} velocity", fontsize=12, color=BLACK)

    fig.tight_layout()
    suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = FIGS_DIR / f"section2_frequency_velocity_overview{suffix}_by_dataset"
    save_all(fig, basename)
    print(f"wrote {basename}.eps/.png")


def plot_paired_frequency_velocity_overview(h5_path: Path, dataset_key: str, dataset_label: str, acquisition: int) -> None:
    """Fig. 1 of the paired arterial/venous draft: fRMS map (vessel-agnostic,
    shown identically in both rows) and global raw velocity waveform for ONE
    acquisition, arteries on top and veins on the bottom."""
    with h5py.File(h5_path, "r") as h5:
        frms = np.asarray(h5[FRMS_MAP_PATH], dtype=float)
        velocity = {
            "artery": np.asarray(h5[GLOBAL_ARTERY_RAW_PATH], dtype=float).reshape(-1),
            "vein": np.asarray(h5[GLOBAL_VEIN_RAW_PATH], dtype=float).reshape(-1),
        }

    fig, axes = plt.subplots(len(VESSELS), 2, figsize=(8.0, 3.4 * len(VESSELS)), squeeze=False)
    for row_idx, vessel in enumerate(VESSELS):
        ax_map = axes[row_idx, 0]
        ax_map.imshow(frms, cmap="gray", aspect="equal")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_ylabel(f"{vessel.capitalize()}\n{dataset_label} (acq. {acquisition})", fontsize=11, color=BLACK)
        if row_idx == 0:
            ax_map.set_title(r"$f_{\mathrm{RMS}}$ map", fontsize=12, color=BLACK)

        ax_wave = axes[row_idx, 1]
        v = velocity[vessel]
        ax_wave.plot(np.arange(v.size), v, color=BLACK, linewidth=1.0)
        ax_wave.axhline(0, color=DARK_GRAY, linewidth=0.6, linestyle=":")
        ax_wave.set_xlabel("Sample index", fontsize=10, color=BLACK)
        ax_wave.set_ylabel("Velocity (mm/s)", fontsize=10, color=BLACK)
        apply_axes(ax_wave, tick_size=9, label_size=10)
        if row_idx == 0:
            ax_wave.set_title("Global raw velocity", fontsize=12, color=BLACK)

    fig.tight_layout()
    basename = FIGS_DIR / f"paired_fig1_frequency_velocity_overview_{dataset_key}_{acquisition:02d}"
    save_all(fig, basename)
    print(f"wrote {basename}.eps/.png")


# =====================================================================
# WAVEFORM-DECOMPOSITION FIGURES -- Fig. 3 (one row per dataset, fixed
# vessel) and paired Fig. 2 (one row per vessel, fixed acquisition).
# Reaches into LowRankWaveformDecomposition's protected
# _normalize_T/_ensure_segment_shape/_compute_representation directly
# (rather than compute_acquisition_endpoints's public summary dict)
# because it needs the raw per-mode reconstruction (U_panel,
# score_panel_flat), which compute_acquisition_endpoints does not expose.
# =====================================================================


def segment_signal_path(signal: str, schema: str, vessel: str = "artery") -> str:
    if signal not in ("raw", "bandlimited"):
        raise ValueError("signal must be one of: raw, bandlimited")
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    key = signal if vessel == "artery" else f"vein_{signal}"
    return SEGMENT_PATHS_BY_SCHEMA[schema][key]


def median_iqr_curve(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(matrix, axis=1)
    q25 = np.nanpercentile(matrix, 25, axis=1)
    q75 = np.nanpercentile(matrix, 75, axis=1)
    return med, q25, q75


def first_baseline_path(path_fn) -> Path | None:
    paths = path_fn().get("baseline1", [])
    return paths[0] if paths else None


def waveform_summary(
    h5_path: Path, helper, dataset_key: str, signal: str = "bandlimited", vessel: str = "artery"
) -> dict[str, np.ndarray]:
    schema = DATASET_SCHEMA[dataset_key]
    signal_path = segment_signal_path(signal, schema, vessel=vessel)
    beat_period_path = SEGMENT_PATHS_BY_SCHEMA[schema]["beat_period"]
    with h5py.File(h5_path, "r") as h5:
        v_block = np.asarray(h5[signal_path], dtype=float)
        T = helper._normalize_T(np.asarray(h5[beat_period_path], dtype=float))
    v_block = helper._ensure_segment_shape(v_block, T)
    rep = helper._compute_representation(v_block=v_block, T=T)
    valid = rep["valid_column_mask"]
    mu = np.nanmean(v_block, axis=0, keepdims=True)
    x_full = v_block - mu

    valid_flat = valid.reshape(-1)
    v_cols = v_block.reshape(v_block.shape[0], -1)[:, valid_flat]
    x_cols = x_full.reshape(x_full.shape[0], -1)[:, valid_flat]
    mu_cols = mu.reshape(1, -1)[:, valid_flat]

    if rep.get("svd_available", False) and rep.get("n_modes_panel", 0) >= 1:
        recon_cols = np.outer(rep["U_panel"][:, 0], rep["score_panel_flat"][0, :])
    else:
        recon_cols = np.full_like(x_cols, np.nan)

    if rep.get("svd_available", False) and rep.get("n_modes_panel", 0) >= 2:
        recon2_cols = np.outer(rep["U_panel"][:, 1], rep["score_panel_flat"][1, :])
    else:
        recon2_cols = np.full_like(x_cols, np.nan)

    return {
        "t": np.linspace(0, 1, v_block.shape[0], endpoint=False),
        "v": v_cols,
        "mu": mu_cols,
        "x": x_cols,
        "a1u1": recon_cols,
        "a2u2": recon2_cols,
    }


def waveform_row_first_two_ylim(summary: dict[str, np.ndarray]) -> tuple[float, float]:
    bounds: list[float] = []
    for key in ("v", "mu"):
        if key == "mu":
            mu_vals = summary[key].reshape(-1)
            med = float(np.nanmedian(mu_vals))
            sd = float(np.nanstd(mu_vals, ddof=1))
            bounds += [med - sd, med + sd]
        else:
            _med, q25, q75 = median_iqr_curve(summary[key])
            bounds += [float(np.nanmin(q25)), float(np.nanmax(q75))]
    lo, hi = min(bounds), max(bounds)
    lo, hi = min(lo, 0.0), max(hi, 0.0)  # keep the zero line inside the visible range
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    return (lo - pad, hi + pad)


def waveform_row_last_three_ylim(summary: dict[str, np.ndarray]) -> tuple[float, float]:
    extents: list[float] = []
    for key in ("x", "a1u1", "a2u2"):
        _med, q25, q75 = median_iqr_curve(summary[key])
        extents.append(max(abs(float(np.nanmin(q25))), abs(float(np.nanmax(q75)))))
    extent = max(extents) if extents else 1.0
    extent = extent if extent > 0 else 1.0
    extent *= 1.12
    return (-extent, extent)


def label_axis_extrema(ax, lo: float, hi: float, n: int = 5) -> None:
    """Force tick marks (and labels) near the axis min and max, plus
    evenly spaced values between them, rounded to the nearest multiple
    of 5, so every panel's extremes are always readable rather than
    left inside unlabeled padding."""
    ticks = np.round(np.linspace(lo, hi, n) / 5.0) * 5.0
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{val:.0f}"))


def draw_waveform_decomposition_grid(rows: list[tuple[str, dict[str, np.ndarray] | None]], basename: str) -> None:
    """Shared grid core for waveform-decomposition figures: one row per
    (row_label, summary) pair, one column per quantity (v, mu, x, a1u1,
    a2u2). Used both for the one-row-per-dataset, fixed-vessel layout
    (plot_waveform_decomposition) and the one-row-per-vessel,
    fixed-acquisition paired layout (plot_paired_waveform_decomposition).

    The mu panel is a scalar per (b,k,r), not a function of t, so it is drawn
    as a flat median line with a shaded band of +/- 1 std across b,k,r,
    rather than the median/IQR-over-t bands used for the other panels.

    Within each row, columns 1-2 (v, mu) share one common y-axis, and
    columns 3-5 (x, a1u1, a2u2) share one common zero-centered y-axis, each
    sized from that row's own data. Axis scale is therefore comparable
    within a row but is allowed to vary across rows.
    """
    panel_defs = [
        ("v", "Beat-aligned\nvelocity"),
        ("mu", r"Baseline level" "\n" r"$\mu$"),
        ("x", r"Baseline-removed" "\n" r"$v-\mu$"),
        ("a1u1", r"Mode-1 recon." "\n" r"$a_1u_1$"),
        ("a2u2", r"Mode-2 recon." "\n" r"$a_2u_2$"),
    ]
    adaptive_zero_centered_cols = {"x", "a1u1", "a2u2"}

    # Figures are drawn much larger than their printed size (this one is
    # placed at 0.96\linewidth inside a figure*, a ~2.6x downscale), so the
    # in-figure font sizes below are chosen to land just under the
    # surrounding body text size once the PDF is scaled down on the page.
    # Titles are wrapped onto two lines since at this font size the full
    # descriptive titles no longer fit on one line within a single column.
    # (Midpoint between the original small sizes and the first, too-large pass.)
    font_title = 15
    font_label = 14
    font_row = 16
    tick_size = 12

    n_rows = len(rows)
    n_cols = len(panel_defs)
    # Square panels: match the endpoint grids (ENDPOINT_PANEL_SIZE + set_box_aspect(1)).
    # Extra width reserves room for the outboard Artery/Vein + unit ylabels.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ENDPOINT_PANEL_SIZE * n_cols + 1.0, ENDPOINT_PANEL_SIZE * n_rows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_2d(axes)
    for row_idx, (row_label, summary) in enumerate(rows):
        if summary is None:
            continue
        t = summary["t"]
        row_ylim_12 = waveform_row_first_two_ylim(summary)
        row_ylim_345 = waveform_row_last_three_ylim(summary)
        for col_idx, (key, title) in enumerate(panel_defs):
            ax = axes[row_idx, col_idx]
            if key == "mu":
                mu_vals = summary[key].reshape(-1)
                med = float(np.nanmedian(mu_vals))
                sd = float(np.nanstd(mu_vals, ddof=1))
                ax.axhline(med, color=BLACK, linewidth=1.8)
                ax.axhspan(med - sd, med + sd, color=BLACK, alpha=0.12, linewidth=0, rasterized=True)
            else:
                med, q25, q75 = median_iqr_curve(summary[key])
                ax.plot(t, med, color=BLACK, linewidth=1.8)
                ax.fill_between(t, q25, q75, color=BLACK, alpha=0.12, linewidth=0, rasterized=True)

            ax.axhline(0, color=BLACK, linewidth=1.0, linestyle=":", zorder=1)
            y_lo, y_hi = row_ylim_345 if key in adaptive_zero_centered_cols else row_ylim_12
            ax.set_ylim(y_lo, y_hi)
            label_axis_extrema(ax, y_lo, y_hi)

            if row_idx == 0:
                ax.set_title(title, fontsize=font_title, color=BLACK)
            if col_idx == 0:
                # Single-line ylabel (compartment + units) avoids the
                # Artery/(mm/s) line-stacking collision that a two-line
                # rotated ylabel produces. Position pinned in axes coords
                # because labelpad alone is unreliable with set_box_aspect(1).
                ax.set_ylabel(f"{row_label} (mm/s)", fontsize=font_label, color=BLACK)
                ax.yaxis.set_label_coords(-0.40, 0.5)
            apply_axes(ax, tick_size=tick_size, label_size=font_label)
            ax.set_box_aspect(1)
    fig.supxlabel("Fraction of cardiac cycle", fontsize=font_label, color=BLACK)
    fig.tight_layout(w_pad=1.1, h_pad=1.1, rect=(0.08, 0.05, 1.0, 1.0))
    save_all(fig, FIGS_DIR / basename)
    print(f"wrote {FIGS_DIR / basename}.eps/.png")


def plot_waveform_decomposition(signal: str = "bandlimited", vessel: str = "artery") -> None:
    """Fig. 3: representative first Baseline-1 acquisition per dataset, one
    row per dataset (fixed vessel)."""
    if signal not in ("raw", "bandlimited"):
        raise ValueError("signal must be one of: raw, bandlimited")
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    helper = make_lowrank_helper()
    rows: list[tuple[str, dict[str, np.ndarray] | None]] = []
    for dataset_key, dataset_label, path_fn in DATASETS:
        path = first_baseline_path(path_fn)
        summary = waveform_summary(path, helper, dataset_key, signal=signal, vessel=vessel) if path is not None else None
        rows.append((dataset_label, summary))
    signal_suffix = "" if signal == "bandlimited" else f"_{signal}"
    vessel_suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = f"section2_waveform_decomposition{vessel_suffix}_by_dataset{signal_suffix}"
    draw_waveform_decomposition_grid(rows, basename)


def plot_paired_waveform_decomposition(
    h5_path: Path, dataset_key: str, dataset_label: str, acquisition: int, signal: str = "bandlimited"
) -> None:
    """Fig. 2 of the paired arterial/venous draft: waveform decomposition
    for ONE acquisition, arteries on top and veins on the bottom."""
    if signal not in ("raw", "bandlimited"):
        raise ValueError("signal must be one of: raw, bandlimited")
    helper = make_lowrank_helper()
    rows = [
        (vessel.capitalize(), waveform_summary(h5_path, helper, dataset_key, signal=signal, vessel=vessel))
        for vessel in VESSELS
    ]
    signal_suffix = "" if signal == "bandlimited" else f"_{signal}"
    basename = f"paired_fig2_waveform_decomposition_{dataset_key}_{acquisition:02d}{signal_suffix}"
    draw_waveform_decomposition_grid(rows, basename)


# =====================================================================
# DIMENSIONALITY / SPECTRUM FIGURES -- Fig. 5 (bottom row) and paired
# Fig. 3. Self-contained via AngioEye's own energy_fraction/Reff/PR/alpha
# endpoints (collect_dataset's mode1..mode12 columns, adapted by
# prepare_all_points_df) for both the main_final.tex artery panels and
# the paired arterial/venous draft -- no external CSV cache involved.
# =====================================================================


def epoch_dot_whisker(
    ax,
    df: pd.DataFrame,
    value_col: str,
    jitter: float = 0.09,
    tick_size: int = 16,
    annotate: bool = False,
) -> None:
    """Fig. 5/6 style: operates on the raw 'branch' column with B/F/B ticks."""
    positions = {branch: i for i, branch in enumerate(BRANCH_ORDER)}
    ax.axvspan(0.5, 1.5, color=FLICKER_BLUE, zorder=0)
    ax.axvline(0.5, color=BLACK, linestyle=":", linewidth=2.2, zorder=1)
    ax.axvline(1.5, color=BLACK, linestyle=":", linewidth=2.2, zorder=1)
    for branch in BRANCH_ORDER:
        vals = df.loc[df["branch"] == branch, value_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        x0 = positions[branch]
        xj = x0 + (RNG.random(vals.size) - 0.5) * 2 * jitter
        med = float(np.nanmedian(vals))
        sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.errorbar(
            [x0],
            [med],
            yerr=[sd],
            fmt="o",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markeredgewidth=1.8,
            ecolor=BLACK,
            elinewidth=2.0,
            capsize=5,
            capthick=1.8,
            zorder=4,
        )
        # Drawn after (on top of) the hollow median marker so individual dots
        # are never hidden behind it when N is small and jitter lands them
        # close to the marker's center (as happens for OSS R's n=3 groups).
        ax.scatter(xj, vals, s=34, color=BLACK, alpha=1.0, zorder=5, edgecolors="none")
    ax.set_xticks([positions[b] for b in BRANCH_ORDER])
    ax.set_xticklabels([BRANCH_LABELS[b] for b in BRANCH_ORDER], fontsize=tick_size)
    ax.set_xlim(-0.5, len(BRANCH_ORDER) - 0.5)
    apply_axes(ax, tick_size=tick_size)

    all_vals = df[value_col].to_numpy(dtype=float)
    if annotate and np.isfinite(all_vals).any():
        p, delta = pooled_test(df, value_col, branch_col="branch", flicker_val="F")
        y_min = float(np.nanmin(all_vals))
        y_max = float(np.nanmax(all_vals))
        pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        ax.text(
            0.03,
            0.97,
            f"{format_p(p)}\n{format_delta(delta)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=BLACK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
        ax.set_ylim(y_min - pad, y_max + 2.2 * pad)


def plot_grid(
    df: pd.DataFrame,
    panels: list[tuple[str, str]],
    basename: str,
    ncols: int,
    figsize_per_panel: tuple[float, float] = (ENDPOINT_PANEL_SIZE, ENDPOINT_PANEL_SIZE),
    annotate: bool = False,
) -> None:
    nrows = len(DATASETS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for row_idx, (dataset_key, dataset_label, _) in enumerate(DATASETS):
        sub = df[df["dataset"] == dataset_key]
        for col_idx, (metric, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            epoch_dot_whisker(ax, sub, metric, jitter=0.08, tick_size=11, annotate=annotate)
            ax.set_box_aspect(1)
            if row_idx == 0:
                ax.set_title(title, fontsize=12, color=BLACK)
            if col_idx == 0:
                ax.set_ylabel(dataset_label, fontsize=12, color=BLACK)
            else:
                ax.set_ylabel("")
    for ax in axes.flatten()[len(DATASETS) * len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_all(fig, FIGS_DIR / basename)
    print(f"wrote {FIGS_DIR / basename}.eps/.png")


def plot_dimensionality(df: pd.DataFrame, vessel: str = "artery") -> None:
    """Fig. 5, top row."""
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    suffix = "" if vessel == "artery" else f"_{vessel}"
    plot_grid(
        df,
        list(DIMENSIONALITY_ENDPOINTS),
        f"section6_C_dimensionality{suffix}_by_dataset",
        ncols=5,
        annotate=True,
    )


def plot_spectrum(df: pd.DataFrame, vessel: str = "artery") -> None:
    """Fig. 5, bottom row."""
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    mode_cols = [f"mode{i}" for i in range(1, 13) if f"mode{i}" in df.columns]
    modes = np.arange(1, len(mode_cols) + 1)
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.6 * n_datasets, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (dataset_key, dataset_label, _) in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset_key]
        for mask, color, style, marker, label in (
            (sub["branch"].isin(["B1", "B2"]), BLACK, "-", "o", "Baseline"),
            (sub["branch"] == "F", DARK_GRAY, "--", "s", "Flicker"),
        ):
            vals = sub.loc[mask, mode_cols].to_numpy(dtype=float)
            med = np.nanmedian(vals, axis=0)
            q25 = np.nanpercentile(vals, 25, axis=0)
            q75 = np.nanpercentile(vals, 75, axis=0)
            ax.plot(modes, med, color=color, linestyle=style, marker=marker, linewidth=2.0, markersize=4, label=label)
            ax.fill_between(modes, q25, q75, color=color, alpha=0.12, linewidth=0, rasterized=True)
        ax.set_title(dataset_label, fontsize=12, color=BLACK)
        ax.set_yscale("log")
        ax.set_xticks(modes)
        ax.set_xlabel("Mode")
        apply_axes(ax, tick_size=10, label_size=11)
    axes[0].set_ylabel("Variance fraction")
    axes[-1].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = f"section6_C_spectrum{suffix}_by_dataset"
    save_all(fig, FIGS_DIR / basename)
    print(f"wrote {FIGS_DIR / basename}.eps/.png")


def plot_spectrum_cumulative(df: pd.DataFrame, vessel: str = "artery") -> None:
    """Fig. 5 (bottom row) duplicate: cumulative variance fraction, log-scaled."""
    if vessel not in VESSELS:
        raise ValueError(f"vessel must be one of {VESSELS}, got {vessel!r}")
    mode_cols = [f"mode{i}" for i in range(1, 13) if f"mode{i}" in df.columns]
    modes = np.arange(1, len(mode_cols) + 1)
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.6 * n_datasets, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (dataset_key, dataset_label, _) in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset_key]
        for mask, color, style, marker, label in (
            (sub["branch"].isin(["B1", "B2"]), BLACK, "-", "o", "Baseline"),
            (sub["branch"] == "F", DARK_GRAY, "--", "s", "Flicker"),
        ):
            vals = sub.loc[mask, mode_cols].to_numpy(dtype=float)
            cum_vals = np.cumsum(vals, axis=1)
            med = np.nanmedian(cum_vals, axis=0)
            q25 = np.nanpercentile(cum_vals, 25, axis=0)
            q75 = np.nanpercentile(cum_vals, 75, axis=0)
            ax.plot(modes, med, color=color, linestyle=style, marker=marker, linewidth=2.0, markersize=4, label=label)
            ax.fill_between(modes, q25, q75, color=color, alpha=0.12, linewidth=0, rasterized=True)
        ax.set_title(dataset_label, fontsize=12, color=BLACK)
        ax.set_yscale("log")
        ax.set_xticks(modes)
        ax.set_xlabel("Mode")
        apply_axes(ax, tick_size=10, label_size=11)
    axes[0].set_ylabel("Cumulative variance fraction")
    axes[-1].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    suffix = "" if vessel == "artery" else f"_{vessel}"
    basename = f"section6_C_spectrum_cumulative{suffix}_by_dataset"
    save_all(fig, FIGS_DIR / basename)
    print(f"wrote {FIGS_DIR / basename}.eps/.png")


def plot_paired_spectrum(dataset_key: str, dataset_label: str, df_artery: pd.DataFrame, df_vein: pd.DataFrame) -> None:
    """Fig. 3 of the paired arterial/venous draft: mode-wise SVD energy
    fractions (left column) and their cumulative sum (right column) for one
    dataset, arteries on top and veins on the bottom. Each panel overlays the
    pooled-baseline and Flicker median +/- IQR curves. Uses collect_dataset()'s
    own "epoch" B1/Flicker/B2 labels directly (no branch-column remap needed)."""
    mode_cols = [f"mode{i}" for i in range(1, 13) if f"mode{i}" in df_artery.columns]
    modes = np.arange(1, len(mode_cols) + 1)

    series = (
        ("Baseline", lambda d: d["epoch"].isin(["B1", "B2"]), BLACK, "-", "o"),
        ("Flicker", lambda d: d["epoch"] == "Flicker", DARK_GRAY, "--", "s"),
    )
    # (column title, cumulative?) -- left: per-mode variance fraction,
    # right: cumulative sum of the variance fraction across modes.
    columns = (("Variance fraction", False), ("Cumulative variance fraction", True))

    fig, axes = plt.subplots(
        len(VESSELS),
        len(columns),
        figsize=(4.2 * len(columns), 3.2 * len(VESSELS)),
        sharex=True,
        squeeze=False,
    )
    for row_idx, (row_label, df) in enumerate(zip(("Artery", "Vein"), (df_artery, df_vein))):
        for col_idx, (col_title, cumulative) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            for label, mask_fn, color, style, marker in series:
                vals = df.loc[mask_fn(df), mode_cols].to_numpy(dtype=float)
                if cumulative:
                    vals = np.cumsum(vals, axis=1)
                med = np.nanmedian(vals, axis=0)
                q25 = np.nanpercentile(vals, 25, axis=0)
                q75 = np.nanpercentile(vals, 75, axis=0)
                ax.plot(modes, med, color=color, linestyle=style, marker=marker, linewidth=2.0, markersize=4, label=label)
                ax.fill_between(modes, q25, q75, color=color, alpha=0.12, linewidth=0, rasterized=True)
            ax.set_yscale("log")
            ax.set_xticks(modes)
            ax.set_xlabel("Mode")
            apply_axes(ax, tick_size=10, label_size=11)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=12, color=BLACK)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=12, color=BLACK)
    axes[0, -1].legend(frameon=False, fontsize=10)
    fig.suptitle(dataset_label, fontsize=13, color=BLACK)
    fig.tight_layout()
    basename = FIGS_DIR / f"paired_fig3_spectrum_{dataset_key}"
    save_all(fig, basename)
    print(f"wrote {basename}.eps/.png")


# =====================================================================
# RESIDUAL-ENERGY FIGURE -- Fig. 6, phase-resolved rho2 residual energy.
# Reads CSV caches written by explore_rho2_residual_waveforms.py (see
# load_rho2_residual_curves); not self-contained via AngioEye the way the
# other figures in this file are.
# =====================================================================


def bootstrap_contrast(curves: pd.DataFrame, metric: str, n_boot: int = 500) -> tuple[np.ndarray, np.ndarray]:
    phases = np.sort(curves["phase"].unique())
    acq_ids = curves[["acquisition", "epoch"]].drop_duplicates()
    base_ids = acq_ids.loc[acq_ids["epoch"].isin(["B1", "B2"]), "acquisition"].to_numpy()
    flicker_ids = acq_ids.loc[acq_ids["epoch"] == "Flicker", "acquisition"].to_numpy()
    pivot = curves.pivot_table(index="acquisition", columns="phase", values=metric, aggfunc="first").reindex(columns=phases)
    draws = np.empty((n_boot, phases.size), dtype=float)
    for idx in range(n_boot):
        b_sample = BOOTSTRAP_RNG.choice(base_ids, size=base_ids.size, replace=True)
        f_sample = BOOTSTRAP_RNG.choice(flicker_ids, size=flicker_ids.size, replace=True)
        b_curve = np.nanmedian(pivot.loc[b_sample].to_numpy(dtype=float), axis=0)
        f_curve = np.nanmedian(pivot.loc[f_sample].to_numpy(dtype=float), axis=0)
        draws[idx, :] = f_curve - b_curve
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def plot_energy(summary: pd.DataFrame, curves: pd.DataFrame) -> None:
    """Fig. 6: phase-resolved residual energy, OSS L/OSS R."""
    fig, axes = plt.subplots(2, len(DATASETS_WITH_OSS_R), figsize=(7.33, 6.0), sharex=True)
    for col, (dataset_key, dataset_label) in enumerate(DATASETS_WITH_OSS_R):
        sub = summary[summary["dataset"] == dataset_key].sort_values("phase")
        raw = curves[curves["dataset"] == dataset_key]
        phase = sub["phase"].to_numpy(dtype=float)

        ax = axes[0, col]
        ax.plot(phase, sub["energy_G2_baseline_median"], color=BLACK, linewidth=2.0, label="Baseline")
        ax.fill_between(
            phase,
            sub["energy_G2_baseline_q25"],
            sub["energy_G2_baseline_q75"],
            color=BLACK,
            alpha=0.12,
            linewidth=0,
            rasterized=True,
        )
        ax.plot(phase, sub["energy_G2_flicker_median"], color=DARK_GRAY, linestyle="--", linewidth=2.0, label="Flicker")
        ax.fill_between(
            phase,
            sub["energy_G2_flicker_q25"],
            sub["energy_G2_flicker_q75"],
            color=DARK_GRAY,
            alpha=0.12,
            linewidth=0,
            rasterized=True,
        )
        ax.set_title(dataset_label, fontsize=12, color=BLACK)
        if col == 0:
            ax.set_ylabel(r"$G_2(t)$")
        apply_axes(ax, tick_size=10, label_size=11)

        ax = axes[1, col]
        lo, hi = bootstrap_contrast(raw, "energy_G2")
        ax.axhline(0, color=BLACK, linestyle=":", linewidth=1.4)
        ax.plot(phase, sub["energy_G2_contrast"], color=BLACK, linewidth=2.0)
        ax.fill_between(phase, lo, hi, color=BLACK, alpha=0.14, linewidth=0, rasterized=True)
        ax.set_xlabel("Cardiac phase")
        if col == 0:
            ax.set_ylabel(r"$\Delta G_2(t)$")
        apply_axes(ax, tick_size=10, label_size=11)
    axes[0, -1].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save_all(fig, FIGS_DIR / "rho2_residual_energy_by_dataset")
    print(f"wrote {FIGS_DIR / 'rho2_residual_energy_by_dataset'}.eps/.png")


# =====================================================================
# TEX COMPILATION
# =====================================================================


def compile_tex() -> None:
    if not MAIN_TEX.exists():
        raise FileNotFoundError(MAIN_TEX)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", str(MAIN_TEX.name)],
        cwd=ARTICLE_DIR,
        check=True,
    )


# =====================================================================
# SINGLE-ACQUISITION ENTRY POINT -- paired arterial/venous draft
# (ARTICLE__Low_rank_modal_decomposition_endpoints), mirroring
# LowRankWaveformDecomposition.compute_acquisition_endpoints(h5_path)'s
# calling convention: one raw acquisition .h5 path in, every figure and
# table this acquisition's parent dataset needs out.
# =====================================================================


def generate_acquisition_report(h5_path: Path | str) -> dict:
    """AngioEye-style entry point (single acquisition h5_path in, matching
    LowRankWaveformDecomposition.compute_acquisition_endpoints(h5_path)):
    given one acquisition's raw .h5 file, regenerates every figure and the
    pooled endpoint table for the paired arterial/venous draft
    (ARTICLE__Low_rank_modal_decomposition_endpoints), with arteries on top
    and veins on the bottom of every panel.

    Figs. 1-2 render this exact acquisition. Figs. 3-6 and Table I are
    dataset-level aggregates (they compare Baseline 1/Flicker/Baseline 2
    epochs), so the acquisition's parent dataset is auto-resolved and its
    full acquisition set is used for those.
    """
    h5_path = Path(h5_path)
    dataset_key, dataset_label, acquisition = resolve_acquisition(h5_path)
    print(f"resolved {h5_path.name} -> dataset={dataset_key} acquisition={acquisition}")
    FIGS_DIR.mkdir(exist_ok=True)

    plot_paired_frequency_velocity_overview(h5_path, dataset_key, dataset_label, acquisition)
    plot_paired_waveform_decomposition(h5_path, dataset_key, dataset_label, acquisition)

    _root, pattern = DATASET_SOURCES[dataset_key]
    paths_by_epoch = dataset_paths(dataset_key)
    df_artery = collect_dataset(dataset_key, dataset_label, paths_by_epoch, pattern, vessel="artery")
    df_vein = collect_dataset(dataset_key, dataset_label, paths_by_epoch, pattern, vessel="vein")

    plot_paired_spectrum(dataset_key, dataset_label, df_artery, df_vein)
    plot_paired_epoch_grid(df_artery, df_vein, PAIRED_NONSVD_ENDPOINTS, f"paired_fig4_nonsvd_endpoints_{dataset_key}")
    plot_paired_epoch_grid(df_artery, df_vein, PAIRED_LOWRANK_ENDPOINTS, f"paired_fig5_lowrank_endpoints_{dataset_key}")
    plot_paired_epoch_grid(
        df_artery, df_vein, PAIRED_DIMENSIONALITY_ENDPOINTS, f"paired_fig6_dimensionality_endpoints_{dataset_key}"
    )
    table = write_paired_table1_holm_stats(dataset_key, dataset_label, df_artery, df_vein)

    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "acquisition": acquisition,
        "table": table,
    }


# =====================================================================
# CLI / MAIN
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute-from-hdf5",
        action="store_true",
        help="Recompute endpoint points from raw HDF5 files through AngioEye. Default reuses the CSV cache.",
    )
    parser.add_argument(
        "--compile-tex",
        action="store_true",
        help="Run latexmk on main_final.tex after regenerating artifacts.",
    )
    parser.add_argument(
        "--acquisition",
        metavar="H5_PATH",
        help=(
            "Generate the paired arterial/venous draft's figures and table for one acquisition's raw .h5 file "
            "(matching AngioEye's compute_acquisition_endpoints(h5_path) input style) instead of running the "
            "main (30).tex/main_final.tex pipeline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.acquisition:
        generate_acquisition_report(args.acquisition)
        return
    FIGS_DIR.mkdir(exist_ok=True)
    df = collect_all_points() if args.recompute_from_hdf5 else load_points_from_csv()
    write_endpoint_stats(df)
    write_full_endpoint_table_stats(df)
    write_table1_holm_pooled_stats()
    plot_context_figures(df)
    plot_frequency_velocity_overview()
    plot_waveform_decomposition()
    plot_lowrank_endpoint_grid(df)

    # df's TPR is AngioEye's native acq["TPR"] (median over (k,r), then median
    # over b) -- the same value Table I reports, so Fig. 5's R0 panel and
    # Table I's TPR row now agree by construction instead of using two
    # different aggregation conventions.
    plot_nonsvd_endpoint_grid(df)

    all_points_df = prepare_all_points_df(df)
    plot_dimensionality(all_points_df)
    plot_spectrum(all_points_df)
    plot_spectrum_cumulative(all_points_df)

    rho2_curves, rho2_summary = load_rho2_residual_curves()
    plot_energy(rho2_summary, rho2_curves)

    # Vein equivalents of Figs. 1-6, self-contained the same way the artery
    # panels above now are: AngioEye's _compute_representation already
    # returns every endpoint (including the full per-mode energy_fraction
    # array) needed for Figs. 3-6, so there is no dependency on the retired
    # plot_section_iv_vi_all_datasets.py script here either. Only the Fig. 6
    # residual-energy curves (not part of main_final.tex) still come from
    # explore_rho2_residual_waveforms.py's CSV cache, loaded above.
    vein_df = collect_all_points(vessel="vein") if args.recompute_from_hdf5 else load_points_from_csv(vessel="vein")
    plot_frequency_velocity_overview(vessel="vein")
    plot_waveform_decomposition(vessel="vein")
    plot_lowrank_endpoint_grid(vein_df, vessel="vein")
    plot_nonsvd_endpoint_grid(vein_df, vessel="vein")

    vein_all_points_df = prepare_all_points_df(vein_df)
    plot_dimensionality(vein_all_points_df, vessel="vein")
    plot_spectrum(vein_all_points_df, vessel="vein")
    plot_spectrum_cumulative(vein_all_points_df, vessel="vein")

    # Pooled "one-eye" endpoint tables (tab:arterial_one_eye_endpoints layout),
    # written per dataset for both vessels as standalone .tex snippets.
    for dataset_key in DATASET_ORDER:
        write_pooled_endpoint_latex_table(df, dataset_key, DATASET_LABELS[dataset_key], vessel="artery")
        write_pooled_endpoint_latex_table(vein_df, dataset_key, DATASET_LABELS[dataset_key], vessel="vein")

    if args.compile_tex:
        compile_tex()


if __name__ == "__main__":
    main()
