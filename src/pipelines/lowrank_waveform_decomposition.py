from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import warnings

from input_output.hdf5_io import (
    create_h5_file,
    find_first_existing_path,
    write_metrics_trees_to_h5,
)
from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT
from input_output.inputs import relative_hdf5_parent
from input_output.output_paths import h5_output_parent

from .core.base import (
    ProcessPipeline,
    ProcessResult,
    process_result_to_metrics_tree,
    registerPipeline,
)

import re
from collections.abc import Iterable

from scipy.stats import kruskal, mannwhitneyu
import matplotlib.pyplot as plt


T_INPUT = "Processing/VelocityPerBeat/BeatPeriodSeconds/value"
V_RAW_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
V_RAW_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
VESSEL_TYPES = ("artery", "vein")

# =====================================================================
# Epoch identity
# =====================================================================
EPOCHS = (("baseline1", "B1"), ("flicker", "Flicker"), ("baseline2", "B2"))
EPOCH_ORDER = tuple(key for key, _short in EPOCHS)
EPOCH_SHORT = dict(EPOCHS)
EPOCH_SHORT_TO_KEY = {short: key for key, short in EPOCHS}
EPOCH_SHORT_ORDER = tuple(short for _key, short in EPOCHS)
EPOCH_LABELS = {"B1": "B1", "Flicker": "F", "B2": "B2"}
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


def _epoch_rank(epoch_short: str) -> int:
    """Position of a short epoch label (B1/Flicker/B2) in EPOCH_SHORT_ORDER, for
    sorting rows into the canonical baseline1 -> flicker -> baseline2 order."""
    return EPOCH_SHORT_ORDER.index(epoch_short)


# =====================================================================
# Patient identification
# =====================================================================
PATIENT_ID_RE = re.compile(r"^(\d{6})")


def extract_patient_id(input_path: Path | str) -> str | None:
    """Returns the 6-digit patient ID prefixing the input ZIP/folder name,
    or None if the name isn't ID-prefixed."""
    match = PATIENT_ID_RE.match(Path(input_path).name)
    return match.group(1) if match else None


def prefixed_filename(filename: str, patient_id: str | None) -> str:
    """Prepends ``{patient_id}_`` to filename when a patient ID is known,
    leaving it unchanged otherwise."""
    return f"{patient_id}_{filename}" if patient_id else filename


# =====================================================================
# Vessel/schema resolution
# =====================================================================


def _enabled_vessels(veins_flag: bool) -> tuple[str, ...]:
    return ("artery", "vein") if veins_flag else ("artery",)


def _filter_vessel_candidates(
    candidates: dict[str, str], veins_flag: bool
) -> dict[str, str]:
    if veins_flag:
        return candidates
    return {k: v for k, v in candidates.items() if not k.startswith("vein/")}


def _resolve_vessel_sources(h5file, veins_flag: bool) -> tuple[dict[str, str], str] | None:
    if find_first_existing_path(h5file, [T_INPUT]) is None:
        return None
    candidates = {
        "artery/raw": V_RAW_SEGMENT_INPUT_ARTERY,
        "artery/bandlimited": V_BAND_SEGMENT_INPUT_ARTERY,
        "vein/raw": V_RAW_SEGMENT_INPUT_VEIN,
        "vein/bandlimited": V_BAND_SEGMENT_INPUT_VEIN,
    }
    return _filter_vessel_candidates(candidates, veins_flag), T_INPUT


# =====================================================================
# Acquisition/epoch classification -- resolving which epoch (baseline1/
# flicker/baseline2) each raw acquisition path belongs to. Real input is a
# ZIP whose root directly contains baseline1/flicker/baseline2 folders, so
# classification is purely folder-name matching (EPOCH_ALIASES) -- there is
# no acquisition-number naming convention to parse, so acquisitions sort by
# file name within their epoch.
# =====================================================================


def _classify_epoch(relative_parts: tuple[str, ...]) -> str | None:
    """Map the first path component that matches a known epoch alias (see
    EPOCH_ALIASES) to its canonical epoch name (e.g. a ``.../flicker/...``
    subfolder). Returns None if none match."""
    for part in relative_parts:
        epoch = EPOCH_ALIASES.get(part.lower())
        if epoch is not None:
            return epoch
    return None


def classify_epoch(h5_path: Path, input_root: Path) -> str | None:
    """Classifies one raw acquisition path (e.g. from context.input_h5_paths)
    into its epoch (baseline1/flicker/baseline2), given the cohort root it
    was discovered under (context.input_path -- the extracted ZIP root,
    which directly contains the baseline1/flicker/baseline2 folders).
    Returns None if the file isn't under input_root, or no epoch could be
    determined."""
    h5_path = Path(h5_path)
    try:
        rel = h5_path.relative_to(input_root)
    except ValueError:
        return None
    return _classify_epoch(rel.parts[:-1])


def classify_cohort(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Classify every raw acquisition path into its epoch via classify_epoch,
    then sort by (epoch order, file name). Returns (records, skipped), where
    ``skipped`` collects every path that couldn't be classified (not under a
    baseline1/flicker/baseline2-named folder)."""
    records: list[tuple[str, Path]] = []
    skipped: list[Path] = []
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        epoch = classify_epoch(h5_path, input_root)
        if epoch is None:
            skipped.append(h5_path)
            continue
        records.append((epoch, h5_path))

    records.sort(key=lambda r: (EPOCH_ORDER.index(r[0]), r[1].name))
    return records, skipped


# =====================================================================
# Input shaping -- preparing a raw waveform block/beat-period array into
# the canonical shape the SVD math operates on.
# =====================================================================


def _normalize_T(T: np.ndarray) -> np.ndarray:
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


def _ensure_segment_shape(
    v_block: np.ndarray, T: np.ndarray | None = None
) -> np.ndarray:
    v_block = np.asarray(v_block, dtype=float)
    if v_block.ndim != 4:
        raise ValueError(
            "Expected segment waveform block with shape "
            f"(n_t, n_beats, n_branches, n_radii), got {v_block.shape}"
        )
    if T is None:
        return v_block

    n_beats = int(_normalize_T(T).shape[1])
    if v_block.shape[1] == n_beats:
        return v_block
    if v_block.shape[0] == n_beats and v_block.shape[1] != n_beats:
        return np.transpose(v_block, (1, 0, 2, 3))
    raise ValueError(
        "Expected segment waveform block with one axis matching the beat-period "
        f"count ({n_beats}) in shape (n_t,n_beats,n_branches,n_radii) or "
        f"(n_beats,n_t,n_branches,n_radii), got {v_block.shape}"
    )


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


class LowRankWaveformMath:
    """Pure-numpy low-rank SVD endpoint math for beat-aligned waveform blocks."""

    eps = 1e-12
    min_valid_samples_fraction = 0.95
    min_valid_columns = 3
    exported_modes = 2

    ALL_ENDPOINTS = (
    ("A1", r"$A_1$", r"Mode-1 amplitude"),
    ("A2", r"$A_2$", r"Mode-2 amplitude"),
    ("TPR", r"$R_0$", r"Total power ratio"),
    ("R1", r"$R_1$", r"Residual level after mode 1"),
    ("R2", r"$R_2$", r"Residual level after modes 1--2"),
    ("rho1", r"$\rho_1$", r"Correlation coefficient 1"),
    ("rho2", r"$\rho_2$", r"Correlation coefficient 2"),
    ("MPR", r"$MPR$", r"Mean power ratio"),
    ("Reff", r"$R_{\mathrm{eff}}$", r"Effective residual"),
    ("PR", r"$PR$", r"Power ratio"),
    ("alpha", r"$\alpha$", r"Alpha parameter"),
    ("G1", r"$G_1$", r"Gain factor 1")
    )

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
    # Aggregation options
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
    # Singular-spectrum diagnostics: scalars summarizing how variance is 
    # spread across modes, applied both to the acquisition-wide spectrum 
    # (joint SVD) and to each beat's own spectrum (per-beat SVD).
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
        s = np.asarray(s, dtype=float)
        if s.size < 2:
            return np.nan
        lam1 = float(s[0])
        lam2 = float(s[1])
        if not (np.isfinite(lam1) and np.isfinite(lam2) and lam1 > 0):
            return np.nan
        return float(1.0 - lam2 / lam1)

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

class LowRankWaveformDecomposition(LowRankWaveformMath, ProcessPipeline):
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

    Vein processing is opt-in via ``veins_flag`` (default False): set
    ``helper.veins_flag = True`` (or subclass / construct with that
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
    veins_flag = False

    # =====================================================================
    # SVD METHOD -- core decomposition
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
        med_score = LowRankWaveformMath._safe_nanmedian(scores)
        return bool(np.isfinite(med_score) and med_score < 0)

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

        mu, x_full = _mean_subtract(v_beat)

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
        T = _normalize_T(T)
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
    # Representation orchestrator
    # =====================================================================

    def _compute_representation(self, v_block: np.ndarray, T: np.ndarray) -> dict:
        T = _normalize_T(T)
        v_block = _ensure_segment_shape(v_block, T)

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

        mu, x_full = _mean_subtract(v_block)
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
    # =====================================================================

    def _build_attrs(self, representations: list[str], input_beat_period_path: str) -> dict:
        return {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD + per-beat (t,kr) SVD robustness variant",
            "aggregation": "median over (k,r), then median over b",
            "mode_panel_max": int(self.exported_modes),
            "vessels": list(_enabled_vessels(self.veins_flag)),
            "veins_flag": bool(self.veins_flag),
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
        legacy_candidates = _filter_vessel_candidates(
            {
                "artery/raw": V_RAW_SEGMENT_INPUT_ARTERY,
                "artery/bandlimited": V_BAND_SEGMENT_INPUT_ARTERY,
                "vein/raw": V_RAW_SEGMENT_INPUT_VEIN,
                "vein/bandlimited": V_BAND_SEGMENT_INPUT_VEIN,
            },
            self.veins_flag,
        )
        schema = _resolve_vessel_sources(h5file, self.veins_flag)
        if schema is None:
            metrics = {}
            for source_name, dataset_path in legacy_candidates.items():
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
            attrs = self._build_attrs([], T_INPUT)
            return ProcessResult(metrics=metrics, attrs=attrs)

        candidates, t_path = schema
        T = _normalize_T(np.asarray(h5file[t_path], dtype=float))
        metrics, resolved = self._compute_source_metrics(h5file, candidates, T)
        attrs = self._build_attrs(resolved, t_path)
        return ProcessResult(metrics=metrics, attrs=attrs)

    def compute_acquisition_endpoints(self, h5_path) -> dict | None:
        """Standalone entry point for callers outside the AngioEye
        pipeline_engine framework (e.g. an external reproduction script):
        given a path to one acquisition's raw .h5 file, resolves whichever
        known schema it uses and, for each enabled vessel (artery always;
        vein only when ``veins_flag`` is True), runs both the joint and
        per-beat SVDs on that vessel's raw segment.

        Returns None if the file has no known beat-period path or no
        enabled vessel's raw segment at all. Otherwise returns a dict keyed
        by enabled vessel name (``{"artery": {...} | None}``, and also
        ``"vein"`` when ``veins_flag``), where a vessel is None when its
        raw segment is absent or its SVD was unavailable (too few valid
        columns). Each vessel dict carries the endpoints plus the raw "mu"
        and "energy_fraction" arrays some callers need directly."""
        with h5py.File(h5_path, "r") as h5file:
            schema = _resolve_vessel_sources(h5file, self.veins_flag)
            if schema is None:
                return None
            candidates, t_path = schema
            T = _normalize_T(np.asarray(h5file[t_path], dtype=float))

            vessel_blocks: dict[str, np.ndarray | None] = {}
            for vessel in _enabled_vessels(self.veins_flag):
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

            v_block = _ensure_segment_shape(v_block, T)

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
        ``veins_flag`` is True) -- to a new .h5 at out_path.

        Uses the same MetricsTree/ProcessResult conversion and
        ANGIOEYE_PROCESSING_ROOT group as the normal batch workflow, so the
        output is found where a downstream reader expects. Only the computed
        metrics are written, not the (often large) source waveform arrays.

        Returns False (and writes nothing) if the file has no known schema.
        """
        with h5py.File(h5_path, "r") as h5file:
            schema = _resolve_vessel_sources(h5file, self.veins_flag)
            if schema is None:
                return False
            candidates, t_path = schema
            T = _normalize_T(np.asarray(h5file[t_path], dtype=float))
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

class LowRankWaveformStatistics:
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
        x = LowRankWaveformStatistics.clean(x)
        y = LowRankWaveformStatistics.clean(y)
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
        b1 = LowRankWaveformStatistics.clean(epoch_values["baseline1"])
        fl = LowRankWaveformStatistics.clean(epoch_values["flicker"])
        b2 = LowRankWaveformStatistics.clean(epoch_values["baseline2"])
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
        holm_p = LowRankWaveformStatistics.holm_adjust(raw_p)
        deltas = [LowRankWaveformStatistics.cliffs_delta(x, y) for _, _, x, y in pairs]

        pooled_p = np.nan
        pooled_delta = np.nan
        if pooled_baseline.size >= 2 and fl.size >= 2:
            pooled_p = float(
                mannwhitneyu(fl, pooled_baseline, alternative="two-sided").pvalue
            )
            pooled_delta = LowRankWaveformStatistics.cliffs_delta(fl, pooled_baseline)

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
    def build_endpoint_table(cls, vessel: str, points_df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def pooled_test(
        df: pd.DataFrame,
        metric: str,
        branch_col: str = "epoch",
        flicker_val: str = "Flicker",
        baseline_vals: tuple[str, str] = ("B1", "B2"),
    ) -> tuple[float, float]:
        """Pooled-baseline-vs-flicker Mann-Whitney p-value and Cliff's delta for
        one metric column, reusing this class's own cliffs_delta rather than a
        separate implementation."""
        baseline = df.loc[df[branch_col].isin(baseline_vals), metric].to_numpy(dtype=float)
        flicker = df.loc[df[branch_col] == flicker_val, metric].to_numpy(dtype=float)
        baseline = baseline[np.isfinite(baseline)]
        flicker = flicker[np.isfinite(flicker)]
        if baseline.size < 2 or flicker.size < 2:
            return float("nan"), float("nan")
        p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
        return p, LowRankWaveformStatistics.cliffs_delta(flicker, baseline)

    @staticmethod
    def format_p(p: float) -> str:
        if not np.isfinite(p):
            return "p=n/a"
        if p < 1e-3:
            return f"p={p:.1e}"
        return f"p={p:.3f}"

    @staticmethod
    def format_delta(delta: float) -> str:
        if not np.isfinite(delta):
            return r"$\delta$=n/a"
        return rf"$\delta$={delta:+.2f}"

    @staticmethod
    def _style_axes(ax) -> None:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="both", labelsize=9)

    @classmethod
    def plot_acquisition_summary(
        cls, vessel_data: dict, out_path: Path, vessel: str
    ) -> Path:
        """Per-acquisition, per-vessel QC figure: beat period, joint-SVD mode
        amplitudes (A1/A2), residual ratios (rho1/rho2), and the singular-value
        energy spectrum, one beat-indexed panel each -- built entirely from the
        public fields compute_acquisition_endpoints already exposes (no raw
        per-timepoint reconstruction needed)."""
        beatwise = vessel_data["beatwise"]
        energy_fraction = np.asarray(vessel_data.get("energy_fraction", []), dtype=float)
        beat_period_b = np.asarray(vessel_data["beat_period_b"], dtype=float)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))

        axes[0].plot(beat_period_b, marker="o", color="black")
        axes[0].set_title("Beat period (s)")
        axes[0].set_xlabel("Beat")

        axes[1].plot(beatwise["A1_b"], marker="o", label="A1")
        axes[1].plot(beatwise["A2_b"], marker="o", label="A2")
        axes[1].set_title("Mode amplitude")
        axes[1].set_xlabel("Beat")
        axes[1].legend(frameon=False, fontsize=8)

        axes[2].plot(beatwise["rho1_b"], marker="o", label=r"$\rho_1$")
        axes[2].plot(beatwise["rho2_b"], marker="o", label=r"$\rho_2$")
        axes[2].set_title("Residual ratio")
        axes[2].set_xlabel("Beat")
        axes[2].legend(frameon=False, fontsize=8)

        n_modes = min(6, energy_fraction.size)
        axes[3].bar(range(1, n_modes + 1), energy_fraction[:n_modes], color="black")
        axes[3].set_title("Energy fraction")
        axes[3].set_xlabel("Mode")

        for ax in axes:
            cls._style_axes(ax)
        fig.suptitle(f"{vessel} -- acquisition summary", fontsize=11)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_cohort_endpoint_grid(cls, points_df: pd.DataFrame, out_path: Path) -> Path:
        """Per-cohort QC figure: one dot-whisker panel per primary endpoint
        (A1, A2, rho1, rho2, TPR, mu), median +/- SD across acquisitions in
        each epoch (B1/Flicker/B2), from the same points table
        build_endpoint_table summarises."""
        metrics = ("A1", "A2", "rho1", "rho2", "TPR", "mu")
        fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 3))
        positions = {epoch: idx for idx, epoch in enumerate(EPOCH_SHORT_ORDER)}
        for ax, metric in zip(axes, metrics, strict=True):
            for epoch in EPOCH_SHORT_ORDER:
                vals = points_df.loc[points_df["epoch"] == epoch, metric].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                med = float(np.nanmedian(vals))
                sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
                ax.errorbar(
                    [positions[epoch]], [med], yerr=[sd],
                    fmt="o", color="black", capsize=4,
                )
            ax.set_xticks(list(positions.values()))
            ax.set_xticklabels(list(positions.keys()), fontsize=8)
            ax.set_title(metric)
            cls._style_axes(ax)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

class LowRankWaveformConfounds:
    """Sec. V.B confound-control grid (analysis-choice sweep + verdicts),
    plus the per-cohort collection/statistics pipeline built on it
    (collect_acquisitions, run_confound_statistics). No dependency on the
    postprocess framework -- see LowRankConfoundStatisticsPostprocess below
    for the registered BatchPostprocess entry point that wraps
    run_confound_statistics for a real batch run."""

    # Metrics with both a joint-SVD and a per-beat-SVD representation --
    # the ones build_grid sweeps over the svd_method axis for. TPR/mpr/
    # alpha/G1/mpr_prime have no such axis and are handled separately.
    METRICS_SVD = ("A1", "A2", "rho1", "rho2")

    def __init__(self, engine: LowRankWaveformDecomposition | None = None) -> None:
        if engine is None:
            # Confound statistics report both compartments; opt into the
            # pipeline's optional vein processing (artery-only is the
            # pipeline default).
            engine = LowRankWaveformDecomposition()
            engine.veins_flag = True
        self._lr = engine

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
        EPOCH_ORDER (the shape LowRankWaveformStatistics.epoch_group_tests /
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
        vessel: str,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """Sec. V.B robustness sweep for one vessel. Runs LowRankWaveformStatistics.epoch_group_tests
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
            return LowRankWaveformStatistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Compute one grid cell and append its flattened row to grid_rows (and
            to verdict_source, grouped by metric for the verdict summary)."""
            tests = run_combo(metric, svd_method, stat, regressed)
            row = {
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

    def plot_cohort_dimensionality(
        self, acqs_by_epoch: dict[str, list[dict]], out_path: Path
    ) -> Path:
        """Per-cohort QC figure: effective rank and participation ratio
        (median +/- SD across acquisitions) in each epoch (B1/Flicker/B2) --
        the same joint-SVD dimensionality diagnostics build_grid's robustness
        sweep is checking, read directly off each acquisition's acq dict."""
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for ax, metric in zip(axes, ("effective_rank", "participation_ratio"), strict=True):
            for epoch in EPOCH_ORDER:
                vals = np.array(
                    [
                        float(a["acq"].get(metric, np.nan))
                        for a in acqs_by_epoch[epoch]
                    ],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                med = float(np.nanmedian(vals))
                sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
                ax.errorbar(
                    [EPOCH_ORDER.index(epoch)], [med], yerr=[sd],
                    fmt="o", color="black", capsize=4,
                )
            ax.set_xticks(range(len(EPOCH_ORDER)))
            ax.set_xticklabels(EPOCH_SHORT_ORDER, fontsize=8)
            ax.set_title(metric.replace("_", " "))
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def build_beat_rows(
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
                    "vessel": vessel,
                    "acquisition": sequence,
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

    @staticmethod
    def build_points_row(
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
            "vessel": vessel,
            "acquisition": sequence,
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

    def collect_acquisitions(
        self,
        records: list[tuple[str, Path]],
        input_root: Path,
        output_dir: Path | None = None,
        patient_id: str | None = None,
    ) -> dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]]:
        """Runs the low-rank engine over the cohort's already-classified,
        already-sorted (epoch, h5_path) records (see classify_cohort), and
        builds each vessel's acqs_by_epoch/points_rows/beat_rows. Returns
        {"artery": (acqs_by_epoch, points_rows, beat_rows), "vein": (...)} --
        an acquisition contributes to a vessel's tables only if that vessel's
        raw segment was present and its joint SVD was available for it (see
        compute_acquisition_endpoints). When output_dir is given, each
        acquisition's endpoints and QC figure are also persisted under
        output_dir/h5/<relative acquisition folder>/ -- the same per-acquisition
        output layout (input_output.output_paths.h5_output_parent, keyed by
        input_output.inputs.relative_hdf5_parent) the pipeline engine uses for
        LowRankWaveformDecomposition's own outputs -- so results land next to
        the acquisition they came from (e.g. under baseline1/)."""
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
            data = self._lr.compute_acquisition_endpoints(h5_path)
            if data is None:
                continue
            acquisition_out_dir = None
            if output_dir is not None:
                # Persist this acquisition's joint-SVD + per-beat endpoints
                # (both vessels, both representations) to their own .h5 (the same
                # MetricsTree/ANGIOEYE_PROCESSING_ROOT structure a normal batch run
                # produces) and its per-vessel QC figure, both mirrored under the
                # acquisition's own relative folder so they land next to that
                # acquisition's other outputs (e.g. under baseline1/).
                relative_parent = relative_hdf5_parent(h5_path, input_root)
                acquisition_out_dir = h5_output_parent(output_dir, relative_parent)
                out_name = prefixed_filename(
                    f"{h5_path.stem}_pipelines_result.h5", patient_id
                )
                self._lr.write_acquisition_h5(h5_path, acquisition_out_dir / out_name)

            for vessel in VESSEL_TYPES:
                vessel_data = data.get(vessel)
                if vessel_data is None:
                    continue
                if acquisition_out_dir is not None:
                    fig_name = prefixed_filename(
                        f"{h5_path.stem}_{vessel}_summary.png", patient_id
                    )
                    LowRankWaveformStatistics.plot_acquisition_summary(
                        vessel_data, acquisition_out_dir / fig_name, vessel
                    )
                state = per_vessel[vessel]
                state["acqs_by_epoch"][epoch].append(vessel_data)
                seq = state["sequence_counters"][epoch]
                state["sequence_counters"][epoch] += 1
                epoch_short = EPOCH_SHORT[epoch]

                state["points_rows"].append(
                    self.build_points_row(vessel, h5_path, seq, epoch_short, vessel_data)
                )
                state["beat_rows"].extend(
                    self.build_beat_rows(vessel, h5_path, seq, epoch_short, vessel_data)
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

    def run_confound_statistics(
        self,
        input_h5_paths: Iterable[Path],
        input_root: Path,
        output_dir: Path,
    ) -> tuple[str, list[Path]]:
        """Classifies input_h5_paths into epochs (classify_cohort), computes
        every acquisition's endpoints, writes per-acquisition metrics-only
        .h5 files and QC figures (mirrored under output_dir/h5/<relative
        acquisition folder>/, e.g. baseline1/) plus this cohort's
        points/beats/tables/confound_control CSVs and figures under
        output_dir/lowrank_confound_statistics/ -- i.e. alongside, not
        inside, the baseline1/flicker/baseline2 acquisition folders. Every
        generated filename is prefixed with the patient ID parsed from
        input_root's name (see extract_patient_id) when one is present.
        Returns (summary, generated_paths)."""
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        patient_id = extract_patient_id(input_root)
        root_dir = output_dir / "lowrank_confound_statistics"
        for sub in ("points", "beats", "tables", "confound_control", "figures"):
            (root_dir / sub).mkdir(parents=True, exist_ok=True)

        records, skipped = classify_cohort(input_h5_paths, input_root)
        if not records:
            raise ValueError(
                "No acquisitions could be classified into an epoch under "
                f"{input_root} (expected baseline1/flicker/baseline2-named "
                "subfolders)."
            )

        generated_paths: list[Path] = []

        def _write_csv(df: pd.DataFrame, subdir: str, filename: str) -> Path:
            path = root_dir / subdir / prefixed_filename(filename, patient_id)
            df.to_csv(path, index=False)
            generated_paths.append(path)
            return path

        per_vessel = self.collect_acquisitions(
            records, input_root, output_dir=output_dir, patient_id=patient_id
        )

        vessel_summaries: list[str] = []
        all_points: list[pd.DataFrame] = []
        for vessel, (acqs_by_epoch, points_rows, beat_rows) in per_vessel.items():
            n_acq = sum(len(acqs_by_epoch[e]) for e in EPOCH_ORDER)
            if n_acq == 0:
                continue
            vessel_summaries.append(
                f"{vessel}={n_acq} (B1={len(acqs_by_epoch['baseline1'])}, "
                f"F={len(acqs_by_epoch['flicker'])}, B2={len(acqs_by_epoch['baseline2'])})"
            )

            points_df = pd.DataFrame(points_rows)
            all_points.append(points_df)
            _write_csv(points_df, "points", f"{vessel}_points.csv")
            _write_csv(pd.DataFrame(beat_rows), "beats", f"{vessel}_beats.csv")
            _write_csv(
                LowRankWaveformStatistics.build_endpoint_table(vessel, points_df),
                "tables",
                f"{vessel}_endpoint_table.csv",
            )

            grid_rows, verdict_rows = self.build_grid(vessel, acqs_by_epoch)
            _write_csv(pd.DataFrame(grid_rows), "confound_control", f"{vessel}_confound_grid.csv")
            _write_csv(
                pd.DataFrame(verdict_rows), "confound_control", f"{vessel}_confound_verdict.csv"
            )

            fig_stem = prefixed_filename(vessel, patient_id)
            generated_paths.append(
                LowRankWaveformStatistics.plot_cohort_endpoint_grid(
                    points_df, root_dir / "figures" / f"{fig_stem}_endpoints.png"
                )
            )
            generated_paths.append(
                self.plot_cohort_dimensionality(
                    acqs_by_epoch, root_dir / "figures" / f"{fig_stem}_dimensionality.png"
                )
            )

        if not all_points:
            raise ValueError(
                "No vessel had any valid acquisitions; nothing to write."
            )

        skipped_note = (
            f"; {len(skipped)} file(s) not classified into any epoch" if skipped else ""
        )
        combined_points = pd.concat(all_points, ignore_index=True)
        summary = (
            f"Low-rank confound-controlled statistics: {len(combined_points)} "
            f"acquisition-vessel row(s) total ({', '.join(vessel_summaries)}){skipped_note}."
        )
        return summary, generated_paths

    def run(self, input_h5_paths, input_root, output_dir):
        """Validate that raw acquisition inputs are present and run the
        statistics pipeline, returning (summary, generated_paths)."""
        if not input_h5_paths:
            raise ValueError(
                "No input acquisition files are available for postprocessing."
            )

        confounds = LowRankWaveformConfounds()
        return confounds.run_confound_statistics(
            input_h5_paths=input_h5_paths,
            input_root=input_root,
            output_dir=output_dir,
        )
