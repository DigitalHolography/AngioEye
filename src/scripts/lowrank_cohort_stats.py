#!/usr/bin/env python
"""Build ``lowrank_cohort.h5`` from packed AngioEye low-rank result H5s.

Usage:
    python src/scripts/lowrank_cohort_stats.py INPUT -o OUTPUT
    python -m scripts.lowrank_cohort_stats INPUT -o OUTPUT

INPUT is a cohort folder or a ZIP of that tree. The script reads packed
``*_AE.h5`` (or legacy result H5s) that already contain low-rank metrics —
it does not recompute SVD or write Figs 5--7. Those figures stay in
``postprocess.lowrank_waveform_cohort``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kruskal, linregress, mannwhitneyu, spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from input_output.archive_io import extracted_zip_tree  # noqa: E402
from input_output.hdf5_io import (  # noqa: E402
    UTF8_STRING_DTYPE,
    open_h5,
    set_attr_safe,
    write_value_dataset,
)
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT  # noqa: E402
from pipelines.lowrank_waveform_decomposition import (  # noqa: E402
    SVD_METHODS,
    aggregate_beatwise,
    aggregate_rho,
)
from postprocess.lowrank_waveform_cohort import (  # noqa: E402
    EPOCH_ORDER,
    EPOCH_SHORT,
    epoch_key_from_folder,
)

if TYPE_CHECKING:
    from postprocess.lowrank_waveform_cohort import CohortCollection


COHORT_H5_ROOT = f"{ANGIOEYE_POSTPROCESS_ROOT}/lowrank_waveform_cohort"
COHORT_H5_BASENAME = "lowrank_cohort.h5"

# Internal acquisitions-table column, published symbol.
CANONICAL_ENDPOINTS = (
    ("A1", "A1"),
    ("A2", "A2"),
    ("TPR", "R0"),
    ("R1", "R1"),
    ("R2", "R2"),
    ("rho1", "rho1"),
    ("rho2", "rho2"),
    ("mpr", "MPR"),
    ("effective_rank", "Reff"),
    ("participation_ratio", "PR"),
)
ENDPOINT_BY_METRIC = dict(CANONICAL_ENDPOINTS)

CONFUND_EIGHT = ("A1", "A2", "R1", "R2", "rho1", "rho2", "TPR", "mpr")
CONFUND_SIX = ("effective_rank", "participation_ratio")

COHORT_DICTIONARY = {
    "columns": {
        "endpoint": "Published symbol for the endpoint.",
        "metric": "Internal acquisitions-table column name.",
        "B1": "Baseline1 median [IQR].",
        "Flicker": "Flicker median [IQR].",
        "B2": "Baseline2 median [IQR].",
        "n_B1": "Finite Baseline1 acquisition count.",
        "n_Flicker": "Finite Flicker acquisition count.",
        "n_B2": "Finite Baseline2 acquisition count.",
        "kw_H": "Kruskal-Wallis H across B1 / Flicker / B2.",
        "kw_p": "Kruskal-Wallis p across B1 / Flicker / B2.",
        "B1_vs_F": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "F_vs_B2": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "B1_vs_B2": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "pooled_p": "Mann-Whitney p for Flicker vs pooled Baseline (B1+B2).",
        "pooled_p_holm": "Pooled Baseline-vs-Flicker p, Holm-adjusted across the 10-endpoint family.",
        "pooled_delta": "Cliff's delta for Flicker vs pooled Baseline.",
        "n_combinations": "Designed size of this endpoint's confound grid, not a missing-data count.",
        "n_significant": "Confound combinations with pooled p < 0.05.",
        "retained": "True when every confound combination has pooled p < 0.05 and one Cliff's delta sign.",
        "aggregation_retained": "Native values only; median vs mean; both SVD methods.",
        "beat_period_retained": "Default aggregation (median, or n/a); native vs residualized; both SVD methods.",
        "spearman_rho": (
            "Spearman correlation of the headline metric vs beat period T, "
            "on B1∪B2 only."
        ),
        "spearman_p": (
            "Spearman p-value vs beat period T, on B1∪B2 only."
        ),
        "r_squared": (
            "OLS R² of the B1∪B2 fit of endpoint vs beat period T "
            "(the line used for residualization)."
        ),
        "r_squared_flicker": (
            "R² of the baseline-fitted line α + βT evaluated on flicker points "
            "only (1 − SS_res/SS_tot). NaN if the baseline fit is missing or "
            "there are fewer than two finite flicker pairs."
        ),
        "slope": "OLS slope of endpoint vs beat period T, fitted on B1∪B2 only.",
        "intercept": (
            "OLS intercept of endpoint vs beat period T, fitted on B1∪B2 only."
        ),
        "n_regression": (
            "Number of B1∪B2 acquisitions used in the OLS / Spearman fit."
        ),
        "svd_method": "Which SVD representation this confound row used.",
        "beat_aggregation": "How beats were collapsed to one acquisition value.",
        "beat_period_control": "Whether the metric was residualized against beat period.",
        "cliffs_delta": "Cliff's delta for Flicker vs pooled Baseline in this combination.",
    },
    "values": {
        "svd_method": {
            "joint": "Acquisition-level joint SVD.",
            "per_beat": "Per-beat SVD, then aggregated across beats.",
            "n/a": "This metric has no SVD axis in this row.",
        },
        "beat_aggregation": {
            "median": "Acquisition value is the median across beats.",
            "mean": "Acquisition value is the mean across beats.",
            "n/a": "No beat-aggregation choice (acquisition scalar).",
        },
        "beat_period_control": {
            "native": "Metric as computed, no period regression.",
            "residualized": (
                "Residual after subtracting α + βT, with α, β fitted on B1∪B2 only."
            ),
        },
        "endpoint": {
            "A1": "Mode-1 amplitude.",
            "A2": "Mode-2 amplitude.",
            "R0": "TPR (total pulsatile RMS).",
            "R1": "Mode-1 residual RMS.",
            "R2": "Mode-2 residual RMS.",
            "rho1": "Mode-1 residual ratio R1/R0.",
            "rho2": "Mode-2 residual ratio R2/R0.",
            "MPR": "mpr (mean-to-pulsatile ratio).",
            "Reff": "effective_rank.",
            "PR": "participation_ratio.",
        },
    },
}

_INT_COLUMNS = {
    "n_B1",
    "n_Flicker",
    "n_B2",
    "n_combinations",
    "n_significant",
    "n_regression",
    "acquisition",
    "beat_index",
    "n_valid_columns",
    "n_total_columns",
}
_BOOL_COLUMNS = {
    "aggregation_retained",
    "beat_period_retained",
    "retained",
}


def acqs_by_canonical_epochs(
    acqs_by_group: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Remap folder-keyed acquisition lists onto baseline1 / flicker / baseline2."""
    out: dict[str, list[dict]] = {epoch: [] for epoch in EPOCH_ORDER}
    for group, acqs in acqs_by_group.items():
        key = epoch_key_from_folder(group)
        if key in out:
            out[key].extend(acqs)
    return out


def _first_finite_array(mapping: dict, *keys: str) -> np.ndarray:
    """First array among ``keys`` that contains a finite value."""
    for key in keys:
        if key not in mapping:
            continue
        arr = np.asarray(mapping[key], dtype=float)
        if np.isfinite(arr).any():
            return arr
    return np.asarray([], dtype=float)


class LowRankWaveformStatistics:
    """Nonparametric tests and the ten-row ``stats`` table."""

    CANONICAL_ENDPOINTS = CANONICAL_ENDPOINTS

    @staticmethod
    def clean(x) -> np.ndarray:
        """Drop non-finite entries so tests never see NaN/inf."""
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Cliff's delta: Pr(X>Y) - Pr(X<Y). Positive means ``x`` tends larger."""
        x = LowRankWaveformStatistics.clean(x)
        y = LowRankWaveformStatistics.clean(y)
        if x.size == 0 or y.size == 0:
            return float("nan")
        diff = x[:, None] - y[None, :]
        return float(np.mean(diff > 0) - np.mean(diff < 0))

    @staticmethod
    def holm_adjust(p_values: list[float]) -> list[float]:
        """Holm-Bonferroni adjusted p-values, same order as the input.

        NaN entries stay NaN and are excluded from the family size.
        """
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
        """Kruskal-Wallis, pairwise MWU+Holm, and pooled Baseline-vs-Flicker tests."""
        b1 = LowRankWaveformStatistics.clean(epoch_values["baseline1"])
        fl = LowRankWaveformStatistics.clean(epoch_values["flicker"])
        b2 = LowRankWaveformStatistics.clean(epoch_values["baseline2"])
        pooled_baseline = np.concatenate([b1, b2])

        kw_H = np.nan
        kw_p = np.nan
        if b1.size >= 1 and fl.size >= 1 and b2.size >= 1:
            try:
                kw = kruskal(b1, fl, b2)
                kw_H = float(kw.statistic)
                kw_p = float(kw.pvalue)
            except ValueError:
                kw_H = np.nan
                kw_p = np.nan

        pairs = [
            ("baseline1", "flicker", b1, fl),
            ("flicker", "baseline2", fl, b2),
            ("baseline1", "baseline2", b1, b2),
        ]
        raw_p: list[float] = []
        raw_U: list[float] = []
        for _, _, x, y in pairs:
            if x.size >= 2 and y.size >= 2:
                result = mannwhitneyu(x, y, alternative="two-sided")
                raw_p.append(float(result.pvalue))
                raw_U.append(float(result.statistic))
            else:
                raw_p.append(float("nan"))
                raw_U.append(float("nan"))
        holm_p = LowRankWaveformStatistics.holm_adjust(raw_p)
        deltas = [LowRankWaveformStatistics.cliffs_delta(x, y) for _, _, x, y in pairs]

        pooled_p = np.nan
        pooled_delta = np.nan
        pooled_U = np.nan
        if pooled_baseline.size >= 2 and fl.size >= 2:
            pooled = mannwhitneyu(fl, pooled_baseline, alternative="two-sided")
            pooled_p = float(pooled.pvalue)
            pooled_U = float(pooled.statistic)
            pooled_delta = LowRankWaveformStatistics.cliffs_delta(fl, pooled_baseline)

        return {
            "kw_H": kw_H,
            "kw_p": kw_p,
            "pairwise": [
                {
                    "pair": f"{a} vs {b}",
                    "p": p,
                    "p_holm": ph,
                    "U": u,
                    "cliffs_delta": d,
                }
                for (a, b, _, _), p, ph, u, d in zip(
                    pairs, raw_p, holm_p, raw_U, deltas
                )
            ],
            "pooled_p": pooled_p,
            "pooled_U": pooled_U,
            "pooled_delta": pooled_delta,
            "n": {"baseline1": b1.size, "flicker": fl.size, "baseline2": b2.size},
        }

    @staticmethod
    def format_median_iqr(vals: np.ndarray) -> str:
        """``median [IQR]`` with 3 significant figures."""
        vals = LowRankWaveformStatistics.clean(vals)
        if vals.size == 0:
            return "--"
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        return f"{med:.3g} [{iqr:.3g}]"

    @staticmethod
    def format_p_scientific(p: float) -> str:
        """Scientific / fixed formatting for a p-value."""
        if not np.isfinite(p):
            return "--"
        if p >= 1e-2:
            return f"{p:.3f}"
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.2f}e{exp}"

    @classmethod
    def format_raw_holm(cls, raw: float, holm: float) -> str:
        """``raw / Holm`` p-value pair."""
        return f"{cls.format_p_scientific(raw)} / {cls.format_p_scientific(holm)}"

    @classmethod
    def beat_period_regression(cls, acquisitions_df: pd.DataFrame) -> dict[str, dict]:
        """OLS / Spearman of each headline metric vs beat period, fitted on B1∪B2."""
        out: dict[str, dict] = {}
        for metric, _endpoint in CANONICAL_ENDPOINTS:
            empty = {
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "r_squared": float("nan"),
                "r_squared_flicker": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
                "n_regression": 0,
            }
            if metric not in acquisitions_df.columns:
                out[metric] = empty
                continue
            values = {
                epoch: acquisitions_df.loc[
                    acquisitions_df["epoch"] == short, metric
                ].to_numpy(dtype=float)
                for epoch, short in EPOCH_SHORT.items()
            }
            periods = {
                epoch: acquisitions_df.loc[
                    acquisitions_df["epoch"] == short, "beat_period"
                ].to_numpy(dtype=float)
                for epoch, short in EPOCH_SHORT.items()
            }
            _, fit = LowRankWaveformConfounds.residualize_against_beat_period(
                values, periods
            )
            n = int(fit["n"])
            rho = p_rho = float("nan")
            if n >= 3:
                y = np.concatenate([values["baseline1"], values["baseline2"]])
                x = np.concatenate([periods["baseline1"], periods["baseline2"]])
                mask = np.isfinite(x) & np.isfinite(y)
                try:
                    spearman = spearmanr(x[mask], y[mask])
                    rho = float(spearman.statistic)
                    p_rho = float(spearman.pvalue)
                except ValueError:
                    pass
            out[metric] = {
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "r_squared": float(fit["r_squared"]),
                "r_squared_flicker": float(fit["r_squared_flicker"]),
                "slope": float(fit["slope"]),
                "intercept": float(fit["intercept"]),
                "n_regression": n,
            }
        return out

    @classmethod
    def build_stats_table(
        cls,
        acquisitions_df: pd.DataFrame,
        confounds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """One row per canonical endpoint: medians, tests, retain flags, regression."""
        retain = LowRankWaveformConfounds.retain_flags_by_metric(confounds_df)
        regression = cls.beat_period_regression(acquisitions_df)

        rows: list[dict] = []
        pooled_raw: list[float] = []
        for metric, endpoint in CANONICAL_ENDPOINTS:
            if metric not in acquisitions_df.columns:
                b1 = fl = b2 = np.asarray([], dtype=float)
            else:
                b1 = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["baseline1"], metric
                ].to_numpy(dtype=float)
                fl = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["flicker"], metric
                ].to_numpy(dtype=float)
                b2 = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["baseline2"], metric
                ].to_numpy(dtype=float)
            tests = cls.epoch_group_tests(
                {"baseline1": b1, "flicker": fl, "baseline2": b2}
            )
            pair = {p["pair"]: p for p in tests["pairwise"]}
            b1_f = pair["baseline1 vs flicker"]
            f_b2 = pair["flicker vs baseline2"]
            b1_b2 = pair["baseline1 vs baseline2"]
            flags = retain.get(metric, {})
            fit = regression.get(metric, {})
            pooled_raw.append(tests["pooled_p"])
            rows.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "B1": cls.format_median_iqr(b1),
                    "Flicker": cls.format_median_iqr(fl),
                    "B2": cls.format_median_iqr(b2),
                    "n_B1": int(tests["n"]["baseline1"]),
                    "n_Flicker": int(tests["n"]["flicker"]),
                    "n_B2": int(tests["n"]["baseline2"]),
                    "kw_H": tests["kw_H"],
                    "kw_p": tests["kw_p"],
                    "B1_vs_F": cls.format_raw_holm(b1_f["p"], b1_f["p_holm"]),
                    "F_vs_B2": cls.format_raw_holm(f_b2["p"], f_b2["p_holm"]),
                    "B1_vs_B2": cls.format_raw_holm(b1_b2["p"], b1_b2["p_holm"]),
                    "B1_vs_F_p": b1_f["p"],
                    "B1_vs_F_p_holm": b1_f["p_holm"],
                    "B1_vs_F_U": b1_f["U"],
                    "F_vs_B2_p": f_b2["p"],
                    "F_vs_B2_p_holm": f_b2["p_holm"],
                    "F_vs_B2_U": f_b2["U"],
                    "B1_vs_B2_p": b1_b2["p"],
                    "B1_vs_B2_p_holm": b1_b2["p_holm"],
                    "B1_vs_B2_U": b1_b2["U"],
                    "pooled_p": tests["pooled_p"],
                    "pooled_U": tests["pooled_U"],
                    "pooled_delta": tests["pooled_delta"],
                    "delta_F_B1": cls.cliffs_delta(cls.clean(fl), cls.clean(b1)),
                    "delta_F_B2": cls.cliffs_delta(cls.clean(fl), cls.clean(b2)),
                    "delta_F_pooled": tests["pooled_delta"],
                    "n_combinations": int(flags.get("n_combinations", 0)),
                    "n_significant": int(flags.get("n_significant", 0)),
                    "aggregation_retained": bool(
                        flags.get("aggregation_retained", False)
                    ),
                    "beat_period_retained": bool(
                        flags.get("beat_period_retained", False)
                    ),
                    "retained": bool(flags.get("retained", False)),
                    "spearman_rho": fit.get("spearman_rho", float("nan")),
                    "spearman_p": fit.get("spearman_p", float("nan")),
                    "r_squared": fit.get("r_squared", float("nan")),
                    "r_squared_flicker": fit.get("r_squared_flicker", float("nan")),
                    "slope": fit.get("slope", float("nan")),
                    "intercept": fit.get("intercept", float("nan")),
                    "n_regression": int(fit.get("n_regression", 0)),
                }
            )

        for row, holm in zip(rows, cls.holm_adjust(pooled_raw)):
            row["pooled_p_holm"] = holm

        column_order = [
            "endpoint",
            "metric",
            "B1",
            "Flicker",
            "B2",
            "n_B1",
            "n_Flicker",
            "n_B2",
            "kw_H",
            "kw_p",
            "B1_vs_F",
            "F_vs_B2",
            "B1_vs_B2",
            "B1_vs_F_p",
            "B1_vs_F_p_holm",
            "B1_vs_F_U",
            "F_vs_B2_p",
            "F_vs_B2_p_holm",
            "F_vs_B2_U",
            "B1_vs_B2_p",
            "B1_vs_B2_p_holm",
            "B1_vs_B2_U",
            "pooled_p",
            "pooled_p_holm",
            "pooled_U",
            "pooled_delta",
            "delta_F_B1",
            "delta_F_B2",
            "delta_F_pooled",
            "n_combinations",
            "n_significant",
            "aggregation_retained",
            "beat_period_retained",
            "retained",
            "spearman_rho",
            "spearman_p",
            "r_squared",
            "r_squared_flicker",
            "slope",
            "intercept",
            "n_regression",
        ]
        return pd.DataFrame(rows)[column_order]


class LowRankWaveformConfounds:
    """Confound-control grid: analysis-choice sweep plus retain flags."""

    @staticmethod
    def residualize_against_beat_period(
        epoch_values: dict[str, np.ndarray],
        epoch_beat_periods: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Fit ``d = α + βT`` on B1∪B2, then residualize every epoch with that line.

        Returns residuals and the baseline-only fit (slope, intercept, R², R² on
        flicker, N). Needs at least 3 finite baseline (value, period) pairs;
        otherwise returns the inputs and NaN diagnostics.
        """
        baseline_epochs = ("baseline1", "baseline2")
        base_values = np.concatenate(
            [
                np.asarray(epoch_values.get(epoch, []), dtype=float)
                for epoch in baseline_epochs
            ]
        )
        base_periods = np.concatenate(
            [
                np.asarray(epoch_beat_periods.get(epoch, []), dtype=float)
                for epoch in baseline_epochs
            ]
        )
        mask = np.isfinite(base_values) & np.isfinite(base_periods)
        n = int(np.sum(mask))
        empty_fit = {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "r_squared_flicker": float("nan"),
            "n": n,
        }
        native = {k: np.asarray(v, dtype=float) for k, v in epoch_values.items()}
        if n < 3:
            return native, empty_fit

        fit = linregress(base_periods[mask], base_values[mask])
        slope = float(fit.slope)
        intercept = float(fit.intercept)
        r_squared = float(fit.rvalue ** 2)

        flicker_values = np.asarray(epoch_values.get("flicker", []), dtype=float)
        flicker_periods = np.asarray(
            epoch_beat_periods.get("flicker", []), dtype=float
        )
        fl_mask = np.isfinite(flicker_values) & np.isfinite(flicker_periods)
        r_squared_flicker = float("nan")
        if int(np.sum(fl_mask)) >= 2:
            y = flicker_values[fl_mask]
            yhat = intercept + slope * flicker_periods[fl_mask]
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot > 0:
                r_squared_flicker = 1.0 - ss_res / ss_tot

        out = {}
        for epoch, values in epoch_values.items():
            values = np.asarray(values, dtype=float)
            periods = np.asarray(epoch_beat_periods[epoch], dtype=float)
            out[epoch] = values - (slope * periods + intercept)
        return out, {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "r_squared_flicker": r_squared_flicker,
            "n": n,
        }

    def acq_dots(
        self,
        acqs: list[dict],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> np.ndarray:
        """One acquisition-level value per acquisition for ``metric``."""
        out = []
        for a in acqs:
            beatwise = a.get("beatwise") or {}
            per_beat = a.get("per_beat_svd") or {}
            acq = a.get("acq") or {}
            if metric == "TPR":
                if svd_method == "per_beat":
                    arr = _first_finite_array(per_beat, "TPR_b_pb", "R0_b_pb")
                else:
                    arr = _first_finite_array(beatwise, "TPR_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric == "mpr":
                if svd_method == "per_beat":
                    arr = _first_finite_array(per_beat, "mpr_b_pb", "MPR_b_pb")
                else:
                    arr = _first_finite_array(beatwise, "mpr_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("A1", "A2", "R1", "R2"):
                if svd_method == "per_beat":
                    arr = _first_finite_array(per_beat, f"{metric}_b_pb")
                else:
                    arr = _first_finite_array(beatwise, f"{metric}_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "per_beat":
                    R_b = _first_finite_array(per_beat, f"R{m}_b_pb")
                    tpr_b = _first_finite_array(per_beat, "TPR_b_pb", "R0_b_pb")
                else:
                    R_b = _first_finite_array(beatwise, f"R{m}_b")
                    tpr_b = _first_finite_array(beatwise, "TPR_b")
                out.append(
                    aggregate_rho(R_b, tpr_b, stat if stat != "n/a" else "median")
                )
            elif metric in ("effective_rank", "participation_ratio"):
                if svd_method == "per_beat":
                    arr = _first_finite_array(per_beat, f"{metric}_b_pb")
                    out.append(
                        aggregate_beatwise(arr, stat if stat != "n/a" else "median")
                    )
                else:
                    out.append(float(acq.get(metric, np.nan)))
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
        """``acq_dots`` per epoch, keyed by baseline1 / flicker / baseline2."""
        return {
            epoch: self.acq_dots(acqs_by_epoch[epoch], metric, svd_method, stat)
            for epoch in EPOCH_ORDER
        }

    @staticmethod
    def epoch_beat_periods(
        acqs_by_epoch: dict[str, list[dict]],
    ) -> dict[str, np.ndarray]:
        """Mean beat period per acquisition, the covariate for residualization."""
        return {
            epoch: np.array(
                [a["beat_period_mean"] for a in acqs_by_epoch[epoch]], dtype=float
            )
            for epoch in EPOCH_ORDER
        }

    def build_grid(
        self,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Sweep metric × SVD × aggregation × period-control into one table."""
        periods = self.epoch_beat_periods(acqs_by_epoch)
        grid_rows: list[dict] = []

        def run_combo(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> dict:
            """Epoch-group tests for one grid cell, residualizing when asked."""
            values = self.epoch_dots(acqs_by_epoch, metric, svd_method, stat)
            if regressed:
                values, _fit = self.residualize_against_beat_period(values, periods)
            return LowRankWaveformStatistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Append one flattened grid row."""
            tests = run_combo(metric, svd_method, stat, regressed)
            grid_rows.append(
                {
                    "endpoint": ENDPOINT_BY_METRIC[metric],
                    "metric": metric,
                    "svd_method": svd_method if svd_method is not None else "n/a",
                    "beat_aggregation": stat,
                    "beat_period_control": "residualized" if regressed else "native",
                    "kw_p": tests["kw_p"],
                    "pooled_p": tests["pooled_p"],
                    "cliffs_delta": tests["pooled_delta"],
                    "n_B1": tests["n"]["baseline1"],
                    "n_Flicker": tests["n"]["flicker"],
                    "n_B2": tests["n"]["baseline2"],
                }
            )

        for metric in CONFUND_EIGHT:
            for svd_method in SVD_METHODS:
                for stat in ("median", "mean"):
                    for regressed in (False, True):
                        add_row(metric, svd_method, stat, regressed)

        for metric in CONFUND_SIX:
            for regressed in (False, True):
                add_row(metric, "joint", "n/a", regressed)
            for stat in ("median", "mean"):
                for regressed in (False, True):
                    add_row(metric, "per_beat", stat, regressed)

        counts = pd.Series([r["metric"] for r in grid_rows]).value_counts()
        for row in grid_rows:
            row["n_combinations"] = int(counts[row["metric"]])

        column_order = [
            "endpoint",
            "metric",
            "svd_method",
            "beat_aggregation",
            "beat_period_control",
            "n_combinations",
            "kw_p",
            "pooled_p",
            "cliffs_delta",
            "n_B1",
            "n_Flicker",
            "n_B2",
        ]
        return pd.DataFrame(grid_rows)[column_order]

    @staticmethod
    def _combination_retained(rows: list[dict]) -> bool:
        """True when every row has pooled p < 0.05 and one Cliff's-delta sign."""
        if not rows:
            return False
        ps = [float(r["pooled_p"]) for r in rows]
        deltas = [float(r["cliffs_delta"]) for r in rows]
        n_significant = sum(1 for p in ps if np.isfinite(p) and p < 0.05)
        signed = [d for d in deltas if np.isfinite(d)]
        consistent = bool(signed) and (
            all(d > 0 for d in signed) or all(d < 0 for d in signed)
        )
        return bool(n_significant == len(rows) and consistent)

    @classmethod
    def retain_flags_by_metric(cls, confounds_df: pd.DataFrame) -> dict[str, dict]:
        """Per-metric combination counts and aggregation / period / overall retain."""
        out: dict[str, dict] = {}
        if confounds_df is None or confounds_df.empty:
            return out
        for metric, group in confounds_df.groupby("metric", sort=False):
            rows = group.to_dict("records")
            agg_rows = [
                r
                for r in rows
                if r["beat_period_control"] == "native"
                and r["beat_aggregation"] in ("median", "mean")
            ]
            period_rows = [
                r for r in rows if r["beat_aggregation"] in ("median", "n/a")
            ]
            ps = [float(r["pooled_p"]) for r in rows]
            out[str(metric)] = {
                "n_combinations": len(rows),
                "n_significant": int(
                    sum(1 for p in ps if np.isfinite(p) and p < 0.05)
                ),
                "aggregation_retained": cls._combination_retained(agg_rows),
                "beat_period_retained": cls._combination_retained(period_rows),
                "retained": cls._combination_retained(rows),
            }
        return out


def _column_dtype(name: str, series: pd.Series):
    """HDF5-friendly dtype for one DataFrame column."""
    if name in _BOOL_COLUMNS or pd.api.types.is_bool_dtype(series):
        return np.bool_
    if name in _INT_COLUMNS or (
        pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series)
    ):
        return np.int64
    if pd.api.types.is_float_dtype(series):
        return np.float64
    return UTF8_STRING_DTYPE


def dataframe_to_compound(df: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame to a numpy structured array for one HDF5 dataset."""
    dtype = [(str(col), _column_dtype(str(col), df[col])) for col in df.columns]
    rec = np.empty(len(df), dtype=dtype)
    for col, field_dtype in dtype:
        series = df[col]
        kind = np.dtype(field_dtype).kind
        if kind == "b":
            rec[col] = series.fillna(False).astype(bool).to_numpy()
        elif kind in "iu":
            rec[col] = (
                pd.to_numeric(series, errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
        elif kind == "f":
            rec[col] = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=np.float64
            )
        else:
            rec[col] = series.fillna("").astype(str).to_numpy()
    return rec


def write_compound_dataset(group: h5py.Group, name: str, df: pd.DataFrame) -> None:
    """Write ``df`` as a single compound dataset ``name`` (not a column group)."""
    if name in group:
        del group[name]
    rec = dataframe_to_compound(df)
    if rec.size == 0:
        group.create_dataset(name, shape=(0,), dtype=rec.dtype)
    else:
        group.create_dataset(name, data=rec)


def dictionary_json() -> str:
    """Pretty JSON for the root ``dictionary`` dataset; must round-trip."""
    text = json.dumps(COHORT_DICTIONARY, indent=2, ensure_ascii=False)
    parsed = json.loads(text)
    if parsed != COHORT_DICTIONARY:
        raise ValueError("cohort dictionary JSON did not round-trip")
    return text


def validate_stats_confounds(
    stats_df: pd.DataFrame, confounds_df: pd.DataFrame
) -> None:
    """Refuse internally inconsistent stats / confounds tables."""
    values = COHORT_DICTIONARY["values"]
    coded = {
        "svd_method": confounds_df["svd_method"],
        "beat_aggregation": confounds_df["beat_aggregation"],
        "beat_period_control": confounds_df["beat_period_control"],
        "endpoint": pd.concat(
            [stats_df["endpoint"], confounds_df["endpoint"]], ignore_index=True
        ),
    }
    for field, series in coded.items():
        allowed = values[field]
        for value in series.astype(str).unique():
            if value not in allowed:
                raise ValueError(
                    f"coded value {field}={value!r} is missing from dictionary"
                )

    stats_endpoints = set(stats_df["endpoint"].astype(str))
    conf_endpoints = set(confounds_df["endpoint"].astype(str))
    missing = stats_endpoints - conf_endpoints
    if missing:
        raise ValueError(
            f"stats endpoints missing from confounds: {sorted(missing)}"
        )
    for _, row in stats_df.iterrows():
        metric = str(row["metric"])
        n_declared = int(row["n_combinations"])
        n_rows = int((confounds_df["metric"] == metric).sum())
        if n_declared != n_rows:
            raise ValueError(
                f"{row['endpoint']}: n_combinations={n_declared} but "
                f"confounds has {n_rows} rows"
            )


def _epoch_counts(acquisitions_df: pd.DataFrame) -> dict[str, int]:
    """Finite-row counts per flicker epoch from the acquisitions table."""
    counts = {"n_B1": 0, "n_Flicker": 0, "n_B2": 0}
    if acquisitions_df is None or acquisitions_df.empty or "epoch" not in acquisitions_df:
        return counts
    value_counts = acquisitions_df["epoch"].astype(str).value_counts()
    counts["n_B1"] = int(value_counts.get(EPOCH_SHORT["baseline1"], 0))
    counts["n_Flicker"] = int(value_counts.get(EPOCH_SHORT["flicker"], 0))
    counts["n_B2"] = int(value_counts.get(EPOCH_SHORT["baseline2"], 0))
    return counts


def write_cohort_h5(
    output_path: Path,
    *,
    acquisitions_by_vessel: dict[str, pd.DataFrame],
    beats_by_vessel: dict[str, pd.DataFrame],
    group_order: list[str],
    patient_id: str | None = None,
    flicker_protocol: bool = False,
    source_files: list[str] | None = None,
    stats_by_vessel: dict[str, pd.DataFrame] | None = None,
    confounds_by_vessel: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Write acquisitions/beats/stats/confounds as compound datasets.

    Layout under ``/AngioEye/Postprocessing/lowrank_waveform_cohort/``:

    * root attributes — provenance
    * ``dictionary`` — UTF-8 JSON of column / value meanings
    * ``{vessel}/acquisitions``, ``{vessel}/beats`` — compound tables
    * ``{vessel}/stats``, ``{vessel}/confounds`` — flicker triad only
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_by_vessel = stats_by_vessel or {}
    confounds_by_vessel = confounds_by_vessel or {}
    dict_text = dictionary_json()
    for vessel, stats_df in stats_by_vessel.items():
        if vessel not in confounds_by_vessel:
            raise ValueError(f"{vessel}: stats present but confounds missing")
        validate_stats_confounds(stats_df, confounds_by_vessel[vessel])

    if output_path.exists():
        output_path.unlink()

    n_acquisitions = (
        len(source_files)
        if source_files is not None
        else int(sum(len(df) for df in acquisitions_by_vessel.values()))
    )
    primary = None
    if "artery" in acquisitions_by_vessel:
        primary = acquisitions_by_vessel["artery"]
    elif acquisitions_by_vessel:
        primary = next(iter(acquisitions_by_vessel.values()))
    root_counts = _epoch_counts(primary if primary is not None else pd.DataFrame())

    with open_h5(output_path, "w") as handle:
        root = handle.require_group(COHORT_H5_ROOT)
        set_attr_safe(
            root, "subject", "UNSET" if not patient_id else str(patient_id)
        )
        set_attr_safe(
            root,
            "protocol",
            "Baseline1 / Flicker / Baseline2" if flicker_protocol else "custom",
        )
        set_attr_safe(root, "flicker_protocol", bool(flicker_protocol))
        set_attr_safe(root, "n_acquisitions", int(n_acquisitions))
        set_attr_safe(root, "n_baseline1", int(root_counts["n_B1"]))
        set_attr_safe(root, "n_flicker", int(root_counts["n_Flicker"]))
        set_attr_safe(root, "n_baseline2", int(root_counts["n_B2"]))
        set_attr_safe(root, "svd_methods", list(SVD_METHODS))
        set_attr_safe(root, "beat_aggregations", ["median", "mean"])
        set_attr_safe(root, "group_order", list(group_order))
        set_attr_safe(root, "source_files", list(source_files or []))
        set_attr_safe(
            root,
            "generated",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        write_value_dataset(root, "dictionary", dict_text)

        for vessel, acquisitions_df in acquisitions_by_vessel.items():
            vessel_group = root.require_group(str(vessel))
            counts = _epoch_counts(acquisitions_df)
            set_attr_safe(vessel_group, "n_B1", counts["n_B1"])
            set_attr_safe(vessel_group, "n_Flicker", counts["n_Flicker"])
            set_attr_safe(vessel_group, "n_B2", counts["n_B2"])
            write_compound_dataset(vessel_group, "acquisitions", acquisitions_df)
            write_compound_dataset(
                vessel_group, "beats", beats_by_vessel.get(vessel, pd.DataFrame())
            )
            stats_df = stats_by_vessel.get(vessel)
            if stats_df is not None and not stats_df.empty:
                write_compound_dataset(vessel_group, "stats", stats_df)
            confounds_df = confounds_by_vessel.get(vessel)
            if confounds_df is not None and not confounds_df.empty:
                write_compound_dataset(vessel_group, "confounds", confounds_df)

    return output_path


def build_stats_tables(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """``stats`` / ``confounds`` frames for the flicker triad (else empty)."""
    stats_by_vessel: dict[str, pd.DataFrame] = {}
    confounds_by_vessel: dict[str, pd.DataFrame] = {}
    if not collection.flicker_protocol:
        return stats_by_vessel, confounds_by_vessel

    for vessel, acquisitions_df in collection.points_by_vessel.items():
        grid = confounds.build_grid(
            acqs_by_canonical_epochs(collection.acqs_by_vessel[vessel])
        )
        stats_by_vessel[vessel] = LowRankWaveformStatistics.build_stats_table(
            acquisitions_df, grid
        )
        confounds_by_vessel[vessel] = grid
    return stats_by_vessel, confounds_by_vessel


def write_stats_h5(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
    output_dir: Path,
) -> Path:
    """Pack ``collection`` into ``{patient_id_}lowrank_cohort.h5`` under output_dir."""
    from postprocess.lowrank_waveform_cohort import prefixed_filename

    stats_by_vessel, confounds_by_vessel = build_stats_tables(collection, confounds)
    return write_cohort_h5(
        Path(output_dir)
        / prefixed_filename(COHORT_H5_BASENAME, collection.patient_id),
        acquisitions_by_vessel=collection.points_by_vessel,
        beats_by_vessel=collection.beats_by_vessel,
        group_order=collection.group_order,
        patient_id=collection.patient_id,
        flicker_protocol=collection.flicker_protocol,
        source_files=collection.source_files,
        stats_by_vessel=stats_by_vessel,
        confounds_by_vessel=confounds_by_vessel,
    )


def _run_on_paths(
    h5_paths: list[Path],
    cohort_root: Path,
    output_dir: Path,
    *,
    veins: bool,
) -> Path:
    """Collect packed H5s, build tables/grid, and write ``lowrank_cohort.h5``."""
    if not h5_paths:
        raise ValueError(
            "No packed AngioEye result H5 files with low-rank metrics were "
            "found. Run lowrank_waveform_decomposition first so result H5s "
            "contain the metrics."
        )
    from postprocess.lowrank_waveform_cohort import collect_payload

    collection = collect_payload(h5_paths, cohort_root, veins=bool(veins))
    confounds = LowRankWaveformConfounds()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_stats_h5(collection, confounds, output_dir)


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = False,
    result_h5_paths: Iterable[Path] | None = None,
) -> Path:
    """Build ``lowrank_cohort.h5`` from packed AngioEye result H5s.

    Accepts a cohort folder, a ZIP of that tree, or an explicit
    ``result_h5_paths`` list. Writes under ``output_dir``. Joint and
    per-beat SVD are both included; ``veins`` selects the vessel set
    for both representations.
    """
    from pipelines.lowrank_waveform_decomposition import (
        find_lowrank_result_h5s,
        result_h5_has_lowrank,
    )
    from postprocess.lowrank_waveform_cohort import (
        _cohort_root_from_paths,
        resolve_cohort_root,
    )

    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if result_h5_paths is not None:
        h5_paths = [Path(p) for p in result_h5_paths if result_h5_has_lowrank(p)]
        preferred = input_path if input_path.is_dir() else None
        cohort_root = _cohort_root_from_paths(h5_paths, preferred)
        return _run_on_paths(
            h5_paths,
            cohort_root,
            output_dir,
            veins=veins,
        )

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with extracted_zip_tree(input_path) as extracted_root:
            cohort_root = resolve_cohort_root(extracted_root)
            h5_paths = find_lowrank_result_h5s(cohort_root)
            return _run_on_paths(
                h5_paths,
                cohort_root,
                output_dir,
                veins=veins,
            )

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        h5_paths = find_lowrank_result_h5s(cohort_root)
        return _run_on_paths(
            h5_paths,
            cohort_root,
            output_dir,
            veins=veins,
        )

    raise ValueError(
        f"Input must be a cohort folder or .zip archive, got: {input_path}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, write ``lowrank_cohort.h5``, print the path."""
    parser = argparse.ArgumentParser(
        description=(
            "Build lowrank_cohort.h5 (acquisitions, beats, stats, confounds) "
            "from packed AngioEye low-rank result H5s. Does not write Figs 5--7."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Cohort folder or ZIP containing packed *_AE.h5 / result H5s.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Directory for lowrank_cohort.h5 (default: current directory).",
    )
    parser.add_argument(
        "--veins",
        action="store_true",
        help="Include the vein compartment for joint and per-beat SVD (default: artery only).",
    )
    args = parser.parse_args(argv)
    path = run(
        args.input,
        args.output,
        veins=args.veins,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
