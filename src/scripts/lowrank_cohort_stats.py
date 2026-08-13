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
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from input_output.archive_io import extracted_zip_tree  # noqa: E402
from input_output.hdf5_io import open_h5, write_value_dataset  # noqa: E402
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT  # noqa: E402
from pipelines.lowrank_waveform_decomposition import (  # noqa: E402
    aggregate_beatwise,
    aggregate_rho,
    normalize_svd_method,
)
from postprocess.lowrank_waveform_cohort import (  # noqa: E402
    EPOCH_ORDER,
    epoch_key_from_folder,
)

if TYPE_CHECKING:
    from postprocess.lowrank_waveform_cohort import CohortCollection


COHORT_H5_ROOT = f"{ANGIOEYE_POSTPROCESS_ROOT}/lowrank_waveform_cohort"
COHORT_H5_BASENAME = "lowrank_cohort.h5"


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


class LowRankWaveformStatistics:
    """Sec. V.A nonparametric tests and Sec. V.C endpoint / Table I frames."""

    TABLE_METRICS = [
        "A1",
        "A2",
        "TPR",
        "R1",
        "R2",
        "rho1",
        "rho2",
        "mpr",
        "effective_rank",
        "participation_ratio",
    ]

    # Article Table I: ten predefined endpoints (Holm family).
    # ``metric`` is the points-table column; ``label`` is the published symbol.
    TABLE1_METRICS = (
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
    TABLE1_EXPLORATORY_METRICS: tuple[tuple[str, str], ...] = ()

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
        """Holm–Bonferroni adjusted p-values, same order as the input.

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
        """Kruskal–Wallis, pairwise MWU+Holm, and pooled Baseline-vs-Flicker tests.

        ``epoch_values`` maps baseline1/flicker/baseline2 to acquisition-level
        dots. Missing tests become NaN rather than raising.
        """
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
        """One row per ``TABLE_METRICS``: per-epoch median/SD/n plus KW/Holm/δ."""
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
                    "cliffs_delta_F_vs_B1": cls.cliffs_delta(flc, b1c),
                    "cliffs_delta_F_vs_B2": cls.cliffs_delta(flc, b2c),
                    "cliffs_delta_F_vs_pooled_baseline": cls.cliffs_delta(
                        flc, pooled_baseline
                    ),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def format_median_iqr(vals: np.ndarray) -> str:
        """``median [IQR]`` with 3 significant figures (paper Table I style)."""
        vals = LowRankWaveformStatistics.clean(vals)
        if vals.size == 0:
            return "--"
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        return f"{med:.3g} [{iqr:.3g}]"

    @staticmethod
    def format_p_scientific(p: float) -> str:
        """Scientific / fixed formatting matching the paper's raw-p / p_H cells."""
        if not np.isfinite(p):
            return "--"
        if p >= 1e-2:
            return f"{p:.3f}"
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.2f}e{exp}"

    @classmethod
    def build_table1_pooled_comparison(
        cls, vessel: str, points_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Article Table I: pooled Baseline (B1+B2) vs Flicker for one vessel.

        Family endpoints get Holm-adjusted p; exploratory rows keep raw p only.
        """
        family = list(cls.TABLE1_METRICS)
        exploratory = list(cls.TABLE1_EXPLORATORY_METRICS)

        raw_ps: list[float] = []
        family_rows: list[dict] = []
        for metric, label in family:
            if metric not in points_df.columns:
                base = np.asarray([], dtype=float)
                fl = np.asarray([], dtype=float)
                p, delta = float("nan"), float("nan")
            else:
                base = points_df.loc[
                    points_df["epoch"].isin(["B1", "B2"]), metric
                ].to_numpy(dtype=float)
                fl = points_df.loc[
                    points_df["epoch"] == "Flicker", metric
                ].to_numpy(dtype=float)
                p, delta = cls.pooled_test(points_df, metric)
            raw_ps.append(p)
            base_c, fl_c = cls.clean(base), cls.clean(fl)
            family_rows.append(
                {
                    "vessel": vessel,
                    "endpoint": label,
                    "metric": metric,
                    "is_exploratory": False,
                    "baseline_median": float(np.median(base_c)) if base_c.size else np.nan,
                    "baseline_iqr": (
                        float(np.percentile(base_c, 75) - np.percentile(base_c, 25))
                        if base_c.size
                        else np.nan
                    ),
                    "n_baseline": int(base_c.size),
                    "flicker_median": float(np.median(fl_c)) if fl_c.size else np.nan,
                    "flicker_iqr": (
                        float(np.percentile(fl_c, 75) - np.percentile(fl_c, 25))
                        if fl_c.size
                        else np.nan
                    ),
                    "n_flicker": int(fl_c.size),
                    "baseline_median_iqr": cls.format_median_iqr(base),
                    "flicker_median_iqr": cls.format_median_iqr(fl),
                    "cliffs_delta": delta,
                    "raw_p": p,
                }
            )

        holm_ps = cls.holm_adjust(raw_ps)
        for row, ph in zip(family_rows, holm_ps):
            row["p_holm"] = ph
            row["raw_p_fmt"] = cls.format_p_scientific(row["raw_p"])
            row["p_holm_fmt"] = cls.format_p_scientific(ph)

        rows = family_rows
        for metric, label in exploratory:
            if metric not in points_df.columns:
                base = np.asarray([], dtype=float)
                fl = np.asarray([], dtype=float)
                p, delta = float("nan"), float("nan")
            else:
                base = points_df.loc[
                    points_df["epoch"].isin(["B1", "B2"]), metric
                ].to_numpy(dtype=float)
                fl = points_df.loc[
                    points_df["epoch"] == "Flicker", metric
                ].to_numpy(dtype=float)
                p, delta = cls.pooled_test(points_df, metric)
            base_c, fl_c = cls.clean(base), cls.clean(fl)
            rows.append(
                {
                    "vessel": vessel,
                    "endpoint": label,
                    "metric": metric,
                    "is_exploratory": True,
                    "baseline_median": float(np.median(base_c)) if base_c.size else np.nan,
                    "baseline_iqr": (
                        float(np.percentile(base_c, 75) - np.percentile(base_c, 25))
                        if base_c.size
                        else np.nan
                    ),
                    "n_baseline": int(base_c.size),
                    "flicker_median": float(np.median(fl_c)) if fl_c.size else np.nan,
                    "flicker_iqr": (
                        float(np.percentile(fl_c, 75) - np.percentile(fl_c, 25))
                        if fl_c.size
                        else np.nan
                    ),
                    "n_flicker": int(fl_c.size),
                    "baseline_median_iqr": cls.format_median_iqr(base),
                    "flicker_median_iqr": cls.format_median_iqr(fl),
                    "cliffs_delta": delta,
                    "raw_p": p,
                    "p_holm": np.nan,
                    "raw_p_fmt": cls.format_p_scientific(p),
                    "p_holm_fmt": "--",
                }
            )

        column_order = [
            "vessel",
            "endpoint",
            "metric",
            "is_exploratory",
            "baseline_median_iqr",
            "flicker_median_iqr",
            "cliffs_delta",
            "raw_p",
            "p_holm",
            "raw_p_fmt",
            "p_holm_fmt",
            "baseline_median",
            "baseline_iqr",
            "n_baseline",
            "flicker_median",
            "flicker_iqr",
            "n_flicker",
        ]
        return pd.DataFrame(rows)[column_order]

    @staticmethod
    def pooled_test(
        df: pd.DataFrame,
        metric: str,
        branch_col: str = "epoch",
        flicker_val: str = "Flicker",
        baseline_vals: tuple[str, str] = ("B1", "B2"),
    ) -> tuple[float, float]:
        """Mann-Whitney p and Cliff's δ for Flicker vs pooled Baseline (B1∪B2)."""
        baseline = df.loc[df[branch_col].isin(baseline_vals), metric].to_numpy(dtype=float)
        flicker = df.loc[df[branch_col] == flicker_val, metric].to_numpy(dtype=float)
        baseline = baseline[np.isfinite(baseline)]
        flicker = flicker[np.isfinite(flicker)]
        if baseline.size < 2 or flicker.size < 2:
            return float("nan"), float("nan")
        p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
        return p, LowRankWaveformStatistics.cliffs_delta(flicker, baseline)


class LowRankWaveformConfounds:
    """Sec. V.B confound-control grid (analysis-choice sweep + verdicts)."""

    METRICS_SVD = ("A1", "A2", "rho1", "rho2")

    def __init__(self, *, svd_method: str = "joint") -> None:
        """``svd_method`` is joint vs per-beat, matching EyeFlow's pipeline options."""
        self.svd_method = normalize_svd_method(svd_method)

    @staticmethod
    def residualize_against_beat_period(
        epoch_values: dict[str, np.ndarray],
        epoch_beat_periods: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Replace each metric value by its residual from a global ``d = α + βT`` fit.

        Needs at least 3 finite (value, period) pairs; otherwise returns the inputs.
        """
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
        """One acquisition-level value per acquisition for ``metric``.

        ``stat`` is beat aggregation (median/mean); ``svd_method`` selects joint
        vs per-beat arrays. TPR and MPR have no SVD axis of their own.
        """
        out = []
        for a in acqs:
            beatwise = a.get("beatwise") or {}
            per_beat = a.get("per_beat_svd") or {}
            if metric == "TPR":
                key = "TPR_b_pb" if svd_method == "per_beat" else "TPR_b"
                src = per_beat if svd_method == "per_beat" else beatwise
                out.append(aggregate_beatwise(src.get(key, []), stat))
            elif metric == "mpr":
                key = "mpr_b_pb" if svd_method == "per_beat" else "mpr_b"
                src = per_beat if svd_method == "per_beat" else beatwise
                out.append(aggregate_beatwise(src.get(key, []), stat))
            elif metric in ("A1", "A2"):
                if svd_method == "joint":
                    arr = beatwise.get(f"{metric}_b", [])
                else:
                    arr = per_beat.get(f"{metric}_b_pb", [])
                out.append(aggregate_beatwise(arr, stat))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "joint":
                    R_b = beatwise.get(f"R{m}_b", [])
                    tpr_b = beatwise.get("TPR_b", [])
                else:
                    R_b = per_beat.get(f"R{m}_b_pb", [])
                    tpr_b = per_beat.get("TPR_b_pb", [])
                out.append(aggregate_rho(R_b, tpr_b, stat))
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
        vessel: str,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """Sweep metric × SVD × aggregation × period-control; return grid + verdicts.

        A metric is retained when every combination has pooled p < 0.05 and a
        consistent Cliff's-delta sign.
        """
        periods = self.epoch_beat_periods(acqs_by_epoch)
        grid_rows: list[dict] = []
        verdict_source: dict[str, list[dict]] = {}

        def run_combo(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> dict:
            """Epoch-group tests for one grid cell, residualizing when asked."""
            values = self.epoch_dots(acqs_by_epoch, metric, svd_method, stat)
            if regressed:
                values = self.residualize_against_beat_period(values, periods)
            return LowRankWaveformStatistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Append one flattened grid row and stash it for the verdict summary."""
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
            for stat in ("median", "mean"):
                for regressed in (False, True):
                    add_row(metric, self.svd_method, stat, regressed)

        for stat in ("median", "mean"):
            for regressed in (False, True):
                add_row("TPR", self.svd_method, stat, regressed)
                add_row("mpr", self.svd_method, stat, regressed)

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


def write_dataframe_to_h5_group(
    parent: h5py.Group, name: str, df: pd.DataFrame
) -> h5py.Group:
    """Write a DataFrame as a named group with one dataset per column."""
    group = parent.require_group(name)
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            write_value_dataset(group, str(column), series.to_numpy(dtype=bool))
        elif pd.api.types.is_integer_dtype(series):
            write_value_dataset(
                group, str(column), series.to_numpy(dtype=np.int64)
            )
        elif pd.api.types.is_float_dtype(series):
            write_value_dataset(
                group, str(column), series.to_numpy(dtype=np.float64)
            )
        else:
            write_value_dataset(
                group,
                str(column),
                series.fillna("").astype(str).tolist(),
            )
    return group


def write_cohort_h5(
    output_path: Path,
    *,
    points_by_vessel: dict[str, pd.DataFrame],
    beats_by_vessel: dict[str, pd.DataFrame],
    group_order: list[str],
    patient_id: str | None = None,
    flicker_protocol: bool = False,
    source_files: list[str] | None = None,
    tables_by_vessel: dict[str, dict[str, pd.DataFrame]] | None = None,
    confound_by_vessel: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> Path:
    """Write cohort points/beats/tables/confounds into one HDF5 product.

    Layout under ``/AngioEye/Postprocessing/lowrank_waveform_cohort/``:

    * ``meta/`` — group_order, patient_id, flicker_protocol, n_acquisitions,
      source_files
    * ``{vessel}/points/``, ``{vessel}/beats/`` — columnar tables
    * ``{vessel}/tables/...`` — endpoint / Table I (flicker triad only)
    * ``{vessel}/confound/{grid,verdict}/`` — confound control (flicker triad)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tables_by_vessel = tables_by_vessel or {}
    confound_by_vessel = confound_by_vessel or {}

    with open_h5(output_path, "w") as handle:
        root = handle.require_group(COHORT_H5_ROOT)
        meta = root.require_group("meta")
        write_value_dataset(meta, "group_order", list(group_order))
        write_value_dataset(
            meta, "patient_id", "" if patient_id is None else str(patient_id)
        )
        write_value_dataset(meta, "flicker_protocol", bool(flicker_protocol))
        n_acquisitions = (
            len(source_files)
            if source_files is not None
            else int(sum(len(df) for df in points_by_vessel.values()))
        )
        write_value_dataset(meta, "n_acquisitions", int(n_acquisitions))
        write_value_dataset(meta, "source_files", list(source_files or []))

        for vessel, points_df in points_by_vessel.items():
            vessel_group = root.require_group(str(vessel))
            write_dataframe_to_h5_group(vessel_group, "points", points_df)
            write_dataframe_to_h5_group(
                vessel_group, "beats", beats_by_vessel[vessel]
            )

            vessel_tables = tables_by_vessel.get(vessel)
            if vessel_tables:
                tables = vessel_group.require_group("tables")
                for table_name, table_df in vessel_tables.items():
                    write_dataframe_to_h5_group(tables, table_name, table_df)

            vessel_confound = confound_by_vessel.get(vessel)
            if vessel_confound:
                confound = vessel_group.require_group("confound")
                for name, confound_df in vessel_confound.items():
                    write_dataframe_to_h5_group(confound, name, confound_df)

    return output_path


def build_stats_tables(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]:
    """Endpoint / Table I / confound frames for the flicker triad (else empty)."""
    tables_by_vessel: dict[str, dict[str, pd.DataFrame]] = {}
    confound_by_vessel: dict[str, dict[str, pd.DataFrame]] = {}
    if not collection.flicker_protocol:
        return tables_by_vessel, confound_by_vessel

    for vessel, points_df in collection.points_by_vessel.items():
        tables_by_vessel[vessel] = {
            "endpoint_table": LowRankWaveformStatistics.build_endpoint_table(
                vessel, points_df
            ),
            "table1_pooled_baseline_vs_flicker": (
                LowRankWaveformStatistics.build_table1_pooled_comparison(
                    vessel, points_df
                )
            ),
        }
        grid_rows, verdict_rows = confounds.build_grid(
            vessel, acqs_by_canonical_epochs(collection.acqs_by_vessel[vessel])
        )
        confound_by_vessel[vessel] = {
            "grid": pd.DataFrame(grid_rows),
            "verdict": pd.DataFrame(verdict_rows),
        }
    return tables_by_vessel, confound_by_vessel


def write_stats_h5(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
    output_dir: Path,
) -> Path:
    """Pack ``collection`` into ``{patient_id_}lowrank_cohort.h5`` under output_dir."""
    from postprocess.lowrank_waveform_cohort import prefixed_filename

    tables_by_vessel, confound_by_vessel = build_stats_tables(collection, confounds)
    return write_cohort_h5(
        Path(output_dir)
        / prefixed_filename(COHORT_H5_BASENAME, collection.patient_id),
        points_by_vessel=collection.points_by_vessel,
        beats_by_vessel=collection.beats_by_vessel,
        group_order=collection.group_order,
        patient_id=collection.patient_id,
        flicker_protocol=collection.flicker_protocol,
        source_files=collection.source_files,
        tables_by_vessel=tables_by_vessel,
        confound_by_vessel=confound_by_vessel,
    )


def _run_on_paths(
    h5_paths: list[Path],
    cohort_root: Path,
    output_dir: Path,
    *,
    veins: bool,
    svd_method: str = "joint",
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
    confounds = LowRankWaveformConfounds(svd_method=svd_method)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_stats_h5(collection, confounds, output_dir)


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = False,
    svd_method: str = "joint",
    result_h5_paths: Iterable[Path] | None = None,
) -> Path:
    """Build ``lowrank_cohort.h5`` from packed AngioEye result H5s.

    Accepts a cohort folder, a ZIP of that tree, or an explicit
    ``result_h5_paths`` list. Writes under ``output_dir``.
    """
    from pipelines.lowrank_waveform_decomposition import (
        find_lowrank_result_h5s,
        normalize_svd_method,
        result_h5_has_lowrank,
    )
    from postprocess.lowrank_waveform_cohort import (
        _cohort_root_from_paths,
        resolve_cohort_root,
    )

    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    svd_method = normalize_svd_method(svd_method)

    if result_h5_paths is not None:
        h5_paths = [Path(p) for p in result_h5_paths if result_h5_has_lowrank(p)]
        preferred = input_path if input_path.is_dir() else None
        cohort_root = _cohort_root_from_paths(h5_paths, preferred)
        return _run_on_paths(
            h5_paths,
            cohort_root,
            output_dir,
            veins=veins,
            svd_method=svd_method,
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
                svd_method=svd_method,
            )

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        h5_paths = find_lowrank_result_h5s(cohort_root)
        return _run_on_paths(
            h5_paths,
            cohort_root,
            output_dir,
            veins=veins,
            svd_method=svd_method,
        )

    raise ValueError(
        f"Input must be a cohort folder or .zip archive, got: {input_path}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, write ``lowrank_cohort.h5``, print the path."""
    parser = argparse.ArgumentParser(
        description=(
            "Build lowrank_cohort.h5 (points, beats, Table I, confound grid) "
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
        help="Include the vein compartment (default: artery only).",
    )
    parser.add_argument(
        "--svd-method",
        choices=("joint", "per_beat"),
        default="joint",
        help="Which SVD representation to use for confound dots (default: joint).",
    )
    args = parser.parse_args(argv)
    path = run(
        args.input,
        args.output,
        veins=args.veins,
        svd_method=args.svd_method,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
