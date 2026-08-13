"""Cohort low-rank Figs 4--8 and ``lowrank_cohort.h5`` from packed result H5s.

Joint-SVD Figs 4--8 keep the article acquisition scalars; a parallel
``*_pb.png`` set uses ``median_b`` of packed per-beat SVD endpoints (ρ is
``median_b(R)/median_b(R0)``). ``T``, ``μ``, R0, and MPR stay joint.
Fig. 4 joint and Fig. 5 ratio plots use packed joint singular values. Stats /
confounds live in the same ``lowrank_cohort.h5``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import kruskal, linregress, mannwhitneyu, spearmanr

from input_output import cohort_results_dir
from input_output.archive_io import extracted_zip_tree
from input_output.hdf5_io import (
    UTF8_STRING_DTYPE,
    open_h5,
    set_attr_safe,
    write_value_dataset,
)
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT

from pipelines.lowrank_waveform_decomposition import (
    SPECTRUM_N_MODES,
    SVD_METHODS,
    aggregate_beatwise,
    aggregate_rho,
    coerce_beat_spectra,
    enabled_vessels,
    finite_std,
    find_lowrank_result_h5s,
    is_usable_beat_spectra,
    load_acquisition_from_result_h5,
    mean_pm_std,
    result_h5_has_lowrank,
)

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


# =====================================================================
# Group / epoch identity
# =====================================================================

EPOCH_ORDER = ("baseline1", "flicker", "baseline2")
EPOCH_SHORT = {
    "baseline1": "B1",
    "flicker": "Flicker",
    "baseline2": "B2",
}
EPOCH_SHORT_TO_KEY = {short: key for key, short in EPOCH_SHORT.items()}
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
_LEADING_INDEX_RE = re.compile(r"^\d+[\s_\-]*")
_PATIENT_ID_RE = re.compile(r"^(\d{6})")
_FIG_LABEL_SIZE = 14
_FIG_TICK_SIZE = 12
_RNG = np.random.default_rng(0)


def _alnum_token_key(text: str) -> tuple:
    """Natural alphanumeric sort key (``2_`` before ``10_``)."""
    tokens: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if part == "":
            continue
        tokens.append((0, int(part)) if part.isdigit() else (1, part.lower()))
    return tuple(tokens)


def epoch_key_from_folder(name: str) -> str | None:
    """Map a folder name to baseline1/flicker/baseline2 when recognizable."""
    lower = name.strip().lower()
    for cand in (lower, _LEADING_INDEX_RE.sub("", lower)):
        if cand and cand in EPOCH_ALIASES:
            return EPOCH_ALIASES[cand]
    return None


def group_display_label(group: str) -> str:
    """B1 / Flicker / B2 when the folder is a flicker epoch, else the folder name."""
    key = epoch_key_from_folder(group)
    return EPOCH_SHORT[key] if key is not None else group


def _group_sort_key(group: str | None) -> tuple:
    """Alphanumeric folder order; ungrouped (None) last."""
    return (1, ()) if group is None else (0, _alnum_token_key(group))


def _row_sort_key(epoch_label: str) -> tuple:
    """Sort points/beats rows: B1 → Flicker → B2, then other labels alphanumerically."""
    short_order = tuple(EPOCH_SHORT.values())
    if epoch_label in short_order:
        return (0, short_order.index(epoch_label))
    return (1, _alnum_token_key(epoch_label))


def ordered_groups(groups: Iterable[str | None]) -> list[str]:
    """Unique named groups in figure x-axis order.

    Bare flicker folders (no leading index) keep B1→Flicker→B2. Indexed names
    such as ``1_baseline1`` follow alphanumeric order of the folder strings.
    """
    ordered = sorted({g for g in groups if g is not None}, key=_group_sort_key)
    if set(ordered) == set(EPOCH_ORDER):
        return list(EPOCH_ORDER)
    keys = [epoch_key_from_folder(g) for g in ordered]
    if set(keys) == set(EPOCH_ORDER) and all(
        key is not None and not _LEADING_INDEX_RE.match(g.strip())
        for g, key in zip(ordered, keys)
    ):
        by_key = {key: g for g, key in zip(ordered, keys)}
        return [by_key[epoch] for epoch in EPOCH_ORDER]
    return ordered


def is_flicker_triad(groups: Iterable[str | None]) -> bool:
    """True when folders encode baseline1 + flicker + baseline2."""
    keys = {epoch_key_from_folder(g) for g in groups if g is not None}
    keys.discard(None)
    return set(EPOCH_ORDER).issubset(keys)


def extract_patient_id(input_path: Path | str) -> str | None:
    """6-digit patient ID prefixing the ZIP/folder name, else None."""
    match = _PATIENT_ID_RE.match(Path(input_path).name)
    return match.group(1) if match else None


def prefixed_filename(filename: str, patient_id: str | None) -> str:
    """Prepend ``{patient_id}_`` when a patient ID is known."""
    return f"{patient_id}_{filename}" if patient_id else filename


def classify_cohort(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[list[tuple[str | None, Path]], list[str]]:
    """Group each H5 by its top-level folder under ``input_root``.

    Returns ``(records, group_order)``. Files sitting directly under the root
    get ``group=None``. Files outside the root are skipped.
    """
    records: list[tuple[str | None, Path]] = []
    input_root = Path(input_root)
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        try:
            rel = h5_path.relative_to(input_root)
        except ValueError:
            continue
        group = rel.parts[0] if len(rel.parts) > 1 else None
        records.append((group, h5_path))
    records.sort(key=lambda r: (_group_sort_key(r[0]), r[1].name))
    return records, ordered_groups(group for group, _ in records)


def resolve_cohort_root(path: Path | str) -> Path:
    """Directory that directly contains the split folders.

    Unwraps a single child folder (typical after extracting a ZIP).
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Cohort root is not a directory: {path}")
    children = sorted(
        child
        for child in path.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and child.name != "__MACOSX"
    )
    return children[0] if len(children) == 1 else path


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


# =====================================================================
# Cohort H5: stats / confounds
# =====================================================================

COHORT_H5_ROOT = f"{ANGIOEYE_POSTPROCESS_ROOT}/lowrank_waveform_cohort"
COHORT_H5_BASENAME = "lowrank_cohort.h5"

# Internal acquisitions-table column, published symbol.
CANONICAL_ENDPOINTS = (
    ("A1", "A1"),
    ("A2", "A2"),
    ("R0", "R0"),
    ("R1", "R1"),
    ("R2", "R2"),
    ("rho1", "rho1"),
    ("rho2", "rho2"),
    ("MPR", "MPR"),
    ("Reff", "Reff"),
    ("PR", "PR"),
)
ENDPOINT_BY_METRIC = dict(CANONICAL_ENDPOINTS)

CONFUND_EIGHT = ("A1", "A2", "R1", "R2", "rho1", "rho2", "R0", "MPR")
CONFUND_SIX = ("Reff", "PR")

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
            "R0": "Total pulsatile RMS.",
            "R1": "Mode-1 residual RMS.",
            "R2": "Mode-2 residual RMS.",
            "rho1": "Mode-1 residual ratio R1/R0.",
            "rho2": "Mode-2 residual ratio R2/R0.",
            "MPR": "Mean-to-pulsatile ratio.",
            "Reff": "Effective rank.",
            "PR": "Participation ratio.",
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

    @classmethod
    def build_stats_tables(
        cls,
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
            stats_by_vessel[vessel] = cls.build_stats_table(acquisitions_df, grid)
            confounds_by_vessel[vessel] = grid
        return stats_by_vessel, confounds_by_vessel


class LowRankWaveformConfounds:
    """Confound-control grid: analysis-choice sweep plus retain flags."""

    @staticmethod
    def _first_finite_array(mapping: dict, *keys: str) -> np.ndarray:
        """First array among ``keys`` that contains a finite value."""
        for key in keys:
            if key not in mapping:
                continue
            arr = np.asarray(mapping[key], dtype=float)
            if np.isfinite(arr).any():
                return arr
        return np.asarray([], dtype=float)

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
            if metric == "R0":
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, "R0")
                else:
                    arr = self._first_finite_array(beatwise, "R0_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric == "MPR":
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, "MPR")
                else:
                    arr = self._first_finite_array(beatwise, "MPR_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("A1", "A2", "R1", "R2"):
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, metric)
                else:
                    arr = self._first_finite_array(beatwise, f"{metric}_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "per_beat":
                    R_b = self._first_finite_array(per_beat, f"R{m}")
                    tpr_b = self._first_finite_array(per_beat, "R0")
                else:
                    R_b = self._first_finite_array(beatwise, f"R{m}_b")
                    tpr_b = self._first_finite_array(beatwise, "R0_b")
                out.append(
                    aggregate_rho(R_b, tpr_b, stat if stat != "n/a" else "median")
                )
            elif metric in ("Reff", "PR"):
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, metric)
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


# =====================================================================
# Figure annotations (p / Cliff's δ on Figs 6--8)
# =====================================================================

def _pooled_test(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Mann-Whitney p and Cliff's δ for Flicker vs pooled Baseline (B1∪B2)."""
    baseline = LowRankWaveformStatistics.clean(
        df.loc[df["epoch"].isin(["B1", "B2"]), metric]
    )
    flicker = LowRankWaveformStatistics.clean(
        df.loc[df["epoch"] == "Flicker", metric]
    )
    if baseline.size < 2 or flicker.size < 2:
        return float("nan"), float("nan")
    p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
    return p, LowRankWaveformStatistics.cliffs_delta(flicker, baseline)


def _format_p(p: float) -> str:
    """Figure annotation for a p-value."""
    if not np.isfinite(p):
        return "p=n/a"
    return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"


def _format_delta(delta: float) -> str:
    """Figure annotation for Cliff's δ."""
    if not np.isfinite(delta):
        return r"$\delta$=n/a"
    return rf"$\delta$={delta:+.2f}"


# =====================================================================
# Figs 4--8
# =====================================================================

class LowRankWaveformCohortFigures:
    """Article Figs. 4--8, written once per cohort (joint and ``_pb``)."""

    PANEL_SIZE = 2.5
    FLICKER_SHADE = "#add8e6"
    SPECTRUM_BASELINE = ("black", "-", "o")
    SPECTRUM_FLICKER = ("#555555", "--", "s")

    FIG5_PANELS = (
        ("beat_period", r"Beat period" "\n" r"$T$"),
        ("mu", r"Baseline level" "\n" r"$\mu$"),
        ("R0", r"Total Pulsatile RMS" "\n" r"$R_0$"),
        ("MPR", r"Mean-to-pulsatile ratio" "\n" r"MPR"),
    )
    FIG6_PANELS = (
        ("A1", r"Mode-1 amplitude" "\n" r"$A_1$"),
        ("A2", r"Mode-2 amplitude" "\n" r"$A_2$"),
        ("R1", r"Residual RMS" "\n" r"$R_1$"),
        ("R2", r"Residual RMS" "\n" r"$R_2$"),
    )
    FIG7_PANELS = (
        ("rho1", r"Residual ratio" "\n" r"$\rho_1$"),
        ("rho2", r"Residual ratio" "\n" r"$\rho_2$"),
        ("Reff", r"Effective rank" "\n" r"$R_{\mathrm{eff}}$"),
        ("PR", r"Participation ratio" "\n" r"PR"),
    )

    @classmethod
    def plot_all(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_dir: Path,
        group_order: list[str],
        *,
        patient_id: str | None = None,
        beats_by_vessel: dict[str, pd.DataFrame] | None = None,
        points_by_vessel_per_beat: dict[str, pd.DataFrame] | None = None,
    ) -> list[Path]:
        """Write Figs. 4--8 (joint and ``_pb``) for the artery cohort."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grids = (
            ("fig6_nonsvd_endpoints.png", cls.FIG5_PANELS),
            ("fig7_lowrank_endpoints.png", cls.FIG6_PANELS),
            ("fig8_residual_spectrum_endpoints.png", cls.FIG7_PANELS),
        )
        written = [
            cls._plot_endpoint_grid(
                points_by_vessel,
                panels,
                out_dir / prefixed_filename(name, patient_id),
                group_order,
            )
            for name, panels in grids
        ]
        pb_points = points_by_vessel_per_beat or {}
        if _has_per_beat_endpoint_dots(pb_points):
            pb_grids = (
                ("fig6_nonsvd_endpoints_pb.png", cls.FIG5_PANELS),
                ("fig7_lowrank_endpoints_pb.png", cls.FIG6_PANELS),
                ("fig8_residual_spectrum_endpoints_pb.png", cls.FIG7_PANELS),
            )
            written.extend(
                cls._plot_endpoint_grid(
                    pb_points,
                    panels,
                    out_dir / prefixed_filename(name, patient_id),
                    group_order,
                )
                for name, panels in pb_grids
            )
        if _has_spectrum_modes(points_by_vessel.get("artery", pd.DataFrame())):
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename("fig4_variance_fraction.png", patient_id),
                    points_by_vessel,
                    cumulative=False,
                )
            )
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename(
                        "fig4_variance_fraction_cumulative.png", patient_id
                    ),
                    points_by_vessel,
                    cumulative=True,
                )
            )
            if is_flicker_triad(group_order):
                written.append(
                    cls._save_spectrum_ratio(
                        out_dir
                        / prefixed_filename("fig5_spectrum_ratio.png", patient_id),
                        points_by_vessel,
                        cumulative=False,
                    )
                )
                written.append(
                    cls._save_spectrum_ratio(
                        out_dir
                        / prefixed_filename(
                            "fig5_spectrum_ratio_cumulative.png", patient_id
                        ),
                        points_by_vessel,
                        cumulative=True,
                    )
                )
        beats = beats_by_vessel or {}
        pb_spectrum_source = (
            pb_points
            if _has_spectrum_modes(pb_points.get("artery", pd.DataFrame()))
            else beats
        )
        if _has_spectrum_modes(pb_spectrum_source.get("artery", pd.DataFrame())):
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename("fig4_variance_fraction_pb.png", patient_id),
                    pb_spectrum_source,
                    cumulative=False,
                )
            )
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename(
                        "fig4_variance_fraction_cumulative_pb.png", patient_id
                    ),
                    pb_spectrum_source,
                    cumulative=True,
                )
            )
        return sorted(written, key=lambda path: path.name)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9) -> None:
        """Hide the grid and set tick-label size."""
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=tick_size)

    @classmethod
    def _draw_epoch_panel(
        cls, ax, df: pd.DataFrame, metric: str, group_order: list[str]
    ) -> None:
        """One endpoint panel: jittered dots, median±SD, pooled p/δ label."""
        labels = [group_display_label(g) for g in group_order]
        positions = {label: idx for idx, label in enumerate(labels)}
        if is_flicker_triad(group_order):
            for idx, group in enumerate(group_order):
                if epoch_key_from_folder(group) == "flicker":
                    ax.axvspan(
                        idx - 0.5, idx + 0.5, color=cls.FLICKER_SHADE, zorder=0
                    )
                    ax.axvline(
                        idx - 0.5, color="black", linestyle=":", linewidth=1.5, zorder=1
                    )
                    ax.axvline(
                        idx + 0.5, color="black", linestyle=":", linewidth=1.5, zorder=1
                    )
                    break

        for label in labels:
            vals = (
                df.loc[df["epoch"] == label, metric].dropna().to_numpy(dtype=float)
                if metric in df.columns
                else np.asarray([], dtype=float)
            )
            if vals.size == 0:
                continue
            x0 = positions[label]
            jitter = (_RNG.random(vals.size) - 0.5) * 0.16
            med = float(np.nanmedian(vals))
            sd = finite_std(vals)
            ax.errorbar(
                [x0],
                [med],
                yerr=[sd],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                zorder=4,
            )
            ax.scatter(
                x0 + jitter, vals, s=20, color="black", edgecolors="none", zorder=5
            )
            ax.scatter(
                [x0],
                [med],
                s=49,
                facecolors="white",
                edgecolors="black",
                linewidths=1.4,
                zorder=6,
            )

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(list(positions.keys()), fontsize=9)
        ax.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
        cls._style_axes(ax)

        if metric in df.columns and is_flicker_triad(group_order):
            all_vals = df[metric].to_numpy(dtype=float)
            if np.isfinite(all_vals).any():
                p, delta = _pooled_test(df, metric)
                y_min = float(np.nanmin(all_vals))
                y_max = float(np.nanmax(all_vals))
                pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
                ax.text(
                    0.03,
                    0.97,
                    f"{_format_p(p)}\n{_format_delta(delta)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.2,
                    },
                )
                ax.set_ylim(y_min - pad, y_max + 2.2 * pad)

    @classmethod
    def _plot_endpoint_grid(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        panels: tuple[tuple[str, str], ...],
        out_path: Path,
        group_order: list[str],
    ) -> Path:
        """Write a 1×N PNG of arterial endpoint panels."""
        out_path = Path(out_path)
        df = points_by_vessel.get("artery", pd.DataFrame())
        n_cols = len(panels)
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(cls.PANEL_SIZE * n_cols, cls.PANEL_SIZE),
            squeeze=False,
        )
        for col_idx, (metric, title) in enumerate(panels):
            ax = axes[0, col_idx]
            cls._draw_epoch_panel(ax, df, metric, group_order)
            ax.set_box_aspect(1)
            ax.set_title(title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Value", fontsize=11)
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def _draw_spectrum(cls, ax, df: pd.DataFrame, mode_cols: list[str], *, cumulative: bool) -> None:
        """Draw Baseline vs Flicker mean±SD singular-value curves on ``ax``."""
        if not mode_cols or df.empty or "epoch" not in df.columns:
            return
        plot_modes = np.arange(1, len(mode_cols) + 1)
        series = (
            (df["epoch"].isin(["B1", "B2"]), *cls.SPECTRUM_BASELINE),
            (df["epoch"] == "Flicker", *cls.SPECTRUM_FLICKER),
        )
        for mask, color, style, marker in series:
            vals = df.loc[mask, mode_cols].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            if cumulative:
                vals = np.nancumsum(vals, axis=1)
            mean, lo, hi = mean_pm_std(vals, axis=0)
            if not cumulative:
                lo = np.maximum(lo, np.where(mean > 0, mean * 1e-6, 1e-12))
            ax.plot(
                plot_modes,
                mean,
                color=color,
                linestyle=style,
                marker=marker,
                linewidth=1.5,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
            )
            ax.fill_between(plot_modes, lo, hi, color=color, alpha=0.12, linewidth=0)

    @classmethod
    def _save_spectrum(
        cls,
        out_path: Path,
        beats_by_vessel: dict[str, pd.DataFrame] | None,
        *,
        cumulative: bool,
    ) -> Path:
        """Write the per-mode or cumulative λ spectrum PNG."""
        out_path = Path(out_path)
        frames = beats_by_vessel or {}
        beats = frames.get("artery", pd.DataFrame())
        mode_cols = _spectrum_mode_cols(beats)
        df = beats if (not beats.empty and mode_cols) else pd.DataFrame()

        fig_h = 3.0
        fig, axes = plt.subplots(1, 1, figsize=(2.0 * fig_h, fig_h), squeeze=False)
        ax = axes[0, 0]
        cls._draw_spectrum(ax, df, mode_cols, cumulative=cumulative)

        modes = np.arange(1, SPECTRUM_N_MODES + 1)
        ax.set_xticks(modes)
        ax.set_xticklabels([str(m) for m in modes])
        ax.set_xlim(0.5, SPECTRUM_N_MODES + 0.5)
        if not cumulative:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=8)
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _pos: f"{y:g}" if y > 0 else "0")
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$m$", fontsize=_FIG_LABEL_SIZE)
        ax.set_ylabel(
            r"$\sum_{i=1}^{m}\lambda_i$" if cumulative else r"$\lambda_m$",
            fontsize=_FIG_LABEL_SIZE,
        )
        ax.set_box_aspect(0.5)
        cls._style_axes(ax, tick_size=_FIG_TICK_SIZE)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _spectrum_ratio_values(
        df: pd.DataFrame,
        mode_cols: list[str],
        *,
        cumulative: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return modes and percent flicker/baseline spectrum change."""
        modes = np.arange(1, len(mode_cols) + 1)
        if df.empty or not mode_cols or "epoch" not in df.columns:
            return modes, np.full(len(mode_cols), np.nan, dtype=float)
        baseline = df.loc[df["epoch"].isin(["B1", "B2"]), mode_cols].to_numpy(
            dtype=float
        )
        flicker = df.loc[df["epoch"] == "Flicker", mode_cols].to_numpy(dtype=float)
        if baseline.size == 0 or flicker.size == 0:
            return modes, np.full(len(mode_cols), np.nan, dtype=float)
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline_mean = np.nanmean(baseline, axis=0)
                flicker_mean = np.nanmean(flicker, axis=0)
        if cumulative:
            baseline_mean = np.nancumsum(baseline_mean)
            flicker_mean = np.nancumsum(flicker_mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flicker_mean / baseline_mean
        percent_change = 100.0 * (ratio - 1.0)
        percent_change = np.where(np.isfinite(percent_change), percent_change, np.nan)
        return modes, percent_change

    @classmethod
    def _save_spectrum_ratio(
        cls,
        out_path: Path,
        points_by_vessel: dict[str, pd.DataFrame] | None,
        *,
        cumulative: bool,
    ) -> Path:
        """Write Fig. 5 spectrum ratio PNG from acquisition-level singular values."""
        out_path = Path(out_path)
        frames = points_by_vessel or {}
        df = frames.get("artery", pd.DataFrame())
        mode_cols = _spectrum_mode_cols(df)
        modes, percent_change = cls._spectrum_ratio_values(
            df,
            mode_cols,
            cumulative=cumulative,
        )

        fig_h = 3.0
        fig, axes = plt.subplots(1, 1, figsize=(2.0 * fig_h, fig_h), squeeze=False)
        ax = axes[0, 0]
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.1)
        ax.plot(
            modes,
            percent_change,
            color="black",
            linestyle="-",
            marker="o",
            linewidth=1.5,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

        finite = percent_change[np.isfinite(percent_change)]
        if finite.size:
            extent = max(5.0, float(np.nanmax(np.abs(finite))) * 1.2)
            ax.set_ylim(-extent, extent)
        ax.set_xticks(np.arange(1, SPECTRUM_N_MODES + 1))
        ax.set_xticklabels([str(m) for m in range(1, SPECTRUM_N_MODES + 1)])
        ax.set_xlim(0.5, SPECTRUM_N_MODES + 0.5)
        ax.set_xlabel(r"$m$", fontsize=_FIG_LABEL_SIZE)
        ax.set_ylabel(
            (
                r"$100(Q^{\mathrm{cum}}_m-1)$ (%)"
                if cumulative
                else r"$100(Q_m-1)$ (%)"
            ),
            fontsize=_FIG_LABEL_SIZE,
        )
        ax.set_box_aspect(0.5)
        cls._style_axes(ax, tick_size=_FIG_TICK_SIZE)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


# =====================================================================
# Collection from packed result H5s
# =====================================================================

PER_BEAT_POINT_METRICS = (
    "A1",
    "A2",
    "R1",
    "R2",
    "rho1",
    "rho2",
    "Reff",
    "PR",
)


def _spectrum_mode_cols(df: pd.DataFrame) -> list[str]:
    """``mode1``…``modeM`` columns present on ``df``."""
    if df is None or df.empty:
        return []
    return [
        f"mode{i}"
        for i in range(1, SPECTRUM_N_MODES + 1)
        if f"mode{i}" in df.columns
    ]


def _has_spectrum_modes(df: pd.DataFrame) -> bool:
    """True if ``df`` has a finite packed singular-value entry."""
    cols = _spectrum_mode_cols(df)
    if not cols:
        return False
    return bool(np.isfinite(df[cols].to_numpy(dtype=float)).any())


def _has_per_beat_endpoint_dots(points_by_vessel: dict[str, pd.DataFrame]) -> bool:
    """True if any vessel has a finite per-beat SVD acquisition endpoint."""
    for df in points_by_vessel.values():
        if df is None or df.empty:
            continue
        for metric in PER_BEAT_POINT_METRICS:
            if metric not in df.columns:
                continue
            if np.isfinite(df[metric].to_numpy(dtype=float)).any():
                return True
    return False


@dataclass
class CohortCollection:
    """Packed-H5 endpoints grouped for figures or ``lowrank_cohort.h5``."""

    points_by_vessel: dict[str, pd.DataFrame]
    beats_by_vessel: dict[str, pd.DataFrame]
    acqs_by_vessel: dict[str, dict[str, list[dict]]]
    group_order: list[str]
    patient_id: str | None
    flicker_protocol: bool
    source_files: list[str]
    points_by_vessel_per_beat: dict[str, pd.DataFrame]


def _finite_at(arr, index: int) -> float:
    """Finite value at ``index``, else NaN."""
    arr = np.asarray(arr, dtype=float)
    if index < arr.size and np.isfinite(arr[index]):
        return float(arr[index])
    return float("nan")


def _beat_get(mapping: dict, key: str, index: int) -> float:
    """Look up beat ``index`` of ``key``, else NaN."""
    return _finite_at(mapping[key], index) if key in mapping else float("nan")


def _beat_rows(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> list[dict]:
    """One long-format row per beat (joint-SVD and per-beat-SVD columns)."""
    beatwise = vessel_data["beatwise"]
    per_beat_svd = vessel_data["per_beat_svd"]
    n_beats = 0
    for mapping, key in (
        (beatwise, "R0_b"),
        (beatwise, "A1_b"),
        (per_beat_svd, "R0"),
        (per_beat_svd, "A1"),
    ):
        if key in mapping:
            n_beats = len(np.asarray(mapping[key]))
            break
    vfb = vessel_data["valid_fraction_per_beat"]
    period_b = vessel_data["beat_period_b"]
    singular_b = coerce_beat_spectra(per_beat_svd.get("singular_values_b", []))
    emit_modes = is_usable_beat_spectra(singular_b)

    rows = []
    for b in range(n_beats):
        row = {
            "vessel": vessel,
            "acquisition": sequence,
            "file": h5_path.name,
            "epoch": epoch_short,
            "beat_index": b,
            "beat_period": _finite_at(period_b, b),
            "valid_fraction": _finite_at(vfb, b),
            "mu": _beat_get(beatwise, "mu_b", b),
            "R0": _beat_get(beatwise, "R0_b", b),
            "MPR": _beat_get(beatwise, "MPR_b", b),
            "A1": _beat_get(beatwise, "A1_b", b),
            "R1": _beat_get(beatwise, "R1_b", b),
            "rho1": _beat_get(beatwise, "rho1_b", b),
            "A2": _beat_get(beatwise, "A2_b", b),
            "R2": _beat_get(beatwise, "R2_b", b),
            "rho2": _beat_get(beatwise, "rho2_b", b),
            "A1_pb": _beat_get(per_beat_svd, "A1", b),
            "R1_pb": _beat_get(per_beat_svd, "R1", b),
            "A2_pb": _beat_get(per_beat_svd, "A2", b),
            "R2_pb": _beat_get(per_beat_svd, "R2", b),
            "rho1_pb": _beat_get(per_beat_svd, "rho1", b),
            "rho2_pb": _beat_get(per_beat_svd, "rho2", b),
            "R0_pb": _beat_get(per_beat_svd, "R0", b),
            "MPR_pb": _beat_get(per_beat_svd, "MPR", b),
            "Reff_pb": _beat_get(per_beat_svd, "Reff", b),
            "PR_pb": _beat_get(per_beat_svd, "PR", b),
        }
        if emit_modes:
            for m in range(1, SPECTRUM_N_MODES + 1):
                if b < singular_b.shape[0] and m <= singular_b.shape[1]:
                    val = float(singular_b[b, m - 1])
                    row[f"mode{m}"] = val if np.isfinite(val) else float("nan")
                else:
                    row[f"mode{m}"] = float("nan")
        rows.append(row)
    return rows


def _points_row(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> dict:
    """One acquisition-level row (a single dot per endpoint)."""
    acq = vessel_data["acq"]
    vfb = vessel_data["valid_fraction_per_beat"]

    def scalar(key: str) -> float:
        return float(acq.get(key, np.nan))

    energy = np.asarray(vessel_data.get("energy_fraction", []), dtype=float)
    singular = np.asarray(vessel_data.get("singular_values", []), dtype=float)
    spectrum = singular if singular.size else energy
    row = {
        "vessel": vessel,
        "acquisition": sequence,
        "file": h5_path.name,
        "epoch": epoch_short,
        "A1": scalar("A1"),
        "A1_sd": scalar("sigma_A1_beat"),
        "R1": scalar("R1"),
        "rho1": scalar("rho1"),
        "rho1_sd": scalar("sigma_rho1_beat"),
        "A2": scalar("A2"),
        "A2_sd": scalar("sigma_A2_beat"),
        "R2": scalar("R2"),
        "rho2": scalar("rho2"),
        "rho2_sd": scalar("sigma_rho2_beat"),
        "R0": scalar("R0"),
        "R0_sd": scalar("sigma_R0_beat"),
        "MPR": scalar("MPR"),
        "MPR_sd": scalar("sigma_MPR_beat"),
        "Reff": scalar("Reff"),
        "PR": scalar("PR"),
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
    for m in range(1, SPECTRUM_N_MODES + 1):
        row[f"mode{m}"] = (
            float(spectrum[m - 1]) if spectrum.size >= m else float("nan")
        )
    return row


def _first_usable_array(mapping: dict, *keys: str) -> np.ndarray:
    """First per-beat array among ``keys`` that has a finite value."""
    for key in keys:
        if key not in mapping:
            continue
        arr = np.asarray(mapping[key], dtype=float)
        if np.isfinite(arr).any():
            return arr
    return np.asarray([], dtype=float)


def _median_per_beat(mapping: dict, *keys: str) -> float:
    """Median of the first usable per-beat array among ``keys``."""
    return aggregate_beatwise(_first_usable_array(mapping, *keys), "median")


def _std_per_beat(mapping: dict, *keys: str) -> float:
    """Sample SD of the first usable per-beat array among ``keys``."""
    arr = _first_usable_array(mapping, *keys)
    if not np.isfinite(arr).any():
        return float("nan")
    return finite_std(arr)


def _rho_from_per_beat(mapping: dict, mode: str) -> float:
    """Article ρ = median_b(R) / median_b(R0) from packed per-beat arrays."""
    return aggregate_rho(
        _first_usable_array(mapping, f"R{mode}"),
        _first_usable_array(mapping, "R0"),
        "median",
    )


def _points_row_per_beat(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> dict:
    """Acquisition row from packed per-beat SVD endpoints.

    ``T``, ``μ``, R0, and MPR are not SVD-derived and stay the joint
    values. Mode amplitudes and residuals use ``median_b``; ρ uses
    ``median_b(R)/median_b(R0)``. Beat SDs are taken from the same
    per-beat arrays.
    """
    row = _points_row(vessel, h5_path, sequence, epoch_short, vessel_data)
    pb = vessel_data.get("per_beat_svd") or {}
    row["A1"] = _median_per_beat(pb, "A1")
    row["A2"] = _median_per_beat(pb, "A2")
    row["R1"] = _median_per_beat(pb, "R1")
    row["R2"] = _median_per_beat(pb, "R2")
    row["rho1"] = _rho_from_per_beat(pb, "1")
    row["rho2"] = _rho_from_per_beat(pb, "2")
    row["Reff"] = _median_per_beat(pb, "Reff")
    row["PR"] = _median_per_beat(pb, "PR")
    row["A1_sd"] = _std_per_beat(pb, "A1")
    row["A2_sd"] = _std_per_beat(pb, "A2")
    row["rho1_sd"] = _std_per_beat(pb, "rho1")
    row["rho2_sd"] = _std_per_beat(pb, "rho2")
    spectrum = np.asarray(vessel_data.get("per_beat_spectrum", []), dtype=float)
    if spectrum.size:
        for m in range(1, SPECTRUM_N_MODES + 1):
            row[f"mode{m}"] = (
                float(spectrum[m - 1]) if spectrum.size >= m else float("nan")
            )
    return row


def collect_payload(
    input_h5_paths: Iterable[Path],
    input_root: Path,
    *,
    veins: bool = True,
) -> CohortCollection:
    """Load packed result H5s into points/beats frames for figures or stats."""
    input_root = Path(input_root)
    records, group_order = classify_cohort(input_h5_paths, input_root)
    if not records:
        raise ValueError(f"No acquisition HDF5 files found under {input_root}.")

    group_keys = list(dict.fromkeys([*group_order, *EPOCH_ORDER]))
    vessels = enabled_vessels(bool(veins))
    acqs_by_vessel = {v: {g: [] for g in group_keys} for v in vessels}
    points_rows = {v: [] for v in vessels}
    points_rows_per_beat = {v: [] for v in vessels}
    beat_rows = {v: [] for v in vessels}
    sequence = {v: {g: 0 for g in group_keys} for v in vessels}
    flat_sequence = {v: 0 for v in vessels}

    for group, h5_path in records:
        data = load_acquisition_from_result_h5(h5_path, veins_flag=bool(veins))
        if data is None:
            continue
        for vessel in vessels:
            vessel_data = data.get(vessel)
            if vessel_data is None:
                continue
            if group is None:
                seq = flat_sequence[vessel]
                flat_sequence[vessel] += 1
                label = "ungrouped"
            else:
                acqs_by_vessel[vessel].setdefault(group, [])
                sequence[vessel].setdefault(group, 0)
                acqs_by_vessel[vessel][group].append(vessel_data)
                seq = sequence[vessel][group]
                sequence[vessel][group] += 1
                label = group_display_label(group)
            points_rows[vessel].append(
                _points_row(vessel, h5_path, seq, label, vessel_data)
            )
            points_rows_per_beat[vessel].append(
                _points_row_per_beat(vessel, h5_path, seq, label, vessel_data)
            )
            beat_rows[vessel].extend(
                _beat_rows(vessel, h5_path, seq, label, vessel_data)
            )

    points_by_vessel: dict[str, pd.DataFrame] = {}
    points_by_vessel_per_beat: dict[str, pd.DataFrame] = {}
    beats_by_vessel: dict[str, pd.DataFrame] = {}
    kept_acqs: dict[str, dict[str, list[dict]]] = {}
    for vessel in vessels:
        rows = points_rows[vessel]
        if not rows:
            continue
        rows.sort(key=lambda r: (_row_sort_key(r["epoch"]), r["acquisition"]))
        pb_rows = points_rows_per_beat[vessel]
        pb_rows.sort(key=lambda r: (_row_sort_key(r["epoch"]), r["acquisition"]))
        beats = beat_rows[vessel]
        beats.sort(
            key=lambda r: (
                _row_sort_key(r["epoch"]),
                r["acquisition"],
                r["beat_index"],
            )
        )
        points_by_vessel[vessel] = pd.DataFrame(rows)
        points_by_vessel_per_beat[vessel] = pd.DataFrame(pb_rows)
        beats_by_vessel[vessel] = pd.DataFrame(beats)
        kept_acqs[vessel] = acqs_by_vessel[vessel]

    if not points_by_vessel:
        raise ValueError("No vessel had any valid acquisitions; nothing to write.")

    return CohortCollection(
        points_by_vessel=points_by_vessel,
        beats_by_vessel=beats_by_vessel,
        acqs_by_vessel=kept_acqs,
        group_order=group_order,
        patient_id=extract_patient_id(input_root),
        flicker_protocol=is_flicker_triad(group_order),
        source_files=[str(path) for _, path in records],
        points_by_vessel_per_beat=points_by_vessel_per_beat,
    )


def _cohort_root_from_paths(
    h5_paths: list[Path], preferred_root: Path | None = None
) -> Path:
    """Pick a directory that exposes group folders for ``h5_paths``."""
    if preferred_root is not None and preferred_root.is_dir():
        return resolve_cohort_root(preferred_root)

    try:
        common = Path(os.path.commonpath([str(Path(p).resolve()) for p in h5_paths]))
    except ValueError:
        common = Path(h5_paths[0]).resolve().parent
    if common.is_file():
        common = common.parent

    probe = common
    for _ in range(5):
        records, _order = classify_cohort(h5_paths, probe)
        if any(group is not None for group, _ in records):
            return probe
        if probe.parent == probe:
            break
        probe = probe.parent
    return common


# =====================================================================
# lowrank_cohort.h5 writer
# =====================================================================

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


def write_stats_h5(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
    output_dir: Path,
) -> Path:
    """Pack ``collection`` into ``{patient_id_}lowrank_cohort.h5`` under output_dir."""
    stats_by_vessel, confounds_by_vessel = (
        LowRankWaveformStatistics.build_stats_tables(collection, confounds)
    )
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
) -> tuple[str, list[Path]]:
    """Collect packed H5s, write ``lowrank_cohort.h5``, and figures when split."""
    if not h5_paths:
        raise ValueError(
            "No packed AngioEye result H5 files with low-rank metrics were "
            "found for cohort postprocess. Run lowrank_waveform_decomposition "
            "first so result H5s contain the metrics."
        )
    collection = collect_payload(h5_paths, cohort_root, veins=veins)
    generated_paths: list[Path] = []
    generated_paths.append(
        write_stats_h5(collection, LowRankWaveformConfounds(), Path(output_dir))
    )
    has_cohort_split = len(collection.group_order) >= 2
    if has_cohort_split and collection.points_by_vessel:
        generated_paths.extend(
            LowRankWaveformCohortFigures.plot_all(
                collection.points_by_vessel,
                Path(output_dir),
                collection.group_order,
                patient_id=collection.patient_id,
                beats_by_vessel=collection.beats_by_vessel,
                points_by_vessel_per_beat=collection.points_by_vessel_per_beat,
            )
        )

    vessel_summaries = []
    for vessel, points_df in collection.points_by_vessel.items():
        acqs_by_group = collection.acqs_by_vessel[vessel]
        n_acq = len(points_df)
        group_counts = ", ".join(
            f"{group_display_label(g)}={len(acqs_by_group.get(g, []))}"
            for g in collection.group_order
        ) or f"ungrouped={n_acq}"
        vessel_summaries.append(f"{vessel}={n_acq} ({group_counts})")
    split_note = (
        f"cohort split={collection.group_order}"
        if has_cohort_split
        else "no cohort split (no Figs 5--7)"
    )
    n_rows = sum(len(df) for df in collection.points_by_vessel.values())
    h5_note = generated_paths[0].name if generated_paths else "no stats h5"
    summary = (
        f"Low-rank waveform run: {n_rows} "
        f"acquisition-vessel row(s) ({', '.join(vessel_summaries)}); "
        f"{split_note}; wrote {h5_note}."
    )
    return summary, generated_paths


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = True,
    result_h5_paths: Iterable[Path] | None = None,
) -> tuple[str, list[Path]]:
    """Build ``lowrank_cohort.h5`` and Figs 4--8 from packed AngioEye result H5s.

    Accepts a cohort folder, a ZIP of that tree, or an explicit
    ``result_h5_paths`` list. Writes under ``output_dir``.
    """
    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if result_h5_paths is not None:
        h5_paths = [Path(p) for p in result_h5_paths if result_h5_has_lowrank(p)]
        preferred = input_path if input_path.is_dir() else None
        return _run_on_paths(
            h5_paths,
            _cohort_root_from_paths(h5_paths, preferred),
            output_dir,
            veins=veins,
        )

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with extracted_zip_tree(input_path) as extracted_root:
            cohort_root = resolve_cohort_root(extracted_root)
            return _run_on_paths(
                find_lowrank_result_h5s(cohort_root),
                cohort_root,
                output_dir,
                veins=veins,
            )

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        return _run_on_paths(
            find_lowrank_result_h5s(cohort_root),
            cohort_root,
            output_dir,
            veins=veins,
        )

    raise ValueError(
        f"Input must be a cohort folder or .zip archive, got: {input_path}"
    )


@registerPostprocess(
    name="Low-rank waveform cohort figures",
    description=(
        "From AngioEye result H5s produced by lowrank_waveform_decomposition, "
        "write ``lowrank_cohort.h5`` (stats, confounds, acquisitions, beats) "
        "and cohort Figs. 4--8 (joint and _pb) when 2+ group folders are "
        "present. Writes under ``cohort-results/``."
    ),
    required_deps=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "h5py>=3.8",
    ],
    required_pipelines=["lowrank_waveform_decomposition"],
)
class LowRankWaveformCohortPostprocess(BatchPostprocess):
    veins_flag = False

    def run(self, context: PostprocessContext) -> PostprocessResult:
        """Registered entry: resolve result H5s and write stats H5 plus figures."""
        input_path = Path(context.input_path).expanduser()
        output_dir = cohort_results_dir(context.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_paths = tuple(context.processed_files) or None
        if result_paths is None and not input_path.exists():
            raise FileNotFoundError(
                "No pipeline result H5s were provided and input path does not "
                f"exist: {input_path}. Run lowrank_waveform_decomposition first."
            )

        summary, generated_paths = run(
            input_path if input_path.exists() else Path(result_paths[0]).parent,
            output_dir,
            veins=bool(self.veins_flag),
            result_h5_paths=result_paths,
        )
        return PostprocessResult(
            summary=summary,
            generated_paths=[str(path) for path in generated_paths],
            metadata={
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "n_generated": len(generated_paths),
                "n_result_h5": len(result_paths) if result_paths else 0,
            },
        )
