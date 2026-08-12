"""Cohort low-rank figures, Table I, and confound H5 from packed result H5s."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from input_output import cohort_results_dir
from input_output.archive_io import extracted_zip_tree
from input_output.hdf5_io import open_h5, write_value_dataset
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT

from pipelines.lowrank_waveform_decomposition import (
    FIGURE_VESSELS,
    aggregate_beatwise,
    aggregate_rho,
    enabled_vessels,
    find_lowrank_result_h5s,
    load_acquisition_from_result_h5,
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
_LEADING_INDEX_RE = re.compile(r"^\d+[\s_\-]*")


def _alnum_token_key(text: str) -> tuple:
    """Natural alphanumeric sort key (``2_`` before ``10_``)."""
    tokens: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if part == "":
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    return tuple(tokens)


def epoch_key_from_folder(name: str) -> str | None:
    """Map a folder/group name to baseline1/flicker/baseline2 when recognizable.

    Accepts bare aliases (``baseline1``, ``bl1``, ``flicker``), short labels
    (``B1``, ``Flicker``), and indexed names (``1_baseline1``, ``2_flicker``).
    """
    lower = name.strip().lower()
    candidates = (lower, _LEADING_INDEX_RE.sub("", lower))
    for cand in candidates:
        if not cand:
            continue
        if cand in EPOCH_ALIASES:
            return EPOCH_ALIASES[cand]
        for short, key in EPOCH_SHORT_TO_KEY.items():
            if cand == short.lower():
                return key
    return None


def canonicalize_group_name(name: str) -> str:
    """Keep the folder name as the group key (alphanumeric figure order).

    Flicker semantics use :func:`epoch_key_from_folder` / display labels
    separately; do not collapse ``1_baseline1`` into ``baseline1`` here.
    """
    return name


def group_display_label(group: str) -> str:
    """Axis/table label: B1/Flicker/B2 when the folder encodes a flicker epoch,
    otherwise the folder name itself."""
    key = epoch_key_from_folder(group)
    if key is not None:
        return EPOCH_SHORT[key]
    return group


def _group_sort_key(group: str | None) -> tuple:
    """Alphanumeric folder order; ungrouped (None) last."""
    if group is None:
        return (1, ())
    return (0, _alnum_token_key(group))


def _epoch_rank(epoch_short: str) -> int:
    """Deprecated sort helper kept for call sites; prefer
    ``row_group_sort_key``."""
    if epoch_short in EPOCH_SHORT_ORDER:
        return EPOCH_SHORT_ORDER.index(epoch_short)
    return len(EPOCH_SHORT_ORDER)


def row_group_sort_key(epoch_label: str) -> tuple:
    """Sort key for points/beats rows keyed by display label."""
    if epoch_label in EPOCH_SHORT_ORDER:
        return (0, EPOCH_SHORT_ORDER.index(epoch_label))
    return (1, _alnum_token_key(epoch_label))


def ordered_groups(groups: Iterable[str | None]) -> list[str]:
    """Unique named groups (None dropped), alphanumeric folder order.

    Bare flicker triad folders (``baseline1`` / ``flicker`` / ``baseline2`` or
    aliases without a leading index) keep article order B1→Flicker→B2. Indexed
    names such as ``1_baseline1``, ``2_flicker``, ``3_baseline2`` follow
    alphanumeric order of the folder strings.
    """
    named = {g for g in groups if g is not None}
    ordered = sorted(named, key=_group_sort_key)
    if set(ordered) == set(EPOCH_ORDER):
        return list(EPOCH_ORDER)
    keys = [epoch_key_from_folder(g) for g in ordered]
    if set(keys) == set(EPOCH_ORDER) and all(
        epoch_key_from_folder(g) is not None
        and not _LEADING_INDEX_RE.match(g.strip())
        for g in ordered
    ):
        by_key = {epoch_key_from_folder(g): g for g in ordered}
        return [by_key[epoch] for epoch in EPOCH_ORDER]
    return ordered


def is_flicker_triad(groups: Iterable[str | None]) -> bool:
    """True when folders encode baseline1 + flicker + baseline2 (any naming
    that :func:`epoch_key_from_folder` recognizes), required for Sec. V
    confound / Table I."""
    keys = {
        epoch_key_from_folder(g)
        for g in groups
        if g is not None and epoch_key_from_folder(g) is not None
    }
    return set(EPOCH_ORDER).issubset(keys)


def acqs_by_canonical_epochs(
    acqs_by_group: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Remap folder-keyed acquisition lists onto baseline1/flicker/baseline2."""
    out: dict[str, list[dict]] = {epoch: [] for epoch in EPOCH_ORDER}
    for group, acqs in acqs_by_group.items():
        key = epoch_key_from_folder(group)
        if key in out:
            out[key].extend(acqs)
    return out


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
# Acquisition/group classification -- the first path component under the
# cohort root is the group folder name (kept as-is for alphanumeric figure
# order). Flat acquisitions (h5 directly under the root) get group=None.
# =====================================================================


def classify_group(h5_path: Path, input_root: Path) -> str | None:
    """Return the group key for one acquisition, or None when the file sits
    directly under ``input_root`` (no split folder)."""
    h5_path = Path(h5_path)
    try:
        rel = h5_path.relative_to(input_root)
    except ValueError:
        return None
    parent_parts = rel.parts[:-1]
    if not parent_parts:
        return None
    return canonicalize_group_name(parent_parts[0])


def classify_epoch(h5_path: Path, input_root: Path) -> str | None:
    """Backward-compatible alias for :func:`classify_group`."""
    return classify_group(h5_path, input_root)


def classify_cohort(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[list[tuple[str | None, Path]], list[str]]:
    """Classify every acquisition by its top-level split folder under
    ``input_root``. Returns ``(records, group_order)`` where ``records`` is
    ``(group | None, h5_path)`` sorted by group then file name, and
    ``group_order`` lists the named groups present (length >= 2 means a
    cohort split that should get cohort-level figures). Flat files are
    kept with ``group=None`` rather than skipped."""
    records: list[tuple[str | None, Path]] = []
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        try:
            h5_path.relative_to(input_root)
        except ValueError:
            continue
        records.append((classify_group(h5_path, input_root), h5_path))

    records.sort(key=lambda r: (_group_sort_key(r[0]), r[1].name))
    group_order = ordered_groups(group for group, _ in records)
    return records, group_order


def resolve_cohort_root(path: Path | str) -> Path:
    """Resolve the cohort root that directly contains the split folders
    (e.g. baseline1/flicker/baseline2, or ctrl/path, or any other set of
    group subfolders). If ``path`` already has two or more child
    directories, it is returned as-is; if it has a single child directory (the
    usual layout after extracting ``260803_Flicker_EF.zip`` into a temp
    tree that wraps ``260803_Flicker_EF/``), that child is returned."""
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Cohort root is not a directory: {path}")

    def _keep_child(child: Path) -> bool:
        name = child.name
        if not child.is_dir():
            return False
        # Ignore Finder metadata trees that macOS often injects into ZIPs.
        if name.startswith(".") or name == "__MACOSX":
            return False
        return True

    children = sorted(child for child in path.iterdir() if _keep_child(child))
    if len(children) >= 2:
        return path
    if len(children) == 1:
        return children[0]
    return path

# =====================================================================
# Statistics
# =====================================================================

class LowRankWaveformStatistics:
    """Sec. V.A nonparametric test bundle and Sec. V.C endpoint tables."""

    TABLE_METRICS = ["rho2", "A2", "rho1", "A1", "TPR", "mpr", "mpr_prime", "alpha", "G1"]

    # Article Table I: ten predefined endpoints (Holm family). No alpha / G1.
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

        Rows follow the predefined endpoints (Holm-adjusted as a family).
        Both the unadjusted ``raw_p`` and Holm-adjusted ``p_holm`` are reported
        (plus formatted ``raw_p_fmt`` / ``p_holm_fmt``). Packed into the cohort
        H5 by the cohort runner.
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

# =====================================================================
# Cohort HDF5 writers
# =====================================================================

COHORT_H5_ROOT = f"{ANGIOEYE_POSTPROCESS_ROOT}/lowrank_waveform_cohort"
COHORT_H5_BASENAME = "lowrank_cohort.h5"

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

# =====================================================================
# Cohort figures (Figs 5--7)
# =====================================================================

# Match AcquisitionFigures typography without importing that class.
_FIG_LABEL_SIZE = 14
_FIG_TICK_SIZE = 12

class LowRankWaveformCohortFigures:
    """Article Figs. 5--7 (arterial), written once per cohort in the cohort
    base directory."""

    PANEL_SIZE = 2.5
    FLICKER_SHADE = "#add8e6"
    SPECTRUM_N_MODES = 12
    SPECTRUM_BASELINE = ("black", "-", "o", "Baseline")
    SPECTRUM_FLICKER = ("#555555", "--", "s", "Flicker")
    _RNG = np.random.default_rng(0)
    _DATASET_LABEL_RE = re.compile(
        r"\b(GOA|OSS[_\s-]?L|OSS[_\s-]?R)\b", re.IGNORECASE
    )

    FIG5_PANELS = (
        ("beat_period", r"Beat period $T$"),
        ("mu", r"Baseline level $\mu$"),
        ("TPR", r"Total Pulsatile RMS $R_0$"),
        ("mpr", r"Mean-to-pulsatile ratio MPR"),
    )
    FIG6_PANELS = (
        ("A1", r"Mode-1 amplitude $A_1$"),
        ("A2", r"Mode-2 amplitude $A_2$"),
        ("TPR", r"Total Pulsatile RMS $R_0$"),
        ("R1", r"Residual RMS ($-A_1$) $R_1$"),
        ("R2", r"Residual RMS ($-A_{1:2}$) $R_2$"),
    )
    FIG7_PANELS = (
        ("rho1", r"Residual ratio $\rho_1$"),
        ("rho2", r"Residual ratio $\rho_2$"),
        ("effective_rank", r"Effective rank $R_{\mathrm{eff}}$"),
        ("participation_ratio", r"Participation ratio PR"),
    )

    @staticmethod
    def output_dir_for(output_dir: Path | str) -> Path:
        return Path(output_dir)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9) -> None:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="both", labelsize=tick_size)

    @classmethod
    def plot_all(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_dir: Path,
        group_order: list[str],
        *,
        patient_id: str | None = None,
        dataset_label: str | None = None,
        beats_by_vessel: dict[str, pd.DataFrame] | None = None,
    ) -> list[Path]:
        """Write Figs. 5, 6, 7 and the artery variance-fraction spectrum."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [
            cls.plot_nonsvd_endpoints(
                points_by_vessel,
                out_dir / prefixed_filename("fig5_nonsvd_endpoints.png", patient_id),
                group_order=group_order,
            ),
            cls.plot_lowrank_endpoints(
                points_by_vessel,
                out_dir / prefixed_filename("fig6_lowrank_endpoints.png", patient_id),
                group_order=group_order,
            ),
            cls.plot_residual_spectrum_endpoints(
                points_by_vessel,
                out_dir
                / prefixed_filename("fig7_residual_spectrum_endpoints.png", patient_id),
                group_order=group_order,
            ),
            cls.plot_variance_fraction(
                points_by_vessel,
                out_dir
                / prefixed_filename("fig_variance_fraction.png", patient_id),
                dataset_label=dataset_label,
                beats_by_vessel=beats_by_vessel,
            ),
        ]
        return written

    @classmethod
    def infer_dataset_label(
        cls, points_by_vessel: dict[str, pd.DataFrame]
    ) -> str | None:
        """Best-effort cohort label (GOA / OSS L / OSS R) from acquisition names."""
        for df in points_by_vessel.values():
            if df.empty or "file" not in df.columns:
                continue
            for name in df["file"].astype(str):
                match = cls._DATASET_LABEL_RE.search(name)
                if match is None:
                    continue
                token = match.group(1).upper().replace(" ", "_").replace("-", "_")
                if token.startswith("OSS"):
                    return "OSS L" if token.endswith("L") else "OSS R"
                return token
        return None

    @classmethod
    def plot_variance_fraction(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        *,
        dataset_label: str | None = None,
        beats_by_vessel: dict[str, pd.DataFrame] | None = None,
    ) -> Path:
        """Pooled-baseline vs flicker singular-value spectrum $\\lambda_m$ (artery).

        Prefers per-beat SVD spectra: log-scaled median±IQR across beats
        (Baseline = all beats from B1+B2; Flicker dashed). Falls back to
        acquisition-level ``mode*`` columns when beat spectra are absent.
        Title/legend omitted for caption placement. Aspect 2:1.
        """
        del dataset_label  # caption-owned; retained for call-site compatibility
        out_path = Path(out_path)
        points = points_by_vessel.get("artery", pd.DataFrame())
        beats = (
            (beats_by_vessel or {}).get("artery", pd.DataFrame())
            if beats_by_vessel is not None
            else pd.DataFrame()
        )
        n_keep = cls.SPECTRUM_N_MODES
        beat_mode_cols = [
            f"mode{i}"
            for i in range(1, n_keep + 1)
            if f"mode{i}" in beats.columns
        ]
        # Use beat-level λ_m when present so the grey band is beat-to-beat IQR.
        if not beats.empty and beat_mode_cols:
            df = beats
            mode_cols = beat_mode_cols
        else:
            df = points
            mode_cols = [
                f"mode{i}"
                for i in range(1, n_keep + 1)
                if f"mode{i}" in df.columns
            ]
        modes = np.arange(1, n_keep + 1)

        fig_h = 3.0
        fig, ax = plt.subplots(figsize=(2.0 * fig_h, fig_h))
        series = (
            (
                df["epoch"].isin(["B1", "B2"]) if "epoch" in df.columns else None,
                *cls.SPECTRUM_BASELINE,
            ),
            (
                (df["epoch"] == "Flicker") if "epoch" in df.columns else None,
                *cls.SPECTRUM_FLICKER,
            ),
        )
        if mode_cols and not df.empty:
            plot_modes = np.arange(1, len(mode_cols) + 1)
            for mask, color, style, marker, _label in series:
                if mask is None:
                    continue
                vals = df.loc[mask, mode_cols].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                med = np.nanmedian(vals, axis=0)
                q25 = np.nanpercentile(vals, 25, axis=0)
                q75 = np.nanpercentile(vals, 75, axis=0)
                ax.plot(
                    plot_modes,
                    med,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    linewidth=1.5,
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                )
                ax.fill_between(
                    plot_modes, q25, q75, color=color, alpha=0.12, linewidth=0
                )

        ax.set_yscale("log")
        ax.set_xticks(modes)
        ax.set_xticklabels([str(m) for m in modes])
        ax.set_xlim(0.5, n_keep + 0.5)
        # Narrow λ_m range (<1 decade) needs non-decade major ticks or the
        # side scale stays blank.
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=8)
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _pos: f"{y:g}" if y > 0 else "0")
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$m$", fontsize=_FIG_LABEL_SIZE)
        ax.set_ylabel(
            r"$\lambda_m$", fontsize=_FIG_LABEL_SIZE
        )
        ax.set_box_aspect(0.5)
        cls._style_axes(
            ax, tick_size=_FIG_TICK_SIZE
        )
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def _draw_epoch_panel(
        cls,
        ax,
        df: pd.DataFrame,
        metric: str,
        group_order: list[str],
        *,
        annotate: bool = False,
    ) -> None:
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
            jitter = (cls._RNG.random(vals.size) - 0.5) * 0.16
            med = float(np.nanmedian(vals))
            sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
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
            # Solid acquisition dots under the hollow median marker.
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

        if annotate and metric in df.columns and is_flicker_triad(group_order):
            all_vals = df[metric].to_numpy(dtype=float)
            if np.isfinite(all_vals).any():
                p, delta = LowRankWaveformStatistics.pooled_test(df, metric)
                y_min = float(np.nanmin(all_vals))
                y_max = float(np.nanmax(all_vals))
                pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
                ax.text(
                    0.03,
                    0.97,
                    (
                        f"{LowRankWaveformStatistics.format_p(p)}\n"
                        f"{LowRankWaveformStatistics.format_delta(delta)}"
                    ),
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
    def _plot_paired_endpoint_grid(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        panels: tuple[tuple[str, str], ...],
        out_path: Path,
        group_order: list[str],
        *,
        ylabel: str | None = None,
        annotate: bool = False,
    ) -> Path:
        out_path = Path(out_path)
        vessels = [
            v
            for v in FIGURE_VESSELS
            if v in points_by_vessel and not points_by_vessel[v].empty
        ]
        if not vessels:
            vessels = list(FIGURE_VESSELS)
            points_by_vessel = {"artery": pd.DataFrame()}

        n_cols = len(panels)
        fig, axes = plt.subplots(
            len(vessels),
            n_cols,
            figsize=(cls.PANEL_SIZE * n_cols, cls.PANEL_SIZE * len(vessels)),
            squeeze=False,
        )
        for row_idx, vessel in enumerate(vessels):
            df = points_by_vessel.get(vessel, pd.DataFrame())
            for col_idx, (metric, title) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                cls._draw_epoch_panel(
                    ax, df, metric, group_order, annotate=annotate
                )
                ax.set_box_aspect(1)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11)
                if col_idx == 0:
                    row_label = (
                        ylabel if ylabel is not None else vessel.capitalize()
                    )
                    ax.set_ylabel(row_label, fontsize=11)
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_nonsvd_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 5: beat period, mu, R0, MPR."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel,
            cls.FIG5_PANELS,
            out_path,
            group_order,
            # Panels mix units (s, mm/s, dimensionless); titles carry meaning.
            ylabel="Value",
            annotate=True,
        )

    @classmethod
    def plot_lowrank_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 6: A1, A2, R0, R1, R2."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel,
            cls.FIG6_PANELS,
            out_path,
            group_order,
            ylabel="Value",
            annotate=True,
        )

    @classmethod
    def plot_residual_spectrum_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 7: rho1, rho2, Reff, PR."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel,
            cls.FIG7_PANELS,
            out_path,
            group_order,
            ylabel="Value",
            annotate=True,
        )

# =====================================================================
# Confounds / collection
# =====================================================================

class LowRankWaveformConfounds:
    """Sec. V.B confound-control grid (analysis-choice sweep + verdicts) for
    the classic flicker triad, plus cohort collection from packed result H5s
    (collect_acquisitions, run_confound_statistics). Cohort figures (Figs. 5--7)
    run for any 2+ folder split; confound tables require
    baseline1/flicker/baseline2. Per-acquisition H5/Figs 2--4 are produced by
    the pipeline into ``{stem}_AE/``, not here."""

    # Metrics with both a joint-SVD and a per-beat-SVD representation --
    # the ones build_grid sweeps over the svd_method axis for. TPR/mpr/
    # alpha/G1/mpr_prime have no such axis and are handled separately.
    METRICS_SVD = ("A1", "A2", "rho1", "rho2")

    def __init__(self, *, veins_flag: bool = True) -> None:
        # Cohort confound tables default to both compartments; the registered
        # postprocess can force artery-only via run(..., veins=False).
        self.veins_flag = bool(veins_flag)

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
                out.append(aggregate_beatwise(a["beatwise"]["TPR_b"], stat))
            elif metric == "mpr":
                # Not SVD-derived (ratio of mu to pulsatile RMS, Eq. 18) -- same
                # beat-aggregation treatment as TPR, no joint/per-beat SVD axis.
                out.append(aggregate_beatwise(a["beatwise"]["mpr_b"], stat))
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
                out.append(aggregate_beatwise(arr, stat))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "joint":
                    R_b = a["beatwise"][f"R{m}_b"]
                else:
                    R_b = a["per_beat_svd"][f"R{m}_b_pb"]
                out.append(aggregate_rho(R_b, a["beatwise"]["TPR_b"], stat))
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
        if "TPR_b" in beatwise:
            n_beats = len(np.asarray(beatwise["TPR_b"]))
        elif "A1_b" in beatwise:
            n_beats = len(np.asarray(beatwise["A1_b"]))
        else:
            n_beats = 0
        vfb = vessel_data["valid_fraction_per_beat"]
        period_b = vessel_data["beat_period_b"]

        def at(arr, b):
            arr = np.asarray(arr, dtype=float)
            return float(arr[b]) if b < arr.size and np.isfinite(arr[b]) else float("nan")

        def beat_get(mapping: dict, key: str, b: int) -> float:
            if key not in mapping:
                return float("nan")
            return at(mapping[key], b)

        rows = []
        singular_b = np.asarray(
            per_beat_svd.get("singular_values_b", []), dtype=float
        )
        if singular_b.ndim == 1:
            singular_b = singular_b.reshape(-1, 1) if singular_b.size else np.zeros((0, 0))
        n_spectrum = LowRankWaveformCohortFigures.SPECTRUM_N_MODES
        for b in range(n_beats):
            row = {
                "vessel": vessel,
                "acquisition": sequence,
                "file": h5_path.name,
                "epoch": epoch_short,
                "beat_index": b,
                "beat_period": at(period_b, b),
                "valid_fraction": at(vfb, b),
                "mu": beat_get(beatwise, "mu_b", b),
                "TPR": beat_get(beatwise, "TPR_b", b),
                "mpr": beat_get(beatwise, "mpr_b", b),
                "A1": beat_get(beatwise, "A1_b", b),
                "R1": beat_get(beatwise, "R1_b", b),
                "rho1": beat_get(beatwise, "rho1_b", b),
                "A2": beat_get(beatwise, "A2_b", b),
                "R2": beat_get(beatwise, "R2_b", b),
                "rho2": beat_get(beatwise, "rho2_b", b),
                "A1_pb": beat_get(per_beat_svd, "A1_b_pb", b),
                "R1_pb": beat_get(per_beat_svd, "R1_b_pb", b),
                "A2_pb": beat_get(per_beat_svd, "A2_b_pb", b),
                "R2_pb": beat_get(per_beat_svd, "R2_b_pb", b),
            }
            # Per-beat SVD singular values λ_m (beat-to-beat spectrum).
            # Only emit mode columns when the packed H5 has a real spectrum so
            # cohort plots do not prefer an all-NaN beat table over acq modes.
            if singular_b.ndim == 2 and singular_b.size and np.any(np.isfinite(singular_b)):
                for m in range(1, n_spectrum + 1):
                    if b < singular_b.shape[0] and m <= singular_b.shape[1]:
                        val = float(singular_b[b, m - 1])
                        row[f"mode{m}"] = val if np.isfinite(val) else float("nan")
                    else:
                        row[f"mode{m}"] = float("nan")
            rows.append(row)
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

        energy = np.asarray(vessel_data.get("energy_fraction", []), dtype=float)
        singular = np.asarray(vessel_data.get("singular_values", []), dtype=float)
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
            "TPR": scalar("TPR"),
            "TPR_sd": scalar("sigma_TPR_beat"),
            "mpr": scalar("mpr"),
            "mpr_sd": scalar("sigma_mpr_beat"),
            "mpr_prime": scalar("mpr_prime"),
            "alpha": scalar("alpha"),
            "G1": scalar("G1"),
            "effective_rank": scalar("effective_rank"),
            "participation_ratio": scalar("participation_ratio"),
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
        # Spectrum columns store singular values λ_m (not λ_m^2 / energy fractions).
        spectrum = singular if singular.size else energy
        for m in range(1, LowRankWaveformCohortFigures.SPECTRUM_N_MODES + 1):
            row[f"mode{m}"] = (
                float(spectrum[m - 1]) if spectrum.size >= m else float("nan")
            )
        return row

    def collect_acquisitions(
        self,
        records: list[tuple[str | None, Path]],
        input_root: Path,
        output_dir: Path | None = None,
        patient_id: str | None = None,
        group_order: list[str] | None = None,
    ) -> dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]]:
        """Collect endpoints from packed AngioEye result H5s.

        ``records`` are ``(group, result_h5_path)`` where each H5 already
        contains ``/AngioEye/Processing/lowrank_waveform_decomposition``
        (typically ``{stem}_AE.h5``). Does not write per-acquisition H5 or
        Figs 2--4 — those are pipeline/AE companions. ``output_dir`` /
        ``patient_id`` are retained for call-site compatibility and ignored.
        Returns
        ``{"artery": (acqs_by_group, points_rows, beat_rows), "vein": (...)}``.
        """
        del output_dir, patient_id, input_root
        if group_order is None:
            group_order = ordered_groups(group for group, _ in records)
        # Always allocate known flicker keys too so build_grid can still
        # index them when the triad is present (empty lists otherwise).
        group_keys = list(dict.fromkeys([*group_order, *EPOCH_ORDER]))

        per_vessel: dict[str, dict] = {
            vessel: {
                "acqs_by_group": {g: [] for g in group_keys},
                "points_rows": [],
                "beat_rows": [],
                "sequence_counters": {g: 0 for g in group_keys},
                "flat_sequence": 0,
            }
            for vessel in enabled_vessels(self.veins_flag)
        }

        for group, h5_path in records:
            data = load_acquisition_from_result_h5(
                h5_path, veins_flag=bool(self.veins_flag)
            )
            if data is None:
                continue

            for vessel in per_vessel:
                vessel_data = data.get(vessel)
                if vessel_data is None:
                    continue
                state = per_vessel[vessel]
                if group is None:
                    seq = state["flat_sequence"]
                    state["flat_sequence"] += 1
                    group_label = "ungrouped"
                else:
                    if group not in state["acqs_by_group"]:
                        state["acqs_by_group"][group] = []
                        state["sequence_counters"][group] = 0
                    state["acqs_by_group"][group].append(vessel_data)
                    seq = state["sequence_counters"][group]
                    state["sequence_counters"][group] += 1
                    group_label = group_display_label(group)

                state["points_rows"].append(
                    self.build_points_row(vessel, h5_path, seq, group_label, vessel_data)
                )
                state["beat_rows"].extend(
                    self.build_beat_rows(vessel, h5_path, seq, group_label, vessel_data)
                )

        result: dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]] = {}
        for vessel, state in per_vessel.items():
            points_rows = state["points_rows"]
            beat_rows = state["beat_rows"]
            points_rows.sort(
                key=lambda r: (row_group_sort_key(r["epoch"]), r["acquisition"])
            )
            beat_rows.sort(
                key=lambda r: (
                    row_group_sort_key(r["epoch"]),
                    r["acquisition"],
                    r["beat_index"],
                )
            )
            result[vessel] = (state["acqs_by_group"], points_rows, beat_rows)
        return result

    def run_confound_statistics(
        self,
        input_h5_paths: Iterable[Path],
        input_root: Path,
        output_dir: Path,
    ) -> tuple[str, list[Path]]:
        """Cohort-only regeneration from packed AngioEye result H5s:

        * reads ``{stem}_AE.h5`` (or legacy result H5s) that already contain
          low-rank metrics — does not recompute SVD or write per-acq products;
        * when the dataset has a multi-folder split (any 2+ named groups --
          bl1/f/bl2, ctrl/path, ...), article Figs. 5--7 are written beside
          the cohort H5 under ``output_dir``;
        * Sec. V.A/V.B endpoint tables + confound grids are packed into the
          cohort H5 only when the classic flicker triad is present.

        Returns ``(summary, generated_paths)``.
        """
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        patient_id = extract_patient_id(input_root)

        records, group_order = classify_cohort(input_h5_paths, input_root)
        if not records:
            raise ValueError(
                f"No acquisition HDF5 files found under {input_root}."
            )

        has_cohort_split = len(group_order) >= 2
        flicker_protocol = is_flicker_triad(group_order)
        figures_dir = LowRankWaveformCohortFigures.output_dir_for(output_dir)
        if has_cohort_split:
            figures_dir.mkdir(parents=True, exist_ok=True)

        generated_paths: list[Path] = []

        per_vessel = self.collect_acquisitions(
            records,
            input_root,
            group_order=group_order,
        )

        vessel_summaries: list[str] = []
        all_points: list[pd.DataFrame] = []
        points_by_vessel: dict[str, pd.DataFrame] = {}
        beats_by_vessel: dict[str, pd.DataFrame] = {}
        tables_by_vessel: dict[str, dict[str, pd.DataFrame]] = {}
        confound_by_vessel: dict[str, dict[str, pd.DataFrame]] = {}
        for vessel, (acqs_by_group, points_rows, beat_rows) in per_vessel.items():
            n_acq = len(points_rows)
            if n_acq == 0:
                continue
            group_counts = ", ".join(
                f"{group_display_label(g)}={len(acqs_by_group.get(g, []))}"
                for g in group_order
            ) or f"ungrouped={n_acq}"
            vessel_summaries.append(f"{vessel}={n_acq} ({group_counts})")

            points_df = pd.DataFrame(points_rows)
            beats_df = pd.DataFrame(beat_rows)
            points_by_vessel[vessel] = points_df
            beats_by_vessel[vessel] = beats_df
            all_points.append(points_df)

            if flicker_protocol:
                tables_by_vessel[vessel] = {
                    "endpoint_table": (
                        LowRankWaveformStatistics.build_endpoint_table(
                            vessel, points_df
                        )
                    ),
                    "table1_pooled_baseline_vs_flicker": (
                        LowRankWaveformStatistics.build_table1_pooled_comparison(
                            vessel, points_df
                        )
                    ),
                }
                grid_rows, verdict_rows = self.build_grid(
                    vessel, acqs_by_canonical_epochs(acqs_by_group)
                )
                confound_by_vessel[vessel] = {
                    "grid": pd.DataFrame(grid_rows),
                    "verdict": pd.DataFrame(verdict_rows),
                }

        if not all_points:
            raise ValueError(
                "No vessel had any valid acquisitions; nothing to write."
            )

        cohort_h5 = write_cohort_h5(
            output_dir / prefixed_filename(COHORT_H5_BASENAME, patient_id),
            points_by_vessel=points_by_vessel,
            beats_by_vessel=beats_by_vessel,
            group_order=group_order,
            patient_id=patient_id,
            flicker_protocol=flicker_protocol,
            source_files=[str(path) for _, path in records],
            tables_by_vessel=tables_by_vessel,
            confound_by_vessel=confound_by_vessel,
        )
        generated_paths.append(cohort_h5)

        if has_cohort_split and points_by_vessel:
            generated_paths.extend(
                LowRankWaveformCohortFigures.plot_all(
                    points_by_vessel,
                    figures_dir,
                    group_order,
                    patient_id=patient_id,
                    beats_by_vessel=beats_by_vessel,
                )
            )

        split_note = (
            f"cohort split={group_order}"
            if has_cohort_split
            else "no cohort split (cohort H5 points/beats only)"
        )
        combined_points = pd.concat(all_points, ignore_index=True)
        summary = (
            f"Low-rank waveform run: {len(combined_points)} "
            f"acquisition-vessel row(s) ({', '.join(vessel_summaries)}); "
            f"{cohort_h5.name}; {split_note}."
        )
        return summary, generated_paths

    def run(self, input_h5_paths, input_root, output_dir):
        """Lower-level cohort entry when paths are already resolved.
        Prefer the module-level :func:`run` for ZIP/folder inputs."""
        if not input_h5_paths:
            raise ValueError(
                "No input acquisition files are available for postprocessing."
            )

        return self.run_confound_statistics(
            input_h5_paths=input_h5_paths,
            input_root=input_root,
            output_dir=output_dir,
        )

# =====================================================================
# Cohort entry
# =====================================================================

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


def _run_on_paths(
    h5_paths: list[Path],
    cohort_root: Path,
    output_dir: Path,
    *,
    veins: bool,
) -> tuple[str, list[Path]]:
    if not h5_paths:
        raise ValueError(
            "No packed AngioEye result H5 files with low-rank metrics were "
            "found for cohort postprocess. Run lowrank_waveform_decomposition "
            "first so result H5s contain the metrics."
        )
    return LowRankWaveformConfounds(veins_flag=bool(veins)).run(
        h5_paths, cohort_root, output_dir
    )


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = True,
    result_h5_paths: Iterable[Path] | None = None,
) -> tuple[str, list[Path]]:
    """Build cohort H5 + Figs 5--7 from packed AngioEye result H5s.

    Accepts a cohort folder, a ZIP of that tree, or an explicit
    ``result_h5_paths`` list (e.g. ``PostprocessContext.processed_files``).
    Writes under ``output_dir`` (typically ``cohort-results/``).
    """
    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if result_h5_paths is not None:
        h5_paths = [Path(p) for p in result_h5_paths if result_h5_has_lowrank(p)]
        preferred = input_path if input_path.is_dir() else None
        cohort_root = _cohort_root_from_paths(h5_paths, preferred)
        return _run_on_paths(h5_paths, cohort_root, output_dir, veins=veins)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with extracted_zip_tree(input_path) as extracted_root:
            cohort_root = resolve_cohort_root(extracted_root)
            h5_paths = find_lowrank_result_h5s(cohort_root)
            return _run_on_paths(h5_paths, cohort_root, output_dir, veins=veins)

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        h5_paths = find_lowrank_result_h5s(cohort_root)
        return _run_on_paths(h5_paths, cohort_root, output_dir, veins=veins)

    raise ValueError(
        f"Input must be a cohort folder or .zip archive, got: {input_path}"
    )

# =====================================================================
# Registered postprocess
# =====================================================================

@registerPostprocess(
    name="Low-rank waveform cohort figures",
    description=(
        "From AngioEye result H5s produced by lowrank_waveform_decomposition "
        "(Holo: `{stem}_AE.h5`; ZIP: `*_pipelines_result.h5` with the ingested "
        "EyeFlow Metrics/lowrank_waveform_decomposition group), build cohort "
        "Figs. 5--7 when 2+ group folders are present, plus one "
        "`lowrank_cohort.h5` (points/beats; Table I / confound tables for the "
        "flicker triad). Writes under ``cohort-results/``. "
        "Per-acquisition Figs 2--4 are produced by the pipeline, not this "
        "postprocess."
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
    def run(self, context: PostprocessContext) -> PostprocessResult:
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
            veins=False,
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

