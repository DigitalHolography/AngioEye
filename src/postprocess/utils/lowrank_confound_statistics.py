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


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Pr(X>Y) - Pr(X<Y), in [-1, 1]. delta > 0 means x tends to be larger."""
    x = _clean(x)
    y = _clean(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float(np.mean(diff > 0) - np.mean(diff < 0))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment; NaN/missing p-values excluded
    from the step-down count rather than inflating it."""
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


def epoch_group_tests(epoch_values: dict[str, np.ndarray]) -> dict:
    """Sec. V.A per-epoch nonparametric tests on one metric's acquisition-
    level dots: Kruskal-Wallis, 3 Holm-adjusted pairwise Mann-Whitney U tests
    with Cliff's delta, and the pooled baseline-vs-flicker test."""
    b1 = _clean(epoch_values["baseline1"])
    fl = _clean(epoch_values["flicker"])
    b2 = _clean(epoch_values["baseline2"])
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
        float(mannwhitneyu(x, y, alternative="two-sided").pvalue) if x.size >= 2 and y.size >= 2 else np.nan
        for _, _, x, y in pairs
    ]
    holm_p = holm_adjust(raw_p)
    deltas = [cliffs_delta(x, y) for _, _, x, y in pairs]

    pooled_p = np.nan
    pooled_delta = np.nan
    if pooled_baseline.size >= 2 and fl.size >= 2:
        pooled_p = float(mannwhitneyu(fl, pooled_baseline, alternative="two-sided").pvalue)
        pooled_delta = cliffs_delta(fl, pooled_baseline)

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


def residualize_against_beat_period(
    epoch_values: dict[str, np.ndarray],
    epoch_beat_periods: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sec. V.B.4: pool the metric and mean beat period across all
    acquisitions/epochs, fit d = alpha + beta*T by OLS, replace each value
    with its residual."""
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in epoch_values.values()])
    all_periods = np.concatenate([np.asarray(v, dtype=float) for v in epoch_beat_periods.values()])
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


_LR = LowRankWaveformDecomposition()


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

_BOM_ACQ_RE = re.compile(r"(\d+)_HD")
_OSS_ACQ_RE = re.compile(r"OSS(?:_R)?_(\d+)")
_GENERIC_ACQ_RE = re.compile(r"_(\d+)(?:_|\.h5$)")


def acquisition_index(path: Path) -> int:
    for pattern in (_BOM_ACQ_RE, _OSS_ACQ_RE, _GENERIC_ACQ_RE):
        for candidate in (path.name, str(path)):
            match = pattern.search(candidate)
            if match:
                return int(match.group(1))
    return 10**9


def _classify_epoch(relative_parts: tuple[str, ...]) -> str | None:
    for part in relative_parts:
        epoch = EPOCH_ALIASES.get(part.lower())
        if epoch is not None:
            return epoch
    return None


MANIP_EPOCH_CODES = {"B1": "baseline1", "B": "baseline1", "F": "flicker", "B2": "baseline2"}


def parse_manip_txt(path: Path) -> dict[int, str]:
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
    """Classifies every raw acquisition path into (dataset, epoch) via
    classify_dataset_and_epoch, grouping and sorting them per dataset the
    same way the old per-dataset directory walk did (epoch order, then
    acquisition index, then path). Returns (records_by_dataset, skipped)
    where skipped is every path that couldn't be classified (e.g. not
    covered by that dataset's manip.txt)."""
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
# PER-ACQUISITION COLLECTION -- a thin wrapper: h5 opening (with retry),
# schema resolution, the joint SVD, and the per-beat SVD variant all live
# in LowRankWaveformDecomposition.compute_acquisition_endpoints now, for
# both artery and vein; this script just calls it and fans the result out
# per vessel.
# =====================================================================

VESSEL_TYPES = ("artery", "vein")


# =====================================================================
# PER-BEAT ENDPOINT EXPORT -- the acquisition-level points table collapses
# each acquisition to one dot per endpoint; this keeps the underlying
# per-beat arrays the engine already computes (same values that get
# median-aggregated into A1/rho1/etc.) as their own long-format table.
# alpha and mpr_prime are acquisition-level-only by
# construction (whole-spectrum / ratio-of-aggregates respectively) and have
# no single-beat value, so they are not included here.
# =====================================================================


def build_beat_rows(
    dataset_name: str,
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> list[dict]:
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
                "rho0": at(beatwise["rho0_b"], b),
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
            # Persists this acquisition's joint-SVD + per-beat endpoints
            # (both vessels, both representations) into their own .h5 (same
            # MetricsTree/ANGIOEYE_PROCESSING_ROOT structure the framework's
            # normal batch run would produce), rather than only ever living
            # in this script's in-memory points/beats tables.
            _LR.write_acquisition_h5(
                h5_path, h5_out_dir / f"{h5_path.stem}_pipelines_result.h5"
            )

        for vessel in VESSEL_TYPES:
            vessel_data = data.get(vessel)
            if vessel_data is None:
                continue
            state = per_vessel[vessel]
            state["acqs_by_epoch"][epoch].append(vessel_data)
            acq = vessel_data["acq"]
            seq = state["sequence_counters"][epoch]
            state["sequence_counters"][epoch] += 1

            vfb = vessel_data["valid_fraction_per_beat"]
            state["points_rows"].append(
                {
                    "dataset": name,
                    "vessel": vessel,
                    "acquisition": acquisition_index(h5_path),
                    "sequence": seq,
                    "file": h5_path.name,
                    "epoch": EPOCH_SHORT[epoch],
                    "A1": float(acq.get("A1", np.nan)),
                    "A1_sd": float(acq.get("sigma_A1_beat", np.nan)),
                    "rho1": float(acq.get("rho1", np.nan)),
                    "rho1_sd": float(acq.get("sigma_rho1_beat", np.nan)),
                    "A2": float(acq.get("A2", np.nan)),
                    "A2_sd": float(acq.get("sigma_A2_beat", np.nan)),
                    "rho2": float(acq.get("rho2", np.nan)),
                    "rho2_sd": float(acq.get("sigma_rho2_beat", np.nan)),
                    "TPR": float(acq.get("TPR", np.nan)),
                    "TPR_sd": float(acq.get("sigma_TPR_beat", np.nan)),
                    "rho0": float(acq.get("rho0", np.nan)),
                    "rho0_sd": float(acq.get("sigma_rho0_beat", np.nan)),
                    "mpr_prime": float(acq.get("mpr_prime", np.nan)),
                    "alpha": float(acq.get("alpha", np.nan)),
                    "beat_period": vessel_data["beat_period_mean"],
                    "beat_period_sd": vessel_data["beat_period_sd"],
                    "mu": float(acq.get("mu_acq", np.nan)),
                    "mu_sd": float(acq.get("sigma_mu_beat", np.nan)),
                    "valid_fraction": float(np.nanmean(vfb)) if vfb.size else float("nan"),
                    "valid_fraction_sd": float(np.nanstd(vfb, ddof=1)) if vfb.size > 1 else float("nan"),
                    "n_valid_columns": vessel_data["n_valid_columns"],
                    "n_total_columns": vessel_data["n_total_columns"],
                }
            )
            state["beat_rows"].extend(
                build_beat_rows(name, vessel, h5_path, seq, EPOCH_SHORT[epoch], vessel_data)
            )

    result: dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]] = {}
    for vessel, state in per_vessel.items():
        points_rows = state["points_rows"]
        beat_rows = state["beat_rows"]
        points_rows.sort(
            key=lambda r: (EPOCH_ORDER.index(EPOCH_SHORT_TO_KEY[r["epoch"]]), r["acquisition"])
        )
        beat_rows.sort(
            key=lambda r: (
                EPOCH_ORDER.index(EPOCH_SHORT_TO_KEY[r["epoch"]]),
                r["acquisition"],
                r["beat_index"],
            )
        )
        result[vessel] = (state["acqs_by_epoch"], points_rows, beat_rows)
    return result


# =====================================================================
# TABLE I/II REPRODUCTION (Sec. V.A/V.C native endpoint tables)
# =====================================================================

TABLE_METRICS = ["rho2", "A2", "rho1", "A1", "TPR", "rho0", "mpr_prime", "alpha"]


def build_endpoint_table(dataset_name: str, vessel: str, points_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TABLE_METRICS:
        b1 = points_df.loc[points_df["epoch"] == "B1", metric].to_numpy(dtype=float)
        fl = points_df.loc[points_df["epoch"] == "Flicker", metric].to_numpy(dtype=float)
        b2 = points_df.loc[points_df["epoch"] == "B2", metric].to_numpy(dtype=float)
        tests = epoch_group_tests({"baseline1": b1, "flicker": fl, "baseline2": b2})
        pair_lookup = {p["pair"]: p for p in tests["pairwise"]}

        b1c, flc, b2c = _clean(b1), _clean(fl), _clean(b2)
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
                "cliffs_delta_F_vs_B1": cliffs_delta(flc, b1c),
                "cliffs_delta_F_vs_B2": cliffs_delta(flc, b2c),
                "cliffs_delta_F_vs_pooled_baseline": cliffs_delta(flc, pooled_baseline),
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# SEC. V.B CONFOUND-CONTROL GRID
# =====================================================================

CONFOUND_METRICS_SVD = ["A1", "A2", "rho1", "rho2"]


def _acq_dots(acqs: list[dict], metric: str, svd_method: str | None, stat: str) -> np.ndarray:
    out = []
    for a in acqs:
        if metric == "TPR":
            out.append(_LR.aggregate_beatwise(a["beatwise"]["TPR_b"], stat))
        elif metric == "rho0":
            # Not SVD-derived (ratio of mu to pulsatile RMS, Eq. 18) -- same
            # beat-aggregation treatment as TPR, no joint/per-beat SVD axis.
            out.append(_LR.aggregate_beatwise(a["beatwise"]["rho0_b"], stat))
        elif metric == "alpha":
            # Single acquisition-level scalar from the joint-SVD singular
            # spectrum (Eq. 19), like effective_rank/participation_ratio --
            # no per-beat array, so svd_method/stat are not applicable axes.
            out.append(float(a["acq"].get("alpha", np.nan)))
        elif metric == "mpr_prime":
            # MPR' (Eq. 17): ratio of two separately-aggregated acquisition
            # scalars (median|mu| / R0), not an average of per-beat ratios --
            # same "no beat-aggregation axis" treatment as alpha.
            out.append(float(a["acq"].get("mpr_prime", np.nan)))
        elif metric in ("A1", "A2"):
            arr = a["beatwise"][f"{metric}_b"] if svd_method == "joint" else a["per_beat_svd"][f"{metric}_b_pb"]
            out.append(_LR.aggregate_beatwise(arr, stat))
        elif metric in ("rho1", "rho2"):
            m = metric[-1]
            if svd_method == "joint":
                R_b = a["beatwise"][f"R{m}_b"]
            else:
                R_b = a["per_beat_svd"][f"R{m}_b_pb"]
            out.append(_LR.aggregate_rho(R_b, a["beatwise"]["TPR_b"], stat))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return np.asarray(out, dtype=float)


def _epoch_dots(acqs_by_epoch: dict[str, list[dict]], metric: str, svd_method: str | None, stat: str):
    return {epoch: _acq_dots(acqs_by_epoch[epoch], metric, svd_method, stat) for epoch in EPOCH_ORDER}


def _epoch_beat_periods(acqs_by_epoch: dict[str, list[dict]]):
    return {
        epoch: np.array([a["beat_period_mean"] for a in acqs_by_epoch[epoch]], dtype=float)
        for epoch in EPOCH_ORDER
    }


def build_confound_grid(
    dataset_name: str, vessel: str, acqs_by_epoch: dict[str, list[dict]]
) -> tuple[list[dict], list[dict]]:
    periods = _epoch_beat_periods(acqs_by_epoch)
    grid_rows: list[dict] = []
    verdict_source: dict[str, list[dict]] = {}

    def run_combo(metric: str, svd_method: str | None, stat: str, regressed: bool) -> dict:
        values = _epoch_dots(acqs_by_epoch, metric, svd_method, stat)
        if regressed:
            values = residualize_against_beat_period(values, periods)
        return epoch_group_tests(values)

    def add_row(metric: str, svd_method: str | None, stat: str, regressed: bool) -> None:
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
            "cliffs_delta_flicker_vs_pooled_baseline": tests["pooled_baseline_vs_flicker_delta"],
            "n_B1": tests["n"]["baseline1"],
            "n_Flicker": tests["n"]["flicker"],
            "n_B2": tests["n"]["baseline2"],
        }
        grid_rows.append(row)
        verdict_source.setdefault(metric, []).append(row)

    for metric in CONFOUND_METRICS_SVD:
        for svd_method in ("joint", "per_beat"):
            for stat in ("median", "mean"):
                for regressed in (False, True):
                    add_row(metric, svd_method, stat, regressed)

    for stat in ("median", "mean"):
        for regressed in (False, True):
            add_row("TPR", None, stat, regressed)
            add_row("rho0", None, stat, regressed)

    for regressed in (False, True):
        add_row("alpha", None, "n/a", regressed)
        add_row("mpr_prime", None, "n/a", regressed)

    verdict_rows = []
    for metric, rows in verdict_source.items():
        ps = [r["pooled_baseline_vs_flicker_p"] for r in rows]
        deltas = [r["cliffs_delta_flicker_vs_pooled_baseline"] for r in rows]
        n_significant = int(sum(1 for p in ps if np.isfinite(p) and p < 0.05))
        signed = [d for d in deltas if np.isfinite(d)]
        consistent_direction = bool(signed) and (all(d > 0 for d in signed) or all(d < 0 for d in signed))
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

            dataset_tables.append(build_endpoint_table(name, vessel, points_df))

            grid_rows, verdict_rows = build_confound_grid(name, vessel, acqs_by_epoch)
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
    def run(self, context: PostprocessContext) -> PostprocessResult:
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
