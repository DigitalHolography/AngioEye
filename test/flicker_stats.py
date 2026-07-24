"""
Statistical testing primitives for the low-rank flicker paper's replication
standard (manuscript Sec. V): per-epoch nonparametric tests (Kruskal-Wallis,
Holm-adjusted pairwise Mann-Whitney U, Cliff's delta, pooled-baseline test)
and the confound-controlled replication check (mean/median beat-combination
x heart-rate-outlier-exclusion/beat-period-regression, 4 combinations, a
candidate is retained only if all four agree).

Not a pytest test -- a library of functions meant to be imported by a
reporting script (the way denoising_benchmark.py is imported by
flicker_denoising_separability.py). Not auto-collected by pytest (filename
doesn't match test_*.py).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

EPOCH_ORDER = ["baseline1", "flicker", "baseline2"]


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def beat_combine(per_beat_values: np.ndarray, method: str) -> float:
    """
    Collapse one acquisition's per-beat values (already reduced over vessel
    location) down to a single acquisition-level dot -- the "beat-
    combination" step of the replication standard (Sec. V.B), either "mean"
    or "median" across beats.
    """
    x = _clean(per_beat_values)
    if x.size == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(x))
    if method == "median":
        return float(np.median(x))
    raise ValueError(f"Unknown beat-combination method: {method}")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta = Pr(X>Y) - Pr(X<Y), computed by enumerating all
    cross-pairs (Sec. V.A), in [-1, 1]. delta > 0 means x tends to be larger
    than y.
    """
    x = _clean(x)
    y = _clean(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    diff = x[:, None] - y[None, :]
    return float(np.mean(diff > 0) - np.mean(diff < 0))


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjustment, controlling family-wise error
    across the given p-values (Sec. V.A: "Holm adjustment controls the
    family-wise error rate within each metric's three pairwise
    comparisons").
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.full(n, np.nan)
    running_max = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        running_max = max(running_max, (n - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def epoch_group_tests(epoch_values: dict[str, np.ndarray]) -> dict:
    """
    Sec. V.A per-epoch nonparametric tests on one metric's acquisition-level
    dots: overall Kruskal-Wallis across baseline1/flicker/baseline2, the
    three Holm-adjusted pairwise Mann-Whitney U tests with Cliff's delta,
    and a pooled-baseline (baseline1+baseline2) vs. flicker test.
    """
    b1 = _clean(epoch_values["baseline1"])
    fl = _clean(epoch_values["flicker"])
    b2 = _clean(epoch_values["baseline2"])
    pooled_baseline = np.concatenate([b1, b2])

    kw_p = np.nan
    if b1.size >= 1 and fl.size >= 1 and b2.size >= 1:
        try:
            kw_p = float(kruskal(b1, fl, b2).pvalue)
        except ValueError:
            # scipy raises rather than returning nan when every pooled value
            # is identical (zero variance -- no group difference to detect).
            kw_p = np.nan

    pairs = [("baseline1", "flicker", b1, fl), ("flicker", "baseline2", fl, b2), ("baseline1", "baseline2", b1, b2)]
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


def exclude_heart_rate_outliers(
    epoch_values: dict[str, np.ndarray],
    epoch_beat_periods: dict[str, np.ndarray],
    n_mad: float = 3.0,
) -> dict[str, np.ndarray]:
    """
    Confound-control strategy 1 (Sec. V.B): drop acquisitions whose
    within-acquisition mean beat period is a heart-rate outlier relative to
    the pooled distribution across all three epochs (robust z-score via
    median/MAD, threshold n_mad).
    """
    all_periods = np.concatenate([_clean(v) for v in epoch_beat_periods.values()])
    med = np.median(all_periods) if all_periods.size else np.nan
    mad = np.median(np.abs(all_periods - med)) * 1.4826 if all_periods.size else 0.0

    out = {}
    for epoch, values in epoch_values.items():
        values = np.asarray(values, dtype=float)
        if mad <= 0 or not np.isfinite(mad):
            out[epoch] = values
            continue
        periods = np.asarray(epoch_beat_periods[epoch], dtype=float)
        z = np.abs(periods - med) / mad
        keep = np.isfinite(z) & (z <= n_mad)
        out[epoch] = values[keep]
    return out


def residualize_against_beat_period(
    epoch_values: dict[str, np.ndarray],
    epoch_beat_periods: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Confound-control strategy 2 (Sec. V.B): regress the metric on beat
    period (pooled across all acquisitions/epochs) and replace each value
    with its residual, so any beat-period-driven trend is removed before
    testing for a flicker effect.
    """
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


def confound_controlled_replication(
    beatwise_values: dict[str, list[np.ndarray]],
    beat_periods: dict[str, list[float]],
) -> dict:
    """
    Sec. V.B replication standard: run epoch_group_tests under all four
    combinations of {mean, median} beat-combination x {exclusion,
    regression} confound control, and retain the candidate only if the
    pooled-baseline-vs-flicker effect is significant (p<0.05) in a
    consistent direction under all four.

    beatwise_values[epoch] is a list, one entry per acquisition in that
    epoch, of that acquisition's per-beat metric array (already reduced
    over vessel location -- e.g. lowrank_pulsatility_metrics.py's
    beatwise["median_kr_A1"]). beat_periods[epoch] is the matching list of
    each acquisition's mean beat period, used for confound control.
    """
    combos = {}
    for beat_combo in ("mean", "median"):
        acq_values = {
            epoch: np.array([beat_combine(v, beat_combo) for v in beatwise_values[epoch]])
            for epoch in EPOCH_ORDER
        }
        acq_periods = {epoch: np.asarray(beat_periods[epoch], dtype=float) for epoch in EPOCH_ORDER}

        combos[f"{beat_combo}_exclusion"] = epoch_group_tests(
            exclude_heart_rate_outliers(acq_values, acq_periods)
        )
        combos[f"{beat_combo}_regression"] = epoch_group_tests(
            residualize_against_beat_period(acq_values, acq_periods)
        )

    ps = [c["pooled_baseline_vs_flicker_p"] for c in combos.values()]
    deltas = [c["pooled_baseline_vs_flicker_delta"] for c in combos.values()]
    significant = all(np.isfinite(p) and p < 0.05 for p in ps)
    signed = [d for d in deltas if np.isfinite(d)]
    consistent_direction = bool(signed) and (all(d > 0 for d in signed) or all(d < 0 for d in signed))

    return {
        "combinations": combos,
        "retained": bool(significant and consistent_direction),
    }
