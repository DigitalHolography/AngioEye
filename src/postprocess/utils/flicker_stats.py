"""
Statistical testing primitives for the low-rank flicker paper's replication
standard (manuscript Sec. V): per-epoch nonparametric tests (Kruskal-Wallis,
Holm-adjusted pairwise Mann-Whitney U, Cliff's delta, pooled-baseline test),
the confound-controlled replication check (mean/median beat-combination x
heart-rate-outlier-exclusion/beat-period-regression, 4 combinations, a
candidate is retained only if all four agree), and the three-check confound
battery (raw separability -> V.B replication -> residualization against an
arbitrary control variable) that flicker-detection analysis scripts have
been re-implementing by hand, one at a time, since the project started.

Promoted from test/flicker_stats.py (an exploratory library module) to a
real postprocess utility, following the same convention as
postprocess/utils/stats_groups_comparison.py: pure functions, no I/O, meant
to be called both from a registered postprocess step
(flicker_replication_pipeline.py) and directly from analysis scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, linregress, mannwhitneyu

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


def mann_whitney_and_auc(baseline: np.ndarray, flicker: np.ndarray) -> tuple[float, float]:
    """
    Raw (no confound control) two-sided Mann-Whitney U test between pooled-
    baseline and flicker acquisition-level values, plus its AUC-equivalent
    (U / (n_baseline * n_flicker)). This is check 1 of run_confound_battery
    below -- previously reimplemented independently in build_qc_control_table.py,
    check_denoising_correction_magnitude.py, and check_bilateral_replication_260720_OSS.py.
    """
    baseline = _clean(baseline)
    flicker = _clean(flicker)
    if baseline.size < 2 or flicker.size < 2:
        return np.nan, np.nan
    stat, p = mannwhitneyu(flicker, baseline, alternative="two-sided")
    return float(p), float(stat / (baseline.size * flicker.size))


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


def residualize_against_control(
    values: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Linear-regress `values` on `control` (pooled across all acquisitions, any
    epoch) and return the residuals plus the fit's R^2. The generic form of
    residualize_against_beat_period's confound-removal step, usable against
    any control variable (total_pulsatile_rms, valid_fraction,
    quality_score, ...), not just beat period -- this is what every
    check_*_vs_*.py / check_*_confounds.py script in this project's history
    did by hand for one specific (metric, control variable) pair at a time.
    """
    values = np.asarray(values, dtype=float)
    control = np.asarray(control, dtype=float)
    mask = np.isfinite(values) & np.isfinite(control)
    if np.sum(mask) < 3:
        return values.copy(), np.nan
    fit = linregress(control[mask], values[mask])
    residuals = values - (fit.slope * control + fit.intercept)
    return residuals, float(fit.rvalue**2)


def run_confound_battery(
    epoch_native_values: dict[str, np.ndarray],
    *,
    beatwise_values: dict[str, list[np.ndarray]] | None = None,
    beat_periods: dict[str, list[float]] | None = None,
    control_values: dict[str, np.ndarray] | None = None,
    control_name: str = "control",
) -> dict:
    """
    The three-check confound battery this project has run by hand, with
    small variations, in roughly fifteen separate analysis scripts
    (build_qc_control_table.py's Check 1/3, check_peak_width_qc_confounds.py,
    check_denoising_correction_magnitude.py,
    check_bilateral_replication_260720_OSS.py, and others):

      1. Raw pooled-baseline-vs-flicker Mann-Whitney/AUC on the metric's
         native (already beat-combined) acquisition-level value.
      2. Sec. V.B 4-combination heart-rate confound-controlled replication
         (confound_controlled_replication), if per-beat values and beat
         periods are supplied.
      3. Residualization against an arbitrary control variable (e.g.
         total_pulsatile_rms), re-testing the raw Mann-Whitney on what's
         left, if control_values are supplied.

    epoch_native_values[epoch] is the metric's one-value-per-acquisition
    array for that epoch (already reduced over beats/vessel-location,
    matching the project's "native" convention -- e.g.
    lowrank_pulsatility_metrics.py's acq["A1"]). All other dict arguments
    are optional and use the same {epoch: ...} shape; omit an argument to
    skip that check rather than passing an empty dict.
    """
    b1 = _clean(epoch_native_values.get("baseline1", []))
    b2 = _clean(epoch_native_values.get("baseline2", []))
    fl = _clean(epoch_native_values.get("flicker", []))
    pooled_baseline = np.concatenate([b1, b2])
    raw_p, raw_auc = mann_whitney_and_auc(pooled_baseline, fl)

    result: dict = {"raw_p": raw_p, "raw_auc": raw_auc}

    if beatwise_values is not None and beat_periods is not None:
        result["vb_report"] = confound_controlled_replication(beatwise_values, beat_periods)

    if control_values is not None:
        all_values = np.concatenate(
            [np.asarray(epoch_native_values.get(epoch, []), dtype=float) for epoch in EPOCH_ORDER]
        )
        all_control = np.concatenate(
            [np.asarray(control_values.get(epoch, []), dtype=float) for epoch in EPOCH_ORDER]
        )
        residuals, r_squared = residualize_against_control(all_values, all_control)

        resid_by_epoch: dict[str, np.ndarray] = {}
        offset = 0
        for epoch in EPOCH_ORDER:
            n = len(np.asarray(epoch_native_values.get(epoch, []), dtype=float))
            resid_by_epoch[epoch] = residuals[offset : offset + n]
            offset += n

        resid_pooled_baseline = np.concatenate(
            [_clean(resid_by_epoch.get("baseline1", [])), _clean(resid_by_epoch.get("baseline2", []))]
        )
        resid_p, resid_auc = mann_whitney_and_auc(resid_pooled_baseline, _clean(resid_by_epoch.get("flicker", [])))
        result["residualized"] = {
            "control_name": control_name,
            "r_squared": r_squared,
            "p": resid_p,
            "auc": resid_auc,
            "survives": bool(np.isfinite(resid_p) and resid_p < 0.05),
        }

    return result
