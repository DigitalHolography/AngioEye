"""
Sec. VII feature vector: assembles the eight retained metrics -- amplitude
(A1), complexity (rho1, effective_rank, participation_ratio), shape
(W80_over_T, Q_d_width, CF, E_LF_over_E_HF) -- into one per-acquisition
table, and reports:

  - Fig. 13: each metric's flicker-vs-pooled-baseline effect size (Cliff's
    delta), using each metric's own "native" acquisition-level value (the
    same un-beat-combined convention Table I itself uses for A1/rho1, and
    a flat median for the four shape descriptors).
  - Fig. 14: pairwise Spearman correlation across the eight metrics.
  - retention: whether each metric independently survives the Sec. V.B
    replication standard (flicker_stats.confound_controlled_replication for
    A1/rho1/the four shape descriptors, which have a genuine per-beat axis
    to combine two ways; plain Kruskal-Wallis/Mann-Whitney/Cliff's-delta --
    flicker_stats.epoch_group_tests -- for effective_rank/
    participation_ratio, which are already single acquisition-level numbers
    with no beat axis, per Fig. 15's caption in the manuscript).

Reuses collect_epoch_data from flicker_epoch_report.py and flicker_stats.py
for every metric and every statistical test; no metric math is
reimplemented here.

Not a pytest test -- a reporting script you run by hand. Not
auto-collected by pytest (filename doesn't match test_*.py).

Usage:
    python3 test/flicker_feature_vector.py /path/to/organized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flicker_epoch_report import CANDIDATE_METRICS, collect_epoch_data  # noqa: E402
from flicker_stats import (  # noqa: E402
    EPOCH_ORDER,
    cliffs_delta,
    confound_controlled_replication,
    epoch_group_tests,
)

AXIS_BY_METRIC = {
    "A1": "amplitude",
    "rho1": "complexity",
    "effective_rank": "complexity",
    "participation_ratio": "complexity",
    "W80_over_T": "shape",
    "Q_d_width": "shape",
    "CF": "shape",
    "E_LF_over_E_HF": "shape",
}
FEATURE_VECTOR_METRICS = list(AXIS_BY_METRIC)

# Metrics with a genuine per-beat axis, so the Sec. V.B replication check
# (mean/median beat-combination x exclusion/regression) applies to them.
BEAT_COMBINED_METRICS = [m for m in FEATURE_VECTOR_METRICS if m in CANDIDATE_METRICS]
# Acquisition-level-only metrics, screened via plain group tests instead
# (Fig. 15's caption: "screened under a different protocol").
SCALAR_ONLY_METRICS = [m for m in FEATURE_VECTOR_METRICS if m not in CANDIDATE_METRICS]


def build_native_table(results: dict) -> dict[str, dict[str, np.ndarray]]:
    """
    table[metric][epoch] -> array of native per-acquisition values, the
    values Fig. 13 and Fig. 14 are computed from.
    """
    table: dict[str, dict[str, np.ndarray]] = {}
    for name in FEATURE_VECTOR_METRICS:
        native_name = name if name in SCALAR_ONLY_METRICS else f"{name}_native"
        table[name] = {
            epoch: np.asarray(results[native_name][epoch], dtype=float) for epoch in EPOCH_ORDER
        }
    return table


def effect_sizes(table: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Fig. 13: Cliff's delta, flicker vs. pooled baseline, per metric."""
    rows = []
    for name in FEATURE_VECTOR_METRICS:
        pooled_baseline = np.concatenate([table[name]["baseline1"], table[name]["baseline2"]])
        delta = cliffs_delta(table[name]["flicker"], pooled_baseline)
        rows.append({"name": name, "axis": AXIS_BY_METRIC[name], "cliffs_delta": delta})
    rows.sort(key=lambda r: abs(r["cliffs_delta"]) if np.isfinite(r["cliffs_delta"]) else -1, reverse=True)
    return rows


def correlation_matrix(table: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Fig. 14: pairwise Spearman correlation across all pooled acquisitions."""
    pooled = {
        name: np.concatenate([table[name][epoch] for epoch in EPOCH_ORDER])
        for name in FEATURE_VECTOR_METRICS
    }
    n = len(FEATURE_VECTOR_METRICS)
    corr = np.full((n, n), np.nan)
    for i, name_i in enumerate(FEATURE_VECTOR_METRICS):
        for j, name_j in enumerate(FEATURE_VECTOR_METRICS):
            x, y = pooled[name_i], pooled[name_j]
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 3:
                continue
            corr[i, j] = float(spearmanr(x[mask], y[mask]).correlation)
    return corr


def retention_flags(results: dict) -> dict[str, bool]:
    """
    Whether each feature-vector metric independently satisfies its
    appropriate replication standard (Sec. V.B for the beat-combined
    metrics, plain Sec. V.A group tests for the scalar-only ones).
    """
    beat_periods = {epoch: results["beat_period"][epoch] for epoch in EPOCH_ORDER}
    retained = {}

    for name in BEAT_COMBINED_METRICS:
        beatwise = {epoch: results[name][epoch] for epoch in EPOCH_ORDER}
        retained[name] = confound_controlled_replication(beatwise, beat_periods)["retained"]

    for name in SCALAR_ONLY_METRICS:
        values = {epoch: results[name][epoch] for epoch in EPOCH_ORDER}
        report = epoch_group_tests(values)
        p = report["pooled_baseline_vs_flicker_p"]
        retained[name] = bool(np.isfinite(p) and p < 0.05)

    return retained


def print_effect_size_table(rows: list[dict]) -> None:
    delta_header = "Cliff's delta"
    print(f"{'metric':>20} {'axis':>12} {delta_header:>15}")
    print("-" * 50)
    for r in rows:
        print(f"{r['name']:>20} {r['axis']:>12} {r['cliffs_delta']:>15.3f}")


def print_correlation_matrix(corr: np.ndarray) -> None:
    header = " " * 20 + "".join(f"{name:>12}" for name in FEATURE_VECTOR_METRICS)
    print(header)
    for name, row in zip(FEATURE_VECTOR_METRICS, corr):
        print(f"{name:>20}" + "".join(f"{v:>12.2f}" for v in row))


def print_retention(retained: dict[str, bool]) -> None:
    for name in FEATURE_VECTOR_METRICS:
        protocol = "V.B replication" if name in BEAT_COMBINED_METRICS else "V.A group test"
        print(f"  {name:>20} ({protocol}): retained={retained[name]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("organized_root", type=Path)
    args = parser.parse_args()

    results = collect_epoch_data(args.organized_root)
    table = build_native_table(results)

    print("\n=== Fig. 13: effect sizes (Cliff's delta, flicker vs. pooled baseline) ===")
    print_effect_size_table(effect_sizes(table))

    print("\n=== Fig. 14: pairwise Spearman correlation ===")
    print_correlation_matrix(correlation_matrix(table))

    print("\n=== Sec. VII retention (each metric's own replication standard) ===")
    print_retention(retention_flags(results))


if __name__ == "__main__":
    main()
