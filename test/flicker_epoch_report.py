"""
Runs the Sec. V.A/V.B replication standard (flicker_stats.py) on real
epoch-organized acquisitions, for the low-rank paper's headline candidates
(A1, rho1) and the Sec. VI.A canonical-mode shape descriptors from
canonical_mode_shape_metrics.py.

Uses the real BOM0753 flicker session (organized into baseline1/flicker/
baseline2 folders by acquisition index, per organize_flicker_files.py),
the same folder convention as flicker_denoising_separability.py.

Not a pytest test -- a reporting script you run by hand. Not auto-collected
by pytest (filename doesn't match test_*.py).

Usage:
    python3 test/flicker_epoch_report.py /path/to/organized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402

from flicker_stats import EPOCH_ORDER, confound_controlled_replication  # noqa: E402
from pipelines.canonical_mode_shape_metrics import CanonicalModeShapeMetrics  # noqa: E402
from pipelines.lowrank_pulsatility_metrics import LowRankPulsatilityMetrics  # noqa: E402

RAW_SEGMENT_PATH = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
BEAT_PERIOD_PATH = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

CANDIDATE_METRICS = ["A1", "rho1", "W80_over_T", "Q_d_width", "CF", "E_LF_over_E_HF"]

# Acquisition-level scalars with no per-beat axis to combine two ways, so
# they sit outside the Sec. V.B replication check's beat-combination step:
# the "native" (flat median over all beat x branch x radius, matching
# lowrank_pulsatility_metrics.py's own acq dict) value of each
# CANDIDATE_METRICS entry, used by Sec. VII/Fig. 13-14 instead of the
# beat-combined variants; plus effective_rank/participation_ratio, which
# are already single acquisition-level numbers straight out of the SVD.
NATIVE_SCALAR_METRICS = [f"{name}_native" for name in CANDIDATE_METRICS] + [
    "effective_rank",
    "participation_ratio",
]


def load_block_and_period(h5_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(h5_path, "r") as f:
        if RAW_SEGMENT_PATH not in f or BEAT_PERIOD_PATH not in f:
            return None
        v_block = np.asarray(f[RAW_SEGMENT_PATH], dtype=float)
        T = np.asarray(f[BEAT_PERIOD_PATH], dtype=float)
    return v_block, T


def acquisition_beatwise_metrics(
    lowrank: LowRankPulsatilityMetrics,
    shape: CanonicalModeShapeMetrics,
    v_block: np.ndarray,
    T: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """
    One acquisition's per-beat values (already reduced over vessel location
    via median) for each name in CANDIDATE_METRICS, its NATIVE_SCALAR_METRICS
    counterparts, and its mean beat period.

    shape's own low-rank fit uses the same default configuration as
    lowrank, so its valid_column_mask is identical; reusing lowrank's mask
    here avoids recomputing that SVD a second time just to re-derive it.
    """
    rep = lowrank._compute_representation(v_block=v_block, T=T)
    if not rep.get("svd_available", False):
        return None

    beat_period_valid = rep["beat_period_valid"]
    beat_period = float(lowrank._safe_nanmean(np.asarray(T)[0][beat_period_valid]))
    valid_column_mask = rep["valid_column_mask"]

    beatwise = rep["beatwise"]
    acq = rep["acq"]
    out: dict[str, np.ndarray] = {
        "A1": beatwise["median_kr_A1"],
        "rho1": beatwise["rho1_beatwise"],
        "A1_native": float(acq["A1"]),
        "rho1_native": float(acq["rho1"]),
        "effective_rank": float(acq["effective_rank"]),
        "participation_ratio": float(acq["participation_ratio"]),
        "beat_period": beat_period,
    }

    shape_block = shape._compute_block(v_block, T)
    for name in shape.shape_metric_names:
        out[name] = lowrank._median_per_beat(shape_block[name], valid_column_mask)
        out[f"{name}_native"] = lowrank._safe_nanmedian(shape_block[name][valid_column_mask])

    return out


def collect_epoch_data(organized_root: Path) -> dict[str, dict[str, list]]:
    """
    results[metric][epoch] -> list of per-acquisition per-beat arrays for
    each name in CANDIDATE_METRICS, ready to hand to
    flicker_stats.confound_controlled_replication; a matching list of plain
    scalars for each name in NATIVE_SCALAR_METRICS; and
    results["beat_period"][epoch] -> matching list of mean beat periods.
    """
    lowrank = LowRankPulsatilityMetrics()
    shape = CanonicalModeShapeMetrics()

    results = {
        name: {epoch: [] for epoch in EPOCH_ORDER}
        for name in CANDIDATE_METRICS + NATIVE_SCALAR_METRICS
    }
    results["beat_period"] = {epoch: [] for epoch in EPOCH_ORDER}

    for epoch in EPOCH_ORDER:
        folder = organized_root / epoch
        if not folder.exists():
            print(f"  WARNING: missing folder {folder}")
            continue
        for h5_path in sorted(folder.glob("*.h5")):
            loaded = load_block_and_period(h5_path)
            if loaded is None:
                print(f"  skipping {h5_path.name}: missing required datasets")
                continue
            v_block, T = loaded
            print(f"  processing {epoch}/{h5_path.name} ...")

            acq = acquisition_beatwise_metrics(lowrank, shape, v_block, T)
            if acq is None:
                print(f"    {h5_path.name}: SVD unavailable, skipping")
                continue

            for name in CANDIDATE_METRICS + NATIVE_SCALAR_METRICS:
                results[name][epoch].append(acq[name])
            results["beat_period"][epoch].append(acq["beat_period"])

    return results


def print_replication_report(name: str, report: dict) -> None:
    print(f"\n=== {name} ===")
    for combo, r in report["combinations"].items():
        p = r["pooled_baseline_vs_flicker_p"]
        d = r["pooled_baseline_vs_flicker_delta"]
        print(f"  {combo:>17}: pooled p={p:>10.4g}  Cliff's delta={d:>7.3f}")
    print(f"  retained (significant, consistent direction, all 4 combos): {report['retained']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("organized_root", type=Path)
    args = parser.parse_args()

    results = collect_epoch_data(args.organized_root)
    beat_periods = results.pop("beat_period")

    for name in CANDIDATE_METRICS:
        report = confound_controlled_replication(results[name], beat_periods)
        print_replication_report(name, report)


if __name__ == "__main__":
    main()
