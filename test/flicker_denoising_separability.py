"""
Does denoising change how well baseline vs. flicker can be told apart?

Uses the real BOM0753 flicker session (organized into baseline1/flicker/
baseline2 folders by acquisition index, per organize_flicker_files.py) and
the low-rank A1/rho1 endpoints from lowrank_pulsatility_metrics.py -- the
same metrics the low-rank paper itself used to show a flicker effect (A1
down, rho1 up) -- computed AFTER running each candidate denoiser on the raw
segment data, instead of on the untouched raw signal.

Reuses the CANDIDATES list and denoiser interface from denoising_benchmark.py
rather than duplicating denoising logic; this script only adds the
epoch-comparison layer on top.

Not a pytest test -- a reporting script you run by hand. Not auto-collected
by pytest (filename doesn't match test_*.py).

Usage:
    python3 test/flicker_denoising_separability.py /path/to/organized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402

from denoising_benchmark import CANDIDATES, ArterialSegExample  # noqa: E402
from pipelines.lowrank_pulsatility_metrics import LowRankPulsatilityMetrics  # noqa: E402

RAW_SEGMENT_PATH = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
BEAT_PERIOD_PATH = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

EPOCH_FOLDERS = ["baseline1", "flicker", "baseline2"]


def load_block_and_period(h5_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(h5_path, "r") as f:
        if RAW_SEGMENT_PATH not in f or BEAT_PERIOD_PATH not in f:
            return None
        v_block = np.asarray(f[RAW_SEGMENT_PATH], dtype=float)
        T = np.asarray(f[BEAT_PERIOD_PATH], dtype=float)
    return v_block, T


def compute_a1_rho1(
    lowrank_helper: LowRankPulsatilityMetrics, v_block: np.ndarray, T: np.ndarray
) -> tuple[float, float]:
    rep = lowrank_helper._compute_representation(v_block=v_block, T=T)
    if not rep.get("svd_available", False):
        return np.nan, np.nan
    acq = rep["acq"]
    return float(acq.get("A1", np.nan)), float(acq.get("rho1", np.nan))


def compute_residual_rms(v_raw: np.ndarray, v_denoised: np.ndarray) -> float:
    """How big a correction the denoiser made: RMS(raw - denoised) over valid samples."""
    diff = v_raw - v_denoised
    mask = np.isfinite(diff)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def compute_residual_rms_per_branch(v_raw: np.ndarray, v_denoised: np.ndarray) -> np.ndarray:
    """Same idea as compute_residual_rms, but one number per branch (vessel)."""
    diff = v_raw - v_denoised  # (n_t, n_beats, n_branches, n_radii)
    n_branches = diff.shape[2]
    out = np.full(n_branches, np.nan)
    for b in range(n_branches):
        branch_diff = diff[:, :, b, :]
        mask = np.isfinite(branch_diff)
        if np.any(mask):
            out[b] = float(np.sqrt(np.mean(branch_diff[mask] ** 2)))
    return out


def run_all(organized_root: Path) -> dict:
    pipeline = ArterialSegExample()
    lowrank_helper = LowRankPulsatilityMetrics()

    # results[candidate_name][epoch] = {"A1": [...], "rho1": [...], "residual_rms": [...], "acq": [...]}
    results = {
        name: {
            epoch: {"A1": [], "rho1": [], "residual_rms": [], "acq": []}
            for epoch in EPOCH_FOLDERS
        }
        for name, _ in CANDIDATES
    }

    for epoch in EPOCH_FOLDERS:
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

            for name, accessor in CANDIDATES:
                if accessor is None:
                    denoised = v_block
                else:
                    method = accessor(pipeline)
                    try:
                        denoised, _diag = method(v_block)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"    {name}: EXCEPTION {type(exc).__name__}: {exc}, "
                            "falling back to raw for this acquisition"
                        )
                        denoised = v_block

                a1, rho1 = compute_a1_rho1(lowrank_helper, denoised, T)
                residual_rms = (
                    compute_residual_rms(v_block, denoised) if accessor is not None else np.nan
                )
                results[name][epoch]["A1"].append(a1)
                results[name][epoch]["rho1"].append(rho1)
                results[name][epoch]["residual_rms"].append(residual_rms)
                results[name][epoch]["acq"].append(h5_path.stem)

    return results


def _clean(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mann_whitney_and_auc(baseline: np.ndarray, flicker: np.ndarray) -> tuple[float, float]:
    if baseline.size < 2 or flicker.size < 2:
        return np.nan, np.nan
    stat, p = mannwhitneyu(flicker, baseline, alternative="two-sided")
    auc = float(stat / (baseline.size * flicker.size))
    return float(p), auc


def summarize(results: dict) -> list[dict]:
    rows = []
    for name in results:
        pooled_baseline_a1 = _clean(
            results[name]["baseline1"]["A1"] + results[name]["baseline2"]["A1"]
        )
        flicker_a1 = _clean(results[name]["flicker"]["A1"])
        pooled_baseline_rho1 = _clean(
            results[name]["baseline1"]["rho1"] + results[name]["baseline2"]["rho1"]
        )
        flicker_rho1 = _clean(results[name]["flicker"]["rho1"])

        p_a1, auc_a1 = mann_whitney_and_auc(pooled_baseline_a1, flicker_a1)
        p_rho1, auc_rho1 = mann_whitney_and_auc(pooled_baseline_rho1, flicker_rho1)

        rows.append(
            {
                "name": name,
                "n_baseline": int(pooled_baseline_a1.size),
                "n_flicker": int(flicker_a1.size),
                "median_A1_baseline": float(np.median(pooled_baseline_a1))
                if pooled_baseline_a1.size
                else np.nan,
                "median_A1_flicker": float(np.median(flicker_a1)) if flicker_a1.size else np.nan,
                "p_A1": p_a1,
                "auc_A1": auc_a1,
                "median_rho1_baseline": float(np.median(pooled_baseline_rho1))
                if pooled_baseline_rho1.size
                else np.nan,
                "median_rho1_flicker": float(np.median(flicker_rho1))
                if flicker_rho1.size
                else np.nan,
                "p_rho1": p_rho1,
                "auc_rho1": auc_rho1,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'candidate':>20} {'n_b':>4} {'n_f':>4} "
        f"{'A1_base':>9} {'A1_flick':>9} {'p_A1':>10} {'AUC_A1':>8}   "
        f"{'rho1_base':>10} {'rho1_flick':>10} {'p_rho1':>10} {'AUC_rho1':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:>20} {r['n_baseline']:>4} {r['n_flicker']:>4} "
            f"{r['median_A1_baseline']:>9.4f} {r['median_A1_flicker']:>9.4f} "
            f"{r['p_A1']:>10.4g} {r['auc_A1']:>8.3f}   "
            f"{r['median_rho1_baseline']:>10.4f} {r['median_rho1_flicker']:>10.4f} "
            f"{r['p_rho1']:>10.4g} {r['auc_rho1']:>9.3f}"
        )


def calibrated_zscores(
    baseline_residuals: list[float], flicker_residuals: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust z-score of "how big was the denoiser's correction" relative to what's
    normal for baseline. Baseline acquisitions are scored leave-one-out (each
    against the OTHER baseline acquisitions) so the comparison isn't rigged in
    baseline's favor; flicker acquisitions are scored against the full baseline
    reference set, since they were never part of calibrating it.
    """
    baseline_arr = np.asarray(baseline_residuals, dtype=float)
    flicker_arr = np.asarray(flicker_residuals, dtype=float)
    eps = 1e-12

    baseline_z = np.full(baseline_arr.shape, np.nan)
    for i in range(baseline_arr.size):
        if not np.isfinite(baseline_arr[i]):
            continue
        others = np.delete(baseline_arr, i)
        others = others[np.isfinite(others)]
        if others.size < 2:
            continue
        med = np.median(others)
        mad = np.median(np.abs(others - med)) * 1.4826
        baseline_z[i] = (baseline_arr[i] - med) / max(mad, eps)

    finite_baseline = baseline_arr[np.isfinite(baseline_arr)]
    flicker_z = np.full(flicker_arr.shape, np.nan)
    if finite_baseline.size >= 2:
        med_all = np.median(finite_baseline)
        mad_all = np.median(np.abs(finite_baseline - med_all)) * 1.4826
        flicker_z = (flicker_arr - med_all) / max(mad_all, eps)

    return baseline_z, flicker_z


def summarize_calibrated_anomaly(results: dict) -> list[dict]:
    rows = []
    for name in results:
        if name == "no_denoising (baseline)":
            continue  # raw - denoised is trivially zero here; not meaningful

        baseline_residuals = (
            results[name]["baseline1"]["residual_rms"]
            + results[name]["baseline2"]["residual_rms"]
        )
        flicker_residuals = results[name]["flicker"]["residual_rms"]

        baseline_z, flicker_z = calibrated_zscores(baseline_residuals, flicker_residuals)
        baseline_z_clean = baseline_z[np.isfinite(baseline_z)]
        flicker_z_clean = flicker_z[np.isfinite(flicker_z)]

        p, auc = mann_whitney_and_auc(baseline_z_clean, flicker_z_clean)
        rows.append(
            {
                "name": name,
                "n_baseline": int(baseline_z_clean.size),
                "n_flicker": int(flicker_z_clean.size),
                "median_z_baseline": float(np.median(baseline_z_clean))
                if baseline_z_clean.size
                else np.nan,
                "median_z_flicker": float(np.median(flicker_z_clean))
                if flicker_z_clean.size
                else np.nan,
                "p_z": p,
                "auc_z": auc,
            }
        )
    return rows


def print_anomaly_table(rows: list[dict]) -> None:
    header = (
        f"{'candidate':>20} {'n_b':>4} {'n_f':>4} "
        f"{'z_baseline':>11} {'z_flicker':>10} {'p_value':>10} {'AUC':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:>20} {r['n_baseline']:>4} {r['n_flicker']:>4} "
            f"{r['median_z_baseline']:>11.3f} {r['median_z_flicker']:>10.3f} "
            f"{r['p_z']:>10.4g} {r['auc_z']:>7.3f}"
        )


def plot_anomaly_separability(rows: list[dict], out_path: Path) -> None:
    names = [r["name"] for r in rows]
    sep = [abs(r["auc_z"] - 0.5) if np.isfinite(r["auc_z"]) else 0.0 for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, sep, color="seagreen")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("|AUC - 0.5|  (baseline-calibrated anomaly separability)")
    ax.set_title("Is a flicker acquisition's correction size abnormal vs. baseline?")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


def plot_separability(rows: list[dict], out_path: Path) -> None:
    names = [r["name"] for r in rows]
    # Separability strength: how far AUC is from 0.5 (chance), for each metric.
    sep_a1 = [abs(r["auc_A1"] - 0.5) if np.isfinite(r["auc_A1"]) else 0.0 for r in rows]
    sep_rho1 = [abs(r["auc_rho1"] - 0.5) if np.isfinite(r["auc_rho1"]) else 0.0 for r in rows]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, sep_a1, width, label="A1 (canonical amplitude)", color="steelblue")
    ax.bar(x + width / 2, sep_rho1, width, label="rho1 (residual fraction)", color="indianred")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("|AUC - 0.5|  (baseline vs. flicker separability)")
    ax.set_title("Does denoising help or hurt baseline-vs-flicker separability?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


DEFAULT_LOWRANK_K_SWEEP = [1, 2, 3, 5, 8, 12, 20, 30, 50]


def _collect_lowrank_residuals(organized_root: Path, k: int) -> dict:
    """
    Run the low-rank denoiser at rank K over every acquisition, once, returning
    both the whole-acquisition pooled residual RMS and the per-branch residual
    RMS array -- so a K-sweep and a per-branch comparison can both reuse a
    single pass over the files instead of reloading/redenoising repeatedly.
    """
    pipeline = ArterialSegExample()
    pipeline.lowrank.max_modes = k

    out = {epoch: {"pooled": [], "max_branch": []} for epoch in EPOCH_FOLDERS}
    for epoch in EPOCH_FOLDERS:
        folder = organized_root / epoch
        if not folder.exists():
            continue
        for h5_path in sorted(folder.glob("*.h5")):
            loaded = load_block_and_period(h5_path)
            if loaded is None:
                continue
            v_block, _T = loaded
            try:
                denoised, _diag = pipeline.lowrank._denoise_segment_block(v_block)
            except Exception:  # noqa: BLE001
                out[epoch]["pooled"].append(np.nan)
                out[epoch]["max_branch"].append(np.nan)
                continue
            out[epoch]["pooled"].append(compute_residual_rms(v_block, denoised))
            per_branch = compute_residual_rms_per_branch(v_block, denoised)
            finite = per_branch[np.isfinite(per_branch)]
            out[epoch]["max_branch"].append(float(np.max(finite)) if finite.size else np.nan)
    return out


def _pooled_baseline_flicker(residuals_by_epoch: dict, feature: str) -> tuple[list, list]:
    baseline = residuals_by_epoch["baseline1"][feature] + residuals_by_epoch["baseline2"][feature]
    flicker = residuals_by_epoch["flicker"][feature]
    return baseline, flicker


def run_lowrank_k_sweep(organized_root: Path, k_values: list[int]) -> list[dict]:
    rows = []
    for k in k_values:
        print(f"  sweeping lowrank max_modes = {k} ...")
        residuals = _collect_lowrank_residuals(organized_root, k)
        baseline, flicker = _pooled_baseline_flicker(residuals, "pooled")
        baseline_z, flicker_z = calibrated_zscores(baseline, flicker)
        baseline_z = baseline_z[np.isfinite(baseline_z)]
        flicker_z = flicker_z[np.isfinite(flicker_z)]
        p, auc = mann_whitney_and_auc(baseline_z, flicker_z)
        rows.append({"k": k, "p": p, "auc": auc})
    return rows


def print_k_sweep_table(rows: list[dict]) -> None:
    print(f"{'K (max_modes)':>15} {'p_value':>10} {'AUC':>8} {'|AUC-0.5|':>10}")
    print("-" * 46)
    for r in rows:
        sep = abs(r["auc"] - 0.5) if np.isfinite(r["auc"]) else float("nan")
        print(f"{r['k']:>15} {r['p']:>10.4g} {r['auc']:>8.3f} {sep:>10.3f}")


def plot_k_sweep(rows: list[dict], out_path: Path) -> None:
    ks = [r["k"] for r in rows]
    sep = [abs(r["auc"] - 0.5) if np.isfinite(r["auc"]) else 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sep, marker="o", color="darkorange")
    ax.set_xlabel("K (lowrank max_modes)")
    ax.set_ylabel("|AUC - 0.5|  (baseline-calibrated anomaly separability)")
    ax.set_title("lowrank_svd: does rank K tuned for detection beat the fidelity-tuned default (12)?")
    ax.axvline(12, color="gray", linestyle="--", linewidth=1, label="current default (K=12)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


def run_worst_branch_comparison(organized_root: Path, k: int) -> list[dict]:
    """
    Compare pooling the residual over the whole acquisition (all branches
    averaged together) against taking the single worst (most anomalous)
    branch. NOTE: branch identity/count isn't stable across acquisitions here
    (different acquisitions were captured with different EyeFlow branch
    configs), so "worst branch" means the most-anomalous vessel *within* that
    acquisition's own set, not "branch index 3 across all acquisitions."
    """
    residuals = _collect_lowrank_residuals(organized_root, k)
    rows = []
    for feature, label in [("pooled", "whole-acquisition average"), ("max_branch", "worst single branch")]:
        baseline, flicker = _pooled_baseline_flicker(residuals, feature)
        baseline_z, flicker_z = calibrated_zscores(baseline, flicker)
        baseline_z = baseline_z[np.isfinite(baseline_z)]
        flicker_z = flicker_z[np.isfinite(flicker_z)]
        p, auc = mann_whitney_and_auc(baseline_z, flicker_z)
        rows.append({"feature": label, "n_baseline": baseline_z.size, "n_flicker": flicker_z.size, "p": p, "auc": auc})
    return rows


def print_worst_branch_table(rows: list[dict]) -> None:
    print(f"{'feature':>28} {'n_b':>4} {'n_f':>4} {'p_value':>10} {'AUC':>7}")
    print("-" * 58)
    for r in rows:
        print(f"{r['feature']:>28} {r['n_baseline']:>4} {r['n_flicker']:>4} {r['p']:>10.4g} {r['auc']:>7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("organized_root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/flicker_separability.png")
    )
    parser.add_argument(
        "--anomaly-out", type=Path, default=Path("/tmp/flicker_anomaly.png")
    )
    parser.add_argument(
        "--k-sweep-out", type=Path, default=Path("/tmp/flicker_lowrank_k_sweep.png")
    )
    args = parser.parse_args()

    results = run_all(args.organized_root)

    rows = summarize(results)
    print()
    print_table(rows)
    plot_separability(rows, args.out)

    anomaly_rows = summarize_calibrated_anomaly(results)
    print()
    print("=== Baseline-calibrated residual-size anomaly detection ===")
    print_anomaly_table(anomaly_rows)
    plot_anomaly_separability(anomaly_rows, args.anomaly_out)

    print()
    print("=== lowrank_svd: rank K sweep, tuned for anomaly separability ===")
    k_sweep_rows = run_lowrank_k_sweep(args.organized_root, DEFAULT_LOWRANK_K_SWEEP)
    print()
    print_k_sweep_table(k_sweep_rows)
    plot_k_sweep(k_sweep_rows, args.k_sweep_out)

    best_k = max(k_sweep_rows, key=lambda r: abs(r["auc"] - 0.5) if np.isfinite(r["auc"]) else -1)["k"]
    print()
    print(f"=== lowrank_svd: whole-acquisition average vs. worst single branch (K={best_k}) ===")
    worst_branch_rows = run_worst_branch_comparison(args.organized_root, best_k)
    print_worst_branch_table(worst_branch_rows)


if __name__ == "__main__":
    main()
