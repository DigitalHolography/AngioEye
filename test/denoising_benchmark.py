"""
Manual validation/benchmark harness for the candidate denoisers in
waveform_shape_metrics_denoised_alternatives.py.

This is NOT a pytest test (no assertions) -- it's a reporting script you run by
hand against real acquisition files, per the approved plan at
~/.claude/plans/polymorphic-hugging-leaf.md. Not auto-collected by pytest
(filename doesn't match test_*.py).

Usage:
    python3 test/denoising_benchmark.py validate /path/to/folder [--limit N]
    python3 test/denoising_benchmark.py sweep-gamma /path/to/folder [--out FILE.png]

Phase 1 (current): validate the ported graph-Laplacian denoiser on real data,
and sweep its `gamma` regularization strength since the ported default
(0.75) turned out to have no documented derivation and produced negligible
smoothing on real data. Later phases will add the SVD/low-rank and Kalman
candidates to this same harness for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

import h5py  # noqa: E402

from pipelines.waveform_shape_metrics_denoised_alternatives import (  # noqa: E402
    ArterialSegExample,
)

RAW_SEGMENT_PATH = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"

STATUS_LABELS = {
    0: "denoised",
    1: "all_nan",
    2: "too_sparse",
    3: "low_variance",
    4: "no_eligible_graph_neighbors",
    5: "graph_solve_failed",
    6: "singleton_graph_component",
    7: "low_reference_corr",
}


def load_raw_segment_block(h5_path: Path) -> np.ndarray | None:
    with h5py.File(h5_path, "r") as f:
        if RAW_SEGMENT_PATH not in f:
            return None
        return np.asarray(f[RAW_SEGMENT_PATH], dtype=float)


def summarize_array(name: str, arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"  {name}: all non-finite ({arr.size} values)"
    return (
        f"  {name}: n={finite.size}/{arr.size} finite, "
        f"mean={np.mean(finite):.4f}, median={np.median(finite):.4f}, "
        f"min={np.min(finite):.4f}, max={np.max(finite):.4f}"
    )


def validate_laplacian_on_file(pipeline: ArterialSegExample, h5_path: Path) -> None:
    v_block = load_raw_segment_block(h5_path)
    print(f"\n=== {h5_path.name} ===")
    if v_block is None:
        print("  raw segment path missing, skipping")
        return
    print(f"  input shape (n_t,n_beats,n_branches,n_radii) = {v_block.shape}")

    try:
        denoised, diag = pipeline.laplacian._denoise_segment_block(v_block)
    except Exception as exc:  # noqa: BLE001
        print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        return

    status = diag["status_code"]
    counts = Counter(int(s) for s in status.ravel())
    total = status.size
    print(f"  status_code distribution (n={total}):")
    for code in sorted(counts):
        label = STATUS_LABELS.get(code, f"unknown({code})")
        print(f"    {code} ({label}): {counts[code]}  [{100*counts[code]/total:.1f}%]")

    print(summarize_array("original_vs_filtered_corr", diag["original_vs_filtered_corr"]))
    print(summarize_array("finite_fraction", diag["finite_fraction"]))
    print(summarize_array("phase_alignment_corr", diag["phase_alignment_corr"]))
    print(summarize_array("best_neighbor_corr", diag["best_neighbor_corr"]))
    print(summarize_array("neighbor_count", diag["neighbor_count"].astype(float)))
    print(summarize_array("effective_neighbor_count", diag["effective_neighbor_count"]))
    print(summarize_array("graph_degree", diag["graph_degree"]))
    print(summarize_array("reference_corr", diag["reference_corr"]))
    print(
        f"  reference gate: kept={diag['reference_kept_count']}, "
        f"rejected={diag['reference_rejected_count']}"
    )
    print(
        f"  graph: edges={diag['graph_edge_count']}, "
        f"components={diag['graph_component_count']}, "
        f"valid_pulses={diag['valid_pulse_count']}"
    )
    print(
        f"  dirichlet energy: before={diag['dirichlet_energy_before']:.4f}, "
        f"after={diag['dirichlet_energy_after']:.4f}"
    )
    print(f"  system_condition_number: {diag['system_condition_number']:.4e}")

    denoised_finite = np.sum(np.isfinite(denoised))
    input_finite = np.sum(np.isfinite(v_block))
    print(f"  finite samples: input={input_finite}, output={denoised_finite}")


LOWRANK_STATUS_LABELS = {
    0: "denoised",
    1: "all_nan",
    2: "excluded_by_lowrank_gate",
    3: "svd_unavailable",
}


def validate_lowrank_on_file(pipeline: ArterialSegExample, h5_path: Path) -> None:
    v_block = load_raw_segment_block(h5_path)
    print(f"\n=== {h5_path.name} ===")
    if v_block is None:
        print("  raw segment path missing, skipping")
        return
    print(f"  input shape (n_t,n_beats,n_branches,n_radii) = {v_block.shape}")

    try:
        denoised, diag = pipeline.lowrank._denoise_segment_block(v_block)
    except Exception as exc:  # noqa: BLE001
        print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        return

    status = diag["status_code"]
    counts = Counter(int(s) for s in status.ravel())
    total = status.size
    print(f"  status_code distribution (n={total}):")
    for code in sorted(counts):
        label = LOWRANK_STATUS_LABELS.get(code, f"unknown({code})")
        print(f"    {code} ({label}): {counts[code]}  [{100*counts[code]/total:.1f}%]")

    print(summarize_array("original_vs_filtered_corr", diag["original_vs_filtered_corr"]))
    print(summarize_array("finite_fraction", diag["finite_fraction"]))
    print(f"  n_modes_used: {diag['n_modes_used']}  (svd_reason={diag['svd_reason']})")
    print(f"  effective_rank: {diag['effective_rank']}")
    print(f"  eta1 (mode-1 energy fraction): {diag['eta1']}")
    print(f"  eta23 (mode-2+3 energy fraction): {diag['eta23']}")

    denoised_finite = np.sum(np.isfinite(denoised))
    input_finite = np.sum(np.isfinite(v_block))
    print(f"  finite samples: input={input_finite}, output={denoised_finite}")


KALMAN_STATUS_LABELS = {
    0: "denoised",
    1: "all_nan",
    2: "too_sparse",
    3: "axis_too_sparse",
}


def validate_kalman_on_file(
    pipeline: ArterialSegExample, h5_path: Path, axis: str
) -> None:
    v_block = load_raw_segment_block(h5_path)
    print(f"\n=== {h5_path.name} (axis={axis}) ===")
    if v_block is None:
        print("  raw segment path missing, skipping")
        return
    print(f"  input shape (n_t,n_beats,n_branches,n_radii) = {v_block.shape}")

    method = (
        pipeline.kalman._denoise_segment_block_beat_axis
        if axis == "beat"
        else pipeline.kalman._denoise_segment_block_radius_axis
    )
    try:
        denoised, diag = method(v_block)
    except Exception as exc:  # noqa: BLE001
        print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        return

    status = diag["status_code"]
    counts = Counter(int(s) for s in status.ravel())
    total = status.size
    print(f"  status_code distribution (n={total}):")
    for code in sorted(counts):
        label = KALMAN_STATUS_LABELS.get(code, f"unknown({code})")
        print(f"    {code} ({label}): {counts[code]}  [{100*counts[code]/total:.1f}%]")

    print(summarize_array("original_vs_filtered_corr", diag["original_vs_filtered_corr"]))
    print(summarize_array("finite_fraction", diag["finite_fraction"]))

    denoised_finite = np.sum(np.isfinite(denoised))
    input_finite = np.sum(np.isfinite(v_block))
    print(f"  finite samples: input={input_finite}, output={denoised_finite}")


DEFAULT_GAMMAS = [0.0, 0.01, 0.03, 0.1, 0.3, 0.75, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def gamma_run_on_file(
    pipeline: ArterialSegExample, v_block: np.ndarray, gamma: float
) -> dict | None:
    """Run the Laplacian denoiser once at a given gamma; return per-file summary."""
    pipeline.laplacian.gamma = gamma
    try:
        _denoised, diag = pipeline.laplacian._denoise_segment_block(v_block)
    except Exception as exc:  # noqa: BLE001
        print(f"    gamma={gamma}: EXCEPTION {type(exc).__name__}: {exc}")
        return None

    valid = diag["status_code"] == 0
    if not np.any(valid):
        return None

    mean_abs_change = diag["mean_abs_change"][valid]
    corr = diag["original_vs_filtered_corr"][valid]
    return {
        "gamma": gamma,
        "n_valid": int(np.sum(valid)),
        "mean_abs_change": float(np.nanmean(mean_abs_change)),
        "corr": float(np.nanmean(corr)),
        "dirichlet_before": float(diag["dirichlet_energy_before"]),
        "dirichlet_after": float(diag["dirichlet_energy_after"]),
        "condition_number": float(diag["system_condition_number"]),
    }


def run_gamma_sweep(
    h5_files: list[Path], gammas: list[float]
) -> dict[float, list[dict]]:
    pipeline = ArterialSegExample()
    per_gamma: dict[float, list[dict]] = {g: [] for g in gammas}

    for h5_path in h5_files:
        v_block = load_raw_segment_block(h5_path)
        if v_block is None:
            continue
        print(f"  sweeping {h5_path.name} ...")
        for gamma in gammas:
            result = gamma_run_on_file(pipeline, v_block, gamma)
            if result is not None:
                per_gamma[gamma].append(result)

    return per_gamma


def summarize_gamma_sweep(per_gamma: dict[float, list[dict]]) -> list[dict]:
    """Aggregate (median across files) one row per gamma value."""
    rows = []
    for gamma, results in per_gamma.items():
        if not results:
            continue
        rows.append(
            {
                "gamma": gamma,
                "n_files": len(results),
                "median_mean_abs_change": float(
                    np.median([r["mean_abs_change"] for r in results])
                ),
                "median_corr": float(np.median([r["corr"] for r in results])),
                "median_dirichlet_after": float(
                    np.median([r["dirichlet_after"] for r in results])
                ),
                "median_dirichlet_ratio": float(
                    np.median(
                        [
                            r["dirichlet_after"] / max(r["dirichlet_before"], 1e-12)
                            for r in results
                        ]
                    )
                ),
                "median_condition_number": float(
                    np.median([r["condition_number"] for r in results])
                ),
            }
        )
    rows.sort(key=lambda r: r["gamma"])
    return rows


def print_gamma_sweep_table(rows: list[dict]) -> None:
    header = (
        f"{'gamma':>10} {'n_files':>8} {'med_corr':>10} {'med_abs_chg':>12} "
        f"{'med_dirichlet_ratio':>20} {'med_cond#':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['gamma']:>10.3g} {r['n_files']:>8d} {r['median_corr']:>10.4f} "
            f"{r['median_mean_abs_change']:>12.4f} {r['median_dirichlet_ratio']:>20.4f} "
            f"{r['median_condition_number']:>12.3g}"
        )


def plot_gamma_sweep(rows: list[dict], out_path: Path) -> None:
    gammas = [r["gamma"] for r in rows]
    corr = [r["median_corr"] for r in rows]
    abs_change = [r["median_mean_abs_change"] for r in rows]
    dirichlet_ratio = [r["median_dirichlet_ratio"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(gammas, corr, "o-")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("gamma")
    ax.set_ylabel("median original_vs_filtered_corr")
    ax.set_title("Self-consistency vs gamma")
    ax.axvline(0.75, color="gray", linestyle="--", alpha=0.6, label="ported default (0.75)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(gammas, abs_change, "o-", color="darkorange")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("gamma")
    ax.set_ylabel("median mean_abs_change (original vs denoised)")
    ax.set_title("Denoising magnitude vs gamma")
    ax.axvline(0.75, color="gray", linestyle="--", alpha=0.6)

    ax = axes[2]
    ax.plot(dirichlet_ratio, abs_change, "o-", color="crimson")
    for r in rows:
        ax.annotate(
            f"{r['gamma']:g}",
            (r["median_dirichlet_ratio"], r["median_mean_abs_change"]),
            fontsize=7,
            textcoords="offset points",
            xytext=(4, 4),
        )
    ax.set_xlabel("Dirichlet-energy ratio (after/before) -- lower = smoother")
    ax.set_ylabel("mean_abs_change (fit residual)")
    ax.set_title("L-curve: fit vs. smoothness, by gamma")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


DEFAULT_MODE_COUNTS = [1, 2, 3, 5, 8, 12, 20, 30, 50]


def modes_run_on_file(
    pipeline: ArterialSegExample, v_block: np.ndarray, n_modes: int
) -> dict | None:
    """Run the low-rank denoiser once at a given rank K; return per-file summary."""
    pipeline.lowrank.max_modes = n_modes
    try:
        _denoised, diag = pipeline.lowrank._denoise_segment_block(v_block)
    except Exception as exc:  # noqa: BLE001
        print(f"    K={n_modes}: EXCEPTION {type(exc).__name__}: {exc}")
        return None

    valid = diag["status_code"] == 0
    if not np.any(valid):
        return None

    corr = diag["original_vs_filtered_corr"][valid]
    return {
        "n_modes": n_modes,
        "n_modes_used": diag["n_modes_used"],
        "n_valid": int(np.sum(valid)),
        "corr": float(np.nanmean(corr)),
        "captured_energy_fraction": float(diag["captured_energy_fraction"]),
        "effective_rank": float(diag["effective_rank"])
        if np.isfinite(diag["effective_rank"])
        else None,
    }


def run_modes_sweep(
    h5_files: list[Path], mode_counts: list[int]
) -> dict[int, list[dict]]:
    pipeline = ArterialSegExample()
    per_k: dict[int, list[dict]] = {k: [] for k in mode_counts}

    for h5_path in h5_files:
        v_block = load_raw_segment_block(h5_path)
        if v_block is None:
            continue
        print(f"  sweeping {h5_path.name} ...")
        for k in mode_counts:
            result = modes_run_on_file(pipeline, v_block, k)
            if result is not None:
                per_k[k].append(result)

    return per_k


def summarize_modes_sweep(per_k: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for k, results in per_k.items():
        if not results:
            continue
        rows.append(
            {
                "n_modes": k,
                "n_files": len(results),
                "median_corr": float(np.median([r["corr"] for r in results])),
                "median_captured_energy_fraction": float(
                    np.median([r["captured_energy_fraction"] for r in results])
                ),
                "median_effective_rank": float(
                    np.median(
                        [r["effective_rank"] for r in results if r["effective_rank"]]
                    )
                ),
            }
        )
    rows.sort(key=lambda r: r["n_modes"])
    return rows


def print_modes_sweep_table(rows: list[dict]) -> None:
    header = (
        f"{'K':>6} {'n_files':>8} {'med_corr':>10} "
        f"{'med_captured_energy':>20} {'med_effective_rank':>20}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n_modes']:>6d} {r['n_files']:>8d} {r['median_corr']:>10.4f} "
            f"{r['median_captured_energy_fraction']:>20.4f} "
            f"{r['median_effective_rank']:>20.2f}"
        )


def plot_modes_sweep(rows: list[dict], out_path: Path) -> None:
    ks = [r["n_modes"] for r in rows]
    corr = [r["median_corr"] for r in rows]
    captured = [r["median_captured_energy_fraction"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ks, corr, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("K (rank / number of modes kept)")
    ax.set_ylabel("median original_vs_filtered_corr")
    ax.set_title("Fit fidelity vs. rank K")
    ax.axvline(3, color="gray", linestyle="--", alpha=0.6, label="reused default (K=3)")
    med_eff_rank = np.median([r["median_effective_rank"] for r in rows])
    ax.axvline(
        med_eff_rank, color="green", linestyle=":", alpha=0.6,
        label=f"median effective_rank ({med_eff_rank:.1f})",
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ks, captured, "o-", color="darkorange")
    ax.set_xscale("log")
    ax.set_xlabel("K (rank / number of modes kept)")
    ax.set_ylabel("median captured energy fraction")
    ax.set_title("Variance retained vs. rank K")
    ax.axvline(3, color="gray", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    pipeline = ArterialSegExample()
    for h5_path in h5_files:
        validate_laplacian_on_file(pipeline, h5_path)


def cmd_validate_lowrank(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    pipeline = ArterialSegExample()
    for h5_path in h5_files:
        validate_lowrank_on_file(pipeline, h5_path)


def cmd_validate_kalman(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    pipeline = ArterialSegExample()
    for h5_path in h5_files:
        validate_kalman_on_file(pipeline, h5_path, axis=args.axis)


def cmd_sweep_gamma(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    gammas = args.gammas if args.gammas else DEFAULT_GAMMAS
    per_gamma = run_gamma_sweep(h5_files, gammas)
    rows = summarize_gamma_sweep(per_gamma)
    print()
    print_gamma_sweep_table(rows)
    plot_gamma_sweep(rows, args.out)


def cmd_sweep_modes(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    mode_counts = args.modes if args.modes else DEFAULT_MODE_COUNTS
    per_k = run_modes_sweep(h5_files, mode_counts)
    rows = summarize_modes_sweep(per_k)
    print()
    print_modes_sweep_table(rows)
    plot_modes_sweep(rows, args.out)


# ----------------------------
# Phase 4, Tier A: synthetic ground-truth benchmark
# ----------------------------
#
# Real retinal pulses have no known "true" clean signal, so this is the only
# way to get an objective ranking across candidates rather than eyeballing
# before/after plots. Generates a synthetic beat/branch/radius ensemble with
# a known clean pulse shape, injects noise (Gaussian/speckle noise, spike
# outliers, missing gaps, some fully-missing pulses), then scores each
# candidate by correlation-to-truth and RMSE-to-truth -- the ground truth
# being the exact clean waveform before noise injection, which no real
# acquisition can give us.

# Each accessor takes the pipeline instance and returns the bound method to
# call with a v_block. Laplacian/lowrank/Kalman live on nested denoiser
# objects (pipeline.laplacian / .lowrank / .kalman) rather than directly on
# the pipeline.
CANDIDATES = [
    ("no_denoising (baseline)", None),
    ("six_step_chain", lambda p: p._denoise_segment_block),
    ("graph_laplacian", lambda p: p.laplacian._denoise_segment_block),
    ("lowrank_svd", lambda p: p.lowrank._denoise_segment_block),
    ("kalman_beat_axis", lambda p: p.kalman._denoise_segment_block_beat_axis),
    ("kalman_radius_axis", lambda p: p.kalman._denoise_segment_block_radius_axis),
]


def synthetic_pulse_shape(n_time: int) -> np.ndarray:
    """One canonical arterial-like pulse: fast systolic upstroke/peak, a
    smaller dicrotic-notch bump, decaying back to a diastolic floor."""
    phase = np.arange(n_time, dtype=float) / n_time
    peak = np.exp(-0.5 * ((phase - 0.15) / 0.07) ** 2)
    notch = 0.35 * np.exp(-0.5 * ((phase - 0.45) / 0.10) ** 2)
    baseline = 0.15
    return baseline + peak + notch


def generate_synthetic_ensemble(
    n_time: int, n_beats: int, n_branches: int, n_radii: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Clean (noise-free) ensemble.

    - branch: categorical -- different vessels entirely, so each branch gets
      an independent, unordered amplitude scale (no expected similarity
      between neighboring branch indices).
    - radius: physically continuous and ordered -- position across one
      vessel's cross-section, so it gets a smooth, monotonic profile
      (fastest near the centre at r=0, slower toward the vessel wall at
      r=n_radii-1), loosely mirroring a Poiseuille-like velocity profile.
      This smoothness is what the radius-axis Kalman candidate assumes and
      is meant to recover -- an i.i.d. random profile here would test the
      wrong thing.
    - beat: each beat gets a small amplitude/timing jitter shared across the
      whole vascular tree for that heartbeat (genuine physiological
      variability a good denoiser should preserve, not erase).
    """
    base_shape = synthetic_pulse_shape(n_time)
    branch_scale = rng.uniform(0.6, 1.4, size=n_branches)

    radius_frac = np.arange(n_radii, dtype=float) / max(1, n_radii - 1)
    radius_scale = 1.3 - 0.9 * radius_frac**2

    v_true = np.zeros((n_time, n_beats, n_branches, n_radii), dtype=float)
    for beat_idx in range(n_beats):
        amp_jitter = rng.normal(1.0, 0.05)
        time_jitter = int(rng.integers(-2, 3))
        shifted_shape = np.roll(base_shape, time_jitter)
        for branch_idx in range(n_branches):
            for radius_idx in range(n_radii):
                amp = (
                    30.0
                    * branch_scale[branch_idx]
                    * radius_scale[radius_idx]
                    * amp_jitter
                )
                v_true[:, beat_idx, branch_idx, radius_idx] = amp * shifted_shape
    return v_true


def inject_synthetic_noise(
    v_true: np.ndarray,
    rng: np.random.Generator,
    noise_std_fraction: float = 0.06,
    spike_prob: float = 0.02,
    spike_scale: float = 6.0,
    gap_prob_per_pulse: float = 0.15,
    gap_len_range: tuple[int, int] = (3, 6),
    all_nan_prob: float = 0.03,
) -> np.ndarray:
    """Gaussian noise + occasional spikes (motion artifact) + short missing
    gaps + a few fully-missing pulses."""
    v_noisy = v_true.copy()
    n_time, n_beats, n_branches, n_radii = v_true.shape

    noise_std = noise_std_fraction * float(np.nanmax(v_true))
    v_noisy += rng.normal(0.0, noise_std, size=v_true.shape)

    spike_mask = rng.random(v_true.shape) < spike_prob
    spikes = rng.normal(0.0, spike_scale * noise_std, size=v_true.shape)
    v_noisy = np.where(spike_mask, v_noisy + spikes, v_noisy)

    for beat_idx in range(n_beats):
        for branch_idx in range(n_branches):
            for radius_idx in range(n_radii):
                if rng.random() < all_nan_prob:
                    v_noisy[:, beat_idx, branch_idx, radius_idx] = np.nan
                    continue
                if rng.random() < gap_prob_per_pulse:
                    gap_len = int(rng.integers(gap_len_range[0], gap_len_range[1] + 1))
                    start = int(rng.integers(0, max(1, n_time - gap_len)))
                    v_noisy[start : start + gap_len, beat_idx, branch_idx, radius_idx] = (
                        np.nan
                    )

    return v_noisy


def _corr_rmse_to_truth(
    pipeline: ArterialSegExample, denoised: np.ndarray, valid_mask: np.ndarray, v_true: np.ndarray
) -> tuple[list[float], list[float]]:
    corrs, rmses = [], []
    for beat_idx, branch_idx, radius_idx in np.argwhere(valid_mask):
        d = denoised[:, beat_idx, branch_idx, radius_idx]
        t = v_true[:, beat_idx, branch_idx, radius_idx]
        finite = np.isfinite(d) & np.isfinite(t)
        if np.sum(finite) < 3:
            continue
        corrs.append(pipeline._pearson_corr(d[finite], t[finite]))
        rmses.append(float(np.sqrt(np.mean((d[finite] - t[finite]) ** 2))))
    return corrs, rmses


def run_synthetic_benchmark(
    pipeline: ArterialSegExample, v_noisy: np.ndarray, v_true: np.ndarray
) -> list[dict]:
    rows = []
    for name, accessor in CANDIDATES:
        if accessor is None:
            # "no denoising" baseline: score the raw noisy input itself
            valid_mask = np.any(np.isfinite(v_noisy), axis=0)
            corrs, rmses = _corr_rmse_to_truth(pipeline, v_noisy, valid_mask, v_true)
            rows.append(
                {
                    "name": name,
                    "n_valid": int(np.sum(valid_mask)),
                    "median_corr_to_truth": float(np.median(corrs)) if corrs else np.nan,
                    "median_rmse_to_truth": float(np.median(rmses)) if rmses else np.nan,
                }
            )
            continue

        method = accessor(pipeline)
        try:
            denoised, diag = method(v_noisy)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "name": name,
                    "n_valid": 0,
                    "median_corr_to_truth": np.nan,
                    "median_rmse_to_truth": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        valid_mask = diag["status_code"] == 0
        corrs, rmses = _corr_rmse_to_truth(pipeline, denoised, valid_mask, v_true)
        rows.append(
            {
                "name": name,
                "n_valid": int(np.sum(valid_mask)),
                "median_corr_to_truth": float(np.median(corrs)) if corrs else np.nan,
                "median_rmse_to_truth": float(np.median(rmses)) if rmses else np.nan,
            }
        )
    return rows


def print_synthetic_benchmark_table(rows: list[dict]) -> None:
    header = f"{'candidate':>26} {'n_valid':>8} {'med_corr_to_truth':>18} {'med_rmse_to_truth':>18}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(f"{r['name']:>26} {'ERROR':>8} {r['error']}")
            continue
        print(
            f"{r['name']:>26} {r['n_valid']:>8d} "
            f"{r['median_corr_to_truth']:>18.4f} {r['median_rmse_to_truth']:>18.4f}"
        )


def plot_synthetic_benchmark(rows: list[dict], out_path: Path) -> None:
    names = [r["name"] for r in rows]
    corr = [r["median_corr_to_truth"] for r in rows]
    rmse = [r["median_rmse_to_truth"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.barh(names, corr, color="steelblue")
    ax.set_xlabel("median correlation to ground truth (higher is better)")
    ax.set_title("Fit fidelity vs. ground truth")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(names, rmse, color="indianred")
    ax.set_xlabel("median RMSE to ground truth (lower is better)")
    ax.set_title("Reconstruction error vs. ground truth")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


def cmd_synthetic_benchmark(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    v_true = generate_synthetic_ensemble(
        n_time=args.n_time,
        n_beats=args.n_beats,
        n_branches=args.n_branches,
        n_radii=args.n_radii,
        rng=rng,
    )
    v_noisy = inject_synthetic_noise(v_true, rng=rng)

    pipeline = ArterialSegExample()
    rows = run_synthetic_benchmark(pipeline, v_noisy, v_true)
    print()
    print_synthetic_benchmark_table(rows)
    plot_synthetic_benchmark(rows, args.out)


# ----------------------------
# Phase 4, Tier B: combined real-data robustness pass
# ----------------------------
#
# Doesn't rank methods (no ground truth on real data) -- just confirms none
# of the candidates crash or misbehave across a batch of real files, side by
# side, before investing further in any one of them.


def run_robustness_check(h5_files: list[Path]) -> list[dict]:
    pipeline = ArterialSegExample()
    rows = []
    for name, accessor in CANDIDATES:
        if accessor is None:
            continue
        method = accessor(pipeline)
        n_files_ok = 0
        n_files_error = 0
        all_corrs = []
        for h5_path in h5_files:
            v_block = load_raw_segment_block(h5_path)
            if v_block is None:
                continue
            try:
                _denoised, diag = method(v_block)
            except Exception as exc:  # noqa: BLE001
                n_files_error += 1
                print(f"  {name} on {h5_path.name}: EXCEPTION {type(exc).__name__}: {exc}")
                continue
            n_files_ok += 1
            valid = diag["status_code"] == 0
            corr = diag["original_vs_filtered_corr"][valid]
            all_corrs.extend(corr[np.isfinite(corr)].tolist())

        rows.append(
            {
                "name": name,
                "n_files_ok": n_files_ok,
                "n_files_error": n_files_error,
                "median_corr": float(np.median(all_corrs)) if all_corrs else np.nan,
            }
        )
    return rows


def print_robustness_table(rows: list[dict]) -> None:
    header = f"{'candidate':>26} {'files_ok':>10} {'files_error':>12} {'median_corr':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:>26} {r['n_files_ok']:>10d} {r['n_files_error']:>12d} "
            f"{r['median_corr']:>14.4f}"
        )


def cmd_robustness_check(args: argparse.Namespace) -> None:
    h5_files = sorted(args.folder.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    rows = run_robustness_check(h5_files)
    print()
    print_robustness_table(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate", help="Run diagnostics on real files")
    p_validate.add_argument("folder", type=Path)
    p_validate.add_argument("--limit", type=int, default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_validate_lr = subparsers.add_parser(
        "validate-lowrank", help="Run diagnostics for the SVD/low-rank candidate"
    )
    p_validate_lr.add_argument("folder", type=Path)
    p_validate_lr.add_argument("--limit", type=int, default=None)
    p_validate_lr.set_defaults(func=cmd_validate_lowrank)

    p_validate_kf = subparsers.add_parser(
        "validate-kalman", help="Run diagnostics for the Kalman-smoother candidates"
    )
    p_validate_kf.add_argument("folder", type=Path)
    p_validate_kf.add_argument("--limit", type=int, default=None)
    p_validate_kf.add_argument(
        "--axis", choices=["beat", "radius"], default="beat", help="Which axis to smooth along"
    )
    p_validate_kf.set_defaults(func=cmd_validate_kalman)

    p_sweep = subparsers.add_parser(
        "sweep-gamma", help="Sweep laplacian_gamma and report fit-vs-smoothness"
    )
    p_sweep.add_argument("folder", type=Path)
    p_sweep.add_argument("--limit", type=int, default=None)
    p_sweep.add_argument(
        "--gammas", type=float, nargs="+", default=None, help="Override the gamma values to sweep"
    )
    p_sweep.add_argument(
        "--out", type=Path, default=Path("/tmp/laplacian_gamma_sweep.png"), help="Plot output path"
    )
    p_sweep.set_defaults(func=cmd_sweep_gamma)

    p_sweep_modes = subparsers.add_parser(
        "sweep-modes", help="Sweep lowrank_max_modes (rank K) and report fit-vs-compression"
    )
    p_sweep_modes.add_argument("folder", type=Path)
    p_sweep_modes.add_argument("--limit", type=int, default=None)
    p_sweep_modes.add_argument(
        "--modes", type=int, nargs="+", default=None, help="Override the K values to sweep"
    )
    p_sweep_modes.add_argument(
        "--out", type=Path, default=Path("/tmp/lowrank_modes_sweep.png"), help="Plot output path"
    )
    p_sweep_modes.set_defaults(func=cmd_sweep_modes)

    p_synth = subparsers.add_parser(
        "synthetic-benchmark",
        help="Objective fit-to-ground-truth comparison across all candidates",
    )
    p_synth.add_argument("--n-time", type=int, default=64)
    p_synth.add_argument("--n-beats", type=int, default=8)
    p_synth.add_argument("--n-branches", type=int, default=5)
    p_synth.add_argument("--n-radii", type=int, default=10)
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument(
        "--out", type=Path, default=Path("/tmp/synthetic_benchmark.png"), help="Plot output path"
    )
    p_synth.set_defaults(func=cmd_synthetic_benchmark)

    p_robust = subparsers.add_parser(
        "robustness-check", help="Run all candidates on a folder of real files, side by side"
    )
    p_robust.add_argument("folder", type=Path)
    p_robust.add_argument("--limit", type=int, default=None)
    p_robust.set_defaults(func=cmd_robustness_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
