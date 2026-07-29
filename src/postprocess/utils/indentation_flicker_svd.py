from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO_SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_SRC))

from pipelines.lowrank_pulsatility_metrics import LowRankPulsatilityMetrics  # noqa: E402
import postprocess.utils.flicker_stats as flicker_stats  # noqa: E402

_LR = LowRankPulsatilityMetrics()
EPS = LowRankPulsatilityMetrics.eps


def resolve_file_paths(base_dir, group, quantity):
    base_dir = Path(base_dir)
    group = group.lower()
    quantity = quantity.lower()

    if group in {"ctrl", "control", "baseline", "bl1", "bl2"}:
        prefix = "ctrl"
    elif group in {"pressure", "indentation"}:
        prefix = "pressure"
    elif group in {"flicker"}:
        prefix = "flicker"
    else:
        raise ValueError(f"Unknown group: {group}")

    if quantity in {"a", "asymmetry"}:
        return (base_dir / f"{prefix}_asymmetry_A.h5", "Asymmetry/A/value", "QualityControl/ParabolicFitValid/value")

    if quantity in {"pa", "p_a", "partition"}:
        return (base_dir / f"{prefix}_partition_functions_pA.h5", "PartitionFunctions/p_A/value", "QualityControl/ParabolicFitValid/value")

    raise ValueError(f"Unknown quantity: {quantity}")


def load_recording_group(base_dir, group, quantity):
    h5path, dataset_path, qc_path = resolve_file_paths(base_dir, group, quantity)

    with h5py.File(h5path, "r") as h:
        values = np.asarray(h[dataset_path], dtype=float)
        phase = np.asarray(h["Axes/CardiacPhase/value"], dtype=float)
        valid_mask = np.asarray(h[qc_path], dtype=bool)

        names = None
        if "Files/Names/value" in h:
            names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h["Files/Names/value"][:]]

    return values, phase, valid_mask, names

# =============================
# PAPER RESULTS
# =============================

def compute_Abar_bkr(A):
    with np.errstate(invalid="ignore"):
        Abar_bkr = np.nanmean(A, axis=0)
    return Abar_bkr


def compute_pA_t_bkr(A, Abar):
    pA_t_bkr = (A - Abar[None, :, :, :]) ** 2
    return pA_t_bkr


def hierarchical_median_bkr(arr_bkr):
    with np.errstate(invalid="ignore"):
        med_over_r = np.nanmedian(arr_bkr, axis=2)
        med_over_k = np.nanmedian(med_over_r, axis=1)
        med_over_b = np.nanmedian(med_over_k, axis=0)
    return float(med_over_b)


def recording_endpoints(A, pA, phase, valid_mask=None):
    A = np.asarray(A, dtype=float).copy()
    pA = np.asarray(pA, dtype=float).copy()
    phase = np.asarray(phase, dtype=float)

    if valid_mask is not None:
        A[:, ~valid_mask] = np.nan
        pA[:, ~valid_mask] = np.nan

    Abar = compute_Abar_bkr(A)
    PFA = hierarchical_median_bkr(np.abs(Abar))

    with np.errstate(invalid="ignore"):
        a_bkr = np.sqrt(np.nanmean(pA, axis=0))
    FFA = hierarchical_median_bkr(a_bkr)

    early_mask = phase < (1.0 / 3.0)
    late_mask = ~early_mask

    with np.errstate(invalid="ignore"):
        aearly_bkr = np.sqrt(np.nanmean(pA[early_mask], axis=0))
        alate_bkr = np.sqrt(np.nanmean(pA[late_mask], axis=0))
        Ra_bkr = aearly_bkr / (alate_bkr + EPS)

    FFAR = hierarchical_median_bkr(Ra_bkr)

    finite_pA_count_bkr = np.sum(np.isfinite(pA), axis=0)
    total_power_bkr = np.nansum(pA, axis=0)

    centroid_phase = np.arange(phase.size, dtype=float) / float(phase.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_A_bkr = (np.nansum(centroid_phase[:, None, None, None] * pA, axis=0) / (total_power_bkr + EPS))

    phi_A_bkr[(finite_pA_count_bkr < 2) | ~(total_power_bkr > 0)] = np.nan
    phi_A = hierarchical_median_bkr(phi_A_bkr)

    return {
        "PFA": PFA,
        "FFA": FFA,
        "FFAR": FFAR,
        "phi_A": phi_A,
        "Abar_bkr": Abar,
        "a_bkr": a_bkr,
        "aearly_bkr": aearly_bkr,
        "alate_bkr": alate_bkr,
        "Ra_bkr": Ra_bkr,
        "phi_A_bkr": phi_A_bkr,
    }


def recording_group_endpoints(base_dir, group):
    A_all, phase, valid_A_all, names = load_recording_group(base_dir, group, "A")
    pA_all, _, valid_pA_all, _ = load_recording_group(base_dir, group, "pA")

    recording_results = []

    for i in range(A_all.shape[0]):
        valid_mask = valid_A_all[i] & valid_pA_all[i]
        result = recording_endpoints(A=A_all[i], pA=pA_all[i], phase=phase, valid_mask=valid_mask)
        if names is not None:
            result["name"] = names[i]
        recording_results.append(result)

    endpoint_names = ["PFA", "FFA", "FFAR", "phi_A"]
    per_recording = {key: np.asarray([result[key] for result in recording_results], dtype=float) for key in endpoint_names}
    group_medians = {key: LowRankPulsatilityMetrics._safe_nanmedian(values) for key, values in per_recording.items()}

    return {
        "recordings": recording_results,
        "per_recording": per_recording,
        "group_medians": group_medians,
        "phase": phase,
        "names": names,
    }


def print_group_summary(group_name, result):
    print(f"\n{group_name}")
    print("-" * len(group_name))
    for key, values in result["per_recording"].items():
        print(f"{key} per recording:")
        print(np.round(values, 4))
        print(f"{key} median: {result['group_medians'][key]:.6g}")


# =============================
# LOW-RANK SVD
# =============================


def validate_pA_matches_A(A, pA):
    Abar = compute_Abar_bkr(A)
    pA_from_A = compute_pA_t_bkr(A, Abar)
    diff = np.abs(np.asarray(pA, dtype=float) - pA_from_A)
    return {
        "max_abs_diff": float(np.nanmax(diff)),
        "median_abs_diff": float(np.nanmedian(diff)),
    }


def build_lowrank_matrix(A, pA=None, valid_mask=None, representation="A_dynamic", min_valid_samples_fraction=_LR.min_valid_samples_fraction):
    A = np.asarray(A, dtype=float).copy()
    representation = representation.lower()

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        A[:, ~valid_mask] = np.nan
    else:
        valid_mask = np.ones(A.shape[1:], dtype=bool)

    if representation == "a_dynamic":
        column_mean_bkr = compute_Abar_bkr(A)
        X_full = A - column_mean_bkr[None, :, :, :]
    elif representation == "a_raw":
        column_mean_bkr = np.full(A.shape[1:], np.nan, dtype=float)
        X_full = A.copy()
    elif representation in {"pa_raw", "p_a_raw"}:
        if pA is None:
            raise ValueError("pA is required for representation='pA_raw'")
        pA = np.asarray(pA, dtype=float).copy()
        pA[:, ~valid_mask] = np.nan

        column_mean_bkr = np.full(pA.shape[1:], np.nan, dtype=float)
        X_full = pA
    elif representation in {"pa_centered", "p_a_centered"}:
        if pA is None:
            raise ValueError("pA is required for representation='pA_centered'")
        pA = np.asarray(pA, dtype=float).copy()
        pA[:, ~valid_mask] = np.nan

        with np.errstate(invalid="ignore"):
            column_mean_bkr = np.nanmean(pA, axis=0)

        X_full = pA - column_mean_bkr[None, :, :, :]
    else:
        raise ValueError(f"Unknown low-rank representation: {representation}")
    finite_fraction_bkr = np.mean(np.isfinite(X_full), axis=0)
    valid_column_mask = (valid_mask & (finite_fraction_bkr >= float(min_valid_samples_fraction)))

    if not np.any(valid_column_mask):
        X = np.empty((A.shape[0], 0), dtype=float)
    else:
        X = X_full[:, valid_column_mask]
        if representation in {"a_dynamic", "pa_centered", "p_a_centered"}:
            X = np.where(np.isfinite(X), X, 0.0)
        else:
            with np.errstate(invalid="ignore"):
                column_mean = np.nanmean(X, axis=0)
            X = np.where(np.isfinite(X), X, column_mean[None, :])
            X = np.where(np.isfinite(X), X, 0.0)

    return {
        "X": X,
        "X_full": X_full,
        "valid_column_mask": valid_column_mask,
        "finite_fraction_bkr": finite_fraction_bkr,
        "column_mean_bkr": column_mean_bkr,
        "representation": representation,
    }


def mode_temporal_centroid(u_t, phase):
    weights = np.asarray(u_t, dtype=float) ** 2
    phase = np.asarray(phase, dtype=float)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= EPS:
        return np.nan
    return float(np.nansum(phase * weights) / (total + EPS))


def mode_early_late_ratio(u_t, phase):
    u_t = np.asarray(u_t, dtype=float)
    phase = np.asarray(phase, dtype=float)
    early_mask = phase < (1.0 / 3.0)
    late_mask = ~early_mask
    early_rms = float(np.sqrt(np.nanmean(u_t[early_mask] ** 2)))
    late_rms = float(np.sqrt(np.nanmean(u_t[late_mask] ** 2)))
    if not np.isfinite(early_rms) or not np.isfinite(late_rms):
        return np.nan
    return float(early_rms / (late_rms + EPS))


def compute_svd_panel(matrix_info, phase=None, max_modes=_LR.max_modes_panel, min_valid_columns=_LR.min_valid_columns):
    X = np.asarray(matrix_info["X"], dtype=float)
    valid_column_mask = np.asarray(matrix_info["valid_column_mask"], dtype=bool)

    shape_bkr = valid_column_mask.shape
    n_t = X.shape[0]
    n_valid_columns = X.shape[1]

    out = {
        "representation": matrix_info["representation"],
        "n_t": n_t,
        "n_valid_columns": n_valid_columns,
        "valid_column_mask": valid_column_mask,
        "finite_fraction_bkr": matrix_info["finite_fraction_bkr"],
    }

    if n_valid_columns < min_valid_columns:
        out["svd_available"] = False
        out["svd_reason"] = "too_few_valid_columns"
        return out

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    energy = s ** 2
    energy_fraction = energy / (np.sum(energy) + EPS)

    n_modes = int(min(max_modes, len(s)))

    U_panel = np.full((n_t, max_modes), np.nan, dtype=float)
    score_panel_bkr = np.full((max_modes, *shape_bkr), np.nan, dtype=float)
    mode_rms_panel_bkr = np.full((max_modes, *shape_bkr), np.nan, dtype=float)
    residual_rms_panel_bkr = np.full((max_modes, *shape_bkr), np.nan, dtype=float)
    rho_panel = np.full((max_modes,), np.nan, dtype=float)
    sign_flips = np.zeros((max_modes,), dtype=np.uint8)
    mode_phi = np.full((max_modes,), np.nan, dtype=float)
    mode_ffar = np.full((max_modes,), np.nan, dtype=float)

    scores_by_mode = []

    for m in range(n_modes):
        scores = s[m] * Vt[m, :]

        median_score = LowRankPulsatilityMetrics._safe_nanmedian(scores)
        if np.isfinite(median_score) and median_score < 0:
            U[:, m] *= -1.0
            Vt[m, :] *= -1.0
            scores *= -1.0
            sign_flips[m] = 1

        U_panel[:, m] = U[:, m]
        score_panel_bkr[m, valid_column_mask] = scores

        # Reuses lowrank_pulsatility_metrics.py's own mode-RMS definition
        mode_rms_panel_bkr[m] = _LR._mode_component_rms(u=U[:, m], scores=scores, valid_mask=valid_column_mask)

        scores_by_mode.append(scores)
        if phase is not None:
            mode_phi[m] = mode_temporal_centroid(U[:, m], phase)
            mode_ffar[m] = mode_early_late_ratio(U[:, m], phase)

    total_rms_valid = np.sqrt(np.mean(X ** 2, axis=0))

    total_rms_bkr = np.full(shape_bkr, np.nan, dtype=float)
    total_rms_bkr[valid_column_mask] = total_rms_valid

    for rank in range(1, n_modes + 1):
        U_rank = U[:, :rank]
        scores_rank = np.vstack(scores_by_mode[:rank])

        X_recon = U_rank @ scores_rank
        residual = X - X_recon

        residual_rms_valid = np.sqrt(np.mean(residual ** 2, axis=0))
        residual_rms_panel_bkr[rank - 1, valid_column_mask] = residual_rms_valid

        num = LowRankPulsatilityMetrics._safe_nanmedian(residual_rms_valid)
        den = LowRankPulsatilityMetrics._safe_nanmedian(total_rms_valid)

        if np.isfinite(num) and np.isfinite(den) and den > EPS:
            rho_panel[rank - 1] = float(num / (den + EPS))

    return {
        **out,
        "svd_available": True,
        "svd_reason": "ok",
        "U": U,
        "s": s,
        "Vt": Vt,
        "energy": energy,
        "energy_fraction": energy_fraction,
        "eta1": float(energy_fraction[0]) if len(energy_fraction) >= 1 else np.nan,
        "eta2": float(energy_fraction[1]) if len(energy_fraction) >= 2 else np.nan,
        "eta3": float(energy_fraction[2]) if len(energy_fraction) >= 3 else np.nan,
        "effective_rank": _LR._effective_rank(energy_fraction),
        "participation_ratio": _LR._participation_ratio(energy_fraction),
        "U_panel": U_panel,
        "score_panel_bkr": score_panel_bkr,
        "mode_rms_panel_bkr": mode_rms_panel_bkr,
        "residual_rms_panel_bkr": residual_rms_panel_bkr,
        "total_rms_bkr": total_rms_bkr,
        "rho_panel": rho_panel,
        "sign_flips": sign_flips,
        "mode_phi": mode_phi,
        "mode_ffar": mode_ffar,
    }

def mode_amplitude(svd_result, mode_idx=0):
    if not svd_result.get("svd_available", False):
        return float("nan")
    mode_rms_panel_bkr = svd_result["mode_rms_panel_bkr"]
    if mode_idx >= mode_rms_panel_bkr.shape[0]:
        return float("nan")
    return LowRankPulsatilityMetrics._safe_nanmedian(mode_rms_panel_bkr[mode_idx])


def recording_lowrank_analysis(A, pA, phase, valid_mask=None, representations=("A_dynamic", "pa_centered"), max_modes=_LR.max_modes_panel, min_valid_samples_fraction=_LR.min_valid_samples_fraction):
    results = {}
    for representation in representations:
        matrix_info = build_lowrank_matrix(
            A=A,
            pA=pA,
            valid_mask=valid_mask,
            representation=representation,
            min_valid_samples_fraction=min_valid_samples_fraction,
        )
        results[representation] = compute_svd_panel(
            matrix_info,
            phase=phase,
            max_modes=max_modes,
        )
    return results

def recording_group_lowrank_analysis(base_dir, group, representations=("A_dynamic", "pa_centered"), max_modes=_LR.max_modes_panel, min_valid_samples_fraction=_LR.min_valid_samples_fraction):
    A_all, phase, valid_A_all, names = load_recording_group(base_dir, group, "A")
    pA_all, _, valid_pA_all, _ = load_recording_group(base_dir, group, "pA")
    recording_results = []
    for i in range(A_all.shape[0]):
        valid_mask = valid_A_all[i] & valid_pA_all[i]
        result = recording_lowrank_analysis(
            A=A_all[i],
            pA=pA_all[i],
            phase=phase,
            valid_mask=valid_mask,
            representations=representations,
            max_modes=max_modes,
            min_valid_samples_fraction=min_valid_samples_fraction,
        )
        if names is not None:
            result["name"] = names[i]
        recording_results.append(result)
    summaries = {}
    for representation in representations:
        summaries[representation] = {}
        scalar_keys = [
            "eta1",
            "eta2",
            "eta3",
            "effective_rank",
            "participation_ratio",
        ]
        for key in scalar_keys:
            summaries[representation][key] = np.asarray([result[representation].get(key, np.nan) for result in recording_results], dtype=float)
        summaries[representation]["rho_panel"] = np.asarray([result[representation].get("rho_panel", np.full(max_modes, np.nan)) for result in recording_results], dtype=float)
        summaries[representation]["A_panel"] = np.asarray(
            [[mode_amplitude(result[representation], m) for m in range(max_modes)] for result in recording_results],
            dtype=float,
        )
        summaries[representation]["mode_phi"] = np.asarray([result[representation].get("mode_phi", np.full(max_modes, np.nan)) for result in recording_results], dtype=float)
        summaries[representation]["mode_ffar"] = np.asarray([result[representation].get("mode_ffar", np.full(max_modes, np.nan)) for result in recording_results], dtype=float)

    return {
        "recordings": recording_results,
        "summaries": summaries,
        "phase": phase,
        "names": names,
    }

def print_lowrank_summary(group_name, result):
    print(f"\n{group_name} low-rank")
    print("-" * (len(group_name) + len(" low-rank")))

    for representation, summary in result["summaries"].items():
        print(f"\n{representation}")

        for key in [
            "eta1",
            "eta2",
            "eta3",
            "effective_rank",
            "participation_ratio",
        ]:
            print(f"{key} median: {LowRankPulsatilityMetrics._safe_nanmedian(summary[key]):.6g}")

        A_panel = summary["A_panel"]
        rho = summary["rho_panel"]
        mode_phi = summary["mode_phi"]
        mode_ffar = summary["mode_ffar"]

        for mode_idx in range(rho.shape[1]):
            print(
                f"mode{mode_idx + 1}: "
                f"A median={LowRankPulsatilityMetrics._safe_nanmedian(A_panel[:, mode_idx]):.6g}, "
                f"rho median={LowRankPulsatilityMetrics._safe_nanmedian(rho[:, mode_idx]):.6g}, "
                f"phi median={LowRankPulsatilityMetrics._safe_nanmedian(mode_phi[:, mode_idx]):.6g}, "
                f"early/late median={LowRankPulsatilityMetrics._safe_nanmedian(mode_ffar[:, mode_idx]):.6g}"
            )

# =============================
# STATISTICS
# =============================

LOWRANK_REPRESENTATIONS = {"A_dynamic": "", "pa_centered": "_pA"}

PAPER_METRIC_KEYS = ["PFA", "FFA", "FFAR", "phi_A"]
LOWRANK_METRIC_KEYS = [
    f"{base}{suffix}" for suffix in LOWRANK_REPRESENTATIONS.values() for base in ("A1", "rho1", "rho2", "rho3")
]
METRIC_KEYS = PAPER_METRIC_KEYS + LOWRANK_METRIC_KEYS
CONTROL_METRIC = "FFA"
RESIDUALIZE_AGAINST_CONTROL = LOWRANK_METRIC_KEYS


def collect_recording_group_table(base_dir, recording_group, representations=LOWRANK_REPRESENTATIONS):
    endpoints = recording_group_endpoints(base_dir, recording_group)
    lowrank = recording_group_lowrank_analysis(base_dir, recording_group, representations=tuple(representations))

    rows = []
    for i, rec in enumerate(lowrank["recordings"]):
        row = {
            "PFA": endpoints["per_recording"]["PFA"][i],
            "FFA": endpoints["per_recording"]["FFA"][i],
            "FFAR": endpoints["per_recording"]["FFAR"][i],
            "phi_A": endpoints["per_recording"]["phi_A"][i],
        }
        for representation, suffix in representations.items():
            svd_result = rec[representation]
            rho_panel = svd_result.get("rho_panel")
            row[f"A1{suffix}"] = mode_amplitude(svd_result, 0)
            row[f"rho1{suffix}"] = float(rho_panel[0]) if rho_panel is not None else float("nan")
            row[f"rho2{suffix}"] = float(rho_panel[1]) if rho_panel is not None else float("nan")
            row[f"rho3{suffix}"] = float(rho_panel[2]) if rho_panel is not None else float("nan")
        rows.append(row)
    return rows

def run_full_analysis(base_dir, representations=LOWRANK_REPRESENTATIONS):
    ctrl_rows = collect_recording_group_table(base_dir, "ctrl", representations)
    pressure_rows = collect_recording_group_table(base_dir, "pressure", representations)

    ctrl_by_metric = {key: np.array([row[key] for row in ctrl_rows], dtype=float) for key in METRIC_KEYS}
    pressure_by_metric = {key: np.array([row[key] for row in pressure_rows], dtype=float) for key in METRIC_KEYS}

    report = {}
    for key in METRIC_KEYS:
        p, auc = flicker_stats.mann_whitney_and_auc(ctrl_by_metric[key], pressure_by_metric[key])
        delta = flicker_stats.cliffs_delta(pressure_by_metric[key], ctrl_by_metric[key])
        report[key] = {
            "ctrl_median": LowRankPulsatilityMetrics._safe_nanmedian(ctrl_by_metric[key]),
            "pressure_median": LowRankPulsatilityMetrics._safe_nanmedian(pressure_by_metric[key]),
            "raw_p": p,
            "raw_auc": auc,
            "cliffs_delta": delta,
        }

    for key, p_holm in zip(
        PAPER_METRIC_KEYS,
        flicker_stats.holm_adjust([report[key]["raw_p"] for key in PAPER_METRIC_KEYS]),
    ):
        report[key]["p_holm"] = p_holm

    for key, p_holm in zip(
        LOWRANK_METRIC_KEYS,
        flicker_stats.holm_adjust([report[key]["raw_p"] for key in LOWRANK_METRIC_KEYS]),
    ):
        report[key]["p_holm"] = p_holm

    control_pooled = np.concatenate([ctrl_by_metric[CONTROL_METRIC], pressure_by_metric[CONTROL_METRIC]])
    n_ctrl = len(ctrl_by_metric[CONTROL_METRIC])
    for key in RESIDUALIZE_AGAINST_CONTROL:
        values_pooled = np.concatenate([ctrl_by_metric[key], pressure_by_metric[key]])
        residuals, r_squared = flicker_stats.residualize_against_control(values_pooled, control_pooled)
        resid_p, resid_auc = flicker_stats.mann_whitney_and_auc(residuals[:n_ctrl], residuals[n_ctrl:])
        report[key]["residualized"] = {
            "control_name": CONTROL_METRIC,
            "r_squared": r_squared,
            "p": resid_p,
            "auc": resid_auc,
            "survives": bool(np.isfinite(resid_p) and resid_p < 0.05),
        }

    return {"ctrl_rows": ctrl_rows, "pressure_rows": pressure_rows, "report": report}

def print_full_report(result):
    report = result["report"]
    print(f"\nctrl vs pressure ({len(result['ctrl_rows'])} vs {len(result['pressure_rows'])} recordings)")
    print("-" * 40)
    for key in METRIC_KEYS:
        r = report[key]
        line = (
            f"{key:<8} ctrl={r['ctrl_median']:.4g} pressure={r['pressure_median']:.4g} "
            f"raw_p={r['raw_p']:.4g} p_holm={r['p_holm']:.4g} "
            f"cliffs_delta={r['cliffs_delta']:.3f} auc={r['raw_auc']:.3f}"
        )
        if "residualized" in r:
            resid = r["residualized"]
            line += (
                f"  | residualized(vs {resid['control_name']}) "
                f"R^2={resid['r_squared']:.3f} p={resid['p']:.4g} survives={resid['survives']}"
            )
        print(line)

# =============================
# RUN
# =============================

def main():
    base = Path("/Users/admin/Desktop/langevin-internship/wobbling-svd")

    for recording_group in ("ctrl", "pressure"):
        result = recording_group_endpoints(base, recording_group)
        print_group_summary(recording_group, result)

    print("\n=== p_A consistency check (Eq. 6) ===")
    for recording_group in ("ctrl", "pressure"):
        A_all, _, _, _ = load_recording_group(base, recording_group, "A")
        pA_all, _, _, _ = load_recording_group(base, recording_group, "pA")
        max_diffs = [validate_pA_matches_A(A_all[i], pA_all[i])["max_abs_diff"] for i in range(A_all.shape[0])]
        print(f"{recording_group}: worst max_abs_diff across recordings = {max(max_diffs):.2e}")

    ctrl_lowrank = recording_group_lowrank_analysis(base, "ctrl")
    pressure_lowrank = recording_group_lowrank_analysis(base, "pressure")
    print_lowrank_summary("ctrl", ctrl_lowrank)
    print_lowrank_summary("pressure", pressure_lowrank)

    full = run_full_analysis(base)
    print_full_report(full)


if __name__ == "__main__":
    raise SystemExit(main())
