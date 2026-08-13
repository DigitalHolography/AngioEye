"""Cohort low-rank Figs 5--7 from packed result H5s.

Statistics tables and ``lowrank_cohort.h5`` are written by
``scripts.lowrank_cohort_stats``, not this postprocess.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from input_output import cohort_results_dir
from input_output.archive_io import extracted_zip_tree

from pipelines.lowrank_waveform_decomposition import (
    SPECTRUM_N_MODES,
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


# =====================================================================
# Figure annotations (p / Cliff's δ on Figs 5--7)
# =====================================================================

def _finite(x) -> np.ndarray:
    """1-D float array with NaN/inf dropped."""
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: Pr(X>Y) - Pr(X<Y). Positive means ``x`` tends larger."""
    x, y = _finite(x), _finite(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float(np.mean(diff > 0) - np.mean(diff < 0))


def _pooled_test(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Mann-Whitney p and Cliff's δ for Flicker vs pooled Baseline (B1∪B2)."""
    baseline = _finite(df.loc[df["epoch"].isin(["B1", "B2"]), metric])
    flicker = _finite(df.loc[df["epoch"] == "Flicker", metric])
    if baseline.size < 2 or flicker.size < 2:
        return float("nan"), float("nan")
    p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
    return p, _cliffs_delta(flicker, baseline)


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
# Figs 5--7
# =====================================================================

class LowRankWaveformCohortFigures:
    """Article Figs. 5--7, written once per cohort."""

    PANEL_SIZE = 2.5
    FLICKER_SHADE = "#add8e6"
    SPECTRUM_BASELINE = ("black", "-", "o")
    SPECTRUM_FLICKER = ("#555555", "--", "s")

    FIG5_PANELS = (
        ("beat_period", r"Beat period" "\n" r"$T$"),
        ("mu", r"Baseline level" "\n" r"$\mu$"),
        ("TPR", r"Total Pulsatile RMS" "\n" r"$R_0$"),
        ("mpr", r"Mean-to-pulsatile ratio" "\n" r"MPR"),
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
        ("effective_rank", r"Effective rank" "\n" r"$R_{\mathrm{eff}}$"),
        ("participation_ratio", r"Participation ratio" "\n" r"PR"),
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
    ) -> list[Path]:
        """Write Figs. 5, 6, 7 and the artery variance-fraction spectrum."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grids = (
            ("fig5_nonsvd_endpoints.png", cls.FIG5_PANELS),
            ("fig6_lowrank_endpoints.png", cls.FIG6_PANELS),
            ("fig7_residual_spectrum_endpoints.png", cls.FIG7_PANELS),
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
        written.append(
            cls._save_spectrum(
                out_dir / prefixed_filename("fig4_variance_fraction.png", patient_id),
                beats_by_vessel,
                cumulative=False,
            )
        )
        written.append(
            cls._save_spectrum(
                out_dir
                / prefixed_filename("fig4_variance_fraction_cumulative.png", patient_id),
                beats_by_vessel,
                cumulative=True,
            )
        )
        return written

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
        beats = (beats_by_vessel or {}).get("artery", pd.DataFrame())
        mode_cols = [
            f"mode{i}"
            for i in range(1, SPECTRUM_N_MODES + 1)
            if f"mode{i}" in beats.columns
        ]
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


# =====================================================================
# Collection from packed result H5s
# =====================================================================

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
        (beatwise, "TPR_b"),
        (beatwise, "A1_b"),
        (per_beat_svd, "TPR_b_pb"),
        (per_beat_svd, "A1_b_pb"),
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
            "TPR": _beat_get(beatwise, "TPR_b", b),
            "mpr": _beat_get(beatwise, "mpr_b", b),
            "A1": _beat_get(beatwise, "A1_b", b),
            "R1": _beat_get(beatwise, "R1_b", b),
            "rho1": _beat_get(beatwise, "rho1_b", b),
            "A2": _beat_get(beatwise, "A2_b", b),
            "R2": _beat_get(beatwise, "R2_b", b),
            "rho2": _beat_get(beatwise, "rho2_b", b),
            "A1_pb": _beat_get(per_beat_svd, "A1_b_pb", b),
            "R1_pb": _beat_get(per_beat_svd, "R1_b_pb", b),
            "A2_pb": _beat_get(per_beat_svd, "A2_b_pb", b),
            "R2_pb": _beat_get(per_beat_svd, "R2_b_pb", b),
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
        "TPR": scalar("TPR"),
        "TPR_sd": scalar("sigma_TPR_beat"),
        "mpr": scalar("mpr"),
        "mpr_sd": scalar("sigma_mpr_beat"),
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
            beat_rows[vessel].extend(
                _beat_rows(vessel, h5_path, seq, label, vessel_data)
            )

    points_by_vessel: dict[str, pd.DataFrame] = {}
    beats_by_vessel: dict[str, pd.DataFrame] = {}
    kept_acqs: dict[str, dict[str, list[dict]]] = {}
    for vessel in vessels:
        rows = points_rows[vessel]
        if not rows:
            continue
        rows.sort(key=lambda r: (_row_sort_key(r["epoch"]), r["acquisition"]))
        beats = beat_rows[vessel]
        beats.sort(
            key=lambda r: (
                _row_sort_key(r["epoch"]),
                r["acquisition"],
                r["beat_index"],
            )
        )
        points_by_vessel[vessel] = pd.DataFrame(rows)
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


def _run_on_paths(
    h5_paths: list[Path],
    cohort_root: Path,
    output_dir: Path,
    *,
    veins: bool,
) -> tuple[str, list[Path]]:
    """Collect packed H5s and write Figs 5--7 when 2+ group folders exist."""
    if not h5_paths:
        raise ValueError(
            "No packed AngioEye result H5 files with low-rank metrics were "
            "found for cohort postprocess. Run lowrank_waveform_decomposition "
            "first so result H5s contain the metrics."
        )
    collection = collect_payload(h5_paths, cohort_root, veins=veins)
    generated_paths: list[Path] = []
    has_cohort_split = len(collection.group_order) >= 2
    if has_cohort_split and collection.points_by_vessel:
        generated_paths.extend(
            LowRankWaveformCohortFigures.plot_all(
                collection.points_by_vessel,
                Path(output_dir),
                collection.group_order,
                patient_id=collection.patient_id,
                beats_by_vessel=collection.beats_by_vessel,
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
    summary = (
        f"Low-rank waveform run: {n_rows} "
        f"acquisition-vessel row(s) ({', '.join(vessel_summaries)}); "
        f"{split_note}."
    )
    return summary, generated_paths


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = True,
    result_h5_paths: Iterable[Path] | None = None,
) -> tuple[str, list[Path]]:
    """Build Figs 5--7 from packed AngioEye result H5s.

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
        "build cohort Figs. 5--7 when 2+ group folders are present. Writes "
        "under ``cohort-results/``. Statistics / ``lowrank_cohort.h5`` are "
        "written by ``scripts.lowrank_cohort_stats``."
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
        """Registered entry: resolve result H5s from context and write Figs 5--7."""
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
