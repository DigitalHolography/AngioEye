"""AngioEye low-rank products from EyeFlow-packed metrics.

EyeFlow owns SVD / endpoints and writes them under
``Processing/Metrics/lowrank_waveform_decomposition/`` in ``*_EF.h5``.
This module does **not** import EyeFlow or recompute decomposition: it
ingests that group into the AngioEye result H5 and writes Figs 2--4.
Joint SVD and per-beat SVD are always ingested together. Figures are
written per vessel and signal source into ``<png dir>/<vessel>/<signal>/``
for every source EyeFlow packed (artery/vein x bandlimited/raw).

Fig. 2 has no SVD. Figs. 3--4 carry a ``<basis>_<observation level>``
suffix -- ``joint_acq``, ``per_beat_acq``, ``per_beat_beats`` -- where the
basis is how the temporal modes were fit and the observation level is what
one summarized value represents. ``joint_beat`` is not canonical: a joint
basis is fit across the whole acquisition, so it has no beat-level row.
Cohort Figs. 4--7 and ``lowrank_cohort.h5`` live in
``postprocess.lowrank_waveform_cohort``.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FuncFormatter,
    MaxNLocator,
    MultipleLocator,
    NullFormatter,
)

from input_output.hdf5_io import find_first_existing_path
from input_output.hdf5_schema import find_pipeline_group
from input_output.inputs import find_hdf5_inputs
from input_output.output_paths import (
    H5_OUTPUT_DIRNAME,
    PNG_OUTPUT_DIRNAME,
    PNGS_OUTPUT_DIRNAME,
    dataset_stem_from_path,
)

from .core.base import (
    ProcessPipeline,
    ProcessResult,
    registerPipeline,
)

T_INPUT = "Processing/VelocityPerBeat/BeatPeriodSeconds/value"

FIGURE_VESSELS = ("artery",)
ALL_VESSELS = ("artery", "vein")
PIPELINE_NAME = "lowrank_waveform_decomposition"

COHORT_SIGNAL = "raw"
SIGNALS = ("bandlimited", "raw")
SVD_METHODS = ("joint", "per_beat")
DEFAULT_SVD_METHOD = "joint"

# EyeFlow velocity node name for each low-rank signal source.
VELOCITY_SIGNAL_NODES = {"raw": "Raw", "bandlimited": "BandLimited"}

# Figure suffix = <SVD basis>_<observation level>. ``joint_beat`` is not
# canonical: a joint basis is fit per acquisition, so it has no beat rows.
FIGURE_VARIANTS = ("joint_acq", "per_beat_acq", "per_beat_beats")

EYEFLOW_LOWRANK_GROUP_CANDIDATES = (
    "Processing/Metrics/lowrank_waveform_decomposition",
    "Metrics/lowrank_waveform_decomposition",
)
SOURCE_NAMES = (
    "artery/raw",
    "artery/bandlimited",
    "vein/raw",
    "vein/bandlimited",
)

SPECTRUM_N_MODES = 12
ENDPOINT_METRICS = (
    "A1",
    "A2",
    "R0",
    "R1",
    "R2",
    "rho1",
    "rho2",
    "MPR",
    "Reff",
    "PR",
)
FIG3_PANEL_KEYS = ("v", "mu", "x", "a1u1", "a2u2")
# EyeFlow ``misc/`` group names for Fig. 3 panels. Internal payload keys
# stay ``v`` / ``mu`` / ``x`` / ``a1u1`` / ``a2u2``.
FIG3_PANEL_EYEFLOW_GROUPS = {
    "v": "segment_velocity",
    "mu": "temporal_baseline",
    "x": "centered_velocity",
    "a1u1": "svd_mode1",
    "a2u2": "svd_mode2",
}
# EyeFlow ``misc/`` datasets for the gray band. Internal keys stay
# ``mean`` / ``lo`` / ``hi``.
FIG3_BAND_EYEFLOW_DATASETS = {
    "mean": "cross_column_mean",
    "lo": "cross_column_lo",
    "hi": "cross_column_hi",
}
EYEFLOW_MISC_ACQUISITION_VELOCITY = "acquisition_level_velocity"
EYEFLOW_MISC_WAVEFORM_JOINT = "waveform_components_joint"
EYEFLOW_MISC_WAVEFORM_PER_BEAT = "waveform_components_per_beat"
EYEFLOW_MISC_SPECTRUM_JOINT = "svd_spectrum_joint"
EYEFLOW_MISC_SPECTRUM_PER_BEAT = "svd_spectrum_joint_per_beat"
FIG3_VARIABILITY_METHODS = (
    "beat_location",
    "beat",
    "pooled_within_location",
)


def enabled_vessels(veins_flag: bool) -> tuple[str, ...]:
    """Return ``("artery",)`` or ``("artery", "vein")`` from the Veins flag."""
    return ("artery", "vein") if veins_flag else ("artery",)


def normalize_svd_method(value: str | None) -> str:
    """Return ``joint`` or ``per_beat`` from a pipeline/postprocess flag."""
    text = str(DEFAULT_SVD_METHOD if value is None else value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "joint": "joint",
        "joint_svd": "joint",
        "per_beat": "per_beat",
        "per_beat_svd": "per_beat",
        "perbeat": "per_beat",
        "perbeat_svd": "per_beat",
    }
    if text not in aliases:
        raise ValueError(
            f"Unknown svd_method {value!r}. Expected one of {SVD_METHODS}."
        )
    return aliases[text]


def aggregate_beatwise(values_per_beat: np.ndarray, stat: str) -> float:
    """Collapse a per-beat array to one acquisition scalar (mean or median)."""
    x = np.asarray(values_per_beat, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x)) if stat == "mean" else float(np.median(x))


def aggregate_rho(R_b: np.ndarray, R0_b: np.ndarray, stat: str) -> float:
    """Residual ratio R/R0 from separately aggregated residual and pulsatile RMS."""
    r = aggregate_beatwise(R_b, stat)
    t = aggregate_beatwise(R0_b, stat)
    if not np.isfinite(r) or not np.isfinite(t) or abs(t) <= 1e-12:
        return float("nan")
    return float(r / (t + 1e-12))


# Serializes every Matplotlib draw in this package. Batch runs render
# figures from a ThreadPoolExecutor, and Matplotlib is not thread-safe.
_FIGURE_LOCK = threading.RLock()


def safe_figure(name: str, plotter, /, **kwargs) -> Path | None:
    """Draw one figure under the global lock, turning failure into a warning.

    Two hazards are handled here. Matplotlib is not thread-safe: pyplot
    keeps a global figure registry and mathtext is a module-global
    pyparsing grammar, so drawing from the batch thread pool corrupts the
    parser and raises ParseException on valid labels such as ``$m$`` or
    Matplotlib's own ``\\mathdefault`` log-tick macro, at random. Every
    figure is therefore serialized through ``_FIGURE_LOCK``.

    Separately, one unplottable source (an absent mode, an empty velocity
    trace) must not cost the later figures of that acquisition, which
    would look like whole vessels or epochs going missing. Any open figure
    is closed so a partial draw cannot leak.
    """
    with _FIGURE_LOCK:
        try:
            return plotter(**kwargs)
        except Exception as exc:  # one bad source must not stop the batch
            plt.close("all")
            # ``exc`` is unbound once the except block exits, so keep the
            # parts of it the warning needs.
            kind = type(exc).__name__
            detail = " ".join(str(exc).split())[:200] or kind
    warnings.warn(
        f"low-rank figure {name!r} could not be drawn: {kind}: {detail}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def report_missing_figures(
    source: str,
    expected: list[str],
    written: list[Path],
    *,
    source_label: str = "",
) -> list[str]:
    """Warn once naming the figures a source did not produce.

    A figure is skipped whenever its packed payload is absent, which is a
    normal outcome for some sources but indistinguishable from a bug when
    the run stays silent. Naming the gaps makes an incomplete output set
    self-explaining instead of something to reverse-engineer from folders.
    """
    produced = {path.stem for path in written}
    missing = [
        name
        for name in expected
        if not any(stem.endswith(name) for stem in produced)
    ]
    if missing:
        where = f"{source_label} {source}".strip()
        warnings.warn(
            f"low-rank figures not produced for {where} "
            f"({len(missing)} of {len(expected)}): {', '.join(missing)}. "
            "The packed EyeFlow payload for these variants is absent.",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


def _finite_min(values: np.ndarray) -> float:
    """Smallest finite value, or NaN when there is none."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.min()) if x.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    """Largest finite value, or NaN when there is none."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else float("nan")


def finite_std(values: np.ndarray, *, ddof: int = 1) -> float:
    """Sample SD of finite values; 0 when fewer than two samples."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=ddof))


def mean_pm_std(
    values: np.ndarray,
    *,
    axis: int = 0,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nan-aware mean ± 1 sample SD along ``axis``.

    Used for every beat-to-beat (or along-sample) gray band: sample SD
    (``ddof=1``), SD = 0 when fewer than two finite samples. Returns
    ``(mean, mean - sd, mean + sd)``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(arr, axis=axis)
        n = np.sum(np.isfinite(arr), axis=axis)
        # Avoid numpy's "dof <= 0" warning when a slice has <2 finite values.
        sd = np.zeros_like(mean, dtype=float)
        ok = n > 1
        if np.any(ok):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd_all = np.nanstd(arr, axis=axis, ddof=ddof)
            sd = np.where(ok & np.isfinite(sd_all), sd_all, 0.0)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    return mean, mean - sd, mean + sd


def coerce_beat_spectra(
    values: np.ndarray | None,
    *,
    n_modes: int = SPECTRUM_N_MODES,
) -> np.ndarray:
    """Return ``(n_beats, n_modes)`` per-beat singular values, or empty."""
    arr = np.asarray([] if values is None else values, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, n_modes), dtype=float)
    if arr.shape[1] < n_modes:
        out = np.full((arr.shape[0], n_modes), np.nan, dtype=float)
        out[:, : arr.shape[1]] = arr
        return out
    return np.asarray(arr[:, :n_modes], dtype=float)


def is_usable_beat_spectra(values: np.ndarray | None) -> bool:
    """True if ``values`` is a 2-D spectrum with at least one finite entry."""
    arr = np.asarray([] if values is None else values, dtype=float)
    return arr.ndim == 2 and arr.shape[1] >= 2 and bool(np.any(np.isfinite(arr)))


def normalize_figure3_variability_method(value: str | None) -> str:
    """Return the Figure 3 gray-band variability method name."""
    text = str("beat_location" if value is None else value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "beat_location": "beat_location",
        "beat_location_sd": "beat_location",
        "beat_location_waveforms": "beat_location",
        "beatlocation": "beat_location",
        "all_valid": "beat_location",
        "all_valid_columns": "beat_location",
        "beat": "beat",
        "beat_sd": "beat",
        "beat_to_beat": "beat",
        "beat_to_beat_sd": "beat",
        "pooled_within_location": "pooled_within_location",
        "within_location": "pooled_within_location",
        "within_location_sd": "pooled_within_location",
        "pooled": "pooled_within_location",
        "pooled_within_location_beat_sd": "pooled_within_location",
    }
    if text not in aliases:
        raise ValueError(
            f"Unknown Figure 3 variability method {value!r}. Expected one of "
            f"{FIG3_VARIABILITY_METHODS}."
        )
    return aliases[text]


def _prepare_figure3_panel_block(
    block: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(n_t, n_beats, n_locations)`` values and a ``(B, L)`` mask."""
    arr = np.asarray(block, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if arr.ndim == mask.ndim:
        arr = arr[None, ...]
    n_t = int(arr.shape[0])
    if mask.shape != arr.shape[1:]:
        return (
            np.full((n_t, 0, 0), np.nan, dtype=float),
            np.zeros((0, 0), dtype=bool),
            n_t,
        )
    n_beats = int(mask.shape[0])
    return arr.reshape(n_t, n_beats, -1), mask.reshape(n_beats, -1), n_t


def figure3_beat_location_mean_pm_std(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid beat-location mean ± 1 SD for one Figure 3 panel.

    At each cardiac phase, collapse all valid columns ``j = (b, k, r)`` from
    the relevant panel. This keeps the variability across the valid
    beat-location waveform population visible in the gray band. ``block`` is
    ``(n_t, n_beats, ...)`` or ``(n_beats, ...)``; ``valid`` is
    ``(n_beats, ...)``. Panels without a time axis, such as ``mu``, return a
    single value that the caller broadcasts over the plotted time axis.
    """
    arr, mask, n_t = _prepare_figure3_panel_block(block, valid)
    if arr.shape[1] == 0:
        empty = np.full(n_t, np.nan, dtype=float)
        return empty, empty, empty
    flat = np.where(mask[None, ...], arr, np.nan).reshape(n_t, -1)
    return mean_pm_std(flat, axis=1, ddof=ddof)


def figure3_beat_mean_pm_std(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal beat-to-beat mean ± 1 SD using a common spatial location set.

    For each beat, first build one curve by taking the spatial median over
    locations valid in every beat. Then compute mean ± sample SD across those
    beat curves. This estimates variability of the typical waveform from beat
    to beat and excludes persistent location differences.
    """
    arr, mask, n_t = _prepare_figure3_panel_block(block, valid)
    if arr.shape[1] == 0:
        empty = np.full(n_t, np.nan, dtype=float)
        return empty, empty, empty
    common_locations = np.all(mask, axis=0)
    if not np.any(common_locations):
        empty = np.full(n_t, np.nan, dtype=float)
        return empty, empty, empty
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            beat_curves = np.nanmedian(arr[:, :, common_locations], axis=2)
    return mean_pm_std(beat_curves, axis=1, ddof=ddof)


def figure3_pooled_within_location_mean_pm_std(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise mean ± pooled within-location beat SD.

    The black curve is the pointwise mean across all valid beat-location
    columns. The gray-band width pools squared deviations around each
    location's own beat mean, so persistent vessel-location offsets are not
    counted as beat variability.
    """
    arr, mask, n_t = _prepare_figure3_panel_block(block, valid)
    if arr.shape[1] == 0:
        empty = np.full(n_t, np.nan, dtype=float)
        return empty, empty, empty
    masked = np.where(mask[None, ...], arr, np.nan)
    flat = masked.reshape(n_t, -1)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(flat, axis=1)
            loc_mean = np.nanmean(masked, axis=1)
    n_loc = np.sum(np.isfinite(masked), axis=1)
    deviations = masked - loc_mean[:, None, :]
    numerator = np.nansum(deviations * deviations, axis=(1, 2))
    denominator = np.sum(np.maximum(n_loc - ddof, 0), axis=1)
    sd = np.zeros(n_t, dtype=float)
    ok = denominator > 0
    sd[ok] = np.sqrt(numerator[ok] / denominator[ok])
    mean = np.asarray(mean, dtype=float)
    return mean, mean - sd, mean + sd


def beat_mean_pm_std_spatial_median(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for the Figure 3 beat-location SD band."""
    return figure3_beat_location_mean_pm_std(block, valid, ddof=ddof)


def figure3_panel_mean_pm_std(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    method: str | None = "beat_location",
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to one of the explicit Figure 3 gray-band definitions."""
    normalized = normalize_figure3_variability_method(method)
    if normalized == "beat_location":
        return figure3_beat_location_mean_pm_std(block, valid, ddof=ddof)
    if normalized == "beat":
        return figure3_beat_mean_pm_std(block, valid, ddof=ddof)
    return figure3_pooled_within_location_mean_pm_std(block, valid, ddof=ddof)


# =====================================================================
# Packed metrics discovery / load
# =====================================================================


def find_eyeflow_lowrank_group(h5file: h5py.File) -> h5py.Group | None:
    """Return EyeFlow's packed low-rank metrics group, if present."""
    for path in EYEFLOW_LOWRANK_GROUP_CANDIDATES:
        if path in h5file and isinstance(h5file[path], h5py.Group):
            return h5file[path]
    return None


def find_lowrank_metrics_group(h5file: h5py.File) -> h5py.Group | None:
    """Locate packed low-rank metrics in an open HDF5 file.

    Prefers the AngioEye ingest copy under ``/AngioEye/Processing/...``, then
    falls back to EyeFlow's ``Processing/Metrics/lowrank_waveform_decomposition``.
    Same loaders then work on ``*_EF.h5`` (pipeline run) and ``*_AE.h5`` (cohort).
    """
    angio = find_pipeline_group(h5file, PIPELINE_NAME)
    if angio is not None:
        return angio
    return find_eyeflow_lowrank_group(h5file)


def result_h5_has_lowrank(h5_path: Path | str) -> bool:
    """True when ``h5_path`` contains packed low-rank metrics (AE or EyeFlow)."""
    try:
        with h5py.File(h5_path, "r") as h5file:
            return find_lowrank_metrics_group(h5file) is not None
    except OSError:
        return False


def find_lowrank_result_h5s(root: Path | str) -> list[Path]:
    """Discover AngioEye result H5s that already pack low-rank metrics.

    Prefers ``*_AE.h5`` (canonical full-chain product). Also accepts legacy
    ``*_pipelines_result.h5`` files that contain the low-rank group.
    Skips raw EyeFlow ``*_EF.h5`` inputs (ingest those via the pipeline).
    """
    root = Path(root)
    candidates = find_hdf5_inputs(root)
    results: list[Path] = []
    for path in candidates:
        stem = path.stem
        # Skip EyeFlow-only inputs when scanning a mixed tree.
        if stem.endswith("_EF") and not stem.endswith("_AE"):
            continue
        # Require the AngioEye-packed group (not only EyeFlow Metrics/).
        try:
            with h5py.File(path, "r") as h5file:
                if find_pipeline_group(h5file, PIPELINE_NAME) is None:
                    continue
        except OSError:
            continue
        results.append(path)

    def _rank(path: Path) -> tuple[int, str]:
        """Sort key: ``*_AE.h5`` first, then ZIP result H5s, then others."""
        stem = path.stem
        if stem.endswith("_AE"):
            return (0, str(path).lower())
        if "pipelines_result" in stem:
            return (1, str(path).lower())
        return (2, str(path).lower())

    results.sort(key=_rank)
    return results


def companion_png_dir_for_result(output_h5_path: Path | str) -> Path:
    """PNG companion folder for a shared AngioEye result H5.

    Canonical AE product: ``{stem}_AE/png/`` beside ``{stem}_AE/h5/``.
    Epoch layout: ``{epoch}/pngs/`` beside ``{epoch}/{stem}_AE.h5``.
    """
    output_h5_path = Path(output_h5_path)
    parent = output_h5_path.parent
    if parent.name.lower() == H5_OUTPUT_DIRNAME and parent.parent.name.endswith("_AE"):
        return parent.parent / PNG_OUTPUT_DIRNAME
    return parent / PNGS_OUTPUT_DIRNAME


def acquisition_fig_stem(
    output_h5_path: Path | str, source_h5_path: Path | str | None = None
) -> str:
    """Stem used for Figs 2--4 filenames (acquisition name, not ``_pipelines_result``)."""
    output_h5_path = Path(output_h5_path)
    parent = output_h5_path.parent
    if parent.name.lower() == H5_OUTPUT_DIRNAME and parent.parent.name.endswith("_AE"):
        try:
            return dataset_stem_from_path(output_h5_path)
        except ValueError:
            return parent.parent.name[: -len("_AE")] or parent.parent.name

    stem = output_h5_path.stem
    if stem.endswith("_pipelines_result"):
        stem = stem[: -len("_pipelines_result")]
    for suffix in ("_AE", "_EF", "_HD", "_DV"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] or stem
    if source_h5_path is not None:
        src = Path(source_h5_path).stem
        for suffix in ("_AE", "_EF", "_HD", "_DV"):
            if src.endswith(suffix):
                return src[: -len(suffix)] or src
        return src
    return stem


def _group_dataset_map(group: h5py.Group) -> dict[str, np.ndarray]:
    """Copy immediate datasets in ``group`` to a name → array dict (no recursion)."""
    out: dict[str, np.ndarray] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            out[key] = np.asarray(obj)
    return out


def _dataset_vector(group: h5py.Group | None, name: str) -> np.ndarray | None:
    """Read one numeric dataset as a finite-length vector."""
    if group is None or name not in group:
        return None
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        return None
    arr = np.asarray(obj, dtype=float).reshape(-1)
    return arr if arr.size else None


def _group_has_datasets(group: h5py.Group | None) -> bool:
    """True when ``group`` directly contains at least one dataset."""
    return isinstance(group, h5py.Group) and any(
        isinstance(obj, h5py.Dataset) for obj in group.values()
    )


def _endpoint_group(source: h5py.Group, method: str) -> h5py.Group | None:
    """Official EyeFlow endpoint group: ``endpoints/joint`` or ``endpoints/per_beat``."""
    endpoints = source.get("endpoints")
    if not isinstance(endpoints, h5py.Group):
        return None
    child = endpoints.get(method)
    return child if isinstance(child, h5py.Group) else None


def _read_endpoint_values(group: h5py.Group | None) -> dict[str, np.ndarray]:
    """Read the official EyeFlow endpoint metric datasets from one endpoint group."""
    if not isinstance(group, h5py.Group):
        return {}
    out: dict[str, np.ndarray] = {}
    for key in ENDPOINT_METRICS:
        arr = _dataset_vector(group, key)
        if arr is not None:
            out[key] = arr
    return out


def _misc_group(source: h5py.Group, name: str) -> h5py.Group | None:
    """Return one official ``misc/<name>`` group."""
    misc = source.get("misc")
    if not isinstance(misc, h5py.Group):
        return None
    group = misc.get(name)
    return group if isinstance(group, h5py.Group) else None


# Spectrum storage used before EyeFlow moved these products under ``misc``.
# Only the spectrum keeps a legacy reader: the cohort Fig. 4 needs it to
# populate its mode columns, and without it an older archive silently
# reprocesses to all-NaN spectra.
_LEGACY_SPECTRUM_GROUPS = {
    "joint": ("fig4_energy_spectrum", {"mean": "lambda"}),
    "per_beat": (
        "fig4_energy_spectrum_pb",
        {"mean": "lambda_mean", "lo": "lambda_lo", "hi": "lambda_hi"},
    ),
}


def _legacy_figure_group(source: h5py.Group, name: str) -> h5py.Group | None:
    """Return one pre-``misc`` ``figures/<name>`` group."""
    figures = source.get("figures")
    if not isinstance(figures, h5py.Group):
        return None
    group = figures.get(name)
    return group if isinstance(group, h5py.Group) else None


def _fig2_payload_from_source(source: h5py.Group) -> dict[str, np.ndarray] | None:
    """Official Fig. 2 payload from ``misc/acquisition_level_velocity``."""
    group = _misc_group(source, EYEFLOW_MISC_ACQUISITION_VELOCITY)
    t = _dataset_vector(group, "cardiac_phase")
    mean = _dataset_vector(group, "velocity_cross_column_mean")
    std = _dataset_vector(group, "velocity_cross_column_std")
    if t is None or mean is None or std is None:
        return None
    n = min(t.size, mean.size, std.size)
    if n == 0:
        return None
    return {"t": t[:n], "mean": mean[:n], "std": std[:n]}


def _fig3_payload_from_source(
    source: h5py.Group, method: str, *, beats_band: bool = False
) -> dict | None:
    """Official Fig. 3 payload from joint or per-beat ``misc`` waveform groups.

    ``beats_band`` used to select a dedicated ``*_beats`` EyeFlow group.
    That group is no longer packed, so this returns None and callers may
    fall back to reconstructing the band from packed modes.
    """
    if beats_band:
        return None
    name = (
        EYEFLOW_MISC_WAVEFORM_PER_BEAT
        if normalize_svd_method(method) == "per_beat"
        else EYEFLOW_MISC_WAVEFORM_JOINT
    )
    group = _misc_group(source, name)
    t = _dataset_vector(group, "cardiac_phase")
    if t is None:
        return None
    out: dict[str, dict[str, np.ndarray] | np.ndarray] = {"t": t}
    n = int(t.size)
    for panel in FIG3_PANEL_KEYS:
        panel_name = FIG3_PANEL_EYEFLOW_GROUPS[panel]
        panel_group = group.get(panel_name) if isinstance(group, h5py.Group) else None
        if not isinstance(panel_group, h5py.Group):
            return None
        band: dict[str, np.ndarray] = {}
        for stat, ds_name in FIG3_BAND_EYEFLOW_DATASETS.items():
            arr = _dataset_vector(panel_group, ds_name)
            if arr is None:
                return None
            band[stat] = arr
            n = min(n, arr.size)
        out[panel] = band
    if n == 0:
        return None
    out["t"] = np.asarray(out["t"], dtype=float)[:n]
    for panel in FIG3_PANEL_KEYS:
        out[panel] = {
            stat: np.asarray(out[panel][stat], dtype=float)[:n]
            for stat in FIG3_BAND_EYEFLOW_DATASETS
        }
    return out


def _spectrum_payload_from_source(
    source: h5py.Group,
    *,
    method: str,
    cumulative: bool,
) -> dict[str, np.ndarray] | None:
    """Official Fig. 4 payload from joint/per-beat ``misc`` spectrum groups.

    Cumulative spectra are no longer packed by EyeFlow.
    """
    if cumulative:
        return None
    method = normalize_svd_method(method)
    if method == "per_beat":
        group_name = EYEFLOW_MISC_SPECTRUM_PER_BEAT
        y_name = "lambda_mean"
        lo_name = "lambda_lo"
        hi_name = "lambda_hi"
    else:
        group_name = EYEFLOW_MISC_SPECTRUM_JOINT
        y_name = "lambda"
        lo_name = hi_name = ""

    group = _misc_group(source, group_name)
    mode = _dataset_vector(group, "svd_mode_index")
    mean = _dataset_vector(group, y_name)
    if mode is None or mean is None:
        # Archives packed before the ``misc`` layout keep the spectrum under
        # ``figures/fig4_energy_spectrum*`` with a ``mode`` index. Reading
        # both keeps cohort Fig. 4 available when reprocessing older data.
        legacy_group, legacy_names = _LEGACY_SPECTRUM_GROUPS[method]
        group = _legacy_figure_group(source, legacy_group)
        mode = _dataset_vector(group, "mode")
        mean = _dataset_vector(group, legacy_names["mean"])
        if mode is None or mean is None:
            return None
        lo_name = legacy_names.get("lo", "")
        hi_name = legacy_names.get("hi", "")
    n = min(mode.size, mean.size)
    if n == 0:
        return None
    payload = {"mode": mode[:n], "mean": mean[:n]}
    if method == "per_beat":
        lo = _dataset_vector(group, lo_name)
        hi = _dataset_vector(group, hi_name)
        if lo is not None and hi is not None:
            n = min(n, lo.size, hi.size)
            payload = {
                "mode": payload["mode"][:n],
                "mean": payload["mean"][:n],
                "lo": lo[:n],
                "hi": hi[:n],
            }
    return payload


def _source_figure_payload(
    h5_path: Path | str,
    vessel: str,
    signal: str,
    loader,
):
    """Open one vessel/signal group and apply an official figure-payload reader."""
    with h5py.File(h5_path, "r") as h5file:
        root = find_lowrank_metrics_group(h5file)
        if root is None:
            return None
        source = root.get(f"{vessel}/{signal}")
        if not isinstance(source, h5py.Group):
            return None
        if not _group_flag_or_payload(
            source,
            "svd_available",
            default_from_payload=_source_group_has_metric_payload(source),
        ):
            return None
        return loader(source)


def load_figure2_payload(
    h5_path: Path | str,
    vessel: str,
    signal: str = COHORT_SIGNAL,
) -> dict[str, np.ndarray] | None:
    """Load official EyeFlow Fig. 2 storage for one vessel."""
    return _source_figure_payload(h5_path, vessel, signal, _fig2_payload_from_source)


def load_figure3_payload(
    h5_path: Path | str,
    vessel: str,
    signal: str = COHORT_SIGNAL,
    *,
    svd_method: str = DEFAULT_SVD_METHOD,
    beats_band: bool = False,
) -> dict | None:
    """Load official EyeFlow Fig. 3 storage for one vessel/SVD method."""
    method = normalize_svd_method(svd_method)
    return _source_figure_payload(
        h5_path,
        vessel,
        signal,
        lambda source: _fig3_payload_from_source(
            source, method, beats_band=beats_band
        ),
    )


def load_spectrum_payload(
    h5_path: Path | str,
    vessel: str,
    signal: str = COHORT_SIGNAL,
    *,
    svd_method: str = DEFAULT_SVD_METHOD,
    cumulative: bool = False,
) -> dict[str, np.ndarray] | None:
    """Load official EyeFlow Fig. 4 storage for one vessel/SVD method."""
    method = normalize_svd_method(svd_method)
    return _source_figure_payload(
        h5_path,
        vessel,
        signal,
        lambda source: _spectrum_payload_from_source(
            source, method=method, cumulative=bool(cumulative)
        ),
    )


def _source_has_metric_payload(metrics: dict, source_name: str) -> bool:
    """True when the vessel/signal source contains low-rank endpoint content."""
    prefixes = (
        f"{source_name}/endpoints/joint/",
        f"{source_name}/endpoints/per_beat/",
        f"{source_name}/misc/",
        f"{source_name}/decomposition/singular_values",
        f"{source_name}/decomposition/singular_energy_fraction",
    )
    return any(key.startswith(prefix) for prefix in prefixes for key in metrics)


def _source_available(metrics: dict, source_name: str, flag_name: str) -> bool:
    """Honor explicit QC flags, otherwise infer availability from metrics."""
    flag = metrics.get(f"{source_name}/qc/{flag_name}")
    if flag is None:
        return _source_has_metric_payload(metrics, source_name)
    return int(np.asarray(flag).reshape(-1)[0]) == 1


def _group_flag_or_payload(
    source: h5py.Group,
    flag_name: str,
    *,
    default_from_payload: bool,
) -> bool:
    """Read ``qc/<flag_name>`` when present, else use payload inference."""
    flag = _scalar_from_group(source, f"qc/{flag_name}", float("nan"))
    if np.isfinite(flag):
        return int(flag) == 1
    return bool(default_from_payload)


def _source_group_has_metric_payload(source: h5py.Group) -> bool:
    """True when a source group has official endpoints, misc, or spectra."""
    endpoints = source.get("endpoints")
    if isinstance(endpoints, h5py.Group):
        for method in SVD_METHODS:
            if _group_has_datasets(endpoints.get(method)):
                return True
    misc = source.get("misc")
    if isinstance(misc, h5py.Group):
        for obj in misc.values():
            if isinstance(obj, h5py.Group):
                if _group_has_datasets(obj):
                    return True
                if any(isinstance(child, h5py.Group) for child in obj.values()):
                    return True
    decomp = source.get("decomposition")
    if isinstance(decomp, h5py.Group) and any(
        name in decomp for name in ("singular_values", "singular_energy_fraction")
    ):
        return True
    per_beat = source.get("per_beat")
    return isinstance(per_beat, h5py.Group) and any(
        name in per_beat for name in ("singular_values_b", "u_mode1_tb")
    )


def _scalar_from_group(
    group: h5py.Group | None,
    name: str,
    default: float = float("nan"),
) -> float:
    """Read one numeric dataset (nested path ok) as a Python float."""
    if group is None:
        return float(default)
    try:
        obj = group[name]
    except KeyError:
        return float(default)
    arr = np.asarray(obj).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def load_vessel_data_from_result_h5(
    h5_path: Path | str,
    vessel: str,
    *,
    signal: str = COHORT_SIGNAL,
) -> dict | None:
    """Load one vessel's packed endpoints from HDF5 for cohort tables / Fig. 4.

    EyeFlow already wrote A1, λ_m, per-beat arrays, etc. into the file; this
    pipeline never recomputes them, so the H5 is the only source.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5file:
        root = find_lowrank_metrics_group(h5file)
        if root is None:
            return None
        source = root.get(f"{vessel}/{signal}")
        if not isinstance(source, h5py.Group):
            return None
        has_payload = _source_group_has_metric_payload(source)
        if not _group_flag_or_payload(
            source,
            "input_available",
            default_from_payload=has_payload,
        ):
            return None
        if not _group_flag_or_payload(
            source,
            "svd_available",
            default_from_payload=has_payload,
        ):
            return None

        joint_endpoints = _endpoint_group(source, "joint")
        per_beat_endpoints = _endpoint_group(source, "per_beat")
        variability = source["variability"] if "variability" in source else None
        decomposition = source["decomposition"] if "decomposition" in source else None
        baseline = source["baseline"] if "baseline" in source else None
        beatwise_g = source["beatwise"] if "beatwise" in source else None
        inputs = source["inputs"] if "inputs" in source else None
        beat_period = source["beat_period"] if "beat_period" in source else None

        acq: dict[str, float] = {}
        for key, arr in _read_endpoint_values(joint_endpoints).items():
            flat = np.asarray(arr, dtype=float).reshape(-1)
            acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(variability, h5py.Group):
            for key, arr in _group_dataset_map(variability).items():
                flat = np.asarray(arr, dtype=float).reshape(-1)
                acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(decomposition, h5py.Group):
            for key in (
                "Reff",
                "PR",
                "spectrum_mode_count_M",
            ):
                if key in decomposition:
                    acq[key] = _scalar_from_group(decomposition, key)
        if isinstance(baseline, h5py.Group):
            for key in ("mu_acq", "abs_mu_acq", "sigma_mu_beat", "mad_mu_beat"):
                if key in baseline:
                    acq[key] = _scalar_from_group(baseline, key)

        beatwise = (
            _group_dataset_map(beatwise_g) if isinstance(beatwise_g, h5py.Group) else {}
        )
        # EyeFlow packs per-beat means under baseline/, but cohort code expects
        # them on the beatwise dict (same shape cohort collectors expect).
        if isinstance(baseline, h5py.Group):
            if "mu_b" in baseline and "mu_b" not in beatwise:
                beatwise["mu_b"] = np.asarray(baseline["mu_b"], dtype=float)
            if "abs_mu_b" in baseline and "abs_mu_b" not in beatwise:
                beatwise["abs_mu_b"] = np.asarray(baseline["abs_mu_b"], dtype=float)

        per_beat_svd = _read_endpoint_values(per_beat_endpoints)
        legacy_per_beat = source["per_beat"] if "per_beat" in source else None
        if isinstance(legacy_per_beat, h5py.Group):
            legacy_per_beat_svd = {
                key: value
                for key, value in _group_dataset_map(legacy_per_beat).items()
                if not key.startswith("u_mode") and not key.startswith("scores_mode")
            }
            if "singular_values_b" in legacy_per_beat_svd:
                per_beat_svd["singular_values_b"] = legacy_per_beat_svd[
                    "singular_values_b"
                ]
        per_beat_svd["singular_values_b"] = coerce_beat_spectra(
            per_beat_svd.get("singular_values_b")
        )

        singular_values = np.asarray([], dtype=float)
        energy_fraction = np.asarray([], dtype=float)
        per_beat_spectrum = np.asarray([], dtype=float)
        per_beat_spectrum_sd = np.asarray([], dtype=float)
        joint_spectrum_payload = _spectrum_payload_from_source(
            source, method="joint", cumulative=False
        )
        if joint_spectrum_payload is not None:
            singular_values = np.asarray(joint_spectrum_payload["mean"], dtype=float)
        pb_spectrum_payload = _spectrum_payload_from_source(
            source, method="per_beat", cumulative=False
        )
        if pb_spectrum_payload is not None:
            per_beat_spectrum = np.asarray(pb_spectrum_payload["mean"], dtype=float)
            lo = np.asarray(pb_spectrum_payload.get("lo", []), dtype=float)
            hi = np.asarray(pb_spectrum_payload.get("hi", []), dtype=float)
            n_sd = min(lo.size, hi.size, per_beat_spectrum.size)
            if n_sd:
                # EyeFlow packs lambda_lo/hi as mean -/+ 1 SD across beats,
                # so the half-width recovers the across-beats sample SD.
                per_beat_spectrum_sd = (hi[:n_sd] - lo[:n_sd]) / 2.0
        if isinstance(decomposition, h5py.Group):
            if singular_values.size == 0 and "singular_values" in decomposition:
                singular_values = np.asarray(
                    decomposition["singular_values"], dtype=float
                )
            if "singular_energy_fraction" in decomposition:
                energy_fraction = np.asarray(
                    decomposition["singular_energy_fraction"], dtype=float
                )

        if (
            isinstance(inputs, h5py.Group)
            and "valid_fraction_columns_per_beat" in inputs
        ):
            vfb = np.asarray(inputs["valid_fraction_columns_per_beat"], dtype=float)
        else:
            vfb = np.asarray([], dtype=float)

        if isinstance(beat_period, h5py.Group):
            beat_period_mean = _scalar_from_group(beat_period, "mean")
            beat_period_sd = _scalar_from_group(beat_period, "std")
        else:
            beat_period_mean = float("nan")
            beat_period_sd = float("nan")

        if "mu_b" in beatwise:
            period_b = np.full(np.asarray(beatwise["mu_b"]).shape[0], beat_period_mean)
        elif "R0_b" in beatwise:
            period_b = np.full(np.asarray(beatwise["R0_b"]).shape[0], beat_period_mean)
        elif "R0" in per_beat_svd:
            period_b = np.full(
                np.asarray(per_beat_svd["R0"]).shape[0], beat_period_mean
            )
        else:
            period_b = np.asarray([], dtype=float)

        n_valid = (
            int(_scalar_from_group(inputs, "n_valid_columns", 0))
            if isinstance(inputs, h5py.Group)
            else 0
        )
        n_total = (
            int(_scalar_from_group(inputs, "n_total_columns", 0))
            if isinstance(inputs, h5py.Group)
            else 0
        )

    return {
        "acq": acq,
        "beatwise": beatwise,
        "per_beat_svd": per_beat_svd,
        "mu": acq.get("mu_acq", float("nan")),
        "energy_fraction": energy_fraction,
        "singular_values": singular_values,
        "per_beat_spectrum": per_beat_spectrum,
        "per_beat_spectrum_sd": per_beat_spectrum_sd,
        "beat_period_mean": beat_period_mean,
        "beat_period_sd": beat_period_sd,
        "beat_period_b": period_b,
        "valid_fraction_per_beat": vfb,
        "n_valid_columns": n_valid,
        "n_total_columns": n_total,
    }


def load_acquisition_from_result_h5(
    h5_path: Path | str,
    *,
    veins_flag: bool = False,
    signal: str = COHORT_SIGNAL,
) -> dict[str, dict | None] | None:
    """Load artery/(optional) vein bundles from packed AE or EyeFlow metrics."""
    h5_path = Path(h5_path)
    if not result_h5_has_lowrank(h5_path):
        return None
    result: dict[str, dict | None] = {}
    any_ok = False
    for vessel in enabled_vessels(veins_flag):
        data = load_vessel_data_from_result_h5(h5_path, vessel, signal=signal)
        result[vessel] = data
        if data is not None:
            any_ok = True
    return result if any_ok else None


def _flatten_h5_group(group: h5py.Group, prefix: str = "") -> dict:
    """Recursively flatten datasets under ``group`` to relative metric keys."""
    metrics: dict = {}
    for key, obj in group.items():
        rel = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(obj, h5py.Dataset):
            if obj.shape == () and obj.dtype.kind in {"O", "S", "U"}:
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                metrics[rel] = val if isinstance(val, str) else np.asarray(obj)
            else:
                metrics[rel] = np.asarray(obj)
        elif isinstance(obj, h5py.Group):
            metrics.update(_flatten_h5_group(obj, rel))
    return metrics


def metrics_from_lowrank_group(
    group: h5py.Group,
    *,
    veins_flag: bool = False,
) -> tuple[dict, list[str]]:
    """Flatten EyeFlow/AE low-rank group into AngioEye metric keys."""
    metrics = _flatten_h5_group(group)
    if not veins_flag:
        metrics = {
            key: value for key, value in metrics.items() if not key.startswith("vein/")
        }
    resolved: list[str] = []
    for source_name in SOURCE_NAMES:
        if not veins_flag and source_name.startswith("vein/"):
            continue
        if _source_available(
            metrics,
            source_name,
            "input_available",
        ) and _source_available(metrics, source_name, "svd_available"):
            resolved.append(source_name)
    return metrics, resolved


# =====================================================================
# ProcessResult
# =====================================================================


def build_ingest_attrs(
    representations: list[str],
    input_beat_period_path: str,
    *,
    veins_flag: bool = False,
) -> dict:
    """Pipeline-group attributes stored on the AngioEye Processing group."""
    return {
        "pipeline_family": "low_rank_waveform_decomposition",
        "svd_method": "joint and per-beat",
        "aggregation": "median over valid (beat, branch, radius) columns",
        "vessels": list(enabled_vessels(veins_flag)),
        "veins_flag": bool(veins_flag),
        "representations": representations,
        "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
        "context_endpoint": "R0",
        "input_beat_period_path": input_beat_period_path,
        "source": "eyeflow_metrics",
    }


def ingest_lowrank_from_h5(
    h5file,
    *,
    veins_flag: bool = False,
) -> ProcessResult:
    """Flatten packed EyeFlow/AE low-rank metrics into a ProcessResult."""
    root = find_eyeflow_lowrank_group(h5file)
    if root is None:
        root = find_pipeline_group(h5file, PIPELINE_NAME)
    if root is None:
        raise ValueError(
            "EyeFlow low-rank metrics not found. Expected "
            "'Processing/Metrics/lowrank_waveform_decomposition' on the "
            "input HDF5. Run EyeFlow lowrank_waveform_decomposition first."
        )

    metrics, resolved = metrics_from_lowrank_group(root, veins_flag=bool(veins_flag))
    if not resolved:
        raise ValueError(
            "EyeFlow low-rank metrics group is present but no vessel/signal "
            "source has qc/input_available=1 "
            f"(veins_flag={bool(veins_flag)})."
        )
    attrs = build_ingest_attrs(
        resolved,
        T_INPUT,
        veins_flag=bool(veins_flag),
    )
    return ProcessResult(metrics=metrics, attrs=attrs)


def write_acquisition_figures(
    *,
    source_h5_path: Path | str,
    output_h5_path: Path | str,
    veins_flag: bool = False,
) -> list[Path]:
    """Write Figs 2--4 beside the result H5 from packed EyeFlow low-rank metrics.

    One figure set per vessel and signal source, written to
    ``<png dir>/<vessel>/<signal>/``. Figures follow the packed metrics
    rather than ``veins_flag``: every source EyeFlow wrote is plotted, so
    venous figures appear whenever venous metrics exist.
    """
    del veins_flag  # figures follow the packed sources, not the ingest flag
    source_h5_path = Path(source_h5_path)
    output_h5_path = Path(output_h5_path)
    if not result_h5_has_lowrank(source_h5_path):
        return []
    png_dir = companion_png_dir_for_result(output_h5_path)
    stem = acquisition_fig_stem(output_h5_path, source_h5_path)
    written: list[Path] = []
    for vessel in ALL_VESSELS:
        for signal in SIGNALS:
            try:
                data = load_vessel_data_from_result_h5(
                    source_h5_path, vessel, signal=signal
                )
            except Exception as exc:  # keep the other sources plottable
                warnings.warn(
                    f"low-rank metrics for {vessel}/{signal} could not be "
                    f"read from {source_h5_path.name}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if data is None:
                continue
            written.extend(
                LowRankWaveformAcquisitionFigures.plot_all(
                    source_h5_path,
                    {vessel: data},
                    png_dir / vessel / signal,
                    file_stem=stem,
                    signal=signal,
                    vessels=(vessel,),
                )
            )
    return written


# =====================================================================
# Pipeline
# =====================================================================


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformIngest(ProcessPipeline):
    """Ingest EyeFlow-packed low-rank metrics and write Figs 2-4.

    Registry name stays ``lowrank_waveform_decomposition``. SVD lives in
    EyeFlow (joint and per-beat always; Veins selects both). Cohort
    figures and ``lowrank_cohort.h5`` live in
    postprocess.lowrank_waveform_cohort.
    """

    description = (
        "Ingest EyeFlow low-rank waveform metrics "
        "(Processing/Metrics/lowrank_waveform_decomposition) into the "
        "AngioEye result H5 and write Figs 2--4. Requires EyeFlow to have "
        "already run lowrank_waveform_decomposition on the input."
    )

    # EyeFlow now packs arterial and venous sources for every acquisition,
    # so both are ingested and plotted by default.
    veins_flag = True

    def run(self, h5file) -> ProcessResult:
        """Copy EyeFlow-packed low-rank metrics into an AngioEye ProcessResult."""
        return ingest_lowrank_from_h5(
            h5file,
            veins_flag=bool(self.veins_flag),
        )

    def write_companions(
        self,
        result: ProcessResult,
        *,
        source_h5_path: Path | str,
        output_h5_path: Path | str,
    ) -> list[Path]:
        """Write Figs 2--4 next to the result H5 from the source EyeFlow file."""
        del result
        return write_acquisition_figures(
            source_h5_path=source_h5_path,
            output_h5_path=output_h5_path,
            veins_flag=bool(self.veins_flag),
        )


def _dataset_array(group: h5py.Group | None, name: str) -> np.ndarray | None:
    """Read dataset ``name`` from ``group``, or None if missing."""
    if group is None or name not in group:
        return None
    return np.asarray(group[name], dtype=float)


def _mode_recon(u: np.ndarray | None, scores: np.ndarray | None) -> np.ndarray | None:
    """Outer product ``u(t) * score(b,k,r)`` → ``(n_t, n_beats, k, r)``."""
    if u is None or scores is None or u.size == 0 or scores.size == 0:
        return None
    u = np.asarray(u, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float)
    return u.reshape((u.shape[0],) + (1,) * scores.ndim) * scores[None, ...]


def _mode_recon_per_beat(
    u_tb: np.ndarray | None, scores_bkr: np.ndarray | None
) -> np.ndarray | None:
    """Beat-local ``u(t,b) * score(b,k,r)`` → ``(n_t, n_beats, k, r)``."""
    if u_tb is None or scores_bkr is None:
        return None
    u = np.asarray(u_tb, dtype=float)
    scores = np.asarray(scores_bkr, dtype=float)
    if u.size == 0 or scores.size == 0 or u.ndim != 2 or scores.ndim != 3:
        return None
    if u.shape[1] != scores.shape[0]:
        return None
    return u[:, :, None, None] * scores[None, ...]


def load_packed_waveform(
    h5_path: Path | str,
    vessel: str,
    signal: str = COHORT_SIGNAL,
    *,
    svd_method: str = DEFAULT_SVD_METHOD,
) -> dict | None:
    """Reconstruct beat-aligned waveforms from packed EyeFlow low-rank outputs.

    Uses ``decomposition/``, ``baseline/mu_bkr``, ``residuals/r*_t_bkr``, and
    ``inputs/valid_column_mask_bkr``. Does not read Velocity* groups.
    ``v = μ + r1 + a1 u1`` (or ``μ + r2 + a1 u1 + a2 u2`` if r1 is missing).
    ``v``, ``μ``, and ``w`` always come from that joint reconstruction.
    ``svd_method="per_beat"`` replaces ``a1u1`` / ``a2u2`` with beat-local
    ``u_m(t,b) a_m(b,k,r)`` from ``per_beat/``.
    """
    with h5py.File(h5_path, "r") as h5:
        root = find_lowrank_metrics_group(h5)
        if root is None:
            return None
        source = root.get(f"{vessel}/{signal}")
        if not isinstance(source, h5py.Group):
            return None
        if not _group_flag_or_payload(
            source,
            "svd_available",
            default_from_payload=_source_group_has_metric_payload(source),
        ):
            return None

        decomp = source.get("decomposition")
        baseline = source.get("baseline")
        inputs = source.get("inputs")
        residuals = source.get("residuals")
        beat_period = source.get("beat_period")
        per_beat = source.get("per_beat")
        if not isinstance(decomp, h5py.Group):
            return None

        u1 = _dataset_array(decomp, "u_mode1")
        u2 = _dataset_array(decomp, "u_mode2")
        s1 = _dataset_array(decomp, "scores_mode1_bkr")
        s2 = _dataset_array(decomp, "scores_mode2_bkr")
        mu = _dataset_array(
            baseline if isinstance(baseline, h5py.Group) else None, "mu_bkr"
        )
        valid = None
        if isinstance(inputs, h5py.Group) and "valid_column_mask_bkr" in inputs:
            valid = np.asarray(inputs["valid_column_mask_bkr"], dtype=bool)
        r1 = _dataset_array(
            residuals if isinstance(residuals, h5py.Group) else None, "r1_t_bkr"
        )
        r2 = _dataset_array(
            residuals if isinstance(residuals, h5py.Group) else None, "r2_t_bkr"
        )
        period = (
            _scalar_from_group(beat_period, "mean")
            if isinstance(beat_period, h5py.Group)
            else float("nan")
        )
        pb_group = per_beat if isinstance(per_beat, h5py.Group) else None
        u1_tb = _dataset_array(pb_group, "u_mode1_tb")
        u2_tb = _dataset_array(pb_group, "u_mode2_tb")
        s1_bkr = _dataset_array(pb_group, "scores_mode1_bkr")
        s2_bkr = _dataset_array(pb_group, "scores_mode2_bkr")

    a1u1 = _mode_recon(u1, s1)
    a2u2 = _mode_recon(u2, s2)
    if a1u1 is None or mu is None or valid is None:
        return None
    spatial = a1u1.shape[1:]
    if mu.shape != spatial or valid.shape != spatial or not np.any(valid):
        return None
    if r1 is not None and r1.shape == a1u1.shape:
        w = r1 + a1u1
    elif r2 is not None and a2u2 is not None and r2.shape == a1u1.shape:
        w = r2 + a1u1 + a2u2
    else:
        return None
    if a2u2 is None or a2u2.shape != a1u1.shape:
        a2u2 = np.full(a1u1.shape, np.nan, dtype=float)

    method = normalize_svd_method(svd_method)
    if method == "per_beat":
        a1u1_pb = _mode_recon_per_beat(u1_tb, s1_bkr)
        a2u2_pb = _mode_recon_per_beat(u2_tb, s2_bkr)
        if a1u1_pb is None or a1u1_pb.shape != a1u1.shape:
            return None
        a1u1 = a1u1_pb
        if a2u2_pb is None or a2u2_pb.shape != a1u1.shape:
            a2u2 = np.full(a1u1.shape, np.nan, dtype=float)
        else:
            a2u2 = a2u2_pb

    return {
        "v": w + mu[None, ...],
        "mu": mu,
        "w": w,
        "a1u1": a1u1,
        "a2u2": a2u2,
        "valid": valid.astype(bool),
        "n_t": int(a1u1.shape[0]),
        "beat_period_mean": float(period),
        "svd_method": method,
    }


class LowRankWaveformAcquisitionFigures:
    """Per-acquisition arterial Figs. 2--4 from packed EyeFlow low-rank metrics.

    Fig. 2 is the full-acquisition cardiac velocity (spatial mean over
    ``(k, r)`` with 0.1 s std whiskers). It does not use EyeFlow's packed
    one-cycle ``misc/acquisition_level_velocity`` summary.
    """

    VESSEL_VELOCITY_NODES = {"artery": "Artery", "vein": "Vein"}

    @classmethod
    def _velocity_candidates(cls, kind: str, vessel: str, signal: str) -> tuple[str, ...]:
        """EyeFlow velocity paths for one ``segments``/``global`` source.

        Falls back to the raw node when the signal has no dedicated
        velocity trace, so Fig. 2 still renders.
        """
        vessel_node = cls.VESSEL_VELOCITY_NODES.get(str(vessel).lower())
        if vessel_node is None:
            return ()
        nodes = [VELOCITY_SIGNAL_NODES.get(str(signal).lower(), "Raw")]
        if "Raw" not in nodes:
            nodes.append("Raw")
        return tuple(
            f"{prefix}Processing/Velocity/{kind}/{vessel_node}/{node}/value"
            for node in nodes
            for prefix in ("", "EyeFlow/")
        )
    FIG2_ASPECT = 3.0
    FIG2_HEIGHT = 2.8
    FIG2_WHISKER_INTERVAL_S = 0.1
    FIG2_X_PAD_FRAC = 0.02
    FIG3_X_PAD_FRAC = 0.02
    # Panel cell width for Fig. 3. Panels are square (box_aspect=1), so a
    # cell much wider than the axes height shows up as dead space between
    # columns; this keeps the cell close to the drawn panel while leaving
    # room for each panel's own y tick labels.
    FIG3_PANEL_W = 1.95
    FIG3_Y_TICK_BINS = 4
    FIG2_TICK_SIZE = 12
    FIG2_LABEL_SIZE = 14
    FIG2_M_S_PEAK = 1.0
    FIG2_M_S_TO_MM_S = 1000.0
    PANEL_SIZE = 2.5

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9, label_size: int = 10) -> None:
        """Turn off grid, show spines, set tick/label sizes."""
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)

    @classmethod
    def plot_all(
        cls,
        h5_path: Path | str,
        vessel_bundle: dict[str, dict | None],
        out_dir: Path,
        *,
        signal: str = "raw",
        file_stem: str | None = None,
        veins_flag: bool = False,
        vessels: tuple[str, ...] | None = None,
    ) -> list[Path]:
        """Write the Fig. 2--4 set for one vessel/signal into ``out_dir``.

        Fig. 2 has no SVD. Figs. 3--4 carry a ``<basis>_<observation level>``
        suffix: ``joint_acq`` (one acquisition-wide basis), ``per_beat_acq``
        (beat-local bases summarized to the acquisition), and
        ``per_beat_beats`` (beat-local bases kept at beat level, so the band
        spans beats). ``joint_beat`` is not canonical and is not written.
        Variants whose packed payload is absent are skipped.
        """
        h5_path = Path(h5_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or h5_path.stem
        vessels = tuple(vessels or enabled_vessels(bool(veins_flag)))
        written: list[Path] = []
        expected: list[str] = []

        def _keep(name: str, plotter, /, **kwargs) -> None:
            """Draw one figure; never let its failure hide the others."""
            expected.append(name)
            path = safe_figure(name, plotter, **kwargs)
            if path is not None:
                written.append(path)

        _keep(
            "fig2_frequency_velocity",
            cls.plot_frequency_velocity,
            h5_path=h5_path,
            out_path=out_dir / f"{stem}_fig2_frequency_velocity.png",
            signal=signal,
            vessels=vessels,
        )
        fig3_variants = (
            ("joint_acq", "joint", "beat_location"),
            ("per_beat_acq", "per_beat", "beat_location"),
            ("per_beat_beats", "per_beat", "beat"),
        )
        for suffix, method, variability in fig3_variants:
            _keep(
                f"fig3_waveform_decomposition_{suffix}",
                cls.plot_waveform_decomposition,
                h5_path=h5_path,
                out_path=(
                    out_dir / f"{stem}_fig3_waveform_decomposition_{suffix}.png"
                ),
                signal=signal,
                vessels=vessels,
                svd_method=method,
                variability_method=variability,
            )
        fig4_variants = (
            ("joint_acq", "joint", True),
            ("per_beat_acq", "per_beat", False),
            ("per_beat_beats", "per_beat", True),
        )
        for suffix, method, band in fig4_variants:
            _keep(
                f"fig4_energy_spectrum_{suffix}",
                cls.plot_energy_spectrum,
                vessel_bundle=vessel_bundle,
                out_path=out_dir / f"{stem}_fig4_energy_spectrum_{suffix}.png",
                vessels=vessels,
                svd_method=method,
                h5_path=h5_path,
                signal=signal,
                band=band,
            )
        report_missing_figures(
            f"{'/'.join(vessels)}/{signal}",
            expected,
            written,
            source_label=stem,
        )
        return written

    @staticmethod
    def _velocity_dt_seconds(h5: h5py.File) -> float:
        """Sample interval of the Doppler velocity time series (seconds)."""
        dt = float(h5.attrs.get("dt_seconds", np.nan)) if h5.attrs else float("nan")
        if np.isfinite(dt) and dt > 0:
            return dt
        batch = float(h5.attrs.get("batch_stride", np.nan)) if h5.attrs else float("nan")
        fs = float(h5.attrs.get("sampling_freq", np.nan)) if h5.attrs else float("nan")
        if np.isfinite(batch) and np.isfinite(fs) and fs > 0 and batch > 0:
            return batch / fs
        return float("nan")

    @classmethod
    def _velocity_to_mm_s(cls, values: np.ndarray) -> np.ndarray:
        """Display native EyeFlow velocity in mm/s.

        Some files store SI metres per second (peak ≪ 1); those are scaled.
        Values already in mm/s are left unchanged.
        """
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        peak = float(np.nanmax(np.abs(arr)))
        if np.isfinite(peak) and 0.0 < peak < cls.FIG2_M_S_PEAK:
            return arr * cls.FIG2_M_S_TO_MM_S
        return arr

    @classmethod
    def _segment_spatial_mean_std(
        cls, block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse a ``(T, k, r[, ...])`` velocity block to spatial mean ± std."""
        arr = np.asarray(block, dtype=float)
        if arr.ndim < 2:
            flat = arr.reshape(-1)
            return flat, np.zeros_like(flat)
        spatial = arr.reshape(arr.shape[0], -1)
        mean, lo, _hi = mean_pm_std(spatial, axis=1)
        return mean, mean - lo

    @classmethod
    def _fig2_waveform(
        cls,
        h5: h5py.File,
        h5_path: Path,
        vessel: str,
        signal: str,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Full-acquisition spatial mean ± std and sample interval.

        Prefers ``Velocity/segments``, then global velocity, then concatenated
        packed beat-aligned ``v``. Ignores the one-cycle figure payload.
        """
        seg_path = find_first_existing_path(
            h5, list(cls._velocity_candidates("segments", vessel, signal))
        )
        glob_path = find_first_existing_path(
            h5, list(cls._velocity_candidates("global", vessel, signal))
        )
        dt_s = cls._velocity_dt_seconds(h5)
        if seg_path is not None:
            mean, std = cls._segment_spatial_mean_std(
                np.asarray(h5[seg_path], dtype=float)
            )
            return mean, std, dt_s
        if glob_path is not None:
            mean = np.asarray(h5[glob_path], dtype=float).reshape(-1)
            return mean, np.zeros_like(mean), dt_s

        packed = load_packed_waveform(h5_path, vessel, signal=signal)
        if packed is None:
            return None
        v = np.asarray(packed["v"], dtype=float)
        n_t = int(packed["n_t"])
        n_beats = int(v.shape[1]) if v.ndim > 1 else 1
        v_cat = np.transpose(v, (1, 0) + tuple(range(2, v.ndim)))
        v_cat = v_cat.reshape((n_beats * n_t,) + v.shape[2:])
        mean, std = cls._segment_spatial_mean_std(v_cat)
        if not (np.isfinite(dt_s) and dt_s > 0):
            period = float(packed["beat_period_mean"])
            if np.isfinite(period) and period > 0 and n_t > 0:
                dt_s = period / n_t
        return mean, std, dt_s

    @classmethod
    def plot_frequency_velocity(
        cls,
        h5_path: Path | str,
        out_path: Path,
        *,
        signal: str = "raw",
        vessels: tuple[str, ...] | None = None,
    ) -> Path | None:
        """Fig. 2: full-acquisition arterial velocity, time in seconds.

        Spatial mean over vessel locations ``(k, r)`` with whiskers showing
        the spatial standard deviation. Falls back to the global velocity
        trace, then to concatenated packed beats, when segment data are
        missing. Does not plot EyeFlow's one-cycle figure payload.
        """
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        vessels = list(vessels or FIGURE_VESSELS)
        n_rows = max(len(vessels), 1)
        fig_w = cls.FIG2_HEIGHT * cls.FIG2_ASPECT
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(fig_w, cls.FIG2_HEIGHT * n_rows),
            squeeze=False,
        )

        def _two_sig(x, _pos) -> str:
            if not np.isfinite(x) or x == 0:
                return "0"
            return f"{x:.2g}"

        any_data = False
        with h5py.File(h5_path, "r") as h5:
            for row_idx, vessel in enumerate(vessels):
                ax = axes[row_idx, 0]
                series = cls._fig2_waveform(h5, h5_path, vessel, signal)
                mean = std = None
                dt_s = float("nan")
                if series is not None:
                    mean, std, dt_s = series
                    mean = cls._velocity_to_mm_s(mean)
                    std = cls._velocity_to_mm_s(std)

                if mean is not None and mean.size:
                    any_data = True
                    n = int(mean.size)
                    if np.isfinite(dt_s) and dt_s > 0:
                        t_s = np.arange(n, dtype=float) * dt_s
                        xlabel = "Time (s)"
                        stride = max(1, int(round(cls.FIG2_WHISKER_INTERVAL_S / dt_s)))
                    else:
                        t_s = np.arange(n, dtype=float)
                        xlabel = "Sample index"
                        stride = 15

                    ax.plot(t_s, mean, color="black", linewidth=1.1, zorder=3)
                    if std is not None and np.any(std > 0):
                        idx = np.arange(0, n, stride, dtype=int)
                        ax.errorbar(
                            t_s[idx],
                            mean[idx],
                            yerr=std[idx],
                            fmt="none",
                            ecolor="black",
                            elinewidth=0.8,
                            capsize=2.5,
                            capthick=0.8,
                            zorder=2,
                        )
                    t0, t1 = _finite_min(t_s), _finite_max(t_s)
                    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
                        t0, t1 = 0.0, 1.0
                    pad = cls.FIG2_X_PAD_FRAC * (t1 - t0)
                    ax.set_xlim(t0 - pad, t1 + pad)
                    if np.isfinite(dt_s) and dt_s > 0:
                        ax.xaxis.set_major_locator(MultipleLocator(0.5))
                    ax.set_xlabel(xlabel, fontsize=cls.FIG2_LABEL_SIZE)
                else:
                    ax.set_xlabel("Time (s)", fontsize=cls.FIG2_LABEL_SIZE)

                ax.axhline(0, color="#555555", linewidth=0.6, linestyle=":")
                ax.set_ylabel("Velocity (mm/s)", fontsize=cls.FIG2_LABEL_SIZE)
                ax.yaxis.set_major_formatter(FuncFormatter(_two_sig))
                ax.set_box_aspect(1.0 / cls.FIG2_ASPECT)
                cls._style_axes(
                    ax, tick_size=cls.FIG2_TICK_SIZE, label_size=cls.FIG2_LABEL_SIZE
                )

        if not any_data:
            plt.close(fig)
            return None
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def _waveform_summary_for_vessel(
        cls,
        h5_path: Path,
        vessel: str,
        signal: str,
        variability_method: str | None = "beat_location",
        *,
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> dict[str, dict[str, np.ndarray]] | None:
        """Build Fig. 3 curves for one vessel and gray-band definition."""
        beats_band = normalize_figure3_variability_method(variability_method) == "beat"
        payload = load_figure3_payload(
            h5_path,
            vessel,
            signal=signal,
            svd_method=svd_method,
            beats_band=beats_band,
        )
        if payload is not None:
            return payload

        packed = load_packed_waveform(
            h5_path, vessel, signal=signal, svd_method=svd_method
        )
        if packed is None:
            return None
        n_t = int(packed["n_t"])
        valid_mask = packed["valid"]

        def _band(block: np.ndarray) -> dict[str, np.ndarray]:
            mean, lo, hi = figure3_panel_mean_pm_std(
                block,
                valid_mask,
                method=variability_method,
            )
            if mean.size == 1:
                mean = np.full(n_t, float(mean[0]), dtype=float)
                lo = np.full(n_t, float(lo[0]), dtype=float)
                hi = np.full(n_t, float(hi[0]), dtype=float)
            return {"mean": mean, "lo": lo, "hi": hi}

        v_band = _band(packed["v"])
        if not np.any(np.isfinite(v_band["mean"])):
            return None
        return {
            "t": np.linspace(0, 1, n_t, endpoint=False),
            "v": v_band,
            "mu": _band(packed["mu"]),
            "x": _band(packed["w"]),
            "a1u1": _band(packed["a1u1"]),
            "a2u2": _band(packed["a2u2"]),
        }

    @staticmethod
    def _panel_ylim(summary: dict, key: str) -> tuple[float, float]:
        """Per-panel y-limits; centered panels are symmetric about 0.

        Panels with no finite sample (an absent mode, or a source with too
        few valid columns) fall back to a unit range: Matplotlib rejects
        NaN/Inf limits, and one such panel must not abort the figure.
        """
        band = summary[key]
        lo_b = _finite_min(band["lo"])
        hi_b = _finite_max(band["hi"])
        if not (np.isfinite(lo_b) and np.isfinite(hi_b)):
            return -1.0, 1.0
        if key in {"x", "a1u1", "a2u2"}:
            extent = max(abs(lo_b), abs(hi_b), 1e-12) * 1.08
            return -extent, extent
        if key == "mu":
            pad = 0.12 * (hi_b - lo_b if hi_b > lo_b else max(abs(hi_b), 1.0))
            return lo_b - pad, hi_b + pad
        lo, hi = min(lo_b, 0.0), max(hi_b, 0.0)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        return lo - pad, hi + pad

    @classmethod
    def plot_waveform_decomposition(
        cls,
        h5_path: Path | str,
        out_path: Path,
        *,
        signal: str = "raw",
        vessels: tuple[str, ...] | None = None,
        variability_method: str | None = "beat_location",
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> Path | None:
        """Fig. 3: arterial v, mu, w, a1u1, a2u2 from packed modes."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        variability_method = normalize_figure3_variability_method(variability_method)
        svd_method = normalize_svd_method(svd_method)
        rows: list[dict[str, np.ndarray] | None] = []
        for vessel in vessels or FIGURE_VESSELS:
            rows.append(
                cls._waveform_summary_for_vessel(
                    h5_path,
                    vessel,
                    signal=signal,
                    variability_method=variability_method,
                    svd_method=svd_method,
                )
            )
        if all(row is None for row in rows):
            return None

        panel_defs = [
            ("v", r"Beat-aligned velocity" "\n" r"$v$"),
            ("mu", r"Baseline level" "\n" r"$\mu$"),
            ("x", r"Centered waveform" "\n" r"$w = v - \mu$"),
            ("a1u1", r"Mode-1 recon." "\n" r"$a_1u_1$"),
            ("a2u2", r"Mode-2 recon." "\n" r"$a_2u_2$"),
        ]
        zero_cols = {"x", "a1u1", "a2u2"}
        n_rows = len(rows)
        n_cols = len(panel_defs)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cls.FIG3_PANEL_W * n_cols, cls.PANEL_SIZE * n_rows),
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        axes = np.atleast_2d(axes)
        label_pad = 8.0
        for row_idx, summary in enumerate(rows):
            if summary is None:
                for col_idx in range(n_cols):
                    axes[row_idx, col_idx].set_visible(False)
                continue
            t = summary["t"]
            t0, t1 = _finite_min(t), _finite_max(t)
            if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
                t0, t1 = 0.0, 1.0
            x_pad = cls.FIG3_X_PAD_FRAC * (t1 - t0)
            for col_idx, (key, title) in enumerate(panel_defs):
                ax = axes[row_idx, col_idx]
                band = summary[key]
                mean = np.asarray(band["mean"], dtype=float)
                lo = np.asarray(band["lo"], dtype=float)
                hi = np.asarray(band["hi"], dtype=float)
                ax.plot(t, mean, color="black", linewidth=1.8)
                ax.fill_between(t, lo, hi, color="black", alpha=0.12, linewidth=0)
                if key in zero_cols:
                    ax.plot(
                        t,
                        np.zeros_like(t),
                        color="black",
                        linewidth=1.0,
                        linestyle=":",
                    )
                y_lo, y_hi = cls._panel_ylim(summary, key)
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlim(t0 - x_pad, t1 + x_pad)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11, pad=6)
                # Every panel carries its own scale, so every panel keeps
                # its y tick labels; only the axis title is not repeated.
                if col_idx == 0:
                    ax.set_ylabel("Velocity (mm/s)", fontsize=10, labelpad=label_pad)
                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=cls.FIG3_Y_TICK_BINS, prune=None)
                )
                if row_idx == n_rows - 1 and col_idx == n_cols // 2:
                    ax.set_xlabel(
                        "Fraction of cardiac cycle",
                        fontsize=10,
                        labelpad=label_pad,
                    )
                cls._style_axes(ax, tick_size=9, label_size=10)
                ax.set_box_aspect(1)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0, wspace=0.01, hspace=0.02)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out_path

    @staticmethod
    def _joint_spectrum(data: dict | None) -> np.ndarray:
        """Packed joint-SVD singular values λ_m, or empty."""
        if data is None:
            return np.asarray([], dtype=float)
        return np.asarray(data.get("singular_values", []), dtype=float)

    @staticmethod
    def _beat_spectra(data: dict | None) -> np.ndarray:
        """Packed per-beat singular values ``(n_beats, n_modes)``."""
        if data is None:
            return np.zeros((0, SPECTRUM_N_MODES), dtype=float)
        pb = data.get("per_beat_svd") or {}
        return coerce_beat_spectra(pb.get("singular_values_b"))

    @classmethod
    def _draw_spectrum_curve(cls, ax, x: np.ndarray, y: np.ndarray) -> None:
        """Draw one black line-and-circle singular-value curve."""
        ax.plot(
            x,
            y,
            color="black",
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            linewidth=1.5,
        )

    @classmethod
    def _draw_spectrum_mean_std(
        cls,
        ax,
        spectra: np.ndarray,
        *,
        cumulative: bool,
        band: bool = True,
    ) -> None:
        """Plot mean ± SD of per-beat spectra; optionally as a running sum."""
        vals = np.asarray(spectra, dtype=float)
        if vals.ndim != 2 or vals.size == 0:
            return
        if cumulative:
            vals = np.nancumsum(vals, axis=1)
        n_modes = int(vals.shape[1])
        x = np.arange(1, n_modes + 1)
        mean, lo, hi = mean_pm_std(vals, axis=0)
        if not cumulative:
            lo = np.maximum(lo, np.where(mean > 0, mean * 1e-6, 1e-12))
        cls._draw_spectrum_curve(ax, x, mean)
        if band:
            ax.fill_between(x, lo, hi, color="black", alpha=0.12, linewidth=0)

    @classmethod
    def _style_spectrum_panel(
        cls, ax, *, n_keep: int, ylabel: str, log_y: bool
    ) -> None:
        """Mode-index x-axis, optional log y, 2:1 panel aspect."""
        modes = np.arange(1, n_keep + 1)
        ax.set_xticks(modes)
        ax.set_xticklabels([str(m) for m in modes])
        ax.set_xlim(0.5, n_keep + 0.5)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(ylabel, fontsize=cls.FIG2_LABEL_SIZE)
        ax.set_xlabel(r"$m$", fontsize=cls.FIG2_LABEL_SIZE)
        ax.set_box_aspect(0.5)
        cls._style_axes(
            ax, tick_size=cls.FIG2_TICK_SIZE, label_size=cls.FIG2_LABEL_SIZE
        )

    @classmethod
    def plot_energy_spectrum(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
        *,
        vessels: tuple[str, ...] | None = None,
        svd_method: str = "joint",
        h5_path: Path | str | None = None,
        signal: str = "raw",
        band: bool = True,
    ) -> Path | None:
        """Fig. 4: singular values vs mode index for one SVD method.

        ``band=False`` draws the curve alone, for the per-beat spectrum
        summarized to one acquisition-level observation.
        """
        return cls._save_energy_spectrum_figure(
            vessel_bundle,
            out_path,
            cumulative=False,
            vessels=vessels,
            svd_method=svd_method,
            h5_path=h5_path,
            signal=signal,
            band=band,
        )

    @classmethod
    def _save_energy_spectrum_figure(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
        *,
        cumulative: bool,
        vessels: tuple[str, ...] | None = None,
        svd_method: str = "joint",
        h5_path: Path | str | None = None,
        signal: str = "raw",
        band: bool = True,
    ) -> Path | None:
        """Write Fig. 4 from joint or packed per-beat singular values."""
        out_path = Path(out_path)
        method = normalize_svd_method(svd_method)
        wanted = list(vessels or FIGURE_VESSELS)
        plot_vessels = [v for v in wanted if vessel_bundle.get(v) is not None] or wanted
        n_keep = SPECTRUM_N_MODES
        fig_h = 3.0
        fig, axes = plt.subplots(
            len(plot_vessels),
            1,
            figsize=(2.0 * fig_h, fig_h * len(plot_vessels)),
            squeeze=False,
        )
        ylabel = r"$\sum_{i=1}^{m}\lambda_i$" if cumulative else r"$\lambda_m$"
        any_drawn = False
        for vessel_idx, vessel in enumerate(plot_vessels):
            ax = axes[vessel_idx, 0]
            data = vessel_bundle.get(vessel)
            payload = (
                load_spectrum_payload(
                    h5_path,
                    vessel,
                    signal=signal,
                    svd_method=method,
                    cumulative=cumulative,
                )
                if h5_path is not None
                else None
            )
            if payload is not None:
                mode = np.asarray(payload["mode"], dtype=float)
                mean = np.asarray(payload["mean"], dtype=float)
                n_modes = min(n_keep, mode.size, mean.size)
                if n_modes:
                    mode = mode[:n_modes]
                    mean = mean[:n_modes]
                    cls._draw_spectrum_curve(ax, mode, mean)
                    lo = np.asarray(payload.get("lo", []) if band else [], dtype=float)
                    hi = np.asarray(payload.get("hi", []) if band else [], dtype=float)
                    if lo.size >= n_modes and hi.size >= n_modes:
                        ax.fill_between(
                            mode,
                            lo[:n_modes],
                            hi[:n_modes],
                            color="black",
                            alpha=0.12,
                            linewidth=0,
                        )
                    any_drawn = True
            elif method == "per_beat":
                spectra = cls._beat_spectra(data)
                if is_usable_beat_spectra(spectra):
                    cls._draw_spectrum_mean_std(
                        ax, spectra[:, :n_keep], cumulative=cumulative, band=band
                    )
                    any_drawn = True
            else:
                spectrum = cls._joint_spectrum(data)
                n_modes = int(min(n_keep, spectrum.size))
                if n_modes:
                    y = spectrum[:n_modes]
                    if cumulative:
                        y = np.cumsum(y)
                    cls._draw_spectrum_curve(ax, np.arange(1, n_modes + 1), y)
                    any_drawn = True
            cls._style_spectrum_panel(
                ax, n_keep=n_keep, ylabel=ylabel, log_y=not cumulative
            )
        if not any_drawn:
            plt.close(fig)
            return None
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
