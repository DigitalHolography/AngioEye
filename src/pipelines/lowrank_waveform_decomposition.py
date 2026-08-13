"""AngioEye low-rank products from EyeFlow-packed metrics.

EyeFlow owns SVD / endpoints and writes them under
``Processing/Metrics/lowrank_waveform_decomposition/`` in ``*_EF.h5``.
This module does **not** import EyeFlow or recompute decomposition: it
ingests that group into the AngioEye result H5 and writes Figs 2--4.
Select ``veins_flag`` and ``svd_method`` (``joint`` or ``per_beat``) the
same way EyeFlow exposes Veins / Joint SVD / Per-beat SVD options.
Cohort Figs 5--7 live in ``postprocess.lowrank_waveform_cohort``.
``lowrank_cohort.h5`` is written by ``scripts.lowrank_cohort_stats``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

from input_output.hdf5_schema import find_pipeline_group
from input_output.inputs import find_hdf5_inputs
from input_output.output_paths import (
    H5_OUTPUT_DIRNAME,
    PNG_OUTPUT_DIRNAME,
    dataset_stem_from_path,
)

from .core.base import (
    ProcessPipeline,
    ProcessResult,
    registerPipeline,
)


T_INPUT = "Processing/VelocityPerBeat/BeatPeriodSeconds/value"

FIGURE_VESSELS = ("artery",)
PIPELINE_NAME = "lowrank_waveform_decomposition"

COHORT_SIGNAL = "raw"
SVD_METHODS = ("joint", "per_beat")
DEFAULT_SVD_METHOD = "joint"

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


def aggregate_rho(R_b: np.ndarray, TPR_b: np.ndarray, stat: str) -> float:
    """Residual ratio R/R0 from separately aggregated residual and pulsatile RMS."""
    r = aggregate_beatwise(R_b, stat)
    t = aggregate_beatwise(TPR_b, stat)
    if not np.isfinite(r) or not np.isfinite(t) or abs(t) <= 1e-12:
        return float("nan")
    return float(r / (t + 1e-12))


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


def beat_mean_pm_std_spatial_median(
    block: np.ndarray,
    valid: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Typical-location beat-to-beat mean ± 1 SD.

    For each spatial location, take the sample mean and SD **across beats**,
    then the spatial median of those curves. Std is never pooled over
    ``(k,r)``, so the band is beat-to-beat rather than spatial. ``block`` is
    ``(n_t, n_beats, ...)`` or ``(n_beats, ...)``; ``valid`` is
    ``(n_beats, ...)``.
    """
    arr = np.asarray(block, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if arr.ndim == mask.ndim:
        arr = arr[None, ...]
    n_t = int(arr.shape[0])
    if mask.shape != arr.shape[1:]:
        empty = np.full(n_t, np.nan, dtype=float)
        return empty, empty, empty
    masked = np.where(mask[None, ...], arr, np.nan)
    mean_loc, lo_loc, _hi_loc = mean_pm_std(masked, axis=1, ddof=ddof)
    std_loc = np.asarray(mean_loc, dtype=float) - np.asarray(lo_loc, dtype=float)
    ok = np.sum(mask, axis=0) > 1
    ok_t = np.broadcast_to(ok, mean_loc.shape)
    mean_loc = np.where(ok_t, mean_loc, np.nan)
    std_loc = np.where(ok_t, std_loc, np.nan)

    def _median_space(values: np.ndarray) -> np.ndarray:
        """Median over flattened spatial axes, keeping the time axis."""
        flat = np.asarray(values, dtype=float).reshape(n_t, -1)
        with np.errstate(all="ignore"):
            return np.nanmedian(flat, axis=1)

    mean = _median_space(mean_loc)
    sd = _median_space(std_loc)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    mean = np.where(np.isfinite(mean), mean, np.nan)
    return mean, mean - sd, mean + sd


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

    Prefers ``*_AE.h5`` (canonical full-chain product). Also accepts ZIP
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
    ZIP / ``*_pipelines_result.h5`` layout: write PNGs next to the result H5
    (do not treat an ancestor folder named ``*_EF`` as an acquisition root).
    """
    output_h5_path = Path(output_h5_path)
    parent = output_h5_path.parent
    if (
        parent.name.lower() == H5_OUTPUT_DIRNAME
        and parent.parent.name.endswith("_AE")
    ):
        return parent.parent / PNG_OUTPUT_DIRNAME
    return parent


def acquisition_fig_stem(
    output_h5_path: Path | str, source_h5_path: Path | str | None = None
) -> str:
    """Stem used for Figs 2--4 filenames (acquisition name, not ``_pipelines_result``)."""
    output_h5_path = Path(output_h5_path)
    parent = output_h5_path.parent
    if (
        parent.name.lower() == H5_OUTPUT_DIRNAME
        and parent.parent.name.endswith("_AE")
    ):
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


def _scalar_from_group(group: h5py.Group, name: str, default: float = float("nan")) -> float:
    """Read one numeric dataset (nested path ok) as a Python float."""
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
        if int(_scalar_from_group(source, "qc/input_available", 0)) != 1:
            return None
        if int(_scalar_from_group(source, "qc/svd_available", 0)) != 1:
            return None

        endpoints = source["endpoints"] if "endpoints" in source else None
        variability = source["variability"] if "variability" in source else None
        decomposition = source["decomposition"] if "decomposition" in source else None
        baseline = source["baseline"] if "baseline" in source else None
        beatwise_g = source["beatwise"] if "beatwise" in source else None
        per_beat_g = source["per_beat"] if "per_beat" in source else None
        inputs = source["inputs"] if "inputs" in source else None
        beat_period = source["beat_period"] if "beat_period" in source else None

        acq: dict[str, float] = {}
        if isinstance(endpoints, h5py.Group):
            for key, arr in _group_dataset_map(endpoints).items():
                flat = np.asarray(arr, dtype=float).reshape(-1)
                acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(variability, h5py.Group):
            for key, arr in _group_dataset_map(variability).items():
                flat = np.asarray(arr, dtype=float).reshape(-1)
                acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(decomposition, h5py.Group):
            for key in (
                "effective_rank",
                "participation_ratio",
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

        per_beat_svd = (
            _group_dataset_map(per_beat_g) if isinstance(per_beat_g, h5py.Group) else {}
        )
        per_beat_svd["singular_values_b"] = coerce_beat_spectra(
            per_beat_svd.get("singular_values_b")
        )

        singular_values = np.asarray([], dtype=float)
        energy_fraction = np.asarray([], dtype=float)
        if isinstance(decomposition, h5py.Group):
            if "singular_values" in decomposition:
                singular_values = np.asarray(
                    decomposition["singular_values"], dtype=float
                )
            if "singular_energy_fraction" in decomposition:
                energy_fraction = np.asarray(
                    decomposition["singular_energy_fraction"], dtype=float
                )

        if isinstance(inputs, h5py.Group) and "valid_fraction_columns_per_beat" in inputs:
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
            period_b = np.full(
                np.asarray(beatwise["mu_b"]).shape[0], beat_period_mean
            )
        elif "TPR_b" in beatwise:
            period_b = np.full(
                np.asarray(beatwise["TPR_b"]).shape[0], beat_period_mean
            )
        elif "TPR_b_pb" in per_beat_svd:
            period_b = np.full(
                np.asarray(per_beat_svd["TPR_b_pb"]).shape[0], beat_period_mean
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
            key: value
            for key, value in metrics.items()
            if not key.startswith("vein/")
        }
    resolved: list[str] = []
    for source_name in SOURCE_NAMES:
        if not veins_flag and source_name.startswith("vein/"):
            continue
        flag = metrics.get(f"{source_name}/qc/input_available")
        if flag is None:
            continue
        if int(np.asarray(flag).reshape(-1)[0]) == 1:
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
    svd_method: str = DEFAULT_SVD_METHOD,
) -> dict:
    """Pipeline-group attributes stored on the AngioEye Processing group."""
    method = normalize_svd_method(svd_method)
    return {
        "pipeline_family": "low_rank_waveform_decomposition",
        "svd_method": method,
        "aggregation": "median over (k,r), then median over b",
        "vessels": list(enabled_vessels(veins_flag)),
        "veins_flag": bool(veins_flag),
        "representations": representations,
        "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
        "context_endpoint": "TPR",
        "input_beat_period_path": input_beat_period_path,
        "source": "eyeflow_metrics",
    }


def ingest_lowrank_from_h5(
    h5file,
    *,
    veins_flag: bool = False,
    svd_method: str = DEFAULT_SVD_METHOD,
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

    metrics, resolved = metrics_from_lowrank_group(
        root, veins_flag=bool(veins_flag)
    )
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
        svd_method=svd_method,
    )
    return ProcessResult(metrics=metrics, attrs=attrs)


def write_acquisition_figures(
    *,
    source_h5_path: Path | str,
    output_h5_path: Path | str,
    veins_flag: bool = False,
    svd_method: str = DEFAULT_SVD_METHOD,
) -> list[Path]:
    """Write Figs 2--4 beside the result H5 from packed EyeFlow low-rank metrics."""
    source_h5_path = Path(source_h5_path)
    output_h5_path = Path(output_h5_path)
    bundle = load_acquisition_from_result_h5(
        source_h5_path,
        veins_flag=bool(veins_flag),
        signal="raw",
    )
    if bundle is None:
        return []
    png_dir = companion_png_dir_for_result(output_h5_path)
    stem = acquisition_fig_stem(output_h5_path, source_h5_path)
    return LowRankWaveformAcquisitionFigures.plot_all(
        source_h5_path,
        bundle,
        png_dir,
        file_stem=stem,
        signal="raw",
        svd_method=svd_method,
    )


# =====================================================================
# Pipeline
# =====================================================================


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformIngest(ProcessPipeline):
    """Ingest EyeFlow-packed low-rank metrics and write Figs 2-4.

    Registry name stays ``lowrank_waveform_decomposition``. SVD lives in
    EyeFlow (Joint SVD / Per-beat SVD options, same as Veins). Cohort
    figures live in postprocess.lowrank_waveform_cohort; ``lowrank_cohort.h5``
    is written by scripts.lowrank_cohort_stats.
    """

    description = (
        "Ingest EyeFlow low-rank waveform metrics "
        "(Processing/Metrics/lowrank_waveform_decomposition) into the "
        "AngioEye result H5 and write Figs 2--4. Requires EyeFlow to have "
        "already run lowrank_waveform_decomposition on the input."
    )

    veins_flag = False
    svd_method = DEFAULT_SVD_METHOD

    def run(self, h5file) -> ProcessResult:
        """Copy EyeFlow-packed low-rank metrics into an AngioEye ProcessResult."""
        return ingest_lowrank_from_h5(
            h5file,
            veins_flag=bool(self.veins_flag),
            svd_method=self.svd_method,
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
            svd_method=self.svd_method,
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


def load_packed_waveform(
    h5_path: Path | str,
    vessel: str,
    signal: str = COHORT_SIGNAL,
) -> dict | None:
    """Reconstruct beat-aligned waveforms from packed EyeFlow low-rank outputs.

    Uses ``decomposition/``, ``baseline/mu_bkr``, ``residuals/r*_t_bkr``, and
    ``inputs/valid_column_mask_bkr``. Does not read Velocity* groups.
    ``v = μ + r1 + a1 u1`` (or ``μ + r2 + a1 u1 + a2 u2`` if r1 is missing).
    """
    with h5py.File(h5_path, "r") as h5:
        root = find_lowrank_metrics_group(h5)
        if root is None:
            return None
        source = root.get(f"{vessel}/{signal}")
        if not isinstance(source, h5py.Group):
            return None
        if int(_scalar_from_group(source, "qc/svd_available", 0)) != 1:
            return None

        decomp = source.get("decomposition")
        baseline = source.get("baseline")
        inputs = source.get("inputs")
        residuals = source.get("residuals")
        beat_period = source.get("beat_period")
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
    return {
        "v": w + mu[None, ...],
        "mu": mu,
        "w": w,
        "a1u1": a1u1,
        "a2u2": a2u2,
        "valid": valid.astype(bool),
        "n_t": int(a1u1.shape[0]),
        "beat_period_mean": float(period),
    }


class LowRankWaveformAcquisitionFigures:
    """Per-acquisition arterial Figs. 2--4 from packed EyeFlow low-rank metrics."""

    FIG2_ASPECT = 3.0
    FIG2_HEIGHT = 2.8
    FIG2_WHISKER_INTERVAL_S = 0.1
    FIG2_X_PAD_FRAC = 0.02
    FIG3_X_PAD_FRAC = 0.02
    FIG2_TICK_SIZE = 12
    FIG2_LABEL_SIZE = 14
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
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> list[Path]:
        """Write Figs. 2--4 into ``out_dir`` from packed low-rank metrics.

        Fig. 3 needs joint modes, so it is skipped for ``per_beat``.
        """
        h5_path = Path(h5_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or h5_path.stem
        method = normalize_svd_method(svd_method)
        written: list[Path] = []
        fig2 = cls.plot_frequency_velocity(
            h5_path,
            out_dir / f"{stem}_fig2_frequency_velocity.png",
            signal=signal,
        )
        if fig2 is not None:
            written.append(fig2)
        if method == "joint":
            fig3 = cls.plot_waveform_decomposition(
                h5_path,
                out_dir / f"{stem}_fig3_waveform_decomposition.png",
                signal=signal,
            )
            if fig3 is not None:
                written.append(fig3)
        written.append(
            cls.plot_energy_spectrum(
                vessel_bundle,
                out_dir / f"{stem}_fig4_energy_spectrum.png",
                svd_method=method,
            )
        )
        written.append(
            cls.plot_energy_spectrum_cumulative(
                vessel_bundle,
                out_dir / f"{stem}_fig4_energy_spectrum_cumulative.png",
                svd_method=method,
            )
        )
        return written

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
    def plot_frequency_velocity(
        cls,
        h5_path: Path | str,
        out_path: Path,
        *,
        signal: str = "raw",
    ) -> Path | None:
        """Fig. 2: reconstructed arterial velocity, beats concatenated in time."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        vessels = list(FIGURE_VESSELS)
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
        for row_idx, vessel in enumerate(vessels):
            ax = axes[row_idx, 0]
            packed = load_packed_waveform(h5_path, vessel, signal=signal)
            mean = std = None
            dt_s = float("nan")
            if packed is not None:
                v = np.asarray(packed["v"], dtype=float)
                # (n_t, n_beats, k, r) → concatenate beats along time.
                n_t = int(packed["n_t"])
                n_beats = int(v.shape[1])
                v_cat = np.transpose(v, (1, 0) + tuple(range(2, v.ndim)))
                v_cat = v_cat.reshape((n_beats * n_t,) + v.shape[2:])
                mean, std = cls._segment_spatial_mean_std(v_cat)
                period = float(packed["beat_period_mean"])
                if np.isfinite(period) and period > 0 and n_t > 0:
                    dt_s = period / n_t

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
                t0, t1 = float(t_s[0]), float(t_s[-1])
                pad = cls.FIG2_X_PAD_FRAC * (t1 - t0 if t1 > t0 else 1.0)
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
    ) -> dict[str, dict[str, np.ndarray]] | None:
        """Build Fig. 3 curves: typical-location beat-to-beat mean ± SD."""
        packed = load_packed_waveform(h5_path, vessel, signal=signal)
        if packed is None:
            return None
        n_t = int(packed["n_t"])
        valid_mask = packed["valid"]

        def _band(block: np.ndarray) -> dict[str, np.ndarray]:
            mean, lo, hi = beat_mean_pm_std_spatial_median(block, valid_mask)
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
    def _row_ylim_first_two(summary: dict) -> tuple[float, float]:
        """Shared y-limits for Fig. 3 panels ``v`` and ``μ``, including 0."""
        bounds: list[float] = []
        for key in ("v", "mu"):
            band = summary[key]
            bounds += [float(np.nanmin(band["lo"])), float(np.nanmax(band["hi"]))]
        lo, hi = min(bounds), max(bounds)
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        return lo - pad, hi + pad

    @staticmethod
    def _row_ylim_last_three(summary: dict) -> tuple[float, float]:
        """Symmetric y-limits about 0 for Fig. 3 panels ``w``, ``a1u1``, ``a2u2``."""
        extents: list[float] = []
        for key in ("x", "a1u1", "a2u2"):
            band = summary[key]
            extents.append(
                max(
                    abs(float(np.nanmin(band["lo"]))),
                    abs(float(np.nanmax(band["hi"]))),
                )
            )
        extent = max(extents) if extents else 1.0
        extent = extent if extent > 0 else 1.0
        extent *= 1.12
        return -extent, extent

    @classmethod
    def plot_waveform_decomposition(
        cls,
        h5_path: Path | str,
        out_path: Path,
        *,
        signal: str = "raw",
    ) -> Path | None:
        """Fig. 3: arterial v, mu, w, a1u1, a2u2 from packed modes/residuals."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        rows: list[dict[str, np.ndarray] | None] = []
        for vessel in FIGURE_VESSELS:
            rows.append(
                cls._waveform_summary_for_vessel(
                    h5_path, vessel, signal=signal
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
            figsize=(cls.PANEL_SIZE * n_cols, cls.PANEL_SIZE * n_rows),
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
            t0, t1 = float(t[0]), float(t[-1])
            x_pad = cls.FIG3_X_PAD_FRAC * (t1 - t0 if t1 > t0 else 1.0)
            ylim_12 = cls._row_ylim_first_two(summary)
            ylim_345 = cls._row_ylim_last_three(summary)
            for col_idx, (key, title) in enumerate(panel_defs):
                ax = axes[row_idx, col_idx]
                band = summary[key]
                mean = np.asarray(band["mean"], dtype=float)
                lo = np.asarray(band["lo"], dtype=float)
                hi = np.asarray(band["hi"], dtype=float)
                ax.plot(t, mean, color="black", linewidth=1.8)
                ax.fill_between(t, lo, hi, color="black", alpha=0.12, linewidth=0)
                ax.plot(
                    t,
                    np.zeros_like(t),
                    color="black",
                    linewidth=1.0,
                    linestyle=":",
                )
                y_lo, y_hi = ylim_345 if key in zero_cols else ylim_12
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlim(t0 - x_pad, t1 + x_pad)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11, pad=6)
                if col_idx == 0:
                    ax.set_ylabel("Velocity (mm/s)", fontsize=10, labelpad=label_pad)
                elif col_idx != 2:
                    ax.tick_params(labelleft=False)
                if row_idx == n_rows - 1 and col_idx == n_cols // 2:
                    ax.set_xlabel(
                        "Fraction of cardiac cycle",
                        fontsize=10,
                        labelpad=label_pad,
                    )
                cls._style_axes(ax, tick_size=9, label_size=10)
                ax.set_box_aspect(1)
        fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.02, hspace=0.02)
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
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> Path:
        """Fig. 4: arterial singular values vs mode index."""
        return cls._save_energy_spectrum_figure(
            vessel_bundle,
            out_path,
            cumulative=False,
            svd_method=svd_method,
        )

    @classmethod
    def plot_energy_spectrum_cumulative(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
        *,
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> Path:
        """Standalone cumulative-sum panel (same 2:1 aspect as Fig. 4)."""
        return cls._save_energy_spectrum_figure(
            vessel_bundle,
            out_path,
            cumulative=True,
            svd_method=svd_method,
        )

    @classmethod
    def _save_energy_spectrum_figure(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
        *,
        cumulative: bool,
        svd_method: str = DEFAULT_SVD_METHOD,
    ) -> Path:
        """Write Fig. 4 (or its cumulative sibling) from packed singular values."""
        out_path = Path(out_path)
        method = normalize_svd_method(svd_method)
        vessels = [
            v for v in FIGURE_VESSELS if vessel_bundle.get(v) is not None
        ] or list(FIGURE_VESSELS)
        n_keep = SPECTRUM_N_MODES
        fig_h = 3.0
        fig, axes = plt.subplots(
            len(vessels),
            1,
            figsize=(2.0 * fig_h, fig_h * len(vessels)),
            squeeze=False,
        )
        ylabel = r"$\sum_{i=1}^{m}\lambda_i$" if cumulative else r"$\lambda_m$"
        for vessel_idx, vessel in enumerate(vessels):
            ax = axes[vessel_idx, 0]
            if method == "per_beat":
                spectra = cls._beat_spectra(vessel_bundle.get(vessel))
                if is_usable_beat_spectra(spectra):
                    cls._draw_spectrum_mean_std(
                        ax, spectra[:, :n_keep], cumulative=cumulative
                    )
            else:
                spectrum = cls._joint_spectrum(vessel_bundle.get(vessel))
                n_modes = int(min(n_keep, spectrum.size))
                if n_modes:
                    y = spectrum[:n_modes]
                    if cumulative:
                        y = np.cumsum(y)
                    cls._draw_spectrum_curve(ax, np.arange(1, n_modes + 1), y)
            cls._style_spectrum_panel(
                ax, n_keep=n_keep, ylabel=ylabel, log_y=not cumulative
            )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
