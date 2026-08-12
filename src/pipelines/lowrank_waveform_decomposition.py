"""AngioEye low-rank products from EyeFlow-packed metrics.

EyeFlow owns SVD / endpoints and writes them under
``Processing/Metrics/lowrank_waveform_decomposition/`` in ``*_EF.h5``.
This module does **not** import EyeFlow or recompute decomposition: it
ingests that group into the AngioEye result H5 and writes Figs 2--4.
Cohort H5 / Figs 5--7 live in ``postprocess.utils.lowrank_waveform``.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

# Worker threads / process pools must not use an interactive GUI backend.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from input_output.hdf5_io import find_first_existing_path
from input_output.hdf5_schema import find_pipeline_group
from input_output.inputs import find_hdf5_inputs, relative_hdf5_parent
from input_output.output_paths import (
    H5_OUTPUT_DIRNAME,
    PNG_OUTPUT_DIRNAME,
    dataset_stem_from_path,
    h5_output_parent,
)

from .core.base import (
    ProcessPipeline,
    ProcessResult,
    registerPipeline,
)


T_INPUT = "Processing/VelocityPerBeat/BeatPeriodSeconds/value"
V_RAW_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
V_RAW_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
VESSEL_TYPES = ("artery", "vein")
# Figures are arterial-only; vein endpoints may still be used for tables.
FIGURE_VESSELS = ("artery",)
PIPELINE_NAME = "lowrank_waveform_decomposition"
# Cohort tables / figs prefer the joint-SVD raw arterial (venous) pack.
COHORT_SIGNAL = "raw"
# EyeFlow packed metrics (active schema uses Processing/Metrics/...).
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

# =====================================================================
# Patient identification
# =====================================================================

PATIENT_ID_RE = re.compile(r"^(\d{6})")


def extract_patient_id(input_path: Path | str) -> str | None:
    """Returns the 6-digit patient ID prefixing the input ZIP/folder name,
    or None if the name isn't ID-prefixed."""
    match = PATIENT_ID_RE.match(Path(input_path).name)
    return match.group(1) if match else None


def prefixed_filename(filename: str, patient_id: str | None) -> str:
    """Prepends ``{patient_id}_`` to filename when a patient ID is known,
    leaving it unchanged otherwise."""
    return f"{patient_id}_{filename}" if patient_id else filename

# =====================================================================
# Vessel/schema resolution
# =====================================================================


def _enabled_vessels(veins_flag: bool) -> tuple[str, ...]:
    return ("artery", "vein") if veins_flag else ("artery",)


def _filter_vessel_candidates(
    candidates: dict[str, str], veins_flag: bool
) -> dict[str, str]:
    if veins_flag:
        return candidates
    return {k: v for k, v in candidates.items() if not k.startswith("vein/")}


def _resolve_vessel_sources(h5file, veins_flag: bool) -> tuple[dict[str, str], str] | None:
    if find_first_existing_path(h5file, [T_INPUT]) is None:
        return None
    candidates = {
        "artery/raw": V_RAW_SEGMENT_INPUT_ARTERY,
        "artery/bandlimited": V_BAND_SEGMENT_INPUT_ARTERY,
        "vein/raw": V_RAW_SEGMENT_INPUT_VEIN,
        "vein/bandlimited": V_BAND_SEGMENT_INPUT_VEIN,
    }
    return _filter_vessel_candidates(candidates, veins_flag), T_INPUT


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
    """Prefer AngioEye-packed group; fall back to EyeFlow Metrics group."""
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
    out: dict[str, np.ndarray] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            out[key] = np.asarray(obj)
    return out


def _scalar_from_group(group: h5py.Group, name: str, default: float = float("nan")) -> float:
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
    """Rebuild the in-memory vessel bundle used by cohort tables from a result H5."""
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
                "alpha",
                "G1",
                "effective_rank",
                "participation_ratio",
                "eta1",
                "eta2",
                "eta12",
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
    for vessel in _enabled_vessels(veins_flag):
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


def _beat_period_path_from_metrics(metrics: dict) -> str:
    for source_name in SOURCE_NAMES:
        key = f"{source_name}/config/input_dataset_path"
        if key not in metrics:
            continue
        # Prefer an explicit beat-period path if present nearby; else default.
        break
    return T_INPUT


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
        "svd_method": "ingested from EyeFlow Metrics/lowrank_waveform_decomposition",
        "aggregation": "median over (k,r), then median over b",
        "vessels": list(_enabled_vessels(veins_flag)),
        "veins_flag": bool(veins_flag),
        "representations": representations,
        "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
        "context_endpoint": "TPR",
        "input_beat_period_path": input_beat_period_path,
        "source": "eyeflow_metrics",
    }


def ingest_lowrank_from_h5(h5file, *, veins_flag: bool = False) -> ProcessResult:
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
    t_path = _beat_period_path_from_metrics(metrics)
    attrs = build_ingest_attrs(resolved, t_path, veins_flag=bool(veins_flag))
    return ProcessResult(metrics=metrics, attrs=attrs)


def write_acquisition_figures(
    *,
    source_h5_path: Path | str,
    output_h5_path: Path | str,
    veins_flag: bool = False,
) -> list[Path]:
    """Write Figs 2--4 beside the result H5 from packed metrics + EF velocity."""
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
    return LowRankAquisitionFigures.plot_all(
        source_h5_path,
        bundle,
        png_dir,
        file_stem=stem,
        signal="raw",
    )


# =====================================================================
# Pipeline
# =====================================================================


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformIngest(ProcessPipeline):
    """Ingest EyeFlow-packed low-rank metrics and write Figs 2--4.

    Registry name stays ``lowrank_waveform_decomposition``. SVD lives in
    EyeFlow; cohort products live in postprocess.utils.lowrank_waveform.
    """

    description = (
        "Ingest EyeFlow low-rank waveform metrics "
        "(Processing/Metrics/lowrank_waveform_decomposition) into the "
        "AngioEye result H5 and write Figs 2--4. Requires EyeFlow to have "
        "already run lowrank_waveform_decomposition on the input."
    )

    veins_flag = False

    def run(self, h5file) -> ProcessResult:
        return ingest_lowrank_from_h5(h5file, veins_flag=bool(self.veins_flag))

    def write_companions(
        self,
        result: ProcessResult,
        *,
        source_h5_path: Path | str,
        output_h5_path: Path | str,
    ) -> list[Path]:
        del result
        return write_acquisition_figures(
            source_h5_path=source_h5_path,
            output_h5_path=output_h5_path,
            veins_flag=bool(self.veins_flag),
        )


class LowRankWaveformAcquisitionFigures:
    """Per-acquisition arterial figures (article Figs. 2--4). Written by the
    pipeline as AE PNG companions under
    ``{stem}_AE/png/`` Written as AE PNG companions by the ingest pipeline.
    Fig. 2 is the cardiac velocity waveform only (3:1), with spatial ``(k,r)``
    std whiskers."""

    FRMS_MAP_CANDIDATES = (
        "EyeFlow/Processing/FrequencyMaps/fRMS_avg/value",
        "Processing/FrequencyMaps/fRMS_avg/value",
    )
    GLOBAL_VELOCITY_CANDIDATES = {
        "artery": (
            "EyeFlow/Processing/Velocity/global/Artery/Raw/value",
            "Processing/Velocity/global/Artery/Raw/value",
        ),
        "vein": (
            "EyeFlow/Processing/Velocity/global/Vein/Raw/value",
            "Processing/Velocity/global/Vein/Raw/value",
        ),
    }
    SEGMENT_VELOCITY_CANDIDATES = {
        "artery": (
            "EyeFlow/Processing/Velocity/segments/Artery/Raw/value",
            "Processing/Velocity/segments/Artery/Raw/value",
        ),
        "vein": (
            "EyeFlow/Processing/Velocity/segments/Vein/Raw/value",
            "Processing/Velocity/segments/Vein/Raw/value",
        ),
    }
    # Fig. 2 waveform panel: width:height = 3:1
    FIG2_ASPECT = 3.0
    FIG2_HEIGHT = 2.8
    # Whiskers every ~100 ms when dt is available; else every 15 samples.
    FIG2_WHISKER_INTERVAL_S = 0.1
    # H5 velocity is already in the project unit (mm/s); do not rescale for display.
    FIG2_VELOCITY_TO_MM_S = 1.0
    FIG2_X_PAD_FRAC = 0.02  # left/right pad as a fraction of the time span
    # Fig. 3 uses a slightly larger pad so the inset stays obvious on square panels.
    FIG3_X_PAD_FRAC = 0.05
    FIG2_TICK_SIZE = 12
    FIG2_LABEL_SIZE = 14
    PANEL_SIZE = 2.5

    @staticmethod
    def output_dir_for(
        output_dir: Path | str, h5_path: Path | str, input_root: Path | str
    ) -> Path:
        """Legacy helper: ``output_dir/h5/<relative>/``.

        Prefer :func:`companion_png_dir_for_result` for the canonical
        ``{stem}_AE/png/`` layout.
        """
        relative_parent = relative_hdf5_parent(h5_path, input_root)
        return h5_output_parent(output_dir, relative_parent)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9, label_size: int = 10) -> None:
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
        patient_id: str | None = None,
        signal: str = "raw",
        file_stem: str | None = None,
    ) -> list[Path]:
        """Write Figs. 2--4 into ``out_dir``.

        ``out_dir`` should be the AE companion PNG folder (``{stem}_AE/png/``).
        """
        h5_path = Path(h5_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or h5_path.stem
        written: list[Path] = []
        written.append(
            cls.plot_frequency_velocity(
                h5_path,
                out_dir / prefixed_filename(f"{stem}_fig2_frequency_velocity.png", patient_id),
            )
        )
        written.append(
            cls.plot_waveform_decomposition(
                h5_path,
                out_dir / prefixed_filename(f"{stem}_fig3_waveform_decomposition.png", patient_id),
                signal=signal,
            )
        )
        written.append(
            cls.plot_energy_spectrum(
                vessel_bundle,
                out_dir / prefixed_filename(f"{stem}_fig4_energy_spectrum.png", patient_id),
            )
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
    def _segment_spatial_mean_std(
        cls, block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse a ``(T, k, r[, ...])`` velocity block to spatial mean ± std."""
        arr = np.asarray(block, dtype=float)
        if arr.ndim < 2:
            flat = arr.reshape(-1)
            return flat, np.zeros_like(flat)
        # Keep time on axis 0; flatten all spatial axes together.
        spatial = arr.reshape(arr.shape[0], -1)
        with np.errstate(all="ignore"):
            mean = np.nanmean(spatial, axis=1)
            std = np.nanstd(spatial, axis=1, ddof=1)
        std = np.where(np.isfinite(std), std, 0.0)
        return mean, std

    @classmethod
    def plot_frequency_velocity(cls, h5_path: Path | str, out_path: Path) -> Path:
        """Fig. 2: arterial cardiac velocity waveform only (3:1), time in s.

        Plots the spatial mean over vessel locations ``(k, r)`` with whiskers
        showing the spatial standard deviation. Falls back to the global
        velocity trace when segment data are unavailable. Velocity is shown in
        mm/s (see ``FIG2_VELOCITY_TO_MM_S``); time in seconds.
        """
        from matplotlib.ticker import FuncFormatter, MultipleLocator

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

        with h5py.File(h5_path, "r") as h5:
            dt_s = cls._velocity_dt_seconds(h5)
            for row_idx, vessel in enumerate(vessels):
                ax = axes[row_idx, 0]
                seg_path = find_first_existing_path(
                    h5, list(cls.SEGMENT_VELOCITY_CANDIDATES.get(vessel, ()))
                )
                glob_path = find_first_existing_path(
                    h5, list(cls.GLOBAL_VELOCITY_CANDIDATES.get(vessel, ()))
                )

                mean = std = None
                if seg_path is not None:
                    mean, std = cls._segment_spatial_mean_std(
                        np.asarray(h5[seg_path], dtype=float)
                    )
                elif glob_path is not None:
                    mean = np.asarray(h5[glob_path], dtype=float).reshape(-1)
                    std = np.zeros_like(mean)

                if mean is not None and mean.size:
                    # Scale stored velocity to mm/s for axis labels/ticks.
                    scale = float(cls.FIG2_VELOCITY_TO_MM_S)
                    mean = mean * scale
                    if std is not None:
                        std = std * scale

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
                    # Left/right padding (fraction of span), same idea as fig. 3 margins.
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

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _median_iqr_curve(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.nanmedian(matrix, axis=1)
        q25 = np.nanpercentile(matrix, 25, axis=1)
        q75 = np.nanpercentile(matrix, 75, axis=1)
        return med, q25, q75

    @classmethod
    def _waveform_summary_for_vessel(
        cls,
        h5_path: Path,
        vessel: str,
        signal: str,
    ) -> dict[str, np.ndarray] | None:
        """Build Fig. 3 curves from EF velocity + packed u/score modes."""
        source_key = f"{vessel}/{signal}"
        with h5py.File(h5_path, "r") as h5:
            schema = _resolve_vessel_sources(h5, veins_flag=False)
            if schema is None:
                return None
            candidates, _t_path = schema
            dataset_path = candidates.get(source_key)
            if dataset_path is None or dataset_path not in h5:
                return None
            v_block = np.asarray(h5[dataset_path], dtype=float)

            root = find_lowrank_metrics_group(h5)
            if root is None:
                return None
            source = root.get(source_key)
            if not isinstance(source, h5py.Group):
                return None
            if int(_scalar_from_group(source, "qc/svd_available", 0)) != 1:
                return None

            decomp = source["decomposition"] if "decomposition" in source else None
            baseline = source["baseline"] if "baseline" in source else None
            inputs = source["inputs"] if "inputs" in source else None
            if not isinstance(decomp, h5py.Group):
                return None

            u1 = (
                np.asarray(decomp["u_mode1"], dtype=float)
                if "u_mode1" in decomp
                else np.asarray([], dtype=float)
            )
            u2 = (
                np.asarray(decomp["u_mode2"], dtype=float)
                if "u_mode2" in decomp
                else np.asarray([], dtype=float)
            )
            s1 = (
                np.asarray(decomp["scores_mode1_bkr"], dtype=float)
                if "scores_mode1_bkr" in decomp
                else np.asarray([], dtype=float)
            )
            s2 = (
                np.asarray(decomp["scores_mode2_bkr"], dtype=float)
                if "scores_mode2_bkr" in decomp
                else np.asarray([], dtype=float)
            )
            mu_bkr = (
                np.asarray(baseline["mu_bkr"], dtype=float)
                if isinstance(baseline, h5py.Group) and "mu_bkr" in baseline
                else None
            )
            valid = (
                np.asarray(inputs["valid_column_mask_bkr"], dtype=bool)
                if isinstance(inputs, h5py.Group)
                and "valid_column_mask_bkr" in inputs
                else None
            )

        if v_block.ndim < 2 or u1.size == 0:
            return None
        n_t = v_block.shape[0]
        if u1.shape[0] != n_t:
            return None

        # Align spatial axes: velocity (t,b,k,r) vs packed (b,k,r).
        spatial = v_block.shape[1:]
        if mu_bkr is not None and mu_bkr.shape == spatial:
            mu = mu_bkr.reshape(1, -1)
        else:
            with np.errstate(all="ignore"):
                mu = np.nanmean(v_block, axis=0, keepdims=True).reshape(1, -1)

        v_cols = v_block.reshape(n_t, -1)
        x_cols = v_cols - mu
        if valid is not None and valid.shape == spatial:
            valid_flat = valid.reshape(-1).astype(bool)
        else:
            valid_flat = np.any(np.isfinite(v_cols), axis=0)
        if not np.any(valid_flat):
            return None

        v_cols = v_cols[:, valid_flat]
        x_cols = x_cols[:, valid_flat]
        mu_cols = mu[:, valid_flat]

        def _mode_recon(u: np.ndarray, scores: np.ndarray) -> np.ndarray:
            if u.size == 0 or scores.size == 0:
                return np.full_like(x_cols, np.nan)
            flat = scores.reshape(-1)
            if flat.shape[0] != valid_flat.shape[0]:
                return np.full_like(x_cols, np.nan)
            return np.outer(u, flat[valid_flat])

        return {
            "t": np.linspace(0, 1, n_t, endpoint=False),
            "v": v_cols,
            "mu": mu_cols,
            "x": x_cols,
            "a1u1": _mode_recon(u1, s1),
            "a2u2": _mode_recon(u2, s2),
        }

    @classmethod
    def _row_ylim_first_two(cls, summary: dict[str, np.ndarray]) -> tuple[float, float]:
        bounds: list[float] = []
        med, q25, q75 = cls._median_iqr_curve(summary["v"])
        bounds += [float(np.nanmin(q25)), float(np.nanmax(q75))]
        mu_vals = summary["mu"].reshape(-1)
        med_mu = float(np.nanmedian(mu_vals))
        sd_mu = float(np.nanstd(mu_vals, ddof=1)) if mu_vals.size > 1 else 0.0
        bounds += [med_mu - sd_mu, med_mu + sd_mu]
        lo, hi = min(bounds), max(bounds)
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        return lo - pad, hi + pad

    @classmethod
    def _row_ylim_last_three(cls, summary: dict[str, np.ndarray]) -> tuple[float, float]:
        extents: list[float] = []
        for key in ("x", "a1u1", "a2u2"):
            _med, q25, q75 = cls._median_iqr_curve(summary[key])
            extents.append(
                max(abs(float(np.nanmin(q25))), abs(float(np.nanmax(q75))))
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
    ) -> Path:
        """Fig. 3: arterial v, mu, w, a1u1, a2u2."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        rows: list[dict[str, np.ndarray] | None] = []
        for vessel in FIGURE_VESSELS:
            rows.append(
                cls._waveform_summary_for_vessel(
                    h5_path, vessel, signal=signal
                )
            )

        panel_defs = [
            ("v", r"Beat-aligned velocity $v$"),
            ("mu", r"Baseline level" "\n" r"$\mu$"),
            ("x", r"Baseline-removed" "\n" r"$v-\mu$"),
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
        # Space between axis label and tick labels (the scale).
        label_pad = 8.0
        for row_idx, summary in enumerate(rows):
            if summary is None:
                for col_idx in range(n_cols):
                    axes[row_idx, col_idx].set_visible(False)
                continue
            t = summary["t"]
            t0, t1 = float(t[0]), float(t[-1])
            x_pad = cls.FIG3_X_PAD_FRAC * (t1 - t0 if t1 > t0 else 1.0)
            # Match Fig. 2 / H5 unit (mm/s); scale factor is 1.0 today.
            scale = float(cls.FIG2_VELOCITY_TO_MM_S)
            ylim_12 = tuple(scale * y for y in cls._row_ylim_first_two(summary))
            ylim_345 = tuple(scale * y for y in cls._row_ylim_last_three(summary))
            for col_idx, (key, title) in enumerate(panel_defs):
                ax = axes[row_idx, col_idx]
                if key == "mu":
                    mu_vals = summary[key].reshape(-1) * scale
                    med = float(np.nanmedian(mu_vals))
                    sd = float(np.nanstd(mu_vals, ddof=1)) if mu_vals.size > 1 else 0.0
                    # Draw over ``t`` (not full-width axhline) so left/right
                    # padding matches the waveform panels.
                    ax.plot(t, np.full_like(t, med), color="black", linewidth=1.8)
                    ax.fill_between(
                        t, med - sd, med + sd, color="black", alpha=0.12, linewidth=0
                    )
                else:
                    med, q25, q75 = cls._median_iqr_curve(summary[key] * scale)
                    ax.plot(t, med, color="black", linewidth=1.8)
                    ax.fill_between(t, q25, q75, color="black", alpha=0.12, linewidth=0)
                # Zero guide over data only — full-width axhline hides x-padding.
                ax.plot(
                    t,
                    np.zeros_like(t),
                    color="black",
                    linewidth=1.0,
                    linestyle=":",
                )
                y_lo, y_hi = ylim_345 if key in zero_cols else ylim_12
                ax.set_ylim(y_lo, y_hi)
                # Same left/right data pad as Fig. 2 on every panel.
                ax.set_xlim(t0 - x_pad, t1 + x_pad)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11, pad=6)
                if col_idx == 0:
                    # Same quantity/unit as Fig. 2 and H5 ``unit`` attrs.
                    ax.set_ylabel("Velocity (mm/s)", fontsize=10, labelpad=label_pad)
                if row_idx == n_rows - 1 and col_idx == n_cols // 2:
                    ax.set_xlabel(
                        "Fraction of cardiac cycle",
                        fontsize=10,
                        labelpad=label_pad,
                    )
                cls._style_axes(ax, tick_size=9, label_size=10)
                ax.set_box_aspect(1)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.08, hspace=0.05)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_path

    @classmethod
    def plot_energy_spectrum(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
    ) -> Path:
        """Fig. 4: arterial singular values $\\lambda_m$ vs mode index $m$."""
        out_path = Path(out_path)
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
        for row_idx, vessel in enumerate(vessels):
            ax = axes[row_idx, 0]
            data = vessel_bundle.get(vessel)
            spectrum = np.asarray([], dtype=float)
            if data is not None:
                spectrum = np.asarray(
                    data.get("singular_values", []), dtype=float
                )
                if spectrum.size == 0:
                    spectrum = np.asarray(
                        data.get("energy_fraction", []), dtype=float
                    )
            n_modes = int(min(n_keep, spectrum.size))
            modes = np.arange(1, n_keep + 1)
            if n_modes > 0:
                ax.plot(
                    np.arange(1, n_modes + 1),
                    spectrum[:n_modes],
                    color="black",
                    linestyle="-",
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                    linewidth=1.5,
                )
            ax.set_xticks(modes)
            ax.set_xticklabels([str(m) for m in modes])
            ax.set_xlim(0.5, n_keep + 0.5)
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel(r"$\lambda_m$", fontsize=cls.FIG2_LABEL_SIZE)
            ax.set_xlabel(r"$m$", fontsize=cls.FIG2_LABEL_SIZE)
            ax.set_box_aspect(0.5)
            cls._style_axes(
                ax, tick_size=cls.FIG2_TICK_SIZE, label_size=cls.FIG2_LABEL_SIZE
            )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

