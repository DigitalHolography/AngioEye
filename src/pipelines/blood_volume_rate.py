from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from input_output.archive_io import iter_extracted_h5_members, list_h5_members

from .core.base import (
    ArchiveProcessPipeline,
    ArchiveProcessResult,
    registerPipeline,
)

LUMEN_DIAMETER_PATH = "/Segmentation/Artery/LumenDiameter/value"
PIXEL_PITCH_PATH = "/Segmentation/PixelPitch_m/value"
BLOOD_VOLUME_RATE_PATH = "/Processing/BloodVolumeRate/Artery/totalMaskedEdges/value"
OUTPUT_FOLDER_NAME = "lumen_diameter_distributions"
BLOOD_VOLUME_RATE_OUTPUT_FOLDER = "blood_volume_rate"
PHASE_SAMPLE_COUNT = 128

_CONTROL_GROUP_NAMES = {
    "baseline",
    "baseline_cohort",
    "baseline_group",
    "bl",
    "control",
    "control_cohort",
    "control_group",
    "controls",
    "ctl",
    "ctrl",
    "healthy",
    "healthy_control",
    "healthy_controls",
    "reference",
}

_LEFT_EYE_NAMES = {
    "eye_left",
    "gauche",
    "l",
    "left",
    "left_eye",
    "lefteye",
    "oeil_gauche",
    "os",
}
_RIGHT_EYE_NAMES = {
    "droit",
    "eye_right",
    "od",
    "oeil_droit",
    "r",
    "right",
    "right_eye",
    "righteye",
}


@dataclass(frozen=True)
class LumenDiameterCollection:
    values_by_cohort: dict[str, np.ndarray]
    input_file_count: int
    loaded_file_count: int
    skipped_files: tuple[str, ...]


@dataclass(frozen=True)
class BloodVolumeRateCollection:
    waveforms_by_group: dict[str, np.ndarray]
    skipped_files: tuple[str, ...]


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_cohort_name(group_name: object) -> str:
    """Return a stable cohort name, canonicalizing control aliases."""
    normalized = _normalized_name(group_name)
    if not normalized:
        return "cohort"
    if normalized in _CONTROL_GROUP_NAMES:
        return "baseline"
    return normalized


def _normalize_eye_name(value: object) -> str | None:
    normalized = _normalized_name(value)
    if normalized in _LEFT_EYE_NAMES:
        return "left_eye"
    if normalized in _RIGHT_EYE_NAMES:
        return "right_eye"
    return None


def _member_cohort_name(relative_path: Path) -> str:
    directory_parts = tuple(
        part for part in relative_path.parts[:-1] if part.lower() != "h5"
    )
    if not directory_parts:
        return "all"

    for index, directory_name in enumerate(directory_parts):
        eye_name = _normalize_eye_name(directory_name)
        if eye_name is not None and index > 0:
            cohort_name = normalize_cohort_name(directory_parts[index - 1])
            return f"{cohort_name}_{eye_name}"

    return normalize_cohort_name(directory_parts[0])


def read_lumen_diameters(h5_path: str | Path) -> np.ndarray:
    """Read finite lumen diameters and convert pixels to micrometres."""
    with h5py.File(h5_path, "r") as h5file:
        if LUMEN_DIAMETER_PATH not in h5file:
            raise KeyError(f"Missing EyeFlow dataset: {LUMEN_DIAMETER_PATH}")
        if PIXEL_PITCH_PATH not in h5file:
            raise KeyError(f"Missing EyeFlow dataset: {PIXEL_PITCH_PATH}")
        values = np.asarray(h5file[LUMEN_DIAMETER_PATH][...], dtype=float).ravel()
        pixel_pitch = np.asarray(h5file[PIXEL_PITCH_PATH][...], dtype=float).squeeze()
        if pixel_pitch.size != 1:
            raise ValueError(f"Expected a scalar at {PIXEL_PITCH_PATH}")
        pixel_pitch_m = float(pixel_pitch)
    return values[np.isfinite(values)] * pixel_pitch_m * 1e6


def read_blood_volume_rate_waveforms(
    h5_path: str | Path,
    *,
    phase_sample_count: int = PHASE_SAMPLE_COUNT,
) -> np.ndarray:
    """Return beats as rows, resampled over a normalized cardiac phase."""
    with h5py.File(h5_path, "r") as h5file:
        if BLOOD_VOLUME_RATE_PATH not in h5file:
            raise KeyError(f"Missing EyeFlow dataset: {BLOOD_VOLUME_RATE_PATH}")
        values = np.asarray(h5file[BLOOD_VOLUME_RATE_PATH][...], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array at {BLOOD_VOLUME_RATE_PATH}")

    source_phase = np.linspace(0.0, 1.0, values.shape[0])
    target_phase = np.linspace(0.0, 1.0, phase_sample_count)
    waveforms: list[np.ndarray] = []
    for beat in values.T:
        finite = np.isfinite(beat)
        if np.count_nonzero(finite) < 2:
            continue
        waveforms.append(np.interp(target_phase, source_phase[finite], beat[finite]))
    if not waveforms:
        return np.empty((0, phase_sample_count), dtype=float)
    return np.vstack(waveforms)


def collect_lumen_diameters(zip_path: str | Path) -> LumenDiameterCollection:
    """Collect lumen diameters by cohort and eye from an EyeFlow ZIP.

    Files under ``cohort/left_eye`` and ``cohort/right_eye`` are kept as two
    distinct distributions. Files directly under a cohort remain supported,
    as does the repository's optional leading ``h5`` directory convention.
    """
    values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    skipped_files: list[str] = []
    input_file_count = 0
    loaded_file_count = 0

    for member in list_h5_members(zip_path):
        input_file_count += 1
        cohort_name = _member_cohort_name(member.relative_path)
        for extracted in iter_extracted_h5_members(zip_path, [member]):
            try:
                file_values = read_lumen_diameters(extracted.path)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                skipped_files.append(f"{member.name}: {exc}")
                continue

            if file_values.size == 0:
                skipped_files.append(f"{member.name}: no finite lumen-diameter values")
                continue

            values[cohort_name].append(file_values)
            loaded_file_count += 1

    values_by_cohort = {
        cohort_name: np.concatenate(cohort_values)
        for cohort_name, cohort_values in values.items()
        if cohort_values
    }
    return LumenDiameterCollection(
        values_by_cohort=values_by_cohort,
        input_file_count=input_file_count,
        loaded_file_count=loaded_file_count,
        skipped_files=tuple(skipped_files),
    )


def collect_blood_volume_rate(zip_path: str | Path) -> BloodVolumeRateCollection:
    waveforms: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    skipped_files: list[str] = []
    for member in list_h5_members(zip_path):
        group_name = _member_cohort_name(member.relative_path)
        for extracted in iter_extracted_h5_members(zip_path, [member]):
            try:
                file_waveforms = read_blood_volume_rate_waveforms(extracted.path)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                skipped_files.append(f"{member.name}: {exc}")
                continue
            if file_waveforms.size == 0:
                skipped_files.append(f"{member.name}: no finite BVR waveforms")
                continue
            waveforms[group_name].append(file_waveforms)

    return BloodVolumeRateCollection(
        waveforms_by_group={
            group_name: np.vstack(group_waveforms)
            for group_name, group_waveforms in waveforms.items()
            if group_waveforms
        },
        skipped_files=tuple(skipped_files),
    )


def build_lumen_diameter_figure(values: np.ndarray) -> tuple[Figure, float, float]:
    """Build the requested grey histogram and black dashed Gaussian fit."""
    finite_values = np.asarray(values, dtype=float).ravel()
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot plot an empty lumen-diameter distribution.")

    median = float(np.median(finite_values))
    standard_deviation = float(np.std(finite_values))
    mean = float(np.mean(finite_values))

    figure = Figure(figsize=(6.4, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bin_width = 5.0
    bin_start = math.floor(float(np.min(finite_values)) / bin_width) * bin_width
    bin_stop = math.ceil(float(np.max(finite_values)) / bin_width) * bin_width
    if bin_stop <= bin_start:
        bin_stop = bin_start + bin_width
    bins = np.arange(bin_start, bin_stop + bin_width, bin_width)
    axis.hist(
        finite_values,
        bins=bins,
        density=True,
        color="grey",
        edgecolor="white",
        linewidth=0.6,
    )

    if standard_deviation > 0.0:
        x_min, x_max = axis.get_xlim()
        x_values = np.linspace(x_min, x_max, 400)
        gaussian = np.exp(-0.5 * ((x_values - mean) / standard_deviation) ** 2) / (
            standard_deviation * math.sqrt(2.0 * math.pi)
        )
        axis.plot(
            x_values,
            gaussian,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
    else:
        # A constant sample has a degenerate Gaussian. Its limiting position is
        # still shown using the requested black dashed line.
        axis.axvline(
            mean,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

    axis.set_xlabel("Lumen diameter (µm)", fontsize="medium")
    axis.set_ylabel("Density", fontsize="medium")
    axis.xaxis.set_major_locator(MultipleLocator(10.0))
    axis.tick_params(axis="both", labelsize="medium")
    axis.text(
        0.98,
        0.98,
        f"Median: {median:.4g}\nStd: {standard_deviation:.4g}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize="medium",
    )
    # Intentionally no title: cohort identity is carried by the PNG filename.
    return figure, median, standard_deviation


def write_lumen_diameter_histogram(
    values: np.ndarray,
    output_path: str | Path,
    *,
    cohort_name: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, median, standard_deviation = build_lumen_diameter_figure(values)
    try:
        figure.savefig(
            output_path,
            dpi=150,
            format="png",
            metadata={
                "Description": (
                    f"Lumen diameter distribution for {cohort_name}; "
                    f"Median: {median:.6g}; Std: {standard_deviation:.6g}"
                )
            },
        )
    finally:
        figure.clear()
    return output_path


def build_blood_volume_rate_figure(
    baseline_waveforms: np.ndarray,
    indentation_waveforms: np.ndarray,
) -> Figure:
    phase = np.linspace(0.0, 1.0, baseline_waveforms.shape[1])
    baseline_median = np.nanmedian(baseline_waveforms, axis=0)
    baseline_std = np.nanstd(baseline_waveforms, axis=0)
    indentation_median = np.nanmedian(indentation_waveforms, axis=0)
    indentation_std = np.nanstd(indentation_waveforms, axis=0)

    figure = Figure(figsize=(6.4, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.fill_between(
        phase,
        baseline_median - baseline_std,
        baseline_median + baseline_std,
        color="black",
        alpha=0.12,
        linewidth=0,
    )
    axis.fill_between(
        phase,
        indentation_median - indentation_std,
        indentation_median + indentation_std,
        facecolor="none",
        edgecolor="grey",
        hatch="///",
        linewidth=0,
    )
    axis.plot(phase, baseline_median, color="black", linewidth=1.8, label="Baseline")
    axis.plot(
        phase,
        indentation_median,
        color="grey",
        linestyle="--",
        linewidth=1.8,
        label="Indentation",
    )
    axis.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 1.0])
    axis.set_xlabel("Cardiac phase, t/T", fontsize="medium")
    axis.set_ylabel("Q(t) (mm3/s)", fontsize="medium")
    axis.tick_params(axis="both", labelsize="medium")
    axis.legend(frameon=False, fontsize="medium")
    return figure


def generate_blood_volume_rate_figures(
    zip_path: str | Path,
    output_root: str | Path,
) -> tuple[list[Path], BloodVolumeRateCollection]:
    collection = collect_blood_volume_rate(zip_path)
    output_dir = Path(output_root) / BLOOD_VOLUME_RATE_OUTPUT_FOLDER
    generated_paths: list[Path] = []
    for eye_name in ("left_eye", "right_eye"):
        baseline_key = f"baseline_{eye_name}"
        indentation_key = f"indentation_{eye_name}"
        if not {
            baseline_key,
            indentation_key,
        }.issubset(collection.waveforms_by_group):
            continue
        figure = build_blood_volume_rate_figure(
            collection.waveforms_by_group[baseline_key],
            collection.waveforms_by_group[indentation_key],
        )
        output_path = output_dir / f"{eye_name}_blood_volume_rate.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, dpi=150, format="png")
        finally:
            figure.clear()
        generated_paths.append(output_path)

    if not generated_paths:
        raise ValueError(
            "No eye has both baseline and indentation BVR data at "
            f"{BLOOD_VOLUME_RATE_PATH}."
        )
    return generated_paths, collection


def generate_lumen_diameter_distributions(
    zip_path: str | Path,
    output_root: str | Path,
) -> tuple[list[Path], LumenDiameterCollection]:
    """Generate one pooled lumen-diameter histogram PNG per cohort."""
    collection = collect_lumen_diameters(zip_path)
    if not collection.values_by_cohort:
        skipped_detail = (
            f" Skipped: {'; '.join(collection.skipped_files)}"
            if collection.skipped_files
            else ""
        )
        raise ValueError(
            f"No finite values were found at {LUMEN_DIAMETER_PATH}.{skipped_detail}"
        )

    output_dir = Path(output_root) / OUTPUT_FOLDER_NAME
    generated_paths: list[Path] = []
    cohort_names = sorted(
        collection.values_by_cohort,
        key=lambda name: (name != "baseline", name),
    )
    for cohort_name in cohort_names:
        output_path = output_dir / f"{cohort_name}_lumen_diameter_distribution.png"
        generated_paths.append(
            write_lumen_diameter_histogram(
                collection.values_by_cohort[cohort_name],
                output_path,
                cohort_name=cohort_name,
            )
        )
    return generated_paths, collection


@registerPipeline(
    name="blood_volume_rate",
    description=(
        "Read a cohort-organized EyeFlow ZIP and generate one lumen-diameter "
        "distribution histogram per cohort and eye. Left-eye and right-eye "
        "subfolders remain separate."
    ),
    required_deps=["h5py>=3.9", "matplotlib>=3.8", "numpy>=1.24"],
)
class BloodVolumeRatePipeline(ArchiveProcessPipeline):
    def run_archive(
        self,
        zip_path: Path | str,
        output_dir: Path | str,
    ) -> ArchiveProcessResult:
        input_path = Path(zip_path).expanduser().resolve()
        if not input_path.is_file() or input_path.suffix.lower() != ".zip":
            raise ValueError("blood_volume_rate requires a ZIP archive as its input.")

        generated_paths, collection = generate_lumen_diameter_distributions(
            input_path,
            output_dir,
        )
        bvr_paths, bvr_collection = generate_blood_volume_rate_figures(
            input_path,
            output_dir,
        )
        skipped_count = len(collection.skipped_files) + len(
            bvr_collection.skipped_files
        )
        skipped_suffix = (
            f" Skipped {skipped_count} incompatible HDF5 file(s)."
            if skipped_count
            else ""
        )
        return ArchiveProcessResult(
            summary=(
                f"Generated {len(generated_paths)} lumen-diameter cohort "
                f"histogram(s) and {len(bvr_paths)} BVR comparison figure(s)."
                f"{skipped_suffix}"
            ),
            generated_paths=[str(path) for path in (*generated_paths, *bvr_paths)],
            metadata={
                "cohorts": list(collection.values_by_cohort),
                "bvr_groups": list(bvr_collection.waveforms_by_group),
                "input_file_count": collection.input_file_count,
                "loaded_file_count": collection.loaded_file_count,
                "skipped_files": list(collection.skipped_files),
            },
        )


__all__ = [
    "BLOOD_VOLUME_RATE_OUTPUT_FOLDER",
    "BLOOD_VOLUME_RATE_PATH",
    "BloodVolumeRatePipeline",
    "BloodVolumeRateCollection",
    "LUMEN_DIAMETER_PATH",
    "LumenDiameterCollection",
    "OUTPUT_FOLDER_NAME",
    "build_lumen_diameter_figure",
    "build_blood_volume_rate_figure",
    "collect_blood_volume_rate",
    "collect_lumen_diameters",
    "generate_lumen_diameter_distributions",
    "generate_blood_volume_rate_figures",
    "normalize_cohort_name",
    "read_lumen_diameters",
    "read_blood_volume_rate_waveforms",
    "write_lumen_diameter_histogram",
]
