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

from input_output.archive_io import iter_extracted_h5_members, list_h5_members

from .core.base import (
    ArchiveProcessPipeline,
    ArchiveProcessResult,
    registerPipeline,
)

LUMEN_DIAMETER_PATH = "/Segmentation/Artery/LumenDiameter/value"
PIXEL_PITCH_PATH = "/Segmentation/PixelPitch_m/value"
OUTPUT_FOLDER_NAME = "lumen_diameter_distributions"

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
    axis.hist(
        finite_values,
        bins="auto",
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
        skipped_count = len(collection.skipped_files)
        skipped_suffix = (
            f" Skipped {skipped_count} incompatible HDF5 file(s)."
            if skipped_count
            else ""
        )
        return ArchiveProcessResult(
            summary=(
                f"Generated {len(generated_paths)} lumen-diameter cohort "
                f"histogram(s).{skipped_suffix}"
            ),
            generated_paths=[str(path) for path in generated_paths],
            metadata={
                "cohorts": list(collection.values_by_cohort),
                "input_file_count": collection.input_file_count,
                "loaded_file_count": collection.loaded_file_count,
                "skipped_files": list(collection.skipped_files),
            },
        )


__all__ = [
    "BloodVolumeRatePipeline",
    "LUMEN_DIAMETER_PATH",
    "LumenDiameterCollection",
    "OUTPUT_FOLDER_NAME",
    "build_lumen_diameter_figure",
    "collect_lumen_diameters",
    "generate_lumen_diameter_distributions",
    "normalize_cohort_name",
    "read_lumen_diameters",
    "write_lumen_diameter_histogram",
]
