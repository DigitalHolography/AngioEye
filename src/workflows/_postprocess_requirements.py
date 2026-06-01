from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py

from input_output import find_pipeline_group


@dataclass(frozen=True)
class CompatiblePostprocessFiles:
    files: tuple[Path, ...]
    skipped: tuple[Path, ...] = ()


def has_pipeline_output(h5_path: Path, pipeline_name: str) -> bool:
    try:
        with h5py.File(h5_path, "r") as h5file:
            return find_pipeline_group(h5file, pipeline_name) is not None
    except OSError:
        return False


def has_pipeline_outputs(h5_path: Path, pipeline_names: Sequence[str]) -> bool:
    return all(has_pipeline_output(h5_path, name) for name in pipeline_names)


def compatible_postprocess_files(
    *,
    processed_outputs: Sequence[Path],
    input_h5_paths: Sequence[Path],
    required_pipelines: Sequence[str],
) -> CompatiblePostprocessFiles:
    if not required_pipelines:
        return CompatiblePostprocessFiles(tuple(processed_outputs or input_h5_paths))

    compatible: list[Path] = []
    skipped: list[Path] = []
    for output_path, input_path in _paired_paths(processed_outputs, input_h5_paths):
        if output_path is not None and has_pipeline_outputs(
            output_path, required_pipelines
        ):
            compatible.append(output_path)
            continue
        if input_path is not None and has_pipeline_outputs(
            input_path, required_pipelines
        ):
            compatible.append(input_path)
            continue
        skipped.append(input_path or output_path)

    return CompatiblePostprocessFiles(
        files=tuple(compatible),
        skipped=tuple(path for path in skipped if path is not None),
    )


def missing_required_pipeline_errors(
    *,
    postprocesses: Sequence[object],
    selected_pipeline_names: Sequence[str],
    reusable_h5_paths: Sequence[Path] = (),
    defer_when_no_reusable_paths: bool = False,
) -> list[str]:
    selected = set(selected_pipeline_names)
    errors: list[str] = []

    for postprocess in postprocesses:
        required = tuple(getattr(postprocess, "required_pipelines", ()))
        if not required or set(required).issubset(selected):
            continue
        if not reusable_h5_paths and defer_when_no_reusable_paths:
            continue
        if reusable_h5_paths and any(
            has_pipeline_outputs(path, required) for path in reusable_h5_paths
        ):
            continue
        errors.append(
            f"{postprocess.name} requires pipeline data: {', '.join(required)}"
        )

    return errors


def _paired_paths(
    processed_outputs: Sequence[Path],
    input_h5_paths: Sequence[Path],
) -> list[tuple[Path | None, Path | None]]:
    pair_count = max(len(processed_outputs), len(input_h5_paths))
    return [
        (
            processed_outputs[index] if index < len(processed_outputs) else None,
            input_h5_paths[index] if index < len(input_h5_paths) else None,
        )
        for index in range(pair_count)
    ]
