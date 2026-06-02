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


def has_pipeline_output_option(
    h5_path: Path,
    pipeline_options: Sequence[Sequence[str]],
) -> bool:
    return any(has_pipeline_outputs(h5_path, option) for option in pipeline_options)


def compatible_postprocess_files(
    *,
    processed_outputs: Sequence[Path],
    input_h5_paths: Sequence[Path],
    required_pipelines: Sequence[str],
    required_pipeline_options: Sequence[Sequence[str]] = (),
    selected_pipeline_names: Sequence[str] = (),
) -> CompatiblePostprocessFiles:
    pipeline_options = _pipeline_options(
        required_pipelines=required_pipelines,
        required_pipeline_options=required_pipeline_options,
    )
    if not pipeline_options:
        return CompatiblePostprocessFiles(tuple(processed_outputs or input_h5_paths))

    selected = set(selected_pipeline_names)
    if processed_outputs and any(
        set(option).issubset(selected) for option in pipeline_options
    ):
        return CompatiblePostprocessFiles(tuple(processed_outputs))

    compatible: list[Path] = []
    skipped: list[Path] = []
    for output_path, input_path in _paired_paths(processed_outputs, input_h5_paths):
        if output_path is not None and has_pipeline_output_option(
            output_path, pipeline_options
        ):
            compatible.append(output_path)
            continue
        if input_path is not None and has_pipeline_output_option(
            input_path, pipeline_options
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
        pipeline_options = _pipeline_options_for(postprocess)
        if not pipeline_options or any(
            set(option).issubset(selected) for option in pipeline_options
        ):
            continue
        if not reusable_h5_paths and defer_when_no_reusable_paths:
            continue
        if reusable_h5_paths and any(
            has_pipeline_output_option(path, pipeline_options)
            for path in reusable_h5_paths
        ):
            continue
        errors.append(
            f"{postprocess.name} requires pipeline data: "
            f"{_format_pipeline_options(pipeline_options)}"
        )

    return errors


def _pipeline_options(
    *,
    required_pipelines: Sequence[str],
    required_pipeline_options: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    if required_pipeline_options:
        return tuple(tuple(option) for option in required_pipeline_options if option)
    required = tuple(required_pipelines)
    return (required,) if required else ()


def _pipeline_options_for(obj: object) -> tuple[tuple[str, ...], ...]:
    options = getattr(obj, "required_pipeline_options", None)
    if options:
        return tuple(tuple(option) for option in options if option)
    required = tuple(getattr(obj, "required_pipelines", ()))
    return (required,) if required else ()


def _format_pipeline_options(
    pipeline_options: Sequence[Sequence[str]],
) -> str:
    return " or ".join(" + ".join(option) for option in pipeline_options)


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
