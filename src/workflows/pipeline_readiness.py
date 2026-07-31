from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from input_output import prepare_run_input, prepare_run_inputs
from input_output.eyeflow_schema import has_path

from ._stem_inputs import resolve_selected_holo_contexts
from .request_state import WorkflowInputSelection


@dataclass(frozen=True)
class ResolvedPipelineInputs:
    """Reusable HDF5 inputs resolved from a workflow selection."""

    files: tuple[tuple[str, Path], ...] = ()
    missing_sources: tuple[str, ...] = ()
    input_error: str | None = None


def resolve_pipeline_inputs(
    selection: WorkflowInputSelection,
) -> ResolvedPipelineInputs:
    """Resolve the HDF5 files a pipeline would receive for this selection."""
    if selection.convention == "holo":
        if not selection.holo_paths:
            return ResolvedPipelineInputs()
        try:
            resolved = resolve_selected_holo_contexts(selection.holo_paths)
        except (OSError, RuntimeError, ValueError) as exc:
            return ResolvedPipelineInputs(input_error=str(exc))
        return ResolvedPipelineInputs(
            files=tuple(
                (context.holo_path.stem, context.h5_path)
                for context in resolved.contexts
            ),
            missing_sources=tuple(resolved.skipped_stems),
        )

    if not selection.data_value and not selection.legacy_input_paths:
        return ResolvedPipelineInputs()
    try:
        plan = (
            prepare_run_inputs(selection.legacy_input_paths)
            if selection.legacy_input_paths
            else prepare_run_input(Path(selection.data_value).expanduser())
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return ResolvedPipelineInputs(input_error=str(exc))
    if plan.is_zip:
        return ResolvedPipelineInputs()
    return ResolvedPipelineInputs(
        files=tuple((path.stem, path) for path in plan.h5_paths),
    )


def pipeline_input_status(
    pipeline: Any,
    selection: WorkflowInputSelection,
) -> str | None:
    """Return an input-specific status, or ``None`` when no override applies."""
    pipeline_cls = getattr(pipeline, "pipeline_cls", None)
    required_paths = tuple(
        getattr(pipeline_cls, "required_h5_paths", ()) if pipeline_cls else ()
    )
    if not required_paths:
        return None

    inputs = resolve_pipeline_inputs(selection)
    source_label = getattr(pipeline_cls, "h5_source_label", "H5")
    if inputs.missing_sources:
        label = "file" if len(inputs.missing_sources) == 1 else "files"
        return (
            f"Missing {source_label} {label}: "
            + ", ".join(inputs.missing_sources)
        )
    if inputs.input_error:
        return "Unreadable input"
    if not inputs.files:
        return None

    missing_by_input: dict[str, tuple[str, ...]] = {}
    unreadable: list[str] = []
    for input_label, h5_path in inputs.files:
        try:
            with h5py.File(h5_path, "r") as h5file:
                missing = tuple(
                    path for path in required_paths if not has_path(h5file, path)
                )
        except (OSError, ValueError):
            unreadable.append(input_label)
            continue
        if missing:
            missing_by_input[input_label] = missing

    if unreadable:
        return f"Unreadable H5: {', '.join(unreadable)}"
    if not missing_by_input:
        return None
    if len(inputs.files) == 1:
        count = len(next(iter(missing_by_input.values())))
        label = "key" if count == 1 else "keys"
        return f"Missing {count} required {label}"
    return "Missing required keys: " + ", ".join(missing_by_input)
