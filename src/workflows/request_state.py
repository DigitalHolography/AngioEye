from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from input_output import InputPlan, prepare_run_input, prepare_run_inputs

from ._postprocess_requirements import (
    missing_required_option_errors,
    missing_required_pipeline_errors,
)
from .dispatch import OutputFilenameResolver, WorkflowInputError, WorkflowRunRequest
from .runs import ZipOutputDir

InputConvention = Literal["legacy", "holo"]


@dataclass(frozen=True)
class WorkflowInputSelection:
    convention: InputConvention
    data_value: str = ""
    legacy_input_paths: tuple[Path, ...] = ()
    holo_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class WorkflowWorkSelection:
    pipeline_names: tuple[str, ...]
    pipelines: tuple[Any, ...]
    postprocesses: tuple[Any, ...]


@dataclass(frozen=True)
class WorkflowOutputOptions:
    base_output_value: str
    zip_outputs: bool
    zip_name: str
    persist_source: bool = False


@dataclass(frozen=True)
class WorkflowRequestState:
    input_selection: WorkflowInputSelection
    work_selection: WorkflowWorkSelection
    output_options: WorkflowOutputOptions


def resolve_work_selection(
    pipeline_names: Sequence[str],
    pipeline_registry: Mapping[str, Any],
    postprocess_names: Sequence[str],
    postprocess_registry: Mapping[str, Any],
) -> WorkflowWorkSelection:
    """Resolve frontend names into descriptors consumed by the dispatcher."""
    selected_pipelines = [
        pipeline_registry[name]
        for name in pipeline_names
        if name in pipeline_registry
    ]
    missing_pipelines = [
        name for name in pipeline_names if name not in pipeline_registry
    ]
    if missing_pipelines:
        available = ", ".join(pipeline_registry)
        raise WorkflowInputError(
            "Pipeline missing",
            f"Pipeline(s) not registered: {', '.join(missing_pipelines)}"
            + (f". Available: {available}" if available else ""),
        )

    selected_postprocesses = [
        postprocess_registry[name]
        for name in postprocess_names
        if name in postprocess_registry
    ]
    missing_postprocesses = [
        name for name in postprocess_names if name not in postprocess_registry
    ]
    if missing_postprocesses:
        available = ", ".join(postprocess_registry)
        raise WorkflowInputError(
            "Postprocess missing",
            f"Postprocess step(s) not registered: {', '.join(missing_postprocesses)}"
            + (f". Available: {available}" if available else ""),
        )
    if not selected_pipelines and not selected_postprocesses:
        raise WorkflowInputError(
            "No work selected",
            "Select at least one pipeline or postprocess step.",
        )
    return WorkflowWorkSelection(
        pipeline_names=tuple(pipeline_names),
        pipelines=tuple(selected_pipelines),
        postprocesses=tuple(selected_postprocesses),
    )


def build_workflow_request(
    state: WorkflowRequestState,
    *,
    zip_output_dir: ZipOutputDir,
    output_filename_for_run: OutputFilenameResolver,
    cwd: Callable[[], Path] = Path.cwd,
) -> WorkflowRunRequest:
    input_selection = state.input_selection
    work_selection = state.work_selection
    output_options = state.output_options

    input_plan = None
    request_mode = "holo"
    if input_selection.convention != "holo":
        try:
            if input_selection.legacy_input_paths:
                input_plan = prepare_run_inputs(input_selection.legacy_input_paths)
            else:
                input_plan = prepare_run_input(
                    Path(input_selection.data_value).expanduser()
                )
        except Exception as exc:  # noqa: BLE001
            raise WorkflowInputError(
                "Invalid input",
                f"Cannot prepare input: {exc}",
            ) from exc
        request_mode = input_plan.kind

    reusable_h5_paths = (
        input_plan.h5_paths
        if input_plan is not None and not input_plan.is_zip
        else ()
    )
    requirement_errors = missing_required_pipeline_errors(
        postprocesses=work_selection.postprocesses,
        selected_pipeline_names=work_selection.pipeline_names,
        reusable_h5_paths=reusable_h5_paths,
        defer_when_no_reusable_paths=bool(input_plan and input_plan.is_zip)
        or (input_selection.convention == "holo" and not work_selection.pipelines),
    )
    requirement_errors.extend(
        missing_required_option_errors(
            postprocesses=work_selection.postprocesses,
            persist_source=output_options.persist_source,
        )
    )
    if requirement_errors:
        raise WorkflowInputError(
            "Postprocess requirements",
            "\n".join(requirement_errors),
        )

    return WorkflowRunRequest(
        mode=request_mode,
        input_plan=input_plan,
        holo_paths=input_selection.holo_paths
        if input_selection.convention == "holo"
        else (),
        base_output_dir=resolve_workflow_output_dir(
            input_selection=input_selection,
            input_plan=input_plan,
            output_value=output_options.base_output_value,
            cwd=cwd,
        ),
        pipelines=work_selection.pipelines,
        postprocesses=work_selection.postprocesses,
        selected_pipeline_names=work_selection.pipeline_names,
        zip_outputs=output_options.zip_outputs,
        zip_name=output_options.zip_name,
        persist_source=output_options.persist_source,
        zip_output_dir=zip_output_dir,
        output_filename_for_run=output_filename_for_run,
    )


def resolve_workflow_output_dir(
    *,
    input_selection: WorkflowInputSelection,
    input_plan: InputPlan | None,
    output_value: str,
    cwd: Callable[[], Path] = Path.cwd,
) -> Path:
    """Resolve an explicit output or the GUI-equivalent input-based default."""
    if input_selection.convention == "holo":
        return cwd()
    if output_value.strip():
        return resolve_base_output_dir(output_value, cwd=cwd)
    if input_plan is None:
        return cwd()

    input_path = input_plan.input_path
    if input_plan.kind == "zip":
        default_dir = input_path.parent / f"{input_path.stem}_angioeye"
    elif input_plan.kind == "file" and input_path.is_file():
        default_dir = input_path.parent
    else:
        default_dir = input_path
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def resolve_base_output_dir(
    base_output_value: str,
    *,
    cwd: Callable[[], Path] = Path.cwd,
) -> Path:
    current_dir = cwd()
    base_output_dir = (
        Path(base_output_value).expanduser() if base_output_value else current_dir
    )
    if not base_output_dir.is_absolute():
        base_output_dir = current_dir / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir
