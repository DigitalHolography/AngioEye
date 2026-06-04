from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ._postprocess_requirements import missing_required_pipeline_errors
from .dispatch import OutputFilenameResolver, WorkflowInputError, WorkflowRunRequest
from .inputs import prepare_run_input, prepare_run_inputs
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
    trim_source: bool


@dataclass(frozen=True)
class WorkflowRequestState:
    input_selection: WorkflowInputSelection
    work_selection: WorkflowWorkSelection
    output_options: WorkflowOutputOptions


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
        or (
            input_selection.convention == "holo"
            and not work_selection.pipelines
        ),
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
        base_output_dir=(
            cwd()
            if input_selection.convention == "holo"
            else resolve_base_output_dir(output_options.base_output_value, cwd=cwd)
        ),
        pipelines=work_selection.pipelines,
        postprocesses=work_selection.postprocesses,
        selected_pipeline_names=work_selection.pipeline_names,
        zip_outputs=output_options.zip_outputs,
        zip_name=output_options.zip_name,
        trim_source=output_options.trim_source,
        zip_output_dir=zip_output_dir,
        output_filename_for_run=output_filename_for_run,
    )


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
