from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from input_output import relative_hdf5_parent
from pipeline_engine import run_pipeline_file, run_postprocesses

from ._holo import HoloInputContext
from ._holo import find_ae_h5 as find_holo_ae_h5
from ._holo import output_dir as holo_output_dir
from ._holo import reset_output_dir as reset_holo_output_dir
from ._holo import resolve_context as resolve_holo_context
from ._postprocess_requirements import (
    compatible_postprocess_files,
)
from ._zip_batches import ZipBatchSettings
from .inputs import RunInputPlan
from .runs import (
    RunWorkflowResult,
    ZipOutputDir,
    ZipProgressCallback,
    log_throttled_zip_progress,
    run_filesystem_workflow,
    run_holo_workflow,
    run_zip_workflow,
)

WorkflowMode = Literal["holo", "file", "folder", "zip"]
OutputFilenameResolver = Callable[[Path, Sequence[Path]], str | None]


@dataclass(frozen=True)
class WorkflowRunRequest:
    mode: WorkflowMode
    pipelines: Sequence[Any]
    postprocesses: Sequence[Any]
    selected_pipeline_names: Sequence[str]
    base_output_dir: Path
    zip_outputs: bool
    zip_name: str
    trim_source: bool
    zip_output_dir: ZipOutputDir
    input_plan: RunInputPlan | None = None
    holo_paths: Sequence[Path] = ()
    zip_batch_settings: ZipBatchSettings = field(
        default_factory=ZipBatchSettings.from_app_settings
    )
    output_filename_for_run: OutputFilenameResolver = lambda _path, _inputs: None


@dataclass(frozen=True)
class WorkflowCallbacks:
    log: Callable[[str], None]
    start_primary_progress: Callable[[float, str], None]
    start_final_progress: Callable[[float, str], None]
    advance_progress: Callable[[float], None]
    set_progress_units: Callable[[float], None]
    set_status: Callable[[str], None]
    make_zip_progress_callback: Callable[[], ZipProgressCallback | None]
    idle_callback: Callable[[], None] | None = None


@dataclass(frozen=True)
class WorkflowDispatchResult:
    workflow_result: RunWorkflowResult | None
    skipped_holo_stems: tuple[str, ...] = ()


class WorkflowInputError(Exception):
    def __init__(self, title: str, message: str, *, status: str = "Run failed."):
        super().__init__(message)
        self.title = title
        self.message = message
        self.status = status


def dispatch_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    if request.mode == "holo":
        return _dispatch_holo_workflow(request, callbacks)
    if request.mode == "file":
        return _dispatch_file_workflow(request, callbacks)
    if request.mode == "folder":
        return _dispatch_folder_workflow(request, callbacks)
    if request.mode == "zip":
        return _dispatch_zip_workflow(request, callbacks)
    raise WorkflowInputError(
        "Invalid input mode",
        f"Unknown input mode: {request.mode}",
    )


def _dispatch_holo_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    if not request.holo_paths:
        raise WorkflowInputError(
            "Missing input",
            "Select one or more .holo files to process.",
            status="Ready.",
        )

    if not request.pipelines and request.postprocesses:
        return _dispatch_holo_postprocess_workflow(request, callbacks)

    contexts, skipped_holo_stems = _resolve_holo_contexts(request.holo_paths)
    if not contexts:
        return WorkflowDispatchResult(
            workflow_result=None,
            skipped_holo_stems=tuple(skipped_holo_stems),
        )

    for context in contexts:
        try:
            reset_holo_output_dir(context)
        except Exception as exc:  # noqa: BLE001
            raise WorkflowInputError("Invalid output", str(exc)) from exc

    callbacks.start_primary_progress(
        len(contexts) * len(request.pipelines),
        "Running pipelines...",
    )
    workflow_result = run_holo_workflow(
        contexts=contexts,
        pipelines=request.pipelines,
        postprocesses=request.postprocesses,
        selected_pipeline_names=request.selected_pipeline_names,
        run_pipeline_file=_pipeline_file_runner(request, callbacks, worker_safe=True),
        run_postprocesses=_postprocess_runner(request, callbacks),
        log=callbacks.log,
        advance_progress=callbacks.advance_progress,
        start_final_progress=callbacks.start_final_progress,
        settings=request.zip_batch_settings,
        idle_callback=callbacks.idle_callback,
    )
    return WorkflowDispatchResult(
        workflow_result=workflow_result,
        skipped_holo_stems=tuple(skipped_holo_stems),
    )


def _dispatch_holo_postprocess_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    ae_records: list[tuple[Path, Path]] = []
    failures: list[str] = []
    skipped_stems: list[str] = []

    for holo_path in request.holo_paths:
        holo_path = holo_path.expanduser()
        ae_h5 = find_holo_ae_h5(holo_path)
        if ae_h5 is None:
            skipped_stems.append(holo_path.stem)
            failures.append(
                f"{holo_path}: no existing AE HDF5 output found under "
                f"{holo_output_dir(holo_path)}"
            )
            continue
        ae_records.append((holo_path, ae_h5))

    callbacks.start_final_progress(
        len(ae_records) * len(request.postprocesses),
        "Running postprocess...",
    )

    for holo_path, ae_h5 in ae_records:
        run_postprocesses(
            request.postprocesses,
            ae_h5.parent,
            (ae_h5,),
            (ae_h5,),
            holo_path,
            request.selected_pipeline_names,
            failures,
            zip_outputs=False,
            log=callbacks.log,
            advance_progress=callbacks.advance_progress,
            idle_callback=callbacks.idle_callback,
            resolve_postprocess_files=_postprocess_file_resolver(
                request.selected_pipeline_names
            ),
        )

    summary = (
        f"Postprocessed {len(ae_records)} existing AE HDF5 file(s)."
        if ae_records
        else "No existing AE HDF5 files were available for postprocessing."
    )
    return WorkflowDispatchResult(
        workflow_result=RunWorkflowResult(
            output_dir=ae_records[0][1].parent if ae_records else Path("."),
            processed_outputs=[ae_h5 for _, ae_h5 in ae_records],
            processed_input_paths=[ae_h5 for _, ae_h5 in ae_records],
            failures=failures,
            summary_message=summary,
        ),
        skipped_holo_stems=tuple(skipped_stems),
    )


def _resolve_holo_contexts(
    holo_paths: Sequence[Path],
) -> tuple[list[HoloInputContext], list[str]]:
    contexts: list[HoloInputContext] = []
    skipped_holo_stems: list[str] = []
    for holo_path in holo_paths:
        try:
            contexts.append(resolve_holo_context(holo_path))
        except Exception:  # noqa: BLE001
            skipped_holo_stems.append(holo_path.stem)
    return contexts, skipped_holo_stems


def _dispatch_file_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    input_plan = _input_plan_for_mode(request, "file")
    return _dispatch_filesystem_workflow(request, callbacks, input_plan)


def _dispatch_folder_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    input_plan = _input_plan_for_mode(request, "folder")
    return _dispatch_filesystem_workflow(request, callbacks, input_plan)


def _dispatch_zip_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
) -> WorkflowDispatchResult:
    input_plan = _input_plan_for_mode(request, "zip")
    if input_plan.item_count == 0:
        message = (
            f"No .h5/.hdf5 files found inside ZIP archive: {input_plan.input_path}"
        )
        callbacks.log(
            f"Error: No .h5/.hdf5 files found inside {input_plan.input_path}"
        )
        raise WorkflowInputError("Invalid input", message)

    callbacks.start_primary_progress(
        input_plan.item_count * len(request.pipelines),
        "Running pipelines...",
    )
    callbacks.log(
        f"[ZIP] Found {input_plan.item_count} HDF5 file(s). "
        f"Extracting {request.zip_batch_settings.batch_size} at a time; "
        f"running pipelines with {request.zip_batch_settings.batch_size} "
        "worker(s)."
    )
    workflow_result = run_zip_workflow(
        zip_path=input_plan.input_path,
        members=input_plan.iter_zip_members(),
        member_count=input_plan.item_count,
        pipelines=request.pipelines,
        postprocesses=request.postprocesses,
        selected_pipeline_names=request.selected_pipeline_names,
        base_output_dir=request.base_output_dir,
        zip_outputs=request.zip_outputs,
        zip_name=request.zip_name,
        settings=request.zip_batch_settings,
        run_pipeline_file=_pipeline_file_runner(
            request,
            callbacks,
            worker_safe=True,
        ),
        run_postprocesses=_postprocess_runner(request, callbacks),
        zip_output_dir=request.zip_output_dir,
        log=callbacks.log,
        advance_progress=callbacks.advance_progress,
        start_final_progress=callbacks.start_final_progress,
        set_status=callbacks.set_status,
        make_zip_progress_callback=callbacks.make_zip_progress_callback,
        idle_callback=callbacks.idle_callback,
    )
    return WorkflowDispatchResult(workflow_result=workflow_result)


def _input_plan_for_mode(
    request: WorkflowRunRequest,
    expected_kind: Literal["file", "folder", "zip"],
) -> RunInputPlan:
    if request.input_plan is None:
        raise WorkflowInputError(
            "Missing input",
            "Select a folder, HDF5 file, or .zip archive to process.",
            status="Ready.",
        )
    if request.input_plan.kind != expected_kind:
        raise WorkflowInputError(
            "Invalid input mode",
            f"Workflow mode '{request.mode}' cannot run input kind "
            f"'{request.input_plan.kind}'.",
        )
    return request.input_plan


def _dispatch_filesystem_workflow(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
    input_plan,
) -> WorkflowDispatchResult:
    inputs = list(input_plan.h5_paths)
    callbacks.start_primary_progress(
        len(inputs) * len(request.pipelines),
        "Running pipelines...",
    )
    workflow_result = run_filesystem_workflow(
        inputs=inputs,
        data_root=input_plan.input_path,
        pipelines=request.pipelines,
        postprocesses=request.postprocesses,
        selected_pipeline_names=request.selected_pipeline_names,
        input_path=input_plan.input_path,
        base_output_dir=request.base_output_dir,
        zip_outputs=request.zip_outputs,
        zip_name=request.zip_name,
        output_filename=request.output_filename_for_run(
            input_plan.input_path,
            input_plan.h5_paths,
        ),
        settings=request.zip_batch_settings,
        run_pipeline_file=_pipeline_file_runner(
            request,
            callbacks,
            worker_safe=True,
        ),
        run_postprocesses=_postprocess_runner(request, callbacks),
        relative_parent=relative_hdf5_parent,
        zip_output_dir=request.zip_output_dir,
        log=callbacks.log,
        advance_progress=callbacks.advance_progress,
        start_final_progress=callbacks.start_final_progress,
        set_status=callbacks.set_status,
        make_zip_progress_callback=callbacks.make_zip_progress_callback,
        idle_callback=callbacks.idle_callback,
    )
    return WorkflowDispatchResult(workflow_result=workflow_result)


def _pipeline_file_runner(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
    *,
    worker_safe: bool,
):
    if worker_safe:
        return functools.partial(
            run_pipeline_file,
            trim_source=request.trim_source,
            log=None,
            advance_progress=None,
            write_idle_callback=None,
            output_path_allocator=None,
        )

    def _run_pipeline_file(
        h5_path: Path,
        pipelines: Sequence[Any],
        output_root: Path,
        output_relative_parent: Path = Path("."),
        output_filename: str | None = None,
        *,
        record_timing=None,
    ) -> Path:
        return run_pipeline_file(
            h5_path,
            pipelines,
            output_root,
            output_relative_parent,
            output_filename,
            trim_source=request.trim_source,
            log=callbacks.log,
            advance_progress=callbacks.advance_progress,
            write_idle_callback=callbacks.idle_callback,
            output_path_allocator=None,
            record_timing=record_timing,
        )

    return _run_pipeline_file


def _postprocess_runner(
    request: WorkflowRunRequest,
    callbacks: WorkflowCallbacks,
):
    def _run_postprocesses(
        postprocesses: Sequence[Any],
        output_dir: Path,
        processed_outputs: Sequence[Path],
        input_h5_paths: Sequence[Path],
        input_path: Path,
        selected_pipeline_names: Sequence[str],
        failures: list[str],
        *,
        zip_outputs: bool,
        record_timing=None,
    ) -> None:
        run_postprocesses(
            postprocesses,
            output_dir,
            processed_outputs,
            input_h5_paths,
            input_path,
            selected_pipeline_names,
            failures,
            zip_outputs=zip_outputs,
            log=callbacks.log,
            advance_progress=callbacks.advance_progress,
            idle_callback=callbacks.idle_callback,
            resolve_postprocess_files=_postprocess_file_resolver(
                selected_pipeline_names
            ),
            record_timing=record_timing,
        )

    return _run_postprocesses


def _postprocess_file_resolver(selected_pipeline_names):
    return lambda descriptor, processed_outputs, input_h5_paths: (
        _resolve_postprocess_files(
            descriptor,
            processed_outputs,
            input_h5_paths,
            selected_pipeline_names=selected_pipeline_names,
        )
    )


def _resolve_postprocess_files(
    descriptor,
    processed_outputs,
    input_h5_paths,
    *,
    selected_pipeline_names=(),
):
    result = compatible_postprocess_files(
        processed_outputs=processed_outputs,
        input_h5_paths=input_h5_paths,
        required_pipelines=getattr(descriptor, "required_pipelines", ()),
        required_pipeline_options=getattr(
            descriptor,
            "required_pipeline_options",
            (),
        ),
        selected_pipeline_names=selected_pipeline_names,
    )
    return result.files, result.skipped


def make_zip_progress_callback(
    *,
    set_progress_units: Callable[[float], None],
    progress_base: float,
    log: Callable[[str], None],
    update_ui: Callable[[], None] | None = None,
) -> ZipProgressCallback:
    return log_throttled_zip_progress(
        set_progress_units=set_progress_units,
        progress_base=progress_base,
        log=log,
        update_ui=update_ui,
    )
