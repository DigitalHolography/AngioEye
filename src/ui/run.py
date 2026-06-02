import sys
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox

from input_output import create_zip_from_tree, is_hdf5_path
from pipelines import PipelineDescriptor
from postprocess import PostprocessDescriptor
from workflows import (
    ZIP_COMPANION_OUTPUT_FOLDERS,
    WorkflowCallbacks,
    WorkflowInputError,
    WorkflowRunRequest,
    dispatch_workflow,
    make_zip_progress_callback,
    missing_required_pipeline_errors,
    prepare_run_input,
    prepare_run_inputs,
)

@dataclass(frozen=True)
class _RunSelection:
    pipeline_names: list[str]
    pipelines: list[PipelineDescriptor]
    postprocesses: list[PostprocessDescriptor]

def _run_batch_from_app(app) -> None:
    app._reset_progress()
    mode = "holo" if app._uses_holo_input_convention() else "legacy"
    data_value = (app.batch_input_var.get() or "").strip()
    legacy_input_paths = app._selected_batch_input_paths() if mode == "legacy" else []
    if mode == "legacy" and not data_value and not legacy_input_paths:
        messagebox.showwarning(
            "Missing input",
            "Select a folder, HDF5 file, or .zip archive to process.",
        )
        return

    selection = _resolve_run_selection(app)
    if selection is None:
        return

    app._reset_batch_output("Starting batch run...\n")
    app._set_minimal_status("Preparing batch...")

    try:
        request = _build_workflow_request(
            app=app,
            mode=mode,
            data_value=data_value,
            selection=selection,
        )
        dispatch_result = _dispatch_workflow(request, _workflow_callbacks(app))
    except WorkflowInputError as exc:
        _show_workflow_input_error(app, exc)
        return

    _finish_dispatch_result(app, dispatch_result)

def _resolve_run_selection(app) -> _RunSelection | None:
    pipeline_names = [
        pipeline.name
        for pipeline in app.pipeline_rows
        if pipeline.available and app.pipeline_visibility.get(pipeline.name, False)
    ]
    selected_postprocess_names = [
        postprocess.name
        for postprocess in app.postprocess_rows
        if postprocess.available
        and app.postprocess_visibility.get(postprocess.name, False)
    ]
    if not pipeline_names and not selected_postprocess_names:
        messagebox.showwarning(
            "No work selected",
            "Select at least one pipeline or postprocess step.",
        )
        return None

    pipelines: list[PipelineDescriptor] = []
    missing: list[str] = []
    for name in pipeline_names:
        pipeline = app.pipeline_registry.get(name)
        if pipeline is None:
            missing.append(name)
        else:
            pipelines.append(pipeline)
    if missing:
        messagebox.showerror(
            "Pipeline missing",
            f"Pipeline(s) not registered: {', '.join(missing)}",
        )
        return None

    postprocesses: list[PostprocessDescriptor] = []
    missing_postprocesses: list[str] = []
    for name in selected_postprocess_names:
        postprocess = app.postprocess_registry.get(name)
        if postprocess is None:
            missing_postprocesses.append(name)
        else:
            postprocesses.append(postprocess)
    if missing_postprocesses:
        messagebox.showerror(
            "Postprocess missing",
            f"Postprocess step(s) not registered: {', '.join(missing_postprocesses)}",
        )
        return None

    return _RunSelection(
        pipeline_names=pipeline_names,
        pipelines=pipelines,
        postprocesses=postprocesses,
    )


def _dispatch_workflow(request, callbacks):
    compat_module = sys.modules.get("angio_eye")
    dispatch = getattr(compat_module, "dispatch_workflow", dispatch_workflow)
    return dispatch(request, callbacks)


def _build_workflow_request(
    *,
    app,
    mode: str,
    data_value: str,
    selection: _RunSelection,
) -> WorkflowRunRequest:
    input_plan = None
    request_mode = "holo"
    if mode != "holo":
        try:
            selected_paths = app._selected_batch_input_paths()
            if selected_paths:
                input_plan = prepare_run_inputs(selected_paths)
            else:
                input_plan = prepare_run_input(Path(data_value).expanduser())
        except Exception as exc:  # noqa: BLE001
            app._log_batch(f"Error: {exc}")
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
    postprocess_requirement_errors = app._validate_postprocess_selection(
        selection.postprocesses,
        selected_pipeline_names=selection.pipeline_names,
        reusable_h5_paths=reusable_h5_paths,
        defer_when_no_reusable_paths=bool(input_plan and input_plan.is_zip)
        or (mode == "holo" and not selection.pipelines),
    )
    if postprocess_requirement_errors:
        raise WorkflowInputError(
            "Postprocess requirements",
            "\n".join(postprocess_requirement_errors),
        )

    return WorkflowRunRequest(
        mode=request_mode,
        input_plan=input_plan,
        holo_paths=app._selected_holo_paths() if mode == "holo" else (),
        base_output_dir=Path.cwd() if mode == "holo" else _resolve_base_output_dir(app),
        pipelines=selection.pipelines,
        postprocesses=selection.postprocesses,
        selected_pipeline_names=selection.pipeline_names,
        zip_outputs=bool(app.batch_zip_var.get()),
        zip_name=app.batch_zip_name_var.get(),
        trim_source=_trim_eyeflow_source(app),
        zip_output_dir=app._zip_output_dir,
        output_filename_for_run=app._minimal_output_filename_for_run,
    )

def _resolve_base_output_dir(app) -> Path:
    base_output_value = (app.batch_output_var.get() or "").strip()
    base_output_dir = (
        Path(base_output_value).expanduser() if base_output_value else Path.cwd()
    )
    if not base_output_dir.is_absolute():
        base_output_dir = Path.cwd() / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir

def _trim_eyeflow_source(app) -> bool:
    persist_var = getattr(app, "_persist_eyeflow_data", None)
    return not bool(persist_var.get()) if persist_var is not None else True

def _workflow_callbacks(app) -> WorkflowCallbacks:
    return WorkflowCallbacks(
        log=app._log_batch,
        start_primary_progress=lambda units, status: app._start_progress(
            units,
            style_name=app._progress_primary_style,
            status_text=status,
        ),
        start_final_progress=lambda units, status: app._start_progress(
            units,
            style_name=app._progress_final_style,
            status_text=status,
        ),
        advance_progress=app._advance_progress,
        set_progress_units=app._set_progress_units,
        set_status=app._set_minimal_status,
        make_zip_progress_callback=lambda: make_zip_progress_callback(
            set_progress_units=app._set_progress_units,
            progress_base=app._progress_completed_units,
            log=app._log_batch,
            update_ui=lambda: _update_ui(app),
        ),
        idle_callback=lambda: _update_ui(app),
    )

def _show_workflow_input_error(app, error: WorkflowInputError) -> None:
    if error.title == "Missing input":
        messagebox.showwarning(error.title, error.message)
    else:
        messagebox.showerror(error.title, error.message)
    app._set_minimal_status(error.status)

def _finish_dispatch_result(app, dispatch_result) -> None:
    workflow_result = dispatch_result.workflow_result
    if workflow_result is None:
        app._update_holo_status_labels()
        _show_skipped_holo_warning(dispatch_result.skipped_holo_stems)
        app._set_minimal_status("Run skipped.")
        return

    app._set_progress_units(app._progress_total_units)
    app._log_batch(f"Completed. {workflow_result.summary_message}")

    if workflow_result.failures:
        app._set_minimal_status("Completed with errors.")
        app._show_batch_error_dialog(
            f"{len(workflow_result.failures)} failure(s). See log for details.\n\n"
            f"{workflow_result.summary_message}"
        )
    else:
        app._set_minimal_status(
            "Completed with errors." if workflow_result.zip_failed else "Process ended."
        )
    if workflow_result.zip_failed and workflow_result.zip_error:
        messagebox.showerror(
            "ZIP failed",
            f"Could not create ZIP archive: {workflow_result.zip_error}",
        )
    _show_skipped_holo_warning(dispatch_result.skipped_holo_stems)

def _show_skipped_holo_warning(skipped_holo_stems: Sequence[str]) -> None:
    if not skipped_holo_stems:
        return
    messagebox.showwarning(
        "Skipped files",
        f"Skipped {len(skipped_holo_stems)} files: {', '.join(skipped_holo_stems)}",
    )

def _update_ui(app) -> None:
    try:
        update_idletasks = getattr(app, "update_idletasks", None)
        if update_idletasks is not None:
            update_idletasks()
        app.update()
    except (AttributeError, tk.TclError):
        pass


class RunMixin:
    def _minimal_output_filename_for_run(
        self,
        data_path: Path,
        inputs: Sequence[Path],
    ) -> str | None:
        if self.ui_mode != "minimal":
            return None
        if self.batch_zip_var.get():
            return None
        if len(inputs) != 1:
            return None
        if not data_path.is_file():
            return None
        if not is_hdf5_path(data_path):
            return None
        return self._default_output_artifact_name(data_path)

    def run_batch(self) -> None:
        _run_batch_from_app(self)

    def _validate_postprocess_selection(
        self,
        postprocesses: Sequence[PostprocessDescriptor],
        selected_pipeline_names: Sequence[str],
        reusable_h5_paths: Sequence[Path] = (),
        defer_when_no_reusable_paths: bool = False,
    ) -> list[str]:
        return missing_required_pipeline_errors(
            postprocesses=postprocesses,
            selected_pipeline_names=selected_pipeline_names,
            reusable_h5_paths=reusable_h5_paths,
            defer_when_no_reusable_paths=defer_when_no_reusable_paths,
        )

    def _zip_output_dir(
        self,
        folder: Path,
        target_path: Path | None = None,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> Path:
        folder = folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {folder}")
        if target_path is None:
            zip_name = f"{folder.name}_outputs.zip" if folder.name else "outputs.zip"
            zip_path = folder.parent / zip_name
        else:
            zip_path = target_path.expanduser().resolve()
        if zip_path.exists():
            zip_path.unlink()
        return create_zip_from_tree(
            folder,
            zip_path,
            exclude_root_dirs=ZIP_COMPANION_OUTPUT_FOLDERS,
            compresslevel=1,
            progress_callback=progress_callback,
        )
