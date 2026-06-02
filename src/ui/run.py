import sys
import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path

from input_output import create_zip_from_tree, is_hdf5_path
from postprocess import PostprocessDescriptor
from workflows import (
    ZIP_COMPANION_OUTPUT_FOLDERS,
    WorkflowInputSelection,
    WorkflowOutputOptions,
    WorkflowRequestState,
    WorkflowWorkSelection,
    WorkflowCallbacks,
    WorkflowInputError,
    build_workflow_request,
    dispatch_workflow,
    make_zip_progress_callback,
    missing_required_pipeline_errors,
)
from .services import services_for

def _run_batch_from_app(app) -> None:
    app._reset_progress()
    input_selection = _collect_input_selection(app)
    if (
        input_selection.convention == "legacy"
        and not input_selection.data_value
        and not input_selection.legacy_input_paths
    ):
        services_for(app).dialogs.showwarning(
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
        request = build_workflow_request(
            WorkflowRequestState(
                input_selection=input_selection,
                work_selection=selection,
                output_options=_collect_output_options(app),
            ),
            zip_output_dir=app._zip_output_dir,
            output_filename_for_run=app._minimal_output_filename_for_run,
        )
        dispatch_result = _dispatch_workflow(request, _workflow_callbacks(app))
    except WorkflowInputError as exc:
        if exc.title == "Invalid input":
            app._log_batch(f"Error: {exc.message}")
        _show_workflow_input_error(app, exc)
        return

    _finish_dispatch_result(app, dispatch_result)


def _collect_input_selection(app) -> WorkflowInputSelection:
    convention = "holo" if app._uses_holo_input_convention() else "legacy"
    return WorkflowInputSelection(
        convention=convention,
        data_value=(app.batch_input_var.get() or "").strip(),
        legacy_input_paths=tuple(
            app._selected_batch_input_paths() if convention == "legacy" else ()
        ),
        holo_paths=tuple(app._selected_holo_paths() if convention == "holo" else ()),
    )


def _collect_output_options(app) -> WorkflowOutputOptions:
    return WorkflowOutputOptions(
        base_output_value=(app.batch_output_var.get() or "").strip(),
        zip_outputs=bool(app.batch_zip_var.get()),
        zip_name=app.batch_zip_name_var.get(),
        trim_source=_trim_eyeflow_source(app),
    )


def _resolve_run_selection(app) -> WorkflowWorkSelection | None:
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
        services_for(app).dialogs.showwarning(
            "No work selected",
            "Select at least one pipeline or postprocess step.",
        )
        return None

    pipelines = []
    missing: list[str] = []
    for name in pipeline_names:
        pipeline = app.pipeline_registry.get(name)
        if pipeline is None:
            missing.append(name)
        else:
            pipelines.append(pipeline)
    if missing:
        services_for(app).dialogs.showerror(
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
        services_for(app).dialogs.showerror(
            "Postprocess missing",
            f"Postprocess step(s) not registered: {', '.join(missing_postprocesses)}",
        )
        return None

    return WorkflowWorkSelection(
        pipeline_names=tuple(pipeline_names),
        pipelines=tuple(pipelines),
        postprocesses=tuple(postprocesses),
    )


def _dispatch_workflow(request, callbacks):
    compat_module = sys.modules.get("angio_eye")
    dispatch = getattr(compat_module, "dispatch_workflow", dispatch_workflow)
    return dispatch(request, callbacks)


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
    dialogs = services_for(app).dialogs
    if error.title == "Missing input":
        dialogs.showwarning(error.title, error.message)
    else:
        dialogs.showerror(error.title, error.message)
    app._set_minimal_status(error.status)

def _finish_dispatch_result(app, dispatch_result) -> None:
    workflow_result = dispatch_result.workflow_result
    if workflow_result is None:
        app._update_holo_status_labels()
        _show_skipped_holo_warning(app, dispatch_result.skipped_holo_stems)
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
        services_for(app).dialogs.showerror(
            "ZIP failed",
            f"Could not create ZIP archive: {workflow_result.zip_error}",
        )
    _show_skipped_holo_warning(app, dispatch_result.skipped_holo_stems)

def _show_skipped_holo_warning(app, skipped_holo_stems: Sequence[str]) -> None:
    if not skipped_holo_stems:
        return
    services_for(app).dialogs.showwarning(
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
