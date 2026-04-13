from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Any

from input_output import ZipH5Member
from workflows import (
    RunWorkflowResult,
    ZipBatchSettings,
    log_throttled_zip_progress,
    prepare_run_input,
    run_filesystem_workflow,
    run_zip_workflow,
)

from .holo import HoloInputContext

ZIP_BATCH_SETTINGS = ZipBatchSettings.from_env()


@dataclass(frozen=True)
class RunSelection:
    pipeline_names: list[str]
    pipelines: list[Any]
    postprocesses: list[Any]


def run(app: Any) -> None:
    app._reset_progress()
    match _run_mode(app):
        case "holo":
            _run_holo_mode(app)
        case "legacy":
            _run_legacy_mode(app)
        case mode:
            messagebox.showerror("Invalid input mode", f"Unknown input mode: {mode}")
            app._set_minimal_status("Run failed.")


def _run_mode(app: Any) -> str:
    return "holo" if app._uses_holo_input_convention() else "legacy"


def _run_holo_mode(app: Any) -> None:
    holo_paths = app._selected_holo_paths()
    if not holo_paths:
        messagebox.showwarning(
            "Missing input",
            "Select one or more .holo files to process.",
        )
        return

    holo_contexts, skipped_holo_stems = _resolve_holo_contexts(app, holo_paths)
    if not holo_contexts:
        app._update_holo_status_labels()
        _show_skipped_holo_warning(skipped_holo_stems)
        app._set_minimal_status("Run skipped.")
        return

    selection = _resolve_run_selection(app)
    if selection is None:
        return

    for context in holo_contexts:
        try:
            app._reset_holo_output_dir(context)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid output", str(exc))
            app._set_minimal_status("Run failed.")
            return

    _run_holo_inputs(
        app=app,
        holo_contexts=holo_contexts,
        pipelines=selection.pipelines,
        postprocesses=selection.postprocesses,
        selected_names=selection.pipeline_names,
        skipped_holo_stems=skipped_holo_stems,
    )


def _resolve_holo_contexts(
    app: Any,
    holo_paths: Iterable[Path],
) -> tuple[list[HoloInputContext], list[str]]:
    holo_contexts: list[HoloInputContext] = []
    skipped_holo_stems: list[str] = []
    for holo_path in holo_paths:
        try:
            holo_contexts.append(app._resolve_holo_context(holo_path))
        except Exception:  # noqa: BLE001
            skipped_holo_stems.append(holo_path.stem)
    return holo_contexts, skipped_holo_stems


def _run_legacy_mode(app: Any) -> None:
    data_value = (app.batch_input_var.get() or "").strip()
    if not data_value:
        messagebox.showwarning(
            "Missing input",
            "Select a folder, HDF5 file, or .zip archive to process.",
        )
        return

    selection = _resolve_run_selection(app)
    if selection is None:
        return

    _run_legacy_input(
        app=app,
        data_path=Path(data_value).expanduser(),
        pipelines=selection.pipelines,
        postprocesses=selection.postprocesses,
        selected_names=selection.pipeline_names,
    )


def _resolve_run_selection(app: Any) -> RunSelection | None:
    pipeline_names = [
        pipeline.name
        for pipeline in app.pipeline_rows
        if pipeline.available and app.pipeline_visibility.get(pipeline.name, False)
    ]
    if not pipeline_names:
        messagebox.showwarning(
            "No pipelines",
            "Select at least one pipeline in Pipeline Library.",
        )
        return

    selected_postprocess_names = [
        postprocess.name
        for postprocess in app.postprocess_rows
        if postprocess.available
        and app.postprocess_visibility.get(postprocess.name, False)
    ]

    pipelines: list[Any] = []
    missing: list[str] = []
    for name in pipeline_names:
        pipeline = app.pipeline_registry.get(name)
        if pipeline is None:
            missing.append(name)
        else:
            pipelines.append(pipeline)
    if missing:
        messagebox.showerror(
            "Pipeline missing", f"Pipeline(s) not registered: {', '.join(missing)}"
        )
        return

    postprocesses: list[Any] = []
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
        return

    postprocess_requirement_errors = app._validate_postprocess_selection(
        postprocesses,
        selected_pipeline_names=pipeline_names,
    )
    if postprocess_requirement_errors:
        messagebox.showerror(
            "Postprocess requirements",
            "\n".join(postprocess_requirement_errors),
        )
        return

    return RunSelection(
        pipeline_names=pipeline_names,
        pipelines=pipelines,
        postprocesses=postprocesses,
    )


def _run_legacy_input(
    *,
    app: Any,
    data_path: Path,
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
) -> None:
    base_output_dir = _resolve_base_output_dir(app)

    app._reset_batch_output("Starting batch run...\n")
    app._set_minimal_status("Preparing batch...")

    try:
        input_plan = prepare_run_input(data_path)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
        app._log_batch(f"Error: {exc}")
        app._set_minimal_status("Run failed.")
        return

    if input_plan.is_zip:
        if input_plan.item_count == 0:
            messagebox.showerror(
                "Invalid input",
                f"No .h5/.hdf5 files found inside ZIP archive: {data_path}",
            )
            app._log_batch(f"Error: No .h5/.hdf5 files found inside {data_path}")
            app._set_minimal_status("Run failed.")
            return

        _run_zip_input(
            app=app,
            zip_path=data_path,
            members=input_plan.iter_zip_members(),
            member_count=input_plan.item_count,
            pipelines=pipelines,
            postprocesses=postprocesses,
            selected_names=selected_names,
            base_output_dir=base_output_dir,
        )
        return

    _run_hdf5_inputs(
        app=app,
        inputs=list(input_plan.h5_paths),
        data_root=input_plan.input_path,
        pipelines=pipelines,
        postprocesses=postprocesses,
        selected_names=selected_names,
        input_path=input_plan.input_path,
        base_output_dir=base_output_dir,
        output_filename=app._minimal_output_filename_for_run(
            input_plan.input_path,
            input_plan.h5_paths,
        ),
    )


def _resolve_base_output_dir(app: Any) -> Path:
    base_output_value = (app.batch_output_var.get() or "").strip()
    base_output_dir = (
        Path(base_output_value).expanduser() if base_output_value else Path.cwd()
    )
    if not base_output_dir.is_absolute():
        base_output_dir = Path.cwd() / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir


def _run_zip_input(
    *,
    app: Any,
    zip_path: Path,
    members: Iterable[ZipH5Member],
    member_count: int,
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
    base_output_dir: Path,
) -> None:
    pipeline_progress_units = member_count * len(pipelines)
    app._start_progress(
        pipeline_progress_units,
        style_name=app._progress_primary_style,
        status_text="Running pipelines...",
    )
    app._log_batch(
        f"[ZIP] Found {member_count} HDF5 file(s). "
        f"Extracting {ZIP_BATCH_SETTINGS.batch_size} at a time; "
        f"running pipelines with {ZIP_BATCH_SETTINGS.pipeline_workers} worker(s)."
    )

    workflow_result = run_zip_workflow(
        zip_path=zip_path,
        members=members,
        member_count=member_count,
        pipelines=pipelines,
        postprocesses=postprocesses,
        selected_pipeline_names=selected_names,
        base_output_dir=base_output_dir,
        zip_outputs=_zip_outputs_requested(app),
        zip_name=_zip_output_name(app),
        settings=ZIP_BATCH_SETTINGS,
        run_pipeline_file=_zip_pipeline_runner(app),
        run_postprocesses=app._run_postprocesses,
        zip_output_dir=app._zip_output_dir,
        log=app._log_batch,
        advance_progress=app._advance_progress,
        start_final_progress=lambda units, status: _start_final_progress(
            app,
            units,
            status,
        ),
        set_status=app._set_minimal_status,
        make_zip_progress_callback=lambda: _make_zip_progress_callback(app),
        on_zip_error=_show_zip_error,
        idle_callback=lambda: _update_ui(app),
    )
    _finish_workflow_result(app, workflow_result)


def _zip_pipeline_runner(app: Any):
    if ZIP_BATCH_SETTINGS.pipeline_workers <= 1:
        return app._run_pipelines_on_file
    return getattr(app, "_run_zip_pipelines_on_file", app._run_pipelines_on_file)


def _run_holo_inputs(
    *,
    app: Any,
    holo_contexts: list[HoloInputContext],
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
    skipped_holo_stems: list[str],
) -> None:
    app._reset_batch_output("Starting batch run...\n")
    app._set_minimal_status("Preparing batch...")

    failures: list[str] = []
    processed_outputs_by_context: dict[HoloInputContext, list[Path]] = {
        context: [] for context in holo_contexts
    }
    input_paths_by_context: dict[HoloInputContext, list[Path]] = {
        context: [] for context in holo_contexts
    }

    app._start_progress(
        len(holo_contexts) * len(pipelines),
        style_name=app._progress_primary_style,
        status_text="Running pipelines...",
    )
    for context in holo_contexts:
        app._log_batch(f"[INPUT] Holo file -> {context.holo_path}")
        app._log_batch(f"[INPUT] EF h5 -> {context.h5_path}")
        app._log_batch(f"[OUTPUT] AE folder -> {context.output_dir}")
        try:
            combined_output = app._run_pipelines_on_file(
                context.h5_path,
                pipelines,
                context.output_dir,
                output_relative_parent=Path("."),
                output_filename=app._holo_output_filename(context.holo_path),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{context.h5_path}: {exc}")
            app._log_batch(f"[FAIL] {context.h5_path.name}: {exc}")
            continue
        processed_outputs_by_context[context].append(combined_output)
        input_paths_by_context[context].append(context.h5_path)

    if postprocesses:
        app._start_progress(
            len(holo_contexts) * len(postprocesses),
            style_name=app._progress_final_style,
            status_text="Running postprocess...",
        )
        for context in holo_contexts:
            processed_outputs = processed_outputs_by_context[context]
            if processed_outputs:
                app._run_postprocesses(
                    postprocesses=postprocesses,
                    output_dir=context.output_dir,
                    processed_outputs=processed_outputs,
                    input_h5_paths=input_paths_by_context[context],
                    input_path=context.holo_path,
                    selected_pipeline_names=selected_names,
                    failures=failures,
                )
            else:
                app._log_batch(
                    f"[POST SKIP] {context.holo_path.name}: no successful pipeline "
                    "outputs were generated."
                )
                app._advance_progress(len(postprocesses))

    processed_outputs = [
        output
        for context_outputs in processed_outputs_by_context.values()
        for output in context_outputs
    ]
    if len(processed_outputs) == 1:
        summary_msg = f"Output file: {processed_outputs[0]}"
    else:
        summary_msg = f"Outputs generated for {len(processed_outputs)} holo file(s)."
    app._set_progress_units(app._progress_total_units)
    app._log_batch(f"Completed. {summary_msg}")

    if failures:
        app._set_minimal_status("Completed with errors.")
        app._show_batch_error_dialog(
            f"{len(failures)} failure(s). See log for details.\n\n{summary_msg}"
        )
    else:
        app._set_minimal_status("Process ended.")

    _show_skipped_holo_warning(skipped_holo_stems)


def _show_skipped_holo_warning(skipped_holo_stems: list[str]) -> None:
    if not skipped_holo_stems:
        return
    messagebox.showwarning(
        "Skipped files",
        f"Skipped {len(skipped_holo_stems)} files: {', '.join(skipped_holo_stems)}",
    )


def _make_zip_progress_callback(app: Any):
    return log_throttled_zip_progress(
        set_progress_units=app._set_progress_units,
        progress_base=app._progress_completed_units,
        log=app._log_batch,
        update_ui=lambda: _update_ui(app),
    )


def _update_ui(app: Any) -> None:
    try:
        update_idletasks = getattr(app, "update_idletasks", None)
        if update_idletasks is not None:
            update_idletasks()
        app.update()
    except (AttributeError, tk.TclError):
        pass


def _finish_workflow_result(app: Any, workflow_result: RunWorkflowResult) -> None:
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


def _run_hdf5_inputs(
    *,
    app: Any,
    inputs: list[Path],
    data_root: Path,
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
    input_path: Path,
    base_output_dir: Path,
    output_filename: str | None,
) -> None:
    pipeline_progress_units = len(inputs) * len(pipelines)
    app._start_progress(
        pipeline_progress_units,
        style_name=app._progress_primary_style,
        status_text="Running pipelines...",
    )

    workflow_result = run_filesystem_workflow(
        inputs=inputs,
        data_root=data_root,
        pipelines=pipelines,
        postprocesses=postprocesses,
        selected_pipeline_names=selected_names,
        input_path=input_path,
        base_output_dir=base_output_dir,
        zip_outputs=_zip_outputs_requested(app),
        zip_name=_zip_output_name(app),
        output_filename=output_filename,
        run_pipeline_file=app._run_pipelines_on_file,
        run_postprocesses=app._run_postprocesses,
        relative_parent=app._relative_input_parent,
        zip_output_dir=app._zip_output_dir,
        log=app._log_batch,
        advance_progress=app._advance_progress,
        start_final_progress=lambda units, status: _start_final_progress(
            app,
            units,
            status,
        ),
        set_status=app._set_minimal_status,
        make_zip_progress_callback=lambda: _make_zip_progress_callback(app),
        on_zip_error=_show_zip_error,
    )
    _finish_workflow_result(app, workflow_result)


def _zip_outputs_requested(app: Any) -> bool:
    return bool(app.batch_zip_var.get())


def _zip_output_name(app: Any) -> str:
    return app.batch_zip_name_var.get()


def _start_final_progress(app: Any, units: float, status: str) -> None:
    app._start_progress(
        units,
        style_name=app._progress_final_style,
        status_text=status,
    )


def _show_zip_error(error: str) -> None:
    messagebox.showerror(
        "ZIP failed",
        f"Could not create ZIP archive: {error}",
    )

