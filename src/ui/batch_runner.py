from __future__ import annotations

import shutil
import tempfile
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

from .holo import HoloInputContext


def run_batch(app: Any) -> None:
    app._reset_progress()
    holo_mode = app._uses_holo_input_convention()
    holo_paths = app._selected_holo_paths() if holo_mode else []
    data_value = (
        str(holo_paths[0])
        if holo_mode
        else (app.batch_input_var.get() or "").strip()
    )
    if (holo_mode and not holo_paths) or (not holo_mode and not data_value):
        messagebox.showwarning(
            "Missing input",
            (
                "Select one or more .holo files to process."
                if holo_mode
                else "Select a folder, HDF5 file, or .zip archive to process."
            ),
        )
        return
    data_path = Path(data_value).expanduser()
    holo_contexts: list[HoloInputContext] = []
    skipped_holo_stems: list[str] = []
    if holo_mode:
        for holo_path in holo_paths:
            try:
                holo_contexts.append(app._resolve_holo_context(holo_path))
            except Exception:  # noqa: BLE001
                skipped_holo_stems.append(holo_path.stem)
        if not holo_contexts:
            app._update_holo_status_labels()
            _show_skipped_holo_warning(skipped_holo_stems)
            app._set_minimal_status("Run skipped.")
            return

    selected_names = [
        pipeline.name
        for pipeline in app.pipeline_rows
        if pipeline.available and app.pipeline_visibility.get(pipeline.name, False)
    ]
    if not selected_names:
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

    pipelines = []
    missing: list[str] = []
    for name in selected_names:
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

    postprocesses = []
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
        selected_pipeline_names=selected_names,
    )
    if postprocess_requirement_errors:
        messagebox.showerror(
            "Postprocess requirements",
            "\n".join(postprocess_requirement_errors),
        )
        return

    if holo_contexts:
        for context in holo_contexts:
            try:
                app._reset_holo_output_dir(context)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Invalid output", str(exc))
                app._set_minimal_status("Run failed.")
                return
        _run_holo_batch(
            app=app,
            holo_contexts=holo_contexts,
            pipelines=pipelines,
            postprocesses=postprocesses,
            selected_names=selected_names,
            skipped_holo_stems=skipped_holo_stems,
        )
        return

    _run_legacy_batch(
        app=app,
        data_path=data_path,
        pipelines=pipelines,
        postprocesses=postprocesses,
        selected_names=selected_names,
    )


def _run_legacy_batch(
    *,
    app: Any,
    data_path: Path,
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
) -> None:
    base_output_value = (app.batch_output_var.get() or "").strip()
    base_output_dir = (
        Path(base_output_value).expanduser() if base_output_value else Path.cwd()
    )
    if not base_output_dir.is_absolute():
        base_output_dir = Path.cwd() / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    app._reset_batch_output("Starting batch run...\n")
    app._set_minimal_status("Preparing batch...")

    tempdir: tempfile.TemporaryDirectory | None = None
    try:
        data_root, tempdir = app._prepare_data_root(data_path)
        inputs = app._find_h5_inputs(data_root)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
        app._log_batch(f"Error: {exc}")
        app._set_minimal_status("Run failed.")
        if tempdir is not None:
            tempdir.cleanup()
        return

    _run_input_batch(
        app=app,
        inputs=inputs,
        data_root=data_root,
        pipelines=pipelines,
        postprocesses=postprocesses,
        selected_names=selected_names,
        input_path=data_path,
        base_output_dir=base_output_dir,
        output_dir=base_output_dir,
        output_filename=app._minimal_output_filename_for_run(data_path, inputs),
        tempdir=tempdir,
    )


def _run_holo_batch(
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


def _run_input_batch(
    *,
    app: Any,
    inputs: list[Path],
    data_root: Path,
    pipelines: list[Any],
    postprocesses: list[Any],
    selected_names: list[str],
    input_path: Path,
    base_output_dir: Path,
    output_dir: Path,
    output_filename: str | None,
    tempdir: tempfile.TemporaryDirectory | None = None,
) -> None:
    pipeline_progress_units = len(inputs) * len(pipelines)
    final_progress_units = len(postprocesses) + (
        1 if app.batch_zip_var.get() else 0
    )
    app._start_progress(
        pipeline_progress_units,
        style_name=app._progress_primary_style,
        status_text="Running pipelines...",
    )

    work_output_dir: Path | None = None
    clean_work_output = False
    zip_failed = False
    try:
        if app.batch_zip_var.get():
            work_output_dir = Path(tempfile.mkdtemp(dir=base_output_dir))
            output_dir = work_output_dir

        failures: list[str] = []
        processed_outputs: list[Path] = []
        processed_input_paths: list[Path] = []
        for h5_path in inputs:
            try:
                relative_parent = app._relative_input_parent(h5_path, data_root)
                combined_output = app._run_pipelines_on_file(
                    h5_path,
                    pipelines,
                    output_dir,
                    output_relative_parent=relative_parent,
                    output_filename=output_filename,
                )
                processed_outputs.append(combined_output)
                processed_input_paths.append(h5_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                app._log_batch(f"[FAIL] {h5_path.name}: {exc}")

        if final_progress_units:
            final_status = (
                "Running postprocess..." if postprocesses else "Creating ZIP..."
            )
            app._start_progress(
                final_progress_units,
                style_name=app._progress_final_style,
                status_text=final_status,
            )

        if postprocesses and processed_outputs:
            app._run_postprocesses(
                postprocesses=postprocesses,
                output_dir=output_dir,
                processed_outputs=processed_outputs,
                input_h5_paths=processed_input_paths,
                input_path=input_path,
                selected_pipeline_names=selected_names,
                failures=failures,
            )
        elif postprocesses:
            app._log_batch(
                "[POST SKIP] No successful pipeline outputs were generated, "
                "so postprocess steps were skipped."
            )
            app._advance_progress(len(postprocesses))

        summary_msg: str
        if app.batch_zip_var.get():
            try:
                zip_name = app.batch_zip_name_var.get().strip() or "outputs.zip"
                if not zip_name.lower().endswith(".zip"):
                    zip_name += ".zip"
                app._set_minimal_status("Creating ZIP...")
                app._log_batch("[ZIP] Preparing archive...")
                last_progress_log = 0.0
                zip_progress_base = app._progress_completed_units

                def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
                    nonlocal last_progress_log
                    fraction = 1.0 if total == 0 else done / total
                    app._set_progress_units(zip_progress_base + fraction)
                    now = time.monotonic()
                    if done == total or (now - last_progress_log) >= 0.5:
                        pct = 100 if total == 0 else int((done * 100) / total)
                        app._log_batch(f"[ZIP] {done}/{total} files ({pct}%)")
                        last_progress_log = now
                        try:
                            app.update()
                        except tk.TclError:
                            pass

                zip_path = app._zip_output_dir(
                    output_dir,
                    target_path=base_output_dir / zip_name,
                    progress_callback=_zip_progress,
                )
                app._log_batch(f"[ZIP] Archive created: {zip_path}")
                summary_msg = f"ZIP archive: {zip_path}"
                clean_work_output = True
            except Exception as exc:  # noqa: BLE001
                zip_failed = True
                app._set_progress_units(zip_progress_base + 1.0)
                app._log_batch(f"[ZIP FAIL] {exc}")
                messagebox.showerror(
                    "Zip failed", f"Could not create ZIP archive: {exc}"
                )
                summary_msg = f"Outputs stored under: {output_dir}"
        else:
            if len(processed_outputs) == 1:
                summary_msg = f"Output file: {processed_outputs[0]}"
            else:
                summary_msg = f"Outputs stored under: {output_dir}"

        app._set_progress_units(app._progress_total_units)
        app._log_batch(f"Completed. {summary_msg}")

        if failures:
            app._set_minimal_status("Completed with errors.")
            app._show_batch_error_dialog(
                f"{len(failures)} failure(s). See log for details.\n\n{summary_msg}"
            )
        else:
            app._set_minimal_status(
                "Completed with errors." if zip_failed else "Process ended."
            )
    finally:
        if tempdir is not None:
            tempdir.cleanup()
        if clean_work_output and work_output_dir is not None:
            shutil.rmtree(work_output_dir, ignore_errors=True)
