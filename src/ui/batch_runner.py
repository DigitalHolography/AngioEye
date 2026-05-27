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
    data_value = (
        (app.holo_input_var.get() or "").strip()
        if holo_mode
        else (app.batch_input_var.get() or "").strip()
    )
    if not data_value:
        messagebox.showwarning(
            "Missing input",
            (
                "Select a .holo file to process."
                if holo_mode
                else "Select a folder, HDF5 file, or .zip archive to process."
            ),
        )
        return
    data_path = Path(data_value).expanduser()
    holo_context: HoloInputContext | None = None
    if holo_mode:
        try:
            holo_context = app._resolve_holo_context(data_path)
        except Exception as exc:  # noqa: BLE001
            app._update_holo_status_labels()
            messagebox.showerror("Invalid holo input", str(exc))
            app._set_minimal_status("Run failed.")
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

    if holo_context is not None:
        base_output_dir = holo_context.output_dir
        try:
            app._reset_holo_output_dir(holo_context)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid output", str(exc))
            app._set_minimal_status("Run failed.")
            return
    else:
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
        if holo_context is not None:
            data_root = holo_context.ef_dir
            inputs = [holo_context.h5_path]
            tempdir = None
            app._log_batch(f"[INPUT] Holo file -> {holo_context.holo_path}")
            app._log_batch(f"[INPUT] EF h5 -> {holo_context.h5_path}")
            app._log_batch(f"[OUTPUT] AE folder -> {holo_context.output_dir}")
        else:
            data_root, tempdir = app._prepare_data_root(data_path)
            inputs = app._find_h5_inputs(data_root)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
        app._log_batch(f"Error: {exc}")
        app._set_minimal_status("Run failed.")
        if tempdir is not None:
            tempdir.cleanup()
        return

    pipeline_progress_units = len(inputs) * len(pipelines)
    final_progress_units = len(postprocesses) + (
        1 if app.batch_zip_var.get() else 0
    )
    app._start_progress(
        pipeline_progress_units,
        style_name=app._progress_primary_style,
        status_text="Running pipelines...",
    )
    output_filename = (
        app._holo_output_filename(holo_context.holo_path)
        if holo_context is not None
        else app._minimal_output_filename_for_run(data_path, inputs)
    )

    work_output_dir: Path | None = None
    clean_work_output = False
    zip_failed = False
    try:
        output_dir = base_output_dir
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
                input_path=data_path,
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
