from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from input_output.inputs import validate_input_paths
from workflows import run_batch

from ..services import services_for
from .base import ViewController


class RunTabController(ViewController):
    @property
    def input_var(self):
        return self.app.batch_input_var

    @property
    def output_var(self):
        return self.app.batch_output_var

    @property
    def processing_branch_var(self):
        return self.app.processing_branch_var

    @property
    def zip_outputs_var(self):
        return self.app.batch_zip_var

    @property
    def zip_name_var(self):
        return self.app.batch_zip_name_var

    @property
    def form_label_width(self) -> int:
        return self.app._ADVANCED_FORM_LABEL_WIDTH

    @property
    def text_bg(self) -> str:
        return self.app._text_bg

    @property
    def text_fg(self) -> str:
        return self.app._text_fg

    def on_batch_paths_changed(self, *_args) -> None:
        self.update_minimal_path_labels()

    def choose_folder(self) -> None:
        if getattr(self.app, "_batch_running", False):
            return
        path = services_for(self.app).file_dialogs.askdirectory(
            title="Select folder containing .holo recordings (including subfolders)",
        )
        if path:
            self.apply_inputs([Path(path)])

    def choose_file(self) -> None:
        if getattr(self.app, "_batch_running", False):
            return
        paths = services_for(self.app).file_dialogs.askopenfilenames(
            filetypes=[("Hologram recordings", "*.holo")],
            title="Select .holo recordings",
        )
        if paths:
            self.apply_inputs([Path(path) for path in paths])

    def choose_output(self) -> None:
        path = services_for(self.app).file_dialogs.askdirectory(
            initialdir=self.output_var.get() or None,
            title="Select parent folder for AngioEye acquisition outputs",
        )
        if path:
            self.output_var.set(path)

    @staticmethod
    def is_supported_file_input(path: Path) -> bool:
        return path.suffix.lower() == ".holo"

    def apply_inputs(self, input_paths: Sequence[str | Path]) -> bool:
        if getattr(self.app, "_batch_running", False):
            return False
        try:
            paths = list(validate_input_paths(input_paths))
        except (ValueError, OSError) as exc:
            services_for(self.app).dialogs.showwarning("Unsupported input", str(exc))
            return False
        self.app.batch_input_paths = paths
        if len(paths) == 1:
            label = str(paths[0])
            output_dir = paths[0] if paths[0].is_dir() else paths[0].parent
            archive_name = f"{paths[0].stem}_statistics.zip"
        else:
            names = ", ".join(path.name for path in paths[:3])
            if len(paths) > 3:
                names += ", ..."
            label = f"{len(paths)} inputs selected: {names}"
            try:
                common_parent = Path(
                    os.path.commonpath([str(path.parent) for path in paths])
                )
            except ValueError:
                common_parent = paths[0].parent
            output_dir = common_parent
            archive_name = "statistics.zip"
        self.app.batch_selection_label = label
        self.input_var.set(label)
        self.output_var.set(str(output_dir))
        self.app.batch_zip_name_var.set(archive_name)
        self.app._reset_progress()
        self.app._set_minimal_status("Ready.")
        self.update_minimal_path_labels()
        return True

    def selected_inputs(self) -> tuple[Path, ...]:
        raw_value = self.input_var.get().strip()
        paths = getattr(self.app, "batch_input_paths", ())
        if paths and raw_value == getattr(self.app, "batch_selection_label", ""):
            return tuple(paths)
        return (Path(raw_value).expanduser(),) if raw_value else ()

    def selected_pipelines(self):
        return [
            pipeline
            for pipeline in self.app.pipeline_rows
            if pipeline.available
            and self.app.pipeline_visibility.get(pipeline.name, False)
        ]

    def persist_processing_branch(self, *_args) -> None:
        try:
            self.app.settings_store.save_processing_branch(
                self.processing_branch_var.get()
            )
        except OSError as exc:
            self.app._show_settings_warning(
                "Settings not saved",
                f"Could not save processing branch preference:\n{exc}",
            )

    def update_minimal_path_labels(self) -> None:
        value = self.input_var.get().strip()
        self.app.minimal_input_path_var.set(value or "No .holo recordings selected")
        output = self.output_var.get().strip()
        self.app.minimal_output_path_var.set(f"Output: {output}" if output else "")

    def _set_running(self, running: bool) -> None:
        self.app._batch_running = running
        for name in (
            "minimal_run_button",
            "run_button",
            "minimal_select_button",
            "minimal_folder_button",
            "folder_button",
            "file_button",
            "input_entry",
        ):
            button = getattr(self.app, name, None)
            if button is not None:
                try:
                    button.configure(state="disabled" if running else "normal")
                except tk.TclError:
                    pass

    def update_progress(self, completed: int, total: int, message: str) -> None:
        self.app._progress_total_units = max(float(total), 1.0)
        self.app._set_progress_units(completed)
        self.app._set_minimal_status(message)

    def _finish_result(self, result) -> None:
        self.app._set_progress_units(self.app._progress_total_units)
        if result.failures:
            self.app._set_minimal_status("Completed with errors.")
            self.app._show_batch_error_dialog(
                f"{len(result.failures)} failure(s). See log for details.\n\n{result.summary_message}"
            )
        else:
            self.app._set_minimal_status("Completed.")

    def _poll_worker_events(self) -> None:
        for _ in range(200):
            try:
                kind, value = self._worker_events.get_nowait()
            except Empty:
                break
            if getattr(self.app, "_closing", False):
                if kind in {"result", "error"}:
                    self.app._batch_running = False
                    self.app.destroy()
                    return
                continue
            if kind == "log":
                self.app._log_batch(value)
            elif kind == "progress":
                self.update_progress(*value)
            elif kind in {"result", "error"}:
                try:
                    if kind == "result":
                        self._finish_result(value)
                    else:
                        self.app._log_batch(f"[ERROR] {value}")
                        self.app._set_minimal_status("Run failed.")
                        services_for(self.app).dialogs.showerror(
                            "Batch statistics failed", str(value)
                        )
                finally:
                    self._set_running(False)
                return
        self.app.after(50, self._poll_worker_events)

    def run(self) -> None:
        if getattr(self.app, "_batch_running", False):
            return
        dialogs = services_for(self.app).dialogs
        inputs = self.selected_inputs()
        if not inputs:
            dialogs.showwarning(
                "Missing input",
                "Select .holo recordings or one folder containing .holo recordings.",
            )
            return
        try:
            inputs = validate_input_paths(inputs)
        except (ValueError, OSError) as exc:
            dialogs.showwarning("Unsupported input", str(exc))
            return
        pipelines = self.selected_pipelines()
        if not pipelines:
            dialogs.showwarning(
                "No pipelines selected",
                "Select statistical tools in the Pipelines tab.",
            )
            return
        output_value = self.output_var.get().strip()
        if not output_value:
            dialogs.showwarning(
                "Missing output", "Select an output folder for the batch statistics."
            )
            return
        self._set_running(True)
        self.app._reset_progress()
        self.app._reset_batch_output("Starting EyeFlow batch statistics...\n")
        self._worker_events = Queue()
        events = self._worker_events
        options = {
            "zip_outputs": bool(self.zip_outputs_var.get()),
            "zip_name": self.zip_name_var.get(),
            "processing_branch": self.processing_branch_var.get(),
        }

        def work() -> None:
            try:
                result = run_batch(
                    inputs,
                    pipelines,
                    Path(output_value).expanduser(),
                    **options,
                    log=lambda message: events.put(("log", message)),
                    progress=lambda completed, total, message: events.put(
                        ("progress", (completed, total, message))
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                events.put(("error", exc))
            else:
                events.put(("result", result))

        self._worker_thread = Thread(target=work, name="angioeye-batch", daemon=False)
        self._worker_thread.start()
        self.app.after(50, self._poll_worker_events)
