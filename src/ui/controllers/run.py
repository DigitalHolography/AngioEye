from __future__ import annotations

import os
import sys
import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path

from input_output import create_zip_from_tree, is_hdf5_path
from workflows import (
    ZIP_COMPANION_OUTPUT_FOLDERS,
    HoloInputContext,
    WorkflowCallbacks,
    WorkflowInputError,
    WorkflowInputSelection,
    WorkflowOutputOptions,
    WorkflowRequestState,
    WorkflowWorkSelection,
    build_workflow_request,
    dataset_dir as holo_dataset_dir,
    dispatch_workflow,
    ef_dir as holo_ef_dir,
    find_ef_h5 as find_holo_ef_h5,
    make_zip_progress_callback,
    output_dir as holo_output_dir,
    output_filename as holo_output_filename,
    read_stem_list,
    reset_output_dir as reset_holo_output_dir,
    resolve_context as resolve_holo_context,
)

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
    def holo_status_var(self):
        return self.app.holo_status_var

    @property
    def holo_output_path_var(self):
        return self.app.holo_output_path_var

    @property
    def persist_eyeflow_data_var(self):
        return self.app._persist_eyeflow_data

    @property
    def form_label_width(self) -> int:
        return self.app._ADVANCED_FORM_LABEL_WIDTH

    @property
    def bg_color(self) -> str:
        return self.app._bg_color

    @property
    def text_bg(self) -> str:
        return self.app._text_bg

    @property
    def text_fg(self) -> str:
        return self.app._text_fg

    def on_batch_paths_changed(self, *_args) -> None:
        self.update_minimal_path_labels()

    def choose_folder(self) -> None:
        path = services_for(self.app).file_dialogs.askdirectory(
            initialdir=self.app.batch_input_var.get() or None,
            title="Select folder containing HDF5 files",
        )
        if path:
            self.app.batch_input_var.set(path)
            self.apply_input_defaults(Path(path))

    def choose_file(self) -> None:
        selected_paths = services_for(self.app).file_dialogs.askopenfilenames(
            filetypes=[
                ("AngioEye inputs", "*.h5 *.hdf5 *.holo *.txt *.zip"),
                ("HDF5 files", "*.h5 *.hdf5"),
                ("Holo files", "*.holo"),
                ("Stem lists", "*.txt"),
                ("Zip archives", "*.zip"),
                ("All files", "*.*"),
            ],
            initialdir=self.app.batch_input_var.get() or os.path.abspath("h5_example"),
            title="Select HDF5, .holo, or .zip input",
        )
        if not selected_paths:
            return

        input_paths = [Path(path) for path in selected_paths]
        if all(path.suffix.lower() == ".holo" for path in input_paths):
            self.apply_holo_inputs(input_paths)
            return
        if len(input_paths) > 1 and all(is_hdf5_path(path) for path in input_paths):
            self.apply_batch_input_files(input_paths)
            return
        if len(input_paths) != 1:
            services_for(self.app).dialogs.showwarning(
                "Unsupported selection",
                "Select multiple .holo files, multiple HDF5 files, "
                "or one .h5, .hdf5, or .zip file.",
            )
            return

        input_path = input_paths[0]
        if input_path.suffix.lower() == ".holo":
            self.apply_holo_inputs([input_path])
        elif input_path.suffix.lower() == ".txt":
            self.apply_holo_inputs([input_path])
        else:
            self.app.batch_input_var.set(str(input_path))
            self.apply_input_defaults(input_path)

    def choose_output(self) -> None:
        path = services_for(self.app).file_dialogs.askdirectory(
            initialdir=self.app.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.app.batch_output_var.set(path)

    def is_supported_file_input(self, input_path: Path) -> bool:
        return is_hdf5_path(input_path) or input_path.suffix.lower() == ".zip"

    def persist_trim_h5source(self) -> None:
        self.app._persist_trim_h5source()

    def run(self) -> None:
        self.app._reset_progress()
        input_selection = self.collect_input_selection()
        if (
            input_selection.convention == "legacy"
            and not input_selection.data_value
            and not input_selection.legacy_input_paths
        ):
            services_for(self.app).dialogs.showwarning(
                "Missing input",
                "Select a folder, HDF5 file, or .zip archive to process.",
            )
            return

        selection = self.collect_work_selection()
        if selection is None:
            return

        self.app._reset_batch_output("Starting batch run...\n")
        self.app._set_minimal_status("Preparing batch...")

        try:
            request = build_workflow_request(
                WorkflowRequestState(
                    input_selection=input_selection,
                    work_selection=selection,
                    output_options=self.collect_output_options(),
                ),
                zip_output_dir=self.zip_output_dir,
                output_filename_for_run=self.minimal_output_filename_for_run,
            )
            dispatch_result = self.dispatch_workflow(request, self.workflow_callbacks())
        except WorkflowInputError as exc:
            if exc.title == "Invalid input":
                self.app._log_batch(f"Error: {exc.message}")
            self.show_workflow_input_error(exc)
            return

        self.finish_dispatch_result(dispatch_result)

    def collect_input_selection(self) -> WorkflowInputSelection:
        convention = "holo" if self.uses_holo_input_convention() else "legacy"
        return WorkflowInputSelection(
            convention=convention,
            data_value=(self.app.batch_input_var.get() or "").strip(),
            legacy_input_paths=tuple(
                self.selected_batch_input_paths() if convention == "legacy" else ()
            ),
            holo_paths=tuple(self.selected_holo_paths() if convention == "holo" else ()),
        )

    def collect_output_options(self) -> WorkflowOutputOptions:
        return WorkflowOutputOptions(
            base_output_value=(self.app.batch_output_var.get() or "").strip(),
            zip_outputs=bool(self.app.batch_zip_var.get()),
            zip_name=self.app.batch_zip_name_var.get(),
            trim_source=self.trim_eyeflow_source(),
        )

    def collect_work_selection(self) -> WorkflowWorkSelection | None:
        return self.app.workflow_selection_controller.collect_selection()

    def dispatch_workflow(self, request, callbacks):
        compat_module = sys.modules.get("angio_eye")
        dispatch = getattr(compat_module, "dispatch_workflow", dispatch_workflow)
        return dispatch(request, callbacks)

    def workflow_callbacks(self) -> WorkflowCallbacks:
        return WorkflowCallbacks(
            log=self.app._log_batch,
            start_primary_progress=lambda units, status: self.app._start_progress(
                units,
                style_name=self.app._progress_primary_style,
                status_text=status,
            ),
            start_final_progress=lambda units, status: self.app._start_progress(
                units,
                style_name=self.app._progress_final_style,
                status_text=status,
            ),
            advance_progress=self.app._advance_progress,
            set_progress_units=self.app._set_progress_units,
            set_status=self.app._set_minimal_status,
            make_zip_progress_callback=lambda: make_zip_progress_callback(
                set_progress_units=self.app._set_progress_units,
                progress_base=self.app._progress_completed_units,
                log=self.app._log_batch,
                update_ui=self.update_ui,
            ),
            idle_callback=self.update_ui,
        )

    def show_workflow_input_error(self, error: WorkflowInputError) -> None:
        dialogs = services_for(self.app).dialogs
        if error.title == "Missing input":
            dialogs.showwarning(error.title, error.message)
        else:
            dialogs.showerror(error.title, error.message)
        self.app._set_minimal_status(error.status)

    def finish_dispatch_result(self, dispatch_result) -> None:
        workflow_result = dispatch_result.workflow_result
        if workflow_result is None:
            self.update_holo_status_labels()
            self.show_skipped_holo_warning(dispatch_result.skipped_holo_stems)
            self.app._set_minimal_status("Run skipped.")
            return

        self.app._set_progress_units(self.app._progress_total_units)
        completion = (
            "Completed with errors."
            if workflow_result.failures or workflow_result.zip_failed
            else "Completed."
        )
        self.app._log_batch(f"{completion} {workflow_result.summary_message}")

        if workflow_result.failures:
            self.app._set_minimal_status("Completed with errors.")
            self.app._show_batch_error_dialog(
                f"{len(workflow_result.failures)} failure(s). See log for details.\n\n"
                f"{workflow_result.summary_message}"
            )
        else:
            self.app._set_minimal_status(
                "Completed with errors."
                if workflow_result.zip_failed
                else "Process ended."
            )
        if workflow_result.zip_failed and workflow_result.zip_error:
            services_for(self.app).dialogs.showerror(
                "ZIP failed",
                f"Could not create ZIP archive: {workflow_result.zip_error}",
            )
        self.show_skipped_holo_warning(dispatch_result.skipped_holo_stems)

    def show_skipped_holo_warning(self, skipped_holo_stems: Sequence[str]) -> None:
        if not skipped_holo_stems:
            return
        services_for(self.app).dialogs.showwarning(
            "Skipped files",
            f"Skipped {len(skipped_holo_stems)} files: {', '.join(skipped_holo_stems)}",
        )

    def default_output_stem(self, input_path: Path) -> str:
        base_name = input_path.stem if input_path.is_file() else input_path.name
        return f"{base_name or 'output'}_angioeye"

    def default_archive_name(self, input_path: Path) -> str:
        return f"{self.default_output_stem(input_path)}.zip"

    def default_output_artifact_name(self, input_path: Path) -> str:
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            return self.default_archive_name(input_path)
        return f"{self.default_output_stem(input_path)}.h5"

    def apply_input_defaults(self, input_path: Path) -> None:
        self.app.input_convention_var.set("legacy")
        self.app.batch_input_paths = []
        self.app.holo_input_paths = []
        self.app.holo_input_var.set("")
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            output_dir = input_path.parent / self.default_output_stem(input_path)
        else:
            output_dir = input_path if input_path.is_dir() else input_path.parent

        self.app.batch_output_var.set(str(output_dir))
        self.app.batch_zip_name_var.set(self.default_archive_name(input_path))
        self.app.batch_zip_var.set(
            input_path.is_file() and input_path.suffix.lower() == ".zip"
        )
        self.update_holo_status_labels()
        self.app._reset_progress()
        self.app._set_minimal_status("Ready.")

    def apply_batch_input_files(self, input_paths: Sequence[Path]) -> None:
        paths = [path.expanduser() for path in input_paths]
        self.app.input_convention_var.set("legacy")
        self.app.batch_input_paths = paths
        self.app.holo_input_paths = []
        self.app.holo_input_var.set("")
        common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
        names = ", ".join(path.name for path in paths[:3])
        if len(paths) > 3:
            names = f"{names}, ..."
        self.app.batch_input_var.set(f"{len(paths)} HDF5 files selected: {names}")
        self.app.batch_output_var.set(str(common_parent))
        self.app.batch_zip_name_var.set("outputs.zip")
        self.app.batch_zip_var.set(False)
        self.update_holo_status_labels()
        self.app._reset_progress()
        self.app._set_minimal_status("Ready.")

    def apply_holo_inputs(self, holo_paths: Sequence[Path]) -> None:
        paths = [path.expanduser() for path in holo_paths]
        self.app.input_convention_var.set("holo")
        self.app.batch_input_paths = []
        self.app.holo_input_paths = paths
        self.app.holo_input_var.set(os.pathsep.join(str(path) for path in paths))
        if len(paths) == 1 and paths[0].suffix.lower() == ".txt":
            self.app.batch_input_var.set(str(paths[0]))
            self.app.batch_output_var.set("Auto: one *_AE folder per listed stem")
            self.app.batch_zip_name_var.set(self.default_archive_name(paths[0]))
        elif len(paths) == 1:
            self.app.batch_input_var.set(str(paths[0]))
            self.app.batch_output_var.set(str(holo_output_dir(paths[0])))
            self.app.batch_zip_name_var.set(self.default_archive_name(paths[0]))
        else:
            stems = ", ".join(path.stem for path in paths)
            self.app.batch_input_var.set(f"{len(paths)} .holo files selected: {stems}")
            self.app.batch_output_var.set("Auto: one *_AE folder per selected .holo")
            self.app.batch_zip_name_var.set("outputs.zip")
        self.app.batch_zip_var.set(False)

        self.update_holo_status_labels()
        self.app._reset_progress()
        self.app._set_minimal_status("Ready.")

    def selected_holo_paths(self) -> list[Path]:
        paths = getattr(self.app, "holo_input_paths", [])
        if paths:
            return list(paths)
        raw_value = (self.app.holo_input_var.get() or "").strip()
        return [Path(raw_value)] if raw_value else []

    def selected_batch_input_paths(self) -> list[Path]:
        return list(getattr(self.app, "batch_input_paths", []))

    def uses_holo_input_convention(self) -> bool:
        return self.app.input_convention_var.get() == "holo"

    def set_holo_status_visible(self, visible: bool) -> None:
        label_names = (
            "minimal_holo_status_label",
            "minimal_holo_output_label",
            "advanced_holo_status_label",
            "advanced_holo_output_label",
        )
        for label_name in label_names:
            label = getattr(self.app, label_name, None)
            if label is not None:
                if visible:
                    label.grid()
                else:
                    label.grid_remove()

    def set_holo_status_color(self, found: bool) -> None:
        color = "#3fb37f" if found else "#d65f5f"
        for label_name in ("minimal_holo_status_label", "advanced_holo_status_label"):
            label = getattr(self.app, label_name, None)
            if label is not None:
                label.configure(fg=color)

    def update_holo_status_labels(self) -> None:
        if not self.uses_holo_input_convention():
            self.app.holo_status_var.set("")
            self.app.holo_output_path_var.set("Output path: -")
            self.set_holo_status_visible(False)
            return

        holo_paths = self.selected_holo_paths()
        if not holo_paths:
            self.app.holo_status_var.set("")
            self.app.holo_output_path_var.set("Output path: -")
            self.set_holo_status_visible(False)
            return

        self.set_holo_status_visible(True)
        if len(holo_paths) == 1 and holo_paths[0].suffix.lower() == ".txt":
            self.update_stem_list_status_labels(holo_paths[0])
            return
        if len(holo_paths) == 1:
            output_text = f"Output path: {holo_output_dir(holo_paths[0])}"
        else:
            output_text = "Output paths: one *_AE folder per selected .holo"
        self.app.holo_output_path_var.set(output_text)

        missing_stems = [
            path.stem for path in holo_paths if find_holo_ef_h5(path) is None
        ]
        found_count = len(holo_paths) - len(missing_stems)
        status = f"{found_count}/{len(holo_paths)} EF found"
        if missing_stems:
            status += f" - missing: {', '.join(missing_stems)}"
        self.app.holo_status_var.set(status)
        self.set_holo_status_color(not missing_stems)

    def update_stem_list_status_labels(self, path: Path) -> None:
        self.set_holo_status_visible(True)
        try:
            stems = read_stem_list(path)
        except Exception as exc:  # noqa: BLE001
            self.app.holo_status_var.set(f"Stem list error: {exc}")
            self.app.holo_output_path_var.set("Output paths: -")
            self.set_holo_status_color(False)
            return
        self.app.holo_status_var.set(f"{len(stems)} stem(s) listed")
        self.app.holo_output_path_var.set(
            "Output paths: one *_AE folder per listed stem"
        )
        self.set_holo_status_color(True)

    def update_minimal_path_labels(self) -> None:
        holo_mode = self.uses_holo_input_convention()
        holo_paths = self.selected_holo_paths() if holo_mode else []
        raw_value = "" if holo_mode else (self.app.batch_input_var.get() or "").strip()
        if not raw_value:
            if (
                len(holo_paths) == 1
                and holo_paths[0].suffix.lower() == ".txt"
            ):
                self.app.minimal_input_path_var.set(str(holo_paths[0]))
                self.app.minimal_output_name_var.set(
                    "Output names: one *_AE.h5 per listed stem"
                )
            elif holo_paths:
                if len(holo_paths) == 1:
                    self.app.minimal_input_path_var.set(str(holo_paths[0]))
                    self.app.minimal_output_name_var.set(
                        f"Output name: {holo_output_filename(holo_paths[0])}"
                    )
                else:
                    stems = ", ".join(path.stem for path in holo_paths)
                    self.app.minimal_input_path_var.set(
                        f"{len(holo_paths)} .holo files selected: {stems}"
                    )
                    self.app.minimal_output_name_var.set(
                        "Output names: one *_AE.h5 per selected .holo"
                    )
            else:
                self.app.minimal_input_path_var.set("No input selected")
                self.app.minimal_output_name_var.set("Output name: -")
        else:
            input_path = Path(raw_value)
            self.app.minimal_input_path_var.set(str(input_path))
            self.app.minimal_output_name_var.set(
                f"Output name: {self.default_output_artifact_name(input_path)}"
            )

        output_value = (self.app.batch_output_var.get() or "").strip()
        self.app.minimal_output_path_var.set(
            output_value or "No output folder selected"
        )

    def minimal_output_filename_for_run(
        self,
        data_path: Path,
        inputs: Sequence[Path],
    ) -> str | None:
        if self.app.ui_mode != "minimal":
            return None
        if self.app.batch_zip_var.get():
            return None
        if len(inputs) != 1:
            return None
        if not data_path.is_file():
            return None
        if not is_hdf5_path(data_path):
            return None
        return self.default_output_artifact_name(data_path)

    def zip_output_dir(
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

    def trim_eyeflow_source(self) -> bool:
        persist_var = getattr(self.app, "_persist_eyeflow_data", None)
        return not bool(persist_var.get()) if persist_var is not None else True

    def update_ui(self) -> None:
        try:
            update_idletasks = getattr(self.app, "update_idletasks", None)
            if update_idletasks is not None:
                update_idletasks()
            self.app.update()
        except (AttributeError, tk.TclError):
            pass

    def reset_holo_output_dir(self, context: HoloInputContext) -> None:
        reset_holo_output_dir(context)

    def holo_dataset_dir(self, holo_path: Path) -> Path:
        return holo_dataset_dir(holo_path)

    def holo_ef_dir(self, holo_path: Path) -> Path:
        return holo_ef_dir(holo_path)

    def resolve_holo_context(self, holo_path: Path) -> HoloInputContext:
        return resolve_holo_context(holo_path)
