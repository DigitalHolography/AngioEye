import os
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox

from input_output import is_hdf5_path
from workflows import HoloInputContext
from workflows import dataset_dir as holo_dataset_dir
from workflows import ef_dir as holo_ef_dir
from workflows import find_ef_h5 as find_holo_ef_h5
from workflows import output_dir as holo_output_dir
from workflows import output_filename as holo_output_filename
from workflows import reset_output_dir as reset_holo_output_dir
from workflows import resolve_context as resolve_holo_context

class InputStateMixin:
    def _on_batch_paths_changed(self, *_args) -> None:
        self._update_minimal_path_labels()

    def _default_output_stem(self, input_path: Path) -> str:
        if input_path.is_file():
            base_name = input_path.stem
        else:
            base_name = input_path.name
        base_name = base_name or "output"
        return f"{base_name}_angioeye"

    def _default_archive_name(self, input_path: Path) -> str:
        return f"{self._default_output_stem(input_path)}.zip"

    def _default_output_artifact_name(self, input_path: Path) -> str:
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            return self._default_archive_name(input_path)
        return f"{self._default_output_stem(input_path)}.h5"

    def _holo_dataset_dir(self, holo_path: Path) -> Path:
        return holo_dataset_dir(holo_path)

    def _holo_ef_dir(self, holo_path: Path) -> Path:
        return holo_ef_dir(holo_path)

    def _holo_output_dir(self, holo_path: Path) -> Path:
        return holo_output_dir(holo_path)

    def _holo_output_filename(self, holo_path: Path) -> str:
        return holo_output_filename(holo_path)

    def _find_holo_ef_h5(self, holo_path: Path) -> Path | None:
        return find_holo_ef_h5(holo_path)

    def _resolve_holo_context(self, holo_path: Path) -> HoloInputContext:
        return resolve_holo_context(holo_path)

    def _set_holo_status_visible(self, visible: bool) -> None:
        label_names = (
            "minimal_holo_status_label",
            "minimal_holo_output_label",
            "advanced_holo_status_label",
            "advanced_holo_output_label",
        )
        for label_name in label_names:
            label = getattr(self, label_name, None)
            if label is not None:
                if visible:
                    label.grid()
                else:
                    label.grid_remove()

    def _set_holo_status_color(self, found: bool) -> None:
        color = "#3fb37f" if found else "#d65f5f"
        for label_name in ("minimal_holo_status_label", "advanced_holo_status_label"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.configure(fg=color)

    def _update_holo_status_labels(self) -> None:
        if not self._uses_holo_input_convention():
            self.holo_status_var.set("")
            self.holo_output_path_var.set("Output path: -")
            self._set_holo_status_visible(False)
            return

        holo_paths = self._selected_holo_paths()
        if not holo_paths:
            self.holo_status_var.set("")
            self.holo_output_path_var.set("Output path: -")
            self._set_holo_status_visible(False)
            return

        self._set_holo_status_visible(True)
        if len(holo_paths) == 1:
            output_text = f"Output path: {self._holo_output_dir(holo_paths[0])}"
        else:
            output_text = "Output paths: one *_AE folder per selected .holo"
        self.holo_output_path_var.set(output_text)

        missing_stems = [
            path.stem for path in holo_paths if self._find_holo_ef_h5(path) is None
        ]
        found_count = len(holo_paths) - len(missing_stems)
        status = f"{found_count}/{len(holo_paths)} EF found"
        if missing_stems:
            status += f" - missing: {', '.join(missing_stems)}"
        self.holo_status_var.set(status)
        self._set_holo_status_color(not missing_stems)

    def _update_minimal_path_labels(self) -> None:
        holo_mode = self._uses_holo_input_convention()
        holo_paths = self._selected_holo_paths() if holo_mode else []
        raw_value = "" if holo_mode else (self.batch_input_var.get() or "").strip()
        if not raw_value:
            if holo_paths:
                if len(holo_paths) == 1:
                    self.minimal_input_path_var.set(str(holo_paths[0]))
                    self.minimal_output_name_var.set(
                        f"Output name: {self._holo_output_filename(holo_paths[0])}"
                    )
                else:
                    stems = ", ".join(path.stem for path in holo_paths)
                    self.minimal_input_path_var.set(
                        f"{len(holo_paths)} .holo files selected: {stems}"
                    )
                    self.minimal_output_name_var.set(
                        "Output names: one *_AE.h5 per selected .holo"
                    )
            else:
                self.minimal_input_path_var.set("No input selected")
                self.minimal_output_name_var.set("Output name: -")
        else:
            input_path = Path(raw_value)
            self.minimal_input_path_var.set(str(input_path))
            self.minimal_output_name_var.set(
                f"Output name: {self._default_output_artifact_name(input_path)}"
            )

        output_value = (self.batch_output_var.get() or "").strip()
        self.minimal_output_path_var.set(output_value or "No output folder selected")

    def _apply_input_defaults(self, input_path: Path) -> None:
        self.input_convention_var.set("legacy")
        self.batch_input_paths = []
        self.holo_input_paths = []
        self.holo_input_var.set("")
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            output_dir = input_path.parent / self._default_output_stem(input_path)
        else:
            output_dir = input_path if input_path.is_dir() else input_path.parent

        self.batch_output_var.set(str(output_dir))
        self.batch_zip_name_var.set(self._default_archive_name(input_path))
        self.batch_zip_var.set(
            input_path.is_file() and input_path.suffix.lower() == ".zip"
        )
        self._update_holo_status_labels()
        self._reset_progress()
        self._set_minimal_status("Ready.")

    def choose_batch_folder(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_input_var.get() or None,
            title="Select folder containing HDF5 files",
        )
        if path:
            self.batch_input_var.set(path)
            self._apply_input_defaults(Path(path))

    def choose_batch_file(self) -> None:
        selected_paths = filedialog.askopenfilenames(
            filetypes=[
                ("AngioEye inputs", "*.h5 *.hdf5 *.holo *.zip"),
                ("HDF5 files", "*.h5 *.hdf5"),
                ("Holo files", "*.holo"),
                ("Zip archives", "*.zip"),
                ("All files", "*.*"),
            ],
            initialdir=self.batch_input_var.get() or os.path.abspath("h5_example"),
            title="Select HDF5, .holo, or .zip input",
        )
        if selected_paths:
            input_paths = [Path(path) for path in selected_paths]
            if all(path.suffix.lower() == ".holo" for path in input_paths):
                self._apply_holo_inputs(input_paths)
                return
            if len(input_paths) > 1 and all(is_hdf5_path(path) for path in input_paths):
                self._apply_batch_input_files(input_paths)
                return
            if len(input_paths) != 1:
                messagebox.showwarning(
                    "Unsupported selection",
                    "Select multiple .holo files, multiple HDF5 files, "
                    "or one .h5, .hdf5, or .zip file.",
                )
                return

            input_path = input_paths[0]
            if input_path.suffix.lower() == ".holo":
                self._apply_holo_input(input_path)
            else:
                self.batch_input_var.set(str(input_path))
                self._apply_input_defaults(input_path)

    def _apply_batch_input_files(self, input_paths: Sequence[Path]) -> None:
        paths = [path.expanduser() for path in input_paths]
        self.input_convention_var.set("legacy")
        self.batch_input_paths = paths
        self.holo_input_paths = []
        self.holo_input_var.set("")
        common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
        names = ", ".join(path.name for path in paths[:3])
        if len(paths) > 3:
            names = f"{names}, ..."
        self.batch_input_var.set(f"{len(paths)} HDF5 files selected: {names}")
        self.batch_output_var.set(str(common_parent))
        self.batch_zip_name_var.set("outputs.zip")
        self.batch_zip_var.set(False)
        self._update_holo_status_labels()
        self._reset_progress()
        self._set_minimal_status("Ready.")

    def _apply_holo_input(self, holo_path: Path) -> None:
        self._apply_holo_inputs([holo_path])

    def _apply_holo_inputs(self, holo_paths: Sequence[Path]) -> None:
        paths = [path.expanduser() for path in holo_paths]
        self.input_convention_var.set("holo")
        self.batch_input_paths = []
        self.holo_input_paths = paths
        self.holo_input_var.set(os.pathsep.join(str(path) for path in paths))
        if len(paths) == 1:
            self.batch_input_var.set(str(paths[0]))
            self.batch_output_var.set(str(self._holo_output_dir(paths[0])))
            self.batch_zip_name_var.set(self._default_archive_name(paths[0]))
        else:
            stems = ", ".join(path.stem for path in paths)
            self.batch_input_var.set(f"{len(paths)} .holo files selected: {stems}")
            self.batch_output_var.set("Auto: one *_AE folder per selected .holo")
            self.batch_zip_name_var.set("outputs.zip")
        self.batch_zip_var.set(False)

        self._update_holo_status_labels()
        self._reset_progress()
        self._set_minimal_status("Ready.")

    def _selected_holo_paths(self) -> list[Path]:
        paths = getattr(self, "holo_input_paths", [])
        if paths:
            return list(paths)

        raw_value = (self.holo_input_var.get() or "").strip()
        return [Path(raw_value)] if raw_value else []

    def _selected_batch_input_paths(self) -> list[Path]:
        return list(getattr(self, "batch_input_paths", []))

    def choose_batch_output(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.batch_output_var.set(path)

    def _uses_holo_input_convention(self) -> bool:
        return self.input_convention_var.get() == "holo"

    def _reset_holo_output_dir(self, context: HoloInputContext) -> None:
        reset_holo_output_dir(context)
