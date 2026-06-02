import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import messagebox

from input_output import is_hdf5_path

from .compat import DND_FILES

class DragDropMixin:
    def _install_drop_targets(self) -> None:
        if DND_FILES is None:
            return
        self._register_drop_target_tree(self)

    def _register_drop_target_tree(self, widget: tk.Misc) -> None:
        if DND_FILES is None:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_input_drop)
        except (AttributeError, tk.TclError):
            pass

        for child in widget.winfo_children():
            self._register_drop_target_tree(child)

    def _handle_dropped_paths(self, dropped_paths: Sequence[Path]) -> bool:
        if dropped_paths and all(
            path.is_file() and path.suffix.lower() == ".holo"
            for path in dropped_paths
        ):
            self._apply_holo_inputs(dropped_paths)
            self._log_batch(
                f"[INPUT] Drag and drop -> {len(dropped_paths)} .holo file(s)"
            )
            return True

        for dropped_path in dropped_paths:
            if dropped_path.is_file() and (
                is_hdf5_path(dropped_path) or dropped_path.suffix.lower() == ".zip"
            ):
                self.input_convention_var.set("legacy")
                self.batch_input_var.set(str(dropped_path))
                self._apply_input_defaults(dropped_path)
                self._log_batch(f"[INPUT] Drag and drop -> {dropped_path}")
                return True
        return False

    def _on_input_drop(self, event) -> None:
        raw_data = getattr(event, "data", "")
        try:
            dropped_values = self.tk.splitlist(raw_data)
        except tk.TclError:
            dropped_values = (raw_data,)

        dropped_paths = [Path(value) for value in dropped_values if value]
        if self._handle_dropped_paths(dropped_paths):
            return

        messagebox.showwarning(
            "Unsupported drop",
            "Drop .holo file(s), or a single .h5, .hdf5, or .zip file.",
        )
