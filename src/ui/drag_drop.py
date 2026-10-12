import tkinter as tk
from collections.abc import Sequence
from pathlib import Path

from .compat import DND_FILES
from .services import services_for


class DragDropMixin:
    def _install_drop_targets(self) -> None:
        if DND_FILES is not None and getattr(self, "_drag_drop_available", True):
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

    def _handle_dropped_paths(self, paths: Sequence[Path]) -> bool:
        if not paths or not all(
            path.is_dir()
            or (path.is_file() and self.run_controller.is_supported_file_input(path))
            for path in paths
        ):
            return False
        if len(paths) > 1 and any(
            not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}
            for path in paths
        ):
            return False
        self.run_controller.apply_inputs(paths)
        self._log_batch(f"[INPUT] Drag and drop: {len(paths)} input(s)")
        return True

    def _on_input_drop(self, event) -> None:
        raw_data = getattr(event, "data", "")
        try:
            values = self.tk.splitlist(raw_data)
        except tk.TclError:
            values = (raw_data,)
        paths = [Path(value) for value in values if value]
        if not self._handle_dropped_paths(paths):
            services_for(self).dialogs.showwarning(
                "Unsupported drop",
                "Drop EyeFlow HDF5 files, one folder, or one ZIP archive.",
            )
