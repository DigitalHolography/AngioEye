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
            widget.dnd_bind("<<DropEnter>>", self._on_input_drag)
            widget.dnd_bind("<<DropPosition>>", self._on_input_drag)
            widget.dnd_bind("<<Drop>>", self._on_input_drop)
        except (AttributeError, tk.TclError):
            pass
        for child in widget.winfo_children():
            self._register_drop_target_tree(child)

    def _handle_dropped_paths(self, paths: Sequence[Path]) -> bool:
        if not self.run_controller.apply_inputs(paths):
            return False
        self._log_batch(f"[INPUT] Drag and drop: {len(paths)} input(s)")
        return True

    def _on_input_drag(self, _event) -> str:
        return "refuse_drop" if getattr(self, "_batch_running", False) else "copy"

    def _on_input_drop(self, event) -> str:
        if getattr(self, "_batch_running", False):
            return "refuse_drop"
        raw_data = getattr(event, "data", "")
        try:
            values = self.tk.splitlist(raw_data)
        except tk.TclError:
            services_for(self).dialogs.showwarning(
                "Unsupported drop",
                "Could not read the dropped paths. Drop .holo recordings or one folder containing .holo recordings.",
            )
            return "refuse_drop"
        paths = [Path(value) for value in values if value]
        return "copy" if self._handle_dropped_paths(paths) else "refuse_drop"
