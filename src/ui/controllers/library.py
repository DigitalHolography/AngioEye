from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

from postprocess import format_required_pipeline_options

from ..services import services_for
from .base import ViewController


class LibraryController(ViewController):
    STATUS_DEFAULT_WRAP_PX = 420
    STATUS_MIN_WRAP_PX = 220

    @staticmethod
    def mousewheel_scroll_units(event: tk.Event) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            steps = max(1, abs(delta) // 120) if abs(delta) >= 120 else 1
            return -steps if delta > 0 else steps

        button = getattr(event, "num", None)
        if button == 4:
            return -1
        if button == 5:
            return 1
        return 0

    def bind_mousewheel(self, widget, canvas) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(
                sequence,
                lambda event, target_canvas=canvas: self.on_vertical_mousewheel(
                    event, target_canvas
                ),
                add="+",
            )

    def on_vertical_mousewheel(self, event: tk.Event, canvas) -> str | None:
        scroll_units = self.mousewheel_scroll_units(event)
        if not scroll_units:
            return None
        canvas.yview_scroll(scroll_units, "units")
        return "break"

    def configure_status_column_wrapping(
        self,
        inner: tk.Widget,
        canvas: tk.Canvas,
        status_labels: list[tk.Widget],
    ) -> None:
        canvas._library_status_wrap_inner = inner
        canvas._library_status_wrap_labels = status_labels

        def update_wraplength(_event: tk.Event | None = None) -> None:
            current_inner = getattr(canvas, "_library_status_wrap_inner", inner)
            current_labels = getattr(canvas, "_library_status_wrap_labels", [])
            canvas_width = canvas.winfo_width()
            if canvas_width <= 1:
                wraplength = self.STATUS_DEFAULT_WRAP_PX
            else:
                selected_column_width = current_inner.grid_bbox(column=0)[2]
                available_width = canvas_width - selected_column_width - 36
                wraplength = max(self.STATUS_MIN_WRAP_PX, available_width)
                wraplength = min(self.STATUS_DEFAULT_WRAP_PX, wraplength)
            for label in current_labels:
                if label.winfo_exists():
                    label.configure(wraplength=wraplength)

        update_wraplength()
        if not getattr(canvas, "_library_status_wrap_bound", False):
            canvas.bind("<Configure>", update_wraplength, add="+")
            canvas._library_status_wrap_bound = True

    def descriptor_tooltip_text(self, descriptor) -> str:
        parts: list[str] = []
        description = getattr(descriptor, "description", "")
        if description:
            parts.append(description)
        required_pipelines = format_required_pipeline_options(descriptor)
        if required_pipelines:
            parts.append(f"Requires pipelines: {required_pipelines}")
        missing_pipelines = getattr(descriptor, "missing_pipelines", [])
        if missing_pipelines:
            parts.append(
                "Unavailable until these pipelines are available: "
                f"{', '.join(missing_pipelines)}"
            )
        missing_deps = getattr(descriptor, "missing_deps", []) or getattr(
            descriptor, "requires", []
        )
        if missing_deps:
            parts.append(f"Install: {', '.join(missing_deps)}")
        return "\n".join(parts)

    def package_folder(self, package_name: str) -> Path | None:
        module = sys.modules.get(package_name)
        module_path = getattr(module, "__path__", None)
        if module_path:
            for path_value in module_path:
                folder = Path(path_value).resolve()
                if folder.is_dir():
                    return folder

        module_file = getattr(module, "__file__", None)
        if module_file:
            folder = Path(module_file).resolve().parent
            if folder.is_dir():
                return folder

        for root in self.app._resource_roots():
            folder = root / package_name
            if folder.is_dir():
                return folder
        return None

    def open_folder_path(self, folder: Path | None, label: str) -> None:
        if folder is None or not folder.is_dir():
            services_for(self.app).dialogs.showerror(
                label,
                f"Could not find the {label.lower()}.",
            )
            return
        try:
            services_for(self.app).folder_opener.open_folder(folder)
        except Exception as exc:  # noqa: BLE001
            services_for(self.app).dialogs.showerror(
                label,
                f"Could not open folder:\n{folder}\n\n{exc}",
            )
