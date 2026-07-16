from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from postprocess import format_required_pipeline_options, required_options_for

from ..services import services_for
from .base import ViewController


class LibraryController(ViewController):
    _STATUS_COLUMN_PADDING = (12, 12)

    def configure_library_columns(self, inner, *, row_count: int = 1) -> None:
        inner.columnconfigure(0, weight=1, minsize=180)
        inner.columnconfigure(1, weight=0, minsize=6)
        inner.columnconfigure(2, weight=0, minsize=0)
        divider_slot = ttk.Frame(inner, cursor="sb_h_double_arrow")
        divider_slot.grid(
            row=0,
            column=1,
            rowspan=max(1, row_count),
            sticky="nsew",
        )
        divider_slot.grid_propagate(False)
        divider_slot._library_resize_inner = inner
        for relx in (0.35, 0.65):
            handle_line = ttk.Separator(divider_slot, orient="vertical")
            handle_line._library_resize_inner = inner
            handle_line.place(
                relx=relx,
                rely=0.5,
                anchor="center",
                width=1,
                height=22,
            )
            handle_line.bind("<ButtonPress-1>", self._start_library_column_resize)
            handle_line.bind("<B1-Motion>", self._resize_library_columns)
            handle_line.bind("<ButtonRelease-1>", self._finish_library_column_resize)
        divider_slot.bind("<ButtonPress-1>", self._start_library_column_resize)
        divider_slot.bind("<B1-Motion>", self._resize_library_columns)
        divider_slot.bind("<ButtonRelease-1>", self._finish_library_column_resize)

    def _start_library_column_resize(self, event) -> None:
        inner = getattr(event.widget, "_library_resize_inner", event.widget.master)
        inner.update_idletasks()
        name_width = inner.grid_bbox(0, 0)[2]
        status_width = max(1, self._library_column_requested_width(inner, 2))
        self._library_resize_state = (
            inner,
            event.x_root,
            name_width,
            status_width,
        )

    def _resize_library_columns(self, event) -> None:
        state = getattr(self, "_library_resize_state", None)
        if state is None:
            return
        inner, start_x, start_width, status_min_width = state
        divider_width = 6
        new_width = start_width + event.x_root - start_x
        max_width = max(180, inner.winfo_width() - status_min_width - divider_width)
        new_width = min(max(180, new_width), max_width)
        inner.columnconfigure(0, weight=0, minsize=new_width)
        inner.columnconfigure(2, weight=1, minsize=status_min_width)

    @staticmethod
    def _library_column_requested_width(inner, column: int) -> int:
        widths = []
        for widget in inner.grid_slaves(column=column):
            grid_info = widget.grid_info()
            padx = grid_info.get("padx", 0)
            if isinstance(padx, (tuple, list)):
                padding = sum(int(value) for value in padx)
            elif isinstance(padx, str) and " " in padx:
                padding = sum(int(value) for value in padx.split())
            else:
                padding = int(padx or 0) * 2
            widths.append(widget.winfo_reqwidth() + padding)
        return max(widths, default=0)

    def _finish_library_column_resize(self, _event) -> None:
        self._library_resize_state = None

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

    def descriptor_tooltip_text(self, descriptor) -> str:
        parts: list[str] = []
        description = getattr(descriptor, "description", "")
        if description:
            parts.append(description)
        required_pipelines = format_required_pipeline_options(descriptor)
        if required_pipelines:
            parts.append(f"Requires pipelines: {required_pipelines}")
        required_options = self.format_required_options(descriptor)
        if required_options:
            parts.append(f"Requires options: {required_options}")
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

    @staticmethod
    def format_required_options(descriptor) -> str:
        return ", ".join(required_options_for(descriptor))

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
