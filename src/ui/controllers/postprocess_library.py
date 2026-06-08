from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from app_settings import normalize_postprocess_visibility
from postprocess import (
    PostprocessDescriptor,
    format_required_pipeline_options,
    load_postprocess_catalog,
)

from ..widgets import _Tooltip
from .library import LibraryController


class PostprocessLibraryController(LibraryController):
    @property
    def summary_var(self):
        return self.app.postprocess_library_summary_var

    @property
    def bg_color(self) -> str:
        return self.app._bg_color

    def set_canvas_widgets(self, canvas, inner, window) -> None:
        self.app.postprocess_library_canvas = canvas
        self.app.postprocess_library_inner = inner
        self.app.postprocess_library_window = window

    def register(self) -> None:
        available, missing = load_postprocess_catalog()
        rows = sorted(
            [*available, *missing], key=lambda postprocess: postprocess.name.lower()
        )
        self.app.postprocess_registry = {p.name: p for p in available}
        self.app.postprocess_catalog = {p.name: p for p in rows}
        self.app.postprocess_rows = rows
        self.sync_visibility(rows)
        self.populate(rows)
        self.app._install_drop_targets()

    def sync_visibility(self, rows: list[PostprocessDescriptor]) -> None:
        visibility, changed = normalize_postprocess_visibility(
            (postprocess.name for postprocess in rows),
            self.app.settings_store.load_postprocess_visibility(),
        )
        for postprocess in rows:
            if not postprocess.available and visibility.get(postprocess.name, False):
                visibility[postprocess.name] = False
                changed = True
        self.app.postprocess_visibility = visibility
        if changed:
            self.persist_visibility()

    def persist_visibility(self) -> None:
        try:
            self.app.settings_store.save_postprocess_visibility(
                self.app.postprocess_visibility
            )
        except OSError as exc:
            self.app._show_settings_warning(
                "Settings not saved",
                f"Could not save postprocess selection preferences:\n{exc}",
            )

    def set_visibility(self, name: str, visible: bool) -> None:
        postprocess = self.app.postprocess_catalog.get(name)
        if postprocess is not None and not postprocess.available:
            visible = False
        if self.app.postprocess_visibility.get(name) == visible:
            return
        self.app.postprocess_visibility[name] = visible
        self.persist_visibility()
        self.update_summary()

    def set_all(self, visible: bool) -> None:
        changed = False
        target_values = {
            postprocess.name: visible and postprocess.available
            for postprocess in self.app.postprocess_rows
        }
        for name, target_value in target_values.items():
            if self.app.postprocess_visibility.get(name) != target_value:
                self.app.postprocess_visibility[name] = target_value
                changed = True
        if not changed:
            return
        for name, var in self.app.postprocess_visibility_vars.items():
            var.set(self.app.postprocess_visibility.get(name, False))
        self.persist_visibility()
        self.update_summary()

    def select_all(self) -> None:
        self.set_all(True)

    def deselect_all(self) -> None:
        self.set_all(False)

    def refresh(self) -> None:
        self.register()

    def open_folder(self) -> None:
        self.open_folder_path(
            self.package_folder("postprocess"),
            "Postprocess folder",
        )

    def status_text(self, postprocess: PostprocessDescriptor) -> str:
        if not postprocess.available:
            if postprocess.missing_pipelines:
                return f"Missing pipelines: {', '.join(postprocess.missing_pipelines)}"
            if postprocess.missing_deps:
                return f"Missing deps: {', '.join(postprocess.missing_deps)}"
            return "Unavailable"
        required_pipelines = format_required_pipeline_options(postprocess)
        if required_pipelines:
            return f"Requires: {required_pipelines}"
        return "Available"

    def populate(self, rows: list[PostprocessDescriptor]) -> None:
        for child in self.app.postprocess_library_inner.winfo_children():
            child.destroy()
        self.app.postprocess_visibility_vars = {}
        self.app.postprocess_library_inner.columnconfigure(0, weight=1)
        status_labels: list[tk.Widget] = []

        selected_header = ttk.Label(self.app.postprocess_library_inner, text="Selected")
        selected_header.grid(row=0, column=0, sticky="w", pady=(0, 6))
        status_header = ttk.Label(self.app.postprocess_library_inner, text="Status")
        status_header.grid(row=0, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
        self.bind_mousewheel(selected_header, self.app.postprocess_library_canvas)
        self.bind_mousewheel(status_header, self.app.postprocess_library_canvas)

        for idx, postprocess in enumerate(rows, start=1):
            is_available = getattr(postprocess, "available", True)
            var = tk.BooleanVar(
                value=self.app.postprocess_visibility.get(postprocess.name, False)
                and is_available
            )
            check = ttk.Checkbutton(
                self.app.postprocess_library_inner,
                text=postprocess.name,
                variable=var,
                state="normal" if is_available else "disabled",
                command=lambda name=postprocess.name, visible_var=var: (
                    self.set_visibility(name, visible_var.get())
                ),
            )
            check.grid(row=idx, column=0, sticky="nw", pady=(0, 6))
            self.bind_mousewheel(check, self.app.postprocess_library_canvas)

            status = ttk.Label(
                self.app.postprocess_library_inner,
                text=self.status_text(postprocess),
            )
            status.grid(row=idx, column=1, sticky="nw", padx=(12, 18), pady=(0, 6))
            self.bind_mousewheel(status, self.app.postprocess_library_canvas)
            status_labels.append(status)

            tip_text = self.descriptor_tooltip_text(postprocess)
            if tip_text:
                _Tooltip(
                    check,
                    tip_text,
                    bg=self.app._surface_color,
                    fg=self.app._text_fg,
                )
                _Tooltip(
                    status,
                    tip_text,
                    bg=self.app._surface_color,
                    fg=self.app._text_fg,
                )

            self.app.postprocess_visibility_vars[postprocess.name] = var

        self.configure_status_column_wrapping(
            self.app.postprocess_library_inner,
            self.app.postprocess_library_canvas,
            status_labels,
        )
        self.update_summary()

    def update_summary(self) -> None:
        selected_count = sum(
            1
            for postprocess in self.app.postprocess_rows
            if postprocess.available
            and self.app.postprocess_visibility.get(postprocess.name, False)
        )
        available_count = sum(
            1 for postprocess in self.app.postprocess_rows if postprocess.available
        )
        self.app.postprocess_library_summary_var.set(
            f"Selected: {selected_count}/{available_count}"
        )
