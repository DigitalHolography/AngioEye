from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from app_settings import normalize_pipeline_visibility
from pipelines import PipelineDescriptor, load_pipeline_catalog

from ..widgets import _Tooltip
from .library import LibraryController


class PipelineLibraryController(LibraryController):
    @property
    def summary_var(self):
        return self.app.pipeline_library_summary_var

    @property
    def bg_color(self) -> str:
        return self.app._bg_color

    def set_canvas_widgets(self, canvas, inner, window) -> None:
        self.app.pipeline_library_canvas = canvas
        self.app.pipeline_library_inner = inner
        self.app.pipeline_library_window = window

    def register(self) -> None:
        available, missing = load_pipeline_catalog()
        rows = sorted(
            [*available, *missing], key=lambda pipeline: pipeline.name.lower()
        )
        self.app.pipeline_registry = {p.name: p for p in available}
        self.app.pipeline_catalog = {p.name: p for p in rows}
        self.app.pipeline_rows = rows
        self.sync_visibility(rows)
        self.populate(rows)
        self.app._install_drop_targets()

    def sync_visibility(self, rows: list[PipelineDescriptor]) -> None:
        visibility, changed = normalize_pipeline_visibility(
            (pipeline.name for pipeline in rows),
            self.app.settings_store.load_pipeline_visibility(),
        )
        for pipeline in rows:
            if not pipeline.available and visibility.get(pipeline.name, False):
                visibility[pipeline.name] = False
                changed = True
        self.app.pipeline_visibility = visibility
        if changed:
            self.persist_visibility()

    def persist_visibility(self) -> None:
        try:
            self.app.settings_store.save_pipeline_visibility(
                self.app.pipeline_visibility
            )
        except OSError as exc:
            self.app._show_settings_warning(
                "Settings not saved",
                f"Could not save pipeline selection preferences:\n{exc}",
            )

    def set_visibility(self, name: str, visible: bool) -> None:
        pipeline = self.app.pipeline_catalog.get(name)
        if pipeline is not None and not pipeline.available:
            visible = False
        if self.app.pipeline_visibility.get(name) == visible:
            return
        self.app.pipeline_visibility[name] = visible
        self.persist_visibility()
        self.update_summary()

    def set_all(self, visible: bool) -> None:
        changed = False
        target_values = {
            pipeline.name: visible and pipeline.available
            for pipeline in self.app.pipeline_rows
        }
        for name, target_value in target_values.items():
            if self.app.pipeline_visibility.get(name) != target_value:
                self.app.pipeline_visibility[name] = target_value
                changed = True
        if not changed:
            return
        for name, var in self.app.pipeline_visibility_vars.items():
            var.set(self.app.pipeline_visibility.get(name, False))
        self.persist_visibility()
        self.update_summary()

    def select_all(self) -> None:
        self.set_all(True)

    def deselect_all(self) -> None:
        self.set_all(False)

    def refresh(self) -> None:
        self.register()
        self.app.postprocess_library_controller.register()

    def open_folder(self) -> None:
        self.open_folder_path(self.package_folder("pipelines"), "Pipeline folder")

    def status_text(self, pipeline: PipelineDescriptor) -> str:
        if pipeline.available:
            return "Available"
        if pipeline.missing_deps:
            return f"Missing deps: {', '.join(pipeline.missing_deps)}"
        return "Unavailable"

    def populate(self, rows: list[PipelineDescriptor]) -> None:
        for child in self.app.pipeline_library_inner.winfo_children():
            child.destroy()
        self.app.pipeline_visibility_vars = {}
        self.app.pipeline_library_inner.columnconfigure(0, weight=1)

        selected_header = ttk.Label(self.app.pipeline_library_inner, text="Selected")
        selected_header.grid(row=0, column=0, sticky="w", pady=(0, 6))
        status_header = ttk.Label(self.app.pipeline_library_inner, text="Status")
        status_header.grid(row=0, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
        self.bind_mousewheel(selected_header, self.app.pipeline_library_canvas)
        self.bind_mousewheel(status_header, self.app.pipeline_library_canvas)

        for idx, pipeline in enumerate(rows, start=1):
            is_available = getattr(pipeline, "available", True)
            var = tk.BooleanVar(
                value=self.app.pipeline_visibility.get(pipeline.name, False)
                and is_available
            )
            check = ttk.Checkbutton(
                self.app.pipeline_library_inner,
                text=pipeline.name,
                variable=var,
                state="normal" if is_available else "disabled",
                command=lambda name=pipeline.name, visible_var=var: (
                    self.set_visibility(name, visible_var.get())
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))
            self.bind_mousewheel(check, self.app.pipeline_library_canvas)

            status = ttk.Label(
                self.app.pipeline_library_inner,
                text=self.status_text(pipeline),
            )
            status.grid(row=idx, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
            self.bind_mousewheel(status, self.app.pipeline_library_canvas)

            tip_text = self.descriptor_tooltip_text(pipeline)
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

            self.app.pipeline_visibility_vars[pipeline.name] = var

        self.update_summary()

    def update_summary(self) -> None:
        selected_count = sum(
            1
            for pipeline in self.app.pipeline_rows
            if pipeline.available
            and self.app.pipeline_visibility.get(pipeline.name, False)
        )
        available_count = sum(
            1 for pipeline in self.app.pipeline_rows if pipeline.available
        )
        self.app.pipeline_library_summary_var.set(
            f"Selected: {selected_count}/{available_count}"
        )
