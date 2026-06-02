import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from app_settings import normalize_pipeline_visibility
from pipelines import PipelineDescriptor, load_pipeline_catalog

from .services import services_for
from .widgets import _Tooltip


class PipelineLibraryTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        app = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(
            self,
            text="Select the pipelines to run. "
            "This preference is saved between app launches.",
        ).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        controls.columnconfigure(4, weight=1)
        ttk.Button(
            controls,
            text="Select all",
            command=app.select_all_pipelines,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Deselect all",
            command=app.deselect_all_pipelines,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload pipelines",
            command=app.refresh_pipeline_catalog,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Open folder",
            command=app.open_pipeline_folder,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(controls, textvariable=app.pipeline_library_summary_var).grid(
            row=0, column=4, sticky="e"
        )

        library_container = ttk.Frame(self)
        library_container.grid(row=2, column=0, sticky="nsew")
        library_container.columnconfigure(0, weight=1)
        library_container.rowconfigure(0, weight=1)

        app.pipeline_library_canvas = tk.Canvas(
            library_container, highlightthickness=0, bg=app._bg_color
        )
        app.pipeline_library_canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_container,
            orient="vertical",
            command=app.pipeline_library_canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        app.pipeline_library_canvas.configure(yscrollcommand=library_scroll.set)
        app.pipeline_library_inner = ttk.Frame(app.pipeline_library_canvas)
        app.pipeline_library_window = app.pipeline_library_canvas.create_window(
            (0, 0), window=app.pipeline_library_inner, anchor="nw"
        )
        app.pipeline_library_inner.bind(
            "<Configure>",
            lambda _evt: app.pipeline_library_canvas.configure(
                scrollregion=app.pipeline_library_canvas.bbox("all")
            ),
        )
        app.pipeline_library_canvas.bind(
            "<Configure>",
            lambda evt: app.pipeline_library_canvas.itemconfigure(
                app.pipeline_library_window, width=evt.width
            ),
        )
        app._bind_vertical_mousewheel(
            app.pipeline_library_canvas, app.pipeline_library_canvas
        )
        app._bind_vertical_mousewheel(
            app.pipeline_library_inner, app.pipeline_library_canvas
        )
        app._bind_vertical_mousewheel(library_scroll, app.pipeline_library_canvas)


class PipelineLibraryMixin:
    def _build_pipeline_library_tab(self, parent: ttk.Frame) -> None:
        tab = PipelineLibraryTab(parent, self)
        tab.pack(fill="both", expand=True)
        self.pipeline_library_tab = tab

    def _bind_vertical_mousewheel(self, widget: tk.Misc, canvas: tk.Canvas) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(
                sequence,
                lambda event, target_canvas=canvas: self._on_vertical_mousewheel(
                    event, target_canvas
                ),
                add="+",
            )

    @staticmethod
    def _mousewheel_scroll_units(event: tk.Event) -> int:
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

    def _on_vertical_mousewheel(self, event: tk.Event, canvas: tk.Canvas) -> str | None:
        scroll_units = self._mousewheel_scroll_units(event)
        if not scroll_units:
            return None
        canvas.yview_scroll(scroll_units, "units")
        return "break"

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        rows = sorted(
            [*available, *missing], key=lambda pipeline: pipeline.name.lower()
        )
        self.pipeline_registry = {p.name: p for p in available}
        self.pipeline_catalog = {p.name: p for p in rows}
        self.pipeline_rows = rows
        self._sync_pipeline_visibility(rows)
        self._populate_pipeline_library(rows)
        self._install_drop_targets()

    def _descriptor_tooltip_text(self, descriptor) -> str:
        parts: list[str] = []
        description = getattr(descriptor, "description", "")
        if description:
            parts.append(description)
        required_pipelines = getattr(descriptor, "required_pipelines", [])
        if required_pipelines:
            parts.append(f"Requires pipelines: {', '.join(required_pipelines)}")
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

    def _pipeline_status_text(self, pipeline: PipelineDescriptor) -> str:
        if pipeline.available:
            return "Available"
        if pipeline.missing_deps:
            return f"Missing deps: {', '.join(pipeline.missing_deps)}"
        return "Unavailable"

    def _populate_pipeline_library(self, rows: list[PipelineDescriptor]) -> None:
        for child in self.pipeline_library_inner.winfo_children():
            child.destroy()
        self.pipeline_visibility_vars = {}
        self.pipeline_library_inner.columnconfigure(0, weight=1)

        selected_header = ttk.Label(self.pipeline_library_inner, text="Selected")
        selected_header.grid(row=0, column=0, sticky="w", pady=(0, 6))
        status_header = ttk.Label(self.pipeline_library_inner, text="Status")
        status_header.grid(row=0, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
        self._bind_vertical_mousewheel(selected_header, self.pipeline_library_canvas)
        self._bind_vertical_mousewheel(status_header, self.pipeline_library_canvas)

        for idx, pipeline in enumerate(rows, start=1):
            is_available = getattr(pipeline, "available", True)
            var = tk.BooleanVar(
                value=self.pipeline_visibility.get(pipeline.name, False)
                and is_available
            )
            check = ttk.Checkbutton(
                self.pipeline_library_inner,
                text=pipeline.name,
                variable=var,
                state="normal" if is_available else "disabled",
                command=lambda name=pipeline.name, visible_var=var: (
                    self._set_pipeline_visibility(name, visible_var.get())
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))
            self._bind_vertical_mousewheel(check, self.pipeline_library_canvas)

            status_text = self._pipeline_status_text(pipeline)
            status = ttk.Label(self.pipeline_library_inner, text=status_text)
            status.grid(row=idx, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
            self._bind_vertical_mousewheel(status, self.pipeline_library_canvas)

            tip_text = self._descriptor_tooltip_text(pipeline)
            if tip_text:
                _Tooltip(check, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(status, tip_text, bg=self._surface_color, fg=self._text_fg)

            self.pipeline_visibility_vars[pipeline.name] = var

        self._update_pipeline_library_summary()

    def _sync_pipeline_visibility(self, rows: list[PipelineDescriptor]) -> None:
        visibility, changed = normalize_pipeline_visibility(
            (pipeline.name for pipeline in rows),
            self.settings_store.load_pipeline_visibility(),
        )
        for pipeline in rows:
            if not pipeline.available and visibility.get(pipeline.name, False):
                visibility[pipeline.name] = False
                changed = True
        self.pipeline_visibility = visibility
        if changed:
            self._persist_pipeline_visibility()

    def _persist_pipeline_visibility(self) -> None:
        try:
            self.settings_store.save_pipeline_visibility(self.pipeline_visibility)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save pipeline selection preferences:\n{exc}",
            )

    def _set_pipeline_visibility(self, name: str, visible: bool) -> None:
        pipeline = self.pipeline_catalog.get(name)
        if pipeline is not None and not pipeline.available:
            visible = False
        if self.pipeline_visibility.get(name) == visible:
            return
        self.pipeline_visibility[name] = visible
        self._persist_pipeline_visibility()
        self._update_pipeline_library_summary()

    def _set_all_pipeline_visibility(self, visible: bool) -> None:
        changed = False
        target_values = {
            pipeline.name: visible and pipeline.available
            for pipeline in self.pipeline_rows
        }
        for name, target_value in target_values.items():
            if self.pipeline_visibility.get(name) != target_value:
                self.pipeline_visibility[name] = target_value
                changed = True
        if not changed:
            return
        for name, var in self.pipeline_visibility_vars.items():
            var.set(self.pipeline_visibility.get(name, False))
        self._persist_pipeline_visibility()
        self._update_pipeline_library_summary()

    def _update_pipeline_library_summary(self) -> None:
        selected_count = sum(
            1
            for pipeline in self.pipeline_rows
            if pipeline.available and self.pipeline_visibility.get(pipeline.name, False)
        )
        available_count = sum(
            1 for pipeline in self.pipeline_rows if pipeline.available
        )
        self.pipeline_library_summary_var.set(
            f"Selected: {selected_count}/{available_count}"
        )

    def _package_folder(self, package_name: str) -> Path | None:
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

        for root in self._resource_roots():
            folder = root / package_name
            if folder.is_dir():
                return folder
        return None

    def _open_folder(self, folder: Path | None, label: str) -> None:
        if folder is None or not folder.is_dir():
            services_for(self).dialogs.showerror(
                label,
                f"Could not find the {label.lower()}.",
            )
            return
        try:
            services_for(self).folder_opener.open_folder(folder)
        except Exception as exc:  # noqa: BLE001
            services_for(self).dialogs.showerror(
                label,
                f"Could not open folder:\n{folder}\n\n{exc}",
            )

    def open_pipeline_folder(self) -> None:
        self._open_folder(self._package_folder("pipelines"), "Pipeline folder")

    def select_all_pipelines(self) -> None:
        self._set_all_pipeline_visibility(True)

    def deselect_all_pipelines(self) -> None:
        self._set_all_pipeline_visibility(False)

    def refresh_pipeline_catalog(self) -> None:
        self._register_pipelines()
        self._register_postprocesses()
