import tkinter as tk
from tkinter import ttk

from app_settings import normalize_postprocess_visibility
from postprocess import PostprocessDescriptor, load_postprocess_catalog

from .widgets import _Tooltip

class PostprocessLibraryMixin:
    def _build_postprocess_library_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text="Select the postprocess steps to run after pipelines. "
            "This preference is saved between app launches.",
        ).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        controls.columnconfigure(4, weight=1)
        ttk.Button(
            controls,
            text="Select all",
            command=self.select_all_postprocesses,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Deselect all",
            command=self.deselect_all_postprocesses,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload postprocess",
            command=self.refresh_postprocess_catalog,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Open folder",
            command=self.open_postprocess_folder,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(controls, textvariable=self.postprocess_library_summary_var).grid(
            row=0, column=4, sticky="e"
        )

        library_container = ttk.Frame(parent)
        library_container.grid(row=2, column=0, sticky="nsew")
        library_container.columnconfigure(0, weight=1)
        library_container.rowconfigure(0, weight=1)

        self.postprocess_library_canvas = tk.Canvas(
            library_container, highlightthickness=0, bg=self._bg_color
        )
        self.postprocess_library_canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_container,
            orient="vertical",
            command=self.postprocess_library_canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        self.postprocess_library_canvas.configure(yscrollcommand=library_scroll.set)
        self.postprocess_library_inner = ttk.Frame(self.postprocess_library_canvas)
        self.postprocess_library_window = self.postprocess_library_canvas.create_window(
            (0, 0), window=self.postprocess_library_inner, anchor="nw"
        )
        self.postprocess_library_inner.bind(
            "<Configure>",
            lambda _evt: self.postprocess_library_canvas.configure(
                scrollregion=self.postprocess_library_canvas.bbox("all")
            ),
        )
        self.postprocess_library_canvas.bind(
            "<Configure>",
            lambda evt: self.postprocess_library_canvas.itemconfigure(
                self.postprocess_library_window, width=evt.width
            ),
        )
        self._bind_vertical_mousewheel(
            self.postprocess_library_canvas, self.postprocess_library_canvas
        )
        self._bind_vertical_mousewheel(
            self.postprocess_library_inner, self.postprocess_library_canvas
        )
        self._bind_vertical_mousewheel(library_scroll, self.postprocess_library_canvas)

    def _register_postprocesses(self) -> None:
        available, missing = load_postprocess_catalog()
        rows = sorted(
            [*available, *missing], key=lambda postprocess: postprocess.name.lower()
        )
        self.postprocess_registry = {p.name: p for p in available}
        self.postprocess_catalog = {p.name: p for p in rows}
        self.postprocess_rows = rows
        self._sync_postprocess_visibility(rows)
        self._populate_postprocess_library(rows)
        self._install_drop_targets()

    def _postprocess_status_text(self, postprocess: PostprocessDescriptor) -> str:
        if not postprocess.available:
            if postprocess.missing_pipelines:
                return f"Missing pipelines: {', '.join(postprocess.missing_pipelines)}"
            if postprocess.missing_deps:
                return f"Missing deps: {', '.join(postprocess.missing_deps)}"
            return "Unavailable"
        if postprocess.required_pipelines:
            return f"Requires: {', '.join(postprocess.required_pipelines)}"
        return "Available"

    def _populate_postprocess_library(self, rows: list[PostprocessDescriptor]) -> None:
        for child in self.postprocess_library_inner.winfo_children():
            child.destroy()
        self.postprocess_visibility_vars = {}
        self.postprocess_library_inner.columnconfigure(0, weight=1)

        selected_header = ttk.Label(self.postprocess_library_inner, text="Selected")
        selected_header.grid(row=0, column=0, sticky="w", pady=(0, 6))
        status_header = ttk.Label(self.postprocess_library_inner, text="Status")
        status_header.grid(row=0, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
        self._bind_vertical_mousewheel(selected_header, self.postprocess_library_canvas)
        self._bind_vertical_mousewheel(status_header, self.postprocess_library_canvas)

        for idx, postprocess in enumerate(rows, start=1):
            is_available = getattr(postprocess, "available", True)
            var = tk.BooleanVar(
                value=self.postprocess_visibility.get(postprocess.name, False)
                and is_available
            )
            check = ttk.Checkbutton(
                self.postprocess_library_inner,
                text=postprocess.name,
                variable=var,
                state="normal" if is_available else "disabled",
                command=lambda name=postprocess.name, visible_var=var: self._set_postprocess_visibility(
                    name, visible_var.get()
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))
            self._bind_vertical_mousewheel(check, self.postprocess_library_canvas)

            status_text = self._postprocess_status_text(postprocess)
            status = ttk.Label(self.postprocess_library_inner, text=status_text)
            status.grid(row=idx, column=1, sticky="w", padx=(12, 18), pady=(0, 6))
            self._bind_vertical_mousewheel(status, self.postprocess_library_canvas)

            tip_text = self._descriptor_tooltip_text(postprocess)
            if tip_text:
                _Tooltip(check, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(status, tip_text, bg=self._surface_color, fg=self._text_fg)

            self.postprocess_visibility_vars[postprocess.name] = var

        self._update_postprocess_library_summary()

    def _sync_postprocess_visibility(self, rows: list[PostprocessDescriptor]) -> None:
        visibility, changed = normalize_postprocess_visibility(
            (postprocess.name for postprocess in rows),
            self.settings_store.load_postprocess_visibility(),
        )
        for postprocess in rows:
            if not postprocess.available and visibility.get(postprocess.name, False):
                visibility[postprocess.name] = False
                changed = True
        self.postprocess_visibility = visibility
        if changed:
            self._persist_postprocess_visibility()

    def _persist_postprocess_visibility(self) -> None:
        try:
            self.settings_store.save_postprocess_visibility(self.postprocess_visibility)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save postprocess selection preferences:\n{exc}",
            )

    def _set_postprocess_visibility(self, name: str, visible: bool) -> None:
        postprocess = self.postprocess_catalog.get(name)
        if postprocess is not None and not postprocess.available:
            visible = False
        if self.postprocess_visibility.get(name) == visible:
            return
        self.postprocess_visibility[name] = visible
        self._persist_postprocess_visibility()
        self._update_postprocess_library_summary()

    def _set_all_postprocess_visibility(self, visible: bool) -> None:
        changed = False
        target_values = {
            postprocess.name: visible and postprocess.available
            for postprocess in self.postprocess_rows
        }
        for name, target_value in target_values.items():
            if self.postprocess_visibility.get(name) != target_value:
                self.postprocess_visibility[name] = target_value
                changed = True
        if not changed:
            return
        for name, var in self.postprocess_visibility_vars.items():
            var.set(self.postprocess_visibility.get(name, False))
        self._persist_postprocess_visibility()
        self._update_postprocess_library_summary()

    def _update_postprocess_library_summary(self) -> None:
        selected_count = sum(
            1
            for postprocess in self.postprocess_rows
            if postprocess.available
            and self.postprocess_visibility.get(postprocess.name, False)
        )
        available_count = sum(
            1 for postprocess in self.postprocess_rows if postprocess.available
        )
        self.postprocess_library_summary_var.set(
            f"Selected: {selected_count}/{available_count}"
        )

    def open_postprocess_folder(self) -> None:
        self._open_folder(
            self._package_folder("postprocess"),
            "Postprocess folder",
        )

    def select_all_postprocesses(self) -> None:
        self._set_all_postprocess_visibility(True)

    def deselect_all_postprocesses(self) -> None:
        self._set_all_postprocess_visibility(False)

    def refresh_postprocess_catalog(self) -> None:
        self._register_postprocesses()
