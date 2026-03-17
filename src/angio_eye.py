import os
import tempfile
import tkinter as tk
import zipfile
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import h5py

from app_settings import AppSettingsStore, normalize_pipeline_visibility

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

from pipelines import PipelineDescriptor, ProcessResult, load_pipeline_catalog
from pipelines.core.errors import format_pipeline_exception
from pipelines.core.utils import write_combined_results_h5


class _Tooltip:
    """Lightweight tooltip that shows on hover."""

    def __init__(
        self, widget: tk.Widget, text: str, bg: str = "#333333", fg: str = "#f7f7f7"
    ) -> None:
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background=self.bg,
            foreground=self.fg,
            relief="solid",
            borderwidth=1,
            wraplength=360,
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ProcessApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HDF5 Process")
        self.geometry("800x600")
        self.settings_store = AppSettingsStore()
        self.pipeline_registry: dict[str, PipelineDescriptor] = {}
        self.pipeline_catalog: dict[str, PipelineDescriptor] = {}
        self.pipeline_rows: list[PipelineDescriptor] = []
        self.pipeline_check_vars: dict[str, tk.BooleanVar] = {}
        self.pipeline_visibility: dict[str, bool] = {}
        self.pipeline_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.pipeline_library_summary_var = tk.StringVar(value="")
        self._settings_warning_shown = False

        self._apply_theme()
        self._build_ui()
        self._register_pipelines()
        self._reset_batch_output()

    def _apply_theme(self) -> None:
        """
        Apply the Sun Valley ttk theme when available; otherwise fall back to a simple dark palette.
        """
        style = ttk.Style(self)
        if sv_ttk:
            try:
                sv_ttk.set_theme("dark")
            except Exception:
                pass

        # Fallback palette aligned with Sun Valley dark.
        fallback_bg = "#0f1116"
        fallback_surface = "#1b1f27"
        fallback_fg = "#e8eef5"
        fallback_muted = "#9aa6b5"
        fallback_accent = "#4f9dff"

        # Derive colors from the active theme when possible to keep consistency.
        bg = style.lookup("TFrame", "background") or fallback_bg
        fg = style.lookup("TLabel", "foreground") or fallback_fg
        surface = (
            style.lookup("TEntry", "fieldbackground")
            or style.lookup("TEntry", "background")
            or fallback_surface
        )
        muted = (
            style.lookup("TLabel", "foreground", state=("disabled",)) or fallback_muted
        )
        accent = (
            style.lookup("TButton", "bordercolor")
            or style.lookup("TNotebook", "foreground")
            or fallback_accent
        )

        self.configure(bg=bg)
        # set texts colors when created.
        self._text_bg = surface
        self._text_fg = fg
        self._muted_fg = muted
        self._bg_color = bg
        self._surface_color = surface
        self._accent_color = accent

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.batch_tab = ttk.Frame(self.notebook, padding=10)
        self.library_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.batch_tab, text="Batch")
        self.notebook.add(self.library_tab, text="Pipeline Library")

        self._build_batch_tab(self.batch_tab)
        self._build_pipeline_library_tab(self.library_tab)

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)
        parent.columnconfigure(3, weight=0)
        parent.rowconfigure(5, weight=1)

        ttk.Label(parent, text="Input (folder / .h5 / .hdf5 / .zip)").grid(
            row=0, column=0, sticky="w"
        )
        input_entry = ttk.Entry(parent, textvariable=self.batch_input_var)
        input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        input_btn_frame = ttk.Frame(parent)
        input_btn_frame.grid(row=0, column=2, sticky="w")
        ttk.Button(
            input_btn_frame, text="Browse folder", command=self.choose_batch_folder
        ).pack(side="left")
        ttk.Button(
            input_btn_frame, text="Browse file/zip", command=self.choose_batch_file
        ).pack(side="left", padx=(4, 0))

        ttk.Label(parent, text="Pipelines").grid(
            row=1, column=0, sticky="nw", pady=(8, 0)
        )
        pipelines_wrapper = ttk.Frame(parent)
        pipelines_wrapper.grid(
            row=1, column=1, columnspan=2, sticky="nsew", pady=(8, 0)
        )
        pipelines_wrapper.columnconfigure(0, weight=1)
        pipelines_wrapper.rowconfigure(1, weight=1)

        actions = ttk.Frame(pipelines_wrapper)
        actions.grid(row=0, column=0, sticky="e", pady=(0, 4))
        ttk.Button(actions, text="Select all", command=self.select_all_pipelines).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(actions, text="Clear all", command=self.clear_all_pipelines).pack(
            side="left"
        )
        ttk.Button(
            actions,
            text="Pipeline library",
            command=self.open_pipeline_library,
        ).pack(side="left", padx=(4, 0))

        pipelines_container = ttk.Frame(pipelines_wrapper)
        pipelines_container.grid(row=1, column=0, sticky="nsew")
        pipelines_container.columnconfigure(0, weight=1)
        pipelines_container.rowconfigure(0, weight=1)

        self.pipeline_checks_canvas = tk.Canvas(
            pipelines_container, highlightthickness=0, height=220, bg=self._bg_color
        )
        self.pipeline_checks_canvas.grid(row=0, column=0, sticky="nsew")
        pipeline_scroll = ttk.Scrollbar(
            pipelines_container,
            orient="vertical",
            command=self.pipeline_checks_canvas.yview,
        )
        pipeline_scroll.grid(row=0, column=1, sticky="ns")
        self.pipeline_checks_canvas.configure(yscrollcommand=pipeline_scroll.set)
        self.pipeline_checks_inner = ttk.Frame(self.pipeline_checks_canvas)
        self.pipeline_checks_window = self.pipeline_checks_canvas.create_window(
            (0, 0), window=self.pipeline_checks_inner, anchor="nw"
        )
        self.pipeline_checks_inner.bind(
            "<Configure>",
            lambda _evt: self.pipeline_checks_canvas.configure(
                scrollregion=self.pipeline_checks_canvas.bbox("all")
            ),
        )
        self.pipeline_checks_canvas.bind(
            "<Configure>",
            lambda evt: self.pipeline_checks_canvas.itemconfigure(
                self.pipeline_checks_window, width=evt.width
            ),
        )

        ttk.Label(parent, text="Output folder").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        batch_output_entry = ttk.Entry(parent, textvariable=self.batch_output_var)
        batch_output_entry.grid(row=2, column=1, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(parent, text="Browse", command=self.choose_batch_output).grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )

        run_btn = ttk.Button(parent, text="Run batch", command=self.run_batch)
        run_btn.grid(row=3, column=0, sticky="w", pady=(10, 4))
        ttk.Checkbutton(
            parent,
            text="Zip outputs after run",
            variable=self.batch_zip_var,
            command=self._toggle_zip_name_visibility,
        ).grid(row=3, column=1, sticky="w", pady=(10, 4))

        # Archive name placed on its own row to avoid resizing the log/list area.
        self.batch_zip_label = ttk.Label(parent, text="Archive name")
        self.batch_zip_label.grid(row=4, column=0, sticky="w", pady=(2, 8), padx=(0, 4))
        self.batch_zip_entry = ttk.Entry(
            parent, textvariable=self.batch_zip_name_var, width=28
        )
        self.batch_zip_entry.grid(
            row=4, column=1, columnspan=3, sticky="w", pady=(2, 8)
        )
        self._toggle_zip_name_visibility()

        ttk.Label(parent, text="Batch log").grid(
            row=5, column=0, sticky="nw", pady=(8, 2)
        )
        batch_output_frame = ttk.Frame(parent)
        batch_output_frame.grid(row=5, column=1, columnspan=3, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        self.batch_output = tk.Text(
            batch_output_frame,
            height=18,
            state="disabled",
            bg=self._text_bg,
            fg=self._text_fg,
            insertbackground=self._text_fg,
        )
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=self.batch_output.yview
        )
        self.batch_output.configure(yscrollcommand=batch_output_scroll.set)
        self.batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")

    def _build_pipeline_library_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text="Choose which pipelines are shown in the Batch tab. "
            "This preference is saved between app launches.",
        ).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        controls.columnconfigure(4, weight=1)
        ttk.Button(controls, text="Open batch", command=self.open_batch_tab).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            controls,
            text="Show all",
            command=self.show_all_pipelines_in_main_ui,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Hide all",
            command=self.hide_all_pipelines_from_main_ui,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload pipelines",
            command=self.refresh_pipeline_catalog,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(
            controls, textvariable=self.pipeline_library_summary_var
        ).grid(row=0, column=4, sticky="e")

        library_container = ttk.Frame(parent)
        library_container.grid(row=2, column=0, sticky="nsew")
        library_container.columnconfigure(0, weight=1)
        library_container.rowconfigure(0, weight=1)

        self.pipeline_library_canvas = tk.Canvas(
            library_container, highlightthickness=0, bg=self._bg_color
        )
        self.pipeline_library_canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_container,
            orient="vertical",
            command=self.pipeline_library_canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        self.pipeline_library_canvas.configure(yscrollcommand=library_scroll.set)
        self.pipeline_library_inner = ttk.Frame(self.pipeline_library_canvas)
        self.pipeline_library_window = self.pipeline_library_canvas.create_window(
            (0, 0), window=self.pipeline_library_inner, anchor="nw"
        )
        self.pipeline_library_inner.bind(
            "<Configure>",
            lambda _evt: self.pipeline_library_canvas.configure(
                scrollregion=self.pipeline_library_canvas.bbox("all")
            ),
        )
        self.pipeline_library_canvas.bind(
            "<Configure>",
            lambda evt: self.pipeline_library_canvas.itemconfigure(
                self.pipeline_library_window, width=evt.width
            ),
        )

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        rows = sorted([*available, *missing], key=lambda pipeline: pipeline.name.lower())
        self.pipeline_registry = {p.name: p for p in available}
        self.pipeline_catalog = {p.name: p for p in rows}
        self.pipeline_rows = rows
        self._sync_pipeline_visibility(rows)
        selection_state = {
            name: var.get() for name, var in self.pipeline_check_vars.items()
        }
        self._populate_pipeline_checks(rows, selection_state)
        self._populate_pipeline_library(rows)

    def _populate_pipeline_checks(
        self,
        rows: list[PipelineDescriptor],
        selection_state: dict[str, bool] | None = None,
    ) -> None:
        for child in self.pipeline_checks_inner.winfo_children():
            child.destroy()
        self.pipeline_check_vars = {}
        visible_rows = [row for row in rows if self.pipeline_visibility.get(row.name, False)]
        if not visible_rows:
            ttk.Label(
                self.pipeline_checks_inner,
                text="No pipelines are visible here. Open Pipeline Library to enable some.",
            ).grid(row=0, column=0, sticky="w", pady=(0, 6))
            return

        for idx, pipeline in enumerate(visible_rows):
            is_available = getattr(pipeline, "available", True)
            default_value = (
                selection_state.get(pipeline.name, is_available)
                if selection_state is not None
                else is_available
            )
            var = tk.BooleanVar(value=default_value if is_available else False)
            var._enabled = is_available  # type: ignore[attr-defined]
            label = pipeline.name if is_available else f"{pipeline.name} (missing deps)"
            state = "normal" if is_available else "disabled"
            check = ttk.Checkbutton(
                self.pipeline_checks_inner, text=label, variable=var, state=state
            )
            check.grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
            tip_text = pipeline.description or ""
            missing_deps = getattr(pipeline, "missing_deps", []) or getattr(
                pipeline, "requires", []
            )
            if missing_deps:
                tip_suffix = f"\nInstall: {', '.join(missing_deps)}"
                tip_text = (tip_text + tip_suffix) if tip_text else tip_suffix
            if tip_text:
                _Tooltip(
                    check,
                    tip_text,
                    bg=self._surface_color,
                    fg=self._text_fg,
                )
            self.pipeline_check_vars[pipeline.name] = var

    def _populate_pipeline_library(self, rows: list[PipelineDescriptor]) -> None:
        for child in self.pipeline_library_inner.winfo_children():
            child.destroy()
        self.pipeline_visibility_vars = {}
        self.pipeline_library_inner.columnconfigure(0, weight=1)

        ttk.Label(self.pipeline_library_inner, text="Show in Batch").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(self.pipeline_library_inner, text="Status").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
        )

        for idx, pipeline in enumerate(rows, start=1):
            var = tk.BooleanVar(value=self.pipeline_visibility.get(pipeline.name, False))
            check = ttk.Checkbutton(
                self.pipeline_library_inner,
                text=pipeline.name,
                variable=var,
                command=lambda name=pipeline.name, visible_var=var: self._set_pipeline_visibility(
                    name, visible_var.get()
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))

            if pipeline.available:
                status_text = "Available"
            elif pipeline.missing_deps:
                status_text = f"Missing deps: {', '.join(pipeline.missing_deps)}"
            else:
                status_text = "Unavailable"
            status = ttk.Label(self.pipeline_library_inner, text=status_text)
            status.grid(row=idx, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

            tip_text = pipeline.description or ""
            missing_deps = getattr(pipeline, "missing_deps", []) or getattr(
                pipeline, "requires", []
            )
            if missing_deps:
                tip_suffix = f"\nInstall: {', '.join(missing_deps)}"
                tip_text = (tip_text + tip_suffix) if tip_text else tip_suffix
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
        self.pipeline_visibility = visibility
        if changed:
            self._persist_pipeline_visibility()

    def _persist_pipeline_visibility(self) -> None:
        try:
            self.settings_store.save_pipeline_visibility(self.pipeline_visibility)
        except OSError as exc:
            if self._settings_warning_shown:
                return
            self._settings_warning_shown = True
            messagebox.showwarning(
                "Settings not saved",
                f"Could not save pipeline visibility preferences:\n{exc}",
            )

    def _set_pipeline_visibility(self, name: str, visible: bool) -> None:
        if self.pipeline_visibility.get(name) == visible:
            return
        self.pipeline_visibility[name] = visible
        self._persist_pipeline_visibility()
        selection_state = {
            pipeline_name: var.get()
            for pipeline_name, var in self.pipeline_check_vars.items()
        }
        self._populate_pipeline_checks(self.pipeline_rows, selection_state)
        self._update_pipeline_library_summary()

    def _set_all_pipeline_visibility(self, visible: bool) -> None:
        changed = False
        for name in self.pipeline_visibility:
            if self.pipeline_visibility[name] != visible:
                self.pipeline_visibility[name] = visible
                changed = True
        if not changed:
            return
        for var in self.pipeline_visibility_vars.values():
            var.set(visible)
        self._persist_pipeline_visibility()
        selection_state = {
            pipeline_name: var.get()
            for pipeline_name, var in self.pipeline_check_vars.items()
        }
        self._populate_pipeline_checks(self.pipeline_rows, selection_state)
        self._update_pipeline_library_summary()

    def _update_pipeline_library_summary(self) -> None:
        visible_count = sum(self.pipeline_visibility.values())
        total_count = len(self.pipeline_visibility)
        self.pipeline_library_summary_var.set(
            f"Visible in Batch: {visible_count}/{total_count}"
        )

    def open_pipeline_library(self) -> None:
        self.notebook.select(self.library_tab)

    def open_batch_tab(self) -> None:
        self.notebook.select(self.batch_tab)

    def show_all_pipelines_in_main_ui(self) -> None:
        self._set_all_pipeline_visibility(True)

    def hide_all_pipelines_from_main_ui(self) -> None:
        self._set_all_pipeline_visibility(False)

    def refresh_pipeline_catalog(self) -> None:
        self._register_pipelines()

    def _reset_batch_output(
        self, message: str = "Select an input path, choose pipelines, then run batch."
    ) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.delete("1.0", "end")
        self.batch_output.insert("end", message)
        self.batch_output.configure(state="disabled")

    def _log_batch(self, text: str) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.insert("end", f"{text}\n")
        self.batch_output.see("end")
        self.batch_output.configure(state="disabled")
        self.batch_output.update_idletasks()
        self.update_idletasks()

    def _export_batch_log(self, initial_dir: Path | None = None) -> Path | None:
        if initial_dir is None:
            initial_dir = Path(self.batch_output_var.get() or Path.cwd())
        if not initial_dir.exists():
            initial_dir = Path.cwd()
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            initialdir=str(initial_dir),
            initialfile="batch_log.txt",
            title="Export batch log",
        )
        if not path:
            return None
        try:
            log_text = self.batch_output.get("1.0", "end").rstrip()
            Path(path).write_text(log_text, encoding="utf-8")
            self._log_batch(f"[LOG] Exported batch log -> {path}")
            return Path(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", f"Could not save log: {exc}")
            return None

    def _show_batch_error_dialog(self, message: str, initial_dir: Path) -> None:
        self.bell()
        export = messagebox.askyesno(
            "Batch completed with errors",
            f"{message}\n\nExport log to .txt?",
            icon="warning",
        )
        if export:
            self._export_batch_log(initial_dir)

    def select_all_pipelines(self) -> None:
        for var in self.pipeline_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(True)

    def clear_all_pipelines(self) -> None:
        for var in self.pipeline_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(False)

    def choose_batch_folder(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_input_var.get() or None,
            title="Select folder containing HDF5 files",
        )
        if path:
            self.batch_input_var.set(path)

    def choose_batch_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("HDF5 or zip", "*.h5 *.hdf5 *.zip"), ("All files", "*.*")],
            initialdir=self.batch_input_var.get() or os.path.abspath("h5_example"),
            title="Select HDF5 file or .zip archive",
        )
        if path:
            self.batch_input_var.set(path)

    def choose_batch_output(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.batch_output_var.set(path)

    def run_batch(self) -> None:
        data_value = (self.batch_input_var.get() or "").strip()
        if not data_value:
            messagebox.showwarning(
                "Missing input",
                "Select a folder, HDF5 file, or .zip archive to process.",
            )
            return
        data_path = Path(data_value).expanduser()

        selected_names = [
            name for name, var in self.pipeline_check_vars.items() if var.get()
        ]
        if not selected_names:
            messagebox.showwarning(
                "No pipelines", "Select at least one pipeline to run."
            )
            return

        pipelines: list[PipelineDescriptor] = []
        missing: list[str] = []
        for name in selected_names:
            pipeline = self.pipeline_registry.get(name)
            if pipeline is None:
                missing.append(name)
            else:
                pipelines.append(pipeline)
        if missing:
            messagebox.showerror(
                "Pipeline missing", f"Pipeline(s) not registered: {', '.join(missing)}"
            )
            return

        base_output_value = (self.batch_output_var.get() or "").strip()
        base_output_dir = (
            Path(base_output_value).expanduser() if base_output_value else Path.cwd()
        )
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._reset_batch_output("Starting batch run...\n")

        tempdir: tempfile.TemporaryDirectory | None = None
        temp_output_dir: tempfile.TemporaryDirectory | None = None
        clean_temp_output = False
        try:
            data_root, tempdir = self._prepare_data_root(data_path)
            inputs = self._find_h5_inputs(data_root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
            self._log_batch(f"Error: {exc}")
            if tempdir is not None:
                tempdir.cleanup()
            return

        output_dir = base_output_dir
        if self.batch_zip_var.get():
            temp_output_dir = tempfile.TemporaryDirectory(dir=base_output_dir)
            output_dir = Path(temp_output_dir.name)

        failures: list[str] = []
        for h5_path in inputs:
            try:
                self._run_pipelines_on_file(h5_path, pipelines, output_dir)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                self._log_batch(f"[FAIL] {h5_path.name}: {exc}")

        summary_msg: str
        if self.batch_zip_var.get():
            try:
                zip_name = self.batch_zip_name_var.get().strip() or "outputs.zip"
                if not zip_name.lower().endswith(".zip"):
                    zip_name += ".zip"
                zip_path = self._zip_output_dir(
                    output_dir, target_path=base_output_dir / zip_name
                )
                self._log_batch(f"[ZIP] Archive created: {zip_path}")
                summary_msg = f"ZIP archive: {zip_path}"
                # Mark for cleanup so only the archive remains
                clean_temp_output = True
            except Exception as exc:  # noqa: BLE001
                self._log_batch(f"[ZIP FAIL] {exc}")
                messagebox.showerror(
                    "Zip failed", f"Could not create ZIP archive: {exc}"
                )
                summary_msg = f"Outputs stored under: {output_dir}"
        else:
            summary_msg = f"Outputs stored under: {output_dir}"

        self._log_batch(f"Completed. {summary_msg}")

        if failures:
            self._show_batch_error_dialog(
                f"{len(failures)} failure(s). See log for details.\n\n{summary_msg}",
                initial_dir=base_output_dir,
            )
        else:
            messagebox.showinfo("Batch completed", summary_msg)

        if clean_temp_output and temp_output_dir is not None:
            temp_output_dir.cleanup()
        if tempdir is not None:
            tempdir.cleanup()

    def _toggle_zip_name_visibility(self) -> None:
        if self.batch_zip_var.get():
            self.batch_zip_label.grid()
            self.batch_zip_entry.grid()
        else:
            self.batch_zip_label.grid_remove()
            self.batch_zip_entry.grid_remove()

    def _prepare_data_root(
        self, data_path: Path
    ) -> tuple[Path, tempfile.TemporaryDirectory | None]:
        if data_path.is_file() and data_path.suffix.lower() == ".zip":
            tempdir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(tempdir.name)
            return Path(tempdir.name), tempdir
        return data_path, None

    def _find_h5_inputs(self, path: Path) -> list[Path]:
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                return [path]
            raise ValueError(f"File is not an HDF5 file: {path}")
        if path.is_dir():
            files = sorted({*path.rglob("*.h5"), *path.rglob("*.hdf5")})
            return files
        raise FileNotFoundError(f"Input path does not exist: {path}")

    def _run_pipelines_on_file(
        self,
        h5_path: Path,
        pipelines: Sequence[PipelineDescriptor],
        output_root: Path,
    ) -> None:
        # Place combined output directly in the output root (no per-file subfolder).
        combined_h5_out = output_root / f"{h5_path.stem}_pipelines_result.h5"
        suffix = 1
        while combined_h5_out.exists():
            combined_h5_out = (
                output_root / f"{h5_path.stem}_{suffix}_pipelines_result.h5"
            )
            suffix += 1

        pipeline_results: list[tuple[str, ProcessResult]] = []
        with h5py.File(h5_path, "r") as h5file:
            for pipeline_desc in pipelines:
                pipeline = pipeline_desc.instantiate()
                try:
                    result = pipeline.run(h5file)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        format_pipeline_exception(exc, pipeline)
                    ) from exc
                pipeline_results.append((pipeline.name, result))
                self._log_batch(f"[OK] {h5_path.name} -> {pipeline.name}")
        write_combined_results_h5(
            pipeline_results, combined_h5_out, source_file=str(h5_path)
        )
        for _, result in pipeline_results:
            result.output_h5_path = str(combined_h5_out)
        self._log_batch(f"[OK] {h5_path.name}: combined results -> {combined_h5_out}")

    def _zip_output_dir(self, folder: Path, target_path: Path | None = None) -> Path:
        folder = folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {folder}")
        if target_path is None:
            zip_name = f"{folder.name}_outputs.zip" if folder.name else "outputs.zip"
            zip_path = folder.parent / zip_name
        else:
            zip_path = target_path.expanduser().resolve()
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(folder))
        return zip_path


def main():
    app = ProcessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
