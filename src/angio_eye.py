import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence

import h5py

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

from pipelines import (
    ProcessPipeline,
    ProcessResult,
    load_pipeline_catalog,
)
from pipelines.core.utils import write_combined_results_h5, write_result_h5


class _Tooltip:
    """Lightweight tooltip that shows on hover."""

    def __init__(
        self, widget: tk.Widget, text: str, bg: str = "#333333", fg: str = "#f7f7f7"
    ) -> None:
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tipwindow: Optional[tk.Toplevel] = None
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
        self.h5_file: Optional[h5py.File] = None
        self.pipeline_registry: Dict[str, ProcessPipeline] = {}
        self.pipeline_check_vars: Dict[str, tk.BooleanVar] = {}
        self.last_process_result: Optional[ProcessResult] = None
        self.last_process_pipeline: Optional[ProcessPipeline] = None
        self.output_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.last_output_dir: Optional[Path] = None
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")

        self._apply_theme()
        self._build_ui()
        self._register_pipelines()
        self._show_placeholder()
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
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        single_tab = ttk.Frame(notebook, padding=10)
        batch_tab = ttk.Frame(notebook, padding=10)
        notebook.add(single_tab, text="Single file")
        notebook.add(batch_tab, text="Batch")

        self._build_single_tab(single_tab)
        self._build_batch_tab(batch_tab)

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        top_bar = ttk.Frame(parent)
        top_bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        open_btn = ttk.Button(top_bar, text="Open .h5 file", command=self.open_file)
        open_btn.pack(side="left")
        self.file_label = ttk.Label(top_bar, text="No file loaded", wraplength=500)
        self.file_label.pack(side="left", padx=8)

        ttk.Label(parent, text="Pipeline").grid(row=1, column=0, sticky="w")
        self.pipeline_var = tk.StringVar()
        self.pipeline_combo = ttk.Combobox(
            parent, textvariable=self.pipeline_var, state="readonly", width=40
        )
        self.pipeline_combo.grid(row=1, column=1, sticky="w")
        run_btn = ttk.Button(
            parent, text="Run pipeline", command=self.run_selected_pipeline
        )
        run_btn.grid(row=1, column=2, sticky="w", padx=6)

        ttk.Label(parent, text="Result").grid(row=2, column=0, sticky="nw", pady=(8, 2))
        output_frame = ttk.Frame(parent)
        output_frame.grid(row=2, column=1, columnspan=2, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.process_output = tk.Text(
            output_frame,
            height=18,
            state="disabled",
            bg=self._text_bg,
            fg=self._text_fg,
            insertbackground=self._text_fg,
        )
        output_scroll = ttk.Scrollbar(
            output_frame, orient="vertical", command=self.process_output.yview
        )
        self.process_output.configure(yscrollcommand=output_scroll.set)
        self.process_output.grid(row=0, column=0, sticky="nsew")
        output_scroll.grid(row=0, column=1, sticky="ns")

        export_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        export_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
        export_frame.columnconfigure(1, weight=1)
        ttk.Label(export_frame, text="Output folder").grid(row=0, column=0, sticky="w")
        output_dir_entry = ttk.Entry(export_frame, textvariable=self.output_dir_var)
        output_dir_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(export_frame, text="Browse", command=self.choose_output_dir).grid(
            row=0, column=2, sticky="w"
        )

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
        self.batch_zip_label.grid(
            row=4, column=0, sticky="w", pady=(2, 8), padx=(0, 4)
        )
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

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        self.pipeline_registry = {p.name: p for p in available}
        self.missing_pipelines = {p.name: p for p in missing}
        self.pipeline_combo["values"] = list(self.pipeline_registry.keys())
        if available:
            self.pipeline_combo.current(0)
            self.pipeline_var.set(available[0].name)
        self._populate_pipeline_checks(available, missing)

    def _populate_pipeline_checks(
        self, available: List[ProcessPipeline], missing: List[ProcessPipeline]
    ) -> None:
        for child in self.pipeline_checks_inner.winfo_children():
            child.destroy()
        self.pipeline_check_vars = {}
        rows: List[ProcessPipeline] = [*available, *missing]
        for idx, pipeline in enumerate(rows):
            is_available = getattr(pipeline, "available", True)
            var = tk.BooleanVar(value=is_available)
            var._enabled = is_available  # type: ignore[attr-defined]
            label = pipeline.name if is_available else f"{pipeline.name} (missing deps)"
            state = "normal" if is_available else "disabled"
            check = ttk.Checkbutton(
                self.pipeline_checks_inner, text=label, variable=var, state=state
            )
            check.grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
            tip_text = pipeline.description or ""
            missing_deps = getattr(pipeline, "missing_deps", []) or getattr(pipeline, "requires", [])
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

    def _show_placeholder(
        self, message: str = "Load a .h5 file then run a pipeline"
    ) -> None:
        self.process_output.configure(state="normal")
        self.process_output.delete("1.0", "end")
        self.process_output.insert("end", message)
        self.process_output.configure(state="disabled")

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

    def select_all_pipelines(self) -> None:
        for var in self.pipeline_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(True)

    def clear_all_pipelines(self) -> None:
        for var in self.pipeline_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(False)

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=os.path.abspath("h5_example"),
        )
        if not path:
            return
        try:
            if self.h5_file is not None:
                self.h5_file.close()
            self.h5_file = h5py.File(path, "r")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Cannot open {path}: {exc}")
            return
        self.file_label.config(text=path)
        self.last_process_result = None
        self.last_process_pipeline = None
        self.last_output_dir = None
        self._show_placeholder("File loaded. Pick a pipeline and run.")

    def run_selected_pipeline(self) -> None:
        name = self.pipeline_var.get()
        if not name:
            messagebox.showwarning(
                "Missing pipeline", "Select a pipeline before running."
            )
            return
        pipeline = self.pipeline_registry.get(name)
        if pipeline is None:
            messagebox.showerror(
                "Pipeline missing", f"Pipeline '{name}' is not registered."
            )
            return
        if self.h5_file is None:
            messagebox.showwarning("Missing file", "Load a .h5 file first.")
            return
        try:
            result = pipeline.run(self.h5_file)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Pipeline error", f"Pipeline failed: {exc}")
            return
        try:
            output_dir = self._prepare_output_dir()
            output_path = self._default_output_path(name, output_dir)
            self._write_result_h5(result, output_path, pipeline_name=name)
            result.output_h5_path = output_path
            self.last_output_dir = output_dir
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Output error", f"Cannot write outputs: {exc}")
            return
        self.last_process_result = result
        self.last_process_pipeline = pipeline
        file_label = self.h5_file.filename or "(in-memory file)"
        self._render_process_result(result, pipeline_name=name, file_path=file_label)

    def _render_process_result(
        self, result: ProcessResult, pipeline_name: str, file_path: str
    ) -> None:
        self.process_output.configure(state="normal")
        self.process_output.delete("1.0", "end")
        self.process_output.insert("end", f"Pipeline: {pipeline_name}\n")
        self.process_output.insert("end", f"File: {file_path}\n\n")
        self.process_output.insert("end", "Metrics:\n")
        for key, value in result.metrics.items():
            self.process_output.insert("end", f" - {key}: {value}\n")
        if result.artifacts:
            self.process_output.insert("end", "\nArtifacts:\n")
            for key, value in result.artifacts.items():
                self.process_output.insert("end", f" - {key}: {value}\n")
        if result.output_h5_path:
            self.process_output.insert(
                "end", f"\nResult HDF5: {result.output_h5_path}\n"
            )
        self.process_output.configure(state="disabled")

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

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.output_dir_var.get() or None,
            title="Select base folder for outputs",
        )
        if path:
            self.output_dir_var.set(path)

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

        pipelines: List[ProcessPipeline] = []
        missing: List[str] = []
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

        tempdir: Optional[tempfile.TemporaryDirectory] = None
        temp_output_dir: Optional[tempfile.TemporaryDirectory] = None
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

        failures: List[str] = []
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
            messagebox.showwarning(
                "Batch completed with errors",
                f"{len(failures)} failure(s). See log for details.\n\n{summary_msg}",
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
    ) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
        if data_path.is_file() and data_path.suffix.lower() == ".zip":
            tempdir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(tempdir.name)
            return Path(tempdir.name), tempdir
        return data_path, None

    def _find_h5_inputs(self, path: Path) -> List[Path]:
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                return [path]
            raise ValueError(f"File is not an HDF5 file: {path}")
        if path.is_dir():
            files = sorted({*path.rglob("*.h5"), *path.rglob("*.hdf5")})
            return files
        raise FileNotFoundError(f"Input path does not exist: {path}")

    def _safe_pipeline_suffix(self, name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "pipeline"

    def _run_pipelines_on_file(
        self,
        h5_path: Path,
        pipelines: Sequence[ProcessPipeline],
        output_root: Path,
    ) -> None:
        data_dir = output_root / h5_path.stem
        data_dir.mkdir(parents=True, exist_ok=True)
        combined_h5_out = data_dir / f"{h5_path.stem}_pipelines_result.h5"
        pipeline_results: List[tuple[str, ProcessResult]] = []
        with h5py.File(h5_path, "r") as h5file:
            for pipeline in pipelines:
                result = pipeline.run(h5file)
                pipeline_results.append((pipeline.name, result))
                self._log_batch(f"[OK] {h5_path.name} -> {pipeline.name}")
        write_combined_results_h5(
            pipeline_results, combined_h5_out, source_file=str(h5_path)
        )
        for _, result in pipeline_results:
            result.output_h5_path = str(combined_h5_out)
        self._log_batch(
            f"[OK] {h5_path.name}: combined results -> {combined_h5_out.name}"
        )

    def _prepare_output_dir(self) -> Path:
        base_dir_value = (self.output_dir_var.get() or "").strip()
        base_dir = Path(base_dir_value).expanduser() if base_dir_value else Path.cwd()
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = (
            Path(self.h5_file.filename).stem
            if self.h5_file and self.h5_file.filename
            else "output"
        )
        output_dir = base_dir / f"{base_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _default_output_path(self, pipeline_name: str, output_dir: Path) -> str:
        safe_name = self._safe_pipeline_suffix(pipeline_name)
        base = (
            Path(self.h5_file.filename).stem
            if self.h5_file and self.h5_file.filename
            else "output"
        )
        return str(output_dir / f"{base}_{safe_name}_result.h5")

    def _zip_output_dir(self, folder: Path, target_path: Optional[Path] = None) -> Path:
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

    def _write_result_h5(
        self, result: ProcessResult, path: str, pipeline_name: str
    ) -> None:
        source_file = (
            self.h5_file.filename if self.h5_file and self.h5_file.filename else None
        )
        write_result_h5(
            result, path, pipeline_name=pipeline_name, source_file=source_file
        )


if __name__ == "__main__":
    app = ProcessApp()
    app.mainloop()
