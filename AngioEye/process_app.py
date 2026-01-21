import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence

import h5py

from pipelines import (
    BasicStatsPipeline,
    ProcessPipeline,
    ProcessResult,
    VelocityComparisonPipeline,
)
from pipelines.utils import write_result_h5


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

        self._build_ui()
        self._register_pipelines()
        self._show_placeholder()
        self._reset_batch_output()

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
        self.process_output = tk.Text(output_frame, height=18, state="disabled")
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

        ttk.Label(export_frame, text="Export CSV").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.export_path_var = tk.StringVar(value="process_result.csv")
        export_entry = ttk.Entry(export_frame, textvariable=self.export_path_var)
        export_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(export_frame, text="Browse", command=self.choose_export_path).grid(
            row=1, column=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            export_frame, text="Export", command=self.export_process_result
        ).grid(row=1, column=3, sticky="w", padx=6, pady=(6, 0))

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(4, weight=1)

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
            pipelines_container, highlightthickness=0, height=220
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

        ttk.Label(parent, text="Batch log").grid(
            row=4, column=0, sticky="nw", pady=(8, 2)
        )
        batch_output_frame = ttk.Frame(parent)
        batch_output_frame.grid(row=4, column=1, columnspan=2, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        self.batch_output = tk.Text(batch_output_frame, height=18, state="disabled")
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=self.batch_output.yview
        )
        self.batch_output.configure(yscrollcommand=batch_output_scroll.set)
        self.batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")

    def _register_pipelines(self) -> None:
        pipelines = [BasicStatsPipeline(), VelocityComparisonPipeline()]
        self.pipeline_registry = {p.name: p for p in pipelines}
        self.pipeline_combo["values"] = list(self.pipeline_registry.keys())
        if pipelines:
            self.pipeline_combo.current(0)
            self.pipeline_var.set(pipelines[0].name)
        self._populate_pipeline_checks(pipelines)

    def _populate_pipeline_checks(self, pipelines: List[ProcessPipeline]) -> None:
        for child in self.pipeline_checks_inner.winfo_children():
            child.destroy()
        self.pipeline_check_vars = {}
        for idx, pipeline in enumerate(pipelines):
            var = tk.BooleanVar(value=True)
            check = ttk.Checkbutton(
                self.pipeline_checks_inner, text=pipeline.name, variable=var
            )
            check.grid(row=idx * 2, column=0, sticky="w", padx=(0, 8))
            if pipeline.description:
                desc = ttk.Label(
                    self.pipeline_checks_inner,
                    text=pipeline.description,
                    wraplength=520,
                    foreground="#555555",
                    anchor="w",
                    justify="left",
                )
                desc.grid(row=idx * 2 + 1, column=0, sticky="w", pady=(0, 8))
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
            var.set(True)

    def clear_all_pipelines(self) -> None:
        for var in self.pipeline_check_vars.values():
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
            self._update_export_default(output_dir)
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

    def choose_export_path(self) -> None:
        initial_dir = self.last_output_dir or Path(
            self.output_dir_var.get() or Path.cwd()
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialdir=str(initial_dir),
            initialfile=Path(self.export_path_var.get()).name,
        )
        if path:
            self.export_path_var.set(Path(path).name)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.output_dir_var.get() or None,
            title="Select base folder for outputs",
        )
        if path:
            self.output_dir_var.set(path)

    def export_process_result(self) -> None:
        if self.last_process_result is None or self.last_process_pipeline is None:
            messagebox.showwarning("No result", "Run a pipeline before exporting.")
            return
        if self.last_output_dir is None:
            try:
                self.last_output_dir = self._prepare_output_dir()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(
                    "Export failed", f"Cannot prepare output folder: {exc}"
                )
                return
        export_name = Path(self.export_path_var.get() or "process_result.csv").name
        final_path = self.last_output_dir / export_name
        self.last_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            final_path_str = self.last_process_pipeline.export(
                self.last_process_result, str(final_path)
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", f"Cannot export: {exc}")
            return
        self.export_path_var.set(final_path_str)
        messagebox.showinfo("Export done", f"Result exported to: {final_path_str}")

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

        output_dir_value = (self.batch_output_var.get() or "").strip()
        output_dir = (
            Path(output_dir_value).expanduser() if output_dir_value else Path.cwd()
        )
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._reset_batch_output("Starting batch run...\n")

        tempdir: Optional[tempfile.TemporaryDirectory] = None
        try:
            data_root, tempdir = self._prepare_data_root(data_path)
            inputs = self._find_h5_inputs(data_root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
            self._log_batch(f"Error: {exc}")
            if tempdir is not None:
                tempdir.cleanup()
            return

        failures: List[str] = []
        for h5_path in inputs:
            try:
                self._run_pipelines_on_file(h5_path, pipelines, output_dir)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                self._log_batch(f"[FAIL] {h5_path.name}: {exc}")

        self._log_batch(f"Completed. Outputs stored under: {output_dir}")
        if failures:
            messagebox.showwarning(
                "Batch completed with errors",
                f"{len(failures)} failure(s). See log for details.",
            )
        else:
            messagebox.showinfo(
                "Batch completed", f"Outputs stored under: {output_dir}"
            )

        if tempdir is not None:
            tempdir.cleanup()

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
        with h5py.File(h5_path, "r") as h5file:
            for pipeline in pipelines:
                suffix = self._safe_pipeline_suffix(pipeline.name)
                h5_out = data_dir / f"{h5_path.stem}_{suffix}_result.h5"
                csv_out = data_dir / f"{h5_path.stem}_{suffix}_metrics.csv"
                result = pipeline.run(h5file)
                write_result_h5(
                    result,
                    h5_out,
                    pipeline_name=pipeline.name,
                    source_file=str(h5_path),
                )
                result.output_h5_path = str(h5_out)
                pipeline.export(result, str(csv_out))
                self._log_batch(f"[OK] {h5_path.name} -> {pipeline.name}")

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
        safe_name = pipeline_name.lower().replace(" ", "_")
        base = (
            Path(self.h5_file.filename).stem
            if self.h5_file and self.h5_file.filename
            else "output"
        )
        return str(output_dir / f"{base}_{safe_name}_result.h5")

    def _update_export_default(self, output_dir: Path) -> None:
        export_name = Path(self.export_path_var.get() or "process_result.csv").name
        self.export_path_var.set(str(output_dir / export_name))

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
