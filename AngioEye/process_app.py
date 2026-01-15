import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import h5py

from pipelines import BasicStatsPipeline, ProcessPipeline, ProcessResult, VelocityComparisonPipeline


class ProcessApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HDF5 Process")
        self.geometry("800x600")
        self.h5_file: Optional[h5py.File] = None
        self.pipeline_registry: Dict[str, ProcessPipeline] = {}
        self.last_process_result: Optional[ProcessResult] = None
        self.last_process_pipeline: Optional[ProcessPipeline] = None

        self._build_ui()
        self._register_pipelines()
        self._show_placeholder()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)

        top_bar = ttk.Frame(container)
        top_bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        open_btn = ttk.Button(top_bar, text="Open .h5 file", command=self.open_file)
        open_btn.pack(side="left")
        self.file_label = ttk.Label(top_bar, text="No file loaded", wraplength=500)
        self.file_label.pack(side="left", padx=8)

        ttk.Label(container, text="Pipeline").grid(row=1, column=0, sticky="w")
        self.pipeline_var = tk.StringVar()
        self.pipeline_combo = ttk.Combobox(container, textvariable=self.pipeline_var, state="readonly", width=40)
        self.pipeline_combo.grid(row=1, column=1, sticky="w")
        run_btn = ttk.Button(container, text="Run pipeline", command=self.run_selected_pipeline)
        run_btn.grid(row=1, column=2, sticky="w", padx=6)

        ttk.Label(container, text="Result").grid(row=2, column=0, sticky="nw", pady=(8, 2))
        output_frame = ttk.Frame(container)
        output_frame.grid(row=2, column=1, columnspan=2, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.process_output = tk.Text(output_frame, height=18, state="disabled")
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.process_output.yview)
        self.process_output.configure(yscrollcommand=output_scroll.set)
        self.process_output.grid(row=0, column=0, sticky="nsew")
        output_scroll.grid(row=0, column=1, sticky="ns")

        export_frame = ttk.Frame(container, padding=(0, 8, 0, 0))
        export_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
        export_frame.columnconfigure(1, weight=1)
        ttk.Label(export_frame, text="Export CSV").grid(row=0, column=0, sticky="w")
        self.export_path_var = tk.StringVar(value="process_result.csv")
        export_entry = ttk.Entry(export_frame, textvariable=self.export_path_var)
        export_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(export_frame, text="Browse", command=self.choose_export_path).grid(row=0, column=2, sticky="w")
        ttk.Button(export_frame, text="Export", command=self.export_process_result).grid(row=0, column=3, sticky="w", padx=6)

    def _register_pipelines(self) -> None:
        pipelines = [BasicStatsPipeline(), VelocityComparisonPipeline()]
        self.pipeline_registry = {p.name: p for p in pipelines}
        self.pipeline_combo["values"] = list(self.pipeline_registry.keys())
        if pipelines:
            self.pipeline_combo.current(0)
            self.pipeline_var.set(pipelines[0].name)

    def _show_placeholder(self, message: str = "Load a .h5 file then run a pipeline") -> None:
        self.process_output.configure(state="normal")
        self.process_output.delete("1.0", "end")
        self.process_output.insert("end", message)
        self.process_output.configure(state="disabled")

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
        self._show_placeholder("File loaded. Pick a pipeline and run.")

    def run_selected_pipeline(self) -> None:
        name = self.pipeline_var.get()
        if not name:
            messagebox.showwarning("Missing pipeline", "Select a pipeline before running.")
            return
        pipeline = self.pipeline_registry.get(name)
        if pipeline is None:
            messagebox.showerror("Pipeline missing", f"Pipeline '{name}' is not registered.")
            return
        if self.h5_file is None:
            messagebox.showwarning("Missing file", "Load a .h5 file first.")
            return
        try:
            result = pipeline.run(self.h5_file)
            output_path = self._default_output_path(name)
            self._write_result_h5(result, output_path, pipeline_name=name)
            result.output_h5_path = output_path
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Pipeline error", f"Pipeline failed: {exc}")
            return
        self.last_process_result = result
        self.last_process_pipeline = pipeline
        file_label = self.h5_file.filename or "(in-memory file)"
        self._render_process_result(result, pipeline_name=name, file_path=file_label)

    def _render_process_result(self, result: ProcessResult, pipeline_name: str, file_path: str) -> None:
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
            self.process_output.insert("end", f"\nResult HDF5: {result.output_h5_path}\n")
        self.process_output.configure(state="disabled")

    def choose_export_path(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=self.export_path_var.get(),
        )
        if path:
            self.export_path_var.set(path)

    def export_process_result(self) -> None:
        if self.last_process_result is None or self.last_process_pipeline is None:
            messagebox.showwarning("No result", "Run a pipeline before exporting.")
            return
        output_path = self.export_path_var.get() or "process_result.csv"
        try:
            final_path = self.last_process_pipeline.export(self.last_process_result, output_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", f"Cannot export: {exc}")
            return
        messagebox.showinfo("Export done", f"Result exported to: {final_path}")

    def _default_output_path(self, pipeline_name: str) -> str:
        safe_name = pipeline_name.lower().replace(" ", "_")
        base = Path(self.h5_file.filename).stem if self.h5_file and self.h5_file.filename else "output"
        return str(Path.cwd() / f"{base}_{safe_name}_result.h5")

    def _write_result_h5(self, result: ProcessResult, path: str, pipeline_name: str) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["pipeline"] = pipeline_name
            if self.h5_file and self.h5_file.filename:
                f.attrs["source_file"] = self.h5_file.filename
            metrics_grp = f.create_group("metrics")
            for key, value in result.metrics.items():
                metrics_grp.create_dataset(key, data=value)
            if result.artifacts:
                artifacts_grp = f.create_group("artifacts")
                for key, value in result.artifacts.items():
                    artifacts_grp.create_dataset(key, data=value)


if __name__ == "__main__":
    app = ProcessApp()
    app.mainloop()
