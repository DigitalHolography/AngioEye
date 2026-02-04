import os
import tempfile
import tkinter as tk
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import h5py

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

from pipelines import PipelineDescriptor, ProcessResult, load_pipeline_catalog
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
        self.pipeline_registry: dict[str, PipelineDescriptor] = {}
        self.available_pipelines: list[PipelineDescriptor] = []
        self.missing_pipelines: list[PipelineDescriptor] = []
        self.pipeline_check_vars: dict[str, tk.BooleanVar] = {}
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.scheduler_jobs: list[dict] = []
        self.scheduler_running = False

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
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        batch_tab = ttk.Frame(notebook, padding=10)
        scheduler_tab = ttk.Frame(notebook, padding=10)
        notebook.add(batch_tab, text="Batch")
        notebook.add(scheduler_tab, text="Scheduler")

        self._build_batch_tab(batch_tab)
        self._build_scheduler_tab(scheduler_tab)

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

    def _build_scheduler_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        actions = ttk.Frame(parent)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(actions, text="Add job", command=self._add_scheduler_job).pack(
            side="left"
        )
        ttk.Button(actions, text="Run queue", command=self.run_scheduler_queue).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Clear jobs", command=self._clear_scheduler_jobs).pack(
            side="left", padx=(8, 0)
        )
        self.scheduler_status = ttk.Label(actions, text="Idle")
        self.scheduler_status.pack(side="left", padx=(12, 0))

        jobs_wrapper = ttk.Frame(parent)
        jobs_wrapper.grid(row=1, column=0, sticky="nsew")
        jobs_wrapper.columnconfigure(0, weight=1)
        jobs_wrapper.rowconfigure(0, weight=1)

        self.scheduler_canvas = tk.Canvas(
            jobs_wrapper, highlightthickness=0, height=260, bg=self._bg_color
        )
        self.scheduler_canvas.grid(row=0, column=0, sticky="nsew")
        jobs_scroll = ttk.Scrollbar(
            jobs_wrapper, orient="vertical", command=self.scheduler_canvas.yview
        )
        jobs_scroll.grid(row=0, column=1, sticky="ns")
        self.scheduler_canvas.configure(yscrollcommand=jobs_scroll.set)
        self.scheduler_jobs_frame = ttk.Frame(self.scheduler_canvas)
        self.scheduler_jobs_window = self.scheduler_canvas.create_window(
            (0, 0), window=self.scheduler_jobs_frame, anchor="nw"
        )
        self.scheduler_jobs_frame.bind(
            "<Configure>",
            lambda _evt: self.scheduler_canvas.configure(
                scrollregion=self.scheduler_canvas.bbox("all")
            ),
        )
        self.scheduler_canvas.bind(
            "<Configure>",
            lambda evt: self.scheduler_canvas.itemconfigure(
                self.scheduler_jobs_window, width=evt.width
            ),
        )

        ttk.Label(parent, text="Scheduler log").grid(
            row=2, column=0, sticky="w", pady=(8, 2)
        )
        log_frame = ttk.Frame(parent)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.scheduler_log = tk.Text(
            log_frame,
            height=12,
            state="disabled",
            bg=self._text_bg,
            fg=self._text_fg,
            insertbackground=self._text_fg,
        )
        sched_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.scheduler_log.yview
        )
        self.scheduler_log.configure(yscrollcommand=sched_scroll.set)
        self.scheduler_log.grid(row=0, column=0, sticky="nsew")
        sched_scroll.grid(row=0, column=1, sticky="ns")

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        self.available_pipelines = available
        self.missing_pipelines = missing
        self.pipeline_registry = {p.name: p for p in available}
        self._populate_pipeline_checks(available, missing)

    def _populate_pipeline_checks(
        self, available: list[PipelineDescriptor], missing: list[PipelineDescriptor]
    ) -> None:
        for child in self.pipeline_checks_inner.winfo_children():
            child.destroy()
        self.pipeline_check_vars = {}
        rows: list[PipelineDescriptor] = [*available, *missing]
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

    def _build_pipeline_checklist(
        self, parent: ttk.Frame
    ) -> dict[str, tk.BooleanVar]:
        """
        Create a set of pipeline checkbuttons inside parent and return the associated variables.
        """
        vars_map: dict[str, tk.BooleanVar] = {}
        rows: list[PipelineDescriptor] = [*self.available_pipelines, *self.missing_pipelines]
        for idx, pipeline in enumerate(rows):
            is_available = getattr(pipeline, "available", True)
            var = tk.BooleanVar(value=is_available)
            var._enabled = is_available  # type: ignore[attr-defined]
            label = pipeline.name if is_available else f"{pipeline.name} (missing deps)"
            state = "normal" if is_available else "disabled"
            check = ttk.Checkbutton(
                parent, text=label, variable=var, state=state, padding=(0, 2)
            )
            check.grid(row=idx, column=0, sticky="w", padx=(0, 6))
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
            vars_map[pipeline.name] = var
        return vars_map

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

    def _log_scheduler(self, text: str) -> None:
        self.scheduler_log.configure(state="normal")
        self.scheduler_log.insert("end", f"{text}\n")
        self.scheduler_log.see("end")
        self.scheduler_log.configure(state="disabled")
        self.scheduler_log.update_idletasks()
        self.update_idletasks()

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

        self._reset_batch_output("Starting batch run...\n")
        self._run_batch_job(
            data_path=data_path,
            pipelines=pipelines,
            base_output_dir=base_output_dir,
            zip_outputs=self.batch_zip_var.get(),
            zip_name=self.batch_zip_name_var.get(),
            log_fn=self._log_batch,
            show_dialogs=True,
        )

    def _run_batch_job(
        self,
        data_path: Path,
        pipelines: Sequence[PipelineDescriptor],
        base_output_dir: Path,
        zip_outputs: bool,
        zip_name: str | None,
        log_fn: Callable[[str], None],
        show_dialogs: bool = True,
    ) -> tuple[bool, str]:
        if not pipelines:
            if show_dialogs:
                messagebox.showwarning("No pipelines", "Select at least one pipeline.")
            log_fn("[SKIP] No pipelines selected.")
            return False, "No pipelines selected."

        base_output_dir = base_output_dir.expanduser()
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        base_output_dir.mkdir(parents=True, exist_ok=True)

        tempdir: tempfile.TemporaryDirectory | None = None
        temp_output_dir: tempfile.TemporaryDirectory | None = None
        clean_temp_output = False

        try:
            data_root, tempdir = self._prepare_data_root(data_path)
            inputs = self._find_h5_inputs(data_root)
        except Exception as exc:  # noqa: BLE001
            if show_dialogs:
                messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
            log_fn(f"Error: {exc}")
            if tempdir is not None:
                tempdir.cleanup()
            return False, f"Input error: {exc}"

        output_dir = base_output_dir
        if zip_outputs:
            temp_output_dir = tempfile.TemporaryDirectory(dir=base_output_dir)
            output_dir = Path(temp_output_dir.name)

        failures: list[str] = []
        for h5_path in inputs:
            try:
                self._run_pipelines_on_file(
                    h5_path, pipelines, output_dir, log_fn=log_fn
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                log_fn(f"[FAIL] {h5_path.name}: {exc}")

        summary_msg: str
        if zip_outputs:
            try:
                archive_name = (zip_name or "outputs.zip").strip() or "outputs.zip"
                if not archive_name.lower().endswith(".zip"):
                    archive_name += ".zip"
                zip_path = self._zip_output_dir(
                    output_dir, target_path=base_output_dir / archive_name
                )
                log_fn(f"[ZIP] Archive created: {zip_path}")
                summary_msg = f"ZIP archive: {zip_path}"
                clean_temp_output = True
            except Exception as exc:  # noqa: BLE001
                log_fn(f"[ZIP FAIL] {exc}")
                if show_dialogs:
                    messagebox.showerror(
                        "Zip failed", f"Could not create ZIP archive: {exc}"
                    )
                summary_msg = f"Outputs stored under: {output_dir}"
        else:
            summary_msg = f"Outputs stored under: {output_dir}"

        log_fn(f"Completed. {summary_msg}")

        if show_dialogs:
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
        return (len(failures) == 0), summary_msg

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

    def _add_scheduler_job(self) -> None:
        job_idx = len(self.scheduler_jobs) + 1
        frame = ttk.Labelframe(
            self.scheduler_jobs_frame, text=f"Job {job_idx}", padding=8
        )
        frame.columnconfigure(1, weight=1)

        input_var = tk.StringVar()
        output_var = tk.StringVar(value=str(Path.cwd()))
        zip_var = tk.BooleanVar(value=False)
        zip_name_var = tk.StringVar(value="outputs.zip")
        status_var = tk.StringVar(value="Idle")

        # Input row
        ttk.Label(frame, text="Input").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(frame, textvariable=input_var)
        input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        input_btns = ttk.Frame(frame)
        input_btns.grid(row=0, column=2, sticky="w")
        ttk.Button(
            input_btns,
            text="Folder",
            command=lambda: self._scheduler_choose_folder(input_var),
        ).pack(side="left")
        ttk.Button(
            input_btns,
            text="File/zip",
            command=lambda: self._scheduler_choose_file(input_var),
        ).pack(side="left", padx=(4, 0))

        # Output row
        ttk.Label(frame, text="Output").grid(row=1, column=0, sticky="w", pady=(6, 0))
        output_entry = ttk.Entry(frame, textvariable=output_var)
        output_entry.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=(6, 0))
        ttk.Button(
            frame,
            text="Browse",
            command=lambda: self._scheduler_choose_output(output_var),
        ).grid(row=1, column=2, sticky="w", pady=(6, 0))

        # Pipelines
        ttk.Label(frame, text="Pipelines").grid(
            row=2, column=0, sticky="nw", pady=(6, 0)
        )
        pipeline_wrapper = ttk.Frame(frame)
        pipeline_wrapper.grid(row=2, column=1, columnspan=2, sticky="nsew", pady=(6, 4))
        pipeline_wrapper.columnconfigure(0, weight=1)
        pipeline_wrapper.rowconfigure(1, weight=1)

        pipeline_actions = ttk.Frame(pipeline_wrapper)
        pipeline_actions.grid(row=0, column=0, sticky="e")
        # vars set after creation so lambdas can close over it
        pipeline_container = ttk.Frame(pipeline_wrapper)
        pipeline_container.grid(row=1, column=0, sticky="nsew")
        pipeline_container.columnconfigure(0, weight=1)
        pipeline_vars = self._build_pipeline_checklist(pipeline_container)
        ttk.Button(
            pipeline_actions,
            text="Select all",
            command=lambda: self._set_scheduler_pipeline_vars(pipeline_vars, True),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            pipeline_actions,
            text="Clear all",
            command=lambda: self._set_scheduler_pipeline_vars(pipeline_vars, False),
        ).pack(side="left")

        # Zip row
        ttk.Checkbutton(
            frame,
            text="Zip outputs after run",
            variable=zip_var,
            command=lambda: self._toggle_scheduler_zip_for_job(job),
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))
        zip_label = ttk.Label(frame, text="Archive name")
        zip_label.grid(row=3, column=1, sticky="w", pady=(6, 0))
        zip_entry = ttk.Entry(frame, textvariable=zip_name_var, width=24)
        zip_entry.grid(row=3, column=2, sticky="w", pady=(6, 0))

        # Footer row with status and remove
        status_label = ttk.Label(frame, textvariable=status_var, foreground=self._muted_fg)
        status_label.grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Button(
            frame,
            text="Remove",
            command=lambda: self._remove_scheduler_job(job),
        ).grid(row=4, column=2, sticky="e", pady=(6, 0))

        job = {
            "frame": frame,
            "input_var": input_var,
            "output_var": output_var,
            "zip_var": zip_var,
            "zip_name_var": zip_name_var,
            "pipeline_vars": pipeline_vars,
            "status_var": status_var,
            "zip_label": zip_label,
            "zip_entry": zip_entry,
        }
        self.scheduler_jobs.append(job)
        frame.grid(row=len(self.scheduler_jobs) - 1, column=0, sticky="ew", pady=(0, 10))
        self._toggle_scheduler_zip_for_job(job)
        self._relabel_scheduler_jobs()

    def _set_scheduler_pipeline_vars(
        self, pipeline_vars: dict[str, tk.BooleanVar], value: bool
    ) -> None:
        for var in pipeline_vars.values():
            if getattr(var, "_enabled", True):
                var.set(value)

    def _toggle_scheduler_zip_for_job(self, job: dict) -> None:
        if job["zip_var"].get():
            job["zip_label"].grid()
            job["zip_entry"].grid()
        else:
            job["zip_label"].grid_remove()
            job["zip_entry"].grid_remove()

    def _remove_scheduler_job(self, job: dict) -> None:
        if job not in self.scheduler_jobs:
            return
        job["frame"].destroy()
        self.scheduler_jobs.remove(job)
        self._relabel_scheduler_jobs()

    def _clear_scheduler_jobs(self) -> None:
        for job in list(self.scheduler_jobs):
            job["frame"].destroy()
        self.scheduler_jobs.clear()
        self._relabel_scheduler_jobs()

    def _relabel_scheduler_jobs(self) -> None:
        for idx, job in enumerate(self.scheduler_jobs, start=1):
            job["frame"].configure(text=f"Job {idx}")
            job["frame"].grid(row=idx - 1, column=0, sticky="ew", pady=(0, 10))
        self.scheduler_jobs_frame.update_idletasks()
        self.scheduler_canvas.configure(
            scrollregion=self.scheduler_canvas.bbox("all")
        )

    def _scheduler_choose_folder(self, target_var: tk.StringVar) -> None:
        path = filedialog.askdirectory(
            initialdir=target_var.get() or os.path.abspath("h5_example"),
            title="Select folder containing HDF5 files",
        )
        if path:
            target_var.set(path)

    def _scheduler_choose_file(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("HDF5 or zip", "*.h5 *.hdf5 *.zip"), ("All files", "*.*")],
            initialdir=target_var.get() or os.path.abspath("h5_example"),
            title="Select HDF5 file or .zip archive",
        )
        if path:
            target_var.set(path)

    def _scheduler_choose_output(self, target_var: tk.StringVar) -> None:
        path = filedialog.askdirectory(
            initialdir=target_var.get() or None,
            title="Select base output folder",
        )
        if path:
            target_var.set(path)

    def run_scheduler_queue(self) -> None:
        if self.scheduler_running:
            return
        if not self.scheduler_jobs:
            messagebox.showwarning(
                "No jobs", "Add at least one job to the queue before running."
            )
            return

        self.scheduler_running = True
        self.scheduler_status.configure(text="Running")
        self._log_scheduler("Starting scheduler queue...")

        for idx, job in enumerate(list(self.scheduler_jobs), start=1):
            job["status_var"].set("Running")
            self.update_idletasks()
            success = self._execute_scheduler_job(job, idx)
            if success:
                job["status_var"].set("Done")
            elif job["status_var"].get() == "Running":
                job["status_var"].set("Failed")

        self.scheduler_running = False
        self.scheduler_status.configure(text="Idle")
        self._log_scheduler("Queue finished.")

    def _execute_scheduler_job(self, job: dict, job_index: int) -> bool:
        def job_log(message: str) -> None:
            self._log_scheduler(f"[Job {job_index}] {message}")

        data_value = (job["input_var"].get() or "").strip()
        if not data_value:
            job["status_var"].set("Skipped")
            job_log("Skipped: no input selected.")
            return False
        data_path = Path(data_value).expanduser()

        selected_names = [
            name for name, var in job["pipeline_vars"].items() if var.get()
        ]
        if not selected_names:
            job["status_var"].set("Skipped")
            job_log("Skipped: no pipelines selected.")
            return False

        pipelines: list[PipelineDescriptor] = []
        missing: list[str] = []
        for name in selected_names:
            pipeline = self.pipeline_registry.get(name)
            if pipeline is None:
                missing.append(name)
            else:
                pipelines.append(pipeline)
        if missing:
            job["status_var"].set("Failed")
            job_log(f"Missing pipeline(s): {', '.join(missing)}")
            return False

        base_output_value = (job["output_var"].get() or "").strip()
        base_output_dir = (
            Path(base_output_value).expanduser()
            if base_output_value
            else Path.cwd()
        )

        job_log(f"Starting: {data_path}")
        success, summary = self._run_batch_job(
            data_path=data_path,
            pipelines=pipelines,
            base_output_dir=base_output_dir,
            zip_outputs=job["zip_var"].get(),
            zip_name=job["zip_name_var"].get(),
            log_fn=job_log,
            show_dialogs=False,
        )
        job_log(f"Finished: {summary}")
        return success

    def _run_pipelines_on_file(
        self,
        h5_path: Path,
        pipelines: Sequence[PipelineDescriptor],
        output_root: Path,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        logger = log_fn or self._log_batch
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
                result = pipeline.run(h5file)
                pipeline_results.append((pipeline.name, result))
                logger(f"[OK] {h5_path.name} -> {pipeline.name}")
        write_combined_results_h5(
            pipeline_results, combined_h5_out, source_file=str(h5_path)
        )
        for _, result in pipeline_results:
            result.output_h5_path = str(combined_h5_out)
        logger(f"[OK] {h5_path.name}: combined results -> {combined_h5_out}")

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
