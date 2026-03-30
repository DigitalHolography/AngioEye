import os
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import h5py

from app_settings import (
    AppSettingsStore,
    normalize_pipeline_visibility,
    normalize_postprocess_visibility,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

from pipelines import PipelineDescriptor, ProcessResult, load_pipeline_catalog
from pipelines.core.errors import format_pipeline_exception
from pipelines.core.utils import write_combined_results_h5
from postprocess import (
    PostprocessContext,
    PostprocessDescriptor,
    load_postprocess_catalog,
)

_BaseAppTk = TkinterDnD.Tk if TkinterDnD is not None else tk.Tk


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


class ProcessApp(_BaseAppTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AngioEye")
        self.settings_store = AppSettingsStore()
        self.ui_mode = self.settings_store.load_ui_mode()
        self.pipeline_registry: dict[str, PipelineDescriptor] = {}
        self.pipeline_catalog: dict[str, PipelineDescriptor] = {}
        self.pipeline_rows: list[PipelineDescriptor] = []
        self.pipeline_check_vars: dict[str, tk.BooleanVar] = {}
        self.pipeline_visibility: dict[str, bool] = {}
        self.pipeline_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.postprocess_registry: dict[str, PostprocessDescriptor] = {}
        self.postprocess_catalog: dict[str, PostprocessDescriptor] = {}
        self.postprocess_rows: list[PostprocessDescriptor] = []
        self.postprocess_check_vars: dict[str, tk.BooleanVar] = {}
        self.postprocess_visibility: dict[str, bool] = {}
        self.postprocess_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.pipeline_library_summary_var = tk.StringVar(value="")
        self.postprocess_library_summary_var = tk.StringVar(value="")
        self.minimal_input_path_var = tk.StringVar(value="No input selected")
        self.minimal_output_path_var = tk.StringVar(value=str(Path.cwd()))
        self.minimal_output_name_var = tk.StringVar(value="Output name: -")
        self._settings_warning_shown = False
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self._window_icon_image: tk.PhotoImage | None = None
        self._minimal_logo_image: tk.PhotoImage | None = None

        self._set_initial_window_size()
        self._apply_theme()
        self._set_window_icon()
        self._build_ui()
        self._install_drop_targets()
        self.batch_input_var.trace_add("write", self._on_batch_paths_changed)
        self.batch_output_var.trace_add("write", self._on_batch_paths_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._register_pipelines()
        self._register_postprocesses()
        self._reset_batch_output()
        self._update_minimal_path_labels()
        self._apply_ui_mode(self.ui_mode, persist=False)

    def _set_initial_window_size(self) -> None:
        width, height, min_width, min_height = self._window_size_for_mode(
            self.ui_mode
        )
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(width, screen_width)
        height = min(height, screen_height)
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min_width, min_height)

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
        self._build_menu()

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = ttk.Frame(container, padding=10)
        self.advanced_view = ttk.Frame(container, padding=10)

        self._build_minimal_view(self.minimal_view)
        self._build_advanced_view(self.advanced_view)

    def _build_menu(self) -> None:
        self.ui_mode_var = tk.StringVar(value=self.ui_mode)
        menu_bar = tk.Menu(self)
        view_menu = tk.Menu(menu_bar, tearoff=False)
        view_menu.add_radiobutton(
            label="Minimal UI",
            value="minimal",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        view_menu.add_radiobutton(
            label="Advanced UI",
            value="advanced",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        menu_bar.add_cascade(label="View", menu=view_menu)
        self.configure(menu=menu_bar)

    def _build_minimal_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.grid_anchor("center")

        content = ttk.Frame(parent, padding=(24, 24))
        content.grid(row=0, column=0)
        content.columnconfigure(0, minsize=420)
        self.minimal_content = content

        self.minimal_title_label = ttk.Label(content, text="AngioEye")
        self.minimal_title_label.grid(row=0, column=0, pady=(0, 10))

        minimal_logo = self._load_scaled_logo_image(max_width=360, max_height=144)
        if minimal_logo is not None:
            self._minimal_logo_image = minimal_logo
            self.minimal_logo_label = ttk.Label(content, image=self._minimal_logo_image)
            self.minimal_logo_label.grid(row=1, column=0, pady=(0, 18))

        self.minimal_browse_button = ttk.Button(
            content,
            text="Browse .h5 or zip archive",
            command=self.choose_batch_file,
        )
        self.minimal_browse_button.grid(row=2, column=0, pady=(0, 10))
        self.minimal_input_path_label = tk.Label(
            content,
            textvariable=self.minimal_input_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_input_path_label.grid(row=3, column=0, pady=(0, 18), sticky="ew")

        self.minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=self.choose_batch_output,
        )
        self.minimal_output_button.grid(row=4, column=0, pady=(0, 10))
        self.minimal_output_path_label = tk.Label(
            content,
            textvariable=self.minimal_output_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_path_label.grid(
            row=5, column=0, pady=(0, 6), sticky="ew"
        )
        self.minimal_output_name_label = tk.Label(
            content,
            textvariable=self.minimal_output_name_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_name_label.grid(
            row=6, column=0, pady=(0, 18), sticky="ew"
        )

        self.minimal_run_button = ttk.Button(content, text="Run", command=self.run_batch)
        self.minimal_run_button.grid(row=7, column=0, pady=(0, 18))

        self.minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.batch_progress_var,
            length=340,
        )
        self.minimal_progress.grid(row=8, column=0, sticky="ew")

    def _build_advanced_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.batch_tab = ttk.Frame(self.notebook, padding=10)
        self.export_tab = ttk.Frame(self.notebook, padding=10)
        self.pipeline_library_tab = ttk.Frame(self.notebook, padding=10)
        self.postprocess_library_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.batch_tab, text="Run")
        self.notebook.add(self.export_tab, text="Export")
        self.notebook.add(self.pipeline_library_tab, text="Pipeline Library")
        self.notebook.add(self.postprocess_library_tab, text="Postprocess Library")

        self._build_batch_tab(self.batch_tab)
        self._build_export_tab(self.export_tab)
        self._build_pipeline_library_tab(self.pipeline_library_tab)
        self._build_postprocess_library_tab(self.postprocess_library_tab)

    def _install_drop_targets(self) -> None:
        if DND_FILES is None:
            return
        self._register_drop_target_tree(self)

    def _register_drop_target_tree(self, widget: tk.Misc) -> None:
        if DND_FILES is None:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_input_drop)
        except (AttributeError, tk.TclError):
            pass

        for child in widget.winfo_children():
            self._register_drop_target_tree(child)

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)
        parent.columnconfigure(3, weight=0)
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

        ttk.Label(parent, text="Postprocess").grid(
            row=2, column=0, sticky="nw", pady=(8, 0)
        )
        postprocess_wrapper = ttk.Frame(parent)
        postprocess_wrapper.grid(
            row=2, column=1, columnspan=2, sticky="nsew", pady=(8, 0)
        )
        postprocess_wrapper.columnconfigure(0, weight=1)
        postprocess_wrapper.rowconfigure(1, weight=1)

        postprocess_actions = ttk.Frame(postprocess_wrapper)
        postprocess_actions.grid(row=0, column=0, sticky="e", pady=(0, 4))
        ttk.Button(
            postprocess_actions,
            text="Select all",
            command=self.select_all_postprocesses,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            postprocess_actions,
            text="Clear all",
            command=self.clear_all_postprocesses,
        ).pack(side="left")
        ttk.Button(
            postprocess_actions,
            text="Postprocess library",
            command=self.open_postprocess_library,
        ).pack(side="left", padx=(4, 0))

        postprocess_container = ttk.Frame(postprocess_wrapper)
        postprocess_container.grid(row=1, column=0, sticky="nsew")
        postprocess_container.columnconfigure(0, weight=1)
        postprocess_container.rowconfigure(0, weight=1)

        self.postprocess_checks_canvas = tk.Canvas(
            postprocess_container,
            highlightthickness=0,
            height=140,
            bg=self._bg_color,
        )
        self.postprocess_checks_canvas.grid(row=0, column=0, sticky="nsew")
        postprocess_scroll = ttk.Scrollbar(
            postprocess_container,
            orient="vertical",
            command=self.postprocess_checks_canvas.yview,
        )
        postprocess_scroll.grid(row=0, column=1, sticky="ns")
        self.postprocess_checks_canvas.configure(
            yscrollcommand=postprocess_scroll.set
        )
        self.postprocess_checks_inner = ttk.Frame(self.postprocess_checks_canvas)
        self.postprocess_checks_window = self.postprocess_checks_canvas.create_window(
            (0, 0), window=self.postprocess_checks_inner, anchor="nw"
        )
        self.postprocess_checks_inner.bind(
            "<Configure>",
            lambda _evt: self.postprocess_checks_canvas.configure(
                scrollregion=self.postprocess_checks_canvas.bbox("all")
            ),
        )
        self.postprocess_checks_canvas.bind(
            "<Configure>",
            lambda evt: self.postprocess_checks_canvas.itemconfigure(
                self.postprocess_checks_window, width=evt.width
            ),
        )

        controls = ttk.Frame(parent)
        controls.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        controls.columnconfigure(3, weight=1)

        run_btn = ttk.Button(controls, text="Run batch", command=self.run_batch)
        run_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Open export tab",
            command=self.open_export_tab,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.advanced_progress = ttk.Progressbar(
            controls,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.batch_progress_var,
            length=320,
        )
        self.advanced_progress.grid(row=0, column=3, sticky="e")

        ttk.Label(parent, text="Batch log").grid(
            row=4, column=0, sticky="nw", pady=(8, 2)
        )
        batch_output_frame = ttk.Frame(parent)
        batch_output_frame.grid(row=4, column=1, columnspan=3, sticky="nsew")
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

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(
            parent,
            text=(
                "Export settings stay hidden from the minimal UI. "
                "For a loaded .zip, the default archive is recreated next to the input."
            ),
            wraplength=760,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(parent, text="Output folder").grid(row=1, column=0, sticky="w")
        batch_output_entry = ttk.Entry(parent, textvariable=self.batch_output_var)
        batch_output_entry.grid(row=1, column=1, sticky="ew", padx=(0, 4))
        ttk.Button(parent, text="Browse", command=self.choose_batch_output).grid(
            row=1, column=2, sticky="w"
        )

        ttk.Checkbutton(
            parent,
            text="Zip outputs after run",
            variable=self.batch_zip_var,
            command=self._toggle_zip_name_visibility,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))

        self.batch_zip_label = ttk.Label(parent, text="Archive name")
        self.batch_zip_label.grid(row=3, column=0, sticky="w", pady=(2, 0))
        self.batch_zip_entry = ttk.Entry(
            parent, textvariable=self.batch_zip_name_var, width=28
        )
        self.batch_zip_entry.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(2, 0)
        )

        ttk.Button(
            parent,
            text="Export batch log",
            command=self._export_batch_log,
        ).grid(row=4, column=0, sticky="w", pady=(16, 0))
        self._toggle_zip_name_visibility()

    def _resource_roots(self) -> list[Path]:
        roots: list[Path] = []
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            roots.append(Path(frozen_root))
        roots.append(Path(__file__).resolve().parents[1])
        roots.append(Path.cwd())
        return roots

    def _resolve_logo_path(self) -> Path | None:
        for root in self._resource_roots():
            candidate = root / "Angioeye_logo.png"
            if candidate.is_file():
                return candidate
        return None

    def _load_logo_image(self) -> tk.PhotoImage | None:
        logo_path = self._resolve_logo_path()
        if logo_path is None:
            return None
        try:
            return tk.PhotoImage(file=str(logo_path))
        except tk.TclError:
            return None

    def _load_scaled_logo_image(
        self,
        *,
        max_width: int,
        max_height: int,
    ) -> tk.PhotoImage | None:
        image = self._load_logo_image()
        if image is None:
            return None

        scale_x = max(1, (image.width() + max_width - 1) // max_width)
        scale_y = max(1, (image.height() + max_height - 1) // max_height)
        scale = max(scale_x, scale_y)
        if scale > 1:
            image = image.subsample(scale, scale)
        return image

    def _set_window_icon(self) -> None:
        image = self._load_logo_image()
        if image is None:
            return
        self._window_icon_image = image
        try:
            self.iconphoto(True, self._window_icon_image)
        except tk.TclError:
            pass

    def _show_settings_warning(self, title: str, details: str) -> None:
        if self._settings_warning_shown:
            return
        self._settings_warning_shown = True
        messagebox.showwarning(title, details)

    def _persist_ui_mode(self) -> None:
        try:
            self.settings_store.save_ui_mode(self.ui_mode)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save UI mode preference:\n{exc}",
            )

    def _window_size_for_mode(self, mode: str) -> tuple[int, int, int, int]:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        if mode == "advanced":
            width = max(980, min(1320, screen_width - 80))
            height = max(820, min(1020, screen_height - 80))
            min_width = min(980, width)
            min_height = min(760, height)
        else:
            width = max(560, min(660, screen_width - 260))
            height = max(420, min(520, screen_height - 260))
            min_width = min(500, width)
            min_height = min(360, height)
        return width, height, min_width, min_height

    def _ensure_window_size_for_mode(
        self,
        mode: str,
        *,
        force_target_size: bool = False,
    ) -> None:
        target_width, target_height, min_width, min_height = self._window_size_for_mode(
            mode
        )
        self.minsize(min_width, min_height)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        current_width = max(self.winfo_width(), 1)
        current_height = max(self.winfo_height(), 1)
        if (
            not force_target_size
            and current_width >= min_width
            and current_height >= min_height
        ):
            return

        if force_target_size:
            if mode == "minimal":
                try:
                    if self.state() != "normal":
                        self.state("normal")
                except tk.TclError:
                    pass
                self.minimal_view.update_idletasks()
                requested_width = self.minimal_view.winfo_reqwidth() + 24
                requested_height = self.minimal_view.winfo_reqheight() + 24
                width = min(
                    max(requested_width, min_width),
                    min(target_width, screen_width),
                )
                height = min(
                    max(requested_height, min_height),
                    min(target_height, screen_height),
                )
            else:
                width = min(target_width, screen_width)
                height = min(target_height, screen_height)
        else:
            width = min(max(current_width, min_width), screen_width)
            height = min(max(current_height, min_height), screen_height)
        x = max(min(self.winfo_x(), screen_width - width), 0)
        y = max(min(self.winfo_y(), screen_height - height), 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _apply_ui_mode(self, mode: str, *, persist: bool = True) -> None:
        normalized_mode = "advanced" if mode == "advanced" else "minimal"
        previous_mode = self.ui_mode
        self.ui_mode = normalized_mode
        self.ui_mode_var.set(normalized_mode)

        self.minimal_view.pack_forget()
        self.advanced_view.pack_forget()
        if normalized_mode == "advanced":
            self.advanced_view.pack(fill="both", expand=True)
        else:
            self.minimal_view.pack(fill="both", expand=True)

        self.update_idletasks()

        self._ensure_window_size_for_mode(
            normalized_mode,
            force_target_size=(
                normalized_mode == "minimal"
                and (previous_mode == "advanced" or not persist)
            ),
        )
        if persist:
            self._persist_ui_mode()

    def _on_close(self) -> None:
        self._persist_ui_mode()
        self.destroy()

    def _on_batch_paths_changed(self, *_args) -> None:
        self._update_minimal_path_labels()

    def _handle_dropped_paths(self, dropped_paths: Sequence[Path]) -> bool:
        for dropped_path in dropped_paths:
            if (
                dropped_path.is_file()
                and dropped_path.suffix.lower() in {".h5", ".hdf5", ".zip"}
            ):
                self.batch_input_var.set(str(dropped_path))
                self._apply_input_defaults(dropped_path)
                self._log_batch(f"[INPUT] Drag and drop -> {dropped_path}")
                return True
        return False

    def _on_input_drop(self, event) -> None:
        raw_data = getattr(event, "data", "")
        try:
            dropped_values = self.tk.splitlist(raw_data)
        except tk.TclError:
            dropped_values = (raw_data,)

        dropped_paths = [Path(value) for value in dropped_values if value]
        if self._handle_dropped_paths(dropped_paths):
            return

        messagebox.showwarning(
            "Unsupported drop",
            "Drop a single .h5, .hdf5, or .zip file into the window.",
        )

    def _default_output_stem(self, input_path: Path) -> str:
        if input_path.is_file():
            base_name = input_path.stem
        else:
            base_name = input_path.name
        base_name = base_name or "output"
        return f"{base_name}_angioeye"

    def _default_archive_name(self, input_path: Path) -> str:
        return f"{self._default_output_stem(input_path)}.zip"

    def _default_output_artifact_name(self, input_path: Path) -> str:
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            return self._default_archive_name(input_path)
        return f"{self._default_output_stem(input_path)}.h5"

    def _update_minimal_path_labels(self) -> None:
        raw_value = (self.batch_input_var.get() or "").strip()
        if not raw_value:
            self.minimal_input_path_var.set("No input selected")
            self.minimal_output_name_var.set("Output name: -")
        else:
            input_path = Path(raw_value)
            self.minimal_input_path_var.set(str(input_path))
            self.minimal_output_name_var.set(
                f"Output name: {self._default_output_artifact_name(input_path)}"
            )

        output_value = (self.batch_output_var.get() or "").strip()
        self.minimal_output_path_var.set(output_value or "No output folder selected")

    def _reset_progress(self) -> None:
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self.batch_progress_var.set(0.0)
        self.update_idletasks()

    def _start_progress(self, total_units: float) -> None:
        self._progress_total_units = max(float(total_units), 1.0)
        self._progress_completed_units = 0.0
        self.batch_progress_var.set(0.0)
        self.update_idletasks()

    def _set_progress_units(self, completed_units: float) -> None:
        clamped_units = min(
            max(float(completed_units), 0.0),
            max(self._progress_total_units, 1.0),
        )
        self._progress_completed_units = clamped_units
        self.batch_progress_var.set(
            (clamped_units / max(self._progress_total_units, 1.0)) * 100.0
        )
        self.update_idletasks()

    def _advance_progress(self, units: float = 1.0) -> None:
        self._set_progress_units(self._progress_completed_units + units)

    def _apply_input_defaults(self, input_path: Path) -> None:
        output_dir = input_path if input_path.is_dir() else input_path.parent

        self.batch_output_var.set(str(output_dir))
        self.batch_zip_name_var.set(self._default_archive_name(input_path))
        self.batch_zip_var.set(
            input_path.is_file() and input_path.suffix.lower() == ".zip"
        )
        self._toggle_zip_name_visibility()
        self._reset_progress()

    def _minimal_output_filename_for_run(
        self,
        data_path: Path,
        inputs: Sequence[Path],
    ) -> str | None:
        if self.ui_mode != "minimal":
            return None
        if self.batch_zip_var.get():
            return None
        if len(inputs) != 1:
            return None
        if not data_path.is_file():
            return None
        if data_path.suffix.lower() not in {".h5", ".hdf5"}:
            return None
        return self._default_output_artifact_name(data_path)

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
        ttk.Label(controls, textvariable=self.pipeline_library_summary_var).grid(
            row=0, column=4, sticky="e"
        )

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

    def _build_postprocess_library_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text="Choose which postprocess steps are shown in the Batch tab. "
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
            command=self.show_all_postprocesses_in_main_ui,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Hide all",
            command=self.hide_all_postprocesses_from_main_ui,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload postprocess",
            command=self.refresh_postprocess_catalog,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(
            controls, textvariable=self.postprocess_library_summary_var
        ).grid(row=0, column=4, sticky="e")

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

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        rows = sorted(
            [*available, *missing], key=lambda pipeline: pipeline.name.lower()
        )
        self.pipeline_registry = {p.name: p for p in available}
        self.pipeline_catalog = {p.name: p for p in rows}
        self.pipeline_rows = rows
        self._sync_pipeline_visibility(rows)
        selection_state = {
            name: var.get() for name, var in self.pipeline_check_vars.items()
        }
        self._populate_pipeline_checks(rows, selection_state)
        self._populate_pipeline_library(rows)
        self._install_drop_targets()

    def _register_postprocesses(self) -> None:
        available, missing = load_postprocess_catalog()
        rows = sorted(
            [*available, *missing], key=lambda postprocess: postprocess.name.lower()
        )
        self.postprocess_registry = {p.name: p for p in available}
        self.postprocess_catalog = {p.name: p for p in rows}
        self.postprocess_rows = rows
        self._sync_postprocess_visibility(rows)
        selection_state = {
            name: var.get() for name, var in self.postprocess_check_vars.items()
        }
        self._populate_postprocess_checks(rows, selection_state)
        self._populate_postprocess_library(rows)
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

    def _populate_pipeline_checks(
        self,
        rows: list[PipelineDescriptor],
        selection_state: dict[str, bool] | None = None,
    ) -> None:
        for child in self.pipeline_checks_inner.winfo_children():
            child.destroy()
        self.pipeline_check_vars = {}
        visible_rows = [
            row for row in rows if self.pipeline_visibility.get(row.name, False)
        ]
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
            label = pipeline.name if is_available else f"{pipeline.name} (unavailable)"
            state = "normal" if is_available else "disabled"
            check = ttk.Checkbutton(
                self.pipeline_checks_inner, text=label, variable=var, state=state
            )
            check.grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
            tip_text = self._descriptor_tooltip_text(pipeline)
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
            var = tk.BooleanVar(
                value=self.pipeline_visibility.get(pipeline.name, False)
            )
            check = ttk.Checkbutton(
                self.pipeline_library_inner,
                text=pipeline.name,
                variable=var,
                command=lambda name=pipeline.name, visible_var=var: (
                    self._set_pipeline_visibility(name, visible_var.get())
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))

            status_text = self._pipeline_status_text(pipeline)
            status = ttk.Label(self.pipeline_library_inner, text=status_text)
            status.grid(row=idx, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

            tip_text = self._descriptor_tooltip_text(pipeline)
            if tip_text:
                _Tooltip(check, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(status, tip_text, bg=self._surface_color, fg=self._text_fg)

            self.pipeline_visibility_vars[pipeline.name] = var

        self._update_pipeline_library_summary()

    def _populate_postprocess_checks(
        self,
        rows: list[PostprocessDescriptor],
        selection_state: dict[str, bool] | None = None,
    ) -> None:
        for child in self.postprocess_checks_inner.winfo_children():
            child.destroy()
        self.postprocess_check_vars = {}
        visible_rows = [
            row for row in rows if self.postprocess_visibility.get(row.name, False)
        ]
        if not visible_rows:
            ttk.Label(
                self.postprocess_checks_inner,
                text=(
                    "No postprocess steps are visible here. "
                    "Open Postprocess Library to enable some."
                ),
            ).grid(row=0, column=0, sticky="w", pady=(0, 6))
            return

        for idx, postprocess in enumerate(visible_rows):
            is_available = getattr(postprocess, "available", True)
            default_value = (
                selection_state.get(postprocess.name, is_available)
                if selection_state is not None
                else is_available
            )
            var = tk.BooleanVar(value=default_value if is_available else False)
            var._enabled = is_available  # type: ignore[attr-defined]
            label = (
                postprocess.name
                if is_available
                else f"{postprocess.name} (unavailable)"
            )
            state = "normal" if is_available else "disabled"
            check = ttk.Checkbutton(
                self.postprocess_checks_inner, text=label, variable=var, state=state
            )
            check.grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
            tip_text = self._descriptor_tooltip_text(postprocess)
            if tip_text:
                _Tooltip(
                    check,
                    tip_text,
                    bg=self._surface_color,
                    fg=self._text_fg,
                )
            self.postprocess_check_vars[postprocess.name] = var

    def _populate_postprocess_library(
        self, rows: list[PostprocessDescriptor]
    ) -> None:
        for child in self.postprocess_library_inner.winfo_children():
            child.destroy()
        self.postprocess_visibility_vars = {}
        self.postprocess_library_inner.columnconfigure(0, weight=1)

        ttk.Label(self.postprocess_library_inner, text="Show in Batch").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(self.postprocess_library_inner, text="Status").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
        )

        for idx, postprocess in enumerate(rows, start=1):
            var = tk.BooleanVar(
                value=self.postprocess_visibility.get(postprocess.name, False)
            )
            check = ttk.Checkbutton(
                self.postprocess_library_inner,
                text=postprocess.name,
                variable=var,
                command=lambda name=postprocess.name, visible_var=var: self._set_postprocess_visibility(
                    name, visible_var.get()
                ),
            )
            check.grid(row=idx, column=0, sticky="w", pady=(0, 6))

            status_text = self._postprocess_status_text(postprocess)
            status = ttk.Label(self.postprocess_library_inner, text=status_text)
            status.grid(row=idx, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

            tip_text = self._descriptor_tooltip_text(postprocess)
            if tip_text:
                _Tooltip(check, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(status, tip_text, bg=self._surface_color, fg=self._text_fg)

            self.postprocess_visibility_vars[postprocess.name] = var

        self._update_postprocess_library_summary()

    def _sync_pipeline_visibility(self, rows: list[PipelineDescriptor]) -> None:
        visibility, changed = normalize_pipeline_visibility(
            (pipeline.name for pipeline in rows),
            self.settings_store.load_pipeline_visibility(),
        )
        self.pipeline_visibility = visibility
        if changed:
            self._persist_pipeline_visibility()

    def _sync_postprocess_visibility(self, rows: list[PostprocessDescriptor]) -> None:
        visibility, changed = normalize_postprocess_visibility(
            (postprocess.name for postprocess in rows),
            self.settings_store.load_postprocess_visibility(),
        )
        self.postprocess_visibility = visibility
        if changed:
            self._persist_postprocess_visibility()

    def _persist_pipeline_visibility(self) -> None:
        try:
            self.settings_store.save_pipeline_visibility(self.pipeline_visibility)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save pipeline visibility preferences:\n{exc}",
            )

    def _persist_postprocess_visibility(self) -> None:
        try:
            self.settings_store.save_postprocess_visibility(self.postprocess_visibility)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save postprocess visibility preferences:\n{exc}",
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

    def _set_postprocess_visibility(self, name: str, visible: bool) -> None:
        if self.postprocess_visibility.get(name) == visible:
            return
        self.postprocess_visibility[name] = visible
        self._persist_postprocess_visibility()
        selection_state = {
            postprocess_name: var.get()
            for postprocess_name, var in self.postprocess_check_vars.items()
        }
        self._populate_postprocess_checks(self.postprocess_rows, selection_state)
        self._update_postprocess_library_summary()

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

    def _set_all_postprocess_visibility(self, visible: bool) -> None:
        changed = False
        for name in self.postprocess_visibility:
            if self.postprocess_visibility[name] != visible:
                self.postprocess_visibility[name] = visible
                changed = True
        if not changed:
            return
        for var in self.postprocess_visibility_vars.values():
            var.set(visible)
        self._persist_postprocess_visibility()
        selection_state = {
            postprocess_name: var.get()
            for postprocess_name, var in self.postprocess_check_vars.items()
        }
        self._populate_postprocess_checks(self.postprocess_rows, selection_state)
        self._update_postprocess_library_summary()

    def _update_pipeline_library_summary(self) -> None:
        visible_count = sum(self.pipeline_visibility.values())
        total_count = len(self.pipeline_visibility)
        self.pipeline_library_summary_var.set(
            f"Visible in Batch: {visible_count}/{total_count}"
        )

    def _update_postprocess_library_summary(self) -> None:
        visible_count = sum(self.postprocess_visibility.values())
        total_count = len(self.postprocess_visibility)
        self.postprocess_library_summary_var.set(
            f"Visible in Batch: {visible_count}/{total_count}"
        )

    def open_pipeline_library(self) -> None:
        self.notebook.select(self.pipeline_library_tab)

    def open_postprocess_library(self) -> None:
        self.notebook.select(self.postprocess_library_tab)

    def open_batch_tab(self) -> None:
        self.notebook.select(self.batch_tab)

    def open_export_tab(self) -> None:
        self.notebook.select(self.export_tab)

    def show_all_pipelines_in_main_ui(self) -> None:
        self._set_all_pipeline_visibility(True)

    def hide_all_pipelines_from_main_ui(self) -> None:
        self._set_all_pipeline_visibility(False)

    def show_all_postprocesses_in_main_ui(self) -> None:
        self._set_all_postprocess_visibility(True)

    def hide_all_postprocesses_from_main_ui(self) -> None:
        self._set_all_postprocess_visibility(False)

    def refresh_pipeline_catalog(self) -> None:
        self._register_pipelines()
        self._register_postprocesses()

    def refresh_postprocess_catalog(self) -> None:
        self._register_postprocesses()

    def _reset_batch_output(
        self,
        message: str = (
            "Select an input path, choose pipelines and optional postprocess steps, "
            "then run batch."
        ),
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

    def select_all_postprocesses(self) -> None:
        for var in self.postprocess_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(True)

    def clear_all_postprocesses(self) -> None:
        for var in self.postprocess_check_vars.values():
            if getattr(var, "_enabled", True):
                var.set(False)

    def choose_batch_folder(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_input_var.get() or None,
            title="Select folder containing HDF5 files",
        )
        if path:
            self.batch_input_var.set(path)
            self._apply_input_defaults(Path(path))

    def choose_batch_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("HDF5 or zip", "*.h5 *.hdf5 *.zip"), ("All files", "*.*")],
            initialdir=self.batch_input_var.get() or os.path.abspath("h5_example"),
            title="Select HDF5 file or .zip archive",
        )
        if path:
            self.batch_input_var.set(path)
            self._apply_input_defaults(Path(path))

    def choose_batch_output(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.batch_output_var.set(path)

    def run_batch(self) -> None:
        self._reset_progress()
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

        selected_postprocess_names = [
            name for name, var in self.postprocess_check_vars.items() if var.get()
        ]

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

        postprocesses: list[PostprocessDescriptor] = []
        missing_postprocesses: list[str] = []
        for name in selected_postprocess_names:
            postprocess = self.postprocess_registry.get(name)
            if postprocess is None:
                missing_postprocesses.append(name)
            else:
                postprocesses.append(postprocess)
        if missing_postprocesses:
            messagebox.showerror(
                "Postprocess missing",
                f"Postprocess step(s) not registered: {', '.join(missing_postprocesses)}",
            )
            return

        postprocess_requirement_errors = self._validate_postprocess_selection(
            postprocesses,
            selected_pipeline_names=selected_names,
        )
        if postprocess_requirement_errors:
            messagebox.showerror(
                "Postprocess requirements",
                "\n".join(postprocess_requirement_errors),
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
        try:
            data_root, tempdir = self._prepare_data_root(data_path)
            inputs = self._find_h5_inputs(data_root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid input", f"Cannot prepare input: {exc}")
            self._log_batch(f"Error: {exc}")
            if tempdir is not None:
                tempdir.cleanup()
            return

        total_progress_units = (
            (len(inputs) * len(pipelines))
            + len(postprocesses)
            + (1 if self.batch_zip_var.get() else 0)
        )
        self._start_progress(total_progress_units)
        minimal_output_filename = self._minimal_output_filename_for_run(
            data_path,
            inputs,
        )

        work_output_dir: Path | None = None
        clean_work_output = False
        try:
            output_dir = base_output_dir
            if self.batch_zip_var.get():
                work_output_dir = Path(tempfile.mkdtemp(dir=base_output_dir))
                output_dir = work_output_dir

            failures: list[str] = []
            processed_outputs: list[Path] = []
            for h5_path in inputs:
                try:
                    relative_parent = self._relative_input_parent(h5_path, data_root)
                    combined_output = self._run_pipelines_on_file(
                        h5_path,
                        pipelines,
                        output_dir,
                        output_relative_parent=relative_parent,
                        output_filename=minimal_output_filename,
                    )
                    processed_outputs.append(combined_output)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{h5_path}: {exc}")
                    self._log_batch(f"[FAIL] {h5_path.name}: {exc}")

            if postprocesses and processed_outputs:
                self._run_postprocesses(
                    postprocesses=postprocesses,
                    output_dir=output_dir,
                    processed_outputs=processed_outputs,
                    input_path=data_path,
                    selected_pipeline_names=selected_names,
                    failures=failures,
                )
            elif postprocesses:
                self._log_batch(
                    "[POST SKIP] No successful pipeline outputs were generated, "
                    "so postprocess steps were skipped."
                )

            summary_msg: str
            if self.batch_zip_var.get():
                try:
                    zip_name = self.batch_zip_name_var.get().strip() or "outputs.zip"
                    if not zip_name.lower().endswith(".zip"):
                        zip_name += ".zip"
                    self._log_batch("[ZIP] Preparing archive...")
                    last_progress_log = 0.0
                    zip_progress_base = self._progress_completed_units

                    def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
                        nonlocal last_progress_log
                        fraction = 1.0 if total == 0 else done / total
                        self._set_progress_units(zip_progress_base + fraction)
                        now = time.monotonic()
                        if done == total or (now - last_progress_log) >= 0.5:
                            pct = 100 if total == 0 else int((done * 100) / total)
                            self._log_batch(f"[ZIP] {done}/{total} files ({pct}%)")
                            last_progress_log = now
                            try:
                                # Keep the UI responsive while archiving large batches.
                                self.update()
                            except tk.TclError:
                                pass

                    zip_path = self._zip_output_dir(
                        output_dir,
                        target_path=base_output_dir / zip_name,
                        progress_callback=_zip_progress,
                    )
                    self._log_batch(f"[ZIP] Archive created: {zip_path}")
                    summary_msg = f"ZIP archive: {zip_path}"
                    clean_work_output = True
                except Exception as exc:  # noqa: BLE001
                    self._set_progress_units(zip_progress_base + 1.0)
                    self._log_batch(f"[ZIP FAIL] {exc}")
                    messagebox.showerror(
                        "Zip failed", f"Could not create ZIP archive: {exc}"
                    )
                    summary_msg = f"Outputs stored under: {output_dir}"
            else:
                if len(processed_outputs) == 1:
                    summary_msg = f"Output file: {processed_outputs[0]}"
                else:
                    summary_msg = f"Outputs stored under: {output_dir}"

            self._set_progress_units(self._progress_total_units)
            self._log_batch(f"Completed. {summary_msg}")

            if failures:
                self._show_batch_error_dialog(
                    f"{len(failures)} failure(s). See log for details.\n\n{summary_msg}",
                    initial_dir=base_output_dir,
                )
            else:
                messagebox.showinfo("Batch completed", summary_msg)
        finally:
            if tempdir is not None:
                tempdir.cleanup()
            if clean_work_output and work_output_dir is not None:
                shutil.rmtree(work_output_dir, ignore_errors=True)

    def _validate_postprocess_selection(
        self,
        postprocesses: Sequence[PostprocessDescriptor],
        selected_pipeline_names: Sequence[str],
    ) -> list[str]:
        selected_set = set(selected_pipeline_names)
        errors: list[str] = []
        for postprocess in postprocesses:
            missing_required = [
                name
                for name in postprocess.required_pipelines
                if name not in selected_set
            ]
            if missing_required:
                errors.append(
                    f"{postprocess.name} requires pipeline(s): "
                    f"{', '.join(missing_required)}"
                )
        return errors

    def _run_postprocesses(
        self,
        postprocesses: Sequence[PostprocessDescriptor],
        output_dir: Path,
        processed_outputs: Sequence[Path],
        input_path: Path,
        selected_pipeline_names: Sequence[str],
        failures: list[str],
    ) -> None:
        context = PostprocessContext(
            output_dir=output_dir,
            processed_files=tuple(processed_outputs),
            selected_pipelines=tuple(selected_pipeline_names),
            input_path=input_path,
            zip_outputs=self.batch_zip_var.get(),
        )
        for descriptor in postprocesses:
            postprocess = descriptor.instantiate()
            self._log_batch(f"[POST] Running {descriptor.name}...")
            try:
                result = postprocess.run(context)
            except Exception as exc:  # noqa: BLE001
                error_message = (
                    f"Postprocess '{descriptor.name}' failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                failures.append(error_message)
                self._log_batch(f"[POST FAIL] {error_message}")
                self._advance_progress()
                continue

            summary = (result.summary or "").strip()
            if summary:
                self._log_batch(f"[POST OK] {descriptor.name}: {summary}")
            else:
                self._log_batch(f"[POST OK] {descriptor.name}")
            self._advance_progress()

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

    def _relative_input_parent(self, h5_path: Path, input_root: Path) -> Path:
        if input_root.is_dir():
            try:
                return h5_path.resolve().relative_to(input_root.resolve()).parent
            except ValueError:
                pass
        return Path(".")

    def _run_pipelines_on_file(
        self,
        h5_path: Path,
        pipelines: Sequence[PipelineDescriptor],
        output_root: Path,
        output_relative_parent: Path = Path("."),
        output_filename: str | None = None,
    ) -> Path:
        target_dir = output_root / output_relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        if output_filename:
            base_output_path = target_dir / output_filename
            combined_h5_out = base_output_path
        else:
            combined_h5_out = target_dir / f"{h5_path.stem}_pipelines_result.h5"
        suffix = 1
        while combined_h5_out.exists():
            if output_filename:
                combined_h5_out = (
                    target_dir
                    / f"{base_output_path.stem}_{suffix}{base_output_path.suffix}"
                )
            else:
                combined_h5_out = (
                    target_dir / f"{h5_path.stem}_{suffix}_pipelines_result.h5"
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
                self._advance_progress()
        self._log_batch(f"[SAVE] Writing output file -> {combined_h5_out.name}")
        self._write_combined_results_with_ui_pump(
            pipeline_results=pipeline_results,
            combined_h5_out=combined_h5_out,
            source_file=str(h5_path),
        )
        for _, result in pipeline_results:
            result.output_h5_path = str(combined_h5_out)
        self._log_batch(f"[OK] {h5_path.name}: combined results -> {combined_h5_out}")
        return combined_h5_out

    def _write_combined_results_with_ui_pump(
        self,
        pipeline_results: Sequence[tuple[str, ProcessResult]],
        combined_h5_out: Path,
        source_file: str,
    ) -> None:
        errors: list[Exception] = []
        done_event = threading.Event()

        def _worker() -> None:
            try:
                write_combined_results_h5(
                    pipeline_results,
                    combined_h5_out,
                    source_file=source_file,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                done_event.set()

        writer_thread = threading.Thread(target=_worker, daemon=True)
        writer_thread.start()
        while not done_event.wait(timeout=0.05):
            try:
                # Let Tk process paint/events while output file is being written.
                self.update_idletasks()
                self.update()
            except tk.TclError:
                break
        writer_thread.join()
        if errors:
            raise errors[0]

    def _zip_output_dir(
        self,
        folder: Path,
        target_path: Path | None = None,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> Path:
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
        files = sorted(
            (file_path for file_path in folder.rglob("*") if file_path.is_file()),
            key=lambda path: str(path.relative_to(folder)),
        )
        total_files = len(files)
        if progress_callback is not None:
            progress_callback(0, total_files, Path("."))
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as zf:
            for idx, file_path in enumerate(files, start=1):
                rel_path = file_path.relative_to(folder)
                zf.write(file_path, rel_path)
                if progress_callback is not None:
                    progress_callback(idx, total_files, rel_path)
        return zip_path


def main():
    app = ProcessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
