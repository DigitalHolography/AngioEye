import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

from app_settings import AppSettingsStore
from pipelines import PipelineDescriptor
from postprocess import PostprocessDescriptor

from .compat import BaseAppTk, sv_ttk
from .controllers import (
    PipelineLibraryController,
    PostprocessLibraryController,
    RunTabController,
    WorkflowSelectionController,
)
from .drag_drop import DragDropMixin
from .progress import ProgressMixin
from .resources import ResourceMixin
from .services import UiServices
from .settings import SettingsMixin
from .views import ViewBuilderMixin

class ProcessApp(
    ViewBuilderMixin,
    DragDropMixin,
    ResourceMixin,
    SettingsMixin,
    ProgressMixin,
    BaseAppTk,
):
    _ADVANCED_FORM_LABEL_WIDTH = 74

    def __init__(self) -> None:
        super().__init__()
        self.title("AngioEye")
        self.ui_services = UiServices()
        self.settings_store = AppSettingsStore()
        self._settings_warning_shown = False
        self._ensure_default_settings()
        self.ui_mode = self.settings_store.load_ui_mode()
        self.pipeline_registry: dict[str, PipelineDescriptor] = {}
        self.pipeline_catalog: dict[str, PipelineDescriptor] = {}
        self.pipeline_rows: list[PipelineDescriptor] = []
        self.pipeline_visibility: dict[str, bool] = {}
        self.pipeline_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.postprocess_registry: dict[str, PostprocessDescriptor] = {}
        self.postprocess_catalog: dict[str, PostprocessDescriptor] = {}
        self.postprocess_rows: list[PostprocessDescriptor] = []
        self.postprocess_visibility: dict[str, bool] = {}
        self.postprocess_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.input_convention_var = tk.StringVar(value="legacy")
        self.batch_input_paths: list[Path] = []
        self.holo_input_paths: list[Path] = []
        self.holo_input_var = tk.StringVar()
        self.holo_status_var = tk.StringVar()
        self.holo_output_path_var = tk.StringVar(value="Output path: -")
        self.minimal_status_var = tk.StringVar(value="Ready.")
        self.pipeline_library_summary_var = tk.StringVar(value="")
        self.postprocess_library_summary_var = tk.StringVar(value="")
        self.minimal_input_path_var = tk.StringVar(value="No input")
        self.minimal_output_path_var = tk.StringVar(value="")
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self._last_saved_batch_log_path: Path | None = None
        self._progress_primary_style = "MinimalPrimary.Horizontal.TProgressbar"
        self._progress_final_style = "MinimalFinal.Horizontal.TProgressbar"
        self._window_icon_image: tk.PhotoImage | None = None
        self._minimal_logo_image: tk.PhotoImage | None = None
        self._minimal_title_font: tkfont.Font | None = None
        self._persist_eyeflow_data = tk.BooleanVar(
            value=not self.settings_store.load_trim_h5source()
        )
        self.run_controller = RunTabController(self)
        self.workflow_selection_controller = WorkflowSelectionController(self)
        self.pipeline_library_controller = PipelineLibraryController(self)
        self.postprocess_library_controller = PostprocessLibraryController(self)

        self._set_initial_window_size()
        self._apply_theme()
        self._set_window_icon()
        self._build_ui()
        self._install_drop_targets()
        self.batch_input_var.trace_add("write", self.run_controller.on_batch_paths_changed)
        self.batch_output_var.trace_add("write", self.run_controller.on_batch_paths_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.pipeline_library_controller.register()
        self.postprocess_library_controller.register()
        self._reset_batch_output()
        self.run_controller.update_minimal_path_labels()
        self._apply_ui_mode(self.ui_mode, persist=False)

    def _set_initial_window_size(self) -> None:
        width, height, min_width, min_height = self._window_size_for_mode(self.ui_mode)
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
        self._style = style
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
        self._configure_progress_styles()

    def _configure_progress_styles(self) -> None:
        progress_colors = {
            self._progress_primary_style: self._accent_color,
            self._progress_final_style: "#3fb37f",
        }
        for style_name, color in progress_colors.items():
            try:
                self._style.configure(
                    style_name,
                    troughcolor=self._surface_color,
                    background=color,
                    bordercolor=color,
                    lightcolor=color,
                    darkcolor=color,
                )
            except tk.TclError:
                self._style.configure(style_name, background=color)

        try:
            self._style.configure("Advanced.TNotebook", tabmargins=(0, 0, 0, 0))
        except tk.TclError:
            pass


def main():
    app = ProcessApp()
    app.mainloop()
