from __future__ import annotations

from .base import ViewController
from .pipeline_library import PipelineLibraryController
from .run import RunTabController


class MinimalViewController(ViewController):
    @property
    def input_path_var(self):
        return self.app.minimal_input_path_var

    @property
    def output_path_var(self):
        return self.app.minimal_output_path_var

    @property
    def status_var(self):
        return self.app.minimal_status_var

    @property
    def progress_var(self):
        return self.app.batch_progress_var

    @property
    def bg_color(self) -> str:
        return self.app._bg_color

    @property
    def muted_fg(self) -> str:
        return self.app._muted_fg

    @property
    def text_fg(self) -> str:
        return self.app._text_fg

    @property
    def progress_style(self) -> str:
        return self.app._progress_primary_style

    def title_font(self):
        return self.app._get_minimal_title_font()

    def load_logo(self):
        return self.app._load_scaled_logo_image(max_width=360, max_height=144)

    def keep_logo(self, image) -> None:
        self.app._minimal_logo_image = image

    def choose_input(self) -> None:
        self.app.run_controller.choose_file()

    def choose_folder(self) -> None:
        self.app.run_controller.choose_folder()

    def run(self) -> None:
        self.app.run_controller.run()

    def choose_pipelines(self) -> None:
        self.app._apply_ui_mode("advanced")
        self.app.notebook.select(self.app.pipeline_library_tab)


class AdvancedViewController(ViewController):
    def create_run_controller(self) -> RunTabController:
        return self.app.run_controller

    def create_pipeline_library_controller(self) -> PipelineLibraryController:
        return self.app.pipeline_library_controller
