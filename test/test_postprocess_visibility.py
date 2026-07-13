import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_settings import AppSettingsStore  # noqa: E402
from postprocess import PostprocessDescriptor  # noqa: E402
from postprocess.core import base as postprocess_base  # noqa: E402
from postprocess.core.base import (  # noqa: E402
    DEFAULT_INPUT_METHODS,
    PostprocessResult,
    format_required_pipeline_options,
    registerPostprocess,
)
from postprocess.pipeline_metrics_manifest import (  # noqa: E402
    PipelineMetricsManifestPostprocess,
)
from postprocess.stats_groups_comparison_pipeline import (  # noqa: E402
    GraphicsDashboardPostprocess,
)
from postprocess.tutorial_postprocess import PostprocessTutorial  # noqa: E402
from ui.controllers.postprocess_library import (  # noqa: E402
    PostprocessLibraryController,
)


class PostprocessVisibilityTests(unittest.TestCase):
    def test_builtin_hidden_postprocesses_declare_hidden_visibility(self) -> None:
        self.assertEqual(GraphicsDashboardPostprocess.visibility, "hidden")
        self.assertEqual(PipelineMetricsManifestPostprocess.visibility, "hidden")
        self.assertEqual(PostprocessTutorial.visibility, "hidden")

    def test_register_postprocess_applies_visibility_to_function_targets(self) -> None:
        original_registry = postprocess_base.POSTPROCESS_REGISTRY.copy()
        postprocess_base.POSTPROCESS_REGISTRY.clear()
        try:

            @registerPostprocess(name="Hidden Function", visibility="hidden")
            def hidden_function(_context):
                return PostprocessResult()

            registered = postprocess_base.POSTPROCESS_REGISTRY["Hidden Function"]

            self.assertIs(registered.func, hidden_function)
            self.assertEqual(registered.visibility, "hidden")
        finally:
            postprocess_base.POSTPROCESS_REGISTRY.clear()
            postprocess_base.POSTPROCESS_REGISTRY.update(original_registry)

    def test_register_postprocess_applies_input_methods_to_function_targets(
        self,
    ) -> None:
        original_registry = postprocess_base.POSTPROCESS_REGISTRY.copy()
        postprocess_base.POSTPROCESS_REGISTRY.clear()
        try:

            @registerPostprocess(
                name="Batch Function",
                input_methods=["zip_batch", "file_batch", "zip_batch"],
            )
            def batch_function(_context):
                return PostprocessResult()

            registered = postprocess_base.POSTPROCESS_REGISTRY["Batch Function"]

            self.assertIs(registered.func, batch_function)
            self.assertEqual(
                registered.input_methods,
                ["zip_batch", "file_batch"],
            )
        finally:
            postprocess_base.POSTPROCESS_REGISTRY.clear()
            postprocess_base.POSTPROCESS_REGISTRY.update(original_registry)

    def test_register_postprocess_defaults_input_methods(self) -> None:
        original_registry = postprocess_base.POSTPROCESS_REGISTRY.copy()
        postprocess_base.POSTPROCESS_REGISTRY.clear()
        try:

            @registerPostprocess(name="Default Function")
            def default_function(_context):
                return PostprocessResult()

            registered = postprocess_base.POSTPROCESS_REGISTRY["Default Function"]

            self.assertIs(registered.func, default_function)
            self.assertEqual(
                registered.input_methods,
                list(DEFAULT_INPUT_METHODS),
            )
        finally:
            postprocess_base.POSTPROCESS_REGISTRY.clear()
            postprocess_base.POSTPROCESS_REGISTRY.update(original_registry)

    def test_register_postprocess_groups_pipeline_options_as_or_with_and_groups(
        self,
    ) -> None:
        original_registry = postprocess_base.POSTPROCESS_REGISTRY.copy()
        postprocess_base.POSTPROCESS_REGISTRY.clear()
        try:

            @registerPostprocess(
                name="Grouped Pipelines",
                required_pipeline_options=[
                    ["one", "one_alternative"],
                    ["two", "two_alternative"],
                ],
            )
            def grouped_pipelines(_context):
                return PostprocessResult()

            registered = postprocess_base.POSTPROCESS_REGISTRY["Grouped Pipelines"]

            self.assertEqual(
                registered.required_pipeline_options,
                [["one", "one_alternative"], ["two", "two_alternative"]],
            )
            self.assertEqual(
                format_required_pipeline_options(registered),
                "(one or one_alternative) and (two or two_alternative)",
            )
        finally:
            postprocess_base.POSTPROCESS_REGISTRY.clear()
            postprocess_base.POSTPROCESS_REGISTRY.update(original_registry)

    def test_postprocess_library_register_filters_hidden_rows(self) -> None:
        available = [
            PostprocessDescriptor(
                name="Visible Available",
                description="",
                available=True,
            ),
            PostprocessDescriptor(
                name="Hidden Available",
                description="",
                available=True,
                visibility="hidden",
            ),
        ]
        missing = [
            PostprocessDescriptor(
                name="Visible Missing",
                description="",
                available=False,
            ),
            PostprocessDescriptor(
                name="Hidden Missing",
                description="",
                available=False,
                visibility="hidden",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            app = SimpleNamespace(
                settings_store=AppSettingsStore(Path(tmp_dir) / "settings.json"),
                _install_drop_targets=lambda: None,
                _show_settings_warning=lambda *_args, **_kwargs: None,
            )
            app.settings_store.save_postprocess_visibility(
                {
                    "Hidden Available": True,
                    "Visible Available": True,
                    "Visible Missing": True,
                }
            )

            controller = PostprocessLibraryController(app)

            def capture_populate(_controller, rows):
                app.populated_rows = rows

            with (
                mock.patch(
                    "ui.controllers.postprocess_library.load_postprocess_catalog",
                    return_value=(available, missing),
                ),
                mock.patch.object(
                    PostprocessLibraryController,
                    "populate",
                    capture_populate,
                ),
            ):
                controller.register()

            self.assertEqual(
                [row.name for row in app.postprocess_rows],
                ["Visible Available", "Visible Missing"],
            )
            self.assertEqual(
                [row.name for row in app.populated_rows],
                ["Visible Available", "Visible Missing"],
            )
            self.assertEqual(list(app.postprocess_registry), ["Visible Available"])
            self.assertEqual(
                list(app.postprocess_catalog),
                ["Visible Available", "Visible Missing"],
            )
            self.assertEqual(
                app.settings_store.load_postprocess_visibility(),
                {"Visible Available": True, "Visible Missing": False},
            )


if __name__ == "__main__":
    unittest.main()
