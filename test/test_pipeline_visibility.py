import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ui.controllers.pipeline_library import PipelineLibraryController  # noqa: E402


class PipelineVisibilityTests(unittest.TestCase):
    def test_unavailable_pipeline_is_deselected_and_preferences_are_saved(self):
        settings = mock.Mock()
        settings.load_pipeline_visibility.return_value = {
            "missing": True,
            "HTML summary": True,
        }
        app = SimpleNamespace(settings_store=settings, pipeline_visibility={})
        controller = PipelineLibraryController(app)
        rows = [
            SimpleNamespace(name="missing", available=False),
            SimpleNamespace(name="HTML summary", available=True),
        ]
        controller.sync_visibility(rows)
        self.assertEqual(
            app.pipeline_visibility, {"missing": False, "HTML summary": True}
        )
        settings.save_pipeline_visibility.assert_called_once_with(
            app.pipeline_visibility
        )

    def test_status_describes_required_eyeflow_metrics(self):
        controller = PipelineLibraryController(SimpleNamespace())
        descriptor = SimpleNamespace(
            available=True,
            required_metric_options=[["Windkessel_RC", "waveform_shape_metrics"]],
            required_metrics=[],
            missing_deps=[],
        )
        text = controller.status_text(descriptor)
        self.assertIn("EyeFlow metrics", text)
        self.assertIn("Windkessel_RC", text)
        self.assertIn("waveform_shape_metrics", text)

    def test_dependency_failure_is_visible(self):
        controller = PipelineLibraryController(SimpleNamespace())
        descriptor = SimpleNamespace(
            available=False,
            required_metric_options=[],
            required_metrics=[],
            missing_deps=["scipy"],
        )
        self.assertEqual(
            controller.status_text(descriptor), "Missing dependencies: scipy"
        )
