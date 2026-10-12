import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_settings import (  # noqa: E402
    AppSettingsStore,
    app_display_name,
    default_settings_path,
    normalize_pipeline_visibility,
)


class AppSettingsTests(unittest.TestCase):
    def test_default_settings_path_prefers_appdata_and_version(self):
        with mock.patch.dict(
            "os.environ",
            {"APPDATA": r"C:\Users\Test\AppData\Roaming", "ANGIOEYE_VERSION": "9.9.9"},
            clear=True,
        ):
            self.assertEqual(
                default_settings_path(),
                Path(r"C:\Users\Test\AppData\Roaming\AngioEye\9.9.9\settings.json"),
            )

    def test_display_name_includes_version(self):
        with mock.patch.dict("os.environ", {"ANGIOEYE_VERSION": "9.9.9"}):
            self.assertEqual(app_display_name(), "AngioEye v9.9.9")

    def test_first_launch_selects_summary_and_manifest(self):
        visibility, changed = normalize_pipeline_visibility(
            ["HTML summary", "Pipeline Metrics Manifest", "Composite Scoring"], {}
        )
        self.assertEqual(
            visibility,
            {
                "HTML summary": True,
                "Pipeline Metrics Manifest": True,
                "Composite Scoring": False,
            },
        )
        self.assertTrue(changed)

    def test_new_pipelines_remain_unselected_and_removed_names_are_pruned(self):
        visibility, changed = normalize_pipeline_visibility(
            ["a", "b"], {"a": True, "removed": False}
        )
        self.assertEqual(visibility, {"a": True, "b": False})
        self.assertTrue(changed)

    def test_legacy_settings_preserve_batch_selection_and_clean_obsolete_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "pipeline_visibility": {"waveform_shape_metrics": True},
                        "postprocess_visibility": {
                            "Composite Scoring": True,
                            "HTML summary": False,
                        },
                        "postprocess_batching": {"variability": {"batch_size": 4}},
                        "persist_source": True,
                        "trim_h5source": False,
                        "ui_mode": "advanced",
                    }
                ),
                encoding="utf-8",
            )
            store = AppSettingsStore(settings_path)
            self.assertEqual(
                store.load_pipeline_visibility(),
                {"Composite Scoring": True, "HTML summary": False},
            )
            self.assertEqual(
                store.load_pipeline_batching(), {"variability": {"batch_size": 4}}
            )
            store.save_ui_mode("minimal")
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            for key in (
                "postprocess_visibility",
                "postprocess_batching",
                "persist_source",
                "trim_h5source",
            ):
                self.assertNotIn(key, saved)
            self.assertEqual(
                saved["pipeline_visibility"],
                {"Composite Scoring": True, "HTML summary": False},
            )

    def test_branch_round_trip_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / "settings.json")
            self.assertEqual(store.load_processing_branch(), "Processing")
            store.save_processing_branch("ProcessingAlt")
            self.assertEqual(store.load_processing_branch(), "ProcessingAlt")
            with self.assertRaises(ValueError):
                store.save_processing_branch("missing")
            for value in ("unknown", {}, []):
                store.save({"processing_branch": value})
                self.assertEqual(store.load_processing_branch(), "Processing")

    def test_renamed_batch_tools_preserve_legacy_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / "settings.json")
            store.save(
                {
                    "postprocess_visibility": {
                        "Variability and heterogeneity tables": True,
                        "groups comparison dashboard": True,
                    }
                }
            )
            self.assertEqual(
                store.load_pipeline_visibility(),
                {
                    "Variability and heterogeneity": True,
                    "Group comparison (Dashboard) - Waveform Shape Metrics": True,
                },
            )

    def test_defaults_initialization_preserves_existing_user_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "defaults.json"
            template.write_text(
                json.dumps(
                    {
                        "pipeline_visibility": {"HTML summary": True},
                        "ui_mode": "advanced",
                    }
                ),
                encoding="utf-8",
            )
            store = AppSettingsStore(root / "settings.json", template)
            self.assertTrue(store.initialize_from_defaults())
            store.save_pipeline_visibility({"HTML summary": False})
            self.assertFalse(store.initialize_from_defaults())
            self.assertEqual(store.load_pipeline_visibility(), {"HTML summary": False})
            self.assertEqual(store.load_ui_mode(), "advanced")

    def test_malformed_json_returns_default_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("not json", encoding="utf-8")
            store = AppSettingsStore(path)
            self.assertEqual(store.load(), {})
            self.assertEqual(store.load_ui_mode(), "minimal")
