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
    default_settings_path,
    normalize_pipeline_visibility,
    normalize_postprocess_visibility,
)


class AppSettingsTests(unittest.TestCase):
    def test_default_settings_path_prefers_appdata_and_version(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "APPDATA": r"C:\Users\Test\AppData\Roaming",
                "ANGIOEYE_VERSION": "9.9.9",
            },
            clear=True,
        ):
            self.assertEqual(
                default_settings_path(),
                Path(r"C:\Users\Test\AppData\Roaming\AngioEye\9.9.9\settings.json"),
            )

    def test_normalize_pipeline_visibility_defaults_first_run_to_visible(self) -> None:
        visibility, changed = normalize_pipeline_visibility(["a", "b"], {})

        self.assertEqual(visibility, {"a": True, "b": True})
        self.assertTrue(changed)

    def test_normalize_pipeline_visibility_hides_new_pipelines_after_first_run(
        self,
    ) -> None:
        visibility, changed = normalize_pipeline_visibility(
            ["a", "b", "c"],
            {"a": True, "b": False},
        )

        self.assertEqual(visibility, {"a": True, "b": False, "c": False})
        self.assertTrue(changed)

    def test_store_round_trips_pipeline_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")
            expected = {"Basic Stats": True, "Dummy Heavy": False}

            store.save_pipeline_visibility(expected)

            self.assertEqual(store.load_pipeline_visibility(), expected)

    def test_normalize_postprocess_visibility_defaults_first_run_to_visible(self) -> None:
        visibility, changed = normalize_postprocess_visibility(
            ["Graphics Dashboard"],
            {},
        )

        self.assertEqual(visibility, {"Graphics Dashboard": True})
        self.assertTrue(changed)

    def test_store_round_trips_postprocess_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")
            expected = {"Graphics Dashboard": True}

            store.save_postprocess_visibility(expected)

            self.assertEqual(store.load_postprocess_visibility(), expected)

    def test_load_ui_mode_defaults_to_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")

            self.assertEqual(store.load_ui_mode(), "minimal")

    def test_load_trim_h5source_defaults_to_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")

            self.assertTrue(store.load_trim_h5source())

    def test_load_uses_default_template_when_user_settings_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            template_path = tmp_path / "default_settings.json"
            settings_path = tmp_path / "settings.json"
            template_path.write_text(
                json.dumps(
                    {
                        "pipeline_visibility": {"Demo": True},
                        "postprocess_visibility": {"Report": False},
                        "ui_mode": "advanced",
                    }
                ),
                encoding="utf-8",
            )
            store = AppSettingsStore(settings_path, template_path)

            self.assertEqual(store.load_ui_mode(), "advanced")
            self.assertEqual(store.load_pipeline_visibility(), {"Demo": True})

    def test_initialize_from_defaults_does_not_overwrite_existing_settings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            template_path = tmp_path / "default_settings.json"
            settings_path = tmp_path / "settings.json"
            template_path.write_text(
                json.dumps({"ui_mode": "advanced"}),
                encoding="utf-8",
            )
            settings_path.write_text(
                json.dumps({"ui_mode": "minimal"}),
                encoding="utf-8",
            )
            store = AppSettingsStore(settings_path, template_path)

            self.assertFalse(store.initialize_from_defaults())
            self.assertEqual(store.load_ui_mode(), "minimal")

    def test_initialize_from_defaults_writes_missing_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            template_path = tmp_path / "default_settings.json"
            settings_path = tmp_path / "settings.json"
            template_path.write_text(
                json.dumps({"ui_mode": "advanced"}),
                encoding="utf-8",
            )
            store = AppSettingsStore(settings_path, template_path)

            self.assertTrue(store.initialize_from_defaults())
            self.assertEqual(
                json.loads(settings_path.read_text(encoding="utf-8")),
                {"ui_mode": "advanced"},
            )

    def test_store_round_trips_ui_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")

            store.save_ui_mode("advanced")

            self.assertEqual(store.load_ui_mode(), "advanced")

    def test_store_round_trips_trim_h5source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AppSettingsStore(Path(tmp_dir) / "settings.json")

            store.save_trim_h5source(False)

            self.assertFalse(store.load_trim_h5source())


if __name__ == "__main__":
    unittest.main()
