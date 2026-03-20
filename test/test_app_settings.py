import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_settings import (
    AppSettingsStore,
    default_settings_path,
    normalize_pipeline_visibility,
    normalize_postprocess_visibility,
)


class AppSettingsTests(unittest.TestCase):
    def test_default_settings_path_prefers_appdata(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"APPDATA": r"C:\Users\Test\AppData\Roaming"},
            clear=True,
        ):
            self.assertEqual(
                default_settings_path(),
                Path(r"C:\Users\Test\AppData\Roaming\AngioEye\settings.json"),
            )

    def test_normalize_pipeline_visibility_defaults_first_run_to_visible(self) -> None:
        visibility, changed = normalize_pipeline_visibility(["a", "b"], {})

        self.assertEqual(visibility, {"a": True, "b": True})
        self.assertTrue(changed)

    def test_normalize_pipeline_visibility_hides_new_pipelines_after_first_run(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
