import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_settings import (
    AppSettingsStore,
    default_settings_path,
    normalize_pipeline_visibility,
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


if __name__ == "__main__":
    unittest.main()
