# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dependency_utils


class DependencyUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        dependency_utils.is_module_available.cache_clear()

    def tearDown(self) -> None:
        dependency_utils.is_module_available.cache_clear()

    def test_is_module_available_falls_back_to_import(self) -> None:
        with (
            mock.patch("dependency_utils.importlib.util.find_spec", return_value=None),
            mock.patch(
                "dependency_utils.importlib.import_module", return_value=object()
            ) as import_module,
        ):
            self.assertTrue(dependency_utils.is_module_available("matplotlib"))
            import_module.assert_called_once_with("matplotlib")

    def test_find_missing_dependencies_uses_import_fallback(self) -> None:
        with (
            mock.patch("dependency_utils.importlib.util.find_spec", return_value=None),
            mock.patch(
                "dependency_utils.importlib.import_module", return_value=object()
            ),
        ):
            missing = dependency_utils.find_missing_dependencies(["matplotlib>=3.8"])

        self.assertEqual([], missing)

    def test_find_missing_dependencies_reports_missing_module(self) -> None:
        with (
            mock.patch("dependency_utils.importlib.util.find_spec", return_value=None),
            mock.patch(
                "dependency_utils.importlib.import_module",
                side_effect=ModuleNotFoundError("missing"),
            ),
        ):
            missing = dependency_utils.find_missing_dependencies(["matplotlib>=3.8"])

        self.assertEqual(["matplotlib"], missing)


if __name__ == "__main__":
    unittest.main()
