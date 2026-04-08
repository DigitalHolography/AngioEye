# ruff: noqa: E402

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess.core import groups_comparison_dashboard as dashboard


class GroupsComparisonDashboardTests(unittest.TestCase):
    def test_optional_eps_export_skips_missing_backend_ps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export_eps"
            output_dir.mkdir()
            buffer = io.StringIO()
            exc = ModuleNotFoundError(
                "No module named 'matplotlib.backends.backend_ps'"
            )
            exc.name = dashboard.POSTSCRIPT_BACKEND_MODULE

            with redirect_stdout(buffer):
                result = dashboard._run_optional_eps_export(
                    lambda: (_ for _ in ()).throw(exc),
                    str(output_dir),
                )

            self.assertFalse(result)
            self.assertFalse(output_dir.exists())
            self.assertIn("EPS export skipped", buffer.getvalue())

    def test_optional_eps_export_reraises_other_missing_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export_eps"
            output_dir.mkdir()
            exc = ModuleNotFoundError("No module named 'plotly'")
            exc.name = "plotly"

            with self.assertRaises(ModuleNotFoundError):
                dashboard._run_optional_eps_export(
                    lambda: (_ for _ in ()).throw(exc),
                    str(output_dir),
                )


if __name__ == "__main__":
    unittest.main()
