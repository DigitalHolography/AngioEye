# ruff: noqa: E402

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.utils import groups_comparison_dashboard as dashboard


class GroupsComparisonDashboardTests(unittest.TestCase):
    def test_nested_diagnostics_are_skipped_and_one_beat_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample_EF.h5"
            with h5py.File(path, "w") as h5:
                group = h5.require_group(
                    "/Processing/Metrics/waveform_shape_metrics/artery/global/bandlimited"
                )
                group.create_dataset("signal_mean", data=np.arange(12).reshape(4, 3))
                group.create_dataset("tau", data=np.tile(np.arange(4)[:, None], (1, 3)))
                group.create_dataset("RI", data=[0.3, 0.4, 0.5])
                group.create_dataset("diagnostics/spectrum", data=np.ones((3, 5)))
                group.create_dataset("label", data="metadata")
            support = dashboard.extract_graphics_support(path)["bandlimited"]
            self.assertNotIn("diagnostics", support)
            self.assertNotIn("label", support)
            beat = dashboard.select_support_beat(support, 1)
            np.testing.assert_array_equal([1, 4, 7, 10], beat["signal_mean"])
            np.testing.assert_array_equal([0, 1, 2, 3], beat["tau"])
            self.assertEqual(0.4, beat["RI"])

    def test_legacy_sample_curves_keep_their_sample_axis(self) -> None:
        support = {"tau": np.arange(4), "signal_mean": np.arange(4), "RI": [0.7]}
        beat = dashboard.select_support_beat(support, 0)
        np.testing.assert_array_equal(support["tau"], beat["tau"])
        np.testing.assert_array_equal(support["signal_mean"], beat["signal_mean"])

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
