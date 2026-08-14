"""EPS companion path mirroring and figure export."""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output.figure_export import save_figure  # noqa: E402
from input_output.output_paths import eps_path_for_png  # noqa: E402


class EpsPathForPngTests(unittest.TestCase):
    def test_mirrors_png_and_pngs_folders(self) -> None:
        self.assertEqual(
            eps_path_for_png("/data/sample_AE/png/artery/raw/fig2.png"),
            Path("/data/sample_AE/eps/artery/raw/fig2.eps"),
        )
        self.assertEqual(
            eps_path_for_png("/data/1_BL1/pngs/artery/raw/fig3.png"),
            Path("/data/1_BL1/eps/artery/raw/fig3.eps"),
        )

    def test_mirrors_export_and_suffix_folders(self) -> None:
        self.assertEqual(
            eps_path_for_png("/tmp/export_png/metric.png"),
            Path("/tmp/export_eps/metric.eps"),
        )
        self.assertEqual(
            eps_path_for_png("/tmp/export_png_html/metric.png"),
            Path("/tmp/export_eps_html/metric.eps"),
        )
        self.assertEqual(
            eps_path_for_png("/tmp/sample_png/plot.png"),
            Path("/tmp/sample_eps/plot.eps"),
        )

    def test_fallback_beside_png(self) -> None:
        self.assertEqual(
            eps_path_for_png("/tmp/plots/figure.png"),
            Path("/tmp/plots/figure.eps"),
        )


class SaveFigureTests(unittest.TestCase):
    def test_writes_png_and_mirrored_eps(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp_dir:
            png_path = Path(tmp_dir) / "png" / "demo.png"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            save_figure(fig, png_path, dpi=80, bbox_inches="tight")
            plt.close(fig)

            self.assertTrue(png_path.is_file())
            eps_path = Path(tmp_dir) / "eps" / "demo.eps"
            self.assertTrue(eps_path.is_file())

    def test_writes_eps_with_translucent_fill(self) -> None:
        """Gray bands bake to opaque light gray for EPS (PS has no alpha)."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from input_output.figure_export import _eps_prepare_vector

        with tempfile.TemporaryDirectory() as tmp_dir:
            png_path = Path(tmp_dir) / "png" / "band.png"
            fig, ax = plt.subplots()
            x = [0.0, 0.5, 1.0]
            collection = ax.fill_between(
                x, [0.0, 0.1, 0.0], [1.0, 0.9, 1.0], color="black", alpha=0.12
            )
            ax.plot(x, [0.5, 0.5, 0.5], color="black")
            ax.set_title(r"Mode $m=1$")
            alpha_before = collection.get_alpha()
            with _eps_prepare_vector(fig):
                face = collection.get_facecolor()[0]
                self.assertAlmostEqual(float(face[0]), 0.88, places=5)
                self.assertAlmostEqual(float(face[3]), 1.0, places=5)
                self.assertEqual(collection.get_alpha(), 1.0)
            save_figure(fig, png_path, dpi=80, bbox_inches="tight", pad_inches=0.02)
            # Temporary EPS flattening must not stick on the live figure.
            self.assertEqual(collection.get_alpha(), alpha_before)
            plt.close(fig)

            self.assertTrue(png_path.is_file())
            self.assertTrue((Path(tmp_dir) / "eps" / "band.eps").is_file())

    def test_skips_eps_when_all_backends_missing(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import input_output.figure_export as figure_export

        with tempfile.TemporaryDirectory() as tmp_dir:
            png_path = Path(tmp_dir) / "png" / "demo.png"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            previous = figure_export._EPS_BACKEND_AVAILABLE
            warned = figure_export._EPS_BACKEND_WARNED
            figure_export._EPS_BACKEND_AVAILABLE = False
            figure_export._EPS_BACKEND_WARNED = False
            original_cairo = figure_export._cairo_backend_available
            figure_export._cairo_backend_available = lambda: False
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    save_figure(fig, png_path, bbox_inches="tight")
            finally:
                figure_export._EPS_BACKEND_AVAILABLE = previous
                figure_export._EPS_BACKEND_WARNED = warned
                figure_export._cairo_backend_available = original_cairo
                plt.close(fig)

            self.assertTrue(png_path.is_file())
            self.assertFalse((Path(tmp_dir) / "eps" / "demo.eps").exists())
            self.assertTrue(
                any("EPS export skipped" in str(item.message) for item in caught)
            )


if __name__ == "__main__":
    unittest.main()
