import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess.composite_scoring import optimal_split  # noqa: E402
from postprocess.composite_scoring.dataclasses import Metric  # noqa: E402
from postprocess.utils import groups_comparison_dashboard as dashboard  # noqa: E402


def _write_dataset(group: h5py.Group, path: str, value: float) -> None:
    parent = group
    parts = path.split("/")
    for part in parts[:-1]:
        parent = parent.require_group(part)
    parent.create_dataset(parts[-1], data=value)


def _create_metric_file(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        pipeline = h5.require_group("/AngioEye/Processing/waveform_shape_metrics")
        pipeline.attrs["pipeline"] = "waveform_shape_metrics"
        for vessel_type, offset in (("artery", 0.0), ("vein", 0.25)):
            base = f"{vessel_type}/global/bandlimited"
            _write_dataset(pipeline, f"{base}/direct", value + offset)
            _write_dataset(pipeline, f"{base}/numerator", 2.0 * (value + offset))
            _write_dataset(pipeline, f"{base}/denominator", 2.0)


class PostprocessH5ReadEfficiencyTests(unittest.TestCase):
    def test_optimal_split_opens_each_file_once_for_the_whole_metric_panel(self):
        panel = {
            "direct": Metric(name="direct"),
            "ratio": Metric(
                name="ratio",
                numerator_name="numerator",
                denominator_name="denominator",
            ),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            paths = [
                output_dir / "control" / "control_1.h5",
                output_dir / "control" / "control_2.h5",
                output_dir / "pathology" / "pathology_1.h5",
                output_dir / "pathology" / "pathology_2.h5",
            ]
            for path, value in zip(paths, (1.0, 1.2, 2.0, 2.2), strict=True):
                _create_metric_file(path, value)

            original_file = h5py.File
            open_count = 0

            def counting_file(*args, **kwargs):
                nonlocal open_count
                open_count += 1
                return original_file(*args, **kwargs)

            with mock.patch.object(optimal_split.h5py, "File", counting_file):
                calibrated, stats = (
                    optimal_split.calibrate_metrics_from_processed_files(
                        paths,
                        output_dir,
                        metric_panel=panel,
                        control_group="control",
                        pathology_group="pathology",
                    )
                )

            self.assertEqual(len(paths), open_count)
            self.assertEqual(set(panel), set(calibrated))
            self.assertEqual(len(panel), len(stats))

    def test_dashboard_support_cache_reuses_the_loaded_h5_block(self):
        cache = {}
        expected = {"bandlimited": {"support": [1.0]}}

        with mock.patch.object(
            dashboard,
            "extract_graphics_support",
            return_value=expected,
        ) as extract:
            first = dashboard._extract_graphics_support_cached(
                cache,
                Path("patient.h5"),
                vessel="artery",
                mode="bandlimited",
            )
            second = dashboard._extract_graphics_support_cached(
                cache,
                Path("patient.h5"),
                vessel="artery",
                mode="bandlimited",
            )

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
