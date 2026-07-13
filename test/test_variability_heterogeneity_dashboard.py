import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess.utils import (  # noqa: E402
    variability_heterogeneity_dashboard as dashboard,
)


class SegmentMetricExtractionTests(unittest.TestCase):
    def test_iter_segment_metrics_accepts_custom_metric_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "custom_metrics.h5"
            with h5py.File(h5_path, "w") as h5file:
                group = h5file.require_group("/custom/metrics/custom_mode")
                group.create_dataset("RI", data=np.full((2, 2, 2), 3.0))

            extracted = list(
                dashboard.iter_segment_metrics(
                    h5_path,
                    ("RI",),
                    mode="custom_mode",
                    metric_folders=("/custom/metrics",),
                )
            )

            self.assertEqual(len(extracted), 1)
            metric_name, values = extracted[0]
            self.assertEqual(metric_name, "RI")
            np.testing.assert_array_equal(values, np.full((2, 2, 2), 3.0))

    def test_compute_blocks_opens_h5_once_and_preserves_folder_priority(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "metrics.h5"
            with h5py.File(h5_path, "w") as h5file:
                denoised = h5file.require_group(
                    "/AngioEye/Processing/waveform_shape_metrics_denoised/"
                    "artery/by_segment/bandlimited_segment"
                )
                raw = h5file.require_group(
                    "/AngioEye/Processing/waveform_shape_metrics/"
                    "artery/by_segment/bandlimited_segment"
                )
                denoised.create_dataset("RI", data=np.ones((2, 2, 2)))
                raw.create_dataset("RI", data=np.full((2, 2, 2), 9.0))
                raw.create_dataset("PI", data=np.full((2, 2, 2), 2.0))

            real_h5_file = h5py.File

            def summarize(arr):
                return {"mean": float(np.mean(arr))}

            with (
                mock.patch.object(
                    dashboard.h5py,
                    "File",
                    wraps=real_h5_file,
                ) as open_h5,
                mock.patch.object(
                    dashboard,
                    "compute_file_higher_metrics_from_segment_array",
                    side_effect=summarize,
                ),
            ):
                blocks = dashboard.compute_file_higher_metric_blocks(
                    h5_path,
                    metrics=("RI", "PI", "missing"),
                )

            self.assertEqual(open_h5.call_count, 1)
            self.assertEqual(
                blocks,
                {
                    "RI": {"mean": 1.0},
                    "PI": {"mean": 2.0},
                },
            )


if __name__ == "__main__":
    unittest.main()
