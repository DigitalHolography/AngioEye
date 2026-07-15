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

import input_output.hdf5_io as hdf5_io  # noqa: E402
from math_utils import (  # noqa: E402
    cv_1d,
    finite_1d,
    iqr_1d,
    mad_1d,
    median_1d,
    nanmedian_or_nan,
    std_1d,
)
from postprocess.utils import (  # noqa: E402
    variability_heterogeneity_dashboard as dashboard,
)


def _reference_axis_statistics(slices, eps):
    stats = {name: [] for name in ("median", "std", "iqr", "mad", "cv")}
    for values in slices:
        finite = finite_1d(values)
        stats["median"].append(median_1d(finite))
        stats["std"].append(std_1d(finite))
        stats["iqr"].append(iqr_1d(finite))
        stats["mad"].append(mad_1d(finite))
        stats["cv"].append(cv_1d(finite, eps=eps))
    return {name: np.asarray(values, dtype=float) for name, values in stats.items()}


def _reference_higher_metrics(arr, eps=dashboard.EPS):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        return None

    spatial = _reference_axis_statistics(
        (arr[beat_idx, :, :] for beat_idx in range(arr.shape[0])),
        eps,
    )
    temporal = _reference_axis_statistics(
        (
            arr[:, branch_idx, radius_idx]
            for branch_idx in range(arr.shape[1])
            for radius_idx in range(arr.shape[2])
        ),
        eps,
    )
    return {
        "MED_seg_medbeat": nanmedian_or_nan(spatial["median"]),
        "STD_seg_medbeat": nanmedian_or_nan(spatial["std"]),
        "IQR_seg_medbeat": nanmedian_or_nan(spatial["iqr"]),
        "MAD_seg_medbeat": nanmedian_or_nan(spatial["mad"]),
        "CV_seg_medbeat": nanmedian_or_nan(spatial["cv"]),
        "STD_beat_medseg": nanmedian_or_nan(temporal["std"]),
        "IQR_beat_medseg": nanmedian_or_nan(temporal["iqr"]),
        "MAD_beat_medseg": nanmedian_or_nan(temporal["mad"]),
        "CV_beat_medseg": nanmedian_or_nan(temporal["cv"]),
    }


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
                    hdf5_io.h5py,
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


class HigherMetricComputationTests(unittest.TestCase):
    def test_vectorized_computation_matches_reference_loop(self):
        rng = np.random.default_rng(12345)
        random_values = rng.normal(size=(5, 3, 4))
        random_values[0, :, :] = np.nan
        random_values[:, 0, 0] = np.inf
        random_values[2, 1, 1] = -np.inf

        cases = {
            "random nonfinite": random_values,
            "singleton": np.asarray([[[3.0]]]),
            "all invalid": np.full((3, 2, 2), np.nan),
            "no beats": np.empty((0, 2, 2)),
            "no branches": np.empty((2, 0, 2)),
            "no radii": np.empty((2, 2, 0)),
        }

        for name, values in cases.items():
            with self.subTest(name=name):
                expected = _reference_higher_metrics(values)
                actual = dashboard.compute_file_higher_metrics_from_segment_array(
                    values
                )
                self.assertEqual(actual.keys(), expected.keys())
                np.testing.assert_allclose(
                    list(actual.values()),
                    list(expected.values()),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )

    def test_non_3d_array_is_rejected(self):
        self.assertIsNone(
            dashboard.compute_file_higher_metrics_from_segment_array(
                np.ones((2, 2))
            )
        )


class ThresholdSweepTests(unittest.TestCase):
    def test_cumulative_sweep_matches_legacy_median_direction_search(self):
        cases = {
            "group higher": (
                np.asarray([0.0, 1.0, 2.0, 2.0, np.nan]),
                np.asarray([1.0, 3.0, 4.0, 4.0, np.inf]),
            ),
            "group lower": (
                np.asarray([3.0, 4.0, 5.0, 5.0]),
                np.asarray([0.0, 1.0, 2.0, 4.0]),
            ),
            "crossing with ties": (
                np.asarray([0.0, 0.0, 2.0, 4.0, 4.0]),
                np.asarray([0.0, 1.0, 1.0, 4.0, 5.0]),
            ),
        }

        for name, (control, group) in cases.items():
            with self.subTest(name=name):
                expected = dashboard.best_threshold_sensitivity_specificity(
                    control,
                    group,
                )
                actual = (
                    dashboard.best_threshold_sensitivity_specificity_cumulative_sweep(
                        control,
                        group,
                    )
                )
                np.testing.assert_allclose(actual[:3], expected[:3])
                self.assertEqual(actual[3], expected[3])

    def test_cumulative_sweep_matches_legacy_across_random_tied_samples(self):
        rng = np.random.default_rng(9876)
        for sample_index in range(100):
            control = rng.integers(-4, 5, size=rng.integers(2, 30)).astype(float)
            group = rng.integers(-4, 5, size=rng.integers(2, 30)).astype(float)
            expected = dashboard.best_threshold_sensitivity_specificity(
                control,
                group,
            )
            actual = (
                dashboard.best_threshold_sensitivity_specificity_cumulative_sweep(
                    control,
                    group,
                )
            )

            with self.subTest(sample_index=sample_index):
                np.testing.assert_allclose(actual[:3], expected[:3])
                self.assertEqual(actual[3], expected[3])

    def test_cumulative_sweep_can_evaluate_both_directions(self):
        control = np.asarray([-100.0] * 5 + [5.0] + [300.0] * 5)
        group = np.asarray([-200.0] * 4 + [6.0] * 2 + [200.0] * 5)

        preferred = (
            dashboard.best_threshold_sensitivity_specificity_cumulative_sweep(
                control,
                group,
            )
        )
        both = dashboard.best_threshold_sensitivity_specificity_cumulative_sweep(
            control,
            group,
            evaluate_both_directions=True,
        )

        self.assertEqual(preferred[3], ">=")
        self.assertEqual(both[3], "<=")
        preferred_youden = preferred[1] + preferred[2] - 1.0
        both_youden = both[1] + both[2] - 1.0
        self.assertGreater(both_youden, preferred_youden)

    def test_cumulative_sweep_preserves_degenerate_results(self):
        for control, group in (
            ([], [1.0]),
            ([1.0], []),
            ([1.0, 1.0], [1.0, 1.0]),
        ):
            with self.subTest(control=control, group=group):
                expected = dashboard.best_threshold_sensitivity_specificity(
                    control,
                    group,
                )
                actual = (
                    dashboard.best_threshold_sensitivity_specificity_cumulative_sweep(
                        control,
                        group,
                    )
                )
                np.testing.assert_allclose(
                    actual[:3],
                    expected[:3],
                    equal_nan=True,
                )
                self.assertEqual(actual[3], expected[3])


if __name__ == "__main__":
    unittest.main()
