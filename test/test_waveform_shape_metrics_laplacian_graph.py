import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.waveform_shape_metrics_denoised_laplace import (  # noqa: E402
    WaveformShapeMetricsDenoisedLaplace,
)

try:
    from pipelines.waveform_shape_metrics_denoised_joint import (  # noqa: E402
        WaveformShapeMetricsDenoisedJoint,
    )
except ModuleNotFoundError:  # pragma: no cover - optional pipeline in this workspace
    WaveformShapeMetricsDenoisedJoint = None


def _configure_laplace(pipeline):
    pipeline.laplacian_min_corr = 0.99
    pipeline.laplacian_reference_min_corr = -1.0
    pipeline.laplacian_corr_power = 1.0
    pipeline.laplacian_max_lag_fraction = 0.0
    pipeline.laplacian_lag_sigma_fraction = 0.0
    pipeline.laplacian_branch_sigma = 0.0
    pipeline.laplacian_radius_sigma = 0.0
    pipeline.laplacian_preserve_nan_mask = True
    pipeline.laplacian_restore_pulse_scale = True
    pipeline.laplacian_clip_output = False
    return pipeline


def _configure_joint(pipeline):
    pipeline.joint_min_corr = 0.99
    pipeline.joint_corr_power = 1.0
    pipeline.joint_max_lag_fraction = 0.0
    pipeline.joint_lag_sigma_fraction = 0.0
    pipeline.joint_branch_sigma = 0.0
    pipeline.joint_radius_sigma = 0.0
    pipeline.joint_preserve_nan_mask = True
    pipeline.joint_restore_pulse_scale = True
    pipeline.joint_clip_output = False
    pipeline.joint_temporal_gamma = 0.03
    return pipeline


def _component_test_block():
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    base = 2.0 + np.sin(t)
    other = 2.0 + np.cos(t)
    block = np.empty((t.size, 3, 1, 1), dtype=float)
    block[:, 0, 0, 0] = base
    block[:, 1, 0, 0] = base
    block[:, 2, 0, 0] = other
    return block


def _configure_laplace_reference_gate(pipeline):
    pipeline.laplacian_min_corr = 0.99
    pipeline.laplacian_reference_min_corr = 0.80
    pipeline.laplacian_corr_power = 1.0
    pipeline.laplacian_lag_sigma_fraction = 0.0
    pipeline.laplacian_branch_sigma = 0.0
    pipeline.laplacian_radius_sigma = 0.0
    pipeline.laplacian_preserve_nan_mask = True
    pipeline.laplacian_restore_pulse_scale = True
    pipeline.laplacian_clip_output = False
    return pipeline


def _reference_gate_test_block():
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    base = 2.0 + np.sin(t) + 0.20 * np.sin(2.0 * t)
    wrong_shape = 2.0 + np.sin(3.0 * t)
    block = np.empty((t.size, 3, 1, 1), dtype=float)
    block[:, 0, 0, 0] = base
    block[:, 1, 0, 0] = base
    block[:, 2, 0, 0] = wrong_shape
    return block


class LaplacianGraphConstructionTests(unittest.TestCase):
    def test_weight_matrix_is_symmetric_all_pairs_threshold_graph(self):
        pipeline = WaveformShapeMetricsDenoisedLaplace()
        pipeline.laplacian_min_corr = 0.70
        pipeline.laplacian_corr_power = 1.0
        pipeline.laplacian_lag_sigma_fraction = 0.0
        pipeline.laplacian_branch_sigma = 0.0
        pipeline.laplacian_radius_sigma = 0.0

        aligned_norm = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.9, np.sqrt(1.0 - 0.9**2), 0.0],
            ],
            dtype=float,
        )
        valid_flat_indices = np.asarray([0, 1, 2], dtype=np.int32)
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        pulse_shape = (3, 1, 1)
        best_neighbor_corr = np.full(pulse_shape, np.nan, dtype=float)
        neighbor_count = np.zeros(pulse_shape, dtype=np.int32)
        effective_neighbor_count = np.zeros(pulse_shape, dtype=float)
        graph_degree = np.zeros(pulse_shape, dtype=float)

        W = pipeline._build_laplacian_weight_matrix(
            aligned_norm=aligned_norm,
            valid_flat_indices=valid_flat_indices,
            coords=coords,
            alignment_lags_by_row=np.zeros(3, dtype=int),
            n_time=aligned_norm.shape[1],
            best_neighbor_corr=best_neighbor_corr,
            neighbor_count=neighbor_count,
            effective_neighbor_count=effective_neighbor_count,
            graph_degree=graph_degree,
        )

        self.assertTrue(np.allclose(W, W.T))
        self.assertTrue(np.allclose(np.diag(W), 0.0))
        self.assertEqual(3, int(np.sum(W > 0.0) // 2))
        self.assertTrue(np.all(neighbor_count[:, 0, 0] == 2))
        self.assertGreater(W[0, 1], 0.0)
        self.assertAlmostEqual(W[0, 1], (0.8 - 0.7) / 0.3)

    def test_laplace_detects_components_and_leaves_singletons_unchanged(self):
        pipeline = _configure_laplace(WaveformShapeMetricsDenoisedLaplace())
        block = _component_test_block()

        out, diag = pipeline._denoise_segment_block(block)

        np.testing.assert_allclose(out[:, 2, 0, 0], block[:, 2, 0, 0])
        self.assertEqual(2, diag["graph_component_count"])
        self.assertEqual(1, diag["graph_edge_count"])
        self.assertEqual(6, diag["status_code"][2, 0, 0])
        self.assertTrue(np.all(diag["status_code"][:2, 0, 0] == 0))
        self.assertEqual(diag["graph_component_label"][0, 0, 0], 0)
        self.assertEqual(diag["graph_component_label"][1, 0, 0], 0)
        self.assertNotEqual(
            diag["graph_component_label"][0, 0, 0],
            diag["graph_component_label"][2, 0, 0],
        )

    def test_laplace_removed_params_are_not_packed_or_used(self):
        block = _component_test_block()
        baseline = _configure_laplace(WaveformShapeMetricsDenoisedLaplace())
        stale_attrs = _configure_laplace(WaveformShapeMetricsDenoisedLaplace())
        stale_attrs.laplacian_top_k_neighbors = 1
        stale_attrs.laplacian_blend_alpha = 0.0
        stale_attrs.laplacian_self_jitter = 100.0

        out_baseline, diag = baseline._denoise_segment_block(block)
        out_stale, _ = stale_attrs._denoise_segment_block(block)

        np.testing.assert_allclose(out_baseline, out_stale)
        baseline._last_laplacian_denoise_diag = diag
        metrics = {}
        baseline._pack_laplacian_denoising_outputs(metrics)
        packed_names = {key.rsplit("/", 1)[-1] for key in metrics}
        self.assertNotIn("top_k_neighbors", packed_names)
        self.assertNotIn("blend_alpha", packed_names)
        self.assertNotIn("self_jitter", packed_names)
        self.assertIn("graph_component_count", packed_names)

    def test_laplace_reference_gate_rejects_wrong_shape_as_nan(self):
        pipeline = _configure_laplace_reference_gate(
            WaveformShapeMetricsDenoisedLaplace()
        )
        pipeline.laplacian_max_lag_fraction = 0.0
        block = _reference_gate_test_block()

        out, diag = pipeline._denoise_segment_block(block)

        self.assertTrue(np.all(np.isnan(out[:, 2, 0, 0])))
        self.assertEqual(7, diag["status_code"][2, 0, 0])
        self.assertTrue(np.all(diag["status_code"][:2, 0, 0] == 0))
        self.assertEqual(2, diag["valid_pulse_count"])
        self.assertEqual(2, diag["reference_kept_count"])
        self.assertEqual(1, diag["reference_rejected_count"])
        self.assertEqual(1, diag["graph_edge_count"])
        self.assertTrue(np.all(diag["reference_keep_mask"][:2, 0, 0]))
        self.assertFalse(bool(diag["reference_keep_mask"][2, 0, 0]))

        pipeline._last_laplacian_denoise_diag = diag
        metrics = {}
        pipeline._pack_laplacian_denoising_outputs(metrics)
        self.assertIn("artery/by_segment/denoising/reference_corr", metrics)
        self.assertIn(
            "artery/by_segment/denoising/params/reference_min_corr", metrics
        )

    def test_laplace_reference_gate_accepts_phase_shifted_shape(self):
        pipeline = _configure_laplace_reference_gate(
            WaveformShapeMetricsDenoisedLaplace()
        )
        pipeline.laplacian_max_lag_fraction = 0.10
        block = _reference_gate_test_block()
        block[:, 2, 0, 0] = np.roll(block[:, 0, 0, 0], 4)

        out, diag = pipeline._denoise_segment_block(block)

        self.assertFalse(np.any(np.isnan(out[:, 2, 0, 0])))
        self.assertTrue(np.all(diag["reference_keep_mask"][:, 0, 0]))
        self.assertEqual(3, diag["reference_kept_count"])
        self.assertEqual(0, diag["reference_rejected_count"])
        self.assertGreaterEqual(diag["reference_corr"][2, 0, 0], 0.99)
        self.assertEqual(-4, int(diag["reference_lag_samples"][2, 0, 0]))
        self.assertTrue(np.all(diag["status_code"][:, 0, 0] == 0))

    def test_laplace_reference_gate_uses_nan_ignoring_median_reference(self):
        pipeline = _configure_laplace_reference_gate(
            WaveformShapeMetricsDenoisedLaplace()
        )
        pipeline.laplacian_min_corr = 0.90
        pipeline.laplacian_max_lag_fraction = 0.0
        block = _reference_gate_test_block()
        block[10:15, 0, 0, 0] = np.nan

        out, diag = pipeline._denoise_segment_block(block)

        self.assertTrue(bool(diag["reference_keep_mask"][0, 0, 0]))
        self.assertTrue(bool(diag["reference_keep_mask"][1, 0, 0]))
        self.assertFalse(bool(diag["reference_keep_mask"][2, 0, 0]))
        self.assertGreaterEqual(diag["reference_corr"][0, 0, 0], 0.95)
        self.assertEqual(7, diag["status_code"][2, 0, 0])
        self.assertTrue(np.all(np.isnan(out[:, 2, 0, 0])))
        self.assertTrue(np.all(np.isnan(out[10:15, 0, 0, 0])))


@unittest.skipIf(
    WaveformShapeMetricsDenoisedJoint is None,
    "waveform_shape_metrics_denoised_joint is not present",
)
class JointGraphConstructionTests(unittest.TestCase):
    def test_joint_detects_components_and_leaves_singletons_unchanged(self):
        pipeline = _configure_joint(WaveformShapeMetricsDenoisedJoint())
        block = _component_test_block()

        out, diag = pipeline._denoise_segment_block(block)

        np.testing.assert_allclose(out[:, 2, 0, 0], block[:, 2, 0, 0])
        self.assertEqual(2, diag["graph_component_count"])
        self.assertEqual(1, diag["graph_edge_count"])
        self.assertEqual(6, diag["status_code"][2, 0, 0])
        self.assertTrue(np.all(diag["status_code"][:2, 0, 0] == 0))

    def test_joint_removed_params_are_not_packed_or_used(self):
        block = _component_test_block()
        baseline = _configure_joint(WaveformShapeMetricsDenoisedJoint())
        stale_attrs = _configure_joint(WaveformShapeMetricsDenoisedJoint())
        stale_attrs.joint_top_k_neighbors = 1
        stale_attrs.joint_blend_alpha = 0.0
        stale_attrs.joint_self_jitter = 100.0

        out_baseline, diag = baseline._denoise_segment_block(block)
        out_stale, _ = stale_attrs._denoise_segment_block(block)

        np.testing.assert_allclose(out_baseline, out_stale)
        baseline._last_joint_denoise_diag = diag
        metrics = {}
        baseline._pack_joint_denoising_outputs(metrics)
        packed_names = {key.rsplit("/", 1)[-1] for key in metrics}
        self.assertNotIn("top_k_neighbors", packed_names)
        self.assertNotIn("blend_alpha", packed_names)
        self.assertNotIn("self_jitter", packed_names)
        self.assertIn("graph_component_count", packed_names)


if __name__ == "__main__":
    unittest.main()
