from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_engine import run_pipeline_file  # noqa: E402
from pipelines.core.base import PipelineDescriptor  # noqa: E402
from pipelines.topological_metrics import (  # noqa: E402
    BEAT_PERIOD_PATH,
    OPTIC_DISC_CENTER_PATH,
    OPTIC_DISC_LABEL,
    OPTIC_DISC_MASK_PATH,
    REGION_AXIS_LABEL,
    REGION_NAMES,
    TOPOLOGICAL_METRICS_REQUIRED_PATHS,
    VESSEL_PATHS,
    TopologicalMetricsPipeline,
)
from ui.controllers.pipeline_library import PipelineLibraryController  # noqa: E402
from workflows import WorkflowInputSelection  # noqa: E402
from workflows.pipeline_readiness import resolve_pipeline_inputs  # noqa: E402


def _waveform(sample_count: int, peak_index: int, scale: float) -> np.ndarray:
    x = np.arange(sample_count, dtype=np.float32)
    width = max(sample_count / 7.0, 1.0)
    return np.float32(0.2) + np.float32(scale) * np.exp(
        -(((x - np.float32(peak_index)) / np.float32(width)) ** 2)
    )


def _write_vessel(
    h5file: h5py.File,
    vessel_index: int,
    centers: np.ndarray,
    *,
    n_beats: int,
    sample_count: int,
) -> None:
    paths = VESSEL_PATHS[vessel_index]
    n_branches, n_radii = centers.shape[:2]
    branch_ids = np.arange(1, n_branches + 1, dtype=np.int32)
    label_map = np.zeros((21, 21), dtype=np.int32)
    for branch_index, branch_id in enumerate(branch_ids):
        label_map[2 + branch_index : 4 + branch_index, 2:19] = branch_id

    raw = np.full(
        (sample_count, n_beats, n_branches, n_radii),
        np.nan,
        dtype=np.float32,
    )
    for beat_index in range(n_beats):
        for branch_index in range(n_branches):
            for radius_index in range(n_radii):
                if not np.all(np.isfinite(centers[branch_index, radius_index])):
                    continue
                raw[:, beat_index, branch_index, radius_index] = _waveform(
                    sample_count,
                    peak_index=(2 + 2 * branch_index + 3 * radius_index + beat_index),
                    scale=1.0 + branch_index + radius_index,
                )

    h5file.create_dataset(paths.branch_ids, data=branch_ids)
    h5file.create_dataset(paths.branch_label_map, data=label_map)
    h5file.create_dataset(paths.segment_center_xy, data=centers)
    h5file.create_dataset(paths.raw_waveform, data=raw)
    h5file.create_dataset(paths.bandlimited_waveform, data=raw)


def _write_topological_input(path: Path, *, source_metrics: bool = False) -> None:
    # EyeFlow SegmentCenterXY uses a bottom-up Y axis: larger Y values are top.
    artery_centers = np.asarray(
        [
            [[5.0, 5.0], [15.0, 15.0]],
            [[15.0, 5.0], [5.0, 15.0]],
        ],
        dtype=np.float32,
    )
    vein_centers = np.asarray(
        [[[5.0, 5.0], [15.0, 15.0]]],
        dtype=np.float32,
    )
    n_beats = 2
    sample_count = 24

    with h5py.File(path, "w") as h5file:
        optic_mask = np.zeros((21, 21), dtype=bool)
        optic_mask[8:13, 8:13] = True
        h5file.create_dataset(OPTIC_DISC_CENTER_PATH, data=[10.0, 10.0])
        h5file.create_dataset(OPTIC_DISC_MASK_PATH, data=optic_mask)
        h5file.create_dataset(BEAT_PERIOD_PATH, data=[[1.0, 1.0]])
        _write_vessel(
            h5file,
            0,
            artery_centers,
            n_beats=n_beats,
            sample_count=sample_count,
        )
        _write_vessel(
            h5file,
            1,
            vein_centers,
            n_beats=n_beats,
            sample_count=sample_count,
        )

        if source_metrics:
            pipeline = TopologicalMetricsPipeline()
            for vessel_index, centers in enumerate((artery_centers, vein_centers)):
                paths = VESSEL_PATHS[vessel_index]
                shape = (n_beats, *centers.shape[:2])
                for signal_type in ("raw", "bandlimited"):
                    for metric_index, metric_name in enumerate(
                        pipeline._metric_names(),
                        start=1,
                    ):
                        values = np.full(shape, metric_index, dtype=np.float32)
                        h5file.create_dataset(
                            paths.source_metric_path(signal_type, metric_name),
                            data=values,
                        )


class TopologicalMetricsTests(unittest.TestCase):
    def test_regions_split_each_branch_by_radius_and_aggregate_per_beat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path)

            with h5py.File(path, "r") as h5file:
                result = TopologicalMetricsPipeline().run(h5file)
                raw = np.asarray(h5file[VESSEL_PATHS[0].raw_waveform])

            self.assertEqual(
                "angioeye_recomputed_from_waveforms",
                result.attrs["metric_source"],
            )
            self.assertNotIn("topology/artery/region_membership", result.metrics)
            self.assertNotIn("regions/names", result.metrics)
            self.assertEqual(list(REGION_NAMES), result.metrics["topology/names"])
            label_map = result.metrics["topology/artery/branch_label_map"].data
            self.assertEqual((21, 21), label_map.shape)
            np.testing.assert_array_equal(
                label_map[:, 10],
                np.full(21, REGION_AXIS_LABEL, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                label_map[10, :],
                np.full(21, REGION_AXIS_LABEL, dtype=np.int32),
            )
            self.assertEqual(1, int(label_map[2, 2]))
            self.assertEqual(OPTIC_DISC_LABEL, int(label_map[8, 8]))
            self.assertNotIn(
                "topology/artery/branch_label_map_visualization_rgb",
                result.metrics,
            )
            self.assertNotIn("topology/artery/branch_color_rgb", result.metrics)

            pipeline = TopologicalMetricsPipeline()
            expected = np.asarray(
                [
                    pipeline._compute_metrics_1d(raw[:, beat, 1, 1], 1.0)[
                        "t_max_over_T"
                    ]
                    for beat in range(raw.shape[1])
                ]
            )
            actual = result.metrics["artery/top_left/global/raw/t_max_over_T"].data
            np.testing.assert_allclose(actual, expected, equal_nan=True)

            branch_prefix = "artery/top_left/by_branch/branch_2"
            np.testing.assert_allclose(
                result.metrics[f"{branch_prefix}/raw/t_max_over_T"].data,
                expected,
                equal_nan=True,
            )
            self.assertFalse(
                any("segment" in key for key in result.metrics),
                result.metrics.keys(),
            )
            self.assertIn(
                "artery/bottom_left/by_branch/branch_1/raw/t_max_over_T",
                result.metrics,
            )
            self.assertNotIn(
                "artery/top_left/by_branch/branch_1/raw/t_max_over_T",
                result.metrics,
            )

            for vessel in ("artery", "vein"):
                for region in REGION_NAMES:
                    for signal_type in ("raw", "bandlimited"):
                        for metric_name in pipeline._metric_names():
                            self.assertIn(
                                f"{vessel}/{region}/global/{signal_type}/{metric_name}",
                                result.metrics,
                            )

    def test_branch_metrics_only_aggregate_segments_of_that_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path, source_metrics=True)
            paths = VESSEL_PATHS[0]

            with h5py.File(path, "r+") as h5file:
                pi_path = paths.source_metric_path("raw", "PI")
                values = np.asarray(h5file[pi_path])
                values[:, 0, 0] = 10.0
                values[:, 0, 1] = 11.0
                values[:, 1, 0] = 30.0
                values[:, 1, 1] = 31.0
                del h5file[pi_path]
                h5file.create_dataset(pi_path, data=values)

            with h5py.File(path, "r") as h5file:
                result = TopologicalMetricsPipeline().run(h5file)

            region_prefix = "artery/top"
            np.testing.assert_array_equal(
                result.metrics[f"{region_prefix}/global/raw/PI"].data,
                [21.0, 21.0],
            )
            np.testing.assert_array_equal(
                result.metrics[f"{region_prefix}/by_branch/branch_1/raw/PI"].data,
                [11.0, 11.0],
            )
            np.testing.assert_array_equal(
                result.metrics[f"{region_prefix}/by_branch/branch_2/raw/PI"].data,
                [31.0, 31.0],
            )

    def test_branch_by_region_hierarchy_is_written_to_h5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.h5"
            _write_topological_input(input_path, source_metrics=True)
            descriptor = PipelineDescriptor(
                name="topological_metrics",
                description=TopologicalMetricsPipeline.description,
                available=True,
                pipeline_cls=TopologicalMetricsPipeline,
            )

            output_path = run_pipeline_file(
                input_path,
                [descriptor],
                tmp_path / "outputs",
            )

            prefix = (
                "/AngioEye/Processing/topological_metrics/artery/"
                "top_left/by_branch/branch_2"
            )
            with h5py.File(output_path, "r") as h5file:
                topology_group = h5file[
                    "/AngioEye/Processing/topological_metrics/topology"
                ]
                self.assertEqual(
                    {"artery", "vein", "optic_disc", "names"},
                    set(topology_group.keys()),
                )
                self.assertIn(f"{prefix}/raw/PI", h5file)
                label_map_path = (
                    "/AngioEye/Processing/topological_metrics/topology/artery/"
                    "branch_label_map"
                )
                self.assertIn(label_map_path, h5file)
                self.assertEqual((21, 21), h5file[label_map_path].shape)
                self.assertEqual(
                    REGION_AXIS_LABEL,
                    h5file[label_map_path].attrs["axis_label"],
                )
                self.assertEqual(
                    OPTIC_DISC_LABEL,
                    h5file[label_map_path].attrs["optic_disc_label"],
                )
                self.assertNotIn(
                    "/AngioEye/Processing/topological_metrics/topology/artery/"
                    "region_membership",
                    h5file,
                )
                region_group = h5file[
                    "/AngioEye/Processing/topological_metrics/artery/top_left"
                ]
                self.assertEqual(
                    {"by_branch", "global"},
                    set(region_group.keys()),
                )
                self.assertNotIn(
                    "/AngioEye/Processing/topological_metrics/regions",
                    h5file,
                )
                self.assertNotIn(
                    "/AngioEye/Processing/topological_metrics/artery/by_region",
                    h5file,
                )

    def test_prefers_complete_eyeflow_per_segment_metric_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path, source_metrics=True)

            with h5py.File(path, "r") as h5file:
                result = TopologicalMetricsPipeline().run(h5file)

            self.assertEqual(
                "eyeflow_per_segment_metrics",
                result.attrs["metric_source"],
            )
            metric_index = (
                list(TopologicalMetricsPipeline()._metric_names()).index("PI") + 1
            )
            actual = result.metrics["vein/top_right/global/raw/PI"].data
            np.testing.assert_array_equal(
                actual,
                np.full(2, metric_index, dtype=np.float32),
            )

    def test_missing_topology_key_has_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path)
            with h5py.File(path, "r+") as h5file:
                del h5file[OPTIC_DISC_CENTER_PATH]

            with h5py.File(path, "r") as h5file:
                with self.assertRaisesRegex(
                    KeyError,
                    "Missing required EyeFlow topology dataset",
                ):
                    TopologicalMetricsPipeline().run(h5file)

    def test_empty_vessel_placeholder_produces_empty_region_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path)
            vein = VESSEL_PATHS[1]
            with h5py.File(path, "r+") as h5file:
                for dataset_path in (
                    vein.branch_ids,
                    vein.branch_label_map,
                    vein.segment_center_xy,
                    vein.raw_waveform,
                    vein.bandlimited_waveform,
                ):
                    del h5file[dataset_path]
                h5file.create_dataset(vein.branch_ids, data=np.empty(0, np.int32))
                h5file.create_dataset(
                    vein.branch_label_map,
                    data=np.zeros((21, 21), np.int32),
                )
                h5file.create_dataset(
                    vein.segment_center_xy,
                    data=np.empty((0, 2, 2), np.float32),
                )
                dummy = np.full((24, 2, 1, 2), np.nan, np.float32)
                h5file.create_dataset(vein.raw_waveform, data=dummy)
                h5file.create_dataset(vein.bandlimited_waveform, data=dummy)

            with h5py.File(path, "r") as h5file:
                result = TopologicalMetricsPipeline().run(h5file)

            self.assertTrue(
                np.all(np.isnan(result.metrics["vein/top/global/raw/PI"].data))
            )


class PipelineInputStatusTests(unittest.TestCase):
    @staticmethod
    def _descriptor() -> PipelineDescriptor:
        return PipelineDescriptor(
            name="topological_metrics",
            description=TopologicalMetricsPipeline.description,
            available=True,
            pipeline_cls=TopologicalMetricsPipeline,
        )

    @staticmethod
    def _controller(selection: WorkflowInputSelection):
        run_controller = SimpleNamespace(
            collect_input_selection=lambda: selection,
        )
        app = SimpleNamespace(run_controller=run_controller)
        return PipelineLibraryController(app)

    def test_status_is_available_when_all_required_keys_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path)
            controller = self._controller(
                WorkflowInputSelection(
                    convention="legacy",
                    legacy_input_paths=(path,),
                )
            )

            self.assertEqual("Available", controller.status_text(self._descriptor()))

    def test_status_reports_missing_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            _write_topological_input(path)
            with h5py.File(path, "r+") as h5file:
                del h5file[TOPOLOGICAL_METRICS_REQUIRED_PATHS[0]]
            controller = self._controller(
                WorkflowInputSelection(
                    convention="legacy",
                    legacy_input_paths=(path,),
                )
            )

            self.assertEqual(
                "Missing 1 required key",
                controller.status_text(self._descriptor()),
            )

    def test_status_reports_missing_eyeflow_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            holo_path = Path(tmp) / "sample.holo"
            holo_path.write_text("", encoding="utf-8")
            controller = self._controller(
                WorkflowInputSelection(
                    convention="holo",
                    holo_paths=(holo_path,),
                )
            )

            self.assertEqual(
                "Missing EF file: sample",
                controller.status_text(self._descriptor()),
            )

    def test_holo_selection_resolves_nested_eyeflow_h5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holo_path = root / "sample.holo"
            holo_path.write_text("", encoding="utf-8")
            ef_h5 = root / "sample" / "sample_EF" / "h5" / "sample_EF.h5"
            ef_h5.parent.mkdir(parents=True)
            with h5py.File(ef_h5, "w"):
                pass
            resolved = resolve_pipeline_inputs(
                WorkflowInputSelection(
                    convention="holo",
                    holo_paths=(holo_path,),
                )
            )

            self.assertEqual((("sample", ef_h5),), resolved.files)
            self.assertEqual((), resolved.missing_sources)


if __name__ == "__main__":
    unittest.main()
