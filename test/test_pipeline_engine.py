import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import (  # noqa: E402
    ANGIOEYE_PROCESSING_ROOT,
    ANGIOEYE_SIGNALS_ROOT,
    EYEFLOW_ROOT,
    find_eyeflow_dataset,
    find_pipeline_group,
    read_signal_datasets,
)
from pipeline_engine import run_pipeline_file, run_postprocesses  # noqa: E402
from pipelines import ProcessResult  # noqa: E402


class _PipelineDescriptor:
    name = "Demo"

    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail

    def instantiate(self):
        return _Pipeline(self.should_fail)


class _Pipeline:
    name = "Demo"

    def __init__(self, should_fail: bool):
        self.should_fail = should_fail

    def run(self, _h5file):
        if self.should_fail:
            raise RuntimeError("boom")
        return ProcessResult(metrics={"value": 3.0})


class PipelineEngineTests(unittest.TestCase):
    def test_run_pipeline_file_without_pipelines_omits_empty_processing_group(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass

            output_path = run_pipeline_file(
                input_path,
                [],
                tmp_path / "outputs",
            )

            with h5py.File(output_path, "r") as h5file:
                self.assertNotIn(ANGIOEYE_PROCESSING_ROOT, h5file)

    def test_run_pipeline_file_uses_unique_output_names_and_writes_h5(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass

            first_output = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
            )
            second_output = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
            )

            self.assertEqual("sample_pipelines_result.h5", first_output.name)
            self.assertEqual("sample_1_pipelines_result.h5", second_output.name)
            self.assertEqual(tmp_path / "outputs" / "h5", first_output.parent)
            self.assertTrue(first_output.exists())
            with h5py.File(first_output, "r") as h5:
                self.assertIn(f"{ANGIOEYE_PROCESSING_ROOT}/demo/value", h5)
                self.assertIn(ANGIOEYE_SIGNALS_ROOT, h5)

    def test_run_pipeline_file_formats_pipeline_failures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass

            with self.assertRaisesRegex(RuntimeError, "Pipeline 'Demo' failed"):
                run_pipeline_file(
                    input_path,
                    [_PipelineDescriptor(should_fail=True)],
                    tmp_path / "outputs",
                )

    def test_persist_source_is_opt_in_and_copies_source_contents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w") as h5file:
                h5file.attrs["source_attr"] = "from-eyeflow"
                h5file.create_dataset("eyeflow", data=[1, 2, 3])

            default_output = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "default",
            )
            persisted_output = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "persisted",
                persist_source=True,
            )

            with h5py.File(default_output, "r") as h5file:
                self.assertNotIn("eyeflow", h5file)
            with h5py.File(persisted_output, "r") as h5file:
                self.assertIn(f"{EYEFLOW_ROOT}/eyeflow", h5file)
                self.assertNotIn("eyeflow", h5file)
                self.assertEqual(
                    "from-eyeflow",
                    h5file[EYEFLOW_ROOT].attrs["source_attr"],
                )

    def test_persisted_eyeflow_v2_source_stays_under_eyeflow_namespace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample_EF.h5"
            with h5py.File(input_path, "w") as h5file:
                h5file.attrs["output_schema"] = "eyeflow_v2"
                h5file.create_dataset(
                    "/Processing/VelocityPerBeat/Artery/Raw/value",
                    data=[1.0, 2.0],
                )
                h5file.require_group(
                    "/Processing/Metrics/waveform_shape_metrics"
                )

            output_path = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "persisted",
                persist_source=True,
            )

            with h5py.File(output_path, "r") as h5file:
                self.assertIn(
                    f"{EYEFLOW_ROOT}/Processing/VelocityPerBeat/Artery/Raw/value",
                    h5file,
                )
                self.assertNotIn(
                    "/Processing/VelocityPerBeat/Artery/Raw/value",
                    h5file,
                )
                self.assertEqual(
                    "eyeflow_v2",
                    h5file[EYEFLOW_ROOT].attrs["output_schema"],
                )
                self.assertIsNotNone(
                    find_pipeline_group(h5file, "waveform_shape_metrics")
                )

    def test_signals_are_copied_without_persisting_the_eye_flow_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            expected_signals = {
                "artery/raw": [1.0, 2.0, 3.0],
                "artery/bandlimited": [4.0, 5.0, 6.0],
                "vein/raw": [7.0, 8.0, 9.0],
                "vein/bandlimited": [10.0, 11.0, 12.0],
            }
            source_paths = {
                "artery/raw": (
                    "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
                ),
                "artery/bandlimited": (
                    "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
                ),
                "vein/raw": "/Vein/VelocityPerBeat/VelocitySignalPerBeat/value",
                "vein/bandlimited": (
                    "/Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
                ),
            }
            with h5py.File(input_path, "w") as h5file:
                h5file.create_dataset("eyeflow", data=[99.0])
                for signal_name, source_path in source_paths.items():
                    h5file.create_dataset(
                        source_path,
                        data=expected_signals[signal_name],
                    )

            with mock.patch(
                "input_output.hdf5_io.copy_signal_datasets",
                side_effect=AssertionError(
                    "signal datasets should be captured from the pipeline input pass"
                ),
            ):
                output_path = run_pipeline_file(
                    input_path,
                    [_PipelineDescriptor()],
                    tmp_path / "outputs",
                    persist_source=False,
                )

            with h5py.File(output_path, "r") as h5file:
                self.assertIn(ANGIOEYE_SIGNALS_ROOT, h5file)
                self.assertNotIn("eyeflow", h5file)
                for signal_name, expected in expected_signals.items():
                    signal_path = f"{ANGIOEYE_SIGNALS_ROOT}/{signal_name}"
                    self.assertIn(signal_path, h5file)
                    self.assertEqual(expected, h5file[signal_path][()].tolist())

    def test_signal_reader_accepts_namespaced_eyeflow_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "namespaced.h5"
            with h5py.File(path, "w") as h5file:
                h5file.create_dataset(
                    f"{EYEFLOW_ROOT}/Artery/VelocityPerBeat/"
                    "VelocitySignalPerBeat/value",
                    data=[1.0, 2.0],
                )

            with h5py.File(path, "r") as h5file:
                signals = read_signal_datasets(h5file)
                dataset = find_eyeflow_dataset(
                    h5file,
                    "Artery/VelocityPerBeat/VelocitySignalPerBeat/value",
                )

            self.assertEqual([1.0, 2.0], signals["artery/raw"].tolist())
            self.assertIsNotNone(dataset)
    def test_eyeflow_v2_signals_are_copied_without_persisting_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample_EF.h5"
            source_paths = {
                "artery/raw": "/Processing/VelocityPerBeat/Artery/Raw/value",
                "artery/bandlimited": (
                    "/Processing/VelocityPerBeat/Artery/BandLimited/value"
                ),
                "vein/raw": "/Processing/VelocityPerBeat/Vein/Raw/value",
                "vein/bandlimited": (
                    "/Processing/VelocityPerBeat/Vein/BandLimited/value"
                ),
            }
            with h5py.File(input_path, "w") as h5file:
                h5file.attrs["output_schema"] = "eyeflow_v2"
                for index, source_path in enumerate(source_paths.values(), start=1):
                    h5file.create_dataset(
                        source_path,
                        data=[[index, index + 0.5]],
                    )

            output_path = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
                persist_source=False,
            )

            with h5py.File(output_path, "r") as h5file:
                for index, signal_name in enumerate(source_paths, start=1):
                    signal_path = f"{ANGIOEYE_SIGNALS_ROOT}/{signal_name}"
                    self.assertEqual(
                        [[index, index + 0.5]],
                        h5file[signal_path][()].tolist(),
                    )

    def test_run_postprocesses_propagates_metadata_failures(self):
        calls: list[str] = []
        logs: list[str] = []
        failures: list[str] = []
        progress: list[float] = []

        class _Postprocess:
            def __init__(self, name, result):
                self._name = name
                self._result = result

            def run(self, _context):
                calls.append(self._name)
                return self._result

        class _Descriptor:
            def __init__(self, name, result):
                self.name = name
                self._result = result

            def instantiate(self):
                return _Postprocess(self.name, self._result)

        first_result = SimpleNamespace(
            summary="partial",
            metadata={"failures": ["Composite Scoring skipped broken.h5"]},
        )
        second_result = SimpleNamespace(summary="done", metadata={})

        with mock.patch(
            "pipeline_engine.execution.PostprocessContext",
            lambda **kwargs: kwargs,
        ):
            run_postprocesses(
                postprocesses=(
                    _Descriptor("Composite Scoring", first_result),
                    _Descriptor("Next Postprocess", second_result),
                ),
                output_dir=Path("."),
                processed_outputs=(Path("ok.h5"),),
                input_h5_paths=(Path("ok_input.h5"),),
                input_path=Path("archive.zip"),
                selected_pipeline_names=("waveform_shape_metrics",),
                failures=failures,
                zip_outputs=False,
                log=logs.append,
                advance_progress=progress.append,
            )

        self.assertEqual(["Composite Scoring", "Next Postprocess"], calls)
        self.assertEqual(["Composite Scoring skipped broken.h5"], failures)
        self.assertIn("[POST WARN] Composite Scoring skipped broken.h5", logs)
        self.assertEqual([1.0, 1.0], progress)

    def test_run_pipeline_file_allows_worker_safe_noop_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass

            output_path = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
                log=None,
                advance_progress=None,
                write_idle_callback=None,
            )

            self.assertTrue(output_path.exists())

    def test_run_pipeline_file_preserves_relative_parent_under_h5_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass

            output_path = run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
                Path("cohort") / "subject",
            )

            self.assertEqual(
                tmp_path
                / "outputs"
                / "h5"
                / "cohort"
                / "subject"
                / "sample_pipelines_result.h5",
                output_path,
            )

    def test_run_pipeline_file_records_detailed_timing_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            with h5py.File(input_path, "w"):
                pass
            timings: list[tuple[str, float]] = []

            run_pipeline_file(
                input_path,
                [_PipelineDescriptor()],
                tmp_path / "outputs",
                record_timing=lambda label, seconds: timings.append(
                    (label, seconds)
                ),
            )

        labels = {label for label, _seconds in timings}
        self.assertIn("per-file output path allocation", labels)
        self.assertIn("per-file input HDF5 open for pipeline compute", labels)
        self.assertIn("per-file input HDF5 close after pipeline compute", labels)
        self.assertIn("per-pipeline instantiate [Demo]", labels)
        self.assertIn("per-pipeline compute [Demo]", labels)
        self.assertIn("per-pipeline callback/log/progress [Demo]", labels)
        self.assertIn("per-file pipeline compute", labels)
        self.assertIn("per-file output write", labels)
        self.assertIn(
            "per-file output write: create output HDF5 (source copy disabled)",
            labels,
        )
        self.assertIn(
            "per-file output write: convert process results to metric trees",
            labels,
        )
        self.assertIn(
            "per-file output write: write metric trees into HDF5",
            labels,
        )


if __name__ == "__main__":
    unittest.main()
