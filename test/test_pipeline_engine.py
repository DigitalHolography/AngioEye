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

from input_output import ANGIOEYE_PROCESSING_ROOT  # noqa: E402
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
            self.assertTrue(first_output.exists())
            with h5py.File(first_output, "r") as h5:
                self.assertIn(f"{ANGIOEYE_PROCESSING_ROOT}/demo/value", h5)

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
