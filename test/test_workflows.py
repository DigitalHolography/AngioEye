import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines import BatchPipeline, PipelineDescriptor, PipelineResult  # noqa: E402
from pipelines.core.grouped_batch import extract_group_name  # noqa: E402
from workflows import run_batch  # noqa: E402


class SummaryPipeline(BatchPipeline):
    def run(self, context):
        records = []
        for path in context.processed_files:
            with h5py.File(path, "a") as h5file:
                records.append(
                    {
                        "group": extract_group_name(path.parent, context.output_dir),
                        "value": float(
                            h5file["/Processing/Metrics/waveform_shape_metrics/RI"][0]
                        ),
                    }
                )
                h5file.create_dataset("/AngioEye/Statistics/test/value", data=12)
        report = context.output_dir / "summary.json"
        report.write_text(json.dumps(records), encoding="utf-8")
        return PipelineResult(summary="Wrote summary", generated_paths=[str(report)])


def descriptor(name="Test summary", required="waveform_shape_metrics"):
    return PipelineDescriptor(
        name=name,
        description="Test",
        available=True,
        required_metrics=[required],
        pipeline_cls=SummaryPipeline,
    )


def write_eyeflow(path, *, primary=2, alternate=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5file:
        h5file.attrs["output_schema"] = "eyeflow_v2"
        for branch, value in (("Processing", primary), ("ProcessingAlt", alternate)):
            h5file.create_dataset(
                f"/{branch}/Metrics/waveform_shape_metrics/RI", data=[value]
            )


class StatisticalBatchTests(unittest.TestCase):
    def test_folder_preserves_groups_and_sources_and_uses_selected_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "cohort"
            sources = [inputs / "control" / "a_EF.h5", inputs / "patient" / "a_EF.h5"]
            for source in sources:
                write_eyeflow(source)
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
            progress = []
            result = run_batch(
                [inputs],
                [descriptor()],
                root / "results",
                processing_branch="ProcessingAlt",
                zip_outputs=True,
                progress=lambda *args: progress.append(args),
            )
            self.assertEqual([], result.failures)
            records = json.loads((result.output_dir / "summary.json").read_text())
            self.assertEqual(
                ["control", "patient"], [record["group"] for record in records]
            )
            self.assertEqual([9, 9], [record["value"] for record in records])
            self.assertEqual(
                before,
                [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources],
            )
            self.assertEqual((3, 3), progress[-1][:2])
            with zipfile.ZipFile(result.zip_path) as archive:
                self.assertIn("h5/control/a_AE.h5", archive.namelist())
                self.assertIn("summary.json", archive.namelist())
                self.assertNotIn("outputs.zip", archive.namelist())

    def test_zip_preserves_cohort_names_and_skips_macos_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "a_EF.h5"
            write_eyeflow(source)
            archive_path = root / "cohort.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(source, "control/a_EF.h5")
                archive.write(source, "patient/a_EF.h5")
                archive.writestr("__MACOSX/._a_EF.h5", b"metadata")
            result = run_batch([archive_path], [descriptor()], root / "results")
            self.assertEqual([], result.failures)
            records = json.loads((result.output_dir / "summary.json").read_text())
            self.assertEqual(
                ["control", "patient"], [record["group"] for record in records]
            )

    def test_bad_input_and_missing_metric_do_not_stop_other_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "good.h5"
            bad = root / "bad.h5"
            write_eyeflow(source)
            bad.write_text("corrupt", encoding="utf-8")
            result = run_batch(
                [source, bad],
                [descriptor("Unavailable metrics", "unknown"), descriptor()],
                root / "results",
            )
            self.assertEqual(2, len(result.failures))
            self.assertTrue((result.output_dir / "summary.json").is_file())
            manifest = json.loads((result.output_dir / "batch_run.json").read_text())
            self.assertEqual(
                ["failed", "completed"],
                [entry["status"] for entry in manifest["pipelines"]],
            )

    def test_rerun_excludes_previous_results_inside_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_eyeflow(root / "control" / "a_EF.h5")
            output = root / "results"
            for _run in range(2):
                result = run_batch([root], [descriptor()], output)
                self.assertEqual(1, len(result.input_files))
                self.assertEqual([], result.failures)

    def test_rejects_empty_selection_and_unsafe_archive_name(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_batch(["file.h5"], [], "results")
        with self.assertRaisesRegex(ValueError, "filename"):
            run_batch(
                ["file.h5"],
                [descriptor()],
                "results",
                zip_outputs=True,
                zip_name="../out.zip",
            )

    def test_output_zip_cannot_replace_source_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "a.h5"
            write_eyeflow(source)
            archive_path = root / "outputs.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(source, "a.h5")
            before = archive_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "differ from the input"):
                run_batch([archive_path], [descriptor()], root, zip_outputs=True)
            self.assertEqual(before, archive_path.read_bytes())

    def test_archive_failure_is_recorded_in_final_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "a.h5"
            write_eyeflow(source)
            with mock.patch(
                "workflows.batch.create_zip_from_tree", side_effect=OSError("disk full")
            ):
                result = run_batch(
                    [source], [descriptor()], root / "results", zip_outputs=True
                )
            manifest = json.loads((result.output_dir / "batch_run.json").read_text())
            self.assertEqual(1, len(result.failures))
            self.assertIn("1 failure(s)", result.summary_message)
            self.assertEqual(result.failures, manifest["failures"])
            self.assertEqual(result.summary_message, manifest["summary"])


if __name__ == "__main__":
    unittest.main()
