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

from input_output.archive_io import reset_output_dir  # noqa: E402
from pipelines import BatchPipeline, PipelineDescriptor, PipelineResult  # noqa: E402
from pipelines.core.grouped_batch import extract_group_name  # noqa: E402
from pipelines.html_summary import WaveformMetricSummaryTablesPipeline  # noqa: E402
from workflows import run_batch  # noqa: E402


class SummaryPipeline(BatchPipeline):
    def run(self, context):
        records = []
        for path in context.processed_files:
            with h5py.File(path, "a") as h5file:
                records.append(
                    {
                        "group": extract_group_name(path.parent, context.batch_root),
                        "value": float(
                            h5file["/Processing/Metrics/waveform_shape_metrics/RI"][0]
                        ),
                    }
                )
                h5file.create_dataset("/AngioEye/Statistics/test/value", data=12)
        report = context.output_dir / "summary.json"
        report.write_text(json.dumps(records), encoding="utf-8")
        return PipelineResult(summary="Wrote summary", generated_paths=[str(report)])


class ReplacingReportPipeline(BatchPipeline):
    def run(self, context):
        report_dir = context.output_dir / "Variability and heterogeneity"
        reset_output_dir(report_dir)
        report = report_dir / "report.txt"
        report.write_text("Shared cohort report", encoding="utf-8")
        return PipelineResult(summary="Wrote report", generated_paths=[str(report)])


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


def write_acquisition(folder, name="a", **options):
    folder.mkdir(parents=True, exist_ok=True)
    holo = folder / f"{name}.holo"
    holo.write_bytes(b"raw acquisition")
    source = folder / name / f"{name}_EF" / "h5" / f"{name}_EF.h5"
    write_eyeflow(source, **options)
    return holo, source


class StatisticalBatchTests(unittest.TestCase):
    def test_folder_preserves_groups_and_sources_and_uses_selected_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "cohort"
            acquisitions = [
                write_acquisition(inputs / group) for group in ("control", "patient")
            ]
            sources = [source for _holo, source in acquisitions]
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
            with h5py.File(root / "results/control/a/a_AE/h5/a_AE.h5") as analysis:
                self.assertEqual(analysis.attrs["angioeye_input_file"], str(sources[0]))
                self.assertEqual(
                    analysis.attrs["angioeye_input_holo"], str(acquisitions[0][0])
                )
            with zipfile.ZipFile(result.zip_path) as archive:
                self.assertIn("control/a/a_AE/h5/a_AE.h5", archive.namelist())
                self.assertIn("cohort-results/summary.json", archive.namelist())
                self.assertNotIn("outputs.zip", archive.namelist())

    def test_folder_ignores_other_exports_and_unselected_acquisitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "cohort"
            _holo, source = write_acquisition(inputs / "control")
            write_eyeflow(inputs / "stray_EF.h5", primary=100)
            write_eyeflow(inputs / "a" / "a_HD" / "h5" / "a_HD_output.h5")
            write_eyeflow(inputs / "b" / "b_EF" / "h5" / "b_EF.h5")
            result = run_batch([inputs], [descriptor()], root / "results")
            self.assertEqual([], result.failures)
            self.assertEqual([source], result.input_files)

    def test_bad_input_and_missing_metric_do_not_stop_other_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good_holo, _source = write_acquisition(root, "good")
            bad_holo, bad = write_acquisition(root, "bad")
            bad.write_text("corrupt", encoding="utf-8")
            result = run_batch(
                [good_holo, bad_holo],
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
            write_acquisition(root / "control")
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

    def test_rejects_direct_hdf5_and_zip_inputs_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _holo, source = write_acquisition(root)
            archive_path = root / "outputs.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(source, "a.h5")
            for path in (source, archive_path):
                with self.subTest(path=path):
                    before = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, ".holo"):
                        run_batch([path], [descriptor()], root / "results")
                    self.assertEqual(before, path.read_bytes())
                    self.assertFalse((root / "results").exists())

    def test_archive_failure_is_recorded_in_final_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, _source = write_acquisition(root)
            with mock.patch(
                "workflows.batch.create_zip_from_tree", side_effect=OSError("disk full")
            ):
                result = run_batch(
                    [holo], [descriptor()], root / "results", zip_outputs=True
                )
            manifest = json.loads((result.output_dir / "batch_run.json").read_text())
            self.assertEqual(1, len(result.failures))
            self.assertIn("1 failure(s)", result.summary_message)
            self.assertEqual(result.failures, manifest["failures"])
            self.assertEqual(result.summary_message, manifest["summary"])

    def test_single_hologram_only_analyzes_its_matching_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, source = write_acquisition(root, "a")
            write_acquisition(root, "b", primary=100)
            result = run_batch([holo], [descriptor()], root / "results")
            self.assertEqual([source], result.input_files)
            self.assertEqual([], result.failures)
            self.assertTrue((result.output_dir / "h5/a_AE.h5").is_file())
            records = json.loads((result.output_dir / "summary.json").read_text())
            self.assertEqual([{"group": "all", "value": 2.0}], records)
            self.assertEqual(b"raw acquisition", holo.read_bytes())

    def test_default_single_output_sits_beside_eyeflow_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, source = write_acquisition(root, "scan.v2_AE")
            before = source.read_bytes()
            result = run_batch([holo], [descriptor()])
            app_dir = root / "scan.v2_AE" / "scan.v2_AE_AE"
            self.assertEqual(app_dir, result.output_dir)
            self.assertTrue((app_dir / "h5/scan.v2_AE_AE.h5").is_file())
            self.assertTrue((app_dir / "summary.json").is_file())
            self.assertTrue((app_dir / "batch_run.json").is_file())
            self.assertEqual(before, source.read_bytes())
            self.assertFalse((root / "h5").exists())

    def test_custom_parent_keeps_named_acquisition_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, source = write_acquisition(root / "data", "sample")
            output = root / "results"
            result = run_batch([holo], [descriptor()], output)
            self.assertEqual(output / "sample/sample_AE", result.output_dir)
            analysis = output / "sample/sample_AE/h5/sample_AE.h5"
            self.assertTrue(analysis.is_file())
            manifest = json.loads((result.output_dir / "batch_run.json").read_text())
            self.assertEqual([str(Path("h5/sample_AE.h5"))], manifest["analysis_files"])
            self.assertEqual([str(source)], manifest["input_files"])

    def test_html_reports_share_each_acquisition_ae_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquisitions = [
                write_acquisition(root / group) for group in ("control", "patient")
            ]
            for _holo, source in acquisitions:
                with h5py.File(source, "a") as original:
                    for vessel in ("artery", "vein"):
                        original.create_dataset(
                            f"/Processing/Metrics/waveform_shape_metrics/{vessel}/global/bandlimited/RI",
                            data=[0.6, 0.7],
                        )
            html_tool = PipelineDescriptor(
                name="HTML summary",
                description="",
                available=True,
                required_metrics=["waveform_shape_metrics"],
                pipeline_cls=WaveformMetricSummaryTablesPipeline,
            )
            before = [source.read_bytes() for _holo, source in acquisitions]
            single = run_batch([acquisitions[0][0]], [html_tool])
            self.assertEqual([], single.failures)
            single_html = root / "control/a/a_AE/html/a_AE.html"
            self.assertIn(single_html, single.generated_paths)
            self.assertIn("RI", single_html.read_text(encoding="utf-8"))
            output = root / "custom"
            batch = run_batch([root], [html_tool], output, zip_outputs=True)
            self.assertEqual([], batch.failures)
            for group in ("control", "patient"):
                html = output / group / "a/a_AE/html/a_AE.html"
                self.assertIn(html, batch.generated_paths)
                self.assertTrue(html.is_file())
            with zipfile.ZipFile(batch.zip_path) as archive:
                self.assertIn("control/a/a_AE/html/a_AE.html", archive.namelist())
                self.assertIn("patient/a/a_AE/html/a_AE.html", archive.namelist())
            self.assertFalse((output / "html").exists())
            self.assertEqual(
                before, [source.read_bytes() for _holo, source in acquisitions]
            )

    def test_default_batch_exports_only_ae_outputs_in_zip_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquisitions = [
                write_acquisition(root / group) for group in ("control", "patient")
            ]
            before = [source.read_bytes() for _holo, source in acquisitions]
            stray = root / "personal_notes.txt"
            stray.write_text("not an output", encoding="utf-8")
            hd = root / "control/a/a_HD/h5/a_HD_output.h5"
            write_eyeflow(hd)
            for _run in range(2):
                result = run_batch([root], [descriptor()], zip_outputs=True)
                self.assertEqual(root / "cohort-results", result.output_dir)
                self.assertEqual([], result.failures)
                self.assertEqual(2, len(result.input_files))
                records = json.loads((result.output_dir / "summary.json").read_text())
                self.assertEqual(
                    ["control", "patient"], [record["group"] for record in records]
                )
                with zipfile.ZipFile(result.zip_path) as archive:
                    names = set(archive.namelist())
                    self.assertIn("control/a/a_AE/h5/a_AE.h5", names)
                    self.assertIn("patient/a/a_AE/h5/a_AE.h5", names)
                    self.assertIn("cohort-results/summary.json", names)
                    self.assertIn("cohort-results/batch_run.json", names)
                    self.assertNotIn("personal_notes.txt", names)
                    self.assertFalse(
                        any(
                            name.endswith(".holo") or "_EF/" in name or "_HD/" in name
                            for name in names
                        )
                    )
                self.assertEqual(
                    before, [source.read_bytes() for _holo, source in acquisitions]
                )

    def test_folder_with_one_acquisition_keeps_its_group_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_acquisition(root / "control")
            result = run_batch([root], [descriptor()])
            records = json.loads((result.output_dir / "summary.json").read_text())
            self.assertEqual(["control"], [record["group"] for record in records])
            self.assertEqual([], result.failures)

    def test_batch_report_replacement_cannot_delete_same_named_input_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquisitions = [
                write_acquisition(root / group)
                for group in ("control", "Variability and heterogeneity")
            ]
            sources = [path for acquisition in acquisitions for path in acquisition]
            before = [path.read_bytes() for path in sources]
            report_tool = descriptor()
            report_tool.pipeline_cls = ReplacingReportPipeline
            for _run in range(2):
                result = run_batch([root], [report_tool])
                self.assertEqual([], result.failures)
                self.assertTrue(
                    (
                        root / "cohort-results/Variability and heterogeneity/report.txt"
                    ).is_file()
                )
                self.assertEqual(before, [path.read_bytes() for path in sources])

    def test_shared_report_folder_cannot_contain_selected_holograms(self):
        for processed in (True, False):
            with (
                self.subTest(processed=processed),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                good, _source = write_acquisition(root / "control")
                blocked, blocked_source = write_acquisition(root / "cohort-results")
                if not processed:
                    blocked_source.unlink()
                before = blocked.read_bytes()
                with self.assertRaisesRegex(ValueError, "contains selected inputs"):
                    run_batch([root], [descriptor()])
                self.assertEqual(before, blocked.read_bytes())
                self.assertFalse((good.parent / "a/a_AE").exists())
                self.assertFalse((root / "cohort-results/batch_run.json").exists())

    def test_missing_companion_is_reported_while_other_acquisitions_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "cohort"
            _holo, source = write_acquisition(inputs / "control")
            missing = inputs / "patient" / "missing.holo"
            missing.parent.mkdir()
            missing.touch()
            progress = []
            messages = []
            result = run_batch(
                [inputs],
                [descriptor()],
                root / "results",
                log=messages.append,
                progress=lambda *args: progress.append(args),
            )
            self.assertEqual([source], result.input_files)
            self.assertEqual(1, len(result.failures))
            self.assertIn("missing_EF.h5", result.failures[0])
            self.assertTrue((result.output_dir / "summary.json").is_file())
            self.assertIn("Analyzed 1/2", result.summary_message)
            self.assertEqual((3, 3), progress[-1][:2])
            self.assertTrue(any("[INPUT FAIL]" in message for message in messages))
            manifest = json.loads((result.output_dir / "batch_run.json").read_text())
            self.assertEqual(result.failures, manifest["failures"])

    def test_missing_single_result_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo = root / "missing.holo"
            holo.touch()
            with self.assertRaisesRegex(FileNotFoundError, "missing_EF.h5"):
                run_batch([holo], [descriptor()], root / "results")
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
