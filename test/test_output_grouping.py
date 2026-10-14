from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.core.base import PipelineContext  # noqa: E402
from pipelines.core.grouped_batch import (  # noqa: E402
    extract_group_name,
    iter_grouped_h5_files,
    iter_grouped_h5_files_in_zip,
)
from pipelines.lowrank_waveform_cohort import (  # noqa: E402
    LowRankWaveformCohortPipeline,
    _cohort_root_from_paths,
    classify_cohort,
    find_cohort_input_h5s,
    validate_cohort_layout,
)
from pipelines.utils.groups_comparison_html import analyze_zip  # noqa: E402


def write_metrics_h5(path: Path, *, value: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.require_group("Processing/Metrics/lowrank_waveform_decomposition")
        handle.create_dataset(
            "Processing/Metrics/waveform_shape_metrics/artery/global/raw/RI",
            data=[value, value],
        )
    return path


class AcquisitionOutputGroupingTests(unittest.TestCase):
    def test_acquisition_wrappers_do_not_count_toward_group_depth(self) -> None:
        batch_root = Path("batch")
        acquisition_h5 = batch_root / "control" / "left" / "sample" / "sample_AE" / "h5"
        self.assertEqual("control", extract_group_name(acquisition_h5, batch_root))
        self.assertEqual(
            "control/left",
            extract_group_name(acquisition_h5, batch_root, group_depth=2),
        )
        self.assertEqual(
            "control/left",
            extract_group_name(acquisition_h5, batch_root, group_depth=4),
        )

    def test_ungrouped_acquisition_stays_ungrouped_at_each_output_root(self) -> None:
        batch_root = Path("batch")
        acquisition = batch_root / "sample"
        analysis = acquisition / "sample_AE"
        for selected_root in (batch_root, acquisition, analysis, analysis / "h5"):
            with self.subTest(selected_root=selected_root):
                self.assertEqual(
                    "all", extract_group_name(analysis / "h5", selected_root)
                )

    def test_legacy_layout_and_unrelated_folder_names_keep_their_groups(self) -> None:
        batch_root = Path("batch")
        self.assertEqual(
            "control", extract_group_name(batch_root / "h5" / "control", batch_root)
        )
        self.assertEqual(
            "control/left",
            extract_group_name(
                batch_root / "h5" / "control" / "left", batch_root, group_depth=2
            ),
        )
        self.assertEqual(
            "sample/other_AE",
            extract_group_name(
                batch_root / "sample" / "other_AE" / "h5",
                batch_root,
                group_depth=2,
            ),
        )

    def test_directory_and_zip_grouping_agree_for_new_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir) / "batch"
            paths = [
                write_metrics_h5(batch_root / relative)
                for relative in (
                    "control/left/a/a_AE/h5/a_AE.h5",
                    "patient/right/b/b_AE/h5/b_AE.h5",
                    "c/c_AE/h5/c_AE.h5",
                )
            ]
            archive_path = Path(tmp_dir) / "batch.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in paths:
                    archive.write(path, path.relative_to(batch_root).as_posix())
            directory_records = [
                (record.group_name, record.file_name)
                for record in iter_grouped_h5_files(batch_root, group_depth=2)
            ]
            archive_records = [
                (record.group_name, record.file_name)
                for record in iter_grouped_h5_files_in_zip(archive_path, group_depth=2)
            ]
            self.assertEqual(
                [
                    ("all", "c_AE.h5"),
                    ("control/left", "a_AE.h5"),
                    ("patient/right", "b_AE.h5"),
                ],
                directory_records,
            )
            self.assertEqual(directory_records, archive_records)

    def test_cohort_grouping_and_root_inference_keep_numbered_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            paths = [
                write_metrics_h5(batch_root / relative)
                for relative in (
                    "1_ctrl/a/a_AE/h5/a_AE.h5",
                    "2_path/b/b_AE/h5/b_AE.h5",
                )
            ]
            records, group_order = classify_cohort(paths, batch_root)
            self.assertEqual(["1_ctrl", "2_path"], group_order)
            self.assertEqual(["1_ctrl", "2_path"], [group for group, _ in records])
            self.assertEqual(
                ("1_ctrl", "2_path"),
                validate_cohort_layout(paths, batch_root).group_order,
            )
            self.assertEqual(batch_root, _cohort_root_from_paths([paths[0]]))

    def test_cohort_classifier_does_not_make_an_acquisition_a_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            path = write_metrics_h5(batch_root / "sample/sample_AE/h5/sample_AE.h5")
            records, group_order = classify_cohort([path], batch_root)
            self.assertEqual([(None, path)], records)
            self.assertEqual([], group_order)

    def test_cohort_root_is_stable_when_previous_reports_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            cohort_root = batch_root / "260101_Patient"
            path = write_metrics_h5(
                cohort_root / "1_ctrl/sample/sample_AE/h5/sample_AE.h5"
            )
            first_root = _cohort_root_from_paths([path], batch_root)
            self.assertEqual(cohort_root, first_root)
            write_metrics_h5(batch_root / "cohort-results/h5/lowrank_cohort.h5")
            (batch_root / "unrelated").mkdir()
            rerun_root = _cohort_root_from_paths([path], batch_root)
            self.assertEqual(first_root, rerun_root)
            self.assertEqual(
                ("1_ctrl",), validate_cohort_layout([path], rerun_root).group_order
            )

    def test_single_numbered_group_and_legacy_flat_root_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            grouped = write_metrics_h5(
                batch_root
                / "1_ctrl/260101_sample/260101_sample_AE/h5/260101_sample_AE.h5"
            )
            (batch_root / "cohort-results").mkdir()
            self.assertEqual(batch_root, _cohort_root_from_paths([grouped], batch_root))
            flat = write_metrics_h5(batch_root / "legacy_EF.h5")
            self.assertEqual(batch_root, _cohort_root_from_paths([flat], batch_root))

    def test_cohort_pipeline_uses_the_shared_report_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            path = write_metrics_h5(
                batch_root / "1_ctrl/sample/sample_AE/h5/sample_AE.h5"
            )
            report_dir = batch_root / "cohort-results"
            report_dir.mkdir()
            context = PipelineContext(
                output_dir=report_dir,
                processed_files=(path,),
                selected_pipelines=(),
                input_path=batch_root,
                zip_outputs=False,
                analysis_root=batch_root,
            )
            with patch(
                "pipelines.lowrank_waveform_cohort._run_on_paths",
                return_value=("written", []),
            ) as run_on_paths:
                result = LowRankWaveformCohortPipeline().run(context)
            self.assertEqual(batch_root, run_on_paths.call_args.args[1])
            self.assertEqual(report_dir, run_on_paths.call_args.args[2])
            self.assertEqual(str(report_dir), result.metadata["output_dir"])

    def test_legacy_cohort_context_keeps_its_report_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            path = write_metrics_h5(batch_root / "h5/1_ctrl/sample_AE.h5")
            context = PipelineContext(
                output_dir=batch_root,
                processed_files=(path,),
                selected_pipelines=(),
                input_path=batch_root,
                zip_outputs=False,
            )
            with patch(
                "pipelines.lowrank_waveform_cohort._run_on_paths",
                return_value=("written", []),
            ) as run_on_paths:
                LowRankWaveformCohortPipeline().run(context)
            self.assertEqual(batch_root / "h5", run_on_paths.call_args.args[1])
            self.assertEqual(
                batch_root / "cohort-results", run_on_paths.call_args.args[2]
            )

    def test_source_discovery_skips_generated_acquisition_h5s(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir)
            source = write_metrics_h5(
                batch_root / "1_ctrl/sample/sample_EF/h5/sample_EF.h5"
            )
            write_metrics_h5(batch_root / "1_ctrl/sample/sample_AE/h5/sample_AE.h5")
            write_metrics_h5(batch_root / "cohort-results/h5/lowrank_cohort.h5")
            self.assertEqual([source], find_cohort_input_h5s(batch_root))

    def test_html_analysis_retains_groups_in_modern_staged_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_root = Path(tmp_dir) / "batch"
            control = write_metrics_h5(
                batch_root / "control/a/a_AE/h5/a_AE.h5", value=1.0
            )
            patient = write_metrics_h5(
                batch_root / "patient/b/b_AE/h5/b_AE.h5", value=2.0
            )
            archive_path = Path(tmp_dir) / "batch.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in (control, patient):
                    archive.write(path, path.relative_to(batch_root).as_posix())
            rows = analyze_zip(archive_path)["raw"]["artery"]["RI"]
            self.assertEqual(
                {("control", 1.0), ("patient", 2.0)},
                {(row["group"], row["median"]) for row in rows},
            )


if __name__ == "__main__":
    unittest.main()
