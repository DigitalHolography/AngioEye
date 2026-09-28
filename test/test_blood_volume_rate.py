from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import list_h5_members  # noqa: E402
from pipelines import load_pipeline_catalog  # noqa: E402
from pipelines.blood_volume_rate import (  # noqa: E402
    OUTPUT_FOLDER_NAME,
    build_lumen_diameter_figure,
    collect_lumen_diameters,
)
from workflows import (  # noqa: E402
    WorkflowInputError,
    WorkflowInputSelection,
    WorkflowOutputOptions,
    WorkflowRequestState,
    WorkflowWorkSelection,
    ZipBatchSettings,
    build_workflow_request,
    run_zip_workflow,
    zip_output_dir,
)


def _write_eyeflow_h5(
    path: Path,
    values: list[float],
    *,
    pixel_pitch_m: float = 2e-6,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset(
            "/Segmentation/Artery/LumenDiameter/value",
            data=np.asarray(values, dtype=float),
        )
        h5file.create_dataset(
            "/Segmentation/PixelPitch_m/value",
            data=pixel_pitch_m,
        )


def _archive_tree(tree_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(tree_root.rglob("*.h5")):
            archive.write(file_path, file_path.relative_to(tree_root).as_posix())


class BloodVolumeRateTests(unittest.TestCase):
    def test_pipeline_rejects_non_zip_workflow_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_h5 = root / "input.h5"
            _write_eyeflow_h5(input_h5, [1.0, 2.0])
            available, _missing = load_pipeline_catalog()
            descriptor = next(
                pipeline
                for pipeline in available
                if pipeline.name == "blood_volume_rate"
            )
            state = WorkflowRequestState(
                input_selection=WorkflowInputSelection(
                    convention="legacy",
                    data_value=str(input_h5),
                ),
                work_selection=WorkflowWorkSelection(
                    pipeline_names=("blood_volume_rate",),
                    pipelines=(descriptor,),
                    postprocesses=(),
                ),
                output_options=WorkflowOutputOptions(
                    base_output_value=str(root / "outputs"),
                    zip_outputs=True,
                    zip_name="results.zip",
                ),
            )

            with self.assertRaisesRegex(
                WorkflowInputError,
                "accepts only zip input; received file",
            ):
                build_workflow_request(state, zip_output_dir=zip_output_dir)

    def test_collects_eye_subfolders_and_control_aliases_by_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_tree = root / "input"
            archive_root = input_tree / "260923_IndentationGOA"
            _write_eyeflow_h5(
                archive_root / "Baseline" / "left_eye" / "left.h5",
                [1.0, 2.0, np.nan],
            )
            _write_eyeflow_h5(
                archive_root / "control" / "right-eye" / "right.h5",
                [3.0, 4.0],
            )
            _write_eyeflow_h5(
                archive_root / "indentation" / "left_eye" / "left.h5",
                [10.0, 12.0],
            )
            _write_eyeflow_h5(
                archive_root / "indentation" / "right_eye" / "right.h5",
                [20.0, 22.0],
            )
            zip_path = root / "cohorts.zip"
            _archive_tree(input_tree, zip_path)

            collection = collect_lumen_diameters(zip_path)

            self.assertEqual(
                {
                    "baseline_left_eye",
                    "baseline_right_eye",
                    "indentation_left_eye",
                    "indentation_right_eye",
                },
                set(collection.values_by_cohort),
            )
            np.testing.assert_array_equal(
                np.array([2.0, 4.0]),
                np.sort(collection.values_by_cohort["baseline_left_eye"]),
            )
            np.testing.assert_array_equal(
                np.array([6.0, 8.0]),
                np.sort(collection.values_by_cohort["baseline_right_eye"]),
            )
            self.assertEqual(4, collection.input_file_count)
            self.assertEqual(4, collection.loaded_file_count)
            self.assertEqual((), collection.skipped_files)

    def test_figure_has_requested_style_and_statistics(self) -> None:
        figure, median, standard_deviation = build_lumen_diameter_figure(
            np.array([1.0, 2.0, 3.0, 4.0])
        )
        self.addCleanup(figure.clear)
        axis = figure.axes[0]

        self.assertEqual("", axis.get_title())
        self.assertEqual("Lumen diameter (µm)", axis.get_xlabel())
        self.assertEqual("Density", axis.get_ylabel())
        self.assertEqual("--", axis.lines[0].get_linestyle())
        self.assertEqual("black", axis.lines[0].get_color())
        self.assertIsNone(axis.get_legend())
        self.assertTrue(axis.patches)
        np.testing.assert_allclose(
            axis.patches[0].get_facecolor()[:3],
            (128 / 255, 128 / 255, 128 / 255),
        )
        self.assertIn("Median:", axis.texts[0].get_text())
        self.assertIn("Std:", axis.texts[0].get_text())
        self.assertEqual(2.5, median)
        self.assertAlmostEqual(float(np.std([1.0, 2.0, 3.0, 4.0])), standard_deviation)

    def test_archive_pipeline_generates_named_pngs_in_final_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_tree = root / "input"
            archive_root = input_tree / "260923_IndentationGOA"
            _write_eyeflow_h5(
                archive_root / "baseline" / "left_eye" / "one.h5",
                [2.0, 3.0, 4.0],
            )
            _write_eyeflow_h5(
                archive_root / "baseline" / "right_eye" / "two.h5",
                [5.0, 6.0, 8.0],
            )
            _write_eyeflow_h5(
                archive_root / "indentation" / "left_eye" / "three.h5",
                [9.0, 10.0, 11.0],
            )
            _write_eyeflow_h5(
                archive_root / "indentation" / "right_eye" / "four.h5",
                [12.0, 13.0, 14.0],
            )
            input_zip = root / "cohorts.zip"
            _archive_tree(input_tree, input_zip)
            available, _missing = load_pipeline_catalog()
            descriptor = next(
                pipeline
                for pipeline in available
                if pipeline.name == "blood_volume_rate"
            )
            self.assertEqual(
                "BloodVolumeRatePipeline",
                descriptor.pipeline_cls.__name__,
            )
            self.assertEqual("archive", descriptor.execution_scope)
            self.assertEqual(("zip",), descriptor.accepted_input_modes)

            output_dir = root / "outputs"
            result = run_zip_workflow(
                zip_path=input_zip,
                members=list_h5_members(input_zip),
                member_count=4,
                pipelines=[descriptor],
                postprocesses=[],
                selected_pipeline_names=["blood_volume_rate"],
                base_output_dir=output_dir,
                zip_outputs=True,
                zip_name="result.zip",
                settings=ZipBatchSettings(batch_size=2, process_workers=1),
                run_pipeline_file=lambda *_args, **_kwargs: self.fail(
                    "archive pipeline must not use the per-file runner"
                ),
                run_postprocesses=lambda *_args, **_kwargs: None,
                zip_output_dir=zip_output_dir,
                log=lambda _message: None,
                advance_progress=lambda _units: None,
                start_final_progress=lambda _units, _status: None,
                set_status=lambda _status: None,
                make_zip_progress_callback=lambda: None,
            )

            expected_names = {
                "baseline_left_eye_lumen_diameter_distribution.png",
                "baseline_right_eye_lumen_diameter_distribution.png",
                "indentation_left_eye_lumen_diameter_distribution.png",
                "indentation_right_eye_lumen_diameter_distribution.png",
            }
            self.assertEqual(4, len(result.generated_outputs))
            self.assertEqual(
                expected_names,
                {path.name for path in result.generated_outputs},
            )
            self.assertEqual(output_dir / "result.zip", result.zip_path)
            with zipfile.ZipFile(result.zip_path) as archive:
                archived_names = set(archive.namelist())
            self.assertTrue(
                {
                    f"result/{OUTPUT_FOLDER_NAME}/{name}" for name in expected_names
                }.issubset(archived_names)
            )


if __name__ == "__main__":
    unittest.main()
