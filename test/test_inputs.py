import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import (  # noqa: E402
    find_hdf5_inputs,
    prepare_run_input,
    prepare_run_inputs,
    resolve_eyeflow_h5,
    validate_input_paths,
)


def _acquisition(folder: Path, name: str, *, processed: bool = True):
    folder.mkdir(parents=True, exist_ok=True)
    holo = folder / name
    holo.write_bytes(b"HOLO contents are not read by input discovery")
    result = folder / holo.stem / f"{holo.stem}_EF" / "h5" / f"{holo.stem}_EF.h5"
    if processed:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"Input discovery leaves HDF5 validation to processing")
    return holo.resolve(), result.resolve()


class AcquisitionInputTests(unittest.TestCase):
    def test_single_holo_uses_exact_eyeflow_layout_and_full_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            holo, result = _acquisition(Path(temp), "scan.v2.HOLO")

            plan = prepare_run_input(holo)

            self.assertEqual("file", plan.kind)
            self.assertEqual(holo, plan.input_path)
            self.assertEqual((holo,), plan.holo_paths)
            self.assertEqual((result,), plan.h5_paths)
            self.assertEqual(1, plan.item_count)
            self.assertEqual((), plan.input_failures)
            self.assertEqual("scan.v2_EF.h5", result.name)

    def test_folder_discovers_acquisitions_recursively_and_preserves_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            last, last_result = _acquisition(root / "pathology", "Z.holo")
            first, first_result = _acquisition(root / "control", "a.Holo")
            unrelated = root / "stray.h5"
            unrelated.write_bytes(b"not an acquisition companion")

            plan = prepare_run_input(root)

            self.assertEqual("folder", plan.kind)
            self.assertEqual(root.resolve(), plan.input_path)
            self.assertEqual((first, last), plan.holo_paths)
            self.assertEqual((first_result, last_result), plan.h5_paths)
            self.assertEqual(2, plan.item_count)

    def test_multiple_files_use_common_acquisition_parent_and_selection_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, first_result = _acquisition(root / "control", "first.holo")
            second, second_result = _acquisition(root / "pathology", "second.holo")

            plan = prepare_run_inputs((second, first))

            self.assertEqual(root.resolve(), plan.input_path)
            self.assertEqual((second, first), plan.holo_paths)
            self.assertEqual((second_result, first_result), plan.h5_paths)

    def test_validation_normalizes_relative_paths_and_removes_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            holo, result = _acquisition(Path(temp), "sample.holo")
            relative = Path(os.path.relpath(holo, Path.cwd()))

            self.assertEqual((holo,), validate_input_paths((relative, holo)))
            self.assertEqual((result,), prepare_run_inputs((relative, holo)).h5_paths)

    def test_direct_hdf5_zip_and_other_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("sample.h5", "sample.hdf5", "data.zip", "list.txt"):
                with self.subTest(name=name):
                    path = Path(temp) / name
                    path.touch()
                    with self.assertRaisesRegex(ValueError, r"\.holo"):
                        prepare_run_input(path)

    def test_empty_selection_and_blank_path_are_rejected(self):
        for selection in ((), ("",), ("  ",)):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, r"Select a \.holo"):
                    validate_input_paths(selection)

    def test_missing_selected_holo_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.holo"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                prepare_run_input(missing)

    def test_multiple_folders_and_folder_file_mixtures_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, _result = _acquisition(root / "first", "sample.holo")
            _acquisition(root / "second", "other.holo")
            for selections in (
                (root / "first", root / "second"),
                (root / "first", holo),
            ):
                with self.subTest(selections=selections):
                    with self.assertRaisesRegex(ValueError, "cannot be combined"):
                        validate_input_paths(selections)

    def test_folder_without_holo_is_rejected_even_if_it_contains_hdf5(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sample_EF.h5").touch()
            with self.assertRaisesRegex(ValueError, r"No \.holo files"):
                prepare_run_input(root)

    def test_discovery_ignores_macos_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid, result = _acquisition(root, "valid.holo")
            _acquisition(root / "__MACOSX", "metadata.holo", processed=False)
            _acquisition(root, "._sidecar.holo", processed=False)

            plan = prepare_run_input(root)

            self.assertEqual((valid,), plan.holo_paths)
            self.assertEqual((result,), plan.h5_paths)
            self.assertEqual((), plan.input_failures)

    def test_folder_with_only_sidecars_is_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _acquisition(root / "__MACOSX", "metadata.holo", processed=False)
            _acquisition(root, "._sidecar.holo", processed=False)
            with self.assertRaisesRegex(ValueError, r"No \.holo files"):
                validate_input_paths((root,))

    def test_single_missing_result_explains_expected_path_and_eyeflow_step(self):
        with tempfile.TemporaryDirectory() as temp:
            holo, expected = _acquisition(Path(temp), "sample.holo", processed=False)
            with self.assertRaises(FileNotFoundError) as caught:
                prepare_run_input(holo)
            self.assertIn(str(expected), str(caught.exception))
            self.assertIn("Run EyeFlow", str(caught.exception))

    def test_other_companions_and_renamed_results_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo, expected = _acquisition(root, "sample.holo", processed=False)
            expected.parent.mkdir(parents=True)
            (expected.parent / "renamed_EF.h5").touch()
            hd = root / "sample" / "sample_HD" / "h5"
            hd.mkdir(parents=True)
            (hd / "sample_HD_output.h5").touch()
            with self.assertRaises(FileNotFoundError):
                resolve_eyeflow_h5(holo)

    def test_partial_folder_retains_valid_inputs_and_reports_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good, result = _acquisition(root, "good.holo")
            missing, expected = _acquisition(root, "missing.holo", processed=False)

            plan = prepare_run_input(root)

            self.assertEqual((good,), plan.holo_paths)
            self.assertEqual((result,), plan.h5_paths)
            self.assertEqual(2, plan.item_count)
            self.assertEqual(1, len(plan.input_failures))
            self.assertIn(missing.name, plan.input_failures[0])
            self.assertIn(str(expected), plan.input_failures[0])

    def test_partial_multiple_selection_retains_matching_holo_result_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing, _expected = _acquisition(root, "missing.holo", processed=False)
            good, result = _acquisition(root, "good.holo")

            plan = prepare_run_inputs((missing, good))

            self.assertEqual(root.resolve(), plan.input_path)
            self.assertEqual((good,), plan.holo_paths)
            self.assertEqual((result,), plan.h5_paths)
            self.assertEqual(2, plan.item_count)
            self.assertEqual(1, len(plan.input_failures))

    def test_no_processed_results_raises_summary_for_every_acquisition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _result = _acquisition(root, "first.holo", processed=False)
            second, _result = _acquisition(root, "second.holo", processed=False)
            for selections in ((root,), (first, second)):
                with self.subTest(selections=selections):
                    with self.assertRaises(FileNotFoundError) as caught:
                        prepare_run_inputs(selections)
                    self.assertIn("No processed EyeFlow results", str(caught.exception))
                    self.assertIn(first.name, str(caught.exception))
                    self.assertIn(second.name, str(caught.exception))

    def test_cross_drive_selection_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _result = _acquisition(root, "first.holo")
            second, _result = _acquisition(root, "second.holo")
            with mock.patch(
                "input_output.inputs.os.path.commonpath", side_effect=ValueError
            ):
                with self.assertRaisesRegex(ValueError, "same drive"):
                    prepare_run_inputs((first, second))

    def test_low_level_hdf5_discovery_remains_available_for_pipeline_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "results" / "sample_AE.h5"
            nested.parent.mkdir()
            nested.touch()
            (root / "sample.holo").touch()
            self.assertEqual([nested], find_hdf5_inputs(root))
            self.assertEqual([nested], find_hdf5_inputs(nested))


if __name__ == "__main__":
    unittest.main()
