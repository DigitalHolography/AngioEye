from __future__ import annotations

import hashlib
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

from input_output import (  # noqa: E402
    list_h5_members,
    prepare_run_inputs,
    stage_analysis_file,
)
from input_output.archive_io import (  # noqa: E402
    create_zip_from_tree,
    replace_folder_in_zip,
)
from input_output.eyeflow_schema import BEAT_PERIOD, require_dataset  # noqa: E402
from input_output.hdf5_schema import (  # noqa: E402
    ANGIOEYE_STATISTICS_ROOT,
    find_pipeline_group,
)


def write_eyeflow_fixture(path: Path) -> None:
    with h5py.File(path, "w") as output:
        output.attrs["eyeflow_version"] = "1.18.0"
        output.attrs["source_file"] = "upstream_HD.h5"
        for branch_name, method, value in (
            ("Processing", "doppler_moments", 1.0),
            ("ProcessingAlt", "frequency_bands", 9.0),
        ):
            branch = output.create_group(branch_name)
            branch.attrs["velocity_estimation_method"] = method
            branch.attrs["output_schema"] = "eyeflow_v2"
            metrics = branch.require_group("Metrics/waveform_shape_metrics")
            metric = metrics.create_dataset(
                "artery/global/bandlimited/RI", data=[value, value + 1.0]
            )
            metric.attrs["definition"] = "resistivity index"
            metrics.create_dataset(
                "artery/by_segment/bandlimited_segment/RI",
                data=np.full((2, 3, 4), value),
            )
            branch.create_dataset(
                "CardiacCycle/Systole/CycleDurationSeconds/value", data=[1.0, 1.2]
            )
            vessel = branch.require_group("VelocityPerBeat/Artery")
            vessel.attrs["vessel_type"] = "artery"
            signal = vessel.create_dataset("Raw/value", data=np.full((2, 8), value))
            signal.attrs["dimDesc"] = ["beat", "sample"]
            vessel.create_dataset("BandLimited/value", data=np.full((2, 8), value))
            vessel.create_dataset("Segments/Raw/value", data=np.ones((8, 2, 3, 4)))
            branch.create_dataset("Maps/FRMSAverage/value", data=np.ones((16, 16)))
        output.create_dataset("Meta/FrameIntervalSeconds/value", data=0.01)
        output.create_dataset("Segmentation/OpticDisc/Center/value", data=[4.0, 8.0])
        output.create_dataset("Segmentation/Artery/Mask/value", data=np.ones((16, 16)))
        output.create_dataset("Videos/full", data=np.ones((8, 16, 16)))


class EyeFlowAnalysisSchemaTests(unittest.TestCase):
    def test_staging_keeps_metrics_axes_attrs_and_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample_EF.h5"
            target = Path(tmp) / "results" / "h5" / "sample_AE.h5"
            write_eyeflow_fixture(source)
            original_digest = hashlib.sha256(source.read_bytes()).digest()

            self.assertEqual(stage_analysis_file(source, target), target)

            self.assertEqual(
                hashlib.sha256(source.read_bytes()).digest(), original_digest
            )
            with h5py.File(target, "r+") as staged:
                group = find_pipeline_group(staged, "waveform_shape_metrics")
                self.assertIsNotNone(group)
                np.testing.assert_array_equal(
                    group["artery/global/bandlimited/RI"], [1.0, 2.0]
                )
                self.assertEqual(
                    group["artery/global/bandlimited/RI"].attrs["definition"],
                    "resistivity index",
                )
                self.assertEqual(
                    group["artery/by_segment/bandlimited_segment/RI"].shape, (2, 3, 4)
                )
                np.testing.assert_array_equal(
                    require_dataset(staged, BEAT_PERIOD), [1.0, 1.2]
                )
                self.assertEqual(staged.attrs["source_file"], "upstream_HD.h5")
                self.assertEqual(
                    staged.attrs["angioeye_input_file"], str(source.resolve())
                )
                self.assertEqual(
                    staged.attrs["angioeye_processing_branch"], "Processing"
                )
                self.assertEqual(
                    staged["Processing/VelocityPerBeat/Artery"].attrs["vessel_type"],
                    "artery",
                )
                self.assertIn("Segmentation/OpticDisc/Center/value", staged)
                for omitted in (
                    "ProcessingAlt",
                    "Processing/Maps",
                    "Videos",
                    "Processing/VelocityPerBeat/Artery/Segments",
                    "Segmentation/Artery/Mask",
                ):
                    self.assertNotIn(omitted, staged)
                staged.require_group(ANGIOEYE_STATISTICS_ROOT)

    def test_selected_alt_branch_is_mapped_without_primary_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample_EF.h5"
            target = Path(tmp) / "sample_AE.h5"
            write_eyeflow_fixture(source)
            stage_analysis_file(source, target, processing_branch="ProcessingAlt")
            with h5py.File(target) as staged:
                self.assertEqual(
                    staged["Processing"].attrs["velocity_estimation_method"],
                    "frequency_bands",
                )
                np.testing.assert_array_equal(
                    staged[
                        "Processing/Metrics/waveform_shape_metrics/artery/global/bandlimited/RI"
                    ],
                    [9.0, 10.0],
                )
                np.testing.assert_array_equal(
                    staged["Processing/VelocityPerBeat/Artery/Raw/value"],
                    np.full((2, 8), 9.0),
                )
                self.assertNotIn("ProcessingAlt", staged)

    def test_invalid_or_missing_branch_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample_EF.h5"
            target = Path(tmp) / "existing.h5"
            write_eyeflow_fixture(source)
            target.write_bytes(b"existing report")
            with self.assertRaisesRegex(
                ValueError, "Unknown EyeFlow processing branch"
            ):
                stage_analysis_file(source, target, processing_branch="unknown")
            with h5py.File(source, "r+") as original:
                del original["ProcessingAlt"]
            with self.assertRaisesRegex(ValueError, "ProcessingAlt/Metrics"):
                stage_analysis_file(source, target, processing_branch="ProcessingAlt")
            self.assertEqual(target.read_bytes(), b"existing report")

    def test_output_cannot_overwrite_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample_EF.h5"
            write_eyeflow_fixture(source)
            with self.assertRaisesRegex(ValueError, "must differ"):
                stage_analysis_file(source, source)

    def test_multiple_holograms_resolve_eyeflow_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            holos = [Path(tmp) / f"{name}.holo" for name in ("first", "second")]
            exports = []
            for holo in holos:
                holo.touch()
                source = (
                    holo.parent
                    / holo.stem
                    / f"{holo.stem}_EF"
                    / "h5"
                    / f"{holo.stem}_EF.h5"
                )
                source.parent.mkdir(parents=True)
                write_eyeflow_fixture(source)
                exports.append(source)
            plan = prepare_run_inputs(holos)
            self.assertEqual(plan.h5_paths, tuple(exports))
            with self.assertRaises(ValueError):
                prepare_run_inputs((holos[0], exports[1]))

    def test_zip_ignores_metadata_and_rejects_absolute_windows_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "batch.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("control/sample_EF.h5", b"fixture")
                archive.writestr("control/._sample_EF.h5", b"metadata")
                archive.writestr("__MACOSX/control/sample_EF.h5", b"metadata")
            self.assertEqual(
                [member.relative_path for member in list_h5_members(archive_path)],
                [Path("control/sample_EF.h5")],
            )
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("C:/outside_EF.h5", b"fixture")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member path"):
                list_h5_members(archive_path)


class ArchiveOutputTests(unittest.TestCase):
    def test_repeated_archive_creation_excludes_destination_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "summary.csv"
            source.write_text("subject,value\na,1\n", encoding="utf-8")
            destination = root / "outputs.zip"
            for _ in range(2):
                create_zip_from_tree(root, destination)
                with zipfile.ZipFile(destination) as archive:
                    self.assertEqual(archive.namelist(), ["summary.csv"])
                    self.assertEqual(archive.read("summary.csv"), source.read_bytes())
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_explicit_source_cannot_be_overwritten_by_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample_EF.h5"
            write_eyeflow_fixture(source)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "must differ from its source"):
                create_zip_from_tree(root, source, source_paths=[source])
            self.assertEqual(source.read_bytes(), before)

    def test_failed_archive_creation_keeps_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "summary.csv"
            source.write_text("subject,value\na,1\n", encoding="utf-8")
            destination = root / "outputs.zip"
            create_zip_from_tree(root, destination)
            before = destination.read_bytes()

            def interrupted_progress(_index, _total, _path):
                raise OSError("interrupted archive export")

            with self.assertRaisesRegex(OSError, "interrupted archive export"):
                create_zip_from_tree(
                    root, destination, progress_callback=interrupted_progress
                )
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_folder_replacement_preserves_other_members_and_excludes_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "figure.txt"
            source.write_text("new figure", encoding="utf-8")
            destination = root / "outputs.zip"
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("reports/old.txt", "old figure")
                archive.writestr("control/sample_EF.h5", "unchanged input")
            replace_folder_in_zip(destination, root, arc_folder="reports")
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(
                    archive.namelist(), ["control/sample_EF.h5", "reports/figure.txt"]
                )
                self.assertEqual(archive.read("reports/figure.txt"), b"new figure")
                self.assertEqual(
                    archive.read("control/sample_EF.h5"), b"unchanged input"
                )
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
