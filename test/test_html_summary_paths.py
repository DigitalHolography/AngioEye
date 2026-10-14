# ruff: noqa: E402

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT
from input_output.output_paths import companion_output_dir
from pipelines.core.base import PipelineContext
from pipelines.html_summary import WaveformMetricSummaryTablesPipeline
from pipelines.utils.html_summary_dashboard import (
    extract_waveform_shape_metrics,
    find_segmentation_map_png,
    generate_metric_table_html_for_file,
)


def _write_minimal_waveform_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.require_group(f"{ANGIOEYE_PROCESSING_ROOT}/waveform_shape_metrics")
        group.attrs["pipeline"] = "waveform_shape_metrics"
        for vessel_type in ("artery", "vein"):
            metrics_group = group.require_group(f"{vessel_type}/global/bandlimited")
            metrics_group.create_dataset("RI", data=[0.7])


def _write_ef_companion_pngs_to_zip(archive: zipfile.ZipFile, ef_prefix: str) -> None:
    stem = Path(ef_prefix).name.removesuffix("_EF")
    for filename in (
        f"{stem}_RI_v_artery.png",
        f"{stem}_RI_v_vein.png",
        f"{stem}_artery_seg_map_bkg.png",
        f"{stem}_vein_seg_map_bkg.png",
    ):
        archive.writestr(f"{ef_prefix}/png/{filename}", b"png")


def _write_eyeflow_hemifield_file(path: Path, *, namespaced: bool) -> None:
    root = "/EyeFlow" if namespaced else ""
    metrics_root = f"{root}/Processing/Metrics/waveform_shape_metrics"
    segmentation_root = f"{root}/Segmentation"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5file:
        for vessel in ("artery", "vein"):
            h5file.create_dataset(
                f"{metrics_root}/{vessel}/global/bandlimited/RI",
                data=[0.7],
            )
            for region, value in (("nasal", 0.6), ("temporal", 0.8)):
                h5file.create_dataset(
                    f"{metrics_root}/{vessel}/hemifield/{region}/global/bandlimited/RI",
                    data=[value],
                )
                h5file.create_dataset(
                    f"{metrics_root}/{vessel}/hemifield/"
                    f"{region}/by_branch/branch_1/bandlimited/RI",
                    data=[value + 0.01],
                )
            h5file.create_dataset(
                f"{segmentation_root}/{vessel.title()}/BranchLabelMap/value",
                data=[[1, 2], [3, 4]],
            )


class HtmlSummaryPathTests(unittest.TestCase):
    def test_html_summary_is_written_beside_acquisition_h5_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            processed_h5 = (
                output_dir / "control" / "sample" / "sample_AE" / "h5" / "sample_AE.h5"
            )
            _write_minimal_waveform_metrics_file(processed_h5)
            context = PipelineContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=Path(tmp_dir) / "batch",
                zip_outputs=False,
            )

            result = WaveformMetricSummaryTablesPipeline().run(context)

            expected = processed_h5.parent.parent / "html" / "sample_AE.html"
            self.assertEqual(
                [expected.resolve()],
                [Path(path).resolve() for path in result.generated_paths],
            )
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "html").exists())

    def test_html_summary_keeps_holo_stem_ending_in_app_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            processed_h5 = (
                output_dir / "sample_AE" / "sample_AE_AE" / "h5" / "sample_AE_AE.h5"
            )
            _write_minimal_waveform_metrics_file(processed_h5)
            context = PipelineContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=output_dir / "sample_AE.holo",
                zip_outputs=False,
            )

            result = WaveformMetricSummaryTablesPipeline().run(context)

            expected = processed_h5.parent.parent / "html" / "sample_AE_AE.html"
            self.assertEqual([str(expected)], result.generated_paths)
            self.assertTrue(expected.exists())

    def test_html_summary_reads_new_hemifield_metrics_and_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "sample_EF.h5"
            _write_eyeflow_hemifield_file(h5_path, namespaced=False)

            metrics = extract_waveform_shape_metrics(h5_path)

            self.assertIn("hemifield", metrics)
            self.assertIn("nasal", metrics["hemifield"])
            self.assertIn(
                "RI",
                metrics["hemifield"]["nasal"]["bandlimited"]["artery"],
            )
            self.assertAlmostEqual(
                0.6,
                metrics["hemifield"]["nasal"]["bandlimited"]["artery"]["RI"]["median"],
            )

            html_path = tmp_path / "html" / "sample.html"
            generate_metric_table_html_for_file(h5_path, html_path)
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Hemifield Analysis", html_text)
            self.assertIn("Nasal", html_text)
            self.assertIn('id="hemifield-select"', html_text)
            self.assertIn("showHemifield", html_text)
            self.assertIn("Artery Branch Label Map", html_text)

    def test_html_summary_reads_namespaced_eyeflow_hemifield_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "sample_AE.h5"
            _write_eyeflow_hemifield_file(h5_path, namespaced=True)

            metrics = extract_waveform_shape_metrics(h5_path)

            self.assertIn("temporal", metrics["hemifield"])
            self.assertIn(
                "RI",
                metrics["hemifield"]["temporal"]["bandlimited"]["vein"],
            )

    def test_html_summary_reads_legacy_eyeflow_metrics_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "legacy_EF.h5"
            with h5py.File(h5_path, "w") as h5file:
                h5file.create_dataset(
                    "/Metrics/waveform_shape_metrics/artery/global/bandlimited/RI",
                    data=[0.7],
                )
                h5file.create_dataset(
                    "/Topology/Artery/BranchLabelMap/value",
                    data=[[1, 2], [3, 4]],
                )

            metrics = extract_waveform_shape_metrics(h5_path)

            self.assertAlmostEqual(
                0.7,
                metrics["bandlimited"]["artery"]["RI"]["median"],
            )

    def test_companion_paths_resolve_across_app_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "sample"
            ae_h5 = root / "sample_AE" / "h5" / "sample_AE.h5"
            ae_h5.parent.mkdir(parents=True)
            ae_h5.write_text("h5", encoding="utf-8")
            artery_signal = root / "sample_EF" / "png" / "sample_RI_v_artery.png"
            vein_signal = root / "sample_EF" / "png" / "sample_RI_v_vein.png"
            artery_seg = root / "sample_EF" / "png" / "sample_artery_seg_map_bkg.png"
            vein_seg = root / "sample_EF" / "png" / "sample_vein_seg_map_bkg.png"
            artery_signal.parent.mkdir(parents=True)
            artery_signal.write_text("png", encoding="utf-8")
            vein_signal.write_text("png", encoding="utf-8")
            artery_seg.write_text("png", encoding="utf-8")
            vein_seg.write_text("png", encoding="utf-8")

            self.assertEqual(
                root / "sample_AE" / "html",
                companion_output_dir(ae_h5, app_suffix="AE", query_type="html"),
            )
            self.assertEqual(
                artery_seg,
                find_segmentation_map_png(
                    ae_h5,
                    vessel_type="artery",
                ),
            )
            self.assertEqual(
                vein_seg,
                find_segmentation_map_png(
                    ae_h5,
                    vessel_type="vein",
                ),
            )

    def test_zip_html_summary_drops_h5_folders_from_archive_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            processed_h5 = output_dir / "h5" / "h5" / "CNTRL" / "sample_AE.h5"
            _write_minimal_waveform_metrics_file(processed_h5)
            zip_path = Path(tmp_dir) / "batch.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("h5/CNTRL/sample.h5", "source")

            context = PipelineContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=zip_path,
                zip_outputs=True,
                input_h5_paths=(),
            )

            result = WaveformMetricSummaryTablesPipeline().run(context)

            expected = output_dir / "html" / "CNTRL" / "sample_AE.html"
            self.assertEqual(
                [expected.resolve()],
                [Path(path).resolve() for path in result.generated_paths],
            )
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "html" / "h5").exists())

    def test_zip_html_summary_does_not_read_unstaged_legacy_companions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            processed_h5 = (
                output_dir / "h5" / "sample_EF" / "h5" / "sample_pipelines_result.h5"
            )
            _write_minimal_waveform_metrics_file(processed_h5)
            zip_path = tmp_path / "old_ef.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("sample_EF/h5/sample.h5", "source")
                _write_ef_companion_pngs_to_zip(archive, "sample_EF")

            context = PipelineContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=zip_path,
                zip_outputs=True,
                input_h5_paths=(
                    tmp_path / "batch_00001" / "sample_EF" / "h5" / "sample.h5",
                ),
            )

            result = WaveformMetricSummaryTablesPipeline().run(context)

            expected = (
                output_dir / "html" / "sample_EF" / "sample_pipelines_result.html"
            )
            html_text = expected.read_text(encoding="utf-8")
            self.assertEqual(
                [expected.resolve()],
                [Path(path).resolve() for path in result.generated_paths],
            )
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "html" / "sample_EF" / "h5").exists())
            self.assertNotIn("data:image/png;base64,cG5n", html_text)

    def test_zip_html_summary_does_not_read_unstaged_companions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            processed_h5 = (
                output_dir
                / "h5"
                / "sample"
                / "sample_EF"
                / "h5"
                / "sample_pipelines_result.h5"
            )
            _write_minimal_waveform_metrics_file(processed_h5)
            zip_path = tmp_path / "new_ef.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("sample/sample_EF/h5/sample.h5", "source")
                _write_ef_companion_pngs_to_zip(archive, "sample/sample_EF")

            context = PipelineContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=zip_path,
                zip_outputs=True,
                input_h5_paths=(
                    tmp_path
                    / "batch_00001"
                    / "sample"
                    / "sample_EF"
                    / "h5"
                    / "sample.h5",
                ),
            )

            result = WaveformMetricSummaryTablesPipeline().run(context)

            expected = (
                output_dir
                / "html"
                / "sample"
                / "sample_EF"
                / "sample_pipelines_result.html"
            )
            html_text = expected.read_text(encoding="utf-8")
            self.assertEqual(
                [expected.resolve()],
                [Path(path).resolve() for path in result.generated_paths],
            )
            self.assertTrue(expected.exists())
            self.assertFalse(
                (output_dir / "html" / "sample" / "sample_EF" / "h5").exists()
            )
            self.assertNotIn("data:image/png;base64,cG5n", html_text)
