# ruff: noqa: E402

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

from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT
from input_output.output_paths import companion_output_dir, h5_output_parent
from postprocess.core.base import PostprocessContext
from postprocess.html_summary import WaveformMetricSummaryTablesPostprocess
from postprocess.utils.html_summary_dashboard import (
    extract_waveform_shape_metrics,
    find_segmentation_map_png,
    find_velocity_signal_png,
    generate_metric_table_html_for_file,
)
from workflows import HoloInputContext, ZipBatchSettings, find_ae_h5, run_holo_workflow


def _write_minimal_waveform_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.require_group(
            f"{ANGIOEYE_PROCESSING_ROOT}/waveform_shape_metrics"
        )
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
                    f"{metrics_root}/{vessel}/hemifield/"
                    f"{region}/global/bandlimited/RI",
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
                metrics["hemifield"]["nasal"]["bandlimited"]["artery"][
                    "RI"
                ]["median"],
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
                    "/Metrics/waveform_shape_metrics/artery/"
                    "global/bandlimited/RI",
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
                artery_signal,
                find_velocity_signal_png(
                    ae_h5,
                    vessel_type="artery",
                ),
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
            self.assertEqual(
                vein_signal,
                find_velocity_signal_png(
                    ae_h5,
                    vessel_type="vein",
                ),
            )

    def test_velocity_signal_resolver_falls_back_to_legacy_png_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            h5_path = root / "sample.h5"
            h5_path.write_text("h5", encoding="utf-8")
            artery_signal = root / "png" / "sample_RI_v_artery.png"
            artery_signal.parent.mkdir()
            artery_signal.write_text("png", encoding="utf-8")

            self.assertEqual(
                artery_signal,
                find_velocity_signal_png(
                    h5_path,
                    vessel_type="artery",
                    stem="sample",
                ),
            )

    def test_html_summary_writes_to_ae_html_for_holo_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "sample"
            holo_path = Path(tmp_dir) / "sample.holo"
            holo_path.write_text("holo", encoding="utf-8")
            ef_h5 = root / "sample_EF" / "h5" / "sample.h5"
            ef_h5.parent.mkdir(parents=True)
            ef_h5.write_text("h5", encoding="utf-8")
            ae_dir = root / "sample_AE"
            processed_h5 = ae_dir / "h5" / "sample_AE.h5"
            processed_h5.parent.mkdir(parents=True)
            processed_h5.write_text("h5", encoding="utf-8")

            def _write_report(_processed_h5, html_path, *, source_path=None):
                del source_path
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_text("html", encoding="utf-8")
                return html_path

            context = PostprocessContext(
                output_dir=ae_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=holo_path,
                zip_outputs=False,
                input_h5_paths=(ef_h5,),
            )

            with mock.patch(
                "postprocess.utils.html_summary_dashboard."
                "generate_metric_table_html_for_file",
                side_effect=_write_report,
            ):
                result = WaveformMetricSummaryTablesPostprocess().run(context)

            expected = root / "sample_AE" / "html" / "sample_AE.html"
            self.assertEqual([str(expected)], result.generated_paths)
            self.assertTrue(expected.exists())
            self.assertFalse((ae_dir / "html" / "h5").exists())

    def test_zip_html_summary_drops_h5_folders_from_archive_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            processed_h5 = output_dir / "h5" / "h5" / "CNTRL" / "sample_AE.h5"
            _write_minimal_waveform_metrics_file(processed_h5)
            zip_path = Path(tmp_dir) / "batch.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("h5/CNTRL/sample.h5", "source")

            context = PostprocessContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=zip_path,
                zip_outputs=True,
                input_h5_paths=(),
            )

            result = WaveformMetricSummaryTablesPostprocess().run(context)

            expected = output_dir / "html" / "CNTRL" / "sample_AE.html"
            self.assertEqual([str(expected)], result.generated_paths)
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "html" / "h5").exists())

    def test_zip_html_summary_uses_old_eyeflow_png_companions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            processed_h5 = (
                output_dir
                / "h5"
                / "sample_EF"
                / "h5"
                / "sample_pipelines_result.h5"
            )
            _write_minimal_waveform_metrics_file(processed_h5)
            zip_path = tmp_path / "old_ef.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("sample_EF/h5/sample.h5", "source")
                _write_ef_companion_pngs_to_zip(archive, "sample_EF")

            context = PostprocessContext(
                output_dir=output_dir,
                processed_files=(processed_h5,),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=zip_path,
                zip_outputs=True,
                input_h5_paths=(
                    tmp_path / "batch_00001" / "sample_EF" / "h5" / "sample.h5",
                ),
            )

            result = WaveformMetricSummaryTablesPostprocess().run(context)

            expected = (
                output_dir
                / "html"
                / "sample_EF"
                / "sample_pipelines_result.html"
            )
            html_text = expected.read_text(encoding="utf-8")
            self.assertEqual([str(expected)], result.generated_paths)
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "html" / "sample_EF" / "h5").exists())
            self.assertIn("data:image/png;base64,cG5n", html_text)

    def test_zip_html_summary_uses_new_eyeflow_png_companions(self) -> None:
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

            context = PostprocessContext(
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

            result = WaveformMetricSummaryTablesPostprocess().run(context)

            expected = (
                output_dir
                / "html"
                / "sample"
                / "sample_EF"
                / "sample_pipelines_result.html"
            )
            html_text = expected.read_text(encoding="utf-8")
            self.assertEqual([str(expected)], result.generated_paths)
            self.assertTrue(expected.exists())
            self.assertFalse(
                (output_dir / "html" / "sample" / "sample_EF" / "h5").exists()
            )
            self.assertIn("data:image/png;base64,cG5n", html_text)

    def test_holo_workflow_writes_h5_under_ae_h5_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "sample"
            holo_path = Path(tmp_dir) / "sample.holo"
            holo_path.write_text("holo", encoding="utf-8")
            ef_dir = root / "sample_EF"
            ef_h5 = ef_dir / "h5" / "sample.h5"
            ef_h5.parent.mkdir(parents=True)
            ef_h5.write_text("h5", encoding="utf-8")
            ae_dir = root / "sample_AE"
            context = HoloInputContext(
                holo_path=holo_path,
                ef_dir=ef_dir,
                h5_path=ef_h5,
                output_dir=ae_dir,
            )

            def _run_pipeline_file(
                h5_path,
                _pipelines,
                output_root,
                output_relative_parent=Path("."),
                output_filename=None,
            ):
                target_dir = h5_output_parent(output_root, output_relative_parent)
                target_dir.mkdir(parents=True, exist_ok=True)
                output_path = target_dir / (output_filename or f"{h5_path.stem}.h5")
                output_path.write_text("result", encoding="utf-8")
                return output_path

            result = run_holo_workflow(
                contexts=[context],
                pipelines=["waveform_shape_metrics"],
                postprocesses=[],
                selected_pipeline_names=("waveform_shape_metrics",),
                run_pipeline_file=_run_pipeline_file,
                run_postprocesses=lambda *args, **kwargs: None,
                log=lambda _message: None,
                advance_progress=lambda _units: None,
                start_final_progress=lambda _units, _status: None,
                settings=ZipBatchSettings(batch_size=1),
            )

            expected = ae_dir / "h5" / "sample_AE.h5"
            self.assertEqual([expected], result.processed_outputs)
            self.assertEqual(expected, find_ae_h5(holo_path))
            self.assertFalse((ae_dir / "h5" / "h5").exists())


if __name__ == "__main__":
    unittest.main()
