# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output.output_paths import companion_output_dir
from postprocess.core.base import PostprocessContext
from postprocess.html_summary import WaveformMetricSummaryTablesPostprocess
from postprocess.utils.html_summary_dashboard import (
    find_segmentation_map_png,
    find_velocity_signal_png,
)
from workflows import HoloInputContext, ZipBatchSettings, find_ae_h5, run_holo_workflow


class HtmlSummaryPathTests(unittest.TestCase):
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
                target_dir = output_root / output_relative_parent
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


if __name__ == "__main__":
    unittest.main()
