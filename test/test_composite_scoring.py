import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import input_output.hdf5_io as hdf5_io  # noqa: E402
import input_output.hdf5_schema as hdf5_schema  # noqa: E402
import postprocess.composite_biomarkers as composite_scoring  # noqa: E402
from postprocess.core.base import PostprocessContext  # noqa: E402

hdf5_io = importlib.reload(hdf5_io)
hdf5_schema = importlib.reload(hdf5_schema)
composite_scoring = importlib.reload(composite_scoring)
ANGIOEYE_POSTPROCESS_ROOT = hdf5_schema.ANGIOEYE_POSTPROCESS_ROOT
ANGIOEYE_PROCESSING_ROOT = hdf5_schema.ANGIOEYE_PROCESSING_ROOT
POSTPROCESS_GROUP = composite_scoring.POSTPROCESS_GROUP
CompositeScoringPostprocess = composite_scoring.CompositeScoringPostprocess


def _write_dataset(group: h5py.Group, path: str, value: float) -> None:
    parent = group
    parts = path.split("/")
    for part in parts[:-1]:
        parent = parent.require_group(part)
    parent.create_dataset(parts[-1], data=value)


def _create_waveform_metrics_file(path: Path, *, abnormal: bool) -> None:
    if abnormal:
        values = {
            "SF_VTI": 0.55,
            "t50_over_T": 0.35,
            "E_low": 0.8,
            "E_total": 1.0,
            "v_end_over_vbar": 0.58,
            "N_eff_over_T": 0.89,
            "RI": 0.8,
            "PI": 1.4,
            "W50_over_T": 0.5,
        }
    else:
        values = {
            "SF_VTI": 0.45,
            "t50_over_T": 0.4,
            "E_low": 0.5,
            "E_total": 1.0,
            "v_end_over_vbar": 0.7,
            "N_eff_over_T": 0.95,
            "RI": 0.7,
            "PI": 1.0,
            "W50_over_T": 0.7,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.require_group(
            f"{ANGIOEYE_PROCESSING_ROOT}/waveform_shape_metrics"
        )
        group.attrs["pipeline"] = "waveform_shape_metrics"
        for vessel_type in ("artery", "vein"):
            for representation in ("raw", "bandlimited"):
                for metric_name, value in values.items():
                    _write_dataset(
                        group,
                        f"{vessel_type}/global/{representation}/{metric_name}",
                        value,
                    )


@unittest.skipIf(
    importlib.util.find_spec("matplotlib") is None,
    "matplotlib is required for composite scoring PNG plots",
)
class CompositeScoringPostprocessTests(unittest.TestCase):
    def test_run_writes_scores_and_cohort_png_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "archive_outputs"
            control_file = output_dir / "CNTRL" / "control_result.h5"
            patient_file = output_dir / "PAT" / "patient_result.h5"
            _create_waveform_metrics_file(control_file, abnormal=False)
            _create_waveform_metrics_file(patient_file, abnormal=True)

            context = PostprocessContext(
                output_dir=output_dir,
                processed_files=(control_file, patient_file),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=Path(tmp_dir) / "archive.zip",
                zip_outputs=True,
            )

            result = CompositeScoringPostprocess().run(context)

            with h5py.File(patient_file, "r") as h5:
                score_group = h5[f"{ANGIOEYE_POSTPROCESS_ROOT}/{POSTPROCESS_GROUP}"]
                self.assertIn("artery/global/raw/RWAS", score_group)
                self.assertIn("artery/global/raw/RWAS4", score_group)
                self.assertIn("artery/global/bandlimited/RWAS", score_group)
                self.assertIn("artery/global/bandlimited/RWAS4", score_group)
                self.assertEqual(4, int(score_group["artery/global/raw/RWAS4"][()]))

            png_dir = output_dir / "png"
            png_paths = sorted(png_dir.glob("*.png"))
            self.assertEqual(10, len(png_paths))
            self.assertTrue(
                (
                    png_dir / "composite_scoring_raw_rwas_violin_by_cohort.png"
                ).exists()
            )
            self.assertTrue(
                (
                    png_dir / "composite_scoring_bandlimited_rwas4_violin_by_cohort.png"
                ).exists()
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in png_paths))
            self.assertIn("Generated 10 PNG plot(s)", result.summary)
            self.assertTrue(
                all(str(path) in result.generated_paths for path in png_paths)
            )

    def test_run_skips_unscorable_files_and_still_writes_png_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "archive_outputs"
            valid_file = output_dir / "CNTRL" / "control_result.h5"
            invalid_file = output_dir / "PAT" / "patient_result.h5"
            _create_waveform_metrics_file(valid_file, abnormal=False)
            invalid_file.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(invalid_file, "w"):
                pass

            context = PostprocessContext(
                output_dir=output_dir,
                processed_files=(invalid_file, valid_file),
                selected_pipelines=("waveform_shape_metrics",),
                input_path=Path(tmp_dir) / "archive.zip",
                zip_outputs=True,
            )

            result = CompositeScoringPostprocess().run(context)

            png_paths = sorted((output_dir / "png").glob("*.png"))
            self.assertEqual(10, len(png_paths))
            self.assertIn("Skipped 1 file(s)", result.summary)
            self.assertEqual(1, len(result.metadata["failures"]))
            self.assertIn(str(invalid_file), result.metadata["failures"][0])


if __name__ == "__main__":
    unittest.main()
