# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines import PipelineContext  # noqa: E402
from pipelines.composite_scoring.run import run_composite_scoring  # noqa: E402
from pipelines.patient_evaluation.run import (
    run_fixed_threshold_evaluation,  # noqa: E402
)


class PipelineFailureTests(unittest.TestCase):
    def test_scoring_missing_cohorts_reports_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sample.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset(
                    "/Processing/Metrics/waveform_shape_metrics/artery/global/bandlimited/RI",
                    data=[0.7],
                )
            context = PipelineContext(root, (path,), (), root, False)
            with self.assertRaisesRegex(ValueError, "calibration failed"):
                run_composite_scoring(context)

    def test_evaluation_without_any_compatible_case_reports_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sample.h5"
            with h5py.File(path, "w"):
                pass
            context = PipelineContext(root, (path,), (), root, False)
            with self.assertRaisesRegex(ValueError, "No patients could be evaluated"):
                run_fixed_threshold_evaluation(context)
