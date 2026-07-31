from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output.eyeflow_schema import (  # noqa: E402
    BEAT_PERIOD,
    SEGMENT_VELOCITY_PER_BEAT,
    VELOCITY_PER_BEAT,
)
from pipelines.waveform_shape_metrics import ArterialSegExample  # noqa: E402


class EyeFlowV2SchemaTests(unittest.TestCase):
    def test_waveform_pipeline_reads_v2_axes_and_empty_vessel_class(self) -> None:
        beat_count = 2
        sample_count = 16
        time = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
        artery_global = np.stack(
            [2.0 + np.sin(time), 2.0 + 0.5 * np.sin(time)],
            axis=0,
        ).astype(np.float32)
        artery_segments = np.repeat(
            artery_global.T[:, :, None, None],
            2,
            axis=3,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_EF.h5"
            with h5py.File(path, "w") as h5file:
                h5file.attrs["output_schema"] = "eyeflow_v2"
                h5file.create_dataset(BEAT_PERIOD, data=[[1.0, 1.0]])
                for representation in ("raw", "bandlimited"):
                    h5file.create_dataset(
                        VELOCITY_PER_BEAT[("artery", representation)],
                        data=artery_global,
                    )
                    h5file.create_dataset(
                        SEGMENT_VELOCITY_PER_BEAT[("artery", representation)],
                        data=artery_segments,
                    )
                    h5file.create_dataset(
                        VELOCITY_PER_BEAT[("vein", representation)],
                        data=artery_global,
                    )
                    h5file.create_dataset(
                        SEGMENT_VELOCITY_PER_BEAT[("vein", representation)],
                        data=np.empty(
                            (sample_count, beat_count, 0, 10),
                            dtype=np.float32,
                        ),
                    )

            with h5py.File(path, "r") as h5file:
                result = ArterialSegExample().run(h5file)

            self.assertIn("artery/global/raw/mu_t", result.metrics)
            self.assertEqual(
                (beat_count,),
                result.metrics["artery/global/raw/mu_t"].data.shape,
            )
            self.assertIn("artery/by_segment/raw_segment/mu_t", result.metrics)
            self.assertIn("vein/global/bandlimited/mu_t", result.metrics)


if __name__ == "__main__":
    unittest.main()
