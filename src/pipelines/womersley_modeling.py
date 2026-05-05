import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline
from .dev_womersley_modeling.v_pulse_meas_extraction import (
    extract_v_pulse_meas,
)

R0 = 0.05  # Vessel radius in mm
num_interp_points = 128  # Number of temporal points for interpolation
n_harmonic = 1


@registerPipeline(name="WomersleyModeling")
class WomersleyModeling(ProcessPipeline):
    description = "Womersley Modeling Pipeline"

    v_profile_path = "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"

    def run(self, h5file: h5py.File) -> ProcessResult:
        """
        Executes the Womersley Modeling pipeline.
        """
        obj = h5file[self.v_profile_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.v_profile_path}, but found {type(obj)}"
            )
        dataset = obj[:]

        v_pulse_fft, v_pulse_meas, v_pulse_meas_dc = extract_v_pulse_meas(
            dataset=dataset, num_interp_points=num_interp_points, n_harmonic=n_harmonic
        )

        metrics: dict = {}
        metrics["v_pulse_fft"] = np.asarray(v_pulse_fft)
        metrics["v_pulse_meas"] = np.asarray(v_pulse_meas)
        metrics["v_pulse_meas_dc"] = np.asarray(v_pulse_meas_dc)

        return ProcessResult(metrics=metrics)
