import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline
from .dev_v_profile_modeling.v_profile_meas_extraction import (
    extract_v_profile_meas,
)

R0 = 0.05  # Vessel radius in mm
num_interp_points = 16  # Number of spatial points for interpolation
n_harmonic = 1


@registerPipeline(name="VProfileModeling")
class VProfileModeling(ProcessPipeline):
    description = "Velocity Profile Modeling Pipeline"

    v_profile_path = "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"

    def run(self, h5file: h5py.File) -> ProcessResult:
        """
        Executes the Velocity Profile Modeling pipeline.
        """
        obj = h5file[self.v_profile_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.v_profile_path}, but found {type(obj)}"
            )
        dataset = obj[:]

        # x_coord = np.linspace(0, R0, num=num_interp_points)
        v_profile_fft, v_profile_meas = extract_v_profile_meas(
            dataset=dataset, num_interp_points=num_interp_points, n_harmonic=n_harmonic
        )

        metrics: dict = {}
        metrics["v_profile_fft"] = np.asarray(v_profile_fft)
        metrics["v_profile_meas"] = np.asarray(v_profile_meas)

        return ProcessResult(metrics=metrics)
