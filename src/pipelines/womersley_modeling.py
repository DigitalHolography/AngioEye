import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline
from .dev_womersley_modeling.v_meas_extraction import (
    preprocess_and_interpolate,
)

R0 = 0.05  # Placeholder for vessel radius in mm
num_interp_points = 16  # Number of spatial points for interpolation
n_harmonic = 1  # Placeholder for number of beats to process


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

        # Expected shape: (n_t, n_x, n_branches, n_radii) -> (128, 33, 14, 10)
        n_t, n_x, n_branches, n_radii = dataset.shape
        # x_coords = np.linspace(0, R0, num=num_interp_points)
        v_profile_fft = np.zeros(
            (n_t, num_interp_points, n_branches, n_radii), dtype=complex
        )
        v_profile_meas = np.zeros(
            (n_t, num_interp_points, n_branches, n_radii), dtype=complex
        )

        for branch_idx in range(n_branches):
            for radii_idx in range(n_radii):
                for t_idx in range(n_t):
                    v_segment_onebeat = np.asarray(
                        dataset[t_idx, :, branch_idx, radii_idx]
                    )

                    v_interp = preprocess_and_interpolate(
                        num_interp_points=num_interp_points,
                        v_segment_onebeat=v_segment_onebeat,
                    )

                    v_fft = np.fft.fft(np.asarray(v_interp), n=num_interp_points)
                    v_profile_fft[t_idx, :, branch_idx, radii_idx] = v_fft

                    v_meas = np.zeros_like(v_fft)
                    v_meas[1] = v_fft[n_harmonic]
                    v_meas[-1] = v_fft[-n_harmonic]
                    v_profile_meas[t_idx, :, branch_idx, radii_idx] = np.fft.ifft(
                        v_meas
                    ).real

        metrics: dict = {}
        metrics["v_profile_fft"] = np.asarray(v_profile_fft)
        metrics["v_profile_meas"] = np.asarray(v_profile_meas)

        return ProcessResult(metrics=metrics)
