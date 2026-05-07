import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline
from .dev_womersley_modeling.dev_forward_modeling import (
    generate_harmonic_flow_profile,
)
from .dev_womersley_modeling.v_profile_meas_extraction import (
    extract_v_profile_meas,
)
from .dev_womersley_modeling.v_pulse_meas_extraction import (
    extract_v_pulse_meas,
)

R0 = 0.00004  # Vessel radius in m
num_interp_points_t = 128  # Number of temporal points for interpolation
num_interp_points_x = 16  # Number of spatial points for interpolation
n_harmonic = 1
Cn = 1.0 + 0.2j
Dn = 0.1 + 0.05j
Nu = 0.0000035  # Viscosity in m^2/s
psf_kernel = None


@registerPipeline(name="WomersleyModeling")
class WomersleyModeling(ProcessPipeline):
    description = "Womersley Modeling Pipeline"

    v_profile_path = "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"
    b_period_path = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

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

        dx = R0 / num_interp_points_x
        x_coord = np.linspace(dx / 2, R0, num=num_interp_points_x // 2)
        r_coord = np.linspace(0, R0, num=num_interp_points_x // 2 + 1)

        obj = h5file[self.b_period_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.b_period_path}, but found {type(obj)}"
            )
        b_period = np.mean(obj[:])

        dataset_x, v_profile_fft, v_profile_meas, v_profile_meas_dc = (
            extract_v_profile_meas(
                dataset=dataset,
                num_interp_points_x=num_interp_points_x,
                n_harmonic=n_harmonic,
            )
        )

        v_pulse_fft, v_pulse_meas, v_pulse_meas_dc = extract_v_pulse_meas(
            dataset=dataset_x,
            num_interp_points_t=num_interp_points_t,
            n_harmonic=n_harmonic,
        )

        v_model = generate_harmonic_flow_profile(
            Cn, Dn, n_harmonic, b_period, R0, Nu, x_coord, r_coord, psf_kernel
        )

        metrics: dict = {}
        metrics["dataset_x"] = np.asarray(dataset_x)
        metrics["v_profile_fft"] = np.asarray(v_profile_fft)
        metrics["v_profile_meas"] = np.asarray(v_profile_meas)
        metrics["v_profile_meas_dc"] = np.asarray(v_profile_meas_dc)
        metrics["v_pulse_fft"] = np.asarray(v_pulse_fft)
        metrics["v_pulse_meas"] = np.asarray(v_pulse_meas)
        metrics["v_pulse_meas_dc"] = np.asarray(v_pulse_meas_dc)
        metrics["v_model"] = np.asarray(v_model)

        return ProcessResult(metrics=metrics)
