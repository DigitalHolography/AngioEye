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

R0 = 100 * 1e-6  # Vessel radius in m
num_interp_points_t = 128  # Number of temporal points for interpolation
num_interp_points_x = 16  # Number of spatial points for interpolation
model_points_x = 32
fwhm = 10 * 1e-6  # Full width at half maximum for Gaussian PSF in m
dx = 2 * R0 / model_points_x  # Spatial resolution of Womersley model in m
Cn = np.zeros(num_interp_points_t // 2 + 1, dtype=complex)
Dn = np.zeros(num_interp_points_t // 2 + 1, dtype=complex)
Cn[0] = -8000.0 - 500j
Cn[1] = -4000.0 - 250j
Cn[2] = -1000.0 - 60j
Cn[3] = -200.0 - 12j

Dn[0] = 20.0 + 5j
Dn[1] = 10.0 + 2.5j
Dn[2] = 2.0 + 0.5j
Dn[3] = 1.0 + 0.255j

Nu = 3.5 * 1e-6  # Viscosity in m^2/s


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

        x_coord = np.linspace(dx / 2, R0, num=model_points_x // 2)
        r_coord = np.linspace(0, R0, num=model_points_x // 2 + 1)

        obj = h5file[self.b_period_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.b_period_path}, but found {type(obj)}"
            )
        b_period = np.mean(obj[:])
        print(f"b_period: {b_period}")

        dataset_x, v_profile_fft, v_profile_meas, v_profile_meas_dc = (
            extract_v_profile_meas(
                dataset=dataset,
                num_interp_points_x=num_interp_points_x,
                n_harmonic=1,
            )
        )

        v_pulse_fft, v_pulse_meas, v_pulse_meas_dc = extract_v_pulse_meas(
            dataset=dataset_x,
            num_interp_points_t=num_interp_points_t,
            n_harmonic=1,
        )

        harmonics = np.arange(0, num_interp_points_t // 2 + 1)

        v_model = generate_harmonic_flow_profile(
            Cn,
            Dn,
            harmonics,
            b_period,
            R0,
            Nu,
            x_coord,
            r_coord,
            fwhm,
            dx,
            v_pulse_fft[[0], :, 3, 2],
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
