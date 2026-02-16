import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterial_waveform_shape_metrics")
class ArterialExample(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."
    v_raw_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_bandlimited_input = (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    )
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    v_bandlimited_max_input = "/Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value"
    v_bandlimited_min_input = "/Artery/VelocityPerBeat/VminPerBeatBandLimited/value"

    def run(self, h5file) -> ProcessResult:
        v_raw = np.asarray(h5file[self.v_raw_input])
        v_raw = np.maximum(v_raw, 0)
        v_bandlimited = np.asarray(h5file[self.v_bandlimited_input])
        v_bandlimited = np.maximum(v_bandlimited, 0)
        T_ds = np.asarray(h5file[self.T])
        v_bandlimited_max = np.asarray(h5file[self.v_bandlimited_max_input])
        v_bandlimited_max = np.maximum(v_bandlimited_max, 0)
        v_bandlimited_min = np.asarray(h5file[self.v_bandlimited_min_input])
        v_bandlimited_min = np.maximum(v_bandlimited_min, 0)
        tau_M1_raw = []
        tau_M1_over_T_raw = []
        tau_M1_bandlimited = []
        tau_M1_over_T_bandlimited = []

        R_VTI_bandlimited = []
        R_VTI_raw = []

        RI_bandlimited = []
        RI_raw = []

        ratio_systole_diastole_R_VTI = 0.5

        for beat_idx in range(len(T_ds[0])):
            t = T_ds[0][beat_idx] / len(v_raw.T[beat_idx])
            D1_raw = np.sum(
                v_raw.T[beat_idx][
                    : int(np.ceil(len(v_raw.T[0]) * ratio_systole_diastole_R_VTI))
                ]
            )
            D2_raw = np.sum(
                v_raw.T[beat_idx][
                    int(np.ceil(len(v_raw.T[0]) * ratio_systole_diastole_R_VTI)) :
                ]
            )
            D1_bandlimited = np.sum(
                v_bandlimited.T[beat_idx][
                    : int(
                        np.ceil(len(v_bandlimited.T[0]) * ratio_systole_diastole_R_VTI)
                    )
                ]
            )
            D2_bandlimited = np.sum(
                v_bandlimited.T[beat_idx][
                    int(
                        np.ceil(len(v_bandlimited.T[0]) * ratio_systole_diastole_R_VTI)
                    ) :
                ]
            )
            R_VTI_bandlimited.append(D1_bandlimited / (D2_bandlimited + 10 ** (-12)))
            R_VTI_raw.append(D1_raw / (D2_raw + 10 ** (-12)))
            M_0 = np.sum(v_raw.T[beat_idx])
            M_1 = 0
            for time_idx in range(len(v_raw.T[beat_idx])):
                M_1 += v_raw[time_idx][beat_idx] * time_idx * t
            TM1 = M_1 / M_0
            tau_M1_raw.append(TM1)
            tau_M1_over_T_raw.append(TM1 / T_ds[0][beat_idx])

        for beat_idx in range(len(T_ds[0])):
            t = T_ds[0][beat_idx] / len(v_raw.T[beat_idx])
            M_0 = np.sum(v_bandlimited.T[beat_idx])
            M_1 = 0
            for time_idx in range(len(v_raw.T[beat_idx])):
                M_1 += v_bandlimited[time_idx][beat_idx] * time_idx * t
            TM1 = M_1 / M_0
            tau_M1_bandlimited.append(TM1)
            tau_M1_over_T_bandlimited.append(TM1 / T_ds[0][beat_idx])

        for beat_idx in range(len(v_bandlimited_max[0])):
            RI_bandlimited_temp = 1 - (
                v_bandlimited_min[0][beat_idx] / v_bandlimited_max[0][beat_idx]
            )
            RI_bandlimited.append(RI_bandlimited_temp)

        for beat_idx in range(len(v_bandlimited_max[0])):
            RI_raw_temp = 1 - (np.min(v_raw.T[beat_idx]) / np.max(v_raw.T[beat_idx]))
            RI_raw.append(RI_raw_temp)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "tau_M1_raw": with_attrs(np.asarray(tau_M1_raw), {"unit": [""]}),
            "tau_M1_bandlimited": np.asarray(tau_M1_bandlimited),
            "tau_M1_over_T_raw": with_attrs(
                np.asarray(tau_M1_over_T_raw), {"unit": [""]}
            ),
            "tau_M1_over_T_bandlimited": np.asarray(tau_M1_over_T_bandlimited),
            "RI_bandlimited": np.asarray(RI_bandlimited),
            "RI_raw": np.asarray(RI_raw),
            "R_VTI_bandlimited": np.asarray(R_VTI_bandlimited),
            "R_VTI_raw": np.asarray(R_VTI_raw),
            "ratio_systole_diastole_R_VTI": np.asarray(ratio_systole_diastole_R_VTI),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
