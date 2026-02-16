import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="old_segment_waveform_shape_metrics")
class ArterialSegExample(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."
    v_raw_input = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_bandlimited_input = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:

        v_raw = np.asarray(h5file[self.v_raw_input])
        v_raw = np.maximum(v_raw, 0)

        v_bandlimited = np.asarray(h5file[self.v_bandlimited_input])
        v_bandlimited = np.maximum(v_bandlimited, 0)

        T = np.asarray(h5file[self.T])

        moment_0_segment = 0
        moment_1_segment = 0

        Tau_M1_bandlimited_segment = []
        Tau_M1_over_T_bandlimited_segment = []
        RI_bandlimited_segment = []
        R_VTI_bandlimited_segment = []

        Tau_M1_raw_segment = []
        Tau_M1_over_T_raw_segment = []
        RI_raw_segment = []
        R_VTI_raw_segment = []

        ratio_systole_diastole_R_VTI = 0.5

        for beat_idx in range(len(v_bandlimited[0, :, 0, 0])):
            Tau_M1_bandlimited_branch = []
            Tau_M1_over_T_bandlimited_branch = []
            RI_bandlimited_branch = []
            R_VTI_bandlimited_branch = []

            for branch_idx in range(len(v_bandlimited[0, beat_idx, :, 0])):
                v_bandlimited_average = np.nanmean(
                    v_bandlimited[:, beat_idx, branch_idx, :], axis=1
                )
                t = T[0][beat_idx] / len(v_bandlimited_average)

                moment_0_segment += np.sum(v_bandlimited_average)
                for time_idx in range(len(v_bandlimited_average)):
                    moment_1_segment += v_bandlimited_average[time_idx] * time_idx * t

                if moment_0_segment != 0:
                    TM1 = moment_1_segment / moment_0_segment
                    Tau_M1_bandlimited_branch.append(TM1)
                    Tau_M1_over_T_bandlimited_branch.append(TM1 / T[0][beat_idx])
                else:
                    Tau_M1_bandlimited_branch.append(0)
                    Tau_M1_over_T_bandlimited_branch.append(0)

                v_bandlimited_max = np.max(v_bandlimited_average)
                v_bandlimited_min = np.min(v_bandlimited_average)

                RI_bandlimited_branch_idx = 1 - (v_bandlimited_min / v_bandlimited_max)
                RI_bandlimited_branch.append(RI_bandlimited_branch_idx)

                epsilon = 10 ** (-12)
                D1_bandlimited = np.sum(
                    v_bandlimited_average[
                        : int(
                            np.ceil(
                                len(v_bandlimited_average)
                                * ratio_systole_diastole_R_VTI
                            )
                        )
                    ]
                )
                D2_bandlimited = np.sum(
                    v_bandlimited_average[
                        int(
                            np.ceil(
                                len(v_bandlimited_average)
                                * ratio_systole_diastole_R_VTI
                            )
                        ) :
                    ]
                )
                R_VTI_bandlimited_branch.append(
                    D2_bandlimited / (D1_bandlimited + epsilon)
                )

            Tau_M1_bandlimited_segment.append(Tau_M1_bandlimited_branch)
            Tau_M1_over_T_bandlimited_segment.append(Tau_M1_over_T_bandlimited_branch)
            RI_bandlimited_segment.append(RI_bandlimited_branch)
            R_VTI_bandlimited_segment.append(R_VTI_bandlimited_branch)

        for beat_idx in range(len(v_raw[0, :, 0, 0])):
            Tau_M1_raw_branch = []
            Tau_M1_over_T_raw_branch = []
            RI_raw_branch = []
            R_VTI_raw_branch = []

            for branch_idx in range(len(v_raw[0, beat_idx, :, 0])):
                v_raw_average = np.nanmean(v_raw[:, beat_idx, branch_idx, :], axis=1)
                t = T[0][beat_idx] / len(v_raw_average)

                moment_0_segment += np.sum(v_raw_average)
                for time_idx in range(len(v_raw_average)):
                    moment_1_segment += v_raw_average[time_idx] * time_idx * t

                if moment_0_segment != 0:
                    TM1 = moment_1_segment / moment_0_segment
                    Tau_M1_raw_branch.append(TM1)
                    Tau_M1_over_T_raw_branch.append(TM1 / T[0][beat_idx])
                else:
                    Tau_M1_raw_branch.append(0)
                    Tau_M1_over_T_raw_branch.append(0)

                v_raw_max = np.max(v_raw_average)
                v_raw_min = np.min(v_raw_average)

                RI_raw_branch_idx = 1 - (v_raw_min / v_raw_max)
                RI_raw_branch.append(RI_raw_branch_idx)

                epsilon = 10 ** (-12)
                D1_raw = np.sum(
                    v_raw_average[
                        : int(
                            np.ceil(len(v_raw_average) * ratio_systole_diastole_R_VTI)
                        )
                    ]
                )
                D2_raw = np.sum(
                    v_raw_average[
                        int(
                            np.ceil(len(v_raw_average) * ratio_systole_diastole_R_VTI)
                        ) :
                    ]
                )
                R_VTI_raw_branch.append(D2_raw / (D1_raw + epsilon))

            Tau_M1_raw_segment.append(Tau_M1_raw_branch)
            Tau_M1_over_T_raw_segment.append(Tau_M1_over_T_raw_branch)
            RI_raw_segment.append(RI_raw_branch)
            R_VTI_raw_segment.append(R_VTI_raw_branch)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "tau_M1_bandlimited_segment": with_attrs(
                np.asarray(Tau_M1_bandlimited_segment),
                {
                    "unit": [""],
                },
            ),
            "R_VTI_bandlimited_segment": with_attrs(
                np.asarray(R_VTI_bandlimited_segment),
                {
                    "unit": [""],
                },
            ),
            "RI_bandlimited_segment": with_attrs(
                np.asarray(RI_bandlimited_segment),
                {
                    "unit": [""],
                },
            ),
            "tau_M1_over_T_bandlimited_segment": with_attrs(
                np.asarray(Tau_M1_over_T_bandlimited_segment),
                {
                    "unit": [""],
                },
            ),
            "tau_M1_raw_segment": with_attrs(
                np.asarray(Tau_M1_raw_segment),
                {
                    "unit": [""],
                },
            ),
            "R_VTI_raw_segment": with_attrs(
                np.asarray(R_VTI_raw_segment),
                {
                    "unit": [""],
                },
            ),
            "RI_raw_segment": with_attrs(
                np.asarray(RI_raw_segment),
                {
                    "unit": [""],
                },
            ),
            "tau_M1_over_T_raw_segment": with_attrs(
                np.asarray(Tau_M1_over_T_raw_segment),
                {
                    "unit": [""],
                },
            ),
            "ratio_systole_diastole_R_VTI": np.asarray(ratio_systole_diastole_R_VTI),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
