import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="segment_waveform_shape_metrics")
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
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:

        v_raw = np.asarray(h5file[self.v_raw_input])
        v_raw = np.maximum(v_raw, 0)

        v_bandlimited = np.asarray(h5file[self.v_bandlimited_input])
        v_bandlimited = np.maximum(v_bandlimited, 0)

        T = np.asarray(h5file[self.T_input])

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

            t = T[0][beat_idx] / len(v_bandlimited)

            for branch_idx in range(len(v_bandlimited[0, beat_idx, :, 0])):
                Tau_M1_bandlimited_radius = []
                Tau_M1_over_T_bandlimited_radius = []
                RI_bandlimited_radius = []
                R_VTI_bandlimited_radius = []

                for radius_idx in range(len(v_bandlimited[0, beat_idx, branch_idx, :])):
                    v_bandlimited_idx = v_bandlimited[
                        :, beat_idx, branch_idx, radius_idx
                    ]

                    moment_0_segment = np.sum(v_bandlimited_idx)
                    moment_1_segment = 0

                    for time_idx in range(len(v_bandlimited_idx)):
                        moment_1_segment += v_bandlimited_idx[time_idx] * time_idx * t

                    TM1 = moment_1_segment / moment_0_segment
                    Tau_M1_bandlimited_radius.append(TM1)
                    Tau_M1_over_T_bandlimited_radius.append(TM1 / T[0][beat_idx])

                    v_bandlimited_max = np.max(v_bandlimited_idx)
                    v_bandlimited_min = np.min(v_bandlimited_idx)

                    RI_bandlimited_radius_idx = 1 - (
                        v_bandlimited_min / v_bandlimited_max
                    )
                    RI_bandlimited_radius.append(RI_bandlimited_radius_idx)

                    epsilon = 10 ** (-12)
                    D1_bandlimited = np.sum(
                        v_bandlimited_idx[
                            : int(
                                np.ceil(
                                    len(v_bandlimited_idx)
                                    * ratio_systole_diastole_R_VTI
                                )
                            )
                        ]
                    )
                    D2_bandlimited = np.sum(
                        v_bandlimited_idx[
                            int(
                                np.ceil(
                                    len(v_bandlimited_idx)
                                    * ratio_systole_diastole_R_VTI
                                )
                            ) :
                        ]
                    )
                    R_VTI_bandlimited_radius.append(
                        D1_bandlimited / (D2_bandlimited + epsilon)
                    )

                Tau_M1_bandlimited_branch.append(Tau_M1_bandlimited_radius)
                Tau_M1_over_T_bandlimited_branch.append(
                    Tau_M1_over_T_bandlimited_radius
                )
                RI_bandlimited_branch.append(RI_bandlimited_radius)
                R_VTI_bandlimited_branch.append(R_VTI_bandlimited_radius)

            Tau_M1_bandlimited_segment.append(Tau_M1_bandlimited_branch)
            Tau_M1_over_T_bandlimited_segment.append(Tau_M1_over_T_bandlimited_branch)
            RI_bandlimited_segment.append(RI_bandlimited_branch)
            R_VTI_bandlimited_segment.append(R_VTI_bandlimited_branch)

        for beat_idx in range(len(v_raw[0, :, 0, 0])):
            Tau_M1_raw_branch = []
            Tau_M1_over_T_raw_branch = []
            RI_raw_branch = []
            R_VTI_raw_branch = []

            t = T[0][beat_idx] / len(v_raw)

            for branch_idx in range(len(v_raw[0, beat_idx, :, 0])):
                Tau_M1_raw_radius = []
                Tau_M1_over_T_raw_radius = []
                RI_raw_radius = []
                R_VTI_raw_radius = []

                for radius_idx in range(len(v_raw[0, beat_idx, branch_idx, :])):
                    v_raw_idx = v_raw[:, beat_idx, branch_idx, radius_idx]

                    moment_0_segment = np.sum(v_raw_idx)
                    moment_1_segment = 0
                    for time_idx in range(len(v_raw_idx)):
                        moment_1_segment += v_raw_idx[time_idx] * time_idx * t

                    TM1 = moment_1_segment / moment_0_segment
                    Tau_M1_raw_radius.append(TM1)
                    Tau_M1_over_T_raw_radius.append(TM1 / T[0][beat_idx])

                    v_raw_max = np.max(v_raw_idx)
                    v_raw_min = np.min(v_raw_idx)

                    RI_raw_radius_idx = 1 - (v_raw_min / v_raw_max)
                    RI_raw_radius.append(RI_raw_radius_idx)

                    epsilon = 10 ** (-12)
                    D1_raw = np.sum(
                        v_raw_idx[
                            : int(
                                np.ceil(len(v_raw_idx) * ratio_systole_diastole_R_VTI)
                            )
                        ]
                    )
                    D2_raw = np.sum(
                        v_raw_idx[
                            int(
                                np.ceil(len(v_raw_idx) * ratio_systole_diastole_R_VTI)
                            ) :
                        ]
                    )
                    R_VTI_raw_radius.append(D1_raw / (D2_raw + epsilon))

                Tau_M1_raw_branch.append(Tau_M1_raw_radius)
                Tau_M1_over_T_raw_branch.append(Tau_M1_over_T_raw_radius)
                RI_raw_branch.append(RI_raw_radius)
                R_VTI_raw_branch.append(R_VTI_raw_radius)

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
