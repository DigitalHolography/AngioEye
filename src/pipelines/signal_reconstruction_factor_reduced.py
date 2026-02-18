import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="signal_reconstruction")
class Reconstruct(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."
    v_profile = "/Artery/CrossSections/VelocityProfileSeg/value"
    v_profile_interp_onebeat = (
        "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"
    )
    vsystol = "/Artery/Velocity/SystolicAccelerationPeakIndexes"
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    systole_idx_input = "/Artery/WaveformAnalysis/SystoleIndices/value"

    def gaussian(x, A, mu, sigma, c):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c

    def run(self, h5file) -> ProcessResult:
        v_seg = np.maximum(np.asarray(h5file[self.v_profile]), 0)
        v_interp_onebeat = np.maximum(
            np.asarray(h5file[self.v_profile_interp_onebeat]), 0
        )
        systole_idx = np.asarray(h5file[self.systole_idx_input])
        T = np.asarray(h5file[self.T_input])

        V = []
        v_profile_beat_threshold = []
        v_profile_beat_ceiled_threshold = []
        v_profile_beat_cropped_threshold = []

        for beat in range(len(T[0])):
            vit_beat = []
            for time_idx in range(
                int(systole_idx[0][beat]), int(systole_idx[0][beat + 1])
            ):
                Vit_br = []
                for br in range(len(v_seg[0, time_idx, :, 0])):
                    vit_seg = []
                    for segment in range(len(v_seg[0, time_idx, br, :])):
                        vit_seg.append(v_seg[:, time_idx, br, segment])
                    Vit_br.append(vit_seg)

                vit_beat.append(Vit_br)
            V.append(vit_beat)
        threshold = 6

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        """for threshold_idx in range(threshold + 1):
            v_profile_beat = []
            v_profile_beat_ceiled = []
            v_profile_beat_cropped = []
            for beat in range(len(T[0])):
                vit_beat = []
                vit_beat_ceiled = []
                vit_beat_cropped = []
                v_raw_temp = np.asarray(V[beat])
                for time_idx in range(len(v_raw_temp[:, 0, 0, 0])):
                    vit_br = []
                    vit_br_ceiled = []
                    vit_br_cropped = []
                    for br in range(len(v_raw_temp[time_idx, :, 0, 0])):
                        vit_seg = []
                        vit_seg_ceiled = []
                        vit_seg_cropped = []
                        for segment in range(len(v_raw_temp[time_idx, br, :, 0])):
                            values_temp = v_raw_temp[time_idx, br, segment, :]
                            values = list(values_temp)
                            try:
                                first = values.index(
                                    next(filter(lambda x: str(x) != "nan", values))
                                )

                                other = values[
                                    np.minimum(
                                        values.index(
                                            next(
                                                filter(
                                                    lambda x: str(x) != "nan", values
                                                )
                                            )
                                        ),
                                        17,
                                    ) :
                                ]
                                last = first + other.index(
                                    next(filter(lambda x: str(x) == "nan", other))
                                )

                                ceil_completion = [
                                    values[first + threshold_idx - 1]
                                    for v in values[
                                        first + threshold_idx : last - threshold_idx
                                    ]
                                ]
                                v_seg_band_ceiled = (
                                    values[first : first + threshold_idx]
                                    + ceil_completion
                                    + values[last - threshold_idx : last]
                                )
                                vit_seg_cropped.append(
                                    np.nanmean(
                                        values[first : first + threshold_idx]
                                        + values[last - threshold_idx : last]
                                    )
                                )
                                vit_seg_ceiled.append(np.nanmean(v_seg_band_ceiled))
                                vit_seg.append(np.nanmean(values_temp))
                            except Exception:  # noqa: BLE001
                                vit_seg_cropped.append(np.nan)
                                vit_seg_ceiled.append(np.nan)
                                vit_seg.append(np.nanmean(values_temp))
                                continue

                        vit_br.append(vit_seg)
                        vit_br_ceiled.append(vit_seg_ceiled)
                        vit_br_cropped.append(vit_seg_cropped)

                    vit_beat.append(vit_br)
                    vit_beat_ceiled.append(vit_br_ceiled)
                    vit_beat_cropped.append(vit_br_cropped)

                v_profile_beat.append(vit_beat)
                v_profile_beat_ceiled.append(vit_beat_ceiled)
                v_profile_beat_cropped.append(vit_beat_cropped)
            v_profile_beat_threshold.append(v_profile_beat)
            v_profile_beat_ceiled_threshold.append(v_profile_beat_ceiled)
            v_profile_beat_cropped_threshold.append(v_profile_beat_cropped)
        target_len = 128
        n_beats = len(v_profile_beat)
        n_branches = len(v_profile_beat[0][0])
        n_segments = len(v_profile_beat[0][0][0])
        v_threshold_beat_segment = np.zeros(
            (threshold + 1, n_beats, target_len, n_branches, n_segments)
        )
        v_threshold_beat_segment_cropped = np.zeros(
            (threshold + 1, n_beats, target_len, n_branches, n_segments)
        )
        v_threshold_beat_segment_ceiled = np.zeros(
            (threshold + 1, n_beats, target_len, n_branches, n_segments)
        )
        for threshold_idx in range(threshold + 1):
            for beat in range(n_beats):
                beat_data = np.asarray(v_profile_beat_threshold[threshold_idx][beat])
                beat_data_ceiled = np.asarray(
                    v_profile_beat_ceiled_threshold[threshold_idx][beat]
                )
                beat_data_cropped = np.asarray(
                    v_profile_beat_cropped_threshold[threshold_idx][beat]
                )
                # shape: (time_len, branches, segments)

                time_len = beat_data.shape[0]

                old_indices = np.linspace(0, 1, time_len)
                new_indices = np.linspace(0, 1, target_len)

                for br in range(n_branches):
                    for seg in range(n_segments):
                        signal = beat_data[:, br, seg]
                        signal_ceiled = beat_data_ceiled[:, br, seg]
                        signal_cropped = beat_data_cropped[:, br, seg]

                        new_values = np.interp(new_indices, old_indices, signal)
                        new_values_ceiled = np.interp(
                            new_indices, old_indices, signal_ceiled
                        )
                        new_values_cropped = np.interp(
                            new_indices, old_indices, signal_cropped
                        )

                        v_threshold_beat_segment[threshold_idx, beat, :, br, seg] = (
                            new_values
                        )
                        v_threshold_beat_segment_cropped[
                            threshold_idx, beat, :, br, seg
                        ] = new_values_ceiled
                        v_threshold_beat_segment_ceiled[
                            threshold_idx, beat, :, br, seg
                        ] = new_values_cropped"""
        v_raw_temp = np.asarray(v_interp_onebeat)
        for threshold_idx in range(threshold + 1):
            v_profile = []
            v_profile_ceiled = []
            v_profile_cropped = []

            for time_idx in range(len(v_raw_temp[:, 0, 0, 0])):
                vit_br = []
                vit_br_ceiled = []
                vit_br_cropped = []
                for br in range(len(v_raw_temp[time_idx, 0, :, 0])):
                    vit_seg = []
                    vit_seg_ceiled = []
                    vit_seg_cropped = []
                    for segment in range(len(v_raw_temp[time_idx, 0, 0, :])):
                        values_temp = v_raw_temp[time_idx, :, br, segment]
                        values = list(values_temp)
                        try:
                            first = values.index(
                                next(filter(lambda x: str(x) != "nan", values))
                            )

                            other = values[
                                np.minimum(
                                    values.index(
                                        next(filter(lambda x: str(x) != "nan", values))
                                    ),
                                    17,
                                ) :
                            ]
                            last = first + other.index(
                                next(filter(lambda x: str(x) == "nan", other))
                            )
                            len_signal = last - first
                            ceil_completion = [
                                values[first + threshold_idx - 1]
                                for v in values[
                                    int(
                                        np.minimum(
                                            first + threshold_idx,
                                            first + np.floor(len_signal / 3),
                                        )
                                    ) : int(
                                        np.maximum(
                                            last - threshold_idx,
                                            last - np.ceil(len_signal * 2 / 3),
                                        )
                                    )
                                ]
                            ]
                            v_seg_band_ceiled = (
                                values[
                                    first : int(
                                        np.minimum(
                                            first + threshold_idx,
                                            first + np.floor(len_signal / 3),
                                        )
                                    )
                                ]
                                + ceil_completion
                                + values[
                                    int(
                                        np.maximum(
                                            last - threshold_idx,
                                            last - np.ceil(len_signal * 2 / 3),
                                        )
                                    ) : last
                                ]
                            )
                            vit_seg_cropped.append(
                                np.nanmean(
                                    values[
                                        : int(
                                            np.minimum(
                                                first + threshold_idx,
                                                first + np.floor(len_signal / 3),
                                            )
                                        )
                                    ]
                                    + values[
                                        int(
                                            np.maximum(
                                                last - threshold_idx,
                                                last - np.ceil(len_signal * 2 / 3),
                                            )
                                        ) :
                                    ]
                                )
                            )
                            vit_seg_ceiled.append(np.nanmean(v_seg_band_ceiled))
                            vit_seg.append(np.nanmean(values_temp))
                        except Exception:  # noqa: BLE001
                            vit_seg_cropped.append(np.nan)
                            vit_seg_ceiled.append(np.nan)
                            vit_seg.append(np.nanmean(values_temp))
                            continue

                    vit_br.append(vit_seg)
                    vit_br_ceiled.append(vit_seg_ceiled)
                    vit_br_cropped.append(vit_seg_cropped)

                v_profile.append(vit_br)
                v_profile_ceiled.append(vit_br_ceiled)
                v_profile_cropped.append(vit_br_cropped)
            v_profile_beat_threshold.append(v_profile)
            v_profile_beat_ceiled_threshold.append(v_profile_ceiled)
            v_profile_beat_cropped_threshold.append(v_profile_cropped)
        v_raw = np.asarray(v_profile_beat_threshold)
        v_raw = np.maximum(v_raw, 0)
        v_raw_ceiled = np.asarray(v_profile_beat_ceiled_threshold)
        v_raw_ceiled = np.maximum(v_raw_ceiled, 0)
        v_raw_cropped = np.asarray(v_profile_beat_cropped_threshold)
        v_raw_cropped = np.maximum(v_raw_cropped, 0)

        moment_1_segment = 0

        moment_1_segment_cropped = 0

        moment_1_segment_ceiled = 0

        Tau_M1_raw_segment = []
        Tau_M1_over_T_raw_segment = []
        RI_raw_segment = []
        R_VTI_raw_segment = []

        Tau_M1_raw_segment_cropped = []
        Tau_M1_over_T_raw_segment_cropped = []
        RI_raw_segment_cropped = []
        R_VTI_raw_segment_cropped = []

        Tau_M1_raw_segment_ceiled = []
        Tau_M1_over_T_raw_segment_ceiled = []
        RI_raw_segment_ceiled = []
        R_VTI_raw_segment_ceiled = []

        ratio_systole_diastole_R_VTI = 0.5

        for threshold_idx in range(len(v_raw[:, 0, 0, 0])):
            Tau_M1_raw_global = []
            Tau_M1_over_T_raw_global = []
            RI_raw_global = []
            R_VTI_raw_global = []

            Tau_M1_raw_global_ceiled = []
            Tau_M1_over_T_raw_global_ceiled = []
            RI_raw_global_ceiled = []
            R_VTI_raw_global_ceiled = []

            Tau_M1_raw_global_cropped = []
            Tau_M1_over_T_raw_global_cropped = []
            RI_raw_global_cropped = []
            R_VTI_raw_global_cropped = []

            for branch_idx in range(len(v_raw[threshold_idx, 0, :, 0])):
                Tau_M1_raw_branch = []
                Tau_M1_over_T_raw_branch = []
                RI_raw_branch = []
                R_VTI_raw_branch = []

                Tau_M1_raw_branch_ceiled = []
                Tau_M1_over_T_raw_branch_ceiled = []
                RI_raw_branch_ceiled = []
                R_VTI_raw_branch_ceiled = []

                Tau_M1_raw_branch_cropped = []
                Tau_M1_over_T_raw_branch_cropped = []
                RI_raw_branch_cropped = []
                R_VTI_raw_branch_cropped = []
                for _radius_idx in range(len(v_raw[threshold_idx, 0, 0, :])):
                    v_raw_average = np.nanmean(
                        v_raw[threshold_idx, :, branch_idx, :], axis=1
                    )
                    v_raw_ceiled_average = np.nanmean(
                        v_raw_ceiled[threshold_idx, :, branch_idx, :], axis=1
                    )
                    v_raw_cropped_average = np.nanmean(
                        v_raw_cropped[threshold_idx, :, branch_idx, :], axis=1
                    )
                    t = T[0][0] / len(v_raw_average)

                    moment_0_segment = np.nansum(v_raw_average)
                    moment_0_segment_cropped = np.nansum(v_raw_cropped_average)
                    moment_0_segment_ceiled = np.nansum(v_raw_ceiled_average)
                    moment_1_segment = 0

                    moment_1_segment_cropped = 0

                    moment_1_segment_ceiled = 0
                    for time_idx in range(len(v_raw_average)):
                        moment_1_segment += v_raw_average[time_idx] * time_idx * t

                        moment_1_segment_cropped += (
                            v_raw_cropped_average[time_idx] * time_idx * t
                        )

                        moment_1_segment_ceiled += (
                            v_raw_ceiled_average[time_idx] * time_idx * t
                        )

                    if moment_0_segment != 0:
                        TM1 = moment_1_segment / moment_0_segment
                        Tau_M1_raw_branch.append(TM1)
                        Tau_M1_over_T_raw_branch.append(TM1 / T[0][0])
                    else:
                        Tau_M1_raw_branch.append(0)
                        Tau_M1_over_T_raw_branch.append(0)
                    if moment_0_segment_cropped != 0:
                        TM1_cropped = (
                            moment_1_segment_cropped / moment_0_segment_cropped
                        )
                        Tau_M1_raw_branch_cropped.append(TM1_cropped)
                        Tau_M1_over_T_raw_branch_cropped.append(TM1_cropped / T[0][0])
                    else:
                        Tau_M1_raw_branch_cropped.append(0)
                        Tau_M1_over_T_raw_branch_cropped.append(0)
                    if moment_0_segment_ceiled != 0:
                        TM1_ceiled = moment_1_segment_ceiled / moment_0_segment_ceiled
                        Tau_M1_raw_branch_ceiled.append(TM1_ceiled)
                        Tau_M1_over_T_raw_branch_ceiled.append(
                            TM1_ceiled / np.nanmean(T[0][:])
                        )
                    else:
                        Tau_M1_raw_branch_ceiled.append(0)
                        Tau_M1_over_T_raw_branch_ceiled.append(0)

                    v_raw_max = np.max(v_raw_average)
                    v_raw_min = np.min(v_raw_average)
                    v_raw_max_cropped = np.max(v_raw_cropped_average)
                    v_raw_min_cropped = np.min(v_raw_cropped_average)
                    v_raw_max_ceiled = np.max(v_raw_ceiled_average)
                    v_raw_min_ceiled = np.min(v_raw_ceiled_average)

                    RI_raw_branch_idx = 1 - (v_raw_min / v_raw_max)
                    RI_raw_branch.append(RI_raw_branch_idx)
                    RI_raw_branch_idx_cropped = 1 - (
                        v_raw_min_cropped / v_raw_max_cropped
                    )
                    RI_raw_branch_cropped.append(RI_raw_branch_idx_cropped)
                    RI_raw_branch_idx_ceiled = 1 - (v_raw_min_ceiled / v_raw_max_ceiled)
                    RI_raw_branch_ceiled.append(RI_raw_branch_idx_ceiled)

                    epsilon = 10 ** (-12)
                    D1_raw = np.sum(
                        v_raw_average[
                            : int(
                                np.ceil(
                                    len(v_raw_average) * ratio_systole_diastole_R_VTI
                                )
                            )
                        ]
                    )
                    D1_raw_cropped = np.sum(
                        v_raw_cropped_average[
                            : int(
                                np.ceil(
                                    len(v_raw_cropped_average)
                                    * ratio_systole_diastole_R_VTI
                                )
                            )
                        ]
                    )
                    D1_raw_ceiled = np.sum(
                        v_raw_ceiled_average[
                            : int(
                                np.ceil(
                                    len(v_raw_ceiled_average)
                                    * ratio_systole_diastole_R_VTI
                                )
                            )
                        ]
                    )
                    D2_raw = np.sum(
                        v_raw_average[
                            int(
                                np.ceil(
                                    len(v_raw_average) * ratio_systole_diastole_R_VTI
                                )
                            ) :
                        ]
                    )
                    D2_raw_cropped = np.sum(
                        v_raw_cropped_average[
                            int(
                                np.ceil(
                                    len(v_raw_cropped_average)
                                    * ratio_systole_diastole_R_VTI
                                )
                            ) :
                        ]
                    )
                    D2_raw_ceiled = np.sum(
                        v_raw_ceiled_average[
                            int(
                                np.ceil(
                                    len(v_raw_ceiled_average)
                                    * ratio_systole_diastole_R_VTI
                                )
                            ) :
                        ]
                    )
                    R_VTI_raw_branch.append(D1_raw / (D2_raw + epsilon))
                    R_VTI_raw_branch_cropped.append(
                        D1_raw_cropped / (D2_raw_cropped + epsilon)
                    )
                    R_VTI_raw_branch_ceiled.append(
                        D1_raw_ceiled / (D2_raw_ceiled + epsilon)
                    )

                Tau_M1_raw_global.append(Tau_M1_raw_branch)
                Tau_M1_over_T_raw_global.append(Tau_M1_over_T_raw_branch)
                RI_raw_global.append(RI_raw_branch)
                R_VTI_raw_global.append(R_VTI_raw_branch)

                Tau_M1_raw_global_cropped.append(Tau_M1_raw_branch_cropped)
                Tau_M1_over_T_raw_global_cropped.append(
                    Tau_M1_over_T_raw_branch_cropped
                )
                RI_raw_global_cropped.append(RI_raw_branch_cropped)
                R_VTI_raw_global_cropped.append(R_VTI_raw_branch_cropped)

                Tau_M1_raw_global_ceiled.append(Tau_M1_raw_branch_ceiled)
                Tau_M1_over_T_raw_global_ceiled.append(Tau_M1_over_T_raw_branch_ceiled)
                RI_raw_global_ceiled.append(RI_raw_branch_ceiled)
                R_VTI_raw_global_ceiled.append(R_VTI_raw_branch_ceiled)
            Tau_M1_raw_segment.append(Tau_M1_raw_global)
            Tau_M1_over_T_raw_segment.append(Tau_M1_over_T_raw_global)
            RI_raw_segment.append(RI_raw_global)
            R_VTI_raw_segment.append(R_VTI_raw_global)

            Tau_M1_raw_segment_cropped.append(Tau_M1_raw_global_cropped)
            Tau_M1_over_T_raw_segment_cropped.append(Tau_M1_over_T_raw_global_cropped)
            RI_raw_segment_cropped.append(RI_raw_global_cropped)
            R_VTI_raw_segment_cropped.append(R_VTI_raw_global_cropped)

            Tau_M1_raw_segment_ceiled.append(Tau_M1_raw_global_ceiled)
            Tau_M1_over_T_raw_segment_ceiled.append(Tau_M1_over_T_raw_global_ceiled)
            RI_raw_segment_ceiled.append(RI_raw_global_ceiled)
            R_VTI_raw_segment_ceiled.append(R_VTI_raw_global_ceiled)

        metrics = {
            "signals/v_profile": np.asarray(v_profile_beat_threshold),
            "signals/v_profile_cropped": np.asarray(v_profile_beat_ceiled_threshold),
            "signals/v_profile_ceiled": np.asarray(v_profile_beat_cropped_threshold),
            "tau_M1/tau_M1_raw": with_attrs(
                np.asarray(Tau_M1_raw_segment), {"unit": [""]}
            ),
            "tau_M1_over_T/tau_M1_over_T_raw": with_attrs(
                np.asarray(Tau_M1_over_T_raw_segment), {"unit": [""]}
            ),
            "RI/RI_raw": np.asarray(RI_raw_segment),
            "R_VTI/R_VTI_raw": np.asarray(R_VTI_raw_segment),
            "tau_M1/tau_M1_raw_cropped": with_attrs(
                np.asarray(Tau_M1_raw_segment_cropped), {"unit": [""]}
            ),
            "tau_M1_over_T/tau_M1_over_T_raw_cropped": with_attrs(
                np.asarray(Tau_M1_over_T_raw_segment_cropped), {"unit": [""]}
            ),
            "RI/RI_raw_cropped": np.asarray(RI_raw_segment_cropped),
            "R_VTI/R_VTI_raw_cropped": np.asarray(R_VTI_raw_segment_cropped),
            "tau_M1/tau_M1_raw_ceiled": with_attrs(
                np.asarray(Tau_M1_raw_segment_ceiled), {"unit": [""]}
            ),
            "tau_M1_over_T/tau_M1_over_T_raw_ceiled": with_attrs(
                np.asarray(Tau_M1_over_T_raw_segment_ceiled), {"unit": [""]}
            ),
            "RI/RI_raw_ceiled": np.asarray(RI_raw_segment_ceiled),
            "R_VTI/R_VTI_raw_ceiled": np.asarray(R_VTI_raw_segment_ceiled),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
