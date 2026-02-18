import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="reconstruct")
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
    vsystol = "/Artery/Velocity/SystolicAccelerationPeakIndexes"
    T_val = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def gaussian(x, A, mu, sigma, c):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c

    def run(self, h5file) -> ProcessResult:
        v_seg = np.asarray(h5file[self.v_profile])
        # t_ds = np.asarray(h5file[self.T_val])

        V = []
        threshold = 3

        V_corrected = []
        V_ceil = []

        for k in range(len(v_seg[0, :, 0, 0])):
            Vit_br = []
            for br in range(len(v_seg[0, k, :, 0])):
                v_branch = np.nanmean(v_seg[:, k, br, :], axis=1)
                Vit_br.append(v_branch)

            V.append(np.mean(Vit_br))
        for k in range(len(v_seg[0, :, 0, 0])):
            Vit = []
            for br in range(len(v_seg[0, k, :, 0])):
                Vit_br = []
                for seg in range(len(v_seg[0, k, br, :])):
                    values = list(v_seg[:, k, br, seg])

                    try:
                        temp = values[
                            : np.minimum(
                                values.index(next(filter(lambda x: x != 0, values)))
                                + threshold,
                                17,
                            )
                        ]
                        other = values[
                            np.minimum(
                                values.index(next(filter(lambda x: x != 0, values)))
                                + threshold,
                                17,
                            ) :
                        ]
                        test = other[
                            np.maximum(
                                other.index(next(filter(lambda x: x == 0, other)))
                                - threshold,
                                0,
                            ) :
                        ]
                        Vit_br.append(np.nanmean(temp + test))
                    except Exception:  # noqa: BLE001
                        Vit_br.append(np.nan)
                        return None

                Vit.append(np.nanmean(Vit_br))

            V_corrected.append(np.nanmean(Vit))

        """vraw_ds = np.asarray(v_threshold_beat_segment)
        vraw_ds_temp = vraw_ds.transpose(1, 0, 2, 3)
        vraw_ds = np.maximum(vraw_ds_temp, 0)
        v_ds = vraw_ds
        t_ds = np.asarray(h5file[self.T_input])

        TMI_seg = []
        TMI_seg_band = []
        RTVI_seg = []
        RTVI_seg_band = []
        RI_seg = []
        RI_seg_band = []
        M0_seg = 0
        M1_seg = 0
        M0_seg_band = 0
        M1_seg_band = 0
        for k in range(len(vraw_ds[0, :, 0, 0])):
            TMI_branch = []
            TMI_branch_band = []
            RTVI_band_branch = []
            RTVI_branch = []
            RI_branch = []
            RI_branch_band = []
            for i in range(len(vraw_ds[0, k, :, 0])):
                avg_speed_band = np.nanmean(v_ds[:, k, i, :], axis=1)
                avg_speed = np.nanmean(vraw_ds[:, k, i, :], axis=1)
                vmin = np.min(avg_speed)
                vmax = np.max(avg_speed)
                vmin_band = np.min(avg_speed_band)
                vmax_band = np.max(avg_speed_band)

                RI_branch.append(1 - (vmin / (vmax + 10 ** (-14))))
                RI_branch_band.append(1 - (vmin_band / (vmax_band + 10 ** (-14))))
                D1_raw = np.sum(avg_speed[:31])
                D2_raw = np.sum(avg_speed[32:])
                D1 = np.sum(avg_speed_band[:31])
                D2 = np.sum(avg_speed_band[32:])
                RTVI_band_branch.append(D1 / (D2 + 10 ** (-12)))
                RTVI_branch.append(D1_raw / (D2_raw + 10 ** (-12)))
                M0_seg += np.sum(avg_speed)
                M0_seg_band += np.sum(avg_speed_band)
                for j in range(len(avg_speed)):
                    M1_seg += avg_speed[j] * j * t_ds[0][k] / 64
                    M1_seg_band += avg_speed_band[j] * j * t_ds[0][k] / 64
                if M0_seg != 0:
                    TMI_branch.append(M1_seg / (t_ds[0][k] * M0_seg))
                else:
                    TMI_branch.append(0)
                if M0_seg_band != 0:
                    TMI_branch_band.append(M1_seg_band / (t_ds[0][k] * M0_seg_band))
                else:
                    TMI_branch_band.append(0)

            TMI_seg.append(TMI_branch)
            TMI_seg_band.append(TMI_branch_band)
            RI_seg.append(RI_branch)
            RI_seg_band.append(RI_branch_band)
            RTVI_seg.append(RTVI_branch)
            RTVI_seg_band.append(RTVI_band_branch)"""
        for k in range(len(v_seg[0, :, 0, 0])):
            Vit = []
            for br in range(len(v_seg[0, k, :, 0])):
                Vit_br = []
                for seg in range(len(v_seg[0, k, br, :])):
                    values = list(v_seg[:, k, br, seg])

                    try:
                        first = values.index(next(filter(lambda x: x != 0, values)))
                        other = values[
                            np.minimum(
                                values.index(next(filter(lambda x: x != 0, values)))
                                + threshold,
                                17,
                            ) :
                        ]
                        last = first + other.index(
                            next(filter(lambda x: x == 0, other))
                        )

                        Comp = [
                            values[first + threshold]
                            for v in values[first + threshold : last - threshold]
                        ]
                        Vit_br.append(
                            np.nanmean(
                                values[: first + threshold]
                                + Comp
                                + values[last - threshold :]
                            )
                        )
                    except Exception:
                        Vit_br.append(np.nan)
                        return None

                Vit.append(np.nanmean(Vit_br))

            V_ceil.append(np.nanmean(Vit))
        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.

        metrics = {
            "Xn": np.asarray(V),
            "Xn_correc": np.asarray(V_corrected),
            "Xn_ceil": np.asarray(V_ceil),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
