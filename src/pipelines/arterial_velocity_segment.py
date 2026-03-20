import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterialformshapesegment")
class ArterialExampleSegment(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."
    v_raw = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    v = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:
        vraw_ds_temp = np.asarray(h5file[self.v_raw])
        vraw_ds = np.maximum(vraw_ds_temp, 0)
        v_ds_temp = np.asarray(h5file[self.v])
        v_ds = np.maximum(v_ds_temp, 0)
        t_ds = np.asarray(h5file[self.T])

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
            RTVI_seg_band.append(RTVI_band_branch)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "TMI_raw_seg": with_attrs(np.asarray(TMI_seg), {"unit": [""]}),
            "TMI_seg": np.asarray(TMI_seg_band),
            "RI": np.asarray(RI_seg_band),
            "RI_raw": np.asarray(RI_seg),
            "RTVI": np.asarray(RTVI_seg_band),
            "RTVI_raw": np.asarray(RTVI_seg),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
