import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="segmentformshape")
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
    v_raw = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    v = "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegmentBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:
        vraw_ds = np.asarray(h5file[self.v_raw])
        v_ds = np.asarray(h5file[self.v])
        t_ds = np.asarray(h5file[self.T])

        moment0_seg = 0
        moment1_seg = 0

        TMI_seg = []
        RI_seg = []
        RVTI_seg = []
        for beat in range(len(v_ds[0, :, 0, 0])):
            TMI_branch = []
            RI_branch = []
            RVTI_branch = []
            for branch in range(len(v_ds[0, beat, :, 0])):
                speed = np.nanmean(v_ds[:, beat, branch, :], axis=1)
                moment0_seg += np.sum(speed)
                for i in range(len(speed)):
                    moment1_seg += speed[i] * i * t_ds[0][beat] / 64
                centroid_seg_branch = (moment1_seg) / (moment0_seg * t_ds[0][0])
                TMI_branch.append(centroid_seg_branch)

                speed_max = np.max(speed)
                speed_min = np.min(speed)
                RI_k = 1 - (speed_min / speed_max)
                RI_branch.append(RI_k)

                epsilon = 10 ** (-12)
                moitie = len(speed) // 2
                d1 = np.sum(speed[:moitie])
                d2 = np.sum(speed[moitie:])
                RVTI_k = d1 / (d2 + epsilon)
                RVTI_branch.append(RVTI_k)

            RI_seg.append(RI_branch)
            TMI_seg.append(TMI_branch)
            RVTI_seg.append(RVTI_branch)

        moment0raw_seg = 0
        moment1raw_seg = 0

        TMIraw_seg = []
        RIraw_seg = []
        RVTIraw_seg = []
        for beat in range(len(vraw_ds[0, :, 0, 0])):
            TMIraw_branch = []
            RIraw_branch = []
            RVTIraw_branch = []
            for branch in range(len(vraw_ds[0, beat, :, 0])):
                speed_raw = np.nanmean(vraw_ds[:, beat, branch, :], axis=1)
                moment0raw_seg += np.sum(speed_raw)
                for i in range(len(speed_raw)):
                    moment1raw_seg += speed_raw[i] * i * t_ds[0][beat] / 64
                centroidraw_seg_branch = (moment1raw_seg) / (
                    moment0raw_seg * t_ds[0][0]
                )
                TMIraw_branch.append(centroidraw_seg_branch)

                speedraw_max = np.max(speed_raw)
                speedraw_min = np.min(speed_raw)
                RIraw_k = 1 - (speedraw_min / speedraw_max)
                RIraw_branch.append(RIraw_k)

                epsilon = 10 ** (-12)
                moitie = len(speed_raw) // 2
                d1raw = np.sum(speed_raw[:moitie])
                d2raw = np.sum(speed_raw[moitie:])
                RVTIraw_k = d1raw / (d2raw + epsilon)
                RVTIraw_branch.append(RVTIraw_k)

            RIraw_seg.append(RIraw_branch)
            TMIraw_seg.append(TMIraw_branch)
            RVTIraw_seg.append(RVTIraw_branch)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "centroid": with_attrs(
                np.asarray(TMI_seg),
                {
                    "unit": [""],
                },
            ),
            "RI": with_attrs(
                np.asarray(RI_seg),
                {
                    "unit": [""],
                },
            ),
            "RTVI": with_attrs(
                np.asarray(RVTI_seg),
                {
                    "unit": [""],
                },
            ),
            "centroid raw": with_attrs(
                np.asarray(TMIraw_seg),
                {
                    "unit": [""],
                },
            ),
            "RI raw": with_attrs(
                np.asarray(RIraw_seg),
                {
                    "unit": [""],
                },
            ),
            "RTVI raw": with_attrs(
                np.asarray(RVTIraw_seg),
                {
                    "unit": [""],
                },
            ),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
