import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="arterialformshape")
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
    v_raw = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    vmax = "/Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value"
    vmin = "/Artery/VelocityPerBeat/VminPerBeatBandLimited/value"

    def run(self, h5file) -> ProcessResult:
        vraw_ds_temp = np.asarray(h5file[self.v_raw])
        vraw_ds = np.maximum(vraw_ds_temp, 0)
        v_ds_temp = np.asarray(h5file[self.v])
        v_ds = np.maximum(v_ds_temp, 0)
        t_ds = np.asarray(h5file[self.T])
        V_max = np.asarray(h5file[self.vmax])
        V_min = np.asarray(h5file[self.vmin])
        TMI_raw = []
        RTVI = []
        RTVI_raw = []
        for k in range(len(t_ds[0])):
            D1_raw = np.sum(vraw_ds.T[k][:31])
            D2_raw = np.sum(vraw_ds.T[k][32:])
            D1 = np.sum(v_ds.T[k][:31])
            D2 = np.sum(v_ds.T[k][32:])
            RTVI.append(D1 / (D2 + 10 ** (-12)))
            RTVI_raw.append(D1_raw / (D2_raw + 10 ** (-12)))
            M_0 = np.sum(vraw_ds.T[k])
            M_1 = 0
            for i in range(len(vraw_ds.T[k])):
                M_1 += vraw_ds[i][k] * i * t_ds[0][k] / 64
            TM1 = M_1 / (t_ds[0][k] * M_0)
            TMI_raw.append(TM1)
        TMI = []
        for k in range(len(t_ds[0])):
            M_0 = np.sum(v_ds.T[k])
            M_1 = 0
            for i in range(len(vraw_ds.T[k])):
                M_1 += v_ds[i][k] * i * t_ds[0][k] / 64
            TM1 = M_1 / (t_ds[0][k] * M_0)
            TMI.append(TM1)
        RI = []
        for i in range(len(V_max[0])):
            RI_temp = 1 - (V_min[0][i] / V_max[0][i])
            RI.append(RI_temp)
        RI_raw = []
        for i in range(len(V_max[0])):
            RI_temp = 1 - (np.min(vraw_ds.T[i]) / np.max(vraw_ds.T[i]))
            RI_raw.append(RI_temp)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "TMI_raw": np.asarray(TMI_raw),
            "TMI": np.asarray(TMI),
            "RI": np.asarray(RI),
            "RI_raw": np.asarray(RI_raw),
            "RTVI": np.asarray(RTVI),
            "RTVI_raw": np.asarray(RTVI_raw),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
