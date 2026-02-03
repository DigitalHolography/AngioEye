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

    def run(self, h5file) -> ProcessResult:
        vraw_ds = np.asarray(h5file[self.v_raw])
        v_ds = np.asarray(h5file[self.v])
        t_ds = np.asarray(h5file[self.T])

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {"vraw": vraw_ds, "vds": v_ds, "tds": t_ds}

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
