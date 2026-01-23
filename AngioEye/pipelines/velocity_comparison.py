import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult


class VelocityComparisonPipeline(ProcessPipeline):
    name = "Artery vs vein velocity"
    description = (
        "Mean of /Artery/CrossSections/velocity_whole_seg_mean and "
        "/Vein/CrossSections/velocity_trunc_seg_mean plus their ratio."
    )
    artery_path = "/Artery/CrossSections/velocity_whole_seg_mean/value"
    vein_path = "/Vein/CrossSections/velocity_whole_seg_mean/value"

    def run(self, h5file: h5py.File) -> ProcessResult:
        missing = [p for p in (self.artery_path, self.vein_path) if p not in h5file]
        if missing:
            raise ValueError(f"Missing dataset(s): {', '.join(missing)}")

        artery_ds = h5file[self.artery_path]
        vein_ds = h5file[self.vein_path]
        artery_mean = float(np.nanmean(np.asarray(artery_ds[...]).ravel()))
        vein_mean = float(np.nanmean(np.asarray(vein_ds[...]).ravel()))
        ratio = float("nan")
        if np.isfinite(vein_mean) and vein_mean != 0:
            ratio = float(artery_mean / vein_mean)
        metrics = {
            "artery_mean": artery_mean,
            "vein_mean": vein_mean,
            "artery_over_vein_ratio": ratio,
        }
        return ProcessResult(metrics=metrics)
