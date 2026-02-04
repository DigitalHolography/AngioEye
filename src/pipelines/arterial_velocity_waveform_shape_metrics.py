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
        
        v_ds = np.asarray(h5file[self.v])
        v_ds_max=np.maximum(v_ds,0)
        t_ds = np.asarray(h5file[self.T])
        
        centroid=[]
        RI=[]
        RVTI=[]
        for k in range (len(v_ds_max[0])):
            moment0=np.sum(v_ds_max.T[k])
            moment1=0
            for i in range (len(v_ds_max.T[k])):
                moment1+=v_ds_max[i][k]*i*t_ds[0][k]/64

            centroid_k=(moment1)/(moment0*t_ds[0][0])
            centroid.append(centroid_k)
        
            v_max=np.max(v_ds_max[k])
            v_min=np.min(v_ds_max[k])
            RI_k=1-(v_min/v_max)
            RI.append(RI_k)

            epsilon=10**(-12)
            moitie=len(v_ds_max.T[k])//2
            d1=np.sum(v_ds_max.T[k][:moitie])  
            d2=np.sum(v_ds_max.T[k][moitie:]) 
            RVTI_k=d1/(d2+epsilon)
            RVTI.append(RVTI_k)
            
        
        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        
        metrics = {"centroid": with_attrs(np.asarray(centroid),{"unit": [""],},),
                "RI": with_attrs(np.asarray(RI),{"unit": [""],},),
                "RTVI": with_attrs(np.asarray(RVTI),{"unit": [""],},),
                }
        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
