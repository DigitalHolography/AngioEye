import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


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
    v_profile = "/Artery/Velocity/VelocityProfiles/value"
    vsystol = "/Artery/Velocity/SystolicAccelerationPeakIndexes"
    T_val = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    vmax = "/Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value"
    vmin = "/Artery/VelocityPerBeat/VminPerBeatBandLimited/value"

    def gaussian(x, A, mu, sigma, c):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c

    def run(self, h5file) -> ProcessResult:
        v_seg = np.asarray(h5file[self.v_profile])
        t_ds = np.asarray(h5file[self.T_val])

        V = []
        threshold = 3

        V_corrected = []
        V_ceil = []
        V_gauss = []

        for k in range(len(v_seg[0, :, 0, 0])):
            VIT_Time = 0
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
                    except:
                        Vit_br.append(np.nan)

                Vit.append(np.nanmean(Vit_br))

            V_corrected.append(np.nanmean(Vit))
        for k in range(len(v_seg[0, :, 0, 0])):
            Vit = []
            Vit_gauss = []
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
                    except:
                        Vit_br.append(np.nan)

                Vit.append(np.nanmean(Vit_br))

            V_ceil.append(np.nanmean(Vit))
        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.

        metrics = {
            "Xn": with_attrs(
                np.asarray(V),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "Xn_correc": with_attrs(
                np.asarray(V_corrected),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "Xn_ceil": with_attrs(
                np.asarray(V_ceil),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
