import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="modal_analysis")
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
    M0_input = "/moment0"
    M1_input = "/moment1"
    M2_input = "/moment2"
    registration_input = "/registration"

    def run(self, h5file) -> ProcessResult:
        from scipy.sparse.linalg import svds

        moment_0 = np.asarray(h5file[self.M0_input])
        moment_1 = np.asarray(h5file[self.M1_input])
        moment_2 = np.asarray(h5file[self.M2_input])

        M0_matrix = []
        M1_matrix = []
        M2_matrix = []
        M2_over_M0_squared = []
        x_size = len(moment_0[0, 0, :, 0])
        y_size = len(moment_0[0, 0, 0, :])
        for time_idx in range(len(moment_0[:, 0, 0, 0])):
            M0_matrix_time = []
            M1_matrix_time = []
            M2_matrix_time = []
            M2_over_M0_squared_time = []
            for x_idx in range(x_size):
                for y_idx in range(y_size):
                    M0 = moment_0[time_idx, 0, x_idx, y_idx]

                    M2 = moment_2[time_idx, 0, x_idx, y_idx]
                    M0_matrix_time.append(M0)
                    M2_over_M0_squared_time.append(np.sqrt(M2 / M0))

                    M1_matrix_time.append(moment_1[time_idx, 0, x_idx, y_idx])

                    M2_matrix_time.append(moment_2[time_idx, 0, x_idx, y_idx])

            M0_matrix.append(M0_matrix_time)
            M1_matrix.append(M1_matrix_time)
            M2_matrix.append(M2_matrix_time)
            M2_over_M0_squared.append(M2_over_M0_squared_time)
        M0_matrix = np.transpose(np.asarray(M0_matrix))
        M2_over_M0_squared = np.transpose(np.asarray(M2_over_M0_squared))
        n_modes = 20
        U_0, S_0, Vt_0 = svds(M0_matrix, k=n_modes)

        idx = np.argsort(S_0)[::-1]
        S_0 = S_0[idx]
        U_0 = U_0[:, idx]
        Vt_0 = Vt_0[idx, :]

        spatial_modes = []
        for mode_idx in range(len(U_0[0])):
            spatial_modes.append(U_0[:, mode_idx].reshape(x_size, y_size))

        # M2 over M0

        U_M2_over_M0_squared, S_M2_over_M0_squared, Vt_M2_over_M0_squared = svds(
            M2_over_M0_squared, k=n_modes
        )
        idx = np.argsort(S_M2_over_M0_squared)[::-1]
        S_M2_over_M0_squared = S_M2_over_M0_squared[idx]
        U_M2_over_M0_squared = U_M2_over_M0_squared[:, idx]
        Vt_M2_over_M0_squared = Vt_M2_over_M0_squared[idx, :]
        spatial_modes_M2_over_M0_squared = []

        for mode_idx in range(len(U_M2_over_M0_squared[0])):
            spatial_modes_M2_over_M0_squared.append(
                U_M2_over_M0_squared[:, mode_idx].reshape(x_size, y_size)
            )

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "Vt_moment0": with_attrs(np.asarray(Vt_0), {"unit": [""]}),
            "spatial_modes_moment0": with_attrs(
                np.asarray(spatial_modes), {"unit": [""]}
            ),
            "S_moment0": with_attrs(np.asarray(S_0), {"unit": [""]}),
            "U_moment0": with_attrs(np.asarray(U_0), {"unit": [""]}),
            "Vt_M2_over_M0_squared": with_attrs(
                np.asarray(Vt_M2_over_M0_squared), {"unit": [""]}
            ),
            "spatial_modes_M2_over_M0_squared": with_attrs(
                np.asarray(spatial_modes_M2_over_M0_squared), {"unit": [""]}
            ),
            "S_M2_over_M0_squared": with_attrs(
                np.asarray(S_M2_over_M0_squared), {"unit": [""]}
            ),
            "U_M2_over_M0_squared": with_attrs(
                np.asarray(U_M2_over_M0_squared), {"unit": [""]}
            ),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
