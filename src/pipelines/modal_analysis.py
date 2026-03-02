import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, medfilt, savgol_filter

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
        max = np.argmax(moment_1[0])
        x_max = int(max / 512)
        y_max = int(max % 512)
        values = moment_2[:, 0, x_max, y_max]
        y_clean = medfilt(values, kernel_size=7)
        y_smooth = savgol_filter(y_clean, window_length=20, polyorder=3)
        dy = np.diff(y_smooth)
        peaks, properties = find_peaks(dy, distance=40, height=3000)
        n_beats = len(peaks) - 1

        M0_matrix = []
        M1_matrix = []
        M2_matrix = []
        M2_over_M0_squared = []
        spatial_modes_moment_0_per_beat = []
        spatial_modes_M2_over_M0_squared_per_beat = []
        S_M2_over_M0_squared_per_beat = []
        S_0_per_beat = []

        Vt_0_per_beat = []
        Vt_M2_over_M0_squared_per_beat = []

        x_size = len(moment_0[0, 0, :, 0])
        y_size = len(moment_0[0, 0, 0, :])

        N_target = 256
        for beat_idx in range(n_beats):
            t0 = peaks[beat_idx]
            t1 = peaks[beat_idx + 1]

            M0_beat = moment_0[t0:t1, 0].reshape(t1 - t0, -1)
            M1_beat = moment_1[t0:t1, 0].reshape(t1 - t0, -1)
            M2_beat = moment_2[t0:t1, 0].reshape(t1 - t0, -1)

            M2_over_M0_squared_beat = np.sqrt(M2_beat / M0_beat)

            M0_matrix.append(M0_beat)
            M1_matrix.append(M1_beat)
            M2_matrix.append(M2_beat)
            M2_over_M0_squared.append(M2_over_M0_squared_beat)
        N_target = 200
        resampled_M0 = []
        resampled_M2_over_M0 = []

        for beat in M0_matrix:
            Nt = beat.shape[0]

            t_old = np.linspace(0, 1, Nt)
            t_new = np.linspace(0, 1, N_target)

            f = interp1d(t_old, beat, axis=0, kind="cubic")
            resampled_M0.append(f(t_new))

        M0_matrix = np.stack(resampled_M0)
        for beat in M2_over_M0_squared:
            Nt = beat.shape[0]

            t_old = np.linspace(0, 1, Nt)
            t_new = np.linspace(0, 1, N_target)

            f = interp1d(t_old, beat, axis=0, kind="cubic")
            resampled_M2_over_M0.append(f(t_new))

        M2_over_M0_squared = np.stack(resampled_M2_over_M0)

        for beat in range(n_beats):
            M0_matrix_temp = np.transpose(np.asarray(M0_matrix[beat]))
            M2_over_M0_squared_temp = np.transpose(np.asarray(M2_over_M0_squared[beat]))
            n_modes = 20
            U_0, S_0, Vt_0 = svds(M0_matrix_temp, k=n_modes)

            idx = np.argsort(S_0)[::-1]
            S_0 = S_0[idx]
            U_0 = U_0[:, idx]
            Vt_0 = Vt_0[idx, :]

            spatial_modes_moment_0 = []
            for mode_idx in range(len(U_0[0])):
                spatial_modes_moment_0.append(U_0[:, mode_idx].reshape(x_size, y_size))

            # M2 over M0

            U_M2_over_M0_squared, S_M2_over_M0_squared, Vt_M2_over_M0_squared = svds(
                M2_over_M0_squared_temp, k=n_modes
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
            spatial_modes_moment_0_per_beat.append(spatial_modes_moment_0)
            spatial_modes_M2_over_M0_squared_per_beat.append(
                spatial_modes_M2_over_M0_squared
            )
            S_M2_over_M0_squared_per_beat.append(S_M2_over_M0_squared)
            S_0_per_beat.append(S_0)

            Vt_0_per_beat.append(Vt_0)
            Vt_M2_over_M0_squared_per_beat.append(Vt_M2_over_M0_squared)

        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "Vt_moment0_per_beat": with_attrs(
                np.asarray(Vt_0_per_beat), {"unit": [""]}
            ),
            "spatial_modes_moment0_per_beat": with_attrs(
                np.asarray(spatial_modes_moment_0_per_beat), {"unit": [""]}
            ),
            "S_moment0_per_beat": with_attrs(np.asarray(S_0_per_beat), {"unit": [""]}),
            "Vt_M2_over_M0_squared_per_beat": with_attrs(
                np.asarray(Vt_M2_over_M0_squared_per_beat), {"unit": [""]}
            ),
            "spatial_modes_M2_over_M0_squared_per_beat": with_attrs(
                np.asarray(spatial_modes_M2_over_M0_squared_per_beat), {"unit": [""]}
            ),
            "S_M2_over_M0_squared_per_beat": with_attrs(
                np.asarray(S_M2_over_M0_squared_per_beat), {"unit": [""]}
            ),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
