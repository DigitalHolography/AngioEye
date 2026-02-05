import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="waveform")
class WaveForm(ProcessPipeline):
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
    T_val = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"
    vmax = "/Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value"
    vmin = "/Artery/VelocityPerBeat/VminPerBeatBandLimited/value"

    def run(self, h5file) -> ProcessResult:
        vraw_ds_temp = np.asarray(h5file[self.v_raw])
        vraw_ds = np.maximum(vraw_ds_temp, 0)
        v_ds_temp = np.asarray(h5file[self.v])
        v_ds = np.maximum(v_ds_temp, 0)
        t_ds = np.asarray(h5file[self.T_val])
        V_max = np.asarray(h5file[self.vmax])
        V_min = np.asarray(h5file[self.vmin])
        N = len(vraw_ds[:, 0])
        # période normalisée (1 beat)
        # pulsation fondamentale

        V_coeff = []
        Xn = []
        H2 = []
        Delta_Phi_2 = []
        Delta_Phi_3 = []
        R10 = []
        HRI_2_10 = []
        HRI_2_10_noisecorrec = []
        S1_10 = []
        nc = []
        Sigma_n = []
        Hspec = []
        Fspec = []
        tau_phi = []
        tau_phi_n = []
        tau_G = []
        BV = []
        tau_M1 = []
        sigma_M = []
        TAU_H = []
        TAU_P_N = []
        TAU_P = []
        for i in range(len(vraw_ds[0])):
            T = t_ds[0][i]
            omega0 = 2 * np.pi / T
            t = np.linspace(0, T, N, endpoint=False)
            Vfft = np.fft.fft(vraw_ds[:, i]) / N
            Vn = Vfft[:11]  # harmonics
            V_coeff.append(Vn)
            Xn.append(Vn[1:] / Vn[1])
            H2.append(np.abs(Xn[i][1]))
            phi = np.angle(Xn[i])
            absV = np.abs(Vn)
            Delta_Phi_2.append(np.angle(Vn[2] * np.conj(Vn[1]) ** 2))
            Delta_Phi_3.append(np.angle(Vn[3] * np.conj(Vn[1]) ** 3))
            R10.append(np.abs(Vn[1]) / np.real(Vn[0]))
            HRI_2_10.append(np.sum(np.abs(Xn[i][2:])))
            magnitudes = absV[1:11]
            phi2 = np.angle(Xn[i][1:])
            absX2 = np.abs(Xn[i][1:])
            p = magnitudes / np.sum(magnitudes)
            n = np.arange(1, 11)
            n2 = np.arange(2, len(Xn[i]) + 1)
            valid = magnitudes > 0

            if np.sum(valid) < 3:
                slope = np.nan

            else:
                log_n = np.log(n[valid])
                log_mag = np.log(magnitudes[valid])

                slope, _ = np.polyfit(log_n, log_mag, 1)
            S1_10.append(slope)
            HRI_2_10_noisecorrec.append([])
            nc_i = np.sum(n * p)
            nc.append(nc_i)
            Sigma_n.append(np.sqrt(np.sum((n - nc_i) ** 2 * p)))
            Hspec.append(-np.sum(p * np.log(p + 1e-12)))
            Fspec.append(np.exp(np.mean(np.log(p + 1e-12))) / np.mean(p))
            omega_n = n2 * omega0

            tau_phi_n_i = -phi2 / omega_n
            tau_phi_n.append(tau_phi_n_i)
            if np.sum(np.isfinite(tau_phi_n_i)) >= 2:
                tau_phi.append(np.median(tau_phi_n_i))
            else:
                tau_phi.append(np.nan)
            phi_unwrap = np.unwrap(phi2)

            A = np.column_stack([-omega_n, np.ones_like(omega_n)])
            params, _, _, _ = np.linalg.lstsq(A, phi_unwrap, rcond=None)

            tau_G.append(params[0])
            bv = np.real(Vn[0]) * np.ones_like(t)

            for k in range(1, 11):
                bv += (
                    Vn[k] * np.exp(1j * k * omega0 * t)
                    + np.conj(Vn[k]) * np.exp(-1j * k * omega0 * t)
                ).real
            BV.append(bv)
            vbase = np.min(bv)
            w = np.maximum(bv - vbase, 0)
            if np.sum(w) > 0:
                tau_M1.append(np.sum(w * t) / np.sum(w))
            else:
                tau_M1.append(np.nan)
            if np.sum(w) > 0:
                t2 = np.sum(w * t**2) / np.sum(w)
                sigma_M.append(np.sqrt(t2 - tau_M1[-1] ** 2))
            else:
                sigma_M.append(np.nan)

            tau_H_n = np.full_like(absX2, np.nan, dtype=float)

            valid2 = absX2 <= 1
            idx2 = np.where(valid2)[0]
            tau_H_n[idx2] = (1 / (omega_n[idx2])) * np.sqrt(
                (1 / (absX2[valid2] ** 2)) - 1
            )
            TAU_H.append(tau_H_n)
            tau_P_n = []

            for k in range(len(Xn[i]) - 1):
                taus = np.linspace(0, 10, 2000)
                H = 1 / (1 + 1j * omega_n[2] * taus)
                err = np.abs(Xn[i][k] - H) ** 2
                tau_P_n.append(taus[np.argmin(err)])

            TAU_P_N.append(tau_P_n)
            if np.sum(np.isfinite(tau_P_n)) >= 2:
                TAU_P.append(np.nanmean(tau_P_n))
            else:
                TAU_P.append(np.nan)
        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "Xn": with_attrs(
                np.asarray(Xn),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "H2": with_attrs(
                np.asarray(H2),
                {
                    "unit": [""],
                    "description": [
                        "Strength of the second harmonic relative to the fundamental"
                    ],
                },
            ),
            "Delta_Phi_2": with_attrs(
                np.asarray(Delta_Phi_2),
                {
                    "unit": ["rad"],
                    "interval": ["-pi , pi"],
                    "description": ["Global-shift-invariant “shape phase” at n = 2"],
                },
            ),
            "Delta_Phi_3": with_attrs(
                np.asarray(Delta_Phi_3),
                {
                    "unit": [""],
                    "description": ["Global-shift-invariant “shape phase” at n = 3"],
                },
            ),
            "R10": with_attrs(
                np.asarray(R10),
                {
                    "unit": [""],
                    "description": ["Pulsatility relative to mean level"],
                },
            ),
            "HRI_2_10": with_attrs(
                np.asarray(HRI_2_10),
                {
                    "unit": [""],
                    "description": [
                        "Relative high-frequency content beyond the fundamental"
                    ],
                },
            ),
            "HRI_2_10_noisecorrec": with_attrs(
                np.asarray(HRI_2_10_noisecorrec),
                {
                    "unit": [""],
                    "nb": ["not computed yet"],
                    "description": [
                        "Richness corrected for an estimated noise floor to reduce bias when high harmonics approach noise"
                    ],
                },
            ),
            "S1_10": with_attrs(
                np.asarray(S1_10),
                {
                    "unit": [""],
                    "description": [
                        "Compact damping descriptor: slope of a linear fit of log "
                    ],
                },
            ),
            "nc": with_attrs(
                np.asarray(nc),
                {
                    "unit": [""],
                    "description": ["nergy location across harmonics using pn"],
                },
            ),
            "Sigma_n": with_attrs(
                np.asarray(Sigma_n),
                {
                    "unit": [""],
                    "description": ["Spread of harmonic content around nc"],
                },
            ),
            "Hspec": with_attrs(
                np.asarray(Hspec),
                {
                    "unit": [""],
                    "description": [
                        "Complexity of harmonic distribution: higher values indicate more evenly distributed energy"
                    ],
                },
            ),
            "Fspec": with_attrs(
                np.asarray(Fspec),
                {
                    "unit": [""],
                    "description": [
                        "Peakedness vs flatness of the harmonic distribution; low values indicate dominance of few harmonics, high values indicate flatter spectra "
                    ],
                },
            ),
            "tau_phi": with_attrs(
                np.asarray(tau_phi),
                {
                    "unit": [""],
                    "description": [
                        "Amplitude-insensitive timing descriptor from harmonic phases: robust aggregate of per-harmonic delays "
                    ],
                },
            ),
            "tau_phi_n": with_attrs(
                np.asarray(tau_phi_n),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "tau_G": with_attrs(
                np.asarray(tau_G),
                {
                    "unit": [""],
                    "description": [
                        "Robust global delay estimated as the slope of (unwrapped) arg (Xn) vs ωn "
                    ],
                },
            ),
            "BV": with_attrs(
                np.asarray(BV),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "tau_M1": with_attrs(
                np.asarray(tau_M1),
                {
                    "unit": [""],
                    "description": [
                        "Point-free timing: centroid (center-of-mass) of the systolic lobe from a band-limited waveform using a positive weight w(t) within a fixed systolic window"
                    ],
                },
            ),
            "sigma_M": with_attrs(
                np.asarray(sigma_M),
                {
                    "unit": [""],
                    "description": ["dispersion proxy "],
                },
            ),
            "TAU_H_N": with_attrs(
                np.asarray(TAU_H),
                {
                    "unit": [""],
                    "description": [
                        "Scalar proxy of low-pass damping from harmonic magnitudes only. Larger τH indicates stronger attenuation of higher harmonics"
                    ],
                },
            ),
            "TAU_P_N": with_attrs(
                np.asarray(TAU_P_N),
                {
                    "unit": [""],
                    "description": [""],
                },
            ),
            "TAU_P": with_attrs(
                np.asarray(TAU_P),
                {
                    "unit": [""],
                    "description": [
                        "Phase-aware scalar summary of vascular damping and phase lag, estimated from complex harmonics under explicit validity gates"
                    ],
                },
            ),
        }

        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)
