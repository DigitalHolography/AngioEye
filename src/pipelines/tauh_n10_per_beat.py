import math

import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, register_pipeline
from .tauh_n10 import _freq_unit


@register_pipeline(name="TauhN10PerBeat")
class TauhN10PerBeat(ProcessPipeline):
    """
    Per-beat τ|H|,10 using per-beat FFT amplitudes and VmaxPerBeatBandLimited.
    Produces per-beat τ plus median/mean across beats.
    """

    description = "Per-beat τ|H| at harmonic n=10 (amplitude-only) for artery and vein."
    harmonic_index = 10

    def run(self, h5file: h5py.File) -> ProcessResult:
        metrics: dict[str, float] = {}
        artifacts: dict[str, float] = {}
        for vessel in ("Artery", "Vein"):
            vessel_metrics, vessel_artifacts = self._compute_per_beat(h5file, vessel)
            metrics.update(vessel_metrics)
            artifacts.update(vessel_artifacts)
        return ProcessResult(metrics=metrics, artifacts=artifacts)

    def _compute_per_beat(
        self, h5file: h5py.File, vessel: str
    ) -> tuple[dict[str, float], dict[str, float]]:
        n = self.harmonic_index
        prefix = vessel.lower()
        # Per-beat FFT amplitudes/phases and per-beat Vmax for the band-limited signal.
        amp_path = f"{vessel}/PerBeat/VelocitySignalPerBeatFFT_abs/value"
        phase_path = f"{vessel}/PerBeat/VelocitySignalPerBeatFFT_arg/value"
        vmax_path = f"{vessel}/PerBeat/VmaxPerBeatBandLimited/value"
        freq_path = (
            f"{vessel}/Velocity/WaveformAnalysis/syntheticSpectralAnalysis/"
            f"{'Arterial' if vessel.lower().startswith('arter') else 'Venous'}PeakFrequencies/value"
        )

        try:
            amps = np.asarray(h5file[amp_path]).astype(np.float64)
            phases = np.asarray(h5file[phase_path]).astype(np.float64)
            vmax = np.asarray(h5file[vmax_path]).astype(np.float64)
            freqs = np.asarray(h5file[freq_path]).astype(np.float64).ravel()
        except KeyError as exc:  # noqa: BLE001
            raise ValueError(
                f"Missing per-beat spectral data for {vessel}: {exc}"
            ) from exc

        if amps.shape[0] <= n or phases.shape[0] <= n:
            raise ValueError(
                f"Not enough harmonics in per-beat FFT for {vessel}: need index {n}"
            )
        if freqs.shape[0] <= n:
            raise ValueError(
                f"Not enough frequency samples for {vessel}: need index {n}"
            )
        if vmax.ndim != 2 or vmax.shape[1] != amps.shape[1]:
            raise ValueError(
                f"Mismatch in beat count for {vessel}: vmax {vmax.shape}, amps {amps.shape}"
            )

        # Frequency handling mirrors the acquisition-level pipeline.
        freq_unit = _freq_unit(h5file, freq_path)
        is_hz = freq_unit == "hz"
        freq_n_raw = freqs[n]
        freq_n_hz = freq_n_raw if is_hz else freq_n_raw / (2 * math.pi)
        omega_n = (2 * math.pi * freq_n_raw) if is_hz else freq_n_raw

        tau_values: list[float] = []
        x_values: list[float] = []
        vmax_values: list[float] = []
        beat_count = amps.shape[1]
        for beat_idx in range(beat_count):
            v_max = float(vmax[0, beat_idx])
            vmax_values.append(v_max)
            v_n = float(amps[n, beat_idx])
            x_abs = math.nan if v_max <= 0 else abs(v_n) / v_max
            x_values.append(x_abs)
            if (
                v_max <= 0
                or not np.isfinite(x_abs)
                or x_abs <= 0
                or x_abs > 1
                or omega_n <= 0
            ):
                tau_values.append(math.nan)
                continue
            denom = (1.0 / (x_abs * x_abs)) - 1.0
            tau_values.append(
                float(math.sqrt(denom) / omega_n) if denom > 0 else math.nan
            )

        metrics: dict[str, float] = {}
        artifacts: dict[str, float] = {f"{prefix}_freq_hz_{n}": freq_n_hz}
        for i, tau in enumerate(tau_values):
            metrics[f"{prefix}_tauH_{n}_beat{i}"] = tau
            artifacts[f"{prefix}_vmax_beat{i}"] = vmax_values[i]
            artifacts[f"{prefix}_X_abs_{n}_beat{i}"] = x_values[i]
        metrics[f"{prefix}_tauH_{n}_median"] = (
            float(np.nanmedian(tau_values)) if tau_values else math.nan
        )
        metrics[f"{prefix}_tauH_{n}_mean"] = (
            float(np.nanmean(tau_values)) if tau_values else math.nan
        )
        return metrics, artifacts
