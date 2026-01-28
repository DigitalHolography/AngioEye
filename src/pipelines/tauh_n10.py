import math
from dataclasses import dataclass
from typing import Dict

import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, register_pipeline


@dataclass
class TauHResult:
    tau: float
    x_abs: float
    vmax: float
    freq_hz: float


def _freq_unit(h5file: h5py.File, path: str) -> str:
    try:
        unit_attr = h5file[path].attrs.get("unit")
    except Exception:
        return "hz"
    if isinstance(unit_attr, bytes):
        unit_attr = unit_attr.decode("utf-8", errors="ignore")
    if isinstance(unit_attr, str) and "rad" in unit_attr.lower():
        return "rad/s"
    return "hz"


@register_pipeline(name="TauhN10")
class TauhN10(ProcessPipeline):
    """
    Acquisition-level τ|H|,10 using synthetic spectral amplitudes.
    τ|H|,n = (1 / ω_n) * sqrt(1 / |X_n|^2 - 1), ω_n = 2π f_n when freqs are in Hz.
    """

    description = "Magnitude-derived damping time constant τ|H| at harmonic n=10 using spectral amplitudes."
    harmonic_index = 10
    synthesis_points = 2048  # samples over one cardiac period for V_max estimation

    def run(self, h5file: h5py.File) -> ProcessResult:
        metrics: Dict[str, float] = {}
        artifacts: Dict[str, float] = {}
        for vessel in ("Artery", "Vein"):
            vessel_result = self._compute_for_vessel(h5file, vessel)
            prefix = vessel.lower()
            metrics[f"{prefix}_tauH_{self.harmonic_index}"] = vessel_result.tau
            metrics[f"{prefix}_X_abs_{self.harmonic_index}"] = vessel_result.x_abs
            artifacts[f"{prefix}_vmax"] = vessel_result.vmax
            artifacts[f"{prefix}_freq_hz_{self.harmonic_index}"] = vessel_result.freq_hz
        return ProcessResult(metrics=metrics, artifacts=artifacts)

    def _compute_for_vessel(self, h5file: h5py.File, vessel: str) -> TauHResult:
        spectral_prefix = "Arterial" if vessel.lower().startswith("arter") else "Venous"
        # Use acquisition-level synthetic spectral amplitudes/phases.
        amp_path = f"{vessel}/Velocity/WaveformAnalysis/syntheticSpectralAnalysis/{spectral_prefix}FourierAmplitude/value"
        phase_path = f"{vessel}/Velocity/WaveformAnalysis/syntheticSpectralAnalysis/{spectral_prefix}FourierPhase/value"
        freq_path = f"{vessel}/Velocity/WaveformAnalysis/syntheticSpectralAnalysis/{spectral_prefix}PeakFrequencies/value"
        try:
            amplitudes = np.asarray(h5file[amp_path]).astype(np.float64).ravel()
            phases = np.asarray(h5file[phase_path]).astype(np.float64).ravel()
            freqs = np.asarray(h5file[freq_path]).astype(np.float64).ravel()
        except KeyError as exc:  # noqa: BLE001
            raise ValueError(f"Missing spectral data for {vessel}: {exc}") from exc

        n = self.harmonic_index
        if amplitudes.shape[0] <= n or phases.shape[0] <= n or freqs.shape[0] <= n:
            raise ValueError(
                f"Not enough harmonics for {vessel}: need index {n}, got {amplitudes.shape[0]} harmonics"
            )

        # Frequencies may be stored in Hz or rad/s (check unit attr). We report freq_hz_* in Hz.
        freq_unit = _freq_unit(h5file, freq_path)
        is_hz = freq_unit == "hz"

        fundamental_hz = freqs[1] if is_hz else freqs[1] / (2 * math.pi)
        freq_n_raw = freqs[n]
        freq_n_hz = freq_n_raw if is_hz else freq_n_raw / (2 * math.pi)

        if fundamental_hz <= 0:
            raise ValueError(
                f"Invalid fundamental frequency for {vessel}: {fundamental_hz}"
            )

        # Reconstruct the band-limited waveform (0..n) to get Vmax.
        vmax = self._estimate_vmax(amplitudes, phases, freqs, is_hz, n)
        if not np.isfinite(vmax) or vmax <= 0:
            return TauHResult(
                tau=math.nan, x_abs=math.nan, vmax=float(vmax), freq_hz=freq_n_hz
            )

        v_n = amplitudes[n]
        x_abs = float(abs(v_n) / vmax)
        if x_abs <= 0 or x_abs > 1:
            tau = math.nan
        else:
            # ω_n = n·ω_0; if freqs are in Hz, multiply by 2π to get rad/s.
            omega_n = (2 * math.pi * freq_n_raw) if is_hz else freq_n_raw
            denom = (1.0 / (x_abs * x_abs)) - 1.0
            if denom <= 0 or omega_n <= 0:
                tau = math.nan
            else:
                tau = float(math.sqrt(denom) / omega_n)
        return TauHResult(
            tau=tau, x_abs=x_abs, vmax=float(vmax), freq_hz=float(freq_n_hz)
        )

    def _estimate_vmax(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        freqs: np.ndarray,
        is_hz: bool,
        n_max: int,
    ) -> float:
        """Reconstruct band-limited waveform (n=0..n_max) and return its maximum magnitude."""
        fundamental_hz = freqs[1] if is_hz else freqs[1] / (2 * math.pi)
        if fundamental_hz <= 0:
            return math.nan
        omega_factor = 2 * math.pi if is_hz else 1.0
        t = np.linspace(
            0.0,
            1.0 / fundamental_hz,
            num=self.synthesis_points,
            endpoint=False,
            dtype=np.float64,
        )
        waveform = np.full_like(t, fill_value=amplitudes[0], dtype=np.float64)
        for k in range(1, n_max + 1):
            # cosine synthesis over one cardiac period
            waveform += amplitudes[k] * np.cos(omega_factor * freqs[k] * t + phases[k])
        return float(np.max(np.abs(waveform)))
