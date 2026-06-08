from ..core.base import registerPipeline
from .pipeline import WaveformShapeMetricsDenoisedLaplaceBase


@registerPipeline(name="waveform_shape_metrics_denoised_laplace")
class WaveformShapeMetricsDenoisedLaplace(WaveformShapeMetricsDenoisedLaplaceBase):
    """Registered graph-Laplacian denoised waveform-shape pipeline."""


__all__ = ["WaveformShapeMetricsDenoisedLaplace"]
