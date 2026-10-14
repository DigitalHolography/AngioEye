"""Fixed-threshold evaluation of patient waveform-shape metrics."""

from pipelines.core.base import (
    PipelineContext,
    PipelineResult,
    registerPipeline,
)

from .run import run_fixed_threshold_evaluation


@registerPipeline(
    name="Fixed Threshold Patient Evaluation",
    description=(
        "Evaluates each patient HDF5 independently against a fixed panel of "
        "waveform-shape thresholds. Supports a single H5 or multiple H5 files, "
        "including H5 files contained in a ZIP archive."
    ),
    required_deps=["matplotlib>=3.8"],
    required_metrics=["waveform_shape_metrics"],
)
def run(ctx: PipelineContext) -> PipelineResult:
    return run_fixed_threshold_evaluation(ctx)
