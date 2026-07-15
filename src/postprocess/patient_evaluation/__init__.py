"""Fixed-threshold evaluation of patient waveform-shape metrics."""

from postprocess.core.base import (
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)

from .run import run_fixed_threshold_evaluation


@registerPostprocess(
    name="Fixed Threshold Patient Evaluation",
    description=(
        "Evaluates each patient HDF5 independently against a fixed panel of "
        "waveform-shape thresholds. Supports a single H5 or multiple H5 files, "
        "including H5 files contained in a ZIP archive."
    ),
    required_deps=["matplotlib>=3.8"],
    required_pipelines=["waveform_shape_metrics"],
)
def run(ctx: PostprocessContext) -> PostprocessResult:
    return run_fixed_threshold_evaluation(ctx)
