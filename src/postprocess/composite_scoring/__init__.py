"""Postprocess composite scoring for Waveform Shape Metrics Pipeline."""

from postprocess.core.base import registerPostprocess, PostprocessContext, PostprocessResult
from .run import run_composite_scoring

@registerPostprocess(
    name="Composite Scoring",
    description=(
        "Appends WAS/WAS-c scores from automatically calibrated waveform "
        "shape metrics and writes cohort score visualizations/reports."
    ),
    required_deps=["matplotlib>=3.8", "scipy>=1.10"],
    required_pipelines=["waveform_shape_metrics"],
)
def run(ctx: PostprocessContext) -> PostprocessResult:
    return run_composite_scoring(ctx)