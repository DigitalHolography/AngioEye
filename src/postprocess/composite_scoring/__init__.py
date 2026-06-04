"""Postprocess composite scoring for Waveform Shape Metrics Pipeline."""

from postprocess.core.base import registerPostprocess, PostprocessContext, PostprocessResult
from .run import run_composite_scoring

@registerPostprocess(
    name="Composite Scoring",
    description=(
        "Appends composite RWAS/RWAS4 scores from dimensionless retinal waveform "
        "shape metrics and writes cohort score visualizations under png/."
    ),
    required_deps=["matplotlib>=3.8"],
    required_pipelines=["waveform_shape_metrics"],
)
def run(ctx: PostprocessContext) -> PostprocessResult:
    return run_composite_scoring(ctx)