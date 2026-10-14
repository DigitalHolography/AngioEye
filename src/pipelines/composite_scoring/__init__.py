"""Pipeline composite scoring for Waveform Shape Metrics Pipeline."""

from pipelines.core.base import (
    PipelineContext,
    PipelineResult,
    registerPipeline,
)

from .run import run_composite_scoring


@registerPipeline(
    name="Composite Scoring",
    description=(
        "Appends WAS/WAS-c scores from automatically calibrated waveform "
        "shape metrics and writes cohort score visualizations/reports."
    ),
    required_deps=["matplotlib>=3.8", "scipy>=1.10"],
    required_metrics=["waveform_shape_metrics"],
    input_methods=["single_file", "file_batch", "cohort_batch", "zip_batch"],
)
def run(ctx: PipelineContext) -> PipelineResult:
    return run_composite_scoring(ctx)
