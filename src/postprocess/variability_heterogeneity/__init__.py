"""Variability and heterogeneity postprocess package."""

from postprocess.core.base import registerPostprocess

from .compute import (
    add_file_blocks_to_results,
    compute_file_higher_metric_blocks,
    compute_file_higher_metrics_from_segment_array,
    iter_segment_metrics,
    variability_tree_from_blocks,
)
from .constants import EPS
from .run import VariabilityHeterogeneityPostprocess

registerPostprocess(
    name="Variability and heterogeneity",
    description=(
        "Build group-level LaTeX and CSV tables for variability and heterogeneity "
        "metrics computed from by-segment arterial waveform shape metrics."
    ),
    required_deps=["pandas>=2.1", "scipy>=1.10"],
    required_pipeline_options=[
        [
            "waveform_shape_metrics",  # OR
            "waveform_shape_metrics_denoised",
        ],
    ],
    input_methods=["file_batch", "cohort_batch", "zip_batch"],
)(VariabilityHeterogeneityPostprocess)


__all__ = [
    "EPS",
    "VariabilityHeterogeneityPostprocess",
    "add_file_blocks_to_results",
    "compute_file_higher_metric_blocks",
    "compute_file_higher_metrics_from_segment_array",
    "iter_segment_metrics",
    "variability_tree_from_blocks",
]
