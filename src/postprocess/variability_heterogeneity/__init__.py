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
    description="""
    The **Variability and Heterogeneity** module is dedicated to the post-processing of segment-level metrics in order to quantify the spatial and temporal variability of the analyzed data. It computes statistical descriptors from HDF5 files, aggregates the results by cohort, and compares each group against a control group using statistical analyses, including the Mann–Whitney test, AUC-based separability, sensitivity and specificity.

    The module automatically generates summary tables, graphical visualizations, and an interactive HTML report to facilitate data interpretation. The entire pipeline is optimized for batch processing and parallel execution, and exports the results in multiple formats, including HDF5, CSV, LaTeX, and HTML.

    --------------------------------------------
    WARNING 

    The **Variability and Heterogeneity** post-processing requires a **control group**. All statistical comparisons are performed against this group, and the analysis cannot be run without it.
    The control group folder name must match one of the names defined in `CONTROL_GROUP_PATTERNS` (`constants.py`). If a different name is used, it must be added to this list before running the post-processing.

""",
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
