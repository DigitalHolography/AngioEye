from __future__ import annotations

import shutil
from collections import defaultdict

from input_output.hdf5_io import append_metrics_trees_to_h5
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT
from postprocess.core.grouped_batch import extract_group_name

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Variability and heterogeneity tables",
    description=(
        "Build group-level LaTeX and CSV tables for variability and heterogeneity "
        "metrics computed from by-segment arterial waveform shape metrics."
    ),
    required_deps=["pandas>=2.1", "scipy>=1.10"],
    required_pipeline_options=[
        ["waveform_shape_metrics"],
        ["waveform_shape_metrics_denoised"],
    ],
)
class VariabilityHeterogeneityPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        if not context.processed_files:
            raise ValueError(
                "No processed HDF5 outputs are available for postprocessing."
            )

        output_dir = context.output_dir.expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

        from .utils import variability_heterogeneity_dashboard

        def _idle() -> None:
            if context.idle_callback is not None:
                context.idle_callback()

        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for file_path in context.processed_files:
            blocks = (
                variability_heterogeneity_dashboard.compute_file_higher_metric_blocks(
                    file_path,
                    mode="raw_segment",
                )
            )
            tree = variability_heterogeneity_dashboard.variability_tree_from_blocks(
                blocks
            )

            if not blocks or tree is None:
                continue

            append_metrics_trees_to_h5(
                file_path,
                ANGIOEYE_POSTPROCESS_ROOT,
                [tree],
                overwrite=True,
            )
            group_name = extract_group_name(file_path.parent, output_dir)
            variability_heterogeneity_dashboard.add_file_blocks_to_results(
                results,
                group_name,
                blocks,
            )
            _idle()

        if not results:
            raise ValueError(
                "No compatible by-segment metrics were found for the variability/heterogeneity tables."
            )

        table_dir = output_dir / "latex_tables"
        if table_dir.exists():
            shutil.rmtree(table_dir)

        table_paths = (
            variability_heterogeneity_dashboard.export_group_tables_from_results(
                results,
                table_dir,
                idle_callback=context.idle_callback,
            )
        )

        created_paths = [str(path) for path in table_paths]
        summary = (
            f"Generated {len(table_paths)} variability/heterogeneity table file(s)."
        )
        return PostprocessResult(summary=summary, generated_paths=created_paths)

