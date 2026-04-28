from __future__ import annotations

from angioeye_io.archive_io import extract_folder_from_zip, temporary_zip_from_tree
from angioeye_io.hdf5_io import MetricsTree, append_metrics_trees_to_h5, read_dataset
from angioeye_io.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT, find_pipeline_group

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
    required_deps=["pandas>=2.1"],
    required_pipelines=["waveform_shape_metrics"],
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

        from .core import variability_heterogeneity_dashboard

        for file_path in context.processed_files:
            tree = variability_heterogeneity_dashboard.write_variability_tree(file_path)

            if tree is None:
                continue

            append_metrics_trees_to_h5(
                file_path,
                ANGIOEYE_POSTPROCESS_ROOT,
                [tree],
                overwrite=True,
            )

        with temporary_zip_from_tree(
            output_dir,
            source_paths=context.processed_files,
        ) as temp_zip:
            results, metrics = variability_heterogeneity_dashboard.analyze_zip(
                str(temp_zip),
                mode="bandlimited_segment",
            )
            if not results:
                raise ValueError(
                    "No compatible by-segment metrics were found for the variability/heterogeneity tables."
                )

            variability_heterogeneity_dashboard.export_group_tables(
                str(temp_zip),
                mode="bandlimited_segment",
            )

            table_paths = extract_folder_from_zip(
                zip_path=temp_zip,
                member_prefix="latex_tables/",
                output_dir=output_dir,
            )

        created_paths = [str(path) for path in table_paths]
        summary = (
            f"Generated {len(table_paths)} variability/heterogeneity table file(s)."
        )
        return PostprocessResult(summary=summary, generated_paths=created_paths)
