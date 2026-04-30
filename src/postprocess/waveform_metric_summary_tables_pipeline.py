from __future__ import annotations

from angioeye_io.archive_io import extract_folder_from_zip, temporary_zip_from_tree

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="waveform metric summary tables",
    description=(
        "Create an HTML report for each processed HDF5 file, including a summary table of waveform metrics and their corresponding visualizations."
    ),
    required_deps=["matplotlib>=3.8", "pandas>=2.1", "plotly>=5.18"],
    required_pipelines=["waveform_shape_metrics"],
)
class WaveformMetricSummaryTablesPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        if not context.processed_files:
            raise ValueError(
                "No processed HDF5 outputs are available for postprocessing."
            )

        output_dir = context.output_dir.expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

        from .utils import waveform_metric_summary_tables

        with temporary_zip_from_tree(
            output_dir,
            source_paths=context.processed_files,
        ) as temp_zip:
            temp_root = temp_zip.parent
            all_results = waveform_metric_summary_tables.analyze_zip(str(temp_zip))
            if not all_results:
                raise ValueError(
                    "No compatible pipeline metrics were found for the dashboard."
                )
            waveform_metric_summary_tables.save_dashboard(
                str(temp_zip),
                output_dir=temp_root / "html_metric_tables",
            )

            table_paths = extract_folder_from_zip(
                zip_path=temp_zip,
                member_prefix="html_metric_tables/",
                output_dir=output_dir,
            )

        created_paths = [*[str(path) for path in table_paths]]
        summary = f"Generated {len(table_paths)} tables."
        return PostprocessResult(summary=summary, generated_paths=created_paths)
