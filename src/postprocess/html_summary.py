from __future__ import annotations

from input_output.archive_io import extract_folder_from_zip, temporary_zip_from_tree
from input_output.output_paths import companion_output_dir, html_output_dir

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


def _is_zip_input(context: PostprocessContext) -> bool:
    return context.input_path.suffix.lower() == ".zip"


def _html_dir_for_path(path, fallback_output_dir):
    for candidate in (path, fallback_output_dir):
        try:
            return companion_output_dir(
                candidate,
                app_suffix="AE",
                query_type="html",
            )
        except ValueError:
            continue
    return html_output_dir(fallback_output_dir)


def _source_path_for_file(context: PostprocessContext, index: int, processed_file):
    if context.input_path.suffix.lower() == ".holo":
        return context.input_path
    if index < len(context.input_h5_paths):
        return context.input_h5_paths[index]
    return processed_file


@registerPostprocess(
    name="HTML summary", 
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

        from .utils import html_summary_dashboard

        if not _is_zip_input(context):
            table_paths = []
            for index, processed_file in enumerate(context.processed_files):
                source_path = _source_path_for_file(context, index, processed_file)
                html_dir = _html_dir_for_path(source_path, output_dir)
                html_path = html_dir / f"{processed_file.stem}.html"
                table_paths.append(
                    html_summary_dashboard.generate_metric_table_html_for_file(
                        processed_file,
                        html_path,
                        source_path=source_path,
                    )
                )

            created_paths = [str(path) for path in table_paths]
            summary = f"Generated {len(table_paths)} tables."
            return PostprocessResult(summary=summary, generated_paths=created_paths)

        with temporary_zip_from_tree(
            output_dir,
            source_paths=context.processed_files,
        ) as temp_zip:
            temp_root = temp_zip.parent
            all_results = html_summary_dashboard.analyze_zip(str(temp_zip))
            if not all_results:
                raise ValueError(
                    "No compatible pipeline metrics were found for the dashboard."
                )
            html_summary_dashboard.save_dashboard(
                str(temp_zip),
                output_dir=temp_root / "html",
            )

            table_paths = extract_folder_from_zip(
                zip_path=temp_zip,
                member_prefix="html",
                output_dir=output_dir,
            )

        created_paths = [*[str(path) for path in table_paths]]
        summary = f"Generated {len(table_paths)} tables."
        return PostprocessResult(summary=summary, generated_paths=created_paths)

