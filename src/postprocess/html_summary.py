from __future__ import annotations

from pathlib import Path

from input_output.archive_io import extracted_zip_tree
from input_output.output_paths import (
    H5_OUTPUT_DIRNAME,
    companion_output_dir,
    html_output_dir,
)

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


def _zip_source_relative_path(input_h5_path) -> Path | None:
    parts = Path(input_h5_path).parts
    for index, part in enumerate(parts):
        if part.lower().startswith("batch_"):
            relative_parts = parts[index + 1 :]
            return Path(*relative_parts) if relative_parts else None
    return None


def _strip_h5_path_parts(path: Path) -> Path:
    kept_parts = [part for part in path.parts if part.lower() != H5_OUTPUT_DIRNAME]
    return Path(*kept_parts) if kept_parts else Path(".")


def _zip_html_relative_parent(output_dir, processed_file, source_relative_path):
    if source_relative_path is not None:
        return _strip_h5_path_parts(source_relative_path.parent)

    output_root = Path(output_dir).expanduser().resolve()
    output_h5_dir = output_root / H5_OUTPUT_DIRNAME
    try:
        relative_parent = (
            Path(processed_file)
            .expanduser()
            .resolve()
            .relative_to(output_h5_dir)
            .parent
        )
    except ValueError:
        try:
            relative_parent = (
                Path(processed_file)
                .expanduser()
                .resolve()
                .relative_to(output_root)
                .parent
            )
        except ValueError:
            relative_parent = Path(".")
    return _strip_h5_path_parts(relative_parent)


@registerPostprocess(
    name="HTML summary",

    description="""
    This module automatically generates an HTML report for each analyzed HDF5 file.

    The report is based on the results produced by the *Waveform Shape Metrics*
    pipeline and provides a comprehensive summary of the computed waveform metrics.

    For each metric, the report displays:

    - the median value across all cardiac cycles;
    - the corresponding standard deviation;
    - the values computed separately for arterial and venous waveforms.

    In addition, the report includes the main visualizations associated with the
    analyzed file:

    - the M0 image;
    - the arterial mask;
    - the arterial velocity waveform;
    - the fAVG mean image;
    - the venous mask;
    - the venous velocity waveform.

    All images are embedded directly into the HTML report and can be viewed in
    full-screen mode to facilitate visual inspection.

    Generated reports are saved in the **HTML summary** output directory.
    """,
    required_deps=["matplotlib>=3.8", "pandas>=2.1", "plotly>=5.18"],
    required_pipeline_options=[
            [
                "waveform_shape_metrics",  # OR
                "topological_metrics",
            ],
        ],
    required_option=["persist_eyeflow_data"],
    input_methods=["single_file", "file_batch", "cohort_batch", "zip_batch"],
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

        table_paths = []
        with extracted_zip_tree(context.input_path) as source_root:
            for index, processed_file in enumerate(context.processed_files):
                source_relative_path = None
                source_path = None
                if index < len(context.input_h5_paths):
                    source_relative_path = _zip_source_relative_path(
                        context.input_h5_paths[index]
                    )
                    if source_relative_path is not None:
                        source_path = source_root / source_relative_path
                html_parent = _zip_html_relative_parent(
                    output_dir,
                    processed_file,
                    source_relative_path,
                )
                html_path = (
                    output_dir / "HTML summary" / html_parent / f"{processed_file.stem}.html"
                )
                table_paths.append(
                    html_summary_dashboard.generate_metric_table_html_for_file(
                        processed_file,
                        html_path,
                        source_path=source_path or processed_file,
                    )
                )

        created_paths = [*[str(path) for path in table_paths]]
        summary = f"Generated {len(table_paths)} tables."
        return PostprocessResult(summary=summary, generated_paths=created_paths)

