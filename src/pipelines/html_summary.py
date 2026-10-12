"""Per-acquisition HTML summaries of exported EyeFlow metrics."""

from pathlib import Path

from input_output.output_paths import H5_OUTPUT_DIRNAME, html_output_dir

from .core.base import BatchPipeline, PipelineContext, PipelineResult, registerPipeline


@registerPipeline(
    name="HTML summary",
    description=(
        "Write one HTML summary per EyeFlow acquisition with arterial and venous "
        "waveform metric statistics, hemifield tables, and available vascular images."
    ),
    required_deps=["matplotlib>=3.8", "pandas>=2.1"],
    required_metric_options=[["waveform_shape_metrics", "topological_metrics"]],
)
class WaveformMetricSummaryTablesPipeline(BatchPipeline):
    def run(self, context: PipelineContext) -> PipelineResult:
        if not context.processed_files:
            raise ValueError("No EyeFlow HDF5 files are available for analysis.")

        from .utils.html_summary_dashboard import generate_metric_table_html_for_file

        output_dir = Path(context.output_dir).expanduser().resolve()
        paths = []
        for index, processed_file in enumerate(context.processed_files):
            try:
                relative_parent = (
                    processed_file.resolve().relative_to(output_dir).parent
                )
            except ValueError:
                relative_parent = Path(".")
            relative_parent = Path(
                *(
                    part
                    for part in relative_parent.parts
                    if part.lower() != H5_OUTPUT_DIRNAME.lower()
                )
            )
            html_path = (
                html_output_dir(output_dir)
                / relative_parent
                / f"{processed_file.stem}.html"
            )
            source_path = processed_file
            if index < len(context.input_h5_paths):
                original = context.input_h5_paths[index]
                if original.is_file():
                    source_path = original
            paths.append(
                generate_metric_table_html_for_file(
                    processed_file,
                    html_path,
                    source_path=source_path,
                )
            )
        return PipelineResult(
            summary=f"Generated {len(paths)} HTML summary file(s).",
            generated_paths=[str(path) for path in paths],
        )
