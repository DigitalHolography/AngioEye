from __future__ import annotations

from angioeye_io.archive_io import (
    extract_file_from_zip,
    extract_folder_from_zip,
    temporary_zip_from_tree,
)

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="groups comparison stats + html",
    description=(
        "Build the cohort HTML dashboard and PNG metric exports from arterial "
        "waveform shape metrics."
    ),
    required_deps=["matplotlib>=3.8", "pandas>=2.1", "plotly>=5.18"],
    required_pipelines=["waveform_shape_metrics"],
)
class GraphicsDashboardPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        if not context.processed_files:
            raise ValueError(
                "No processed HDF5 outputs are available for postprocessing."
            )

        output_dir = context.output_dir.expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

        from .core import groups_comparison_html

        with temporary_zip_from_tree(
            output_dir,
            source_paths=context.processed_files,
        ) as temp_zip:
            temp_root = temp_zip.parent
            all_results = groups_comparison_html.analyze_zip(str(temp_zip))
            if not all_results:
                raise ValueError(
                    "No compatible pipeline metrics were found for the dashboard."
                )
            groups_comparison_html.save_dashboard(
                str(temp_zip),
                export_png_dir=temp_root / "export_png_html",
            )

            png_paths = extract_folder_from_zip(
                zip_path=temp_zip,
                member_prefix="export_png_html/",
                output_dir=output_dir,
            )
            dashboard_path = extract_file_from_zip(
                zip_path=temp_zip,
                member_name="waveform_metrics_dashboard.html",
                output_dir=output_dir,
            )

        created_paths = [str(dashboard_path), *[str(path) for path in png_paths]]
        summary = f"Generated dashboard and {len(png_paths)} PNG illustration(s)."
        return PostprocessResult(summary=summary, generated_paths=created_paths)
