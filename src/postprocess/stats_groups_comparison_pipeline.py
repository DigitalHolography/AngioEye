from __future__ import annotations

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="groups comparison stats",
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

        from .utils import stats_groups_comparison

        all_results = stats_groups_comparison.analyze_batch_root(output_dir)
        if not all_results:
            raise ValueError(
                "No compatible pipeline metrics were found for the dashboard."
            )
        png_dir = output_dir / "export_png"
        eps_dir = output_dir / "export_eps"
        generated_paths = stats_groups_comparison.save_dashboard_outputs(
            all_results,
            export_png_dir=png_dir,
            export_eps_dir=eps_dir,
        )
        png_count = sum(1 for path in generated_paths if path.suffix == ".png")
        created_paths = [str(path) for path in generated_paths]
        summary = f"Generated {png_count} PNG illustration(s)."
        return PostprocessResult(summary=summary, generated_paths=created_paths)

