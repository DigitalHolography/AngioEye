from __future__ import annotations

from input_output.archive_io import (
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
    name="Group comparison (Dashboard) - Waveform Shape Metrics",
    description = """
    This module automatically analyzes vascular Doppler signals stored in HDF5 files to compute waveform metrics and generate the corresponding figures.

    ### Features

    - Analysis of **arterial** and **venous** vessels.
    - Support for **raw** and **bandlimited** signals.
    - Analysis of ZIP archives containing HDF5 files organized into **control** and **pathological** groups.
    - Direct use of the results produced by the **Waveform Shape Metrics** pipeline.

    ### Analysis

    For each waveform metric, the module computes the **median** and **standard deviation** across all cardiac cycles in each file. The analysis can be performed on **raw**, **bandlimited**, or **all** available processing modes.

    All figures are automatically exported in **PNG** and **EPS** formats.

    ### Generated figures

    Each figure includes:

    - A **statistical panel** showing individual subject values, the group mean, and the corresponding standard deviation.
    - An **illustrative panel** showing the physiological meaning of the selected metric using a representative waveform from the corresponding group.
    --------------------------------------------
    WARNING 

    To highlight the control group in **gray**, ensure that its name is correctly defined in `find_control_group_name()` in `..core.grouped_batch`.

    ### Configuration

    The generated figures can be customized by modifying the calls to `export_selected_metric()` in `save_dashboard()`.

    #### Processing mode

    The `mode` parameter selects which processed signals are used:

    - `raw`: use the raw (unfiltered) signals.
    - `bandlimited`: use the bandlimited signals.
    - `all`: generate figures for both processing modes.

    #### Group illustrations

    The `show_group_illustrations` parameter controls whether representative waveform illustrations are displayed alongside the statistical comparison:

    - `True`: include both the statistical comparison and the waveform illustrations.
    - `False`: generate only the statistical comparison.

    By default, figures are generated with `mode="bandlimited"` and `show_group_illustrations=True`.
    """,
    required_deps=["matplotlib>=3.8", "pandas>=2.1", "plotly>=5.18"],
    required_pipelines=["waveform_shape_metrics"],
    input_methods=["file_batch", "cohort_batch", "zip_batch"],
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

        from .utils import groups_comparison_dashboard

        with temporary_zip_from_tree(
            output_dir,
            source_paths=context.processed_files,
        ) as temp_zip:
            all_results, single_group = groups_comparison_dashboard.analyze_zip(
                str(temp_zip)
            )
            if not all_results:
                raise ValueError(
                    "No compatible pipeline metrics were found for the dashboard."
                )
            groups_comparison_dashboard.save_dashboard(
                all_results,
                str(temp_zip),
                single_group,
            )

            png_paths = extract_folder_from_zip(
                zip_path=temp_zip,
                member_prefix="group comparison (Dashboard) - Waveform Shape Metrics/",
                output_dir=output_dir,
            )

        created_paths = [            
            *[str(path) for path in png_paths]
        ]
        summary = f" Generated dashboard {len(png_paths)} PNG illustration(s)."
        return PostprocessResult(summary=summary, generated_paths=created_paths)

