from __future__ import annotations

import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

from app_settings import AppSettingsStore
from batch_engine import DEFAULT_PROCESS_WORKERS, iter_batches, settings_int
from input_output.hdf5_io import append_metrics_trees_to_h5
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT
from postprocess.core.grouped_batch import extract_group_name

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)

DEFAULT_VARIABILITY_BATCH_SIZE = 8


def _compute_variability_batch(file_paths):
    from .utils import variability_heterogeneity_dashboard

    started_at = time.monotonic()
    results = tuple(
        (
            file_path,
            variability_heterogeneity_dashboard.compute_file_higher_metric_blocks(
                file_path,
                mode="bandlimited_segment",
            ),
        )
        for file_path in file_paths
    )
    return time.monotonic() - started_at, results


@registerPostprocess(
    name="Variability and heterogeneity",
    description=(
        "Build group-level LaTeX and CSV tables for variability and heterogeneity "
        "metrics computed from by-segment arterial waveform shape metrics."
    ),
    required_deps=["pandas>=2.1", "scipy>=1.10"],
    required_pipeline_options=[
        [
            "waveform_shape_metrics", # OR
            "waveform_shape_metrics_denoised",
        ],
    ],
    input_methods=["file_batch", "cohort_batch", "zip_batch"],
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

        def _record_timing(label: str, seconds: float) -> None:
            if context.record_timing is not None:
                context.record_timing(label, seconds)

        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        write_queue = deque()
        setup_started_at = time.monotonic()
        settings_store = AppSettingsStore()
        batch_settings = settings_store.load_batch_execution()
        postprocess_batching = settings_store.load_postprocess_batching()
        variability_settings = postprocess_batching.get("variability", {})
        if not isinstance(variability_settings, dict):
            variability_settings = {}
        batch_size = settings_int(
            variability_settings,
            "batch_size",
            settings_int(
                batch_settings,
                "batch_size",
                DEFAULT_VARIABILITY_BATCH_SIZE,
            ),
        )
        process_workers = settings_int(
            variability_settings,
            "process_workers",
            settings_int(
                batch_settings,
                "process_workers",
                DEFAULT_PROCESS_WORKERS,
            ),
        )
        file_batches = list(iter_batches(context.processed_files, batch_size))
        process_count = min(len(file_batches), process_workers)
        _record_timing(
            "variability postprocess setup",
            time.monotonic() - setup_started_at,
        )

        def _drain_write_queue() -> None:
            started_at = time.monotonic()
            drained = 0
            while write_queue:
                file_path, blocks = write_queue.popleft()
                tree = (
                    variability_heterogeneity_dashboard.variability_tree_from_blocks(
                        blocks
                    )
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
                drained += 1
            if drained:
                _record_timing(
                    "variability postprocess H5 write queue drain",
                    time.monotonic() - started_at,
                )

        pool_started_at = time.monotonic()
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            futures = [
                executor.submit(_compute_variability_batch, file_batch)
                for file_batch in file_batches
            ]
            for future in as_completed(futures):
                batch_compute_seconds, batch_results = future.result()
                _record_timing(
                    "variability postprocess worker batch compute",
                    batch_compute_seconds,
                )
                handoff_started_at = time.monotonic()
                write_queue.extend(batch_results)
                _record_timing(
                    "variability postprocess result queue handoff",
                    time.monotonic() - handoff_started_at,
                )
                _drain_write_queue()
        _record_timing(
            "variability postprocess process pool wall time",
            time.monotonic() - pool_started_at,
        )

        if not results:
            raise ValueError(
                "No compatible by-segment metrics were found for the variability/heterogeneity tables."
            )

        table_dir = output_dir / "Variability and heterogeneity"
        if table_dir.exists():
            shutil.rmtree(table_dir)

        table_export_started_at = time.monotonic()
        table_paths = variability_heterogeneity_dashboard.export_group_tables_from_results(
            results,
            table_dir,
            idle_callback=context.idle_callback,
        )
        _record_timing(
            "variability postprocess table export",
            time.monotonic() - table_export_started_at,
        )

        created_paths = [str(path) for path in table_paths]
        summary = (
            f"Generated {len(created_paths)} variability/heterogeneity table file(s)."
        )
        return PostprocessResult(summary=summary, generated_paths=created_paths)
