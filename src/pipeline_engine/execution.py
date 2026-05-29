from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py

from input_output import (
    ANGIOEYE_PROCESSING_ROOT,
    create_h5_file,
    write_metrics_trees_to_h5,
)
from pipelines import (
    PipelineDescriptor,
    ProcessResult,
    process_results_to_metric_trees,
)
from pipelines.core.errors import format_pipeline_exception
from postprocess import PostprocessContext, PostprocessDescriptor

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
IdleCallback = Callable[[], None]


def run_pipeline_file(
    h5_path: Path,
    pipelines: Sequence[PipelineDescriptor],
    output_root: Path,
    output_relative_parent: Path = Path("."),
    output_filename: str | None = None,
    *,
    trim_source: bool = True,
    log: LogCallback | None = None,
    advance_progress: ProgressCallback | None = None,
    write_idle_callback: IdleCallback | None = None,
) -> Path:
    output_path = _unique_pipeline_output_path(
        h5_path=h5_path,
        output_root=output_root,
        output_relative_parent=output_relative_parent,
        output_filename=output_filename,
    )
    pipeline_results = _run_pipeline_descriptors(
        h5_path=h5_path,
        pipelines=pipelines,
        log=log,
        advance_progress=advance_progress,
    )
    _log(log, f"[SAVE] Writing output file -> {output_path.name}")
    _write_pipeline_output(
        pipeline_results=pipeline_results,
        output_path=output_path,
        source_file=str(h5_path),
        trim_source=trim_source,
        idle_callback=write_idle_callback,
    )
    for _, result in pipeline_results:
        result.output_h5_path = str(output_path)
    _log(log, f"[OK] {h5_path.name}: combined results -> {output_path}")
    return output_path


def run_postprocesses(
    postprocesses: Sequence[PostprocessDescriptor],
    output_dir: Path,
    processed_outputs: Sequence[Path],
    input_h5_paths: Sequence[Path],
    input_path: Path,
    selected_pipeline_names: Sequence[str],
    failures: list[str],
    *,
    zip_outputs: bool,
    log: LogCallback,
    advance_progress: ProgressCallback,
) -> None:
    context = PostprocessContext(
        output_dir=output_dir,
        processed_files=tuple(processed_outputs),
        selected_pipelines=tuple(selected_pipeline_names),
        input_path=input_path,
        zip_outputs=zip_outputs,
        input_h5_paths=tuple(input_h5_paths),
    )
    for descriptor in postprocesses:
        postprocess = descriptor.instantiate()
        log(f"[POST] Running {descriptor.name}...")
        try:
            result = postprocess.run(context)
        except Exception as exc:  # noqa: BLE001
            error_message = (
                f"Postprocess '{descriptor.name}' failed: "
                f"{type(exc).__name__}: {exc}"
            )
            failures.append(error_message)
            log(f"[POST FAIL] {error_message}")
            advance_progress(1.0)
            continue

        summary = (result.summary or "").strip()
        if summary:
            log(f"[POST OK] {descriptor.name}: {summary}")
        else:
            log(f"[POST OK] {descriptor.name}")
        for warning in _postprocess_result_failures(result):
            failures.append(warning)
            log(f"[POST WARN] {warning}")
        advance_progress(1.0)


def _unique_pipeline_output_path(
    *,
    h5_path: Path,
    output_root: Path,
    output_relative_parent: Path,
    output_filename: str | None,
) -> Path:
    target_dir = output_root / output_relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        base_output_path = target_dir / output_filename
        output_path = base_output_path
    else:
        base_output_path = target_dir / f"{h5_path.stem}_pipelines_result.h5"
        output_path = base_output_path

    suffix = 1
    while output_path.exists():
        if output_filename:
            output_path = (
                target_dir
                / f"{base_output_path.stem}_{suffix}{base_output_path.suffix}"
            )
        else:
            output_path = target_dir / f"{h5_path.stem}_{suffix}_pipelines_result.h5"
        suffix += 1
    return output_path


def _run_pipeline_descriptors(
    *,
    h5_path: Path,
    pipelines: Sequence[PipelineDescriptor],
    log: LogCallback | None,
    advance_progress: ProgressCallback | None,
) -> list[tuple[str, ProcessResult]]:
    pipeline_results: list[tuple[str, ProcessResult]] = []
    with h5py.File(h5_path, "r") as h5file:
        for pipeline_desc in pipelines:
            pipeline = pipeline_desc.instantiate()
            try:
                result = pipeline.run(h5file)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(format_pipeline_exception(exc, pipeline)) from exc
            pipeline_results.append((pipeline.name, result))
            _log(log, f"[OK] {h5_path.name} -> {pipeline.name}")
            _advance(advance_progress)
    return pipeline_results


def _write_pipeline_output(
    *,
    pipeline_results: Sequence[tuple[str, ProcessResult]],
    output_path: Path,
    source_file: str,
    trim_source: bool,
    idle_callback: IdleCallback | None,
) -> None:
    if idle_callback is None:
        _write_pipeline_output_sync(
            pipeline_results=pipeline_results,
            output_path=output_path,
            source_file=source_file,
            trim_source=trim_source,
        )
        return

    errors: list[Exception] = []
    done_event = threading.Event()

    def _worker() -> None:
        try:
            _write_pipeline_output_sync(
                pipeline_results=pipeline_results,
                output_path=output_path,
                source_file=source_file,
                trim_source=trim_source,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done_event.set()

    writer_thread = threading.Thread(target=_worker, daemon=True)
    writer_thread.start()
    while not done_event.wait(timeout=0.05):
        idle_callback()
    writer_thread.join()
    if errors:
        raise errors[0]


def _write_pipeline_output_sync(
    *,
    pipeline_results: Sequence[tuple[str, ProcessResult]],
    output_path: Path,
    source_file: str,
    trim_source: bool,
) -> None:
    create_h5_file(
        output_path,
        source_file=source_file,
        trim_source=trim_source,
    )
    write_metrics_trees_to_h5(
        output_path,
        ANGIOEYE_PROCESSING_ROOT,
        process_results_to_metric_trees(pipeline_results),
        overwrite=False,
    )


def _postprocess_result_failures(result) -> list[str]:
    failures = getattr(result, "metadata", {}).get("failures", [])
    if isinstance(failures, str):
        return [failures]
    if not isinstance(failures, Sequence):
        return []
    return [str(failure) for failure in failures if str(failure).strip()]


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)


def _advance(advance_progress: ProgressCallback | None) -> None:
    if advance_progress is not None:
        advance_progress(1.0)
