from __future__ import annotations

import shutil
import threading
import time
from functools import cache, partial
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Protocol

from batch_engine import (
    BatchGroupResult,
    BatchTaskResult,
    can_pickle,
    iter_batches,
    run_task_batch,
    run_threaded_batches_in_process_pool,
)
from input_output import ZipH5Member
from pipelines import load_pipeline_catalog

from ._holo import HoloInputContext, output_filename
from ._standard_pipeline_runs import (
    PipelineRunResult,
    RunPipelineFile,
    run_filesystem_pipeline_run,
    run_zip_pipeline_run,
)
from ._zip_batches import ZipBatchSettings
from .timing import TimingSamples

_PIPELINE_RESOLVE_LOCK = threading.Lock()


@dataclass
class RunWorkflowResult:
    output_dir: Path
    processed_outputs: list[Path]
    failures: list[str]
    summary_message: str
    processed_input_paths: list[Path] = field(default_factory=list)
    zip_path: Path | None = None
    zip_failed: bool = False
    zip_error: str | None = None
    cleanup_output_dir: Callable[[], None] = lambda: None


ZipWorkflowResult = RunWorkflowResult


@dataclass(frozen=True)
class _WorkflowWorkspace:
    output_dir: Path
    temporary_output_dir: Path | None = None

    @property
    def uses_temporary_output_dir(self) -> bool:
        return self.temporary_output_dir is not None

    def cleanup_temporary_output_dir(self) -> None:
        if self.temporary_output_dir is not None:
            shutil.rmtree(self.temporary_output_dir, ignore_errors=True)


@dataclass(frozen=True)
class _FinalizedOutputs:
    summary_message: str
    zip_path: Path | None = None
    zip_failed: bool = False
    zip_error: str | None = None


class RunPostprocesses(Protocol):
    def __call__(
        self,
        postprocesses: Sequence[Any],
        output_dir: Path,
        processed_outputs: Sequence[Path],
        input_h5_paths: Sequence[Path],
        input_path: Path,
        selected_pipeline_names: Sequence[str],
        failures: list[str],
        *,
        zip_outputs: bool,
        record_timing: Callable[[str, float], None] | None = None,
    ) -> None: ...


ZipProgressCallback = Callable[[int, int, Path], None]
ZipOutputDir = Callable[[Path, Path | None, ZipProgressCallback | None], Path]
IdleCallback = Callable[[], None]
ZIP_COMPANION_OUTPUT_FOLDERS = ("png",)


@dataclass(frozen=True)
class HoloPipelineJob:
    context: HoloInputContext
    output_filename: str


def copy_zip_companion_output_folders(
    source_root: Path,
    output_root: Path,
) -> list[Path]:
    copied_roots: list[Path] = []
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    for folder_name in ZIP_COMPANION_OUTPUT_FOLDERS:
        source_dir = source_root / folder_name
        if not source_dir.is_dir():
            continue

        target_dir = output_root / folder_name
        copied_any = False
        for source_file in sorted(
            path for path in source_dir.rglob("*") if path.is_file()
        ):
            relative_path = source_file.relative_to(source_dir)
            target_file = target_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied_any = True

        if copied_any:
            copied_roots.append(target_dir)

    return copied_roots


def _prepare_workflow_workspace(
    *,
    base_output_dir: Path,
    zip_outputs: bool,
) -> _WorkflowWorkspace:
    if not zip_outputs:
        return _WorkflowWorkspace(output_dir=base_output_dir)

    temporary_output_dir = Path(mkdtemp(dir=base_output_dir))
    return _WorkflowWorkspace(
        output_dir=temporary_output_dir,
        temporary_output_dir=temporary_output_dir,
    )


def run_filesystem_workflow(
    *,
    inputs: Iterable[Path],
    data_root: Path,
    pipelines: Sequence[Any],
    postprocesses: Sequence[Any],
    selected_pipeline_names: Sequence[str],
    input_path: Path,
    base_output_dir: Path,
    zip_outputs: bool,
    zip_name: str,
    output_filename: str | None,
    settings: ZipBatchSettings,
    run_pipeline_file: RunPipelineFile,
    run_postprocesses: RunPostprocesses,
    relative_parent: Callable[[Path, Path], Path],
    zip_output_dir: ZipOutputDir,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    start_final_progress: Callable[[float, str], None],
    set_status: Callable[[str], None],
    make_zip_progress_callback: Callable[[], ZipProgressCallback | None],
    on_zip_error: Callable[[str], None] | None = None,
    idle_callback: IdleCallback | None = None,
) -> RunWorkflowResult:
    workspace = _prepare_workflow_workspace(
        base_output_dir=base_output_dir,
        zip_outputs=zip_outputs,
    )
    cleanup_workspace = False

    try:
        pipeline_started_at = time.monotonic()
        pipeline_result = run_filesystem_pipeline_run(
            inputs=inputs,
            data_root=data_root,
            pipelines=pipelines,
            output_dir=workspace.output_dir,
            output_filename=output_filename,
            settings=settings,
            run_pipeline_file=run_pipeline_file,
            relative_parent=relative_parent,
            log=log,
            advance_progress=advance_progress,
            idle_callback=idle_callback,
        )
        _log_elapsed(log, "Pipeline phase", pipeline_started_at)

        _run_workflow_postprocesses(
            pipeline_result=pipeline_result,
            postprocesses=postprocesses,
            selected_pipeline_names=selected_pipeline_names,
            input_path=input_path,
            output_dir=workspace.output_dir,
            run_postprocesses=run_postprocesses,
            log=log,
            advance_progress=advance_progress,
            start_final_progress=start_final_progress,
            final_progress_units=len(postprocesses) + (1 if zip_outputs else 0),
            zip_outputs=zip_outputs,
        )

        finalized_outputs = _finalize_workflow_outputs(
            output_dir=workspace.output_dir,
            base_output_dir=base_output_dir,
            processed_outputs=pipeline_result.processed_outputs,
            zip_outputs=zip_outputs,
            zip_name=zip_name,
            zip_output_dir=zip_output_dir,
            log=log,
            advance_progress=advance_progress,
            set_status=set_status,
            make_zip_progress_callback=make_zip_progress_callback,
            on_zip_error=on_zip_error,
            timings=pipeline_result.timings,
        )
        cleanup_workspace = (
            workspace.uses_temporary_output_dir and not finalized_outputs.zip_failed
        )
        workflow_result = _workflow_result(
            output_dir=workspace.output_dir,
            pipeline_result=pipeline_result,
            finalized_outputs=finalized_outputs,
        )
        _log_timing_averages(log, pipeline_result.timings)
        return workflow_result
    finally:
        if cleanup_workspace:
            workspace.cleanup_temporary_output_dir()


def run_zip_workflow(
    *,
    zip_path: Path,
    members: Iterable[ZipH5Member],
    member_count: int,
    pipelines: Sequence[Any],
    postprocesses: Sequence[Any],
    selected_pipeline_names: Sequence[str],
    base_output_dir: Path,
    zip_outputs: bool,
    zip_name: str,
    settings: ZipBatchSettings,
    run_pipeline_file: RunPipelineFile,
    run_postprocesses: RunPostprocesses,
    zip_output_dir: ZipOutputDir,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    start_final_progress: Callable[[float, str], None],
    set_status: Callable[[str], None],
    make_zip_progress_callback: Callable[[], ZipProgressCallback | None],
    on_zip_error: Callable[[str], None] | None = None,
    idle_callback: IdleCallback | None = None,
) -> RunWorkflowResult:
    workspace = _prepare_workflow_workspace(
        base_output_dir=base_output_dir,
        zip_outputs=zip_outputs,
    )
    cleanup_workspace = False

    try:
        pipeline_started_at = time.monotonic()
        pipeline_result = run_zip_pipeline_run(
            zip_path=zip_path,
            members=members,
            member_count=member_count,
            pipelines=pipelines,
            output_dir=workspace.output_dir,
            settings=settings,
            run_pipeline_file=run_pipeline_file,
            log=log,
            advance_progress=advance_progress,
            idle_callback=idle_callback,
        )
        _log_elapsed(log, "Pipeline phase", pipeline_started_at)

        _run_workflow_postprocesses(
            pipeline_result=pipeline_result,
            postprocesses=postprocesses,
            selected_pipeline_names=selected_pipeline_names,
            input_path=zip_path,
            output_dir=workspace.output_dir,
            run_postprocesses=run_postprocesses,
            log=log,
            advance_progress=advance_progress,
            start_final_progress=start_final_progress,
            final_progress_units=len(postprocesses) + (1 if zip_outputs else 0),
            zip_outputs=zip_outputs,
        )

        finalized_outputs = _finalize_workflow_outputs(
            output_dir=workspace.output_dir,
            base_output_dir=base_output_dir,
            processed_outputs=pipeline_result.processed_outputs,
            zip_outputs=zip_outputs,
            zip_name=zip_name,
            zip_output_dir=zip_output_dir,
            log=log,
            advance_progress=advance_progress,
            set_status=set_status,
            make_zip_progress_callback=make_zip_progress_callback,
            on_zip_error=on_zip_error,
            timings=pipeline_result.timings,
        )
        cleanup_workspace = (
            workspace.uses_temporary_output_dir and not finalized_outputs.zip_failed
        )
        workflow_result = _workflow_result(
            output_dir=workspace.output_dir,
            pipeline_result=pipeline_result,
            finalized_outputs=finalized_outputs,
        )
        _log_timing_averages(log, pipeline_result.timings)
        return workflow_result
    finally:
        if cleanup_workspace:
            workspace.cleanup_temporary_output_dir()


def _workflow_result(
    *,
    output_dir: Path,
    pipeline_result: PipelineRunResult,
    finalized_outputs: _FinalizedOutputs,
) -> RunWorkflowResult:
    return RunWorkflowResult(
        output_dir=output_dir,
        processed_outputs=pipeline_result.processed_outputs,
        processed_input_paths=pipeline_result.processed_input_paths,
        failures=pipeline_result.failures,
        summary_message=finalized_outputs.summary_message,
        zip_path=finalized_outputs.zip_path,
        zip_failed=finalized_outputs.zip_failed,
        zip_error=finalized_outputs.zip_error,
    )


def run_holo_workflow(
    *,
    contexts: Sequence[HoloInputContext],
    pipelines: Sequence[Any],
    postprocesses: Sequence[Any],
    selected_pipeline_names: Sequence[str],
    run_pipeline_file: RunPipelineFile,
    run_postprocesses: RunPostprocesses,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    start_final_progress: Callable[[float, str], None],
    settings: ZipBatchSettings,
    idle_callback: IdleCallback | None = None,
) -> RunWorkflowResult:
    result = PipelineRunResult()
    pipeline_started_at = time.monotonic()
    outputs_by_context: dict[HoloInputContext, list[Path]] = {
        context: [] for context in contexts
    }
    inputs_by_context: dict[HoloInputContext, list[Path]] = {
        context: [] for context in contexts
    }

    jobs = [
        HoloPipelineJob(
            context=context,
            output_filename=output_filename(context.holo_path),
        )
        for context in contexts
    ]
    for context in contexts:
        log(f"[INPUT] Holo file -> {context.holo_path}")
        log(f"[INPUT] EF h5 -> {context.h5_path}")
        log(f"[OUTPUT] AE folder -> {context.output_dir}")

    _run_holo_pipeline_jobs(
        jobs=jobs,
        pipelines=pipelines,
        run_pipeline_file=run_pipeline_file,
        settings=settings,
        result=result,
        outputs_by_context=outputs_by_context,
        inputs_by_context=inputs_by_context,
        log=log,
        advance_progress=advance_progress,
        idle_callback=idle_callback,
    )
    _log_elapsed(log, "Pipeline phase", pipeline_started_at)

    if postprocesses:
        start_final_progress(
            len(contexts) * len(postprocesses),
            "Running postprocess...",
        )
        postprocess_started_at = time.monotonic()
        for context in contexts:
            processed_outputs = outputs_by_context[context]
            if processed_outputs:
                run_postprocesses(
                    postprocesses,
                    context.output_dir,
                    processed_outputs,
                    inputs_by_context[context],
                    context.holo_path,
                    selected_pipeline_names,
                    result.failures,
                    zip_outputs=False,
                )
            else:
                log(
                    f"[POST SKIP] {context.holo_path.name}: no successful pipeline "
                    "outputs were generated."
                )
                advance_progress(len(postprocesses))
        _log_elapsed(log, "Postprocess phase", postprocess_started_at)

    return RunWorkflowResult(
        output_dir=contexts[0].output_dir if contexts else Path("."),
        processed_outputs=result.processed_outputs,
        processed_input_paths=result.processed_input_paths,
        failures=result.failures,
        summary_message=_holo_summary(
            result.processed_outputs,
            input_count=len(contexts),
            failure_count=len(result.failures),
        ),
    )


def _run_holo_pipeline_jobs(
    *,
    jobs: Sequence[HoloPipelineJob],
    pipelines: Sequence[Any],
    run_pipeline_file: RunPipelineFile,
    settings: ZipBatchSettings,
    result: PipelineRunResult,
    outputs_by_context: dict[HoloInputContext, list[Path]],
    inputs_by_context: dict[HoloInputContext, list[Path]],
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    idle_callback: IdleCallback | None,
) -> None:
    if not jobs:
        return

    batches = list(iter_batches(jobs, settings.batch_size))
    pipeline_names = _pipeline_names(pipelines)
    run_job = partial(
        _run_holo_pipeline_job_by_name,
        pipeline_names=pipeline_names,
        run_pipeline_file=run_pipeline_file,
    )
    use_process_pool = settings.process_workers > 1 and can_pickle(run_job)
    if use_process_pool:
        process_count = min(len(batches), max(1, settings.process_workers))
        thread_count = max(1, settings.batch_size)
        log(
            f"[PROCESS] Starting ProcessPoolExecutor(max_workers={process_count}) "
            f"for {len(batches)} holo batch(es); each process uses "
            f"ThreadPoolExecutor(max_workers={thread_count})."
        )
        for batch_index, batch in enumerate(batches, start=1):
            log(
                f"[PROCESS] Queued holo batch {batch_index}/{len(batches)} "
                f"({len(batch)} file(s))."
            )
        for batch_result in run_threaded_batches_in_process_pool(
            batches,
            run_item=run_job,
            process_workers=process_count,
            thread_workers=thread_count,
            idle_callback=idle_callback,
        ):
            _record_holo_batch_result(
                batch_result=batch_result,
                jobs=batches[batch_result.index - 1],
                result=result,
                outputs_by_context=outputs_by_context,
                inputs_by_context=inputs_by_context,
                log=log,
                advance_progress=advance_progress,
                pipeline_count=len(pipeline_names),
            )
        log(f"[PROCESS] Process pool completed {len(batches)} holo batch(es).")
        return

    if settings.process_workers > 1:
        log(
            "[HOLO WARN] Process pool disabled because the configured file runner "
            "or pipeline descriptors cannot be pickled; using threads."
        )
    for task_result in run_task_batch(
        jobs,
        run_item=run_job,
        max_workers=settings.batch_size,
        idle_callback=idle_callback,
    ):
        _record_holo_task_result(
            task_result=task_result,
            result=result,
            outputs_by_context=outputs_by_context,
            inputs_by_context=inputs_by_context,
            log=log,
            advance_progress=advance_progress,
            pipeline_count=len(pipeline_names),
        )


def _run_holo_pipeline_job(
    job: HoloPipelineJob,
    *,
    pipelines: Sequence[Any],
    run_pipeline_file: RunPipelineFile,
) -> Path:
    return run_pipeline_file(
        job.context.h5_path,
        pipelines,
        job.context.output_dir,
        Path("."),
        job.output_filename,
    )


def _run_holo_pipeline_job_by_name(
    job: HoloPipelineJob,
    *,
    pipeline_names: tuple[str, ...],
    run_pipeline_file: RunPipelineFile,
) -> Path:
    return _run_holo_pipeline_job(
        job,
        pipelines=_pipeline_descriptors_by_name(pipeline_names),
        run_pipeline_file=run_pipeline_file,
    )


def _pipeline_names(pipelines: Sequence[Any]) -> tuple[str, ...]:
    return tuple(getattr(pipeline, "name", str(pipeline)) for pipeline in pipelines)


@cache
def _pipeline_descriptors_by_name(pipeline_names: tuple[str, ...]) -> tuple[Any, ...]:
    with _PIPELINE_RESOLVE_LOCK:
        available, _missing = load_pipeline_catalog()
        registry = {pipeline.name: pipeline for pipeline in available}
        missing = [name for name in pipeline_names if name not in registry]
        if missing:
            raise ValueError(
                f"Pipeline(s) not available in worker: {', '.join(missing)}"
            )
        return tuple(registry[name] for name in pipeline_names)


def _record_holo_batch_result(
    *,
    batch_result: BatchGroupResult[HoloPipelineJob, Path],
    jobs: Sequence[HoloPipelineJob],
    result: PipelineRunResult,
    outputs_by_context: dict[HoloInputContext, list[Path]],
    inputs_by_context: dict[HoloInputContext, list[Path]],
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    pipeline_count: int,
) -> None:
    if batch_result.error is not None:
        log(
            f"[BATCH FAIL] Holo batch {batch_result.index}/"
            f"{batch_result.count}: {batch_result.error}"
        )
        for job in jobs:
            message = f"Batch process failed: {batch_result.error}"
            result.failures.append(f"{job.context.h5_path}: {message}")
            log(f"[FAIL] {job.context.h5_path.name}: {message}")
            advance_progress(pipeline_count)
        return

    log(
        f"[BATCH OK] Holo batch {batch_result.index}/{batch_result.count} "
        f"finished in {_format_elapsed(batch_result.elapsed_seconds)} "
        f"(process {batch_result.process_id})."
    )
    for task_result in batch_result.results:
        _record_holo_task_result(
            task_result=task_result,
            result=result,
            outputs_by_context=outputs_by_context,
            inputs_by_context=inputs_by_context,
            log=log,
            advance_progress=advance_progress,
            pipeline_count=pipeline_count,
        )


def _record_holo_task_result(
    *,
    task_result: BatchTaskResult[HoloPipelineJob, Path],
    result: PipelineRunResult,
    outputs_by_context: dict[HoloInputContext, list[Path]],
    inputs_by_context: dict[HoloInputContext, list[Path]],
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    pipeline_count: int,
) -> None:
    job = task_result.item
    context = job.context
    if task_result.error is not None:
        result.failures.append(f"{context.h5_path}: {task_result.error}")
        log(f"[FAIL] {context.h5_path.name}: {task_result.error}")
        advance_progress(pipeline_count)
        return

    assert task_result.value is not None
    result.processed_outputs.append(task_result.value)
    result.processed_input_paths.append(context.h5_path)
    outputs_by_context[context].append(task_result.value)
    inputs_by_context[context].append(context.h5_path)
    log(f"[OK] {context.h5_path.name}: combined results -> {task_result.value}")
    advance_progress(pipeline_count)


def _holo_summary(
    processed_outputs: Sequence[Path],
    *,
    input_count: int | None = None,
    failure_count: int = 0,
) -> str:
    if failure_count:
        return (
            f"Processed {len(processed_outputs)}/{input_count or 0} holo file(s); "
            f"{failure_count} failed/skipped."
        )
    if len(processed_outputs) == 1:
        return f"Output file: {processed_outputs[0]}"
    return f"Outputs generated for {len(processed_outputs)} holo file(s)."


def _run_workflow_postprocesses(
    *,
    pipeline_result: PipelineRunResult,
    postprocesses: Sequence[Any],
    selected_pipeline_names: Sequence[str],
    input_path: Path,
    output_dir: Path,
    run_postprocesses: RunPostprocesses,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    start_final_progress: Callable[[float, str], None],
    final_progress_units: int,
    zip_outputs: bool,
) -> None:
    if final_progress_units:
        final_status = "Running postprocess..." if postprocesses else "Creating ZIP..."
        start_final_progress(final_progress_units, final_status)

    postprocess_started_at = time.monotonic() if postprocesses else None
    if postprocesses and pipeline_result.processed_outputs:
        run_postprocesses(
            postprocesses,
            output_dir,
            pipeline_result.processed_outputs,
            pipeline_result.processed_input_paths,
            input_path,
            selected_pipeline_names,
            pipeline_result.failures,
            zip_outputs=zip_outputs,
            record_timing=pipeline_result.timings.add,
        )
    elif postprocesses:
        log(
            "[POST SKIP] No successful pipeline outputs were generated, "
            "so postprocess steps were skipped."
        )
        advance_progress(len(postprocesses))
    if postprocess_started_at is not None:
        _add_timing(
            pipeline_result.timings,
            "postprocess phase total",
            time.monotonic() - postprocess_started_at,
        )
        _log_elapsed(log, "Postprocess phase", postprocess_started_at)


def _finalize_workflow_outputs(
    *,
    output_dir: Path,
    base_output_dir: Path,
    processed_outputs: Sequence[Path],
    zip_outputs: bool,
    zip_name: str,
    zip_output_dir: ZipOutputDir,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    set_status: Callable[[str], None],
    make_zip_progress_callback: Callable[[], ZipProgressCallback | None],
    on_zip_error: Callable[[str], None] | None,
    timings: TimingSamples | None = None,
) -> _FinalizedOutputs:
    finalize_started_at = time.monotonic()
    if zip_outputs:
        return _zip_workflow_outputs(
            output_dir=output_dir,
            base_output_dir=base_output_dir,
            zip_name=zip_name,
            zip_output_dir=zip_output_dir,
            log=log,
            advance_progress=advance_progress,
            set_status=set_status,
            make_zip_progress_callback=make_zip_progress_callback,
            on_zip_error=on_zip_error,
            finalize_started_at=finalize_started_at,
            timings=timings,
        )

    if timings is not None:
        _add_timing(
            timings,
            "workflow output finalization without ZIP",
            time.monotonic() - finalize_started_at,
        )
    if len(processed_outputs) == 1:
        return _FinalizedOutputs(summary_message=f"Output file: {processed_outputs[0]}")
    return _FinalizedOutputs(summary_message=f"Outputs stored under: {output_dir}")


def _zip_workflow_outputs(
    *,
    output_dir: Path,
    base_output_dir: Path,
    zip_name: str,
    zip_output_dir: ZipOutputDir,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    set_status: Callable[[str], None],
    make_zip_progress_callback: Callable[[], ZipProgressCallback | None],
    on_zip_error: Callable[[str], None] | None,
    finalize_started_at: float,
    timings: TimingSamples | None = None,
) -> _FinalizedOutputs:
    try:
        final_zip_name = _zip_filename(zip_name)
        set_status("Creating ZIP...")
        log("[ZIP] Preparing archive...")
        zip_started_at = time.monotonic()
        zip_path = zip_output_dir(
            output_dir,
            base_output_dir / final_zip_name,
            make_zip_progress_callback(),
        )
        if timings is not None:
            _add_timing(timings, "final ZIP creation", time.monotonic() - zip_started_at)
        copy_started_at = time.monotonic()
        companion_paths = copy_zip_companion_output_folders(
            output_dir,
            base_output_dir,
        )
        if timings is not None:
            _add_timing(
                timings,
                "companion output copy",
                time.monotonic() - copy_started_at,
            )
        log(f"[ZIP] Archive created: {zip_path}")
        if timings is not None:
            _add_timing(
                timings,
                "ZIP finalization total",
                time.monotonic() - finalize_started_at,
            )
        _log_elapsed(log, "ZIP finalization", finalize_started_at)
        return _FinalizedOutputs(
            summary_message=_zip_summary(zip_path, companion_paths),
            zip_path=zip_path,
        )
    except Exception as exc:  # noqa: BLE001
        zip_error = str(exc)
        advance_progress(1.0)
        log(f"[ZIP FAIL] {zip_error}")
        if timings is not None:
            _add_timing(
                timings,
                "ZIP finalization failed total",
                time.monotonic() - finalize_started_at,
            )
        _log_elapsed(log, "ZIP finalization", finalize_started_at)
        if on_zip_error is not None:
            on_zip_error(zip_error)
        return _FinalizedOutputs(
            summary_message=f"Outputs stored under: {output_dir}",
            zip_failed=True,
            zip_error=zip_error,
        )


def _zip_filename(zip_name: str) -> str:
    final_zip_name = zip_name.strip() or "outputs.zip"
    if not final_zip_name.lower().endswith(".zip"):
        final_zip_name += ".zip"
    return final_zip_name


def _zip_summary(zip_path: Path, companion_paths: Sequence[Path]) -> str:
    summary_parts = [f"ZIP archive: {zip_path}"]
    if companion_paths:
        companion_summary = ", ".join(str(path) for path in companion_paths)
        summary_parts.append(f"Companion outputs: {companion_summary}")
    return "; ".join(summary_parts)


def _log_elapsed(
    log: Callable[[str], None],
    label: str,
    started_at: float,
) -> None:
    log(f"[TIME] {label} completed in {_format_elapsed(time.monotonic() - started_at)}")


def _log_timing_averages(
    log: Callable[[str], None],
    timings: TimingSamples,
) -> None:
    if not timings:
        return

    samples_by_label = (
        timings.snapshot() if hasattr(timings, "snapshot") else dict(timings)
    )
    summary_items: list[str] = []
    for label in sorted(samples_by_label):
        samples = samples_by_label[label]
        if not samples:
            continue
        average = sum(samples) / len(samples)
        summary_items.append(
            f"{label} average {_format_elapsed(average)} over {len(samples)} sample(s)"
        )

    if summary_items:
        log(f"[TIMING] {'; '.join(summary_items)}")


def _add_timing(timings: TimingSamples, label: str, seconds: float) -> None:
    if hasattr(timings, "add"):
        timings.add(label, seconds)
        return
    timings.setdefault(label, []).append(seconds)


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def log_throttled_zip_progress(
    *,
    set_progress_units: Callable[[float], None],
    progress_base: float,
    log: Callable[[str], None],
    update_ui: Callable[[], None] | None = None,
) -> ZipProgressCallback:
    last_progress_log = 0.0

    def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
        nonlocal last_progress_log
        fraction = 1.0 if total == 0 else done / total
        set_progress_units(progress_base + fraction)
        now = time.monotonic()
        if done == total or (now - last_progress_log) >= 0.5:
            pct = 100 if total == 0 else int((done * 100) / total)
            log(f"[ZIP] {done}/{total} files ({pct}%)")
            last_progress_log = now
            if update_ui is not None:
                update_ui()

    return _zip_progress
