from __future__ import annotations

import inspect
import functools
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from batch_engine import (
    BatchGroupResult,
    BatchExecutionSettings,
    batch_count,
    can_pickle,
    iter_batches,
    run_indexed_threaded_batches_in_process_pool_bounded,
    run_task_batch,
    run_threaded_batches_in_process_pool,
)
from input_output import ZipH5Member
from pipelines import load_pipeline_catalog
from pipeline_engine import OutputPathAllocator

from ._zip_batches import (
    ExtractedZipBatch,
    ZipBatchSettings,
    iter_extracted_zip_batches,
    streamed_extracted_zip_batches,
)
from .timing import TimingRecorder

_PIPELINE_RESOLVE_LOCK = threading.Lock()


@dataclass
class PipelineRunResult:
    processed_outputs: list[Path] = field(default_factory=list)
    processed_input_paths: list[Path] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    timings: TimingRecorder = field(default_factory=TimingRecorder)


RunPipelineFile = Callable[
    [Path, Sequence[Any], Path, Path, str | None],
    Path,
]
ZipMemberPath = tuple[ZipH5Member, Path]


@dataclass(frozen=True)
class PipelineFileJob:
    h5_path: Path
    output_relative_parent: Path
    output_filename: str | None
    input_label: str
    log_label: str


def run_filesystem_pipeline_run(
    *,
    inputs: Iterable[Path],
    data_root: Path,
    pipelines: Sequence[Any],
    output_dir: Path,
    output_filename: str | None,
    settings: BatchExecutionSettings,
    run_pipeline_file: RunPipelineFile,
    relative_parent: Callable[[Path, Path], Path],
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    idle_callback: Callable[[], None] | None = None,
) -> PipelineRunResult:
    result = PipelineRunResult()
    planning_started_at = time.monotonic()
    jobs = [
        PipelineFileJob(
            h5_path=h5_path,
            output_relative_parent=relative_parent(h5_path, data_root),
            output_filename=output_filename,
            input_label=str(h5_path),
            log_label=h5_path.name,
        )
        for h5_path in inputs
    ]
    result.timings.add(
        "filesystem pipeline run: discover/build file jobs",
        time.monotonic() - planning_started_at,
    )
    pipeline_names = _pipeline_names(pipelines)
    use_process_pool = settings.process_workers > 1 and (
        _can_run_pipeline_batches_in_process_pool(
            run_pipeline_file=run_pipeline_file,
            pipeline_names=pipeline_names,
        )
    )
    if use_process_pool:
        reserve_started_at = time.monotonic()
        jobs = _reserve_pipeline_job_output_names(jobs, output_dir)
        result.timings.add(
            "filesystem pipeline run: reserve process-safe output filenames",
            time.monotonic() - reserve_started_at,
        )
    batch_count_value = batch_count(len(jobs), settings.batch_size)
    job_batches = list(iter_batches(jobs, settings.batch_size))
    if use_process_pool:
        _run_threaded_pipeline_batches_in_process_pool(
            job_batches=job_batches,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            run_pipeline_file=run_pipeline_file,
            result=result,
            log=log,
            advance_progress=advance_progress,
            process_workers=settings.process_workers,
            thread_workers=settings.batch_size,
            idle_callback=idle_callback,
            pool_timing_label="filesystem pipeline run: process pool wall time",
            batch_timing_label="filesystem pipeline run: process batch wall time",
        )
        _log_pipeline_run_summary(result, total_files=len(jobs), log=log)
        return result

    if settings.process_workers > 1:
        log(
            "[BATCH WARN] Process pool disabled because the configured file runner "
            "or pipeline descriptors cannot be pickled; using threads only."
        )

    for batch_index, job_batch in enumerate(
        job_batches,
        start=1,
    ):
        log(
            f"[BATCH] Running batch {batch_index}/{batch_count_value} "
            f"({len(job_batch)} file(s)) with "
            f"{min(settings.batch_size, len(job_batch))} worker(s)..."
        )
        batch_started_at = time.monotonic()
        _run_pipeline_job_batch(
            jobs=job_batch,
            pipelines=pipelines,
            output_dir=output_dir,
            run_pipeline_file=run_pipeline_file,
            result=result,
            log=log,
            advance_progress=advance_progress,
            max_workers=settings.batch_size,
            idle_callback=idle_callback,
            timings=result.timings,
        )
        batch_elapsed = time.monotonic() - batch_started_at
        result.timings.add(
            "filesystem pipeline run: one job batch wall time",
            batch_elapsed,
        )
        log(
            f"[BATCH OK] Batch {batch_index}/{batch_count_value} "
            f"finished in {_format_elapsed(batch_elapsed)}."
        )
    _log_pipeline_run_summary(result, total_files=len(jobs), log=log)
    return result


def _reserve_pipeline_job_output_names(
    jobs: Sequence[PipelineFileJob],
    output_dir: Path,
) -> list[PipelineFileJob]:
    allocator = OutputPathAllocator()
    reserved_jobs: list[PipelineFileJob] = []
    for job in jobs:
        output_path = allocator.reserve(
            h5_path=job.h5_path,
            output_root=output_dir,
            output_relative_parent=job.output_relative_parent,
            output_filename=job.output_filename,
        )
        reserved_jobs.append(replace(job, output_filename=output_path.name))
    return reserved_jobs


def _pipeline_names(pipelines: Sequence[Any]) -> tuple[str, ...]:
    return tuple(getattr(pipeline, "name", str(pipeline)) for pipeline in pipelines)


@functools.cache
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


def _can_run_pipeline_batches_in_process_pool(
    *,
    run_pipeline_file: RunPipelineFile,
    pipeline_names: Sequence[str],
) -> bool:
    return can_pickle(run_pipeline_file, tuple(pipeline_names))


def _run_threaded_pipeline_batches_in_process_pool(
    *,
    job_batches: Sequence[Sequence[PipelineFileJob]],
    pipeline_names: Sequence[str],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    process_workers: int,
    thread_workers: int,
    idle_callback: Callable[[], None] | None,
    pool_timing_label: str,
    batch_timing_label: str,
) -> None:
    if not job_batches:
        return

    process_count = min(len(job_batches), max(1, process_workers))
    thread_count = max(1, thread_workers)
    run_job = functools.partial(
        _run_pipeline_job_by_name,
        pipeline_names=tuple(pipeline_names),
        output_dir=output_dir,
        run_pipeline_file=run_pipeline_file,
    )
    log(
        f"[PROCESS] Starting ProcessPoolExecutor(max_workers={process_count}) "
        f"for {len(job_batches)} batch(es); each process uses "
        f"ThreadPoolExecutor(max_workers={thread_count})."
    )
    for batch_index, job_batch in enumerate(job_batches, start=1):
        log(
            f"[PROCESS] Queued batch {batch_index}/{len(job_batches)} "
            f"({len(job_batch)} file(s))."
        )
    pool_started_at = time.monotonic()
    for batch_result in run_threaded_batches_in_process_pool(
        job_batches,
        run_item=run_job,
        process_workers=process_count,
        thread_workers=thread_count,
        idle_callback=idle_callback,
    ):
        _record_process_pool_batch_result(
            batch_result=batch_result,
            jobs=job_batches[batch_result.index - 1],
            result=result,
            log=log,
            advance_progress=advance_progress,
            pipeline_count=len(pipeline_names),
            batch_timing_label=batch_timing_label,
        )

    result.timings.add(
        pool_timing_label,
        time.monotonic() - pool_started_at,
    )
    log(f"[PROCESS] Process pool completed {len(job_batches)} batch(es).")


def _record_process_pool_batch_result(
    *,
    batch_result: BatchGroupResult[PipelineFileJob, Path],
    jobs: Sequence[PipelineFileJob],
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    pipeline_count: int,
    batch_timing_label: str,
) -> None:
    if batch_result.error is not None:
        error_message = _format_exception(batch_result.error)
        log(
            f"[BATCH FAIL] Batch {batch_result.index}/"
            f"{batch_result.count}: {error_message}"
        )
        for job in jobs:
            _record_pipeline_failure_message(
                result,
                input_label=job.input_label,
                log_label=job.log_label,
                error_message=f"Batch process failed: {error_message}",
                log=log,
            )
            advance_progress(pipeline_count)
        return

    log(
        f"[BATCH OK] Batch {batch_result.index}/{batch_result.count} "
        f"finished in {_format_elapsed(batch_result.elapsed_seconds)} "
        f"(process {batch_result.process_id})."
    )
    result.timings.add(
        batch_timing_label,
        batch_result.elapsed_seconds,
    )

    for task_result in batch_result.results:
        job = task_result.item
        if task_result.error is None:
            assert task_result.value is not None
            _record_pipeline_success(result, job.h5_path, task_result.value)
            log(
                f"[OK] Batch {batch_result.index}/{batch_result.count} "
                f"{job.log_label}: combined results -> {task_result.value}"
            )
        else:
            _record_pipeline_failure(
                result,
                input_label=job.input_label,
                log_label=job.log_label,
                error=task_result.error,
                log=log,
            )
        advance_progress(pipeline_count)


def run_zip_pipeline_run(
    *,
    zip_path: Path,
    members: Iterable[ZipH5Member],
    member_count: int,
    pipelines: Sequence[Any],
    output_dir: Path,
    settings: ZipBatchSettings,
    run_pipeline_file: RunPipelineFile,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    idle_callback: Callable[[], None] | None = None,
) -> PipelineRunResult:
    result = PipelineRunResult()
    pipeline_names = _pipeline_names(pipelines)
    use_process_pool = settings.process_workers > 1 and (
        _can_run_pipeline_batches_in_process_pool(
            run_pipeline_file=run_pipeline_file,
            pipeline_names=pipeline_names,
        )
    )
    if use_process_pool:
        _run_threaded_zip_batches_in_process_pool(
            zip_path=zip_path,
            members=members,
            member_count=member_count,
            pipelines=pipelines,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            settings=settings,
            run_pipeline_file=run_pipeline_file,
            result=result,
            log=log,
            advance_progress=advance_progress,
            idle_callback=idle_callback,
        )
        _log_pipeline_run_summary(result, total_files=member_count, log=log)
        return result

    if settings.process_workers > 1:
        log(
            "[ZIP WARN] Process pool disabled because the configured file runner "
            "or pipeline descriptors cannot be pickled; using threads inside each "
            "ZIP batch."
        )

    for extracted_batch in iter_extracted_zip_batches(
        zip_path,
        members,
        member_count=member_count,
        settings=settings,
        timings=result.timings,
    ):
        log(
            f"[ZIP] Extracting batch {extracted_batch.index}/"
            f"{extracted_batch.count} ({len(extracted_batch.members)} file(s))..."
        )
        if extracted_batch.error is not None:
            _record_zip_extraction_failure(
                result=result,
                zip_path=zip_path,
                batch_index=extracted_batch.index,
                member_count=len(extracted_batch.members),
                pipeline_count=len(pipelines),
                error=extracted_batch.error,
                log=log,
                advance_progress=advance_progress,
            )
            continue

        planning_started_at = time.monotonic()
        member_paths: list[ZipMemberPath] = list(
            zip(extracted_batch.members, extracted_batch.h5_paths, strict=True)
        )
        result.timings.add(
            "source ZIP pipeline run: pair extracted members with temp HDF5 paths",
            time.monotonic() - planning_started_at,
        )
        worker_count = min(settings.batch_size, len(member_paths))
        log(
            f"[ZIP] Running pipelines for batch {extracted_batch.index}/"
            f"{extracted_batch.count} with {worker_count} worker(s)..."
        )
        job_build_started_at = time.monotonic()
        jobs = [
            PipelineFileJob(
                h5_path=h5_path,
                output_relative_parent=member.relative_path.parent,
                output_filename=None,
                input_label=member.name,
                log_label=member.name,
            )
            for member, h5_path in member_paths
        ]
        jobs = _reserve_pipeline_job_output_names(jobs, output_dir)
        result.timings.add(
            "source ZIP pipeline run: build pipeline jobs from extracted members",
            time.monotonic() - job_build_started_at,
        )
        batch_started_at = time.monotonic()
        _run_pipeline_job_batch(
            jobs=jobs,
            pipelines=pipelines,
            output_dir=output_dir,
            run_pipeline_file=run_pipeline_file,
            result=result,
            log=log,
            advance_progress=advance_progress,
            max_workers=worker_count,
            idle_callback=idle_callback,
            timings=result.timings,
        )
        batch_elapsed = time.monotonic() - batch_started_at
        result.timings.add(
            "source ZIP pipeline run: one extracted job batch wall time",
            batch_elapsed,
        )
        log(
            f"[ZIP OK] Batch {extracted_batch.index}/{extracted_batch.count} "
            f"finished in {_format_elapsed(batch_elapsed)}."
        )
    _log_pipeline_run_summary(result, total_files=member_count, log=log)
    return result


def _run_threaded_zip_batches_in_process_pool(
    *,
    zip_path: Path,
    members: Iterable[ZipH5Member],
    member_count: int,
    pipelines: Sequence[Any],
    pipeline_names: Sequence[str],
    output_dir: Path,
    settings: ZipBatchSettings,
    run_pipeline_file: RunPipelineFile,
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    idle_callback: Callable[[], None] | None,
) -> None:
    batch_count_value = batch_count(member_count, settings.batch_size)
    log(
        f"[ZIP] Streaming {member_count} file(s) through {batch_count_value} "
        f"process batch(es) of up to {settings.batch_size} file(s); "
        "keeping one extracted batch ready ahead..."
    )
    process_count = min(batch_count_value, max(1, settings.process_workers))
    thread_count = max(1, settings.batch_size)
    run_job = functools.partial(
        _run_pipeline_job_by_name,
        pipeline_names=tuple(pipeline_names),
        output_dir=output_dir,
        run_pipeline_file=run_pipeline_file,
    )
    jobs_by_index: dict[int, list[PipelineFileJob]] = {}
    roots_by_index: dict[int, Path] = {}
    pool_started_at = time.monotonic()
    log(
        f"[PROCESS] Starting ProcessPoolExecutor(max_workers={process_count}) "
        f"for {batch_count_value} streaming ZIP batch(es); each process uses "
        f"ThreadPoolExecutor(max_workers={thread_count})."
    )
    with streamed_extracted_zip_batches(
        zip_path,
        members,
        member_count=member_count,
        settings=settings,
        max_ready_batches=1,
        idle_callback=idle_callback,
        timings=result.timings,
    ) as extracted_stream:
        indexed_job_batches = _iter_zip_job_batches_from_stream(
            zip_path=zip_path,
            extracted_batches=extracted_stream,
            output_dir=output_dir,
            result=result,
            log=log,
            advance_progress=advance_progress,
            pipeline_count=len(pipelines),
            jobs_by_index=jobs_by_index,
            roots_by_index=roots_by_index,
        )
        try:
            for batch_result in run_indexed_threaded_batches_in_process_pool_bounded(
                indexed_job_batches,
                group_count=batch_count_value,
                run_item=run_job,
                process_workers=process_count,
                thread_workers=thread_count,
                max_pending_batches=process_count,
                idle_callback=idle_callback,
            ):
                jobs = jobs_by_index.pop(batch_result.index, [])
                _record_process_pool_batch_result(
                    batch_result=batch_result,
                    jobs=jobs,
                    result=result,
                    log=log,
                    advance_progress=advance_progress,
                    pipeline_count=len(pipeline_names),
                    batch_timing_label="source ZIP pipeline run: process batch wall time",
                )
                root = roots_by_index.pop(batch_result.index, None)
                if root is not None:
                    shutil.rmtree(root, ignore_errors=True)
        finally:
            for root in roots_by_index.values():
                shutil.rmtree(root, ignore_errors=True)
    result.timings.add(
        "source ZIP pipeline run: process pool wall time",
        time.monotonic() - pool_started_at,
    )
    log(f"[PROCESS] Process pool completed {batch_count_value} ZIP batch(es).")


def _iter_zip_job_batches_from_stream(
    *,
    zip_path: Path,
    extracted_batches: Iterable[ExtractedZipBatch],
    output_dir: Path,
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    pipeline_count: int,
    jobs_by_index: dict[int, list[PipelineFileJob]],
    roots_by_index: dict[int, Path],
) -> Iterable[tuple[int, list[PipelineFileJob]]]:
    for extracted_batch in extracted_batches:
        log(
            f"[ZIP] Extracted batch {extracted_batch.index}/"
            f"{extracted_batch.count} ({len(extracted_batch.members)} file(s))."
        )
        roots_by_index[extracted_batch.index] = extracted_batch.root
        if extracted_batch.error is not None:
            _record_zip_extraction_failure(
                result=result,
                zip_path=zip_path,
                batch_index=extracted_batch.index,
                member_count=len(extracted_batch.members),
                pipeline_count=pipeline_count,
                error=extracted_batch.error,
                log=log,
                advance_progress=advance_progress,
            )
            shutil.rmtree(extracted_batch.root, ignore_errors=True)
            roots_by_index.pop(extracted_batch.index, None)
            continue

        member_paths: list[ZipMemberPath] = list(
            zip(extracted_batch.members, extracted_batch.h5_paths, strict=True)
        )
        jobs = [
            PipelineFileJob(
                h5_path=h5_path,
                output_relative_parent=member.relative_path.parent,
                output_filename=None,
                input_label=member.name,
                log_label=member.name,
            )
            for member, h5_path in member_paths
        ]
        reserved_jobs = _reserve_pipeline_job_output_names(jobs, output_dir)
        jobs_by_index[extracted_batch.index] = reserved_jobs
        log(
            f"[PROCESS] Queued ZIP batch {extracted_batch.index}/"
            f"{extracted_batch.count} ({len(reserved_jobs)} file(s))."
        )
        yield extracted_batch.index, reserved_jobs


def _run_pipeline_job_batch(
    *,
    jobs: Sequence[PipelineFileJob],
    pipelines: Sequence[Any],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    max_workers: int,
    idle_callback: Callable[[], None] | None,
    timings: TimingRecorder | None = None,
) -> None:
    batch_started_at = time.monotonic()
    for task_result in run_task_batch(
        jobs,
        run_item=lambda job: _run_pipeline_job(
            job=job,
            pipelines=pipelines,
            output_dir=output_dir,
            run_pipeline_file=run_pipeline_file,
            timings=timings,
        ),
        max_workers=max_workers,
        idle_callback=idle_callback,
    ):
        result_handling_started_at = time.monotonic()
        job = task_result.item
        if task_result.error is not None:
            _record_pipeline_failure(
                result,
                input_label=job.input_label,
                log_label=job.log_label,
                error=task_result.error,
                log=log,
            )
            advance_progress(len(pipelines))
            if timings is not None:
                timings.add(
                    "pipeline job batch: handle failed job result",
                    time.monotonic() - result_handling_started_at,
                )
            continue

        assert task_result.value is not None
        _record_pipeline_success(result, job.h5_path, task_result.value)
        log(f"[OK] {job.log_label}: combined results -> {task_result.value}")
        advance_progress(len(pipelines))
        if timings is not None:
            timings.add(
                "pipeline job batch: handle successful job result",
                time.monotonic() - result_handling_started_at,
            )
    if timings is not None:
        timings.add(
            "pipeline job batch: executor drain plus result handling",
            time.monotonic() - batch_started_at,
        )


def _run_pipeline_job_by_name(
    job: PipelineFileJob,
    *,
    pipeline_names: tuple[str, ...],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
) -> Path:
    return _run_pipeline_job(
        job,
        pipelines=_pipeline_descriptors_by_name(pipeline_names),
        output_dir=output_dir,
        run_pipeline_file=run_pipeline_file,
        timings=None,
    )


def _run_pipeline_job(
    job: PipelineFileJob,
    *,
    pipelines: Sequence[Any],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
    timings: TimingRecorder | None = None,
) -> Path:
    compatibility_started_at = time.monotonic()
    accepts_record_timing = (
        timings is not None and _accepts_record_timing(run_pipeline_file)
    )
    if timings is not None:
        timings.add(
            "per-file pipeline runner: inspect record_timing support",
            time.monotonic() - compatibility_started_at,
        )
    started_at = time.monotonic()
    try:
        args = (
            job.h5_path,
            pipelines,
            output_dir,
            job.output_relative_parent,
            job.output_filename,
        )
        if accepts_record_timing:
            return run_pipeline_file(*args, record_timing=timings.add)
        return run_pipeline_file(*args)
    finally:
        if timings is not None:
            timings.add("per-file pipeline total", time.monotonic() - started_at)


def _accepts_record_timing(run_pipeline_file: RunPipelineFile) -> bool:
    try:
        parameters = inspect.signature(run_pipeline_file).parameters
    except (TypeError, ValueError):
        return False
    return "record_timing" in parameters


def _record_pipeline_success(
    result: PipelineRunResult,
    h5_path: Path,
    combined_output: Path,
) -> None:
    result.processed_outputs.append(combined_output)
    result.processed_input_paths.append(h5_path)


def _record_pipeline_failure(
    result: PipelineRunResult,
    *,
    input_label: str,
    log_label: str,
    error: Exception,
    log: Callable[[str], None],
) -> None:
    _record_pipeline_failure_message(
        result,
        input_label=input_label,
        log_label=log_label,
        error_message=str(error),
        log=log,
    )


def _record_pipeline_failure_message(
    result: PipelineRunResult,
    *,
    input_label: str,
    log_label: str,
    error_message: str,
    log: Callable[[str], None],
) -> None:
    result.failures.append(f"{input_label}: {error_message}")
    log(f"[FAIL] {log_label}: {error_message}")


def _format_exception(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def _record_zip_extraction_failure(
    *,
    result: PipelineRunResult,
    zip_path: Path,
    batch_index: int,
    member_count: int,
    pipeline_count: int,
    error: Exception,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
) -> None:
    result.failures.append(f"{zip_path}: batch {batch_index}: {error}")
    log(f"[ZIP FAIL] Batch {batch_index}: {error}")
    advance_progress(member_count * pipeline_count)


def _log_pipeline_run_summary(
    result: PipelineRunResult,
    *,
    total_files: int,
    log: Callable[[str], None],
) -> None:
    succeeded = len(result.processed_outputs)
    failed = len(result.failures)
    log(
        f"[SUMMARY] Pipeline files: {succeeded}/{total_files} succeeded, "
        f"{failed} failed."
    )
