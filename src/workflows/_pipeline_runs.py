from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from batch_engine import (
    BatchExecutionSettings,
    batch_count,
    iter_batches,
    run_task_batch,
)
from input_output import ZipH5Member

from ._zip_batches import ZipBatchSettings, iter_extracted_zip_batches
from .timing import TimingRecorder


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
    batch_count_value = batch_count(len(jobs), settings.batch_size)
    for batch_index, job_batch in enumerate(
        iter_batches(jobs, settings.batch_size),
        start=1,
    ):
        log(
            f"[BATCH] Running batch {batch_index}/{batch_count_value} "
            f"({len(job_batch)} file(s)) with "
            f"{min(settings.task_workers, len(job_batch))} worker(s)..."
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
            max_workers=settings.task_workers,
            idle_callback=idle_callback,
            timings=result.timings,
        )
        result.timings.add(
            "filesystem pipeline run: one job batch wall time",
            time.monotonic() - batch_started_at,
        )
    return result


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
        worker_count = min(settings.task_workers, len(member_paths))
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
        result.timings.add(
            "source ZIP pipeline run: one extracted job batch wall time",
            time.monotonic() - batch_started_at,
        )
    return result


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


def _run_pipeline_job(
    *,
    job: PipelineFileJob,
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
    result.failures.append(f"{input_label}: {error}")
    log(f"[FAIL] {log_label}: {error}")


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
