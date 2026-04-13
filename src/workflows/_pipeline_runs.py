from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from input_output import ZipH5Member

from ._zip_batches import ZipBatchSettings, iter_extracted_zip_batches


@dataclass
class PipelineRunResult:
    processed_outputs: list[Path] = field(default_factory=list)
    processed_input_paths: list[Path] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


RunPipelineFile = Callable[
    [Path, Sequence[Any], Path, Path, str | None],
    Path,
]
ZipMemberPath = tuple[ZipH5Member, Path]


def run_filesystem_pipeline_run(
    *,
    inputs: Iterable[Path],
    data_root: Path,
    pipelines: Sequence[Any],
    output_dir: Path,
    output_filename: str | None,
    run_pipeline_file: RunPipelineFile,
    relative_parent: Callable[[Path, Path], Path],
    log: Callable[[str], None],
) -> PipelineRunResult:
    result = PipelineRunResult()
    for h5_path in inputs:
        try:
            combined_output = run_pipeline_file(
                h5_path,
                pipelines,
                output_dir,
                relative_parent(h5_path, data_root),
                output_filename,
            )
        except Exception as exc:  # noqa: BLE001
            _record_pipeline_failure(
                result,
                input_label=str(h5_path),
                log_label=h5_path.name,
                error=exc,
                log=log,
            )
            continue

        _record_pipeline_success(result, h5_path, combined_output)
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

        member_paths: list[ZipMemberPath] = list(
            zip(extracted_batch.members, extracted_batch.h5_paths, strict=True)
        )
        worker_count = min(settings.pipeline_workers, len(member_paths))
        if worker_count <= 1:
            _run_zip_pipeline_batch_sequential(
                member_paths=member_paths,
                pipelines=pipelines,
                output_dir=output_dir,
                run_pipeline_file=run_pipeline_file,
                result=result,
                log=log,
            )
            continue

        log(
            f"[ZIP] Running pipelines for batch {extracted_batch.index}/"
            f"{extracted_batch.count} with {worker_count} worker(s)..."
        )
        _run_zip_pipeline_batch_parallel(
            member_paths=member_paths,
            pipelines=pipelines,
            output_dir=output_dir,
            run_pipeline_file=run_pipeline_file,
            result=result,
            log=log,
            advance_progress=advance_progress,
            max_workers=worker_count,
            idle_callback=idle_callback,
        )
    return result


def _run_zip_pipeline_batch_sequential(
    *,
    member_paths: Sequence[ZipMemberPath],
    pipelines: Sequence[Any],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
    result: PipelineRunResult,
    log: Callable[[str], None],
) -> None:
    for member, h5_path in member_paths:
        try:
            combined_output = _run_zip_member_pipeline(
                member=member,
                h5_path=h5_path,
                pipelines=pipelines,
                output_dir=output_dir,
                run_pipeline_file=run_pipeline_file,
            )
        except Exception as exc:  # noqa: BLE001
            _record_pipeline_failure(
                result,
                input_label=member.name,
                log_label=member.name,
                error=exc,
                log=log,
            )
            continue

        _record_pipeline_success(result, h5_path, combined_output)


def _run_zip_pipeline_batch_parallel(
    *,
    member_paths: Sequence[ZipMemberPath],
    pipelines: Sequence[Any],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
    result: PipelineRunResult,
    log: Callable[[str], None],
    advance_progress: Callable[[float], None],
    max_workers: int,
    idle_callback: Callable[[], None] | None,
) -> None:
    worker_count = min(len(member_paths), max(1, max_workers))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_zip_member_pipeline,
                member=member,
                h5_path=h5_path,
                pipelines=pipelines,
                output_dir=output_dir,
                run_pipeline_file=run_pipeline_file,
            ): (member, h5_path)
            for member, h5_path in member_paths
        }
        pending = set(futures)
        while pending:
            done, pending = wait(
                pending,
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if idle_callback is not None:
                    idle_callback()
                continue
            for future in done:
                member, h5_path = futures[future]
                try:
                    combined_output = future.result()
                except Exception as exc:  # noqa: BLE001
                    _record_pipeline_failure(
                        result,
                        input_label=member.name,
                        log_label=member.name,
                        error=exc,
                        log=log,
                    )
                    advance_progress(len(pipelines))
                    continue

                _record_pipeline_success(result, h5_path, combined_output)
                log(f"[OK] {member.name}: combined results -> {combined_output}")
                advance_progress(len(pipelines))
            if idle_callback is not None:
                idle_callback()


def _run_zip_member_pipeline(
    *,
    member: ZipH5Member,
    h5_path: Path,
    pipelines: Sequence[Any],
    output_dir: Path,
    run_pipeline_file: RunPipelineFile,
) -> Path:
    return run_pipeline_file(
        h5_path,
        pipelines,
        output_dir,
        member.relative_path.parent,
        None,
    )


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
