from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from input_output import ZipH5Member

from ._pipeline_runs import (
    PipelineRunResult,
    RunPipelineFile,
    run_filesystem_pipeline_run,
    run_zip_pipeline_run,
)
from ._zip_batches import ZipBatchSettings


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


RunPostprocesses = Callable[
    [
        Sequence[Any],
        Path,
        Sequence[Path],
        Sequence[Path],
        Path,
        Sequence[str],
        list[str],
    ],
    None,
]
ZipProgressCallback = Callable[[int, int, Path], None]
ZipOutputDir = Callable[[Path, Path | None, ZipProgressCallback | None], Path]
IdleCallback = Callable[[], None]
ZIP_COMPANION_OUTPUT_FOLDERS = ("png",)


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
) -> RunWorkflowResult:
    workspace = _prepare_workflow_workspace(
        base_output_dir=base_output_dir,
        zip_outputs=zip_outputs,
    )
    cleanup_workspace = False

    try:
        pipeline_result = run_filesystem_pipeline_run(
            inputs=inputs,
            data_root=data_root,
            pipelines=pipelines,
            output_dir=workspace.output_dir,
            output_filename=output_filename,
            run_pipeline_file=run_pipeline_file,
            relative_parent=relative_parent,
            log=log,
        )

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
        )
        cleanup_workspace = (
            workspace.uses_temporary_output_dir and not finalized_outputs.zip_failed
        )
        return _workflow_result(
            output_dir=workspace.output_dir,
            pipeline_result=pipeline_result,
            finalized_outputs=finalized_outputs,
        )
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
        )
        cleanup_workspace = (
            workspace.uses_temporary_output_dir and not finalized_outputs.zip_failed
        )
        return _workflow_result(
            output_dir=workspace.output_dir,
            pipeline_result=pipeline_result,
            finalized_outputs=finalized_outputs,
        )
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
) -> None:
    if final_progress_units:
        final_status = "Running postprocess..." if postprocesses else "Creating ZIP..."
        start_final_progress(final_progress_units, final_status)

    if postprocesses and pipeline_result.processed_outputs:
        run_postprocesses(
            postprocesses,
            output_dir,
            pipeline_result.processed_outputs,
            pipeline_result.processed_input_paths,
            input_path,
            selected_pipeline_names,
            pipeline_result.failures,
        )
    elif postprocesses:
        log(
            "[POST SKIP] No successful pipeline outputs were generated, "
            "so postprocess steps were skipped."
        )
        advance_progress(len(postprocesses))


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
) -> _FinalizedOutputs:
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
) -> _FinalizedOutputs:
    try:
        final_zip_name = _zip_filename(zip_name)
        set_status("Creating ZIP...")
        log("[ZIP] Preparing archive...")
        zip_path = zip_output_dir(
            output_dir,
            base_output_dir / final_zip_name,
            make_zip_progress_callback(),
        )
        companion_paths = copy_zip_companion_output_folders(
            output_dir,
            base_output_dir,
        )
        log(f"[ZIP] Archive created: {zip_path}")
        return _FinalizedOutputs(
            summary_message=_zip_summary(zip_path, companion_paths),
            zip_path=zip_path,
        )
    except Exception as exc:  # noqa: BLE001
        zip_error = str(exc)
        advance_progress(1.0)
        log(f"[ZIP FAIL] {zip_error}")
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
