from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from input_output import stage_analysis_file
from input_output.archive_io import (
    create_zip_from_tree,
    iter_extracted_h5_members,
    list_h5_members,
)
from input_output.hdf5_schema import find_pipeline_group
from input_output.inputs import prepare_run_inputs
from input_output.output_paths import ae_result_filename
from pipelines import PipelineContext, PipelineDescriptor


@dataclass
class BatchRunResult:
    output_dir: Path
    input_files: list[Path] = field(default_factory=list)
    generated_paths: list[Path] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    summary_message: str = ""
    zip_path: Path | None = None


def _metric_options(descriptor: PipelineDescriptor) -> tuple[tuple[str, ...], ...]:
    options = getattr(descriptor, "required_metric_options", ())
    if options:
        return tuple(tuple(group) for group in options)
    return tuple((name,) for name in getattr(descriptor, "required_metrics", ()))


def _compatible(path: Path, descriptor: PipelineDescriptor) -> bool:
    with h5py.File(path, "r") as h5file:
        return all(
            any(find_pipeline_group(h5file, name) is not None for name in group)
            for group in _metric_options(descriptor)
        )


def run_batch(
    input_paths: Sequence[str | Path],
    pipelines: Sequence[PipelineDescriptor],
    output_dir: str | Path,
    *,
    zip_outputs: bool = False,
    zip_name: str = "outputs.zip",
    processing_branch: str = "Processing",
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BatchRunResult:
    """Run each tool once over compact analysis copies, preserving source files.

    Group folders are preserved under ``h5/`` for cohort labels. Only exported
    metrics and the small signals needed by reports are copied into results.
    """
    if not input_paths:
        raise ValueError("Select EyeFlow HDF5 files, a folder, or a ZIP archive.")
    if not pipelines:
        raise ValueError("Select at least one statistical pipeline.")
    if processing_branch not in {"Processing", "ProcessingAlt"}:
        raise ValueError("Processing branch must be Processing or ProcessingAlt.")
    if zip_outputs and (Path(zip_name).name != zip_name or not zip_name.strip()):
        raise ValueError("ZIP name must be a filename without directory components.")
    plan = prepare_run_inputs(input_paths)
    root = Path(output_dir).expanduser().resolve()
    source_root = plan.input_path.resolve()
    if zip_outputs:
        archive_target = root / (
            zip_name if zip_name.lower().endswith(".zip") else f"{zip_name}.zip"
        )
        if plan.is_zip and (
            archive_target.resolve() == source_root
            or (archive_target.exists() and archive_target.samefile(source_root))
        ):
            raise ValueError("The output ZIP must differ from the input archive.")
    if plan.kind == "folder" and root == source_root:
        raise ValueError("Choose an output folder separate from the input folder.")
    if plan.is_zip:
        members = list_h5_members(plan.input_path)
        input_count = len(members)
    else:
        sources = list(dict.fromkeys(path.resolve() for path in plan.h5_paths))
        sources = [path for path in sources if not path.is_relative_to(root)]
        input_count = len(sources)
    if not input_count:
        raise ValueError("No EyeFlow HDF5 files found in the selected input.")
    root.mkdir(parents=True, exist_ok=True)
    result = BatchRunResult(output_dir=root)
    staged: list[Path] = []
    total = input_count + len(pipelines)
    completed = 0
    report_entries: list[dict] = []
    reserved: set[Path] = set()

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    def advance(message: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, total, message)

    def stage(source: Path, relative: Path, provenance: Path) -> None:
        destination = root / "h5" / relative.parent / ae_result_filename(relative)
        if destination in reserved:
            destination = destination.with_name(
                f"{destination.stem}_{len(reserved)}.h5"
            )
        reserved.add(destination)
        try:
            stage_analysis_file(
                source, destination, processing_branch=processing_branch
            )
            with h5py.File(destination, "a") as analysis:
                analysis.attrs["angioeye_input_file"] = str(provenance)
            staged.append(destination)
            result.input_files.append(provenance)
            emit(f"[INPUT] {provenance} -> {destination.relative_to(root)}")
        except Exception as exc:
            failure = f"Input '{provenance}': {type(exc).__name__}: {exc}"
            result.failures.append(failure)
            emit(f"[INPUT FAIL] {failure}")
        finally:
            advance(f"Read {provenance.name}")

    if progress is not None:
        progress(0, total, "Reading EyeFlow metrics...")
    if plan.is_zip:
        for extracted in iter_extracted_h5_members(plan.input_path, members):
            stage(
                extracted.path,
                extracted.member.relative_path,
                plan.input_path / extracted.member.relative_path,
            )
    else:
        if source_root.is_file():
            source_root = source_root.parent
        if len(input_paths) > 1:
            source_root = Path(
                os.path.commonpath([str(path.parent) for path in sources])
            )
        for source in sources:
            stage(source, source.relative_to(source_root), source)

    mode = (
        "zip_batch"
        if plan.is_zip
        else (
            "cohort_batch"
            if plan.kind == "folder"
            else ("single_file" if input_count == 1 else "file_batch")
        )
    )
    for descriptor in pipelines:
        entry = {"name": descriptor.name, "status": "failed"}
        started = time.monotonic()
        try:
            if not descriptor.available:
                reason = descriptor.error_msg or ", ".join(descriptor.missing_deps)
                raise ValueError(f"Pipeline unavailable: {reason}")
            if mode not in getattr(descriptor, "input_methods", (mode,)):
                raise ValueError(f"This pipeline does not support {mode} inputs.")
            accepted_modes = getattr(descriptor, "accepted_input_modes", ())
            if accepted_modes and plan.kind not in accepted_modes:
                raise ValueError(
                    f"This pipeline requires {' or '.join(accepted_modes)} inputs."
                )
            indices = [
                index
                for index, path in enumerate(staged)
                if _compatible(path, descriptor)
            ]
            compatible = tuple(staged[index] for index in indices)
            minimum = max(1, getattr(descriptor, "minimum_input_files", 1))
            if len(compatible) < minimum:
                requirements = " and ".join(
                    " or ".join(group) for group in _metric_options(descriptor)
                )
                raise ValueError(
                    f"Requires {minimum} compatible EyeFlow file(s); found {len(compatible)}."
                    + (
                        f" Required exported metrics: {requirements}."
                        if requirements
                        else ""
                    )
                )
            if len(compatible) < len(staged):
                emit(
                    f"[SKIP] {descriptor.name}: {len(staged) - len(compatible)} file(s) lack required metrics."
                )
            context = PipelineContext(
                output_dir=root,
                processed_files=compatible,
                selected_pipelines=tuple(item.name for item in pipelines),
                input_path=plan.input_path,
                zip_outputs=zip_outputs,
                input_h5_paths=tuple(result.input_files[index] for index in indices),
                processing_branch=processing_branch,
            )
            emit(
                f"[PIPELINE] Running {descriptor.name} over {len(compatible)} file(s)..."
            )
            pipeline_result = descriptor.instantiate().run(context)
            paths = [Path(path) for path in pipeline_result.generated_paths]
            result.generated_paths.extend(paths)
            warnings = pipeline_result.metadata.get("failures", [])
            if isinstance(warnings, str):
                warnings = [warnings]
            if warnings:
                result.failures.extend(str(warning) for warning in warnings)
                for warning in warnings:
                    emit(f"[WARN] {descriptor.name}: {warning}")
            entry.update(
                status=("completed_with_errors" if paths else "failed")
                if warnings
                else "completed",
                summary=pipeline_result.summary,
                generated_paths=[str(path) for path in paths],
            )
            emit(f"[OK] {descriptor.name}: {pipeline_result.summary}")
        except Exception as exc:
            failure = f"Pipeline '{descriptor.name}': {type(exc).__name__}: {exc}"
            result.failures.append(failure)
            entry["error"] = failure
            emit(f"[FAIL] {failure}")
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started
            report_entries.append(entry)
            advance(descriptor.name)
    succeeded = sum(entry["status"] == "completed" for entry in report_entries)
    result.summary_message = (
        f"Analyzed {len(staged)}/{input_count} EyeFlow file(s); "
        f"{succeeded}/{len(pipelines)} pipeline(s) completed; "
        f"{len(result.failures)} failure(s)."
    )
    manifest = root / "batch_run.json"
    manifest.write_text(
        json.dumps(
            {
                "processing_branch": processing_branch,
                "input_files": [str(path) for path in result.input_files],
                "analysis_files": [str(path.relative_to(root)) for path in staged],
                "pipelines": report_entries,
                "failures": result.failures,
                "summary": result.summary_message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result.generated_paths = list(
        dict.fromkeys([*staged, *result.generated_paths, manifest])
    )
    if zip_outputs:
        archive = root / (
            zip_name if zip_name.lower().endswith(".zip") else f"{zip_name}.zip"
        )
        files = [path for path in root.rglob("*") if path.is_file() and path != archive]
        try:
            result.zip_path = create_zip_from_tree(root, archive, source_paths=files)
            result.generated_paths.append(result.zip_path)
            emit(f"[ZIP] {result.zip_path}")
        except Exception as exc:
            failure = f"Output ZIP: {type(exc).__name__}: {exc}"
            result.failures.append(failure)
            emit(f"[FAIL] {failure}")
            result.summary_message = (
                f"Analyzed {len(staged)}/{input_count} EyeFlow file(s); "
                f"{succeeded}/{len(pipelines)} pipeline(s) completed; "
                f"{len(result.failures)} failure(s)."
            )
            report = json.loads(manifest.read_text(encoding="utf-8"))
            report.update(failures=result.failures, summary=result.summary_message)
            manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit(result.summary_message)
    return result
