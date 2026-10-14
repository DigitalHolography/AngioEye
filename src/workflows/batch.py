from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from input_output import stage_analysis_file
from input_output.archive_io import create_zip_from_tree
from input_output.hdf5_schema import find_pipeline_group
from input_output.inputs import prepare_run_inputs
from input_output.output_paths import (
    acquisition_output_dir,
    cohort_results_dir,
    h5_output_dir,
)
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
    output_dir: str | Path | None = None,
    *,
    zip_outputs: bool = False,
    zip_name: str = "outputs.zip",
    processing_branch: str = "Processing",
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BatchRunResult:
    """Run each tool once over compact analysis copies, preserving source files.

    ``output_dir`` is the parent of the mirrored acquisition trees; by default
    results live beside the selected holograms. Each acquisition owns
    ``<stem>/<stem>_AE/``. Only exported metrics and the small signals needed by
    reports are copied into results.
    """
    if not input_paths:
        raise ValueError("Select .holo files or one folder containing .holo files.")
    if not pipelines:
        raise ValueError("Select at least one statistical pipeline.")
    if processing_branch not in {"Processing", "ProcessingAlt"}:
        raise ValueError("Processing branch must be Processing or ProcessingAlt.")
    if zip_outputs and (Path(zip_name).name != zip_name or not zip_name.strip()):
        raise ValueError("ZIP name must be a filename without directory components.")
    plan = prepare_run_inputs(input_paths)
    source_root = plan.input_path.resolve()
    if source_root.is_file():
        source_root = source_root.parent
    output_parent = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_root
    )
    input_count = plan.item_count
    if not input_count:
        raise ValueError("No EyeFlow HDF5 files found in the selected input.")
    acquisition_dirs = tuple(
        acquisition_output_dir(
            holo,
            output_parent=output_parent,
            relative_parent=holo.parent.relative_to(source_root),
        ).resolve()
        for holo in plan.holo_paths
    )
    batch_root = (
        acquisition_dirs[0]
        if plan.kind == "file" and input_count == 1
        else output_parent
    )
    root = (
        batch_root
        if plan.kind == "file" and input_count == 1
        else cohort_results_dir(output_parent)
    ).resolve()
    source_paths = (*plan.selected_holo_paths, *plan.h5_paths)
    if any(
        path.resolve().is_relative_to(output_root)
        for path in source_paths
        for output_root in (*acquisition_dirs, root)
    ):
        raise ValueError(
            f"The AngioEye report folder contains selected inputs: {root}. "
            "Choose a different output parent folder."
        )
    root.mkdir(parents=True, exist_ok=True)
    result = BatchRunResult(output_dir=root, failures=list(plan.input_failures))
    staged: list[Path] = []
    total = input_count + len(pipelines)
    completed = 0
    report_entries: list[dict] = []

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    def advance(message: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, total, message)

    def stage(source: Path, app_dir: Path, holo: Path) -> None:
        destination = h5_output_dir(app_dir) / f"{holo.stem}_AE.h5"
        try:
            stage_analysis_file(
                source, destination, processing_branch=processing_branch
            )
            with h5py.File(destination, "a") as analysis:
                analysis.attrs["angioeye_input_file"] = str(source)
                analysis.attrs["angioeye_input_holo"] = str(holo)
            staged.append(destination)
            result.input_files.append(source)
            emit(f"[INPUT] {holo} -> {source} -> {destination.relative_to(batch_root)}")
        except Exception as exc:
            failure = f"Input '{holo}' ({source}): {type(exc).__name__}: {exc}"
            result.failures.append(failure)
            emit(f"[INPUT FAIL] {failure}")
        finally:
            advance(f"Read {holo.name}")

    if progress is not None:
        progress(0, total, "Reading EyeFlow metrics...")
    for failure in plan.input_failures:
        emit(f"[INPUT FAIL] {failure}")
        advance("Missing EyeFlow result")
    for source, holo, app_dir in zip(
        plan.h5_paths, plan.holo_paths, acquisition_dirs, strict=True
    ):
        source = source.resolve()
        holo = holo.resolve()
        stage(source, app_dir, holo)

    mode = (
        "cohort_batch"
        if plan.kind == "folder"
        else ("single_file" if input_count == 1 else "file_batch")
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
                analysis_root=batch_root,
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
                "input_holo_files": [
                    str(path.resolve()) for path in plan.selected_holo_paths
                ],
                "input_files": [str(path) for path in result.input_files],
                "analysis_files": [
                    str(path.relative_to(batch_root)) for path in staged
                ],
                "analysis_root": str(batch_root),
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
        files = list(
            dict.fromkeys(path for path in result.generated_paths if path.is_file())
        )
        try:
            result.zip_path = create_zip_from_tree(
                batch_root, archive, source_paths=files
            )
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
