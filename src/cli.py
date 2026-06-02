"""
Command-line interface to run AngioEye pipelines over a collection of HDF5 files.

Usage example:
    python cli.py --data data/ --pipelines pipelines.txt --postprocess postprocess.txt --output ./results --zip --zip-name my_run.zip

Inputs:
    --data / -d        Path to a directory (recursively scanned), a single .h5/.hdf5 file, or a .zip archive of .h5 files.
    --pipelines / -p   Text file listing pipeline names (one per line, '#' and blank lines ignored).
    --postprocess      Optional text file listing postprocess names (one per line, '#' and blank lines ignored).
    --output / -o      Base directory where results will be written (input subfolder layout is preserved).
    --trim-source / -t When set, source HDF5 contents will not be copied into pipeline output files (reducing output size, but losing provenance).
    --zip / -z         When set, compress the outputs into a .zip archive after completion.
                       Companion report folders such as png/ are kept next to it.
    --zip-name         Optional filename for the archive (default: outputs.zip).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from input_output import (
    create_zip_from_tree,
)
from pipelines import (
    PipelineDescriptor,
    load_pipeline_catalog,
)
from postprocess import PostprocessDescriptor, load_postprocess_catalog
from workflows import (
    ZIP_COMPANION_OUTPUT_FOLDERS,
    WorkflowCallbacks,
    WorkflowInputError,
    WorkflowRunRequest,
    ZipBatchSettings,
    dispatch_workflow,
    missing_required_pipeline_errors,
    prepare_run_input,
)


def _build_pipeline_registry() -> dict[str, PipelineDescriptor]:
    available, _ = load_pipeline_catalog()
    # pipelines = load_all_pipelines()
    return {p.name: p for p in available}


def _build_postprocess_registry() -> dict[str, PostprocessDescriptor]:
    available, _ = load_postprocess_catalog()
    return {p.name: p for p in available}


def _load_pipeline_list(
    path: Path, registry: dict[str, PipelineDescriptor]
) -> list[PipelineDescriptor]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    selected: list[PipelineDescriptor] = []
    missing: list[str] = []
    for line in raw_lines:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        pipeline = registry.get(name)
        if pipeline is None:
            missing.append(name)
        else:
            selected.append(pipeline)
    if missing:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unknown pipeline(s): {', '.join(missing)}. Available: {available}"
        )
    if not selected:
        raise ValueError(
            "No pipelines selected (file is empty or only contains comments)."
        )
    return selected


def _load_postprocess_list(
    path: Path, registry: dict[str, PostprocessDescriptor]
) -> list[PostprocessDescriptor]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    selected: list[PostprocessDescriptor] = []
    missing: list[str] = []
    for line in raw_lines:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        postprocess = registry.get(name)
        if postprocess is None:
            missing.append(name)
        else:
            selected.append(postprocess)
    if missing:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unknown postprocess step(s): {', '.join(missing)}. Available: {available}"
        )
    return selected


def _validate_postprocess_selection(
    postprocesses: Sequence[PostprocessDescriptor],
    selected_pipeline_names: Sequence[str],
    reusable_h5_paths: Sequence[Path] = (),
    defer_when_no_reusable_paths: bool = False,
) -> None:
    errors = missing_required_pipeline_errors(
        postprocesses=postprocesses,
        selected_pipeline_names=selected_pipeline_names,
        reusable_h5_paths=reusable_h5_paths,
        defer_when_no_reusable_paths=defer_when_no_reusable_paths,
    )
    if errors:
        raise ValueError("\n".join(errors))


def _zip_output_dir(
    folder: Path,
    target_path: Path | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {folder}")
    if target_path is None:
        zip_name = f"{folder.name}_outputs.zip" if folder.name else "outputs.zip"
        zip_path = folder.parent / zip_name
    else:
        zip_path = target_path.expanduser().resolve()
    if zip_path.exists():
        zip_path.unlink()
    return create_zip_from_tree(
        folder,
        zip_path,
        exclude_root_dirs=ZIP_COMPANION_OUTPUT_FOLDERS,
        compresslevel=1,
        progress_callback=progress_callback,
    )


def run_cli(
    data_path: Path,
    pipelines_file: Path,
    postprocess_file: Path | None,
    output_dir: Path,
    trim_source: bool = False,
    zip_outputs: bool = False,
    zip_name: str | None = None,
) -> int:
    registry = _build_pipeline_registry()
    pipelines = _load_pipeline_list(pipelines_file, registry)
    postprocess_registry = _build_postprocess_registry()
    postprocesses = (
        _load_postprocess_list(postprocess_file, postprocess_registry)
        if postprocess_file is not None
        else []
    )
    input_plan = prepare_run_input(data_path)
    _validate_postprocess_selection(
        postprocesses,
        selected_pipeline_names=[pipeline.name for pipeline in pipelines],
        reusable_h5_paths=() if input_plan.is_zip else input_plan.h5_paths,
        defer_when_no_reusable_paths=input_plan.is_zip,
    )
    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        dispatch_result = dispatch_workflow(
            WorkflowRunRequest(
                mode=input_plan.kind,
                input_plan=input_plan,
                pipelines=pipelines,
                postprocesses=postprocesses,
                selected_pipeline_names=[pipeline.name for pipeline in pipelines],
                base_output_dir=output_root,
                zip_outputs=zip_outputs,
                zip_name=zip_name or "outputs.zip",
                trim_source=trim_source,
                zip_output_dir=_zip_output_dir,
                zip_batch_settings=ZipBatchSettings.from_env(),
            ),
            _cli_workflow_callbacks(),
        )
    except WorkflowInputError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1

    workflow_result = dispatch_result.workflow_result
    if workflow_result is None:
        print("No outputs generated.", file=sys.stderr)
        return 1

    print(f"Completed. {workflow_result.summary_message}")
    if workflow_result.failures:
        print(f"{len(workflow_result.failures)} failure(s):", file=sys.stderr)
        for msg in workflow_result.failures:
            print(f" - {msg}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AngioEye pipelines over a folder of HDF5 files."
    )
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=Path,
        help="Directory containing .h5/.hdf5 files (scanned recursively), a single .h5/.hdf5 file, or a .zip archive.",
    )
    parser.add_argument(
        "-p",
        "--pipelines",
        required=True,
        type=Path,
        help="Text file with pipeline names to run (one per line, '#' and blank lines ignored).",
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=None,
        help="Optional text file with postprocess names to run after pipelines.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Base output directory. Input subfolder layout is preserved for output files.",
    )
    parser.add_argument(
        "-t",
        "--trim-source",
        action="store_true",
        help="When set, source HDF5 contents will not be copied into pipeline output files (reducing output size, but losing provenance).",
    )
    parser.add_argument(
        "-z",
        "--zip",
        action="store_true",
        help=(
            "Zip the outputs after processing, keeping companion report folders "
            "such as png/ next to the archive."
        ),
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="outputs.zip",
        help="Archive filename to place inside the output directory (default: outputs.zip).",
    )
    args = parser.parse_args(argv)

    try:
        return run_cli(
            args.data,
            args.pipelines,
            args.postprocess,
            args.output,
            trim_source=args.trim_source,
            zip_outputs=args.zip,
            zip_name=args.zip_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _log_cli(message: str) -> None:
    if message.startswith("[POST FAIL]") or message.startswith("[POST WARN]"):
        print(message, file=sys.stderr)
    else:
        print(message)


def _cli_workflow_callbacks() -> WorkflowCallbacks:
    return WorkflowCallbacks(
        log=_log_cli,
        start_primary_progress=lambda _units, _status: None,
        start_final_progress=lambda _units, _status: None,
        advance_progress=lambda _units=1.0: None,
        set_progress_units=lambda _units: None,
        set_status=lambda _status: None,
        make_zip_progress_callback=_make_cli_zip_progress_callback,
    )


def _make_cli_zip_progress_callback():
    last_progress_log = 0.0

    def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
        nonlocal last_progress_log
        now = time.monotonic()
        if done == total or (now - last_progress_log) >= 0.5:
            pct = 100 if total == 0 else int((done * 100) / total)
            print(f"[ZIP] {done}/{total} files ({pct}%)")
            last_progress_log = now

    return _zip_progress


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


if __name__ == "__main__":
    raise SystemExit(main())

