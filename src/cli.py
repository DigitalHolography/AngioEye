"""
Run AngioEye pipelines from the command line.

Usage example:
    python cli.py --data data/ --pipelines pipelines.txt \
        --postprocess postprocess.txt --output ./results --zip \
        --zip-name my_run.zip

Inputs:
    --data / -d        Path to a directory (recursively scanned), a single .h5/.hdf5
                       file,
                       a .zip archive of .h5 files, one or more .holo files, or a .txt
                       holo path list. May be repeated for multiple explicit inputs.
    --pipelines / -p   Text file listing pipeline names (one per line, '#' and blank
                       lines ignored).
    --pipeline         Pipeline name; may be repeated. Combined with --pipelines.
    --postprocess      Optional text file listing postprocess names (one per line, '#'
                       and blank lines ignored).
    --postprocess-step Postprocess name; may be repeated. Combined with --postprocess.
    --output / -o      Base directory where results will be written (input subfolder
                       layout is preserved).
    --trim-source / -t When set, source HDF5 contents will not be copied into pipeline
                       output files (reducing output size, but losing provenance).
    --zip / -z         When set, compress the outputs into a .zip archive after
                       completion.
                       Companion report folders such as png/ are kept next to it.
    --zip-name         Optional filename for the archive (default: outputs.zip).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pipelines import PipelineDescriptor, load_pipeline_catalog
from postprocess import PostprocessDescriptor, load_postprocess_catalog
from workflows import (
    WorkflowCallbacks,
    WorkflowInputError,
    WorkflowInputSelection,
    WorkflowOutputOptions,
    WorkflowRequestState,
    build_workflow_request,
    dispatch_workflow,
    resolve_work_selection,
    zip_output_dir,
)

HOLO_SUFFIX = ".holo"
INPUT_LIST_SUFFIX = ".txt"


def _build_pipeline_registry() -> dict[str, PipelineDescriptor]:
    available, _missing = load_pipeline_catalog()
    return {pipeline.name: pipeline for pipeline in available}


def _build_postprocess_registry() -> dict[str, PostprocessDescriptor]:
    available, _missing = load_postprocess_catalog()
    return {
        postprocess.name: postprocess
        for postprocess in available
        if getattr(postprocess, "visibility", "visible") != "hidden"
    }


def _load_name_list(path: Path) -> list[str]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    return [
        name
        for line in raw_lines
        if (name := line.strip()) and not name.startswith("#")
    ]


def _dedupe_names(names: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _selected_names(
    selection_file: Path | None,
    inline_names: Sequence[str],
) -> list[str]:
    names: list[str] = []
    if selection_file is not None:
        names.extend(_load_name_list(selection_file))
    names.extend(inline_names)
    return _dedupe_names(names)


def _is_holo_selection(paths: Sequence[Path]) -> bool:
    if len(paths) == 1 and paths[0].suffix.lower() == INPUT_LIST_SUFFIX:
        return True
    return bool(paths) and all(path.suffix.lower() == HOLO_SUFFIX for path in paths)


def _normalize_data_paths(
    data_paths: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(data_paths, (str, Path)):
        return (Path(data_paths),)
    return tuple(Path(path) for path in data_paths)


def _prepare_cli_input(paths: Sequence[Path]) -> WorkflowInputSelection:
    expanded_paths = tuple(path.expanduser() for path in paths)
    if not expanded_paths:
        raise ValueError("Provide at least one --data path.")
    if _is_holo_selection(expanded_paths):
        return WorkflowInputSelection(
            convention="holo",
            holo_paths=expanded_paths,
        )
    if len(expanded_paths) == 1:
        return WorkflowInputSelection(
            convention="legacy",
            data_value=str(expanded_paths[0]),
        )
    return WorkflowInputSelection(
        convention="legacy",
        legacy_input_paths=expanded_paths,
    )


def run_cli(
    data_paths: str | Path | Sequence[str | Path],
    pipelines_file: Path | None,
    postprocess_file: Path | None,
    output_dir: str | Path,
    pipeline_names: Sequence[str] = (),
    postprocess_names: Sequence[str] = (),
    trim_source: bool = False,
    zip_outputs: bool = False,
    zip_name: str | None = None,
) -> int:
    data_paths = _normalize_data_paths(data_paths)
    try:
        selected_pipeline_names = _selected_names(pipelines_file, pipeline_names)
        selected_postprocess_names = _selected_names(postprocess_file, postprocess_names)
        work_selection = resolve_work_selection(
            selected_pipeline_names,
            _build_pipeline_registry(),
            selected_postprocess_names,
            _build_postprocess_registry(),
        )
        input_selection = _prepare_cli_input(data_paths)
        request = build_workflow_request(
            WorkflowRequestState(
                input_selection=input_selection,
                work_selection=work_selection,
                output_options=WorkflowOutputOptions(
                    base_output_value=str(output_dir),
                    zip_outputs=zip_outputs,
                    zip_name=zip_name or "outputs.zip",
                    trim_source=trim_source,
                ),
            ),
            zip_output_dir=zip_output_dir,
            output_filename_for_run=lambda _path, _inputs: None,
        )
        dispatch_result = dispatch_workflow(
            request,
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


def _print_pipeline_catalog() -> None:
    available, missing = load_pipeline_catalog()
    _print_descriptor_catalog("Pipelines", available, missing)


def _print_postprocess_catalog() -> None:
    available, missing = load_postprocess_catalog()
    visible_available = [
        item for item in available if getattr(item, "visibility", "visible") != "hidden"
    ]
    visible_missing = [
        item for item in missing if getattr(item, "visibility", "visible") != "hidden"
    ]
    _print_descriptor_catalog("Postprocess steps", visible_available, visible_missing)


def _print_descriptor_catalog(title: str, available, missing) -> None:
    print(title)
    for item in available:
        description = (getattr(item, "description", "") or "").strip()
        suffix = f" - {description}" if description else ""
        print(f"  [available] {item.name}{suffix}")
    for item in missing:
        reason = (
            ", ".join(getattr(item, "missing_deps", ()) or ())
            or ", ".join(getattr(item, "missing_pipelines", ()) or ())
            or getattr(item, "error_msg", "")
            or "unavailable"
        )
        print(f"  [unavailable] {item.name} - {reason}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AngioEye pipelines and postprocess steps."
    )
    parser.add_argument(
        "-d",
        "--data",
        action="append",
        type=Path,
        default=[],
        help=(
            "Input path. Use a directory, one .h5/.hdf5 file, a .zip archive, "
            "one .txt holo path list, or repeat for multiple .holo/.h5 inputs."
        ),
    )
    parser.add_argument(
        "-p",
        "--pipelines",
        type=Path,
        default=None,
        help=(
            "Text file with pipeline names to run (one per line, '#' and blank "
            "lines ignored)."
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        help="Pipeline name to run; may be repeated and is combined with --pipelines.",
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=None,
        help="Optional text file with postprocess names to run after pipelines.",
    )
    parser.add_argument(
        "--postprocess-step",
        action="append",
        default=[],
        help=(
            "Postprocess step name to run; may be repeated and is combined with "
            "--postprocess."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Base output directory. Input subfolder layout is preserved for "
            "output files."
        ),
    )
    parser.add_argument(
        "-t",
        "--trim-source",
        action="store_true",
        help=(
            "When set, source HDF5 contents will not be copied into pipeline "
            "output files (reducing output size, but losing provenance)."
        ),
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
        help=(
            "Archive filename to place inside the output directory "
            "(default: outputs.zip)."
        ),
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List discovered pipelines and exit.",
    )
    parser.add_argument(
        "--list-postprocess",
        action="store_true",
        help="List discovered visible postprocess steps and exit.",
    )
    args = parser.parse_args(argv)

    try:
        if args.list_pipelines:
            _print_pipeline_catalog()
            return 0
        if args.list_postprocess:
            _print_postprocess_catalog()
            return 0
        if not args.data:
            parser.error("the following arguments are required: -d/--data")
        if args.output is None:
            parser.error("the following arguments are required: -o/--output")
        return run_cli(
            args.data,
            args.pipelines,
            args.postprocess,
            args.output,
            pipeline_names=args.pipeline,
            postprocess_names=args.postprocess_step,
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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
