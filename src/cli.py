"""
Run AngioEye pipelines from the command line.

Usage example:
    python cli.py --data data/ --pipelines pipelines.txt \
        --postprocesses "HTML summary" --output ./results --zip \
        --zip-name my_run.zip

Inputs:
    --data / -d        Path to a directory (recursively scanned), a single .h5/.hdf5
                       file,
                       a .zip archive of .h5 files, one or more .holo files, or a .txt
                       holo path list. May be repeated for multiple explicit inputs.
    --pipelines        One or more pipeline names, a text file listing names, or a
                       list literal such as ["pipeline1", "pipeline2"].
    --postprocesses    One or more postprocess names, a text file listing names, or
                       a list literal such as ["postprocess1", "postprocess2"].
    --output / -o      Base directory where results will be written (input subfolder
                       layout is preserved).
    --keep-source      When set, persist the source HDF5 contents in pipeline output
                       files (increasing output size but preserving provenance).
    --zip / -z         When set, compress the outputs into a .zip archive after
                       completion.
                       Companion report folders such as png/ are kept next to it.
    --zip-name         Optional filename for the archive (default: outputs.zip).
"""

from __future__ import annotations

import argparse
import ast
import multiprocessing
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from app_settings import (
    AppSettingsStore,
    normalize_pipeline_visibility,
    normalize_postprocess_visibility,
)
from pipelines import PipelineDescriptor, load_pipeline_catalog
from postprocess import (
    PostprocessDescriptor,
    load_postprocess_catalog,
    required_options_for,
)
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


def _default_selected_names(
    registry: dict[str, object],
    *,
    visibility_loader: Callable[[], dict[str, bool]],
    normalize_visibility: Callable[..., tuple[dict[str, bool], bool]],
) -> list[str]:
    visibility, _changed = normalize_visibility(
        registry.keys(),
        visibility_loader(),
    )
    return [name for name in registry if visibility.get(name, False)]


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


def _parse_list_literal(value: str) -> list[str] | None:
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]

    # Some shells remove quotes from an unquoted expression such as
    # [pipeline1, pipeline2]. Keep that convenient form working too.
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    return [part.strip().strip("'\"") for part in inner.replace(",", " ").split()]


def _expand_selection_values(values: Sequence[str | Path] | None) -> list[str]:
    """Flatten repeated values and shell-friendly list literals."""

    if not values:
        return []
    if isinstance(values, (str, Path)):
        values = (values,)
    expanded: list[str] = []
    pending: list[str] = []
    bracket_depth = 0
    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue
        if pending or value.startswith("["):
            pending.append(value)
            bracket_depth += value.count("[") - value.count("]")
            if bracket_depth > 0:
                continue
            value = " ".join(pending)
            pending.clear()
            bracket_depth = 0
        parsed = _parse_list_literal(value)
        expanded.extend(parsed if parsed is not None else [value])
    if pending:
        expanded.append(" ".join(pending))
    return _dedupe_names(expanded)


def _selected_names(selection_values: Sequence[str | Path] | None) -> list[str]:
    names: list[str] = []
    for value in _expand_selection_values(selection_values):
        selection_file = Path(value).expanduser()
        if selection_file.is_file():
            names.extend(_load_name_list(selection_file))
        else:
            names.append(value)
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
    pipelines_file: Path | Sequence[str | Path] | None,
    postprocess_file: Path | Sequence[str | Path] | None,
    output_dir: str | Path | None,
    persist_source: bool = False,
    zip_outputs: bool = False,
    zip_name: str | None = None,
) -> int:
    data_paths = _normalize_data_paths(data_paths)
    try:
        pipeline_registry = _build_pipeline_registry()
        postprocess_registry = _build_postprocess_registry()
        settings_store = AppSettingsStore()
        effective_persist_source = bool(persist_source)
        selected_pipeline_names = _selected_names(pipelines_file)
        if not pipelines_file:
            selected_pipeline_names = _default_selected_names(
                pipeline_registry,
                visibility_loader=settings_store.load_pipeline_visibility,
                normalize_visibility=normalize_pipeline_visibility,
            )
        selected_postprocess_names = _selected_names(postprocess_file)
        if not postprocess_file:
            selected_postprocess_names = _default_selected_names(
                postprocess_registry,
                visibility_loader=settings_store.load_postprocess_visibility,
                normalize_visibility=normalize_postprocess_visibility,
            )
        work_selection = resolve_work_selection(
            selected_pipeline_names,
            pipeline_registry,
            selected_postprocess_names,
            postprocess_registry,
        )
        if any(
            "persist_eyeflow_data" in required_options_for(postprocess)
            for postprocess in work_selection.postprocesses
        ):
            effective_persist_source = True
        input_selection = _prepare_cli_input(data_paths)
        request = build_workflow_request(
            WorkflowRequestState(
                input_selection=input_selection,
                work_selection=work_selection,
                output_options=WorkflowOutputOptions(
                    base_output_value="" if output_dir is None else str(output_dir),
                    zip_outputs=zip_outputs,
                    zip_name=zip_name or "outputs.zip",
                    persist_source=effective_persist_source,
                ),
            ),
            zip_output_dir=zip_output_dir,
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
        "--pipelines",
        nargs="+",
        default=None,
        help=(
            "Pipeline name(s), a text file with one name per line, or a list "
            "such as ['pipeline1', 'pipeline2']. If omitted, use settings."
        ),
    )
    parser.add_argument(
        "--postprocesses",
        nargs="+",
        default=None,
        help=(
            "Postprocess name(s), a text file with one name per line, or a list "
            "such as ['postprocess1', 'postprocess2']. If omitted, run none."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Base output directory. Input subfolder layout is preserved for "
            "output files. If omitted, derive it from the input like the GUI."
        ),
    )
    persist_group = parser.add_mutually_exclusive_group()
    persist_group.add_argument(
        "--keep-source",
        dest="persist_source",
        action="store_true",
        help=(
            "Persist source HDF5 contents into pipeline output files. "
            "Automatically enabled when a selected postprocess requires it."
        ),
    )
    persist_group.set_defaults(persist_source=False)
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
        return run_cli(
            args.data,
            args.pipelines,
            args.postprocesses,
            args.output,
            persist_source=args.persist_source,
            zip_outputs=args.zip,
            zip_name=args.zip_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _log_cli(message: str) -> None:
    if message.startswith("[POST FAIL]") or message.startswith("[POST WARN]"):
        print(message, file=sys.stderr, flush=True)
    else:
        print(message, flush=True)


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
            print(f"[ZIP] {done}/{total} files ({pct}%)", flush=True)
            last_progress_log = now

    return _zip_progress


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
