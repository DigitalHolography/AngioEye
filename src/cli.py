"""Headless statistical pipelines for batches of EyeFlow HDF5 exports."""

from __future__ import annotations

import argparse
import ast
import multiprocessing
import sys
from collections.abc import Sequence
from pathlib import Path

from app_settings import AppSettingsStore, normalize_pipeline_visibility
from pipelines import load_pipeline_catalog
from workflows import run_batch


def _selected_names(values: Sequence[str | Path] | str | Path | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    expanded: list[str] = []
    pending: list[str] = []
    depth = 0
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if pending or value.startswith("["):
            pending.append(value)
            depth += value.count("[") - value.count("]")
            if depth > 0:
                continue
            value = " ".join(pending)
            pending.clear()
            depth = 0
        if value.startswith("[") and value.endswith("]"):
            try:
                names = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                names = [part.strip().strip("'\"") for part in value[1:-1].split(",")]
            if not isinstance(names, (list, tuple)) or not all(
                isinstance(name, str) for name in names
            ):
                raise ValueError(
                    "Pipeline list literals must contain names as strings."
                )
            expanded.extend(names)
        else:
            path = Path(value).expanduser()
            if path.is_file():
                expanded.extend(
                    line.strip()
                    for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
            else:
                expanded.append(value)
    if pending:
        raise ValueError("Unclosed pipeline list literal.")
    return list(dict.fromkeys(name.strip() for name in expanded if name.strip()))


def _log_cli(message: str) -> None:
    stream = (
        sys.stderr
        if any(marker in message for marker in ("[FAIL]", "[INPUT FAIL]", "[WARN]"))
        else sys.stdout
    )
    print(message, file=stream, flush=True)


def run_cli(
    data_paths: str | Path | Sequence[str | Path],
    pipeline_names: Sequence[str | Path] | str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    processing_branch: str = "Processing",
    zip_outputs: bool = False,
    zip_name: str = "outputs.zip",
) -> int:
    paths = [data_paths] if isinstance(data_paths, (str, Path)) else list(data_paths)
    try:
        available, missing = load_pipeline_catalog()
        registry = {
            descriptor.name: descriptor
            for descriptor in [*available, *missing]
            if descriptor.visibility != "hidden"
        }
        names = _selected_names(pipeline_names)
        if pipeline_names is None:
            visibility, _changed = normalize_pipeline_visibility(
                registry, AppSettingsStore().load_pipeline_visibility()
            )
            names = [name for name in registry if visibility.get(name, False)]
        unknown = [name for name in names if name not in registry]
        if unknown:
            raise ValueError(
                f"Unknown statistical pipeline(s): {', '.join(unknown)}. Use --list-pipelines."
            )
        result = run_batch(
            paths,
            [registry[name] for name in names],
            output_dir,
            processing_branch=processing_branch,
            zip_outputs=zip_outputs,
            zip_name=zip_name,
            log=_log_cli,
        )
        return 1 if result.failures else 0
    except (ValueError, OSError) as exc:
        _log_cli(f"[FAIL] {exc}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        action="append",
        help=".holo files or one cohort folder containing .holo files (EyeFlow results must already exist).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        action="append",
        help="Statistical pipeline names, a list literal, or a names text file.",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List statistical tools and required exported metrics.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Parent folder for <holo>/<holo>_AE outputs (default: beside the inputs).",
    )
    parser.add_argument(
        "--processing-branch",
        choices=("Processing", "ProcessingAlt"),
        default="Processing",
        help="EyeFlow method whose exported metrics are analyzed.",
    )
    parser.add_argument(
        "--zip",
        "-z",
        action="store_true",
        help="Also create a ZIP of the completed results.",
    )
    parser.add_argument("--zip-name", default="outputs.zip")
    args = parser.parse_args(argv)
    if args.list_pipelines:
        available, missing = load_pipeline_catalog()
        for descriptor in [*available, *missing]:
            if descriptor.visibility == "hidden":
                continue
            requirements = " and ".join(
                " or ".join(group) for group in descriptor.required_metric_options
            )
            status = (
                "available"
                if descriptor.available
                else f"unavailable: {descriptor.error_msg or ', '.join(descriptor.missing_deps)}"
            )
            print(
                f"{descriptor.name} [{status}]"
                + (f" — exported metrics: {requirements}" if requirements else "")
            )
        return 0
    if not args.data:
        parser.error("--data is required unless --list-pipelines is used")
    return run_cli(
        [path for group in args.data for path in group],
        [name for group in args.pipelines for name in group]
        if args.pipelines is not None
        else None,
        args.output,
        processing_branch=args.processing_branch,
        zip_outputs=args.zip,
        zip_name=args.zip_name,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
