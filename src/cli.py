"""
Command-line interface to run AngioEye pipelines over a collection of HDF5 files.

Usage example:
    python cli.py --data data/ --pipelines pipelines.txt --postprocess postprocess.txt --output ./results --zip --zip-name my_run.zip

Inputs:
    --data / -d        Path to a directory (recursively scanned), a single .h5/.hdf5 file, or a .zip archive of .h5 files.
    --pipelines / -p   Text file listing pipeline names (one per line, '#' and blank lines ignored).
    --postprocess      Optional text file listing postprocess names (one per line, '#' and blank lines ignored).
    --output / -o      Base directory where results will be written (input subfolder layout is preserved).
    --zip / -z         When set, compress the outputs into a .zip archive after completion.
    --zip-name         Optional filename for the archive (default: outputs.zip).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py

from pipelines import (
    PipelineDescriptor,
    ProcessResult,
    load_pipeline_catalog,
)
from pipelines.core.errors import format_pipeline_exception
from pipelines.core.utils import write_combined_results_h5
from postprocess import (
    PostprocessContext,
    PostprocessDescriptor,
    load_postprocess_catalog,
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
) -> None:
    selected_set = set(selected_pipeline_names)
    errors = []
    for postprocess in postprocesses:
        missing_required = [
            name for name in postprocess.required_pipelines if name not in selected_set
        ]
        if missing_required:
            errors.append(
                f"{postprocess.name} requires pipeline(s): "
                f"{', '.join(missing_required)}"
            )
    if errors:
        raise ValueError("\n".join(errors))


def _find_h5_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return [path]
        raise ValueError(f"File is not an HDF5 file: {path}")
    if path.is_dir():
        files = sorted({*path.rglob("*.h5"), *path.rglob("*.hdf5")})
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _safe_pipeline_suffix(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "pipeline"


def _prepare_data_root(
    data_path: Path,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a directory containing HDF5 files; extract zip archives when needed."""
    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        tempdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(data_path, "r") as zf:
            zf.extractall(tempdir.name)
        return Path(tempdir.name), tempdir
    return data_path, None


def _run_pipelines_on_file(
    h5_path: Path,
    pipelines: Sequence[PipelineDescriptor],
    output_root: Path,
    output_relative_parent: Path = Path("."),
) -> Path:
    target_dir = output_root / output_relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    combined_h5_out = target_dir / f"{h5_path.stem}_pipelines_result.h5"
    suffix = 1
    while combined_h5_out.exists():
        combined_h5_out = target_dir / f"{h5_path.stem}_{suffix}_pipelines_result.h5"
        suffix += 1
    pipeline_results: list[tuple[str, ProcessResult]] = []
    with h5py.File(h5_path, "r") as h5file:
        for pipeline_desc in pipelines:
            pipeline = pipeline_desc.instantiate()
            try:
                result = pipeline.run(h5file)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(format_pipeline_exception(exc, pipeline)) from exc
            pipeline_results.append((pipeline.name, result))
            print(f"[OK] {h5_path.name} -> {pipeline.name}")
    write_combined_results_h5(
        pipeline_results, combined_h5_out, source_file=str(h5_path)
    )
    for _, result in pipeline_results:
        result.output_h5_path = str(combined_h5_out)
    print(f"[OK] {h5_path.name}: combined results -> {combined_h5_out}")
    return combined_h5_out


def _relative_input_parent(h5_path: Path, input_root: Path) -> Path:
    if input_root.is_dir():
        try:
            return h5_path.resolve().relative_to(input_root.resolve()).parent
        except ValueError:
            pass
    return Path(".")


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
    files = sorted(
        (file_path for file_path in folder.rglob("*") if file_path.is_file()),
        key=lambda path: str(path.relative_to(folder)),
    )
    total_files = len(files)
    if progress_callback is not None:
        progress_callback(0, total_files, Path("."))
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as zf:
        for idx, file_path in enumerate(files, start=1):
            rel_path = file_path.relative_to(folder)
            zf.write(file_path, rel_path)
            if progress_callback is not None:
                progress_callback(idx, total_files, rel_path)
    return zip_path


def run_cli(
    data_path: Path,
    pipelines_file: Path,
    postprocess_file: Path | None,
    output_dir: Path,
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
    _validate_postprocess_selection(
        postprocesses,
        selected_pipeline_names=[pipeline.name for pipeline in pipelines],
    )
    data_root, tempdir = _prepare_data_root(data_path)
    work_tempdir_path: Path | None = None
    clean_work_output = False
    try:
        inputs = _find_h5_inputs(data_root)
        if not inputs:
            raise ValueError(f"No .h5/.hdf5 files found under {data_path}")

        output_root = output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        work_root = output_root
        if zip_outputs:
            work_tempdir_path = Path(tempfile.mkdtemp(dir=output_root))
            work_root = work_tempdir_path

        failures: list[str] = []
        processed_outputs: list[Path] = []
        processed_input_paths: list[Path] = []
        for h5_path in inputs:
            try:
                relative_parent = _relative_input_parent(h5_path, data_root)
                combined_output = _run_pipelines_on_file(
                    h5_path,
                    pipelines,
                    work_root,
                    output_relative_parent=relative_parent,
                )
                processed_outputs.append(combined_output)
                processed_input_paths.append(h5_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                print(f"[FAIL] {h5_path.name}: {exc}", file=sys.stderr)

        if postprocesses and processed_outputs:
            context = PostprocessContext(
                output_dir=work_root,
                processed_files=tuple(processed_outputs),
                selected_pipelines=tuple(pipeline.name for pipeline in pipelines),
                input_path=data_path,
                zip_outputs=zip_outputs,
                input_h5_paths=tuple(processed_input_paths),
            )
            for descriptor in postprocesses:
                print(f"[POST] Running {descriptor.name}...")
                try:
                    result = descriptor.instantiate().run(context)
                except Exception as exc:  # noqa: BLE001
                    msg = (
                        f"Postprocess '{descriptor.name}' failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    failures.append(msg)
                    print(f"[POST FAIL] {msg}", file=sys.stderr)
                    continue
                if result.summary:
                    print(f"[POST OK] {descriptor.name}: {result.summary}")
                else:
                    print(f"[POST OK] {descriptor.name}")
        elif postprocesses:
            print(
                "[POST SKIP] No successful pipeline outputs were generated, "
                "so postprocess steps were skipped.",
                file=sys.stderr,
            )

        if zip_outputs:
            try:
                final_name = (zip_name or "outputs.zip").strip() or "outputs.zip"
                if not final_name.lower().endswith(".zip"):
                    final_name += ".zip"
                print("[ZIP] Preparing archive...")
                last_progress_log = 0.0

                def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
                    nonlocal last_progress_log
                    now = time.monotonic()
                    if done == total or (now - last_progress_log) >= 0.5:
                        pct = 100 if total == 0 else int((done * 100) / total)
                        print(f"[ZIP] {done}/{total} files ({pct}%)")
                        last_progress_log = now

                zip_path = _zip_output_dir(
                    work_root,
                    target_path=output_root / final_name,
                    progress_callback=_zip_progress,
                )
                print(f"[ZIP] Archive created: {zip_path}")
                summary_msg = f"ZIP archive: {zip_path}"
                clean_work_output = True
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[ZIP FAIL] Could not create ZIP archive: {exc}", file=sys.stderr
                )
                summary_msg = f"Outputs stored under: {work_root}"
        else:
            summary_msg = f"Outputs stored under: {work_root}"

        print(f"Completed. {summary_msg}")

        if failures:
            print(f"{len(failures)} failure(s):", file=sys.stderr)
            for msg in failures:
                print(f" - {msg}", file=sys.stderr)
            return 1
        return 0
    finally:
        if tempdir is not None:
            tempdir.cleanup()
        if clean_work_output and work_tempdir_path is not None:
            shutil.rmtree(work_tempdir_path, ignore_errors=True)


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
        "-z",
        "--zip",
        action="store_true",
        help="Zip the outputs after processing (only the archive is kept).",
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
            zip_outputs=args.zip,
            zip_name=args.zip_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
