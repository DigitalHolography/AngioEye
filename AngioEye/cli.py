"""
Command-line interface to run AngioEye pipelines over a collection of HDF5 files.

Usage example:
    python cli.py --data data/ --pipelines pipelines.txt --output ./results

Inputs:
    --data / -d        Path to a directory (recursively scanned), a single .h5/.hdf5 file, or a .zip archive of .h5 files.
    --pipelines / -p   Text file listing pipeline names (one per line, '#' and blank lines ignored).
    --output / -o      Base directory where results will be written. A subfolder is created per input file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tempfile
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py

from pipelines import (
    BasicStatsPipeline,
    ProcessPipeline,
    TauHarmonic10Pipeline,
    TauHarmonic10PerBeatPipeline,
    VelocityComparisonPipeline,
)
from pipelines.utils import write_result_h5


PIPELINE_CLASSES = [
    BasicStatsPipeline,
    VelocityComparisonPipeline,
    TauHarmonic10Pipeline,
    TauHarmonic10PerBeatPipeline,
]


def _build_pipeline_registry() -> Dict[str, ProcessPipeline]:
    pipelines = [cls() for cls in PIPELINE_CLASSES]
    return {p.name: p for p in pipelines}


def _load_pipeline_list(path: Path, registry: Dict[str, ProcessPipeline]) -> List[ProcessPipeline]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    selected: List[ProcessPipeline] = []
    missing: List[str] = []
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
        raise ValueError(f"Unknown pipeline(s): {', '.join(missing)}. Available: {available}")
    if not selected:
        raise ValueError("No pipelines selected (file is empty or only contains comments).")
    return selected


def _find_h5_inputs(path: Path) -> List[Path]:
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


def _prepare_data_root(data_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return a directory containing HDF5 files; extract zip archives when needed."""
    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        tempdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(data_path, "r") as zf:
            zf.extractall(tempdir.name)
        return Path(tempdir.name), tempdir
    return data_path, None


def _run_pipelines_on_file(
    h5_path: Path,
    pipelines: Sequence[ProcessPipeline],
    output_root: Path,
) -> List[Path]:
    outputs: List[Path] = []
    data_dir = output_root / h5_path.stem
    data_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5file:
        for pipeline in pipelines:
            suffix = _safe_pipeline_suffix(pipeline.name)
            h5_out = data_dir / f"{h5_path.stem}_{suffix}_result.h5"
            csv_out = data_dir / f"{h5_path.stem}_{suffix}_metrics.csv"
            result = pipeline.run(h5file)
            write_result_h5(result, h5_out, pipeline_name=pipeline.name, source_file=str(h5_path))
            result.output_h5_path = str(h5_out)
            pipeline.export(result, str(csv_out))
            outputs.extend([h5_out, csv_out])
            print(f"[OK] {h5_path.name} -> {pipeline.name}")
    return outputs


def run_cli(data_path: Path, pipelines_file: Path, output_dir: Path) -> int:
    registry = _build_pipeline_registry()
    pipelines = _load_pipeline_list(pipelines_file, registry)
    data_root, tempdir = _prepare_data_root(data_path)
    try:
        inputs = _find_h5_inputs(data_root)
        if not inputs:
            raise ValueError(f"No .h5/.hdf5 files found under {data_path}")

        output_root = output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        failures: List[str] = []
        for h5_path in inputs:
            try:
                _run_pipelines_on_file(h5_path, pipelines, output_root)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{h5_path}: {exc}")
                print(f"[FAIL] {h5_path.name}: {exc}", file=sys.stderr)

        print(f"Completed. Outputs stored under: {output_root}")
        if failures:
            print(f"{len(failures)} failure(s):", file=sys.stderr)
            for msg in failures:
                print(f" - {msg}", file=sys.stderr)
            return 1
        return 0
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run AngioEye pipelines over a folder of HDF5 files.")
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
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Base output directory. A subfolder is created per input HDF5 file.",
    )
    args = parser.parse_args(argv)

    try:
        return run_cli(args.data, args.pipelines, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
