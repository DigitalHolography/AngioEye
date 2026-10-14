from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .hdf5_schema import is_hdf5_path

HOLO_SUFFIX = ".holo"
InputKind = Literal["file", "folder"]


@dataclass(frozen=True)
class InputPlan:
    kind: InputKind
    input_path: Path
    h5_paths: tuple[Path, ...] = ()
    holo_paths: tuple[Path, ...] = ()
    input_failures: tuple[str, ...] = ()
    selected_holo_paths: tuple[Path, ...] = ()

    @property
    def item_count(self) -> int:
        return len(self.h5_paths) + len(self.input_failures)


def prepare_run_input(input_path: str | Path) -> InputPlan:
    return prepare_run_inputs((input_path,))


def prepare_run_inputs(input_paths: Sequence[str | Path]) -> InputPlan:
    """Resolve selected acquisitions to their existing EyeFlow result files."""
    paths = validate_input_paths(input_paths)
    is_folder = paths[0].is_dir()
    holos = tuple(_find_holo_inputs(paths[0])) if is_folder else paths
    if is_folder or len(paths) == 1:
        input_path = paths[0]
    else:
        try:
            input_path = Path(os.path.commonpath([str(path.parent) for path in paths]))
        except ValueError as exc:
            raise ValueError("Select .holo files on the same drive.") from exc

    resolved_holos: list[Path] = []
    h5_paths: list[Path] = []
    failures: list[str] = []
    for holo in holos:
        try:
            result = resolve_eyeflow_h5(holo)
        except FileNotFoundError as exc:
            if not is_folder and len(holos) == 1:
                raise
            failures.append(str(exc))
            continue
        resolved_holos.append(holo)
        h5_paths.append(result)
    if not h5_paths:
        raise FileNotFoundError(
            "No processed EyeFlow results found for the selected .holo inputs:\n\n"
            + "\n\n".join(failures)
        )
    return InputPlan(
        kind="folder" if is_folder else "file",
        input_path=input_path,
        h5_paths=tuple(h5_paths),
        holo_paths=tuple(resolved_holos),
        input_failures=tuple(failures),
        selected_holo_paths=holos,
    )


def validate_input_paths(input_paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Accept one folder containing acquisitions, or existing .holo files."""
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in input_paths:
        if not str(raw_path).strip():
            raise ValueError("Select a .holo file or a folder containing .holo files.")
        path = Path(raw_path).expanduser().resolve()
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            paths.append(path)
    if not paths:
        raise ValueError("Select a .holo file or a folder containing .holo files.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist:\n{path}")
        if not path.is_dir():
            _validate_holo_file(path)
    if any(path.is_dir() for path in paths):
        if len(paths) != 1:
            raise ValueError(
                "Select one folder containing .holo files, or one or more .holo files. "
                "Folders cannot be combined with other inputs."
            )
        if next(_iter_holo_inputs(paths[0]), None) is None:
            raise ValueError(
                f"No .holo files found in the selected folder:\n{paths[0]}"
            )
    return tuple(paths)


def resolve_eyeflow_h5(holo_path: str | Path) -> Path:
    """Use EyeFlow's acquisition layout without guessing another export."""
    holo = Path(holo_path).expanduser().resolve()
    _validate_holo_file(holo)
    result = holo.parent / holo.stem / f"{holo.stem}_EF" / "h5" / f"{holo.stem}_EF.h5"
    if not result.is_file():
        raise FileNotFoundError(
            f"Processed EyeFlow result missing for {holo.name}.\n"
            f"Expected:\n{result}\n"
            "Run EyeFlow for this .holo file before analyzing it in AngioEye."
        )
    return result


def _validate_holo_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist:\n{path}")
    if not path.is_file() or path.suffix.lower() != HOLO_SUFFIX:
        raise ValueError(
            f"Input must be a .holo file or a folder containing .holo files:\n{path}"
        )


def _find_holo_inputs(folder: Path) -> list[Path]:
    return sorted(
        _iter_holo_inputs(folder),
        key=lambda path: (str(path).casefold(), str(path)),
    )


def _iter_holo_inputs(folder: Path) -> Iterator[Path]:
    for path in folder.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() == HOLO_SUFFIX
            and not path.name.startswith("._")
            and not any(part.casefold() == "__macosx" for part in path.parts)
        ):
            yield path.resolve()


def iter_hdf5_inputs(path: str | Path) -> Iterator[Path]:
    input_path = Path(path)
    if input_path.is_file():
        if is_hdf5_path(input_path):
            yield input_path
            return
        raise ValueError(f"File is not an HDF5 file: {input_path}")
    if input_path.is_dir():
        files = sorted(
            file_path
            for file_path in input_path.rglob("*")
            if file_path.is_file() and is_hdf5_path(file_path)
        )
        yield from files
        return
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def find_hdf5_inputs(path: str | Path) -> list[Path]:
    return list(iter_hdf5_inputs(path))
