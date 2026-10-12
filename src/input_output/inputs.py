from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .archive_io import count_h5_members
from .hdf5_schema import is_hdf5_path

InputKind = Literal["file", "folder", "zip"]


@dataclass(frozen=True)
class InputPlan:
    kind: InputKind
    input_path: Path
    h5_paths: tuple[Path, ...] = ()
    zip_member_count: int | None = None

    @property
    def is_zip(self) -> bool:
        return self.kind == "zip"

    @property
    def item_count(self) -> int:
        if self.zip_member_count is not None:
            return self.zip_member_count
        return len(self.h5_paths)


def prepare_run_input(input_path: str | Path) -> InputPlan:
    path = Path(input_path).expanduser()
    if _is_zip_file(path):
        return InputPlan(
            kind="zip",
            input_path=path,
            zip_member_count=count_h5_members(path),
        )
    if path.is_file() and is_hdf5_path(path):
        return InputPlan(kind="file", input_path=path, h5_paths=(path,))
    return InputPlan(
        kind="folder",
        input_path=path,
        h5_paths=tuple(find_hdf5_inputs(path)),
    )


def prepare_run_inputs(input_paths: Sequence[str | Path]) -> InputPlan:
    paths = tuple(Path(path).expanduser() for path in input_paths)
    if len(paths) == 1:
        return prepare_run_input(paths[0])
    if not paths:
        return InputPlan(kind="folder", input_path=Path("."), h5_paths=())
    invalid_paths = [
        path for path in paths if not path.is_file() or not is_hdf5_path(path)
    ]
    if invalid_paths:
        invalid_summary = ", ".join(str(path) for path in invalid_paths)
        raise ValueError(
            f"Multiple file selection only supports .h5/.hdf5 files: {invalid_summary}"
        )
    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return InputPlan(
        kind="file",
        input_path=common_parent,
        h5_paths=paths,
    )


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


def _is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"
