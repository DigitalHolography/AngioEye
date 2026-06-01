from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from input_output import (
    ZipH5Member,
    count_h5_members,
    find_hdf5_inputs,
    is_hdf5_path,
    iter_h5_members,
)

RunInputKind = Literal["file", "folder", "zip"]


@dataclass(frozen=True)
class RunInputPlan:
    kind: RunInputKind
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

    def iter_zip_members(self) -> Iterator[ZipH5Member]:
        if not self.is_zip:
            raise ValueError("Run input plan is not a ZIP archive.")
        return iter_h5_members(self.input_path)


def prepare_run_input(input_path: str | Path) -> RunInputPlan:
    path = Path(input_path).expanduser()
    if _is_zip_file(path):
        return RunInputPlan(
            kind="zip",
            input_path=path,
            zip_member_count=count_h5_members(path),
        )
    if path.is_file() and is_hdf5_path(path):
        return RunInputPlan(kind="file", input_path=path, h5_paths=(path,))
    return RunInputPlan(
        kind="folder",
        input_path=path,
        h5_paths=tuple(find_hdf5_inputs(path)),
    )


def prepare_run_inputs(input_paths: Sequence[str | Path]) -> RunInputPlan:
    paths = tuple(Path(path).expanduser() for path in input_paths)
    if len(paths) == 1:
        return prepare_run_input(paths[0])
    if not paths:
        return RunInputPlan(kind="folder", input_path=Path("."), h5_paths=())
    invalid_paths = [
        path for path in paths if not path.is_file() or not is_hdf5_path(path)
    ]
    if invalid_paths:
        invalid_summary = ", ".join(str(path) for path in invalid_paths)
        raise ValueError(
            "Multiple file selection only supports .h5/.hdf5 files: "
            f"{invalid_summary}"
        )
    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return RunInputPlan(
        kind="file",
        input_path=common_parent,
        h5_paths=paths,
    )


def _is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"
