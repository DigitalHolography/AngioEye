from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from input_output import (
    ZipH5Member,
    count_h5_members,
    find_hdf5_inputs,
    iter_h5_members,
)


@dataclass(frozen=True)
class RunInputPlan:
    input_path: Path
    h5_paths: tuple[Path, ...] = ()
    zip_member_count: int | None = None

    @property
    def is_zip(self) -> bool:
        return self.zip_member_count is not None

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
            input_path=path,
            zip_member_count=count_h5_members(path),
        )
    return RunInputPlan(input_path=path, h5_paths=tuple(find_hdf5_inputs(path)))


def _is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"
