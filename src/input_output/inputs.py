from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .archive_io import ZipH5Member, count_h5_members, iter_h5_members
from .hdf5_schema import is_hdf5_path

HOLO_SUFFIX = ".holo"
INPUT_LIST_SUFFIX = ".txt"
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

    def iter_zip_members(self) -> Iterator[ZipH5Member]:
        if not self.is_zip:
            raise ValueError("Run input plan is not a ZIP archive.")
        return iter_h5_members(self.input_path)


@dataclass(frozen=True)
class HoloInputStatus:
    ef: bool


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
            "Multiple file selection only supports .h5/.hdf5 files: "
            f"{invalid_summary}"
        )
    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return InputPlan(
        kind="file",
        input_path=common_parent,
        h5_paths=paths,
    )


def dataset_dir(holo_path: Path) -> Path:
    return holo_path.parent / holo_path.stem


def ef_dir(holo_path: Path) -> Path:
    return dataset_dir(holo_path) / f"{holo_path.stem}_EF"


def find_ef_h5(holo_path: Path) -> Path | None:
    ef_dir_path = ef_dir(holo_path)
    if not ef_dir_path.is_dir():
        return None
    candidates = sorted(
        path for path in ef_dir_path.iterdir() if path.is_file() and is_hdf5_path(path)
    )
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda path: (
            path.stem != holo_path.stem,
            str(path).lower(),
        ),
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


def relative_hdf5_parent(h5_path: str | Path, input_root: str | Path) -> Path:
    h5_path_obj = Path(h5_path)
    input_root_obj = Path(input_root)
    if input_root_obj.is_dir():
        try:
            return h5_path_obj.resolve().relative_to(input_root_obj.resolve()).parent
        except ValueError:
            pass
    return Path(".")


def holo_input_status(
    holo_path: Path,
    *,
    require_holo_file: bool,
) -> HoloInputStatus:
    holo_path = _absolute(holo_path)
    try:
        _validate_holo_file(holo_path, require_file=require_holo_file)
    except (FileNotFoundError, ValueError):
        return HoloInputStatus(ef=False)
    return HoloInputStatus(ef=find_ef_h5(holo_path) is not None)


def stem_input_status(stem: str, root_dir: Path) -> HoloInputStatus:
    root_dir = _absolute(root_dir)
    return HoloInputStatus(ef=find_ef_h5(root_dir / stem) is not None)


def read_stem_list(path: Path) -> tuple[str, ...]:
    path = path.expanduser()
    if path.suffix.lower() != INPUT_LIST_SUFFIX:
        raise ValueError(f"File is not a {INPUT_LIST_SUFFIX} stem list: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Stem list does not exist: {path}")
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def found_status_text(
    label: str,
    found_count: int,
    total_count: int,
    missing_stems: Sequence[str],
) -> str:
    if total_count == 1:
        return f"{label} found" if found_count else f"{label} not found"
    text = f"{label} {found_count}/{total_count} found"
    if missing_stems:
        text += ": missing " + ", ".join(missing_stems)
    return text


def _absolute(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


def _validate_holo_file(holo_path: Path, *, require_file: bool) -> None:
    if holo_path.suffix.lower() != HOLO_SUFFIX:
        raise ValueError(f"HOLO input must be a {HOLO_SUFFIX} file:\n{holo_path}")
    if not require_file:
        return
    if not holo_path.exists():
        raise FileNotFoundError(f"HOLO input does not exist:\n{holo_path}")
    if not holo_path.is_file():
        raise ValueError(f"HOLO input must be a file:\n{holo_path}")


def _is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"
