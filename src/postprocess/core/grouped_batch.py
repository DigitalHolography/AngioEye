from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from angioeye_io.archive_io import extracted_zip_tree


H5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class GroupedH5File:
    group_name: str
    file_name: str
    file_path: Path


def extract_group_name(root: str | Path, batch_root: str | Path) -> str:
    root_path = Path(root)
    batch_root_path = Path(batch_root)
    return "all" if root_path == batch_root_path else root_path.name


def iter_grouped_h5_files(
    batch_root: str | Path,
    *,
    sort_key: Callable[[GroupedH5File], Any] | None = None,
) -> Iterator[GroupedH5File]:
    batch_root_path = Path(batch_root)
    records: list[GroupedH5File] = []

    for root, _, files in batch_root_path.walk():
        h5_files = sorted(
            file_name
            for file_name in files
            if Path(file_name).suffix.lower() in H5_SUFFIXES
        )
        if not h5_files:
            continue

        group_name = extract_group_name(root, batch_root_path)
        root_path = Path(root)
        for file_name in h5_files:
            records.append(
                GroupedH5File(
                    group_name=group_name,
                    file_name=file_name,
                    file_path=root_path / file_name,
                )
            )

    if sort_key is None:
        records.sort(
            key=lambda record: (record.group_name.lower(), record.file_name.lower())
        )
    else:
        records.sort(key=sort_key)

    yield from records


def iter_grouped_h5_files_in_zip(
    zip_path: str | Path,
    *,
    sort_key: Callable[[GroupedH5File], Any] | None = None,
) -> Iterator[GroupedH5File]:
    with extracted_zip_tree(zip_path) as batch_root:
        yield from iter_grouped_h5_files(batch_root, sort_key=sort_key)


def build_grouped_h5_index(
    batch_root: str | Path,
    *,
    sort_key: Callable[[GroupedH5File], Any] | None = None,
) -> dict[str, dict[str, Path]]:
    index: defaultdict[str, dict[str, Path]] = defaultdict(dict)
    for record in iter_grouped_h5_files(batch_root, sort_key=sort_key):
        index[record.group_name][record.file_name] = record.file_path
    return dict(index)


def find_control_group_name(groups: Iterable[object]) -> str | None:
    for group in groups:
        if group is None:
            continue
        group_lower = str(group).lower()
        if "control" in group_lower or group_lower in {"ctrl", "ctl", "controls"}:
            return str(group)
    return None


def build_group_order(groups: Iterable[str]) -> list[str]:
    ordered_groups = sorted(groups)
    control_name = find_control_group_name(ordered_groups)
    if control_name in ordered_groups:
        ordered_groups = [group for group in ordered_groups if group != control_name]
        ordered_groups.append(control_name)
    return ordered_groups


__all__ = [
    "GroupedH5File",
    "build_group_order",
    "build_grouped_h5_index",
    "extract_group_name",
    "find_control_group_name",
    "iter_grouped_h5_files",
    "iter_grouped_h5_files_in_zip",
]
