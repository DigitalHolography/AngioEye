from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from input_output.archive_io import ZipH5Member, iter_extracted_h5_members, list_h5_members
from input_output.hdf5_schema import is_hdf5_path


@dataclass(frozen=True)
class GroupedH5File:
    group_name: str
    file_name: str
    file_path: Path


def extract_group_name(root: str | Path, batch_root: str | Path) -> str:
    root_path = Path(root)
    batch_root_path = Path(batch_root)
    try:
        relative = root_path.relative_to(batch_root_path)
    except ValueError:
        return root_path.name
    return "all" if relative == Path(".") else relative.parts[0]


def _extract_member_group_name(member: ZipH5Member) -> str:
    parts = member.relative_path.parts
    return "all" if len(parts) == 1 else parts[0]


def _grouped_record_for_member(member: ZipH5Member, file_path: Path) -> GroupedH5File:
    return GroupedH5File(
        group_name=_extract_member_group_name(member),
        file_name=member.relative_path.name,
        file_path=file_path,
    )


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
            if is_hdf5_path(file_name)
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
    members = list_h5_members(zip_path)
    sortable_members = [
        (
            member,
            _grouped_record_for_member(member, member.relative_path),
        )
        for member in members
    ]
    if sort_key is None:
        sortable_members.sort(
            key=lambda item: (
                item[1].group_name.lower(),
                item[1].file_name.lower(),
            )
        )
    else:
        sortable_members.sort(key=lambda item: sort_key(item[1]))

    for member, _record in sortable_members:
        for extracted in iter_extracted_h5_members(zip_path, [member]):
            yield _grouped_record_for_member(member, extracted.path)


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

