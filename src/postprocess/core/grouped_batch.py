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


def extract_group_name(
    root: str | Path, batch_root: str | Path, *, group_depth: int = 1
) -> str:
    """
    group_depth controls how many leading path segments under batch_root are
    joined (with "/") into the group name. Default 1 is the original
    single-level cohort/group convention (e.g. "control"/"patient"). Pass 2
    for two-level groupings such as participant/eye x epoch (e.g.
    "L/baseline1"), falling back to whatever depth is actually available if
    the path is shallower than requested.
    """
    root_path = Path(root)
    batch_root_path = Path(batch_root)
    try:
        relative = root_path.relative_to(batch_root_path)
    except ValueError:
        return root_path.name
    if relative == Path("."):
        return "all"
    parts = relative.parts[:group_depth]
    return "/".join(parts)


def _extract_member_group_name(member: ZipH5Member, *, group_depth: int = 1) -> str:
    parts = member.relative_path.parts
    if len(parts) <= 1:
        return "all"
    return "/".join(parts[: min(group_depth, len(parts) - 1)])


def _grouped_record_for_member(
    member: ZipH5Member, file_path: Path, *, group_depth: int = 1
) -> GroupedH5File:
    return GroupedH5File(
        group_name=_extract_member_group_name(member, group_depth=group_depth),
        file_name=member.relative_path.name,
        file_path=file_path,
    )


def iter_grouped_h5_files(
    batch_root: str | Path,
    *,
    sort_key: Callable[[GroupedH5File], Any] | None = None,
    group_depth: int = 1,
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

        group_name = extract_group_name(root, batch_root_path, group_depth=group_depth)
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
    group_depth: int = 1,
) -> Iterator[GroupedH5File]:
    members = list_h5_members(zip_path)
    sortable_members = [
        (
            member,
            _grouped_record_for_member(
                member, member.relative_path, group_depth=group_depth
            ),
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
            yield _grouped_record_for_member(
                member, extracted.path, group_depth=group_depth
            )


def build_grouped_h5_index(
    batch_root: str | Path,
    *,
    sort_key: Callable[[GroupedH5File], Any] | None = None,
    group_depth: int = 1,
) -> dict[str, dict[str, Path]]:
    index: defaultdict[str, dict[str, Path]] = defaultdict(dict)
    for record in iter_grouped_h5_files(
        batch_root, sort_key=sort_key, group_depth=group_depth
    ):
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

