from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from posixpath import normpath
from tempfile import TemporaryDirectory

from .hdf5_schema import is_hdf5_path


@dataclass(frozen=True)
class ZipH5Member:
    name: str
    relative_path: Path
    file_size: int
    compress_size: int


@dataclass(frozen=True)
class ExtractedH5Member:
    member: ZipH5Member
    path: Path


def _safe_zip_relative_path(member_name: str) -> Path:
    normalized = normpath(member_name.replace("\\", "/")).strip("/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    return Path(*parts)


def _zip_h5_member_from_info(info: zipfile.ZipInfo) -> ZipH5Member | None:
    if info.is_dir():
        return None
    relative_path = _safe_zip_relative_path(info.filename)
    if not is_hdf5_path(relative_path):
        return None
    return ZipH5Member(
        name=info.filename,
        relative_path=relative_path,
        file_size=info.file_size,
        compress_size=info.compress_size,
    )


def iter_h5_members(zip_path: str | Path) -> Iterator[ZipH5Member]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            member = _zip_h5_member_from_info(info)
            if member is not None:
                yield member


def count_h5_members(zip_path: str | Path) -> int:
    return sum(1 for _member in iter_h5_members(zip_path))


def list_h5_members(zip_path: str | Path) -> list[ZipH5Member]:
    members = list(iter_h5_members(zip_path))
    members.sort(key=lambda member: member.relative_path.as_posix())
    return members


def batched(items: Iterable[ZipH5Member], batch_size: int) -> Iterator[list[ZipH5Member]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    batch: list[ZipH5Member] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_h5_member_batches(
    members: Iterable[ZipH5Member],
    batch_size: int,
) -> Iterator[tuple[int, list[ZipH5Member]]]:
    for batch_idx, member_batch in enumerate(batched(members, batch_size), start=1):
        yield batch_idx, member_batch


def extract_h5_member(
    zip_path: str | Path,
    member: ZipH5Member,
    output_root: str | Path,
) -> Path:
    output_root_path = Path(output_root)
    target_path = output_root_path / member.relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        _extract_h5_member_from_archive(archive, member, target_path)
    return target_path


def _extract_h5_member_from_archive(
    archive: zipfile.ZipFile,
    member: ZipH5Member,
    target_path: Path,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member.name) as src, target_path.open("wb") as dest:
        shutil.copyfileobj(src, dest, length=1024 * 1024)


def extract_h5_members(
    zip_path: str | Path,
    members: Iterable[ZipH5Member],
    output_root: str | Path,
) -> list[Path]:
    output_root_path = Path(output_root)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in members:
            target_path = output_root_path / member.relative_path
            _extract_h5_member_from_archive(archive, member, target_path)
            extracted.append(target_path)
    return extracted


def iter_extracted_h5_members(
    zip_path: str | Path,
    members: Iterable[ZipH5Member] | None = None,
) -> Iterator[ExtractedH5Member]:
    selected_members = iter_h5_members(zip_path) if members is None else members
    for member in selected_members:
        with TemporaryDirectory() as tmp_dir:
            extracted_path = extract_h5_member(zip_path, member, tmp_dir)
            yield ExtractedH5Member(member=member, path=extracted_path)


@contextmanager
def extracted_zip_tree(zip_path: str | Path) -> Iterator[Path]:
    with TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmp_dir)
        yield Path(tmp_dir)


def create_zip_from_tree(
    tree_root: str | Path,
    zip_path: str | Path,
    *,
    source_paths: Iterable[str | Path] | None = None,
    compresslevel: int = 1,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> Path:
    tree_root_path = Path(tree_root).expanduser().resolve()
    zip_path_obj = Path(zip_path)
    zip_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if source_paths is None:
        files = sorted(
            (path for path in tree_root_path.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(tree_root_path).as_posix(),
        )
    else:
        files = []
        for source_path in source_paths:
            file_path = Path(source_path).expanduser().resolve()
            if not file_path.is_file():
                raise FileNotFoundError(f"Source file does not exist: {file_path}")
            try:
                file_path.relative_to(tree_root_path)
            except ValueError as exc:
                raise ValueError(
                    f"Source file is not inside archive root {tree_root_path}: "
                    f"{file_path}"
                ) from exc
            files.append(file_path)
        files.sort(key=lambda path: path.relative_to(tree_root_path).as_posix())

    with zipfile.ZipFile(
        zip_path_obj,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as archive:
        total_files = len(files)
        if progress_callback is not None:
            progress_callback(0, total_files, Path("."))
        for idx, file_path in enumerate(files, start=1):
            rel_path = file_path.relative_to(tree_root_path)
            archive.write(file_path, rel_path)
            if progress_callback is not None:
                progress_callback(idx, total_files, rel_path)
    return zip_path_obj


@contextmanager
def temporary_zip_from_tree(
    tree_root: str | Path,
    *,
    source_paths: Iterable[str | Path] | None = None,
    archive_name: str = "batch_outputs.zip",
    compresslevel: int = 1,
) -> Iterator[Path]:
    with TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / archive_name
        create_zip_from_tree(
            tree_root,
            zip_path,
            source_paths=source_paths,
            compresslevel=compresslevel,
        )
        yield zip_path


def reset_output_dir(path: str | Path) -> None:
    path_obj = Path(path)
    if path_obj.is_dir():
        shutil.rmtree(path_obj)
    path_obj.mkdir(parents=True, exist_ok=True)


def replace_folder_in_zip(
    zip_path: str | Path,
    folder_path: str | Path,
    *,
    arc_folder: str,
) -> None:
    temp_zip = str(zip_path) + ".tmp"
    folder_path_obj = Path(folder_path)

    with zipfile.ZipFile(zip_path, "r") as source_archive:
        with zipfile.ZipFile(
            temp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target_archive:
            for item in source_archive.infolist():
                if not item.filename.startswith(f"{arc_folder}/"):
                    target_archive.writestr(item, source_archive.read(item.filename))

            for root, _, files in folder_path_obj.walk():
                root_path = Path(root)
                for file_name in files:
                    full_path = root_path / file_name
                    rel_path = full_path.relative_to(folder_path_obj)
                    arcname = (Path(arc_folder) / rel_path).as_posix()
                    target_archive.write(full_path, arcname)

    Path(temp_zip).replace(zip_path)


def replace_file_in_zip(
    zip_path: str | Path,
    file_to_add: str | Path,
    *,
    arcname: str | None = None,
) -> None:
    temp_zip = str(zip_path) + ".tmp"
    file_path = Path(file_to_add)
    archive_name = arcname or file_path.name

    with zipfile.ZipFile(zip_path, "r") as source_archive:
        with zipfile.ZipFile(
            temp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target_archive:
            for item in source_archive.infolist():
                if item.filename != archive_name:
                    target_archive.writestr(item, source_archive.read(item.filename))

            target_archive.write(file_path, archive_name)

    Path(temp_zip).replace(zip_path)


def extract_file_from_zip(
    zip_path: str | Path,
    member_name: str,
    output_dir: str | Path,
) -> Path:
    target = Path(output_dir) / member_name
    target.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member_name) as src, target.open("wb") as dest:
            shutil.copyfileobj(src, dest)

    return target


def extract_folder_from_zip(
    zip_path: str | Path,
    *,
    member_prefix: str,
    output_dir: str | Path,
) -> list[Path]:
    prefix = member_prefix.rstrip("/")
    target_dir = Path(output_dir) / prefix
    if target_dir.exists():
        shutil.rmtree(target_dir)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in sorted(
            item.filename for item in archive.infolist() if not item.is_dir()
        ):
            if not member.startswith(f"{prefix}/"):
                continue
            rel_path = Path(member).relative_to(prefix)
            target = target_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            extracted.append(target)

    return extracted


__all__ = [
    "ZipH5Member",
    "ExtractedH5Member",
    "batched",
    "count_h5_members",
    "create_zip_from_tree",
    "extract_file_from_zip",
    "extract_folder_from_zip",
    "extract_h5_member",
    "extract_h5_members",
    "extracted_zip_tree",
    "iter_extracted_h5_members",
    "iter_h5_member_batches",
    "iter_h5_members",
    "list_h5_members",
    "replace_file_in_zip",
    "replace_folder_in_zip",
    "reset_output_dir",
    "temporary_zip_from_tree",
]
