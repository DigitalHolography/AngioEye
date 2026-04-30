from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import zipfile


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
) -> None:
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
        for file_path in files:
            archive.write(file_path, file_path.relative_to(tree_root_path))


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
    "create_zip_from_tree",
    "extract_file_from_zip",
    "extract_folder_from_zip",
    "extracted_zip_tree",
    "replace_file_in_zip",
    "replace_folder_in_zip",
    "reset_output_dir",
    "temporary_zip_from_tree",
]
