from __future__ import annotations

from collections.abc import Iterator
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


__all__ = [
    "extracted_zip_tree",
    "replace_file_in_zip",
    "replace_folder_in_zip",
    "reset_output_dir",
]
