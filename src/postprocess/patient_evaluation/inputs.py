from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator
import re
import zipfile


H5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class ResolvedH5:
    source_path: Path
    h5_path: Path
    patient_id: str
    archive_member: str | None = None


def default_output_dir(selected_path: Path) -> Path:
    selected_path = Path(selected_path)
    return selected_path.parent / f"{selected_path.stem}_fixed_threshold_evaluation"


@contextmanager
def materialize_h5_inputs(
    selected_paths: Iterable[Path],
) -> Iterator[list[ResolvedH5]]:
    """Resolve HDF5 files from H5 files, directories, and ZIP archives.

    Temporary ZIP extraction directories remain alive for the full context.
    """
    selected = [Path(path) for path in selected_paths]
    resolved: list[ResolvedH5] = []
    used_patient_ids: dict[str, int] = {}

    with TemporaryDirectory(prefix="fixed_threshold_eval_") as temp_root_str:
        temp_root = Path(temp_root_str)

        for source_index, source_path in enumerate(selected):
            if not source_path.exists():
                raise FileNotFoundError(f"Input not found: {source_path}")

            suffix = source_path.suffix.casefold()
            if suffix in H5_SUFFIXES:
                resolved.append(
                    ResolvedH5(
                        source_path=source_path,
                        h5_path=source_path,
                        patient_id=_unique_patient_id(
                            source_path.stem,
                            used_patient_ids,
                        ),
                    )
                )
                continue

            if suffix == ".zip":
                archive_dir = temp_root / f"archive_{source_index}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                members = _safe_extract_h5_members(source_path, archive_dir)
                for member_name, extracted_path in members:
                    member_path = Path(member_name)
                    base_id = "__".join(member_path.with_suffix("").parts)
                    resolved.append(
                        ResolvedH5(
                            source_path=source_path,
                            h5_path=extracted_path,
                            patient_id=_unique_patient_id(
                                base_id or extracted_path.stem,
                                used_patient_ids,
                            ),
                            archive_member=member_name,
                        )
                    )
                continue

            if source_path.is_dir():
                for h5_path in sorted(
                    path
                    for path in source_path.rglob("*")
                    if path.is_file() and path.suffix.casefold() in H5_SUFFIXES
                ):
                    relative = h5_path.relative_to(source_path)
                    base_id = "__".join(relative.with_suffix("").parts)
                    resolved.append(
                        ResolvedH5(
                            source_path=source_path,
                            h5_path=h5_path,
                            patient_id=_unique_patient_id(
                                base_id or h5_path.stem,
                                used_patient_ids,
                            ),
                            archive_member=str(relative),
                        )
                    )
                continue

            raise ValueError(
                f"Unsupported input {source_path}. Expected .h5, .hdf5, .zip, "
                "or a directory containing HDF5 files."
            )

        if not resolved:
            raise ValueError("No .h5 or .hdf5 patient file was found.")

        yield resolved


def _safe_extract_h5_members(
    zip_path: Path,
    destination: Path,
) -> list[tuple[str, Path]]:
    destination_root = destination.resolve()
    extracted: list[tuple[str, Path]] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            member = Path(info.filename)
            if member.suffix.casefold() not in H5_SUFFIXES:
                continue

            target = (destination / member).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe ZIP member path: {info.filename!r}"
                ) from exc

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, open(target, "wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            extracted.append((info.filename, target))

    # Preserve the order stored in the original ZIP. This order is used
    # to label cases C001, C002, ... in reports and plots.
    return extracted


def _unique_patient_id(
    raw_name: str,
    used: dict[str, int],
) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    normalized = normalized or "patient"
    count = used.get(normalized, 0) + 1
    used[normalized] = count
    return normalized if count == 1 else f"{normalized}_{count}"
