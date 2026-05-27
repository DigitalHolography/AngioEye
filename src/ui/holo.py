from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HoloInputContext:
    holo_path: Path
    ef_dir: Path
    h5_path: Path
    output_dir: Path


def dataset_dir(holo_path: Path) -> Path:
    return holo_path.parent / holo_path.stem


def ef_dir(holo_path: Path) -> Path:
    return dataset_dir(holo_path) / f"{holo_path.stem}_EF"


def output_dir(holo_path: Path) -> Path:
    return dataset_dir(holo_path) / f"{holo_path.stem}_AE"


def output_filename(holo_path: Path) -> str:
    return f"{holo_path.stem}_AE.h5"


def find_ef_h5(holo_path: Path) -> Path | None:
    ef_dir_path = ef_dir(holo_path)
    if not ef_dir_path.is_dir():
        return None
    candidates = sorted({*ef_dir_path.rglob("*.h5"), *ef_dir_path.rglob("*.hdf5")})
    if not candidates:
        return None

    def candidate_rank(path: Path) -> tuple[int, str]:
        has_h5_parent = any(parent.name.lower() == "h5" for parent in path.parents)
        exact_stem = path.stem == holo_path.stem
        if has_h5_parent and exact_stem:
            rank = 0
        elif exact_stem:
            rank = 1
        elif has_h5_parent:
            rank = 2
        else:
            rank = 3
        return rank, str(path).lower()

    return min(candidates, key=candidate_rank)


def resolve_context(holo_path: Path) -> HoloInputContext:
    holo_path = holo_path.expanduser()
    if holo_path.suffix.lower() != ".holo":
        raise ValueError(f"File is not a .holo file: {holo_path}")
    if not holo_path.is_file():
        raise FileNotFoundError(f"Holo file does not exist: {holo_path}")

    ef_dir_path = ef_dir(holo_path)
    h5_path = find_ef_h5(holo_path)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5/.hdf5 file found under {ef_dir_path}")
    return HoloInputContext(
        holo_path=holo_path,
        ef_dir=ef_dir_path,
        h5_path=h5_path,
        output_dir=output_dir(holo_path),
    )
