from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from input_output import read_stem_list

from ._holo import (
    HoloInputContext,
    ef_dir,
    find_ef_h5,
    output_dir,
    resolve_context,
)


@dataclass(frozen=True)
class ResolvedInputContexts:
    contexts: list[HoloInputContext]
    skipped_stems: list[str]
    failures: list[str]


def resolve_stem_context(root_dir: Path, stem: str) -> HoloInputContext:
    stem = stem.strip()
    if not stem or Path(stem).name != stem:
        raise ValueError(f"Invalid input stem: {stem!r}")

    stem_path = root_dir.expanduser() / stem
    h5_path = find_ef_h5(stem_path)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5/.hdf5 file found under {ef_dir(stem_path)}")
    return HoloInputContext(
        holo_path=stem_path,
        ef_dir=ef_dir(stem_path),
        h5_path=h5_path,
        output_dir=output_dir(stem_path),
    )


def resolve_selected_holo_contexts(paths: Sequence[Path]) -> ResolvedInputContexts:
    if len(paths) == 1 and paths[0].suffix.lower() == ".txt":
        return _resolve_stem_list_contexts(paths[0])
    if any(path.suffix.lower() == ".txt" for path in paths):
        raise ValueError(
            "Select either one .txt stem list or one or more .holo files."
        )
    return _resolve_holo_contexts(paths)


def _resolve_stem_list_contexts(path: Path) -> ResolvedInputContexts:
    contexts: list[HoloInputContext] = []
    skipped: list[str] = []
    failures: list[str] = []
    input_list = path.expanduser()
    for stem in read_stem_list(input_list):
        try:
            contexts.append(resolve_stem_context(input_list.parent, stem))
        except Exception as exc:  # noqa: BLE001
            skipped.append(stem)
            failures.append(f"{stem}: {exc}")
    return ResolvedInputContexts(contexts, skipped, failures)


def _resolve_holo_contexts(paths: Sequence[Path]) -> ResolvedInputContexts:
    contexts: list[HoloInputContext] = []
    skipped: list[str] = []
    failures: list[str] = []
    for holo_path in paths:
        try:
            contexts.append(resolve_context(holo_path))
        except Exception as exc:  # noqa: BLE001
            skipped.append(holo_path.stem)
            failures.append(f"{holo_path}: {exc}")
    return ResolvedInputContexts(contexts, skipped, failures)
