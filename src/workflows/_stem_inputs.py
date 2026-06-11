from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from input_output import read_holo_path_list

from ._holo import (
    HoloInputContext,
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

    return resolve_context(root_dir.expanduser() / f"{stem}.holo")


def resolve_selected_holo_contexts(paths: Sequence[Path]) -> ResolvedInputContexts:
    if len(paths) == 1 and paths[0].suffix.lower() == ".txt":
        return _resolve_holo_path_list_contexts(paths[0])
    if any(path.suffix.lower() == ".txt" for path in paths):
        raise ValueError(
            "Select either one .txt holo path list or one or more .holo files."
        )
    return _resolve_holo_contexts(paths)


def _resolve_holo_path_list_contexts(path: Path) -> ResolvedInputContexts:
    contexts: list[HoloInputContext] = []
    skipped: list[str] = []
    failures: list[str] = []
    input_list = read_holo_path_list(path.expanduser())
    for stem in input_list.stems:
        try:
            contexts.append(resolve_stem_context(input_list.root_dir, stem))
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
