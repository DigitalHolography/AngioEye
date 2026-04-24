from __future__ import annotations

from collections.abc import Iterator

import h5py

from .hdf5_io import find_child_group_by_attr, safe_h5_key

ANGIOEYE_ROOT = "/AngioEye"
ANGIOEYE_PROCESSING_ROOT = f"{ANGIOEYE_ROOT}/Processing"
ANGIOEYE_POSTPROCESS_ROOT = f"{ANGIOEYE_ROOT}/Postprocessing"
LEGACY_PIPELINES_ROOT = "/pipelines"

def _child_path_candidates(base_candidates: list[str], *parts: str) -> list[str]:
    suffix = "/".join(part.strip("/") for part in parts if part)
    candidates: list[str] = []
    for base in base_candidates:
        candidate = f"{base}/{suffix}" if suffix else base
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def pipeline_path_candidates(pipeline_name: str, *parts: str) -> list[str]:
    safe_name = safe_h5_key(pipeline_name)
    return _child_path_candidates(
        [
            f"{ANGIOEYE_PROCESSING_ROOT}/{safe_name}",
            f"{ANGIOEYE_PROCESSING_ROOT}/{pipeline_name}",
        ],
        *parts,
    )


def postprocess_path_candidates(postprocess_name: str, *parts: str) -> list[str]:
    safe_name = safe_h5_key(postprocess_name)
    return _child_path_candidates(
        [
            f"{ANGIOEYE_POSTPROCESS_ROOT}/{safe_name}",
            f"{ANGIOEYE_POSTPROCESS_ROOT}/{postprocess_name}",
        ],
        *parts,
    )


def get_processing_root(
    h5file: h5py.File,
    *,
    create: bool = False,
) -> h5py.Group | None:
    if ANGIOEYE_PROCESSING_ROOT in h5file:
        group = h5file[ANGIOEYE_PROCESSING_ROOT]
        return group if isinstance(group, h5py.Group) else None
    if create:
        return h5file.require_group(ANGIOEYE_PROCESSING_ROOT)
    if LEGACY_PIPELINES_ROOT in h5file:
        group = h5file[LEGACY_PIPELINES_ROOT]
        return group if isinstance(group, h5py.Group) else None
    return None


def get_postprocess_root(
    h5file: h5py.File,
    *,
    create: bool = False,
) -> h5py.Group | None:
    if group := h5file.get(ANGIOEYE_POSTPROCESS_ROOT, None):
        return group if isinstance(group, h5py.Group) else None
    if create:
        return h5file.require_group(ANGIOEYE_POSTPROCESS_ROOT)
    return None


def find_pipeline_group(h5file: h5py.File, pipeline_name: str) -> h5py.Group | None:
    processing_root = get_processing_root(h5file)
    if processing_root is not None:
        by_attr = find_child_group_by_attr(processing_root, "pipeline", pipeline_name)
        if by_attr is not None:
            return by_attr

        safe_name = safe_h5_key(pipeline_name)
        for key in (safe_name, pipeline_name):
            child = processing_root.get(key)
            if isinstance(child, h5py.Group):
                return child

    for candidate in pipeline_path_candidates(pipeline_name):
        child = h5file.get(candidate)
        if isinstance(child, h5py.Group):
            return child
    return None


def find_postprocess_group(
    h5file: h5py.File,
    postprocess_name: str,
) -> h5py.Group | None:
    postprocess_root = get_postprocess_root(h5file)
    if postprocess_root is not None:
        by_attr = find_child_group_by_attr(
            postprocess_root,
            "pipeline",
            postprocess_name,
        )
        if by_attr is not None:
            return by_attr

        safe_name = safe_h5_key(postprocess_name)
        for key in (safe_name, postprocess_name):
            child = postprocess_root.get(key)
            if isinstance(child, h5py.Group):
                return child

    for candidate in postprocess_path_candidates(postprocess_name):
        child = h5file.get(candidate)
        if isinstance(child, h5py.Group):
            return child
    return None


def iter_metric_datasets(group: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    datasets: list[tuple[str, h5py.Dataset]] = []
    group.visititems(visitor)
    for item in datasets:
        yield item
