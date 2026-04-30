from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class MetricsTree:
    name: str
    metrics: dict[str, Any]
    attrs: dict[str, Any] | None = None


def safe_h5_key(name: str) -> str:
    """Return a filesystem/HDF5-friendly key derived from a logical name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def open_h5(path: Path | str, mode: str = "r") -> h5py.File:
    return h5py.File(Path(path), mode)


def copy_h5_contents(source_file: Path | str | None, dest: h5py.File) -> None:
    """Copy all attributes and top-level objects from an existing HDF5 into dest."""
    if not source_file:
        return
    src_path = Path(source_file)
    if not src_path.exists():
        return
    with open_h5(src_path, "r") as src:
        for key, value in src.attrs.items():
            dest.attrs[key] = value
        for key in src.keys():
            src.copy(src[key], dest, name=key)


def create_h5_file(
    path: Path | str,
    *,
    source_file: Path | str | None = None,
    trim_source: bool = False,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(out_path, "w") as h5file:
        if not trim_source:
            copy_h5_contents(source_file, h5file)
        if source_file:
            h5file.attrs["source_file"] = str(source_file)
    return out_path


def find_first_existing_path(
    group_or_file: h5py.Group | h5py.File,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in group_or_file:
            return candidate
    return None


def find_child_group_by_attr(
    group: h5py.Group,
    attr_name: str,
    attr_value: Any,
) -> h5py.Group | None:
    for child in group.values():
        if isinstance(child, h5py.Group) and child.attrs.get(attr_name) == attr_value:
            return child
    return None


def read_dataset(
    group_or_file: h5py.Group | h5py.File,
    path: str,
    default: Any = None,
) -> Any:
    try:
        dataset = group_or_file[path]
    except Exception:
        return default
    try:
        return dataset[()]
    except Exception:
        return default


def read_array(
    group_or_file: h5py.Group | h5py.File,
    path: str,
    dtype=None,
) -> np.ndarray | None:
    value = read_dataset(group_or_file, path, default=None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    if arr.shape == ():
        return np.asarray([arr.item()], dtype=dtype)
    return np.ravel(arr)


def create_unique_group(parent: h5py.Group, base_name: str) -> h5py.Group:
    candidate = base_name
    idx = 1
    while candidate in parent:
        candidate = f"{base_name}_{idx}"
        idx += 1
    return parent.create_group(candidate)


def resolve_dataset_target(root_group: h5py.Group, key: str) -> tuple[h5py.Group, str]:
    normalized_key = str(key).replace("\\", "/").strip("/")
    parts = [part for part in normalized_key.split("/") if part]
    if not parts:
        raise ValueError("Dataset key cannot be empty.")

    parent = root_group
    for part in parts[:-1]:
        existing = parent.get(part)
        if existing is None:
            parent = parent.create_group(part)
            continue
        if isinstance(existing, h5py.Group):
            parent = existing
            continue
        raise ValueError(
            f"Cannot create subgroup '{part}' for key '{key}': a dataset already exists at that path."
        )

    return parent, parts[-1]


def set_attr_safe(h5obj: h5py.File | h5py.Group | h5py.Dataset, key: str, value) -> None:
    if isinstance(value, str):
        h5obj.attrs.create(key, value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    data = value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            data = np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8"))
        else:
            data = np.asarray(value)
    try:
        h5obj.attrs[key] = data
    except (TypeError, ValueError):
        h5obj.attrs[key] = str(value)


def write_value_dataset(group: h5py.Group, key: str, value) -> None:
    ds_attrs = None
    data = value

    if hasattr(value, "data") and hasattr(value, "attrs"):
        data = value.data
        ds_attrs = value.attrs
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        data, ds_attrs = value

    target_group, dataset_key = resolve_dataset_target(group, str(key))
    if dataset_key in target_group:
        del target_group[dataset_key]

    if isinstance(data, str):
        dataset = target_group.create_dataset(
            dataset_key, data=data, dtype=h5py.string_dtype(encoding="utf-8")
        )
    elif isinstance(data, (list, tuple)) and all(
        isinstance(item, str) for item in data
    ):
        str_dtype = h5py.string_dtype(encoding="utf-8")
        dataset = target_group.create_dataset(
            dataset_key,
            data=np.asarray(data, dtype=object),
            dtype=str_dtype,
        )
    else:
        payload = data
        if isinstance(data, (list, tuple)):
            payload = np.asarray(data)
        try:
            dataset = target_group.create_dataset(dataset_key, data=payload)
        except (TypeError, ValueError):
            if isinstance(payload, np.ndarray) and payload.dtype.kind in {"U", "O"}:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                dataset = target_group.create_dataset(
                    dataset_key,
                    data=np.asarray(payload, dtype=object),
                    dtype=str_dtype,
                )
            else:
                dataset = target_group.create_dataset(
                    dataset_key,
                    data=str(data),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

    if ds_attrs:
        for attr_key, attr_val in ds_attrs.items():
            set_attr_safe(dataset, attr_key, attr_val)


def write_metrics_tree_group(
    parent: h5py.Group,
    tree: MetricsTree,
    *,
    overwrite: bool = False,
) -> h5py.Group:
    group_name = safe_h5_key(tree.name)
    existing = parent.get(group_name)
    if existing is not None:
        if overwrite:
            del parent[group_name]
        else:
            group = create_unique_group(parent, group_name)
            set_attr_safe(group, "pipeline", tree.name)
            if tree.attrs:
                for key, value in tree.attrs.items():
                    if key == "pipeline":
                        continue
                    set_attr_safe(group, key, value)
            for key, value in tree.metrics.items():
                write_value_dataset(group, key, value)
            return group

    group = parent.create_group(group_name)
    set_attr_safe(group, "pipeline", tree.name)
    if tree.attrs:
        for key, value in tree.attrs.items():
            if key == "pipeline":
                continue
            set_attr_safe(group, key, value)
    for key, value in tree.metrics.items():
        write_value_dataset(group, key, value)
    return group


def write_metrics_trees_to_h5(
    h5_path: Path | str,
    root_path: str,
    trees: Sequence[MetricsTree],
    *,
    overwrite: bool = False,
) -> None:
    with open_h5(h5_path, "r+") as h5file:
        root_group = h5file.require_group(root_path)
        for tree in trees:
            write_metrics_tree_group(
                root_group,
                tree,
                overwrite=overwrite,
            )


def append_metrics_trees_to_h5(
    h5_path: Path | str,
    root_path: str,
    trees: Sequence[MetricsTree],
    *,
    overwrite: bool = True,
) -> None:
    write_metrics_trees_to_h5(
        h5_path,
        root_path,
        trees,
        overwrite=overwrite,
    )
