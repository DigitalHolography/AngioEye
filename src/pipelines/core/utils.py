from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from .base import DatasetValue, ProcessResult


def safe_h5_key(name: str) -> str:
    """Return a filesystem/HDF5-friendly key derived from a pipeline name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "pipeline"


def _copy_input_contents(source_file: str | Path | None, dest: h5py.File) -> None:
    """Copy all attributes and top-level objects from the input H5 into dest."""
    if not source_file:
        return
    src_path = Path(source_file)
    if not src_path.exists():
        return
    with h5py.File(src_path, "r") as src:
        for key, value in src.attrs.items():
            dest.attrs[key] = value
        for key in src.keys():
            src.copy(src[key], dest, name=key)


def _ensure_pipelines_group(h5file: h5py.File) -> h5py.Group:
    """Return a pipelines group, creating it when missing."""
    return (
        h5file["pipelines"]
        if "pipelines" in h5file
        else h5file.create_group("pipelines")
    )


def _create_unique_group(parent: h5py.Group, base_name: str) -> h5py.Group:
    """Create a subgroup avoiding name collisions."""
    candidate = base_name
    idx = 1
    while candidate in parent:
        candidate = f"{base_name}_{idx}"
        idx += 1
    return parent.create_group(candidate)


def _write_value_dataset(group: h5py.Group, key: str, value) -> None:
    """
    Create a dataset under group for the given value.

    Handles scalars, numpy arrays, and nested lists/tuples.
    Falls back to a UTF-8 string representation when the value type
    is not directly supported by h5py.
    """
    ds_attrs = None
    data = value

    # Support DatasetValue or tuple(value, attrs) for convenience.
    if isinstance(value, DatasetValue):
        data = value.data
        ds_attrs = value.attrs
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        data, ds_attrs = value

    if isinstance(data, str):
        dataset = group.create_dataset(
            key, data=data, dtype=h5py.string_dtype(encoding="utf-8")
        )
    else:
        payload = data
        if isinstance(data, (list, tuple)):
            payload = np.asarray(data)
        try:
            dataset = group.create_dataset(key, data=payload)
        except (TypeError, ValueError):
            dataset = group.create_dataset(
                key, data=str(data), dtype=h5py.string_dtype(encoding="utf-8")
            )

    if ds_attrs:
        for attr_key, attr_val in ds_attrs.items():
            _set_attr_safe(dataset, attr_key, attr_val)


def _set_attr_safe(h5obj: h5py.File | h5py.Group, key: str, value) -> None:
    """
    Set an attribute on a file or group, falling back to string when the type is unsupported.
    """
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


def write_result_h5(
    result: ProcessResult,
    path: Path | str,
    pipeline_name: str,
    source_file: str | None = None,
) -> str:
    """
    Write pipeline results to an HDF5 file.

    Attributes:
        pipeline: pipeline display name.
        source_file: optional path to the originating HDF5 input.
        metrics: stored under /metrics/<name>.
        artifacts: stored under /artifacts/<name> when present.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _copy_input_contents(source_file, f)
        if "pipeline" not in f.attrs:
            f.attrs["pipeline"] = pipeline_name
        if source_file:
            f.attrs["source_file"] = source_file
        if result.file_attrs:
            for key, value in result.file_attrs.items():
                if key in {"pipeline", "source_file"}:
                    continue
                _set_attr_safe(f, key, value)
        pipelines_grp = _ensure_pipelines_group(f)
        pipeline_grp = _create_unique_group(pipelines_grp, safe_h5_key(pipeline_name))
        pipeline_grp.attrs["pipeline"] = pipeline_name
        if result.attrs:
            for key, value in result.attrs.items():
                if key == "pipeline":
                    continue
                _set_attr_safe(pipeline_grp, key, value)
        metrics_grp = pipeline_grp.create_group("metrics")
        for key, value in result.metrics.items():
            _write_value_dataset(metrics_grp, key, value)
        if result.artifacts:
            artifacts_grp = pipeline_grp.create_group("artifacts")
            for key, value in result.artifacts.items():
                _write_value_dataset(artifacts_grp, key, value)
    return str(out_path)


def write_combined_results_h5(
    results: Sequence[tuple[str, ProcessResult]],
    path: Path | str,
    source_file: str | None = None,
) -> str:
    """
    Write multiple pipeline results into a single HDF5 file.

    The file groups results under /pipelines/<safe_pipeline_name>/{metrics,artifacts}.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _copy_input_contents(source_file, f)
        if source_file:
            f.attrs["source_file"] = source_file
        pipelines_grp = _ensure_pipelines_group(f)
        for pipeline_name, result in results:
            pipeline_grp = _create_unique_group(
                pipelines_grp, safe_h5_key(pipeline_name)
            )
            pipeline_grp.attrs["pipeline"] = pipeline_name
            if result.attrs:
                for key, value in result.attrs.items():
                    if key == "pipeline":
                        continue
                    _set_attr_safe(pipeline_grp, key, value)
            metrics_grp = pipeline_grp.create_group("metrics")
            for key, value in result.metrics.items():
                _write_value_dataset(metrics_grp, key, value)
            if result.artifacts:
                artifacts_grp = pipeline_grp.create_group("artifacts")
                for key, value in result.artifacts.items():
                    _write_value_dataset(artifacts_grp, key, value)
    return str(out_path)
