from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from .base import ProcessResult


def safe_h5_key(name: str) -> str:
    """Return a filesystem/HDF5-friendly key derived from a pipeline name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "pipeline"


def _write_value_dataset(group: h5py.Group, key: str, value) -> None:
    """
    Create a dataset under group for the given value.

    Handles scalars, numpy arrays, and nested lists/tuples.
    Falls back to a UTF-8 string representation when the value type
    is not directly supported by h5py.
    """
    if isinstance(value, str):
        group.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    data = value
    if isinstance(value, (list, tuple)):
        data = np.asarray(value)
    try:
        group.create_dataset(key, data=data)
    except (TypeError, ValueError):
        group.create_dataset(
            key, data=str(value), dtype=h5py.string_dtype(encoding="utf-8")
        )


def write_result_h5(
    result: ProcessResult,
    path: Union[Path, str],
    pipeline_name: str,
    source_file: Optional[str] = None,
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
        f.attrs["pipeline"] = pipeline_name
        if source_file:
            f.attrs["source_file"] = source_file
        metrics_grp = f.create_group("metrics")
        for key, value in result.metrics.items():
            _write_value_dataset(metrics_grp, key, value)
        if result.artifacts:
            artifacts_grp = f.create_group("artifacts")
            for key, value in result.artifacts.items():
                _write_value_dataset(artifacts_grp, key, value)
    return str(out_path)


def write_combined_results_h5(
    results: Sequence[Tuple[str, ProcessResult]],
    path: Union[Path, str],
    source_file: Optional[str] = None,
) -> str:
    """
    Write multiple pipeline results into a single HDF5 file.

    The file groups results under /pipelines/<safe_pipeline_name>/{metrics,artifacts}.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        if source_file:
            f.attrs["source_file"] = source_file
        pipelines_grp = f.create_group("pipelines")
        for pipeline_name, result in results:
            pipeline_grp = pipelines_grp.create_group(safe_h5_key(pipeline_name))
            pipeline_grp.attrs["pipeline"] = pipeline_name
            metrics_grp = pipeline_grp.create_group("metrics")
            for key, value in result.metrics.items():
                _write_value_dataset(metrics_grp, key, value)
            if result.artifacts:
                artifacts_grp = pipeline_grp.create_group("artifacts")
                for key, value in result.artifacts.items():
                    _write_value_dataset(artifacts_grp, key, value)
    return str(out_path)
