from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py

from .base import ProcessResult


def safe_h5_key(name: str) -> str:
    """Return a filesystem/HDF5-friendly key derived from a pipeline name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "pipeline"


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
            metrics_grp.create_dataset(key, data=value)
        if result.artifacts:
            artifacts_grp = f.create_group("artifacts")
            for key, value in result.artifacts.items():
                artifacts_grp.create_dataset(key, data=value)
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
                metrics_grp.create_dataset(key, data=value)
            if result.artifacts:
                artifacts_grp = pipeline_grp.create_group("artifacts")
                for key, value in result.artifacts.items():
                    artifacts_grp.create_dataset(key, data=value)
    return str(out_path)
