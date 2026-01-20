from pathlib import Path
from typing import Optional, Union

import h5py

from .base import ProcessResult


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
