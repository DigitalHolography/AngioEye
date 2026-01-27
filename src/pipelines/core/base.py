import csv
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py


@dataclass
class ProcessResult:
    metrics: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None
    attrs: Optional[Dict[str, Any]] = None  # attributes stored on the pipeline group
    file_attrs: Optional[Dict[str, Any]] = None  # attributes stored on the root H5 file
    output_h5_path: Optional[str] = None


@dataclass
class DatasetValue:
    """Represents a dataset payload plus optional attributes for that dataset."""

    data: Any
    attrs: Optional[Dict[str, Any]] = None


def with_attrs(data: Any, attrs: Dict[str, Any]) -> DatasetValue:
    """Convenience helper to attach attributes to a dataset value."""
    return DatasetValue(data=data, attrs=attrs)


class ProcessPipeline:
    description: str = ""

    def __init__(self) -> None:
        # Derive the pipeline name from the module filename (e.g., basic_stats.py -> basic_stats).
        module_name = (self.__class__.__module__ or "").rsplit(".", 1)[-1]
        self.name: str = module_name or self.__class__.__name__

    def run(self, h5file: h5py.File) -> ProcessResult:
        raise NotImplementedError

    def export(self, result: ProcessResult, output_path: str) -> str:
        """Default CSV export for metrics."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            for key, value in result.metrics.items():
                writer.writerow([key, value])
        return output_path
